"""Annual analysis with explicit provenance and spatially held-out evaluation."""

import logging
import os
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version

import matplotlib

matplotlib.use("Agg")
import ee
import numpy as np
import pandas as pd

import gee as gee_mod
from clustering import choose_optimal_clusters, kmeans_and_label, plot_uhi_distribution
from config import AnalysisConfig
from eda import create_eda_plots
from map import create_map
from ml import (
    evaluate_and_plot,
    prepare_features,
    shap_analysis,
    spatial_holdout,
    train_random_forest,
)
from sampling import sample_to_dataframe
from spatial import plot_hot_spots, run_hot_spot_analysis
from utils import (
    get_city_roi_bounds,
    get_full_year_date_range,
    setup_logging,
    write_json,
)

LAND_COVER = {
    10: "Trees",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare_Sparse_Vegetation",
    70: "Snow_Ice",
    80: "Water",
    90: "Wetland",
    95: "Mangroves",
    100: "Moss_Lichen",
}
RURAL_TYPES = ["Trees", "Shrubland", "Grassland", "Cropland"]


def surface_uhi_summary(df):
    urban = df.loc[df.Land_Cover_Type == "Built-up", "LST_Celsius"]
    rural = df.loc[df.Land_Cover_Type.isin(RURAL_TYPES), "LST_Celsius"]
    return {
        "uhi_intensity": float(urban.mean() - rural.mean())
        if len(urban) and len(rural)
        else None,
        "urban_samples": len(urban),
        "rural_samples": len(rural),
        "rural_reference_classes": RURAL_TYPES,
    }


def clean_samples(df, year):
    required = [
        "LST_Celsius",
        "NDVI",
        "NDBI",
        "Map",
        "Elevation",
        "Slope",
        "Aspect_sin",
        "Aspect_cos",
        "latitude",
        "longitude",
    ]
    result = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    result["Land_Cover_Type"] = result["Map"].map(LAND_COVER).fillna("Unknown")
    result["Year"] = year
    return result.drop(columns="Map")


def run_analysis(cfg: AnalysisConfig) -> bool:
    cfg.validate()
    setup_logging(cfg.log_level, cfg.outputs_dir)
    if not gee_mod.initialize_earth_engine(cfg.ee_project):
        return False
    if cfg.roi_bounds is None:
        cfg.roi_bounds = get_city_roi_bounds(cfg.city_name)
    if cfg.roi_bounds is None:
        logging.error(
            "Could not determine ROI. Supply --roi-bounds west south east north."
        )
        return False
    cfg.validate()  # Also validate geocoded bounds against the supported extent.
    roi = ee.Geometry.Rectangle(cfg.roi_bounds)
    logging.warning(
        "Surface UHI is a sampled built-up minus vegetated-land contrast within the ROI; it is not air-temperature UHI."
    )
    logging.warning(
        "Using static WorldCover %s for all analysis years.", cfg.worldcover_year
    )
    frames, yearly_reports, failures = [], {}, {}
    lc_native = gee_mod.get_worldcover_landcover(roi, cfg.worldcover_year)
    topo_native = gee_mod.get_topographic_features(roi)
    grid = lambda image: gee_mod.to_analysis_grid(image, cfg.common_resolution_m)
    lc_img = gee_mod.to_analysis_grid(
        lc_native, cfg.common_resolution_m, categorical=True
    )
    topo_img = grid(topo_native)

    for year in cfg.analysis_years:
        period = str(year)
        dates = get_full_year_date_range(year)
        if not dates or dates[0] >= dates[1]:
            failures[period] = "No completed date interval available"
            continue
        start_date, end_date = dates
        logging.info("Processing %s: [%s, %s)", year, start_date, end_date)
        try:
            fetch_lst = (
                gee_mod.get_modis_lst
                if cfg.primary_lst_source == "MODIS"
                else gee_mod.get_landsat_lst
            )
            lst_native = fetch_lst(roi, start_date, end_date)
            ndvi_native, ndbi_native = gee_mod.get_landsat_indices(
                roi, start_date, end_date, cfg.index_denominator_epsilon
            )
            if any(img is None for img in (lst_native, ndvi_native, ndbi_native)):
                failures[period] = "Missing source scenes"
                continue
            lst_img, ndvi_img, ndbi_img = map(
                grid, (lst_native, ndvi_native, ndbi_native)
            )
            combined = lst_img.addBands([ndvi_img, ndbi_img, lc_img, topo_img])
            df = sample_to_dataframe(
                combined,
                roi,
                cfg.num_pixels_for_ml,
                cfg.common_resolution_m,
                cfg.random_state,
            )
            if df.empty:
                failures[period] = "No complete samples after masking"
                continue
            df = clean_samples(df, year)
            if df.empty:
                failures[period] = "No finite complete samples"
                continue
            k = (
                choose_optimal_clusters(df, cfg.uhi_cluster_range, cfg.random_state)
                if cfg.uhi_clusters == "auto"
                else cfg.uhi_clusters
            )
            df, zones, colors, _ = kmeans_and_label(df, k, cfg.random_state)
            df = run_hot_spot_analysis(df)
            if (df.Hot_Spot_Class == "Analysis Error").any():
                failures[period + "_hotspots"] = "Hotspot analysis failed; inspect log"
            report = surface_uhi_summary(df)
            report.update(
                {
                    "start_date": start_date,
                    "end_date_exclusive": end_date,
                    "samples": len(df),
                    "hot_spots": int((df.Hot_Spot_Class == "Hot Spot").sum()),
                    "cold_spots": int((df.Hot_Spot_Class == "Cold Spot").sum()),
                    "observation_counts_mean": {
                        c: float(df[c].mean()) for c in df if c.endswith("_count")
                    },
                    "index_outside_unit_interval_samples": {
                        c: int((df[c].abs() > 1).sum()) for c in ("NDVI", "NDBI")
                    },
                }
            )
            yearly_reports[period] = report
            frames.append(df)
            df.to_csv(
                os.path.join(cfg.outputs_dir, f"{period}_samples.csv"), index=False
            )
            logging.info(
                "%s: surface UHI contrast %s C; %d urban / %d rural samples",
                period,
                report["uhi_intensity"],
                report["urban_samples"],
                report["rural_samples"],
            )
            if cfg.index_diagnostics:
                report["index_mask_diagnostics"] = gee_mod.index_mask_diagnostics(
                    roi,
                    start_date,
                    end_date,
                    cfg.index_denominator_epsilon,
                    cfg.common_resolution_m,
                )
            plot_uhi_distribution(
                df,
                zones,
                colors,
                os.path.join(cfg.outputs_dir, f"{period}_uhi_zones.png"),
            )
            plot_hot_spots(df, os.path.join(cfg.outputs_dir, f"{period}_hot_spots.png"))
            create_map(
                roi,
                lst_img.select("LST_Celsius"),
                ndvi_img.select("NDVI"),
                ndbi_img.select("NDBI"),
                lc_img,
                topo_img,
                df,
                zones,
                colors,
                cfg.outputs_dir,
                start_date,
                end_date,
                cfg.city_name,
                cfg.initial_zoom,
            )
        except Exception as exc:
            failures[period] = str(exc)
            logging.exception("Year %s failed or produced incomplete outputs", period)

    metrics, ml_error = None, None
    if frames:
        df_all = pd.concat(frames, ignore_index=True)
        df_all.to_csv(
            os.path.join(cfg.outputs_dir, "samples_all_years.csv"), index=False
        )
        try:
            create_eda_plots(df_all, cfg.outputs_dir)
        except Exception as exc:
            failures["eda"] = str(exc)
            logging.exception("EDA failed")
        try:
            train, test, groups = spatial_holdout(
                df_all, cfg.random_state, cfg.spatial_block_size_m
            )
            X_train, y_train = prepare_features(df_all.iloc[train])
            X_test, y_test = prepare_features(df_all.iloc[test])
            # Learn the categorical feature schema only from training data.
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
            default_rf = {
                "n_estimators": cfg.rf_n_estimators,
                "max_depth": cfg.rf_max_depth,
                "min_samples_split": cfg.rf_min_samples_split,
                "min_samples_leaf": cfg.rf_min_samples_leaf,
            }
            model = train_random_forest(
                X_train,
                y_train,
                cfg.random_state,
                cfg.perform_hyperparameter_tuning,
                cfg.tuning_iterations,
                default_rf,
                groups=groups[train],
            )
            metrics = evaluate_and_plot(model, X_test, y_test, cfg.outputs_dir)
            metrics.update(
                {
                    "split": "spatial_block_holdout",
                    "block_size_m": cfg.spatial_block_size_m,
                    "train_samples": len(train),
                    "train_blocks": len(np.unique(groups[train])),
                    "test_blocks": len(np.unique(groups[test])),
                }
            )
            split_table = df_all[["Year", "longitude", "latitude"]].copy()
            split_table["block"] = groups
            split_table["split"] = "train"
            split_table.loc[test, "split"] = "test"
            split_table.to_csv(
                os.path.join(cfg.outputs_dir, "model_split.csv"), index=False
            )
            shap_analysis(model, X_test, cfg.outputs_dir, cfg.random_state)
        except Exception as exc:
            ml_error = str(exc)
            logging.exception("ML unavailable; preserving annual results")
    versions = {}
    for package in (
        "earthengine-api",
        "numpy",
        "pandas",
        "scikit-learn",
        "esda",
        "libpysal",
        "pyproj",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    summary = {
        "config": asdict(cfg),
        "versions": versions,
        "analysis_crs": gee_mod.ANALYSIS_CRS,
        "yearly_reports": yearly_reports,
        "failures": failures,
        "model_metrics": metrics,
        "model_error": ml_error,
        "status": "complete"
        if frames and not failures and not ml_error
        else "partial"
        if frames
        else "failed",
    }
    write_json(os.path.join(cfg.outputs_dir, "summary.json"), summary)
    logging.info("Run status: %s. Results in %s", summary["status"], cfg.outputs_dir)
    return summary["status"] == "complete"
