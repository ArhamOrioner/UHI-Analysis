import logging
import os
from typing import Dict, List, Optional, Tuple

import ee
import numpy as np
import pandas as pd

from config import AnalysisConfig
from utils import ensure_dir, setup_logging, season_date_range, get_city_roi_bounds
import gee as gee_mod
from sampling import sample_to_dataframe
from eda import create_eda_plots
from clustering import choose_optimal_clusters, kmeans_and_label, plot_uhi_distribution
from spatial import run_hot_spot_analysis, plot_hot_spots
from map import create_map
from ml import prepare_features, train_random_forest, evaluate_and_plot, shap_analysis


def run_analysis(cfg: AnalysisConfig) -> None:
    setup_logging(cfg.log_level, cfg.outputs_dir)
    ensure_dir(cfg.outputs_dir)

    if cfg.city_name and not cfg.roi_bounds:
        bounds = get_city_roi_bounds(cfg.city_name)
        if bounds:
            cfg.roi_bounds = bounds

    if not gee_mod.initialize_earth_engine():
        return

    roi = ee.Geometry.Rectangle(cfg.roi_bounds)

    all_period_frames: List[pd.DataFrame] = []
    for year in cfg.analysis_years:
        for season_name, (start_m, end_m) in cfg.seasons_to_analyze.items():
            start_date, end_date = season_date_range(year, start_m, end_m)
            logging.info("Processing %s %s..%s", season_name, start_date, end_date)

            if cfg.primary_lst_source.upper() == "MODIS":
                lst_img = gee_mod.get_modis_lst(roi, start_date, end_date)
                lst_band = "LST_Celsius"
            elif cfg.primary_lst_source.upper() == "LANDSAT":
                lst_img = gee_mod.get_landsat_lst(roi, start_date, end_date)
                lst_band = "LST_Landsat_Celsius"
            else:
                logging.error("Invalid PRIMARY_LST_SOURCE: %s", cfg.primary_lst_source)
                return

            ndvi_img, ndbi_img = gee_mod.get_landsat_indices(roi, start_date, end_date)
            lc_img = gee_mod.get_worldcover_landcover(roi, cfg.worldcover_year)
            topo_img = gee_mod.get_topographic_features(roi)

            if any(x is None for x in [lst_img, ndvi_img, ndbi_img, lc_img, topo_img]):
                logging.warning("Skipping period due to missing inputs.")
                continue

            combined = lst_img.addBands([ndvi_img, ndbi_img, lc_img, topo_img])
            df = sample_to_dataframe(
                combined, roi, cfg.num_pixels_for_ml, cfg.common_resolution_m, cfg.random_state
            )
            if df is None or df.empty:
                logging.warning("Empty sample; skipping period.")
                continue

            # Normalize LST column name
            if lst_band in df.columns and lst_band != "LST_Celsius":
                df.rename(columns={lst_band: "LST_Celsius"}, inplace=True)

            # Cleaning
            df.dropna(inplace=True)
            valid_lc = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
            df = df[df["Map"].isin(valid_lc)] if "Map" in df.columns else df
            # Rename landcover if present under typical names
            if "Map" in df.columns:
                df.rename(columns={"Map": "Land_Cover"}, inplace=True)
            if "Land_Cover" in df.columns:
                lc_map = {10: "Trees", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up", 60: "Bare_Vegetation", 70: "Snow_Ice", 80: "Water", 90: "Wetland", 95: "Mangroves"}
                df["Land_Cover_Type"] = df["Land_Cover"].map(lc_map)
                df.drop(columns=["Land_Cover"], inplace=True)

            df["Year"] = year
            df["Season"] = season_name
            all_period_frames.append(df)

    if not all_period_frames:
        logging.error("No data processed. Exiting.")
        return

    df_all = pd.concat(all_period_frames, ignore_index=True)
    logging.info("Combined DataFrame shape: %s", df_all.shape)

    # EDA
    try:
        create_eda_plots(df_all, cfg.outputs_dir)
    except Exception as exc:
        logging.warning("EDA plotting skipped: %s", exc)

    # Clustering
    if cfg.uhi_clusters == "auto":
        k = choose_optimal_clusters(df_all, cfg.uhi_cluster_range, cfg.random_state)
    else:
        k = int(cfg.uhi_clusters)
    df_all, ordered_zones, zone_colors, centers = kmeans_and_label(df_all, k, cfg.random_state)
    try:
        plot_uhi_distribution(df_all, ordered_zones, zone_colors, cfg.outputs_dir)
    except Exception as exc:
        logging.warning("UHI distribution plot skipped: %s", exc)

    # UHI intensity
    urban_types = ["Built-up"]
    rural_types = ["Trees", "Shrubland", "Grassland", "Cropland", "Water"]
    urban_mean = df_all[df_all["Land_Cover_Type"].isin(urban_types)]["LST_Celsius"].mean()
    rural_mean = df_all[df_all["Land_Cover_Type"].isin(rural_types)]["LST_Celsius"].mean()
    if not np.isnan(urban_mean) and not np.isnan(rural_mean):
        logging.info("UHI intensity: %.2f°C", float(urban_mean - rural_mean))

    # Hot spot analysis
    df_all = run_hot_spot_analysis(df_all, k_neighbors=8)
    try:
        plot_hot_spots(df_all, cfg.outputs_dir)
    except Exception as exc:
        logging.warning("Hot spot plotting skipped: %s", exc)

    # Map layers for overall period
    lst_map = (
        gee_mod.get_modis_lst(roi, cfg.start_date, cfg.end_date)
        if cfg.primary_lst_source.upper() == "MODIS"
        else gee_mod.get_landsat_lst(roi, cfg.start_date, cfg.end_date)
    )
    ndvi_map, ndbi_map = gee_mod.get_landsat_indices(roi, cfg.start_date, cfg.end_date)
    lc_map = gee_mod.get_worldcover_landcover(roi, cfg.worldcover_year)
    topo_map = gee_mod.get_topographic_features(roi)
    try:
        create_map(roi, lst_map, ndvi_map, ndbi_map, lc_map, topo_map, df_all, ordered_zones, zone_colors, cfg.outputs_dir, cfg.start_date, cfg.end_date, cfg.city_name)
    except Exception as exc:
        logging.warning("Map generation skipped: %s", exc)

    # Machine learning
    try:
        X, y = prepare_features(df_all)
        default_rf = {
            "n_estimators": cfg.rf_n_estimators,
            "max_depth": cfg.rf_max_depth,
            "min_samples_split": cfg.rf_min_samples_split,
            "min_samples_leaf": cfg.rf_min_samples_leaf,
        }
        model = train_random_forest(
            X,
            y,
            random_state=cfg.random_state,
            perform_tuning=cfg.perform_hyperparameter_tuning,
            tuning_iterations=cfg.tuning_iterations,
            default_params=default_rf,
        )
        metrics = evaluate_and_plot(model, X, y, cfg.outputs_dir)
        logging.info("Model metrics: %s", metrics)
        shap_analysis(model, X, cfg.outputs_dir)
    except Exception as exc:
        logging.warning("ML pipeline skipped: %s", exc)

    logging.info("Analysis complete. Outputs in '%s'", cfg.outputs_dir)

