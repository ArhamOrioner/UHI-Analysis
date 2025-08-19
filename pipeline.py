import logging
import os
from typing import Dict, List, Optional, Tuple

import ee
import numpy as np
import pandas as pd

from config import AnalysisConfig
from utils import ensure_dir, setup_logging, get_full_year_date_range, get_city_roi_bounds, generate_summary_report
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

    if not gee_mod.initialize_earth_engine():
        return

    roi = None
    if cfg.roi_bounds and len(cfg.roi_bounds) == 4:
        logging.info("Using user-provided ROI bounds: %s", cfg.roi_bounds)
        roi = ee.Geometry.Rectangle(cfg.roi_bounds)
    elif cfg.city_name:
        logging.info("Attempting to find ROI bounds for city: '%s'", cfg.city_name)
        bounds = get_city_roi_bounds(cfg.city_name)
        if bounds:
            cfg.roi_bounds = bounds
            roi = ee.Geometry.Rectangle(bounds)
            logging.info("Found bounds for %s: %s", cfg.city_name, bounds)

    if roi is None:
        logging.error("Could not determine Region of Interest. Provide --roi-bounds or a valid --city.")
        return

    all_year_frames: List[pd.DataFrame] = []
    yearly_reports = {}

    # FIX: The main loop now iterates through years, not seasons.
    for year in cfg.analysis_years:
        period_id = str(year)
        date_range = get_full_year_date_range(year)
        if not date_range:
            continue
        start_date, end_date = date_range

        logging.info("\n" + "#"*80)
        logging.info("PROCESSING PERIOD: Year %s (%s to %s)", period_id, start_date, end_date)
        logging.info("#"*80)

        # --- Data Fetching & Cleaning ---
        lst_img = (
            gee_mod.get_modis_lst(roi, start_date, end_date)
            if cfg.primary_lst_source.upper() == "MODIS"
            else gee_mod.get_landsat_lst(roi, start_date, end_date)
        )
        lst_band = "LST_Celsius" if cfg.primary_lst_source.upper() == "MODIS" else "LST_Landsat_Celsius"
        
        ndvi_img, ndbi_img = gee_mod.get_landsat_indices(roi, start_date, end_date)
        lc_img = gee_mod.get_worldcover_landcover(roi, cfg.worldcover_year)
        topo_img = gee_mod.get_topographic_features(roi)

        if any(x is None for x in [lst_img, ndvi_img, ndbi_img, lc_img, topo_img]):
            logging.warning("Skipping year %s due to missing GEE data.", period_id)
            continue

        combined = lst_img.addBands([ndvi_img, ndbi_img, lc_img, topo_img])
        df = sample_to_dataframe(combined, roi, cfg.num_pixels_for_ml, cfg.common_resolution_m, cfg.random_state)
        if df is None or df.empty:
            logging.warning("Empty sample for %s; skipping year.", period_id)
            continue

        if lst_band in df.columns and lst_band != "LST_Celsius":
            df.rename(columns={lst_band: "LST_Celsius"}, inplace=True)
        df.dropna(inplace=True)
        if "Map" in df.columns:
            lc_map = {10:"Trees", 20:"Shrubland", 30:"Grassland", 40:"Cropland", 50:"Built-up", 60:"Bare_Vegetation", 80:"Water", 90:"Wetland", 95:"Mangroves"}
            df["Land_Cover_Type"] = df["Map"].map(lc_map)
            df.drop(columns=["Map"], inplace=True)
        
        # --- Perform Full Analysis for this Year ---
        logging.info("--- Starting Analysis for Year %s ---", period_id)
        
        # 1. Clustering
        k = choose_optimal_clusters(df, cfg.uhi_cluster_range, cfg.random_state) if cfg.uhi_clusters == "auto" else int(cfg.uhi_clusters)
        df, ordered_zones, zone_colors, _ = kmeans_and_label(df, k, cfg.random_state)
        plot_uhi_distribution(df, ordered_zones, zone_colors, os.path.join(cfg.outputs_dir, f"{period_id}_uhi_zones.png"))

        # 2. UHI Intensity
        urban_mean = df[df["Land_Cover_Type"] == "Built-up"]["LST_Celsius"].mean()
        rural_types = ["Trees", "Shrubland", "Grassland", "Cropland", "Water"]
        rural_mean = df[df["Land_Cover_Type"].isin(rural_types)]["LST_Celsius"].mean()
        uhi_intensity = urban_mean - rural_mean if pd.notna(urban_mean) and pd.notna(rural_mean) else None
        if uhi_intensity: logging.info("[%s] Annual UHI Intensity: %.2f°C", period_id, uhi_intensity)

        # 3. Hot Spot Analysis
        df = run_hot_spot_analysis(df)
        plot_hot_spots(df, os.path.join(cfg.outputs_dir, f"{period_id}_hot_spots.png"))

        yearly_reports[period_id] = {
            "uhi_intensity": uhi_intensity,
            "hot_spots": (df["Hot_Spot_Class"] == "Hot Spot").sum(),
            "cold_spots": (df["Hot_Spot_Class"] == "Cold Spot").sum(),
        }
        all_year_frames.append(df)

    if not all_year_frames:
        logging.error("No data successfully processed for any year. Exiting.")
        return

    df_all = pd.concat(all_year_frames, ignore_index=True)
    logging.info("\nCombined DataFrame for all years has shape: %s", df_all.shape)

    # --- Post-Loop Analysis on COMBINED data ---
    create_eda_plots(df_all, cfg.outputs_dir)

    logging.info("Fetching GEE layers for overall map...")
    primary_df = all_year_frames[0]
    _, p_ord_zones, p_zone_colors, _ = kmeans_and_label(primary_df.copy(), 3, cfg.random_state)
    lst_map = (gee_mod.get_modis_lst(roi, cfg.start_date, cfg.end_date) if cfg.primary_lst_source.upper() == "MODIS" else gee_mod.get_landsat_lst(roi, cfg.start_date, cfg.end_date))
    ndvi_map, ndbi_map = gee_mod.get_landsat_indices(roi, cfg.start_date, cfg.end_date)
    lc_map = gee_mod.get_worldcover_landcover(roi, cfg.worldcover_year)
    topo_map = gee_mod.get_topographic_features(roi)
    create_map(roi, lst_map, ndvi_map, ndbi_map, lc_map, topo_map, primary_df, p_ord_zones, p_zone_colors, cfg.outputs_dir, cfg.start_date, cfg.end_date, cfg.city_name)

    metrics, model = None, None
    try:
        X, y = prepare_features(df_all)
        default_rf = {"n_estimators": cfg.rf_n_estimators, "max_depth": cfg.rf_max_depth, "min_samples_split": cfg.rf_min_samples_split, "min_samples_leaf": cfg.rf_min_samples_leaf}
        model = train_random_forest(X, y, cfg.random_state, cfg.perform_hyperparameter_tuning, cfg.tuning_iterations, default_rf)
        metrics = evaluate_and_plot(model, X, y, cfg.outputs_dir, cfg.random_state)
        logging.info("Model metrics on combined data: %s", metrics)
        shap_analysis(model, X, cfg.outputs_dir, cfg.random_state)
    except Exception as exc:
        logging.error("ML pipeline failed: %s", exc, exc_info=True)

    generate_summary_report(cfg, df_all, metrics, model, yearly_reports)

    logging.info("Analysis complete. Outputs in '%s'", cfg.outputs_dir)
