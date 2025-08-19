import json
import logging
import os
from typing import List, Optional, Tuple, Dict
import datetime

import requests
import pandas as pd
import numpy as np


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def setup_logging(log_level: str, outputs_dir: str) -> None:
    ensure_dir(outputs_dir)
    log_file = os.path.join(outputs_dir, "uhi_analysis.log")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )

def get_full_year_date_range(year: int) -> Optional[Tuple[str, str]]:
    """
    Generates a start and end date for a full year of analysis.
    - For past years, returns the full year.
    - For the current year, returns from the start of the year to today.
    - For future years, returns None.
    """
    today = datetime.date.today()
    current_year = today.year

    if year > current_year:
        logging.warning("Cannot analyze future year %d. Skipping.", year)
        return None
    
    start_date = f"{year}-01-01"
    
    if year < current_year:
        end_date = f"{year}-12-31"
    else: # year == current_year
        end_date = today.strftime("%Y-%m-%d")
        logging.info("Analyzing current year (%d) up to today's date: %s", year, end_date)

    return start_date, end_date


def get_city_roi_bounds(city_name: str) -> Optional[List[float]]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city_name, "format": "jsonv2", "limit": 1}
    headers = {"User-Agent": "UHI_Analysis_Script/1.0 (github.com/your-repo)"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data or "boundingbox" not in data[0]:
            logging.warning("No results or bounding box for city '%s' from Nominatim.", city_name)
            return None
        bbox = data[0]["boundingbox"]
        return [float(bbox[2]), float(bbox[0]), float(bbox[3]), float(bbox[1])]
    except requests.exceptions.RequestException as exc:
        logging.error("Failed to resolve city bounds via Nominatim API: %s", exc)
        return None


def generate_summary_report(cfg, df: pd.DataFrame, metrics: Optional[Dict], model, yearly_reports: Dict) -> None:
    logging.info("\n" + "="*80)
    logging.info("UHI ANALYSIS - FINAL SUMMARY REPORT")
    logging.info("="*80)
    logging.info("📍 Study Area: %s", cfg.city_name)
    logging.info("📅 Analysis Years: %s", cfg.analysis_years)

    logging.info("\n--- Yearly UHI Findings ---")
    if not yearly_reports:
        logging.info("  No yearly data to report.")
    else:
        for year, report in yearly_reports.items():
            intensity_str = f"{report['uhi_intensity']:.2f}°C" if report.get('uhi_intensity') is not None else "N/A"
            logging.info("  - %s: UHI Intensity: %s | Hot Spots: %d | Cold Spots: %d",
                         year, intensity_str, report.get('hot_spots', 0), report.get('cold_spots', 0))

    if metrics:
        logging.info("\n--- Machine Learning Model (Random Forest on Combined Data) ---")
        # FIX: Add descriptive labels to the metrics for better understanding.
        r2_percentage = metrics['r2'] * 100
        logging.info("📊 Explanatory Power (R² Score): %.3f (The model explains %.1f%% of the temperature variation)", metrics['r2'], r2_percentage)
        logging.info("📉 Avg. Prediction Error (RMSE): %.2f°C (On average, predictions are off by this much)", metrics['rmse'])
        
        if model and hasattr(model, 'feature_importances_'):
            try:
                importances = pd.Series(model.feature_importances_, index=model.feature_names_in_)
                logging.info("🔑 Most Important Feature: %s", importances.idxmax())
            except Exception as e:
                logging.warning("Could not determine the most important feature: %s", e)

    logging.info("\n--- Outputs ---")
    logging.info("📁 All plots, logs, and maps are saved in the '%s' directory.", cfg.outputs_dir)
    logging.info("="*80)
