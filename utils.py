import json
import logging
import os
from typing import List, Optional, Tuple

import requests


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


def season_date_range(year: int, start_month: int, end_month: int) -> Tuple[str, str]:
    import pandas as pd

    start_date = f"{year}-{start_month:02d}-01"
    if start_month > end_month:
        end_year = year + 1
        end_date = f"{end_year}-{end_month:02d}-{pd.Timestamp(f'{end_year}-{end_month:02d}-01').days_in_month}"
    else:
        end_date = f"{year}-{end_month:02d}-{pd.Timestamp(f'{year}-{end_month:02d}-01').days_in_month}"
    return start_date, end_date


def get_city_roi_bounds(city_name: str) -> Optional[List[float]]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city_name, "format": "jsonv2", "limit": 1}
    headers = {"User-Agent": "UHI_Analysis/1.0 (contact@example.com)"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            logging.warning("No Nominatim results for city '%s'", city_name)
            return None
        bbox = data[0]["boundingbox"]
        south_lat = float(bbox[0])
        north_lat = float(bbox[1])
        west_lon = float(bbox[2])
        east_lon = float(bbox[3])
        return [west_lon, south_lat, east_lon, north_lat]
    except Exception as exc:
        logging.error("Failed to resolve city bounds via Nominatim: %s", exc)
        return None