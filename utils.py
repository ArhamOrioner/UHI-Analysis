import datetime
import json
import logging
import os
from typing import List, Optional, Tuple

import requests


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def setup_logging(log_level: str, outputs_dir: str) -> None:
    ensure_dir(outputs_dir)
    log_file = os.path.join(outputs_dir, "uhi_analysis.log")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )


def get_full_year_date_range(year: int) -> Optional[Tuple[str, str]]:
    """
    Generates a start and end date for a full year of analysis.
    - For past years, returns the full year.
    - For the current year, returns from the start of the year to today (exclusive).
    - For future years, returns None.
    """
    today = datetime.date.today()
    current_year = today.year

    if year > current_year:
        logging.warning("Cannot analyze future year %d. Skipping.", year)
        return None

    start_date = f"{year}-01-01"

    if year < current_year:
        end_date = f"{year + 1}-01-01"  # Earth Engine's end date is exclusive.
    else:  # year == current_year
        end_date = today.strftime("%Y-%m-%d")
        logging.info(
            "Analyzing current year (%d) up to today's date: %s", year, end_date
        )

    return start_date, end_date


def get_city_roi_bounds(city_name: str) -> Optional[List[float]]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city_name, "format": "jsonv2", "limit": 1}
    headers = {
        "User-Agent": "UHI-Analysis/2.0 (https://github.com/ArhamOrioner/UHI-Analysis)"
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data or "boundingbox" not in data[0]:
            logging.warning(
                "No results or bounding box for city '%s' from Nominatim.", city_name
            )
            return None
        bbox = data[0]["boundingbox"]
        return [float(bbox[2]), float(bbox[0]), float(bbox[3]), float(bbox[1])]
    except (
        requests.exceptions.RequestException,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
    ) as exc:
        logging.error("Failed to resolve city bounds via Nominatim API: %s", exc)
        return None
