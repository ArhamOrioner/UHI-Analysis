"""Bounded, reproducible sampling on the explicit analysis grid."""

import logging

import ee
import pandas as pd

from gee import ANALYSIS_CRS


def sample_to_dataframe(
    image: ee.Image, roi: ee.Geometry, num_pixels: int, scale_m: int, random_state: int
) -> pd.DataFrame:
    fc = image.sample(
        region=roi,
        projection=ee.Projection(ANALYSIS_CRS).atScale(scale_m),
        numPixels=num_pixels,
        seed=random_state,
        geometries=True,
        dropNulls=True,
        tileScale=4,
    )
    records = []
    for feature in fc.getInfo()["features"]:
        lon, lat = feature["geometry"]["coordinates"][:2]
        records.append({**feature["properties"], "longitude": lon, "latitude": lat})
    df = pd.DataFrame.from_records(records)
    logging.info(
        "Sampled %d complete grid cells (requested approximately %d).",
        len(df),
        num_pixels,
    )
    return df
