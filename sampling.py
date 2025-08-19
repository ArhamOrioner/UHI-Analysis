import logging
from typing import Optional
import ee
import geemap
import pandas as pd

def sample_to_dataframe(image: ee.Image, roi: ee.Geometry, num_pixels: int, scale_m: int, random_state: int) -> Optional[pd.DataFrame]:
    """Samples points from a GEE Image and converts to a Pandas DataFrame."""
    try:
        # Use geometries=True to extract lat/lon coordinates
        fc = image.sample(
            region=roi,
            scale=scale_m,
            numPixels=num_pixels,
            seed=random_state,
            geometries=True,
        )
        # Explicitly add lat/lon as properties for the conversion
        fc = fc.map(lambda f: f.set({
            "longitude": f.geometry().coordinates().get(0),
            "latitude": f.geometry().coordinates().get(1),
        }))
        df = geemap.ee_to_df(fc)
        if df.empty:
            logging.warning("No samples were returned from Google Earth Engine.")
            return None
        logging.info("Successfully sampled %d pixels into a DataFrame.", len(df))
        return df
    except Exception as exc:
        logging.error("GEE sampling failed: %s", exc)
        return None
