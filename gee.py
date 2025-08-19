import logging
from typing import Optional, Tuple

import ee


def initialize_earth_engine() -> bool:
    try:
        ee.Initialize()
        logging.info("Google Earth Engine initialized.")
        return True
    except Exception as exc:
        logging.error("Earth Engine init failed: %s", exc)
        logging.error("Please run 'earthengine authenticate' and try again.")
        return False


def get_modis_lst(roi: ee.Geometry, start_date: str, end_date: str) -> Optional[ee.Image]:
    try:
        col = (
            ee.ImageCollection("MODIS/061/MOD11A2")
            .filterDate(start_date, end_date)
            .filterBounds(roi)
            .select("LST_Day_1km")
        )
        if col.size().getInfo() == 0:
            logging.warning("No MODIS LST for %s..%s", start_date, end_date)
            return None
        mean_kelvin = col.mean()
        lst_c = (
            mean_kelvin.multiply(0.02).subtract(273.15).focal_median(radius=3, units="pixels").rename("LST_Celsius")
        )
        # CRITICAL FIX: Clip the final image to the ROI
        return lst_c.clip(roi)
    except Exception as exc:
        logging.error("MODIS LST error: %s", exc)
        return None


def get_landsat_lst(roi: ee.Geometry, start_date: str, end_date: str) -> Optional[ee.Image]:
    try:
        col = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterDate(start_date, end_date)
            .filterBounds(roi)
            .select(["ST_B10", "QA_PIXEL"])
        )
        if col.size().getInfo() == 0:
            logging.warning("No Landsat LST for %s..%s", start_date, end_date)
            return None

        def mapper(img: ee.Image) -> ee.Image:
            qa = img.select("QA_PIXEL")
            cloud_mask = (
                qa.bitwiseAnd(1 << 1).eq(0)
                .And(qa.bitwiseAnd(1 << 3).eq(0))
                .And(qa.bitwiseAnd(1 << 4).eq(0))
            )
            st_k = img.select("ST_B10").multiply(0.00341802).add(149.0)
            st_c = st_k.subtract(273.15).rename("LST_Landsat_Celsius")
            return st_c.updateMask(cloud_mask)

        lst = col.map(mapper).mean()
        # CRITICAL FIX: Clip the final image to the ROI
        return lst.clip(roi)
    except Exception as exc:
        logging.error("Landsat LST error: %s", exc)
        return None


def get_landsat_indices(roi: ee.Geometry, start_date: str, end_date: str) -> Tuple[Optional[ee.Image], Optional[ee.Image]]:
    try:
        col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate(start_date, end_date).filterBounds(roi)
        if col.size().getInfo() == 0:
            logging.warning("No Landsat data for indices in %s..%s", start_date, end_date)
            return None, None

        def mask_optical(img: ee.Image) -> ee.Image:
            qa = img.select("QA_PIXEL")
            cloud_mask = (
                qa.bitwiseAnd(1 << 1).eq(0)
                .And(qa.bitwiseAnd(1 << 3).eq(0))
                .And(qa.bitwiseAnd(1 << 4).eq(0))
            )
            optical = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
            scaled = img.select(optical).multiply(0.0000275).add(-0.2)
            return scaled.updateMask(cloud_mask)

        def add_indices(img: ee.Image) -> ee.Image:
            ndvi = img.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
            ndbi = img.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")
            return img.addBands([ndvi, ndbi])

        processed = col.map(mask_optical).map(add_indices)
        ndvi = processed.select("NDVI").mean()
        ndbi = processed.select("NDBI").mean()
        # CRITICAL FIX: Clip the final images to the ROI
        return ndvi.clip(roi), ndbi.clip(roi)
    except Exception as exc:
        logging.error("Landsat indices error: %s", exc)
        return None, None


def get_worldcover_landcover(roi: ee.Geometry, year: int) -> Optional[ee.Image]:
    try:
        if year == 2020:
            img = ee.Image("ESA/WorldCover/v100/2020").select("Map")
        else:
            img = ee.Image("ESA/WorldCover/v200/2021").select("Map")
        # CRITICAL FIX: Clip the final image to the ROI
        return img.clip(roi)
    except Exception as exc:
        logging.error("WorldCover error: %s", exc)
        return None


def get_topographic_features(roi: ee.Geometry) -> Optional[ee.Image]:
    try:
        dem = ee.Image("USGS/SRTMGL1_003")
        elevation = dem.select("elevation").rename("Elevation")
        slope = ee.Terrain.slope(dem).rename("Slope")
        aspect = ee.Terrain.aspect(dem).rename("Aspect")
        combined = elevation.addBands([slope, aspect])
        # CRITICAL FIX: Clip the final image to the ROI
        return combined.clip(roi)
    except Exception as exc:
        logging.error("Topographic features error: %s", exc)
        return None
