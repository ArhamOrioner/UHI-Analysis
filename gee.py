"""Earth Engine preprocessing. All date intervals are [start, end)."""

import logging
import math
import os
from typing import Optional, Tuple

import ee

LANDSAT_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
ANALYSIS_CRS = "EPSG:6933"


def initialize_earth_engine(project: Optional[str] = None) -> bool:
    try:
        ee.Initialize(project=project or os.environ.get("EE_PROJECT"))
        logging.info("Google Earth Engine initialized.")
        return True
    except Exception as exc:
        logging.error("Earth Engine init failed: %s", exc)
        logging.error(
            "Run 'earthengine authenticate' and supply --project YOUR_CLOUD_PROJECT."
        )
        return False


def landsat_quality_mask(img: ee.Image) -> ee.Image:
    # QA_PIXEL bits 0..5: fill, dilated cloud, cirrus, cloud, shadow, snow.
    # Conservative QA_RADSAT policy: reject any saturation or terrain occlusion.
    return (
        img.select("QA_PIXEL").bitwiseAnd(63).eq(0).And(img.select("QA_RADSAT").eq(0))
    )


def normalized_index(
    img: ee.Image, first: str, second: str, name: str, epsilon: float = 1e-6
) -> ee.Image:
    """Retain negative reflectances; explicitly mask near-zero denominators.

    Unlike normalizedDifference(), expression() does not silently reject negative
    inputs. Indices may lie outside [-1, 1]; they are intentionally not clamped.
    epsilon is in scaled reflectance units, not raw DN units.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Index denominator epsilon must be positive")
    a, b = img.select(first), img.select(second)
    valid = a.add(b).abs().gt(epsilon)
    return img.expression(
        "(a - b) / (a + b)",
        {
            "a": a.updateMask(valid),
            "b": b.updateMask(valid),
        },
    ).rename(name)


def prepare_landsat_indices(img: ee.Image, epsilon: float = 1e-6) -> ee.Image:
    sr = (
        img.select(["SR_B4", "SR_B5", "SR_B6"])
        .multiply(0.0000275)
        .add(-0.2)
        .updateMask(landsat_quality_mask(img))
    )
    ndvi = normalized_index(sr, "SR_B5", "SR_B4", "NDVI", epsilon)
    ndbi = normalized_index(sr, "SR_B6", "SR_B5", "NDBI", epsilon)
    return ndvi.addBands(ndbi).set("system:time_start", img.get("system:time_start"))


def prepare_modis_lst(img: ee.Image) -> ee.Image:
    qc = img.select("QC_Day")
    # Good mandatory QA, good data quality, emissivity error <=0.01, LST error <=1K.
    good = qc.bitwiseAnd(255).eq(0)
    raw = img.select("LST_Day_1km")
    return (
        raw.updateMask(good.And(raw.gte(7500)))
        .multiply(0.02)
        .subtract(273.15)
        .rename("LST_Celsius")
    )


def prepare_landsat_lst(img: ee.Image) -> ee.Image:
    raw = img.select("ST_B10")
    return (
        raw.updateMask(landsat_quality_mask(img).And(raw.gt(0)))
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
        .rename("LST_Celsius")
    )


def _collection(dataset, roi, start_date, end_date):
    col = ee.ImageCollection(dataset).filterDate(start_date, end_date).filterBounds(roi)
    return col.sort("system:time_start")


def _mean_with_count(col, band, projection, roi):
    # Composites otherwise default to a 1-degree WGS84 projection.
    mean = col.select(band).mean().setDefaultProjection(projection)
    count = (
        col.select(band)
        .count()
        .rename(band + "_count")
        .setDefaultProjection(projection)
    )
    return mean.addBands(count).clip(roi)


def get_modis_lst(
    roi: ee.Geometry, start_date: str, end_date: str
) -> Optional[ee.Image]:
    col = _collection("MODIS/061/MOD11A2", roi, start_date, end_date)
    if col.size().getInfo() == 0:
        logging.warning("No MODIS scenes for %s..%s", start_date, end_date)
        return None
    projection = ee.Image(col.first()).select("LST_Day_1km").projection()
    # No focal smoothing: it changes temperatures and can obscure spatial extremes.
    return _mean_with_count(col.map(prepare_modis_lst), "LST_Celsius", projection, roi)


def get_landsat_lst(
    roi: ee.Geometry, start_date: str, end_date: str
) -> Optional[ee.Image]:
    col = _collection(LANDSAT_COLLECTION, roi, start_date, end_date).filter(
        ee.Filter.eq("PROCESSING_LEVEL", "L2SP")
    )
    if col.size().getInfo() == 0:
        logging.warning("No Landsat L2SP scenes for %s..%s", start_date, end_date)
        return None
    projection = ee.Image(col.first()).select("ST_B10").projection()
    return _mean_with_count(
        col.map(prepare_landsat_lst), "LST_Celsius", projection, roi
    )


def get_landsat_indices(
    roi: ee.Geometry, start_date: str, end_date: str, epsilon: float = 1e-6
) -> Tuple[Optional[ee.Image], Optional[ee.Image]]:
    col = _collection(LANDSAT_COLLECTION, roi, start_date, end_date)
    if col.size().getInfo() == 0:
        logging.warning("No Landsat scenes for %s..%s", start_date, end_date)
        return None, None
    projection = ee.Image(col.first()).select("SR_B5").projection()
    processed = col.map(lambda img: prepare_landsat_indices(img, epsilon))
    return tuple(
        _mean_with_count(processed, band, projection, roi) for band in ("NDVI", "NDBI")
    )


def index_mask_diagnostics(roi, start_date, end_date, epsilon=1e-6, scale_m=1000):
    """Estimate the negative-input masking condition on a fixed diagnostic grid.

    Reports sums of valid scene-pixel observations, not unique pixels or area.
    Both variants use identical QA and denominator masks to isolate EWL203.
    """
    col = _collection(LANDSAT_COLLECTION, roi, start_date, end_date)
    if col.size().getInfo() == 0:
        return {}

    def counts(img):
        sr = img.select(["SR_B4", "SR_B5", "SR_B6"]).multiply(0.0000275).add(-0.2)
        fixed = prepare_landsat_indices(img, epsilon)
        result = []
        for name, a, b in (("NDVI", "SR_B5", "SR_B4"), ("NDBI", "SR_B6", "SR_B5")):
            valid = fixed.select(name).mask()
            lost = valid.And(sr.select(a).lt(0).Or(sr.select(b).lt(0)))
            result.extend(
                [valid.rename(name + "_valid"), lost.rename(name + "_negative_input")]
            )
        return ee.Image.cat(result).unmask(0)

    return (
        col.map(counts)
        .sum()
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            crs=ANALYSIS_CRS,
            scale=scale_m,
            maxPixels=10000000,
            tileScale=4,
        )
        .getInfo()
    )


def get_worldcover_landcover(roi: ee.Geometry, year: int) -> ee.Image:
    if year not in (2020, 2021):
        raise ValueError("WorldCover supports only 2020 or 2021")
    version = "v100" if year == 2020 else "v200"
    return ee.Image(f"ESA/WorldCover/{version}/{year}").select("Map").clip(roi)


def get_topographic_features(roi: ee.Geometry) -> ee.Image:
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    aspect = ee.Terrain.aspect(dem).multiply(3.141592653589793 / 180)
    # Circular encoding avoids averaging 359 and 1 degrees to 180 degrees.
    return (
        dem.rename("Elevation")
        .addBands(
            [
                ee.Terrain.slope(dem).rename("Slope"),
                aspect.sin().rename("Aspect_sin"),
                aspect.cos().rename("Aspect_cos"),
            ]
        )
        .clip(roi)
    )


def to_analysis_grid(
    image: ee.Image, scale_m: int, categorical: bool = False
) -> ee.Image:
    """Area-weighted mean for continuous bands, mode for land-cover classes.

    Input must have a meaningful native/default projection. Counts become mean
    valid observations per native pixel, and can therefore be fractional.
    """
    reducer = ee.Reducer.mode() if categorical else ee.Reducer.mean()
    return image.reduceResolution(reducer=reducer, maxPixels=65536).reproject(
        crs=ANALYSIS_CRS, scale=scale_m
    )
