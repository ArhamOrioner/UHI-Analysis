# This is parameter script

import ee
import os

PROJECT_NAME = "Urban Heat Island (UHI) Detection & Prediction in New Delhi"
AUTHOR = "Arham Ansari"
DATE = "30-06-2025"

CITY_NAME = "New Delhi"
ROI_BOUNDS = [76.5, 28.1, 78.5, 29.5]
ROI_GEOMETRY = ee.Geometry.Rectangle(ROI_BOUNDS)

ANALYSIS_YEARS = [2022, 2023]
SEASONS_TO_ANALYZE = {
    'Summer': (6, 8),
    'Winter': (12, 2)
}

PRIMARY_LST_SOURCE = 'MODIS' # Can be 'MODIS' or 'LANDSAT'

START_DATE = '2022-01-01'
END_DATE = '2022-12-31'

COMMON_RESOLUTION_METERS = 1000

NUM_PIXELS_FOR_ML = 5000

RANDOM_STATE = 42

UHI_CLUSTERS = 'auto' # or an integer number of clusters, e.g., 3
UHI_CLUSTER_RANGE = range(2, 6) # Range for automatic cluster determination

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 15
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 2

PERFORM_HYPERPARAMETER_TUNING = True
TUNING_ITERATIONS = 50

OUTPUTS_DIR = 'outputs'

WORLDCOVER_YEAR = 2021 # Use 2020 or 2021. Script defaults to 2021 if other year is chosen.

LOG_FILE = os.path.join(OUTPUTS_DIR, 'uhi_analysis.log')
LOG_LEVEL = 'INFO'