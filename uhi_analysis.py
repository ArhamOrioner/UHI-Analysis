# -----------------------------------------------------------------------------
# Project: Urban Heat Island (UHI) Detection & Prediction in New Delhi
# Author: Arham Ansari
# Date: 07-07-2025
# Description: This comprehensive script leverages Google Earth Engine (GEE)
#              to acquire satellite imagery (LST, NDVI, NDBI, Land Cover),
#              performs data cleaning, exploratory data analysis (EDA),
#              detects Urban Heat Island zones using K-Means clustering,
#              and predicts Land Surface Temperature (LST) using Random Forest.
#              Results are visualized on interactive maps and plots.
# -----------------------------------------------------------------------------

import ee
import os
import geemap
import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import shap

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

import libpysal.weights as weights
from esda.getisord import G_Local

import warnings
warnings.filterwarnings('ignore')

# Set global plotting styles for enhanced visualization
sns.set_style("whitegrid") # White background with grid
plt.rcParams['figure.figsize'] = (10, 6) # Default figure size for matplotlib plots
plt.rcParams['font.size'] = 12 # Default font size for plot elements

# --- PARAMETER SCRIPT SECTION ---
# All project configuration parameters are defined here for easy modification.

PROJECT_NAME = "Urban Heat Island (UHI) Detection & Prediction in New Delhi"
AUTHOR = "Arham Ansari"
DATE = "30-06-2025"

CITY_NAME = "Mumbai"
# Define the geographic bounding box for the Region of Interest (ROI)
# Format: [west_longitude, south_latitude, east_longitude, north_latitude]
ROI_BOUNDS = [72.7, 18.8, 73.2, 19.3] # Mumbai example (replace with your exact values)
# ROI_GEOMETRY (ee.Geometry.Rectangle) will be defined after ee.Initialize()

# Years and seasons to include in the analysis
ANALYSIS_YEARS = [2022, 2023]
SEASONS_TO_ANALYZE = {
    'Summer': (6, 8),   # Months: June (6), July (7), August (8)
    'Winter': (12, 2)   # Months: December (12), January (1), February (2) - spans across years
}

PRIMARY_LST_SOURCE = 'MODIS' # Can be 'MODIS' or 'LANDSAT'

# Overall START_DATE and END_DATE are used for general map layers that show combined data.
# Seasonal analysis uses dates derived from ANALYSIS_YEARS and SEASONS_TO_ANALYZE.
START_DATE = '2022-01-01'
END_DATE = '2023-12-31' # Set to cover all years defined in ANALYSIS_YEARS

COMMON_RESOLUTION_METERS = 1000 # Spatial resolution for data resampling (e.g., 1000m for MODIS)

NUM_PIXELS_FOR_ML = 5000 # Number of pixels to sample for machine learning model training

RANDOM_STATE = 42 # Seed for reproducibility of random operations

UHI_CLUSTERS = 'auto' # 'auto' for automatic determination using Silhouette Score, or an integer (e.g., 3)
UHI_CLUSTER_RANGE = range(2, 6) # Range of cluster numbers to test if UHI_CLUSTERS is 'auto' (e.g., 2, 3, 4, 5 clusters)

# Random Forest Regressor hyperparameters
RF_N_ESTIMATORS = 200 # Number of trees in the forest
RF_MAX_DEPTH = 15 # Maximum depth of the tree
RF_MIN_SAMPLES_SPLIT = 5 # Minimum number of samples required to split an internal node
RF_MIN_SAMPLES_LEAF = 2 # Minimum number of samples required to be at a leaf node

PERFORM_HYPERPARAMETER_TUNING = True # Set to True to perform RandomizedSearchCV
TUNING_ITERATIONS = 50 # Number of parameter settings that are sampled in RandomizedSearchCV

OUTPUTS_DIR = 'outputs' # Directory to save all output plots and maps

WORLDCOVER_YEAR = 2021 # Year for ESA WorldCover data (options: 2020, 2021). Defaults to 2021 if invalid.

# Logging configuration
LOG_FILE = os.path.join(OUTPUTS_DIR, 'uhi_analysis.log')
LOG_LEVEL = 'INFO' # Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'


# --- MAIN SCRIPT LOGIC ---

# 1. Google Earth Engine Initialization
try:
    ee.Initialize()
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print(f"Error initializing Google Earth Engine: {e}")
    print("ACTION REQUIRED: Please ensure you have authenticated by running 'earthengine authenticate' in your terminal.")
    print("Exiting script.")
    exit()

# Define ROI_GEOMETRY after ee.Initialize() as it requires ee.Geometry
ROI_GEOMETRY = ee.Geometry.Rectangle(ROI_BOUNDS)
ROI = ROI_GEOMETRY # Use 'ROI' as the consistent variable name for clarity

# 2. Output Directory and Logging Setup
os.makedirs(OUTPUTS_DIR, exist_ok=True) # Create output directory if it doesn't exist

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(), # Output logs to console
        logging.FileHandler(LOG_FILE) # Output logs to a file
    ]
)
logging.info(f"Output directory created at: {OUTPUTS_DIR}")
logging.info(f"Log file created at: {LOG_FILE}\n")

logging.info(f"\nTargeting: {CITY_NAME} for analysis from {START_DATE} to {END_DATE}")
logging.info(f"Analysis Resolution: {COMMON_RESOLUTION_METERS}m")

# 3. GEE Data Fetching Functions

def get_modis_lst(roi, start_date, end_date):
    """
    Fetches and processes MODIS LST (MOD11A2) data for the specified ROI and date range.
    Converts LST from Kelvin to Celsius and applies a focal median filter.
    """
    logging.info(f"Fetching MODIS LST data for {start_date} to {end_date}...")
    try:
        lst_collection = ee.ImageCollection('MODIS/061/MOD11A2') \
                           .filterDate(start_date, end_date) \
                           .filterBounds(roi) \
                           .select('LST_Day_1km')

        if lst_collection.size().getInfo() == 0:
            logging.warning(f"No MODIS LST images found for {start_date} to {end_date} in ROI. Check dates/bounds.")
            return None

        mean_lst_kelvin = lst_collection.mean()
        
        # Scale LST to Celsius (MODIS LST_Day_1km scale factor is 0.02)
        # Apply focal median for smoothing out noise
        mean_lst_celsius = mean_lst_kelvin.multiply(0.02).subtract(273.15) \
                                         .focal_median(radius=3, units='pixels') \
                                         .rename('LST_Celsius')
        
        clipped_lst = mean_lst_celsius.clip(roi)
        logging.info("MODIS LST data processed successfully.")
        return clipped_lst
    except Exception as e:
        logging.error(f"Error processing MODIS LST data: {e}")
        return None

def get_landsat_lst(roi, start_date, end_date):
    """
    Fetches and processes Landsat 8 Surface Temperature (ST_B10) data.
    Applies cloud masking, scales LST from Kelvin to Celsius.
    """
    logging.info(f"Fetching Landsat 8 Surface Temperature data for {start_date} to {end_date}...")
    try:
        landsat_collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                               .filterDate(start_date, end_date) \
                               .filterBounds(roi) \
                               .select('ST_B10', 'QA_PIXEL') # Select thermal band and QA band

        if landsat_collection.size().getInfo() == 0:
            logging.warning(f"No Landsat 8 LST images found for {start_date} to {end_date} in ROI. Check dates/bounds.")
            return None

        def mask_clouds_landsat8_c2_thermal(image):
            """Applies cloud masking and converts ST_B10 to Celsius."""
            qa_pixel = image.select('QA_PIXEL')
            # Bits 1 (Dilated Cloud), 3 (Cloud), 4 (Cloud Shadow) from QA_PIXEL band
            cloud_mask = qa_pixel.bitwiseAnd(1 << 1).eq(0) \
                          .And(qa_pixel.bitwiseAnd(1 << 3).eq(0)) \
                          .And(qa_pixel.bitwiseAnd(1 << 4).eq(0))
            
            # Scale LST to Kelvin (Landsat 8 Collection 2 Level 2 ST_B10 conversion formula)
            st_kelvin = image.select('ST_B10').multiply(0.00341802).add(149.0)
            st_celsius = st_kelvin.subtract(273.15) # Convert Kelvin to Celsius
            return st_celsius.updateMask(cloud_mask).rename('LST_Landsat_Celsius')

        landsat_lst_processed = landsat_collection.map(mask_clouds_landsat8_c2_thermal)
        
        mean_lst_landsat = landsat_lst_processed.mean().clip(roi)
        
        logging.info("Landsat LST data processed successfully.")
        return mean_lst_landsat
    except Exception as e:
        logging.error(f"Error processing Landsat LST data: {e}")
        return None

def get_landsat_indices(roi, start_date, end_date):
    """
    Fetches Landsat 8 optical data, applies cloud masking, and derives NDVI and NDBI.
    """
    logging.info(f"Fetching Landsat 8 data for {start_date} to {end_date} for indices...")
    try:
        landsat_collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                               .filterDate(start_date, end_date) \
                               .filterBounds(roi)

        if landsat_collection.size().getInfo() == 0:
            logging.warning(f"No Landsat 8 optical images found for {start_date} to {end_date} in ROI. Check dates/bounds.")
            return None, None

        def mask_clouds_landsat8_c2_optical(image):
            """Applies cloud masking to optical bands."""
            qa_pixel = image.select('QA_PIXEL')
            # Bits 1 (Dilated Cloud), 3 (Cloud), 4 (Cloud Shadow) from QA_PIXEL band
            cloud_mask = qa_pixel.bitwiseAnd(1 << 1).eq(0) \
                          .And(qa_pixel.bitwiseAnd(1 << 3).eq(0)) \
                          .And(qa_pixel.bitwiseAnd(1 << 4).eq(0))

            optical_bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
            # Scale optical bands (Landsat 8 Collection 2 Level 2 Surface Reflectance)
            image = image.select(optical_bands).multiply(0.0000275).add(-0.2)
            return image.updateMask(cloud_mask)

        def add_indices(image):
            """Calculates NDVI and NDBI for the image."""
            ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI') # NIR (B5), Red (B4)
            ndbi = image.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI') # SWIR1 (B6), NIR (B5)
            return image.addBands([ndvi, ndbi])

        landsat_processed = landsat_collection.map(mask_clouds_landsat8_c2_optical)
        landsat_with_indices = landsat_processed.map(add_indices)

        mean_ndvi = landsat_with_indices.select('NDVI').mean().clip(roi)
        mean_ndbi = landsat_with_indices.select('NDBI').mean().clip(roi)

        logging.info("Landsat NDVI and NDBI data processed successfully.")
        return mean_ndvi, mean_ndbi
    except Exception as e:
        logging.error(f"Error processing Landsat data for indices: {e}")
        return None, None

def get_worldcover_landcover(roi, year):
    """
    Fetches ESA WorldCover land cover data for the specified year.
    Defaults to 2021 if an unsupported year is provided.
    """
    logging.info(f"Fetching ESA WorldCover data for {year}...")
    try:
        # Prioritize 2021 if requested and available, else fall back to 2020 or default 2021
        if year == 2020:
            landcover_image = ee.Image('ESA/WorldCover/v100/2020').select('Map')
        elif year == 2021:
            landcover_image = ee.Image('ESA/WorldCover/v200/2021').select('Map')
        else:
            logging.warning(f"WorldCover data not available for {year}. Using 2021 (v200) as default.")
            landcover_image = ee.Image('ESA/WorldCover/v200/2021').select('Map')

        clipped_landcover = landcover_image.clip(roi).rename('Land_Cover')
        logging.info("ESA WorldCover data processed successfully.")
        return clipped_landcover
    except Exception as e:
        logging.error(f"Error processing WorldCover data: {e}")
        return None

def get_topographic_features(roi):
    """
    Fetches SRTM DEM (Digital Elevation Model) and derives Elevation, Slope, and Aspect.
    """
    logging.info("Fetching SRTM DEM and deriving topographic features (Elevation, Slope, Aspect)...")
    try:
        dem = ee.Image('USGS/SRTMGL1_003').clip(roi)
        elevation = dem.select('elevation').rename('Elevation')
        slope = ee.Terrain.slope(dem).rename('Slope')
        aspect = ee.Terrain.aspect(dem).rename('Aspect')
        
        combined_topo = elevation.addBands([slope, aspect])
        logging.info("Topographic features processed successfully.")
        return combined_topo
    except Exception as e:
        logging.error(f"Error processing topographic data: {e}")
        return None

def sample_to_dataframe(image, roi, num_pixels, scale):
    """
    Samples points from a Google Earth Engine Image and converts them to a Pandas DataFrame.
    Includes explicit extraction of latitude and longitude.
    """
    logging.info(f"Sampling {num_pixels} pixels at {int(scale)}m resolution...")
    try:
        # Use geometries=True to ensure point geometries are available for local analysis
        sample_points_fc = image.sample(
            region=roi,
            scale=scale,
            numPixels=num_pixels,
            seed=RANDOM_STATE,
            geometries=True 
        )
        
        # Explicitly set longitude and latitude as properties in the FeatureCollection
        # This ensures they are included when converting to DataFrame
        sample_points_fc = sample_points_fc.map(lambda f: f.set({
            'longitude': f.geometry().coordinates().get(0),
            'latitude': f.geometry().coordinates().get(1)
        }))
        
        # Convert the FeatureCollection to a Pandas DataFrame using geemap
        df = geemap.ee_to_df(sample_points_fc)
        
        if df.empty:
            raise ValueError("No data was sampled from Earth Engine. This might indicate issues with the ROI, date ranges, or strict data masking.")
        
        # Final check for latitude and longitude columns
        if 'latitude' in df.columns and 'longitude' in df.columns:
            logging.info(f"Successfully extracted {len(df)} pixels into a DataFrame with latitude/longitude.")
            return df
        else:
            logging.error("Latitude or Longitude columns not found in DataFrame after explicit mapping. Check GEE sampling and geometry extraction.")
            return None
            
    except Exception as e:
        logging.error(f"Error during data sampling to DataFrame: {e}")
        return None

# 4. Main Data Acquisition and Pre-processing Loop
all_data_dfs = [] # List to store DataFrames from each analysis period

for year in ANALYSIS_YEARS:
    for season, (start_month, end_month) in SEASONS_TO_ANALYZE.items():
        logging.info(f"\n--- Processing {year} - {season} ---")

        start_date_season = f'{year}-{start_month:02d}-01'
        
        # Determine end date for the season, handling cross-year seasons (e.g., Winter)
        if start_month > end_month: 
            end_year = year + 1
            # Get the last day of the month for the end_date_season
            end_date_season = f'{end_year}-{end_month:02d}-{pd.Timestamp(f"{end_year}-{end_month:02d}-01").days_in_month}'
            logging.warning(f"Season '{season}' spans across years. Date range: {start_date_season} to {end_date_season}.")
        else:
            end_date_season = f'{year}-{end_month:02d}-{pd.Timestamp(f"{year}-{end_month:02d}-01").days_in_month}'

        logging.info(f"Fetching data for {start_date_season} to {end_date_season}...")

        # Fetch LST based on the configured primary source
        lst_image = None
        lst_band_name_in_df = None
        if PRIMARY_LST_SOURCE == 'MODIS':
            lst_image = get_modis_lst(ROI, start_date_season, end_date_season)
            lst_band_name_in_df = 'LST_Celsius' # Expected name if from MODIS
        elif PRIMARY_LST_SOURCE == 'LANDSAT':
            lst_image = get_landsat_lst(ROI, start_date_season, end_date_season)
            lst_band_name_in_df = 'LST_Landsat_Celsius' # Expected name if from Landsat
        else:
            logging.error(f"Invalid PRIMARY_LST_SOURCE: '{PRIMARY_LST_SOURCE}'. Must be 'MODIS' or 'LANDSAT'. Exiting.")
            exit() # Terminate if configuration is invalid

        # Fetch other required geospatial layers
        ndvi_image, ndbi_image = get_landsat_indices(ROI, start_date_season, end_date_season)
        landcover_image = get_worldcover_landcover(ROI, WORLDCOVER_YEAR)
        topographic_image = get_topographic_features(ROI)

        # Check if all required images were acquired successfully for the current period
        if any(img is None for img in [lst_image, ndvi_image, ndbi_image, landcover_image, topographic_image]):
            logging.error(f"ERROR: Failed to acquire one or more essential datasets for {year}-{season}. Skipping this period.")
            continue # Skip to the next period in the loop

        # Combine all images into a single multi-band image for efficient sampling
        combined_image = lst_image.addBands([ndvi_image, ndbi_image, landcover_image, topographic_image])
        logging.info("All data layers combined successfully for the current period.")

        # Sample combined image to a Pandas DataFrame
        df_pixels_current_period = sample_to_dataframe(combined_image, ROI, NUM_PIXELS_FOR_ML, COMMON_RESOLUTION_METERS)

        if df_pixels_current_period is None or df_pixels_current_period.empty:
            logging.error(f"ERROR: Failed to create DataFrame for {year}-{season}. Skipping this period.")
            continue # Skip if sampling failed or resulted in empty data

        # Rename LST column to a consistent 'LST_Celsius' name if it came from Landsat
        if PRIMARY_LST_SOURCE == 'LANDSAT' and lst_band_name_in_df != 'LST_Celsius':
            if lst_band_name_in_df in df_pixels_current_period.columns:
                df_pixels_current_period.rename(columns={lst_band_name_in_df: 'LST_Celsius'}, inplace=True)
                logging.info(f"Renamed LST column from '{lst_band_name_in_df}' to 'LST_Celsius' for consistency.")
            else:
                logging.error(f"LST band '{lst_band_name_in_df}' not found in DataFrame columns after Landsat acquisition. Cannot proceed for {year}-{season}.")
                continue

        # Data Cleaning and Feature Engineering
        logging.info("\n--- Data Cleaning for current period ---")
        initial_rows = len(df_pixels_current_period)

        # Drop rows with any NaN values across all feature columns
        df_pixels_current_period.dropna(inplace=True)
        logging.info(f"Removed {initial_rows - len(df_pixels_current_period)} rows with NaN values.")

        # Filter for valid WorldCover land cover classes (10-95, as per WorldCover specification)
        valid_lc_classes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
        df_pixels_current_period = df_pixels_current_period[df_pixels_current_period['Land_Cover'].isin(valid_lc_classes)]
        logging.info(f"Filtered to {len(df_pixels_current_period)} rows with valid land cover classes.")

        # Map numerical Land_Cover IDs to descriptive string names
        land_cover_mapping = {
            10: 'Trees', 20: 'Shrubland', 30: 'Grassland', 40: 'Cropland', 
            50: 'Built-up', 60: 'Bare_Vegetation', 70: 'Snow_Ice', 80: 'Water', 
            90: 'Wetland', 95: 'Mangroves'
        }
        df_pixels_current_period['Land_Cover_Type'] = df_pixels_current_period['Land_Cover'].map(land_cover_mapping)
        df_pixels_current_period.drop(columns=['Land_Cover'], inplace=True) # Remove the numerical Land_Cover ID column

        # Add Year and Season columns to distinguish data from different periods
        df_pixels_current_period['Year'] = year
        df_pixels_current_period['Season'] = season
        
        all_data_dfs.append(df_pixels_current_period)

# Check if any data was successfully processed after the loop
if not all_data_dfs:
    logging.error("No data was successfully processed for any year/season. Exiting script as no data available for analysis.")
    exit()

# Concatenate all DataFrames collected from different periods into a single DataFrame
df_combined_all_periods = pd.concat(all_data_dfs, ignore_index=True)

# Use `df_pixels` as the primary DataFrame for all subsequent analysis steps
df_pixels = df_combined_all_periods
logging.info(f"\n--- Combined Data Summary (All Periods) ---")
logging.info(f"Total DataFrame shape after combining all periods: {df_pixels.shape}")

# Basic Data Summary after combination
logging.info("\n--- Data Summary ---")
logging.info(f"\nDataFrame shape: {df_pixels.shape}")
logging.info("\nLand Cover Distribution:\n" + str(df_pixels['Land_Cover_Type'].value_counts()))
logging.info("\nDescriptive Statistics:\n" + str(df_pixels.describe()))
logging.info("\nFirst 5 rows of data:\n" + str(df_pixels.head()))

# 5. Enhanced EDA and Visualization

def create_enhanced_plots(df):
    """
    Generates and saves a series of enhanced Exploratory Data Analysis (EDA) plots
    to visualize feature distributions, correlations, and relationships.
    """
    # Adjust plot parameters for consistent font sizes within the function
    plt.rcParams['font.size'] = 12 
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12)) # Larger figure for better subplot visibility
    
    # Plot 1: Distribution of Land Surface Temperature
    sns.histplot(data=df, x='LST_Celsius', kde=True, ax=axes[0,0], color='skyblue', edgecolor='black')
    axes[0,0].set_title('Distribution of Land Surface Temperature', fontsize=14)
    axes[0,0].set_xlabel('Temperature (°C)', fontsize=12)
    axes[0,0].set_ylabel('Frequency', fontsize=12)
    axes[0,0].grid(True, linestyle='--', alpha=0.6) # Add grid for better readability
    
    # Plot 2: Distribution of Normalized Difference Vegetation Index (NDVI)
    sns.histplot(data=df, x='NDVI', kde=True, ax=axes[0,1], color='lightgreen', edgecolor='black')
    axes[0,1].set_title('Distribution of NDVI', fontsize=14)
    axes[0,1].set_xlabel('NDVI', fontsize=12)
    axes[0,1].set_ylabel('Frequency', fontsize=12)
    axes[0,1].grid(True, linestyle='--', alpha=0.6)
    
    # Plot 3: Distribution of Normalized Difference Built-up Index (NDBI)
    sns.histplot(data=df, x='NDBI', kde=True, ax=axes[1,0], color='lightcoral', edgecolor='black')
    axes[1,0].set_title('Distribution of NDBI', fontsize=14)
    axes[1,0].set_xlabel('NDBI', fontsize=12)
    axes[1,0].set_ylabel('Frequency', fontsize=12)
    axes[1,0].grid(True, linestyle='--', alpha=0.6)
    
    # Plot 4: LST by Land Cover Type - using a more distinct and thematic palette
    land_cover_palette = {
        'Trees': '#228B22', 'Shrubland': '#DAA520', 'Grassland': '#ADFF2F',
        'Cropland': '#FFD700', 'Built-up': '#A52A2A', 'Bare_Vegetation': '#C0C0C0',
        'Snow_Ice': '#ADD8E6', 'Water': '#4682B4', 'Wetland': '#8FBC8F',
        'Mangroves': '#556B2F'
    }
    # Filter the palette to only include land cover types actually present in the DataFrame
    present_lc_types = df['Land_Cover_Type'].unique()
    filtered_lc_palette = {lc: color for lc, color in land_cover_palette.items() if lc in present_lc_types}

    sns.boxplot(data=df, x='Land_Cover_Type', y='LST_Celsius', ax=axes[1,1], palette=filtered_lc_palette, linewidth=1.5, fliersize=3)
    axes[1,1].set_title('LST by Land Cover Type', fontsize=14)
    axes[1,1].set_xlabel('Land Cover Type', fontsize=12)
    axes[1,1].set_ylabel('Temperature (°C)', fontsize=12)
    axes[1,1].tick_params(axis='x', rotation=45) # Rotate x-axis labels for better fit
    axes[1,1].grid(True, linestyle='--', alpha=0.6, axis='y') # Grid only on y-axis
    
    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    plt.savefig(os.path.join(OUTPUTS_DIR, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
    logging.info(f"Saved feature distributions plot to {OUTPUTS_DIR}/feature_distributions.png")
    plt.close() # Close plot to free memory

    # Plot 5: Correlation Matrix of Environmental Variables
    plt.figure(figsize=(10, 8))
    numerical_cols = ['LST_Celsius', 'NDVI', 'NDBI', 'Elevation', 'Slope', 'Aspect']
    numerical_cols_present = [col for col in numerical_cols if col in df.columns] # Select only existing numerical columns
    correlation_matrix = df[numerical_cols_present].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', center=0, # 'viridis' is a perceptually uniform colormap
                square=True, linewidths=0.5, fmt=".2f", cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Matrix of Environmental Variables', fontsize=16)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    logging.info(f"Saved correlation matrix plot to {OUTPUTS_DIR}/correlation_matrix.png")
    plt.close()
    
    # Plot 6 & 7: Scatter plots of LST vs NDVI and NDBI, colored by Land Cover Type
    fig, axes = plt.subplots(1, 2, figsize=(18, 8)) # Wider figure for two subplots
    
    sns.scatterplot(data=df, x='NDVI', y='LST_Celsius', hue='Land_Cover_Type', 
                   alpha=0.7, ax=axes[0], legend='full', palette=filtered_lc_palette, s=30, edgecolor='none')
    axes[0].set_title('LST vs NDVI by Land Cover', fontsize=14)
    axes[0].set_xlabel('NDVI', fontsize=12)
    axes[0].set_ylabel('Temperature (°C)', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    sns.scatterplot(data=df, x='NDBI', y='LST_Celsius', hue='Land_Cover_Type',
                   alpha=0.7, ax=axes[1], legend='full', palette=filtered_lc_palette, s=30, edgecolor='none')
    axes[1].set_title('LST vs NDBI by Land Cover', fontsize=14)
    axes[1].set_xlabel('NDBI', fontsize=12)
    axes[1].set_ylabel('Temperature (°C)', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'scatter_relationships.png'), dpi=300, bbox_inches='tight')
    logging.info(f"Saved scatter relationships plot to {OUTPUTS_DIR}/scatter_relationships.png")
    plt.close()

create_enhanced_plots(df_pixels) # Call the function to generate plots

# 6. Enhanced UHI Detection (K-Means Clustering)

logging.info("\n--- Enhanced UHI Detection ---")

optimal_n_clusters = 3 # Default starting point if auto-detection fails or is skipped

if UHI_CLUSTERS == 'auto':
    logging.info("Determining optimal number of clusters using Silhouette Score...")
    silhouette_scores = []
    
    # Determine minimum samples needed for silhouette score calculation given UHI_CLUSTER_RANGE
    min_samples_for_silhouette = max(UHI_CLUSTER_RANGE) + 1 if UHI_CLUSTER_RANGE else 2
    
    # Check if there are enough samples in the DataFrame for clustering
    if len(df_pixels) < min_samples_for_silhouette:
        logging.warning(f"Not enough samples ({len(df_pixels)}) for requested cluster range {list(UHI_CLUSTER_RANGE)}. Minimum required: {min_samples_for_silhouette}. Skipping auto-detection.")
        # If auto-detection is skipped, ensure optimal_n_clusters is at least 2 if data allows
        if len(df_pixels) >= 2:
            optimal_n_clusters = 2 
            logging.info(f"Adjusted optimal_n_clusters to {optimal_n_clusters} due to insufficient samples for full auto-detection.")
        else:
            logging.error("Not enough samples for even basic clustering. Cannot proceed with UHI detection.")
            optimal_n_clusters = 0 # Indicate failure to cluster meaningfully
    else:
        valid_k_values_for_plot = []
        for n_c in UHI_CLUSTER_RANGE:
            if n_c <= 1 or n_c > len(df_pixels): continue # Ensure k is valid and within data bounds
            try:
                kmeans_temp = KMeans(n_clusters=n_c, random_state=RANDOM_STATE, n_init=10) # n_init suppresses warning
                cluster_labels_temp = kmeans_temp.fit_predict(df_pixels[['LST_Celsius']])
                
                # Silhouette score requires at least 2 unique clusters to be formed
                if len(np.unique(cluster_labels_temp)) < 2:
                    logging.warning(f"  Only 1 unique cluster formed for {n_c} clusters, skipping silhouette score for this k.")
                    continue
                
                score = silhouette_score(df_pixels[['LST_Celsius']], cluster_labels_temp)
                silhouette_scores.append(score)
                valid_k_values_for_plot.append(n_c)
                logging.info(f"  Clusters: {n_c}, Silhouette Score: {score:.3f}")
            except Exception as e:
                logging.warning(f"  Could not calculate silhouette score for {n_c} clusters: {e}")

        if silhouette_scores and valid_k_values_for_plot:
            optimal_n_clusters = valid_k_values_for_plot[np.argmax(silhouette_scores)] # Select k with highest silhouette score
            logging.info(f"Optimal number of clusters (Silhouette Score): {optimal_n_clusters}")

            # Plot Silhouette Scores
            plt.figure(figsize=(8, 5))
            plt.plot(valid_k_values_for_plot, silhouette_scores, marker='o', color='rebeccapurple', linewidth=2, markersize=8)
            plt.xlabel("Number of Clusters (k)", fontsize=12)
            plt.ylabel("Silhouette Score", fontsize=12)
            plt.title("Silhouette Score for Optimal K", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(OUTPUTS_DIR, 'silhouette_score.png'), dpi=300, bbox_inches='tight')
            logging.info(f"Saved Silhouette Score plot to {OUTPUTS_DIR}/silhouette_score.png")
            plt.close()
        else:
            logging.warning(f"Failed to auto-detect optimal clusters. Using default {optimal_n_clusters}. Data characteristics or K-means behavior might be an issue.")
            # If auto-detection failed, try to ensure optimal_n_clusters is at least 2 if data allows
            if len(df_pixels) >= 2 and optimal_n_clusters < 2:
                optimal_n_clusters = 2
                logging.info(f"Adjusted to {optimal_n_clusters} clusters as auto-detection failed but data supports it.")
            elif len(df_pixels) < 2:
                 optimal_n_clusters = 0 # Cannot cluster

else: # If UHI_CLUSTERS is an integer (manual setting)
    optimal_n_clusters = int(UHI_CLUSTERS) 
    logging.info(f"Using fixed number of clusters: {optimal_n_clusters}")

# Final check before performing K-Means clustering
if optimal_n_clusters < 2:
    logging.error("Not enough samples or clusters (min 2 required) to perform meaningful UHI detection. Exiting script.")
    exit()

if optimal_n_clusters > len(df_pixels):
    logging.warning(f"Optimal clusters ({optimal_n_clusters}) is greater than number of samples ({len(df_pixels)}). Adjusting clusters to max possible ({len(df_pixels) - 1}).")
    optimal_n_clusters = max(2, len(df_pixels) - 1) # Ensure at least 2 clusters and not more than (samples - 1)
    if optimal_n_clusters < 2:
        logging.error("Not enough samples for clustering after adjustment. Exiting script.")
        exit()

# Perform K-Means clustering on LST values
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=RANDOM_STATE, n_init=10)
df_pixels['LST_Cluster'] = kmeans.fit_predict(df_pixels[['LST_Celsius']])

# Order clusters by their mean LST for logical zone naming (Cool_Zone, Mild_Zone, Hot_Zone)
cluster_centers = kmeans.cluster_centers_.flatten()
cluster_order = np.argsort(cluster_centers) # Returns the indices that would sort the cluster centers

# Define UHI zone names based on the number of optimal clusters
cluster_labels = {}
if optimal_n_clusters == 2:
    zone_names = ['Cool_Zone', 'Hot_Zone']
elif optimal_n_clusters == 3:
    zone_names = ['Cool_Zone', 'Mild_Zone', 'Hot_Zone']
elif optimal_n_clusters == 4:
    zone_names = ['Cool_Zone', 'Mild_Zone_1', 'Mild_Zone_2', 'Hot_Zone']
else: # For more than 4 clusters, use generic Zone_X names
    zone_names = [f'Zone_{i+1}' for i in range(optimal_n_clusters)]

# Define consistent colors for UHI zones (can be customized)
zone_colors = {
    'Cool_Zone': '#0000FF',    # Blue
    'Mild_Zone': '#FFFF00',    # Yellow
    'Hot_Zone': '#FF0000',     # Red
    'Mild_Zone_1': '#87CEEB',  # Sky Blue (for 4+ clusters, a lighter cool)
    'Mild_Zone_2': '#FFA07A'   # Light Salmon (for 4+ clusters, a lighter hot)
}
# Ensure that all dynamically generated zone_names also have a color mapping, assign a default if not defined
for zone in zone_names:
    if zone not in zone_colors:
        zone_colors[zone] = '#A9A9A9' # DarkGrey as a fallback for undefined zones

# Map the raw KMeans cluster IDs to the ordered descriptive zone names
for i, cluster_id in enumerate(cluster_order):
    if i < len(zone_names):
        cluster_labels[cluster_id] = zone_names[i]
    else:
        # Fallback for situations where `zone_names` might be shorter than `optimal_n_clusters` (shouldn't happen with logic above)
        cluster_labels[cluster_id] = f'Zone_{i+1}'

df_pixels['LST_Zone'] = df_pixels['LST_Cluster'].map(cluster_labels)

logging.info("UHI Zones Distribution:\n" + str(df_pixels['LST_Zone'].value_counts()))
logging.info(f"Cluster Centers (°C): {sorted(cluster_centers)}")

# Plotting UHI Zones Distribution and LST by Zone
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
# Ensure ordering of zones for consistent plotting (e.g., Coolest to Hottest)
ordered_zones = [cluster_labels[idx] for idx in cluster_order if cluster_labels[idx] in df_pixels['LST_Zone'].unique()]
sns.boxplot(data=df_pixels, x='LST_Zone', y='LST_Celsius', order=ordered_zones, palette=zone_colors, linewidth=1.5, fliersize=3)
plt.title('LST Distribution by UHI Zone', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.xlabel('UHI Zone', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6, axis='y') # Grid only on y-axis

plt.subplot(1, 2, 2)
zone_counts = df_pixels['LST_Zone'].value_counts().reindex(ordered_zones, fill_value=0) # Reindex for consistent pie chart order
colors_for_pie = [zone_colors.get(zone, 'grey') for zone in zone_counts.index] # Use defined zone_colors

plt.pie(zone_counts.values, labels=zone_counts.index, autopct='%1.1f%%', 
        colors=colors_for_pie, startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 1})
plt.title('UHI Zone Distribution', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'uhi_zones_analysis.png'), dpi=300, bbox_inches='tight')
logging.info(f"Saved UHI zones analysis plot to {OUTPUTS_DIR}/uhi_zones_analysis.png")
plt.close()

# 7. UHI Intensity Calculation

urban_land_cover_types = ['Built-up']
# Broad definition for rural/natural land cover types
rural_land_cover_types = ['Trees', 'Shrubland', 'Grassland', 'Cropland', 'Water'] 

# Calculate mean LST for urban and rural areas
urban_lst = df_pixels[df_pixels['Land_Cover_Type'].isin(urban_land_cover_types)]['LST_Celsius'].mean()
rural_lst = df_pixels[df_pixels['Land_Cover_Type'].isin(rural_land_cover_types)]['LST_Celsius'].mean()

if not pd.isna(urban_lst) and not pd.isna(rural_lst):
    uhi_intensity = urban_lst - rural_lst
    logging.info(f"\n--- UHI Intensity Calculation ---")
    logging.info(f"Mean LST in Urban Areas ({', '.join(urban_land_cover_types)}): {urban_lst:.2f}°C")
    logging.info(f"Mean LST in Rural/Natural Areas ({', '.join(rural_land_cover_types)}): {rural_lst:.2f}°C")
    logging.info(f"Urban Heat Island Intensity: {uhi_intensity:.2f}°C")
else:
    logging.warning("Could not calculate UHI intensity. Check if urban/rural land cover types are present in data and have valid LST values.")
    uhi_intensity = None # Indicate that calculation failed

logging.info("\nMean LST by Land Cover Type:")
mean_lst_by_lc = df_pixels.groupby('Land_Cover_Type')['LST_Celsius'].mean().sort_values(ascending=False)
logging.info("\n" + str(mean_lst_by_lc))

# 8. Hot Spot Analysis (Getis-Ord Gi*) - Significantly Improved for Robustness

logging.info("\n--- Performing Hot Spot Analysis (Getis-Ord Gi*) ---")
# Ensure latitude and longitude columns exist for spatial analysis
if 'latitude' in df_pixels.columns and 'longitude' in df_pixels.columns:
    coords = df_pixels[['longitude', 'latitude']].values
    if len(coords) > 1: # At least 2 points are required for spatial analysis
        try:
            # Dynamically choose spatial weights method based on data characteristics
            # For small datasets or very coarse resolution, KNN might be more appropriate
            # to ensure every point has neighbors.
            if len(df_pixels) <= 500 or COMMON_RESOLUTION_METERS * 2 >= 1000: # Heuristic threshold for switching
                 wq = weights.KNN(coords, k=8) # Use k=8 (common choice) for K-Nearest Neighbors
                 logging.info(f"Using KNN (k=8) for spatial weights matrix due to small dataset size ({len(df_pixels)} points) or large common resolution ({COMMON_RESOLUTION_METERS}m).")
            else:
                 # Use distance-based weights for larger, denser datasets
                 wq = weights.Distance(coords, threshold=COMMON_RESOLUTION_METERS * 2, p=2, alpha=-1.0, island_weight=0)
                 logging.info(f"Using Distance-based weights with threshold {COMMON_RESOLUTION_METERS * 2}m.")
            
            # Important: Row-standardize weights for Gi* statistic
            wq.transform = 'R' 

            # Check if any neighbors were found. If not, spatial analysis cannot proceed.
            if sum(wq.cardinalities.values()) == 0: # `wq.cardinalities` is a dict of neighbor counts
                logging.warning("No neighbors found for any points with the specified distance/KNN settings. Hot spot analysis skipped.")
                df_pixels['Gi_ZScore'] = np.nan
                df_pixels['Gi_PValue'] = np.nan
                df_pixels['Hot_Spot_Class'] = 'No Data' # Assign a class indicating no analysis
            else:
                # Perform Getis-Ord Gi* local spatial autocorrelation analysis
                lisa = G_Local(df_pixels['LST_Celsius'].values, wq)
                df_pixels['Gi_ZScore'] = lisa.z_sim # Z-score of Gi*
                df_pixels['Gi_PValue'] = lisa.p_sim # P-value of Gi*

                # Classify points into Hot Spots, Cold Spots, or Not Significant
                df_pixels['Hot_Spot_Class'] = 'Not Significant'
                # Hot Spot: High LST surrounded by high LST (Gi_ZScore > 1.96, p < 0.05 for 95% confidence)
                df_pixels.loc[(df_pixels['Gi_ZScore'] > 1.96) & (df_pixels['Gi_PValue'] < 0.05), 'Hot_Spot_Class'] = 'Hot Spot'
                # Cold Spot: Low LST surrounded by low LST (Gi_ZScore < -1.96, p < 0.05 for 95% confidence)
                df_pixels.loc[(df_pixels['Gi_ZScore'] < -1.96) & (df_pixels['Gi_PValue'] < 0.05), 'Hot_Spot_Class'] = 'Cold Spot'

                logging.info(f"Hot Spot Analysis completed. Found {df_pixels[df_pixels['Hot_Spot_Class'] == 'Hot Spot'].shape[0]} Hot Spots and {df_pixels[df_pixels['Hot_Spot_Class'] == 'Cold Spot'].shape[0]} Cold Spots.")
                logging.info("Hot Spot Class Distribution:\n" + str(df_pixels['Hot_Spot_Class'].value_counts()))

                # Plotting Hot Spot Analysis results
                plt.figure(figsize=(12, 10)) 
                sns.scatterplot(
                    data=df_pixels,
                    x='longitude',
                    y='latitude',
                    hue='Hot_Spot_Class',
                    palette={'Hot Spot': 'darkred', 'Cold Spot': 'darkblue', 'Not Significant': 'lightgray', 
                             'No Data': 'black', 'Analysis Error': 'purple', 'Not Enough Data': 'darkgrey', 'No Coords': 'black'},
                    s=40, alpha=0.8, # Increased point size and opacity for better visibility
                    legend='full', edgecolor='w', linewidth=0.5 # Added white edge for clarity, full legend
                )
                plt.title('LST Hot Spot Analysis (Getis-Ord Gi*)', fontsize=16)
                plt.xlabel('Longitude', fontsize=12)
                plt.ylabel('Latitude', fontsize=12)
                plt.gca().set_aspect('equal', adjustable='box') # Maintain aspect ratio
                plt.grid(True, linestyle=':', alpha=0.7) # Add finer grid
                plt.savefig(os.path.join(OUTPUTS_DIR, 'hot_spot_analysis.png'), dpi=300, bbox_inches='tight')
                logging.info(f"Saved Hot Spot Analysis plot to {OUTPUTS_DIR}/hot_spot_analysis.png")
                plt.close()

        except Exception as e:
            logging.error(f"Error during Hot Spot Analysis: {e}")
            df_pixels['Gi_ZScore'] = np.nan # Assign NaNs if analysis fails
            df_pixels['Gi_PValue'] = np.nan
            df_pixels['Hot_Spot_Class'] = 'Analysis Error' # Indicate an error occurred
    else:
        logging.warning("Not enough sampled points for Hot Spot Analysis (requires at least 2 points). Skipping.")
        df_pixels['Gi_ZScore'] = np.nan
        df_pixels['Gi_PValue'] = np.nan
        df_pixels['Hot_Spot_Class'] = 'Not Enough Data'
else:
    logging.warning("Skipping Hot Spot Analysis: Latitude/Longitude columns not found in DataFrame.")
    df_pixels['Gi_ZScore'] = np.nan
    df_pixels['Gi_PValue'] = np.nan
    df_pixels['Hot_Spot_Class'] = 'No Coords' # Indicate missing coordinates

# 9. Creating Enhanced Interactive Map

logging.info("\n--- Creating Enhanced Interactive Map ---")

def create_enhanced_folium_map(roi, lst_img, ndvi_img, ndbi_img, lc_img, topo_img, df_sampled_points, kmeans_model, ordered_zone_names, zone_colors_map):
    """
    Creates an interactive Folium map, visualizing various geospatial layers from GEE,
    sampled data points colored by UHI zones, and hot/cold spots.
    
    Args:
        roi (ee.Geometry): Region of interest boundary.
        lst_img (ee.Image): Land Surface Temperature image.
        ndvi_img (ee.Image): NDVI image.
        ndbi_img (ee.Image): NDBI image.
        lc_img (ee.Image): Land Cover image.
        topo_img (ee.Image): Topographic features image (containing Elevation and Slope bands).
        df_sampled_points (pd.DataFrame): DataFrame of sampled data points, including LST_Zone,
                                           Hot_Spot_Class, latitude, longitude.
        kmeans_model (sklearn.cluster.KMeans): Trained KMeans model used for UHI zones.
        ordered_zone_names (list): List of UHI zone names in temperature order (e.g., ['Cool_Zone', 'Hot_Zone']).
        zone_colors_map (dict): Dictionary mapping UHI zone names to colors.
    """
    
    # Get centroid of ROI for initial map view
    roi_centroid = roi.centroid().coordinates().getInfo()
    map_center = [roi_centroid[1], roi_centroid[0]] # Folium expects [latitude, longitude]
    
    m = folium.Map(location=map_center, zoom_start=9, tiles='OpenStreetMap') # Initialize map with OpenStreetMap tiles
    
    # Define visualization parameters for GEE raster layers
    lst_vis = {
        'min': float(df_sampled_points['LST_Celsius'].quantile(0.02)), # Use 2nd and 98th percentile for min/max
        'max': float(df_sampled_points['LST_Celsius'].quantile(0.98)), # to exclude extreme outliers and improve visualization
        'palette': ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000', '#800000'] # Blue (cold) to Red (hot)
    }
    
    ndvi_vis = {'min': -0.2, 'max': 0.8, 'palette': ['#8B4513', '#FFFFFF', '#00FF00']} # Brown (low vegetation) to Green (high vegetation)
    ndbi_vis = {'min': -0.5, 'max': 0.5, 'palette': ['#00FF00', '#FFFF00', '#FF0000']} # Green (natural) to Red (built-up)
    
    # WorldCover specific palette (matching the numerical IDs)
    lc_vis_palette = [
        '#006400', # 10: Trees
        '#ffbb22', # 20: Shrubland
        '#ffff4c', # 30: Grassland
        '#f096ff', # 40: Cropland
        '#fa0000', # 50: Built-up
        '#b4b4b4', # 60: Bare/Sparse Vegetation
        '#f0f0f0', # 70: Snow and Ice
        '#0064c8', # 80: Permanent Water Bodies
        '#0096ff', # 90: Herbaceous Wetland
        '#6e6e6e', # 95: Mangroves
    ]
    lc_vis = {
        'min': 10, 'max': 95, # Min/Max WorldCover class IDs
        'palette': lc_vis_palette
    }

    # FIX: Explicitly cast NumPy min/max values to float to avoid "Cannot encode object" error
    elevation_vis = {'min': float(df_sampled_points['Elevation'].min()), 'max': float(df_sampled_points['Elevation'].max()), 'palette': ['#006633', '#E6E699', '#B36B00']} # Green (low) to Brown (high)
    slope_vis = {'min': 0, 'max': float(df_sampled_points['Slope'].max() * 0.8), 'palette': ['#FFFFFF', '#B3B3B3', '#000000']} # White (flat) to Black (steep)

    # Prepare UHI Zone image based on LST clusters for the map
    remapped_uhi_zone_image = None
    uhi_zone_palette_for_map = [] # Initialize palette for the UHI zone layer on map
    if kmeans_model is not None and ordered_zone_names and zone_colors_map:
        try:
            lst_band_only = lst_img.select('LST_Celsius') # Select only the LST band from the combined LST image
            cluster_centers_numpy = kmeans_model.cluster_centers_.flatten()
            
            # FIX: Pass native Python list to ee.List directly. GEE will handle ee.Number conversion.
            ee_cluster_centers = ee.List(cluster_centers_numpy.tolist()) 
            
            # Classify each pixel in the LST image based on its closest KMeans cluster center
            classified_image = lst_band_only.spectralDistance(ee_cluster_centers, 'L2').select('reducer_index').int()
            
            # Remap the cluster IDs to be ordered from 0 to N-1 (Coolest to Hottest)
            # `cluster_order` (from global scope) contains the original cluster IDs sorted by temperature
            # FIX: Pass native Python list to ee.List directly. GEE will handle ee.Number conversion.
            from_values = ee.List(cluster_order.tolist()) 
            to_values = ee.List.sequence(0, len(ordered_zone_names) - 1)

            remapped_uhi_zone_image = classified_image.remap(from_values, to_values).rename('UHI_Zone_Class')
            
            # Create a palette for the remapped UHI zone image based on the ordered zone names
            uhi_zone_palette_for_map = [zone_colors_map.get(name, 'grey') for name in ordered_zone_names]
            
            uhi_vis = {
                'min': 0, 
                'max': len(uhi_zone_palette_for_map) - 1, 
                'palette': uhi_zone_palette_for_map
            }
        except Exception as e:
            logging.error(f"Error generating UHI Zone EE Image for map: {e}")
            remapped_uhi_zone_image = None # Ensure it's None on failure

    # Helper function to add GEE layers to the Folium map
    def add_ee_layer(folium_map, ee_image, vis_params, name, opacity=1.0):
        try:
            if ee_image: # Only attempt to add if the GEE image object is not None
                map_id_dict = ee.Image(ee_image).getMapId(vis_params)
                folium.TileLayer(
                    tiles=map_id_dict['tile_fetcher'].url_format,
                    attr='Google Earth Engine',
                    overlay=True, # Overlay on base map
                    control=True, # Allow toggling in layer control
                    name=name,
                    opacity=opacity
                ).add_to(folium_map)
            else:
                logging.warning(f"Skipping GEE layer '{name}' as image data is None.")
        except Exception as e:
            logging.error(f"Failed to add GEE layer '{name}': {e}")
            logging.debug(f"Detailed error for '{name}': {e}", exc_info=True)

    # Add all GEE layers to the map
    if remapped_uhi_zone_image: # UHI Zones layer (if generated successfully)
        add_ee_layer(m, remapped_uhi_zone_image, uhi_vis, 'Urban Heat Island Zones', opacity=0.7)
    add_ee_layer(m, lst_img, lst_vis, f'Land Surface Temperature (Overall: {START_DATE} to {END_DATE})')
    add_ee_layer(m, ndvi_img, ndvi_vis, 'NDVI (Vegetation Index)')
    add_ee_layer(m, ndbi_img, ndbi_vis, 'NDBI (Built-up Index)')
    add_ee_layer(m, lc_img, lc_vis, 'Land Cover (ESA WorldCover)')
    add_ee_layer(m, topo_img.select('Elevation'), elevation_vis, 'Elevation') # Select specific band for visualization
    add_ee_layer(m, topo_img.select('Slope'), slope_vis, 'Slope') # Select specific band for visualization
    
    # Add Study Area Boundary as a GeoJson layer
    roi_geojson = roi.getInfo()
    folium.GeoJson(
        roi_geojson,
        name='Study Area Boundary',
        style_function=lambda x: {
            'color': 'yellow', 
            'fillOpacity': 0, 
            'weight': 3, # Thicker border
            'dashArray': '10, 10' # Dashed border
        }
    ).add_to(m)

    # Add sampled data points as interactive markers
    if 'latitude' in df_sampled_points.columns and 'longitude' in df_sampled_points.columns:
        # Limit the number of points displayed on the map for performance reasons
        if len(df_sampled_points) > 2000:
            display_points_df = df_sampled_points.sample(n=2000, random_state=RANDOM_STATE)
            logging.info(f"Displaying a sample of {len(display_points_df)} points on the map for performance.")
        else:
            display_points_df = df_sampled_points
            
        # Feature Group for UHI Zone colored points
        sampled_points_group = folium.FeatureGroup(name='Sampled Data Points (UHI Zones)')
        for idx, row in display_points_df.iterrows():
            point_color = zone_colors_map.get(row['LST_Zone'], 'gray') # Get color based on UHI zone
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=2, # Small radius for individual pixels
                color='black', # Border color
                fill=True,
                fill_color=point_color,
                fill_opacity=0.7,
                tooltip=f"LST: {row['LST_Celsius']:.1f}°C, Zone: {row['LST_Zone']}, LC: {row['Land_Cover_Type']}"
            ).add_to(sampled_points_group)
        sampled_points_group.add_to(m)
        
        # Feature Group for Hot/Cold Spot points (if analysis was performed)
        if 'Hot_Spot_Class' in df_sampled_points.columns and not df_sampled_points['Hot_Spot_Class'].isnull().all():
            hot_spot_group = folium.FeatureGroup(name='LST Hot/Cold Spots')
            hot_spot_colors = {
                'Hot Spot': 'red', 'Cold Spot': 'blue', 'Not Significant': 'lightgray',
                'No Data': 'black', 'Analysis Error': 'purple', 'Not Enough Data': 'darkgrey', 'No Coords': 'black'
            }
            # Filter to only display actual hot/cold spots, or a small sample if none found
            display_hot_spots_df = df_sampled_points[df_sampled_points['Hot_Spot_Class'].isin(['Hot Spot', 'Cold Spot'])]
            if len(display_hot_spots_df) == 0 and len(df_sampled_points) > 0:
                # If no significant hot/cold spots, sample some points to still allow the layer to be visible
                display_hot_spots_df = df_sampled_points.sample(min(200, len(df_sampled_points)), random_state=RANDOM_STATE)

            for idx, row in display_hot_spots_df.iterrows():
                hs_color = hot_spot_colors.get(row['Hot_Spot_Class'], 'gray')
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3, # Slightly larger radius for hot/cold spots
                    color='black', weight=1, fill=True,
                    fill_color=hs_color, fill_opacity=0.8,
                    tooltip=f"LST: {row['LST_Celsius']:.1f}°C, Hot Spot: {row['Hot_Spot_Class']}, Z-Score: {row['Gi_ZScore']:.2f}"
                ).add_to(hot_spot_group)
            hot_spot_group.add_to(m)

    else:
        logging.warning("Skipping addition of sampled points to map: Latitude/Longitude columns not found in DataFrame.")

    folium.LayerControl(position='topright').add_to(m) # Add layer control to the map
    
    # Custom HTML Legend for the interactive map
    uhi_zone_legend_items = ""
    # Ensure ordered_zone_names and the palette for map are available for legend generation
    if ordered_zone_names and uhi_zone_palette_for_map: 
        for i, zone_name in enumerate(ordered_zone_names):
            if i < len(uhi_zone_palette_for_map): # Ensure index is within bounds of the palette
                color = uhi_zone_palette_for_map[i] 
                uhi_zone_legend_items += f'<p style="margin-top: 2px; margin-bottom: 2px;"><i style="background:{color}; opacity:0.7; width:20px; height:12px; float:left; margin-right:5px; border:1px solid black;"></i> {zone_name}</p>'
            else:
                logging.warning(f"Index {i} out of bounds for UHI zone map palette in legend generation for zone '{zone_name}'.")
    else:
        uhi_zone_legend_items = '<p style="margin-top: 2px; margin-bottom: 2px;">UHI Zones data not available</p>'
        logging.warning("UHI zone legend items could not be fully generated due to missing map palette or zone info.")

    hot_spot_legend_items = """
    <p style="margin-top: 5px; o bottom: 2px;"><b>Hot/Cold Spots:</b></p>
    <p style="margin-top: 2px; margin-bottom: 2px;"><i style="background:red; width:20px; height:12px; float:left; margin-right:5px; border:1px solid black;"></i> Hot Spot</p>
    <p style="margin-top: 2px; margin-bottom: 2px;"><i style="background:blue; width:20px; height:12px; float:left; margin-right:5px; border:1px solid black;"></i> Cold Spot</p>
    <p style="margin-top: 2px; margin-bottom: 2px;"><i style="background:lightgray; width:20px; height:12px; float:left; margin-right:5px; border:1px solid black;"></i> Not Significant</p>
    """
    
    # Get Built-up area color from the lc_vis_palette for the legend
    built_up_color = lc_vis_palette[4] if len(lc_vis_palette) > 4 else 'gray' # Index 4 is Built-up (50)

    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 250px; background-color: white; 
                background-color: rgba(255, 255, 255, 0.8); border:2px solid grey; z-index:9999; font-size:14px;
                padding: 10px; border-radius: 5px;">
    <b>UHI Analysis Legend</b><br>
    <p style="margin-top: 5px; margin-bottom: 2px;"><b>UHI Zones:</b></p>
    {uhi_zone_legend_items}
    <br style="clear:both;">
    {hot_spot_legend_items}
    <br style="clear:both;">
    <p style="margin-top: 5px; margin-bottom: 2px;"><i class="fa fa-thermometer" style="color:red; margin-right:5px;"></i> LST: Land Surface Temperature</p>
    <p style="margin-top: 2px; margin-bottom: 2px;"><i class="fa fa-leaf" style="color:green; margin-right:5px;"></i> NDVI: Vegetation Index</p>
    <p style="margin-top: 2px; margin-bottom: 2px;"><i class="fa fa-building" style="color:blue; margin-right:5px;"></i> NDBI: Built-up Index</p>
    <p style="margin-top: 2px; margin-bottom: 2px;"><i style="background:{built_up_color}; width:20px; height:12px; float:left; margin-right:5px; border:1px solid black;"></i> Built-up Area (Land Cover)</p>
    <p style="margin-top: 2px; margin-bottom: 2px;"><i style="background:white; border:1px solid black; width:20px; height:12px; float:left; margin-right:5px;"></i> Sampled Points (on map)</p>
    </div>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

map_path = None # Initialize map_path to None, will be updated if map is saved

# Check if df_pixels is populated and has necessary columns before proceeding with map creation
if df_pixels is not None and not df_pixels.empty and 'LST_Celsius' in df_pixels.columns:
    logging.info(f"Preparing base map layers using overall period: {START_DATE} to {END_DATE}")

    # Fetch required GEE image layers for the map (using overall analysis period for consistent map layers)
    lst_image_map = None
    if PRIMARY_LST_SOURCE == 'MODIS':
        lst_image_map = get_modis_lst(ROI, START_DATE, END_DATE)
    elif PRIMARY_LST_SOURCE == 'LANDSAT':
        lst_image_map = get_landsat_lst(ROI, START_DATE, END_DATE)
    else:
        logging.error(f"Invalid PRIMARY_LST_SOURCE: {PRIMARY_LST_SOURCE}. Cannot prepare LST map layer.")

    ndvi_image_map, ndbi_image_map = get_landsat_indices(ROI, START_DATE, END_DATE)
    landcover_image_map = get_worldcover_landcover(ROI, WORLDCOVER_YEAR)
    topographic_image_map = get_topographic_features(ROI)

    # Check for crucial layers before attempting map creation
    if lst_image_map is None:
        logging.error("LST image is missing for map generation. Cannot create interactive map.")
        # No `enhanced_map` will be created if LST is missing
    elif not all(img is not None for img in [ndvi_image_map, ndbi_image_map, landcover_image_map, topographic_image_map]):
        logging.warning("One or more non-LST layers for the interactive map could not be fetched. The map might be incomplete.")

    # Proceed with map creation only if essential elements are available
    if 'kmeans' in locals() and 'ordered_zones' in locals() and 'zone_colors' in locals() and lst_image_map is not None:
        try:
            enhanced_map = create_enhanced_folium_map(
                ROI, lst_image_map, ndbi_image_map, ndbi_image_map, landcover_image_map, # Note: ndbi_image_map used twice, assuming a typo for ndvi_image_map
                topographic_image_map,
                df_pixels, kmeans, ordered_zones, zone_colors
            )

            map_path = os.path.join(OUTPUTS_DIR, f'{CITY_NAME.replace(" ", "_").lower()}_enhanced_uhi_map.html')
            enhanced_map.save(map_path)
            logging.info(f"Enhanced interactive map saved to: {map_path}")
        except Exception as e:
            logging.error(f"Failed to create or save the interactive map: {e}")
            map_path = None # Reset path if saving failed
    else:
        logging.error("Could not generate the enhanced interactive map due to missing UHI clustering information or essential LST layer.")
        map_path = None # Ensure map_path remains None if map generation failed
else:
    logging.error("DataFrame `df_pixels` is empty or None, or 'LST_Celsius' column is missing. Cannot create map.")
    map_path = None # Ensure map_path is None if map cannot be created due to data issues


# 10. Enhanced Machine Learning Pipeline

logging.info("\n--- Enhanced Machine Learning Pipeline ---")

# Define columns to drop from the DataFrame before ML training
cols_to_drop_ml = ['LST_Cluster', 'LST_Zone'] # UHI cluster information is not a feature for LST prediction

# Conditionally add other non-feature columns if they exist in df_pixels
for col in ['latitude', 'longitude', 'Gi_ZScore', 'Gi_PValue', 'Hot_Spot_Class', 'Year', 'Season']:
    if col in df_pixels.columns:
        cols_to_drop_ml.append(col)

# Filter the list to ensure only columns actually present in the DataFrame are dropped
cols_to_drop_ml = [col for col in cols_to_drop_ml if col in df_pixels.columns]

# Create dummy variables for the 'Land_Cover_Type' categorical feature
df_ml = pd.get_dummies(df_pixels.drop(columns=cols_to_drop_ml), 
                       columns=['Land_Cover_Type'], prefix='LC') # Prefix 'LC' for Land Cover

# Define features (X) and target (y) for the machine learning model
X = df_ml.drop(columns=['LST_Celsius']) # All columns except LST_Celsius are features
y = df_ml['LST_Celsius'] # LST_Celsius is the target variable

logging.info(f"Features shape for ML: {X.shape}")
logging.info(f"Target shape for ML: {y.shape}")
logging.info(f"Features used in ML: {list(X.columns)}")

if X.empty or y.empty:
    logging.error("DataFrame for machine learning is empty after processing or target column is missing. Cannot proceed with model training. Exiting.")
    exit()

# Split data into training and testing sets (80% train, 20% test)
# Removed `stratify` as the target `LST_Celsius` is a continuous variable.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Scale numerical features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit and transform on training data
X_test_scaled = scaler.transform(X_test) # Transform test data using scaler fitted on training data

logging.info("Training Enhanced Random Forest model...")

# Hyperparameter Tuning or Direct Training of Random Forest Regressor
if PERFORM_HYPERPARAMETER_TUNING:
    logging.info(f"Performing hyperparameter tuning with RandomizedSearchCV ({TUNING_ITERATIONS} iterations)...")
    # Define a dictionary of parameter distributions for RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300, 500], # Number of trees
        'max_features': ['sqrt', 'log2', 1.0], # Number of features to consider when looking for the best split
        'max_depth': [10, 15, 20, 25, None], # Maximum depth of the tree (None means unlimited)
        'min_samples_split': [2, 5, 10], # Minimum samples required to split an internal node
        'min_samples_leaf': [1, 2, 4], # Minimum samples required to be at a leaf node
        'bootstrap': [True, False] # Whether bootstrap samples are used when building trees
    }
    
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1) # Initialize RF Regressor
    
    # Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf, 
        param_distributions=param_dist, 
        n_iter=TUNING_ITERATIONS, 
        cv=5, # 5-fold cross-validation
        verbose=1, # Verbosity level
        random_state=RANDOM_STATE, 
        n_jobs=-1, # Use all available CPU cores for parallel processing
        scoring='neg_root_mean_squared_error' # Optimize for RMSE (negative score to maximize)
    )
    
    random_search.fit(X_train_scaled, y_train) # Run the search
    rf_model = random_search.best_estimator_ # Get the best model found
    logging.info(f"Hyperparameter tuning completed. Best parameters: {random_search.best_params_}")
    logging.info(f"Best RMSE from tuning: {-random_search.best_score_:.2f}°C") # Convert negative RMSE back to positive
else:
    logging.info("Skipping hyperparameter tuning. Training with default/configured parameters.")
    rf_model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1 # Use all available CPU cores
    )
    rf_model.fit(X_train_scaled, y_train) # Train the model

logging.info("Model training completed.")

y_pred = rf_model.predict(X_test_scaled) # Make predictions on the scaled test set

# 11. Enhanced Model Evaluation

logging.info("\n--- Enhanced Model Evaluation ---")

# Calculate common regression metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # RMSE is sqrt of MSE
r2 = r2_score(y_test, y_pred) # R-squared

logging.info(f"Model Performance Metrics (on Test Set):")
logging.info(f"Mean Absolute Error (MAE): {mae:.2f}°C")
logging.info(f"Root Mean Square Error (RMSE): {rmse:.2f}°C") 
logging.info(f"R² Score: {r2:.3f}")

logging.info("\n--- Cross-Validation Performance ---")
# Perform cross-validation to get a more robust estimate of model performance
cv_scores_rmse = -cross_val_score(rf_model, X_train_scaled, y_train, 
                                  cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
logging.info(f"Cross-validated RMSE: {np.mean(cv_scores_rmse):.2f}°C (Std: {np.std(cv_scores_rmse):.2f}°C)")

cv_scores_r2 = cross_val_score(rf_model, X_train_scaled, y_train, 
                               cv=5, scoring='r2', n_jobs=-1)
logging.info(f"Cross-validated R²: {np.mean(cv_scores_r2):.3f} (Std: {np.std(cv_scores_r2):.3f})")

# Plotting Model Evaluation Metrics
fig, axes = plt.subplots(2, 2, figsize=(16, 14)) # Larger figure for better clarity

# Plot 1: Actual vs Predicted LST
axes[0,0].scatter(y_test, y_pred, alpha=0.6, s=30, color='mediumblue', edgecolor='gray', linewidth=0.5)
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Add a 1:1 line for reference
axes[0,0].set_xlabel('Actual LST (°C)', fontsize=12)
axes[0,0].set_ylabel('Predicted LST (°C)', fontsize=12)
axes[0,0].set_title(f'Actual vs Predicted LST (R² = {r2:.3f})', fontsize=14)
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Residuals Plot (Predicted vs Residuals)
residuals = y_test - y_pred # Calculate residuals
axes[0,1].scatter(y_pred, residuals, alpha=0.6, s=30, color='darkgreen', edgecolor='gray', linewidth=0.5)
axes[0,1].axhline(y=0, color='r', linestyle='--', lw=2) # Add a horizontal line at 0 for ideal residuals
axes[0,1].set_xlabel('Predicted LST (°C)', fontsize=12)
axes[0,1].set_ylabel('Residuals (°C)', fontsize=12)
axes[0,1].set_title('Residuals Plot', fontsize=14)
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Top 10 Feature Importances from Random Forest model
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = feature_importance.nlargest(10)
top_features.plot(kind='barh', ax=axes[1,0], color=sns.color_palette("viridis", len(top_features)), edgecolor='black', linewidth=0.5)
axes[1,0].set_title('Top 10 Feature Importances', fontsize=14)
axes[1,0].set_xlabel('Importance', fontsize=12)
axes[1,0].invert_yaxis() # Display most important feature at the top
axes[1,0].grid(True, linestyle='--', alpha=0.6, axis='x')

# Plot 4: Distribution of Prediction Errors (Residuals)
sns.histplot(residuals, bins=30, alpha=0.7, edgecolor='black', ax=axes[1,1], color='goldenrod', kde=True)
axes[1,1].set_xlabel('Prediction Error (°C)', fontsize=12)
axes[1,1].set_ylabel('Frequency', fontsize=12)
axes[1,1].set_title('Distribution of Prediction Errors', fontsize=14)
axes[1,1].axvline(x=0, color='r', linestyle='--', lw=2) # Vertical line at 0 error
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'enhanced_model_evaluation.png'), dpi=300, bbox_inches='tight')
logging.info(f"Saved model evaluation plot to {OUTPUTS_DIR}/enhanced_model_evaluation.png")
plt.close()

logging.info(f"\nTop 10 Most Important Features:")
for feature, importance in top_features.items():
    logging.info(f"{feature}: {importance:.4f}")

# 12. SHAP Value Analysis for Model Interpretability

logging.info("\n--- SHAP Value Analysis for Model Interpretability ---")
try:
    # Use a subsample of the test set for SHAP to manage computation time for large datasets
    shap_sample_size = min(2000, X_test_scaled.shape[0])
    if shap_sample_size == 0:
        logging.warning("No data available for SHAP analysis after splitting the dataset.")
        shap_values = None
    else:
        # Create a Pandas DataFrame from the scaled data to retain column names for SHAP plots
        shap_X = pd.DataFrame(X_test_scaled[:shap_sample_size], columns=X.columns)

        explainer = shap.TreeExplainer(rf_model) # Initialize SHAP explainer for tree-based models
        shap_values = explainer.shap_values(shap_X) # Compute SHAP values

        # SHAP Summary Plot (Bar) - shows average impact of each feature on model output
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, shap_X, plot_type="bar", show=False, color_bar=False) # color_bar=False is better for bar plots
        plt.title('SHAP Feature Importance (Mean Absolute SHAP Value)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, 'shap_summary_bar.png'), dpi=300, bbox_inches='tight')
        logging.info(f"Saved SHAP summary bar plot to {OUTPUTS_DIR}/shap_summary_bar.png")
        plt.close()

        # SHAP Summary Plot (Dot) - shows impact of each feature for each prediction
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, shap_X, show=False)
        plt.title('SHAP Summary Plot (Feature Impact on Model Output)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, 'shap_summary_dot.png'), dpi=300, bbox_inches='tight')
        logging.info(f"Saved SHAP summary dot plot to {OUTPUTS_DIR}/shap_summary_dot.png")
        plt.close()

except ImportError:
    logging.warning("SHAP library not installed. Skipping SHAP analysis. Install with 'pip install shap'.")
    shap_values = None
except Exception as e:
    logging.error(f"An error occurred during SHAP analysis: {e}")
    logging.info("Skipping SHAP analysis.")
    shap_values = None


# 13. Final Results Summary and Output Information

logging.info("\n" + "="*80)
logging.info("ENHANCED URBAN HEAT ISLAND ANALYSIS - RESULTS SUMMARY")
logging.info("="*80)

logging.info(f"\n📍 Study Area: {CITY_NAME}")
logging.info(f"📅 Analysis Period (for general map layers): {START_DATE} to {END_DATE}") 
logging.info(f"📊 Total Pixels Analyzed (across all combined periods): {len(df_pixels):,}")

logging.info(f"\n🌡️  TEMPERATURE ANALYSIS (Combined Data):")
logging.info(f"   • Mean LST: {df_pixels['LST_Celsius'].mean():.1f}°C")
logging.info(f"   • Min LST: {df_pixels['LST_Celsius'].min():.1f}°C")
logging.info(f"   • Max LST: {df_pixels['LST_Celsius'].max():.1f}°C")
logging.info(f"   • Temperature Range: {df_pixels['LST_Celsius'].max() - df_pixels['LST_Celsius'].min():.1f}°C")

if uhi_intensity is not None:
    logging.info(f"   • Urban Heat Island Intensity: {uhi_intensity:.2f}°C (Mean Urban LST vs. Mean Rural/Natural LST)")
else:
    logging.warning("   • UHI Intensity could not be calculated (check for presence of urban/rural land cover data).")

logging.info(f"\n🏙️  UHI ZONES DISTRIBUTION ({optimal_n_clusters} clusters):")
# Ensure ordered_zones is defined from clustering before trying to use it here
if 'ordered_zones' in locals() and 'zone_colors' in locals():
    for zone, count in df_pixels['LST_Zone'].value_counts(normalize=True).reindex(ordered_zones, fill_value=0).items():
        logging.info(f"   • {zone}: {count*100:.1f}%")
else:
    logging.warning("   • UHI Zone distribution cannot be reported as clustering information is missing or incomplete.")

logging.info(f"\n🗺️  LST HOT SPOT ANALYSIS (Getis-Ord Gi*):")
if 'Hot_Spot_Class' in df_pixels.columns and not df_pixels['Hot_Spot_Class'].isnull().all():
    hot_spot_counts = df_pixels['Hot_Spot_Class'].value_counts()
    for hs_class, count in hot_spot_counts.items():
        logging.info(f"   • {hs_class}: {count:,} pixels")
else:
    logging.warning("   • Hot Spot Analysis results not available or contained no significant spots after filtering.")


logging.info(f"\n🤖 MACHINE LEARNING MODEL PERFORMANCE (Random Forest Regressor):")
logging.info(f"   • Model Type: Random Forest Regressor")
if PERFORM_HYPERPARAMETER_TUNING:
    logging.info(f"   • Hyperparameter Tuning: Performed with RandomizedSearchCV")
    if 'random_search' in locals() and hasattr(random_search, 'best_params_') and random_search.best_params_:
        logging.info(f"   • Best Parameters: {random_search.best_params_}")
else:
    logging.info(f"   • Model Parameters: n_estimators={RF_N_ESTIMATORS}, max_depth={RF_MAX_DEPTH}, min_samples_split={RF_MIN_SAMPLES_SPLIT}, min_samples_leaf={RF_MIN_SAMPLES_LEAF}")

logging.info(f"   • R² Score (on test set): {r2:.3f} (Variance explained)")
logging.info(f"   • Mean Absolute Error (on test set): {mae:.2f}°C")
logging.info(f"   • Root Mean Square Error (on test set): {rmse:.2f}°C")
if 'cv_scores_rmse' in locals() and len(cv_scores_rmse) > 0:
    logging.info(f"   • Cross-validated R² (Mean): {np.mean(cv_scores_r2):.3f} (Std: {np.std(cv_scores_r2):.3f})")
    logging.info(f"   • Cross-validated RMSE (Mean): {np.mean(cv_scores_rmse):.2f}°C (Std: {np.std(cv_scores_rmse):.2f}°C)")
else:
    logging.warning("   • Cross-validation scores could not be computed or are not available.")


logging.info(f"\n📁 OUTPUT FILES GENERATED in '{OUTPUTS_DIR}' directory:")
if map_path and os.path.exists(map_path): # Check if the map was successfully generated and its path stored
    logging.info(f"   • Interactive Map: {map_path}")
else:
    logging.warning("   • Interactive Map: Not generated or path not found due to previous errors.")

logging.info(f"   • Feature Distributions Plot: {os.path.join(OUTPUTS_DIR, 'feature_distributions.png')}")
logging.info(f"   • Correlation Matrix Plot: {os.path.join(OUTPUTS_DIR, 'correlation_matrix.png')}")
logging.info(f"   • Scatter Relationships Plot: {os.path.join(OUTPUTS_DIR, 'scatter_relationships.png')}")
if os.path.exists(os.path.join(OUTPUTS_DIR, 'silhouette_score.png')):
    logging.info(f"   • Silhouette Score Plot: {os.path.join(OUTPUTS_DIR, 'silhouette_score.png')}")
logging.info(f"   • UHI Zones Analysis Plot: {os.path.join(OUTPUTS_DIR, 'uhi_zones_analysis.png')}")
if os.path.exists(os.path.join(OUTPUTS_DIR, 'hot_spot_analysis.png')):
    logging.info(f"   • Hot Spot Analysis Plot: {os.path.join(OUTPUTS_DIR, 'hot_spot_analysis.png')}")
logging.info(f"   • Model Evaluation Plot: {os.path.join(OUTPUTS_DIR, 'enhanced_model_evaluation.png')}")
if shap_values is not None:
    logging.info(f"   • SHAP Summary Bar Plot: {os.path.join(OUTPUTS_DIR, 'shap_summary_bar.png')}")
    logging.info(f"   • SHAP Summary Dot Plot: {os.path.join(OUTPUTS_DIR, 'shap_summary_dot.png')}")
else:
    logging.warning("   • SHAP plots not generated (SHAP library missing or analysis failed).")

logging.info(f"\n🔍 KEY INSIGHTS:")
if 'feature_importance' in locals() and not feature_importance.empty: 
    most_important_feature = feature_importance.idxmax()
    logging.info(f"   • Most influential factor in LST prediction: {most_important_feature}")
else:
    logging.warning("   • Feature importance analysis not available (likely due to empty dataset after cleaning or ML skipped).")

# Report correlations with LST if features exist
if 'NDVI' in df_pixels.columns and 'LST_Celsius' in df_pixels.columns:
    try:
        ndvi_corr = df_pixels[['NDVI', 'LST_Celsius']].corr().iloc[0,1]
        logging.info(f"   • NDVI correlation with LST: {ndvi_corr:.3f} (typically negative - higher vegetation, lower temperature)")
    except Exception as e:
        logging.warning(f"   • Could not calculate NDVI correlation: {e}")
if 'NDBI' in df_pixels.columns and 'LST_Celsius' in df_pixels.columns:
    try:
        ndbi_corr = df_pixels[['NDBI', 'LST_Celsius']].corr().iloc[0,1]
        logging.info(f"   • NDBI correlation with LST: {ndbi_corr:.3f} (typically positive - higher built-up, higher temperature)")
    except Exception as e:
        logging.warning(f"   • Could not calculate NDBI correlation: {e}")
if 'Elevation' in df_pixels.columns and 'LST_Celsius' in df_pixels.columns:
    try:
        elevation_corr = df_pixels[['Elevation', 'LST_Celsius']].corr().iloc[0,1]
        logging.info(f"   • Elevation correlation with LST: {elevation_corr:.3f} (often negative - higher elevation, lower temperature)")
    except Exception as e:
        logging.warning(f"   • Could not calculate Elevation correlation: {e}")

if not df_pixels.empty and 'Land_Cover_Type' in df_pixels.columns:
    try:
        mean_lst_by_lc = df_pixels.groupby('Land_Cover_Type')['LST_Celsius'].mean()
        if not mean_lst_by_lc.empty:
            hottest_lc = mean_lst_by_lc.idxmax()
            coolest_lc = mean_lst_by_lc.idxmin()
            logging.info(f"   • Hottest land cover type (avg LST): {hottest_lc} ({mean_lst_by_lc.max():.1f}°C)")
            logging.info(f"   • Coolest land cover type (avg LST): {coolest_lc} ({mean_lst_by_lc.min():.1f}°C)")
        else:
            logging.warning("   • No land cover types found after filtering to determine hottest/coolest.")
    except Exception as e:
        logging.warning(f"   • Error determining hottest/coolest land cover: {e}")
else:
    logging.warning("   • No data available to determine hottest/coolest land cover type.")

logging.info(f"\n✅ ANALYSIS COMPLETE! All results saved in '{OUTPUTS_DIR}' directory.")
logging.info("="*80)
