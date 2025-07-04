# -----------------------------------------------------------------------------
# Project: Urban Heat Island (UHI) Detection & Prediction in New Delhi
# Author: Arham Ansari
# Date: 30-06-2025
# Description: This comprehensive script leverages Google Earth Engine (GEE)
#              to acquire satellite imagery (LST, NDVI, NDBI, Land Cover),
#              performs data cleaning, exploratory data analysis (EDA),
#              detects Urban Heat Island zones using K-Means clustering,
#              and predicts Land Surface Temperature (LST) using Random Forest.
#              Results are visualized on interactive maps and plots.
# -----------------------------------------------------------------------------

import ee
import geemap
import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
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

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

try:
    ee.Initialize()
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print(f"Error initializing Google Earth Engine: {e}")
    print("ACTION REQUIRED: Please ensure you have authenticated by running 'earthengine authenticate' in your terminal.")
    print("Exiting script.")
    exit()

try:
    from config_params import (
        CITY_NAME, ROI_BOUNDS, ROI_GEOMETRY,
        START_DATE, END_DATE,
        PRIMARY_LST_SOURCE, COMMON_RESOLUTION_METERS,
        NUM_PIXELS_FOR_ML, RANDOM_STATE,
        UHI_CLUSTERS, UHI_CLUSTER_RANGE,
        RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT, RF_MIN_SAMPLES_LEAF,
        PERFORM_HYPERPARAMETER_TUNING, TUNING_ITERATIONS,
        OUTPUTS_DIR, WORLDCOVER_YEAR,
        LOG_FILE, LOG_LEVEL,
        ANALYSIS_YEARS, SEASONS_TO_ANALYZE
    )
except ImportError:
    print("Error: config_params.py not found. Please ensure it's in the same directory.")
    print("Exiting script.")
    exit()

os.makedirs(OUTPUTS_DIR, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logging.info(f"Output directory created at: {OUTPUTS_DIR}")
logging.info(f"Log file created at: {LOG_FILE}\n")

ROI = ROI_GEOMETRY

logging.info(f"\nTargeting: {CITY_NAME} from {START_DATE} to {END_DATE}")
logging.info(f"Analysis Resolution: {COMMON_RESOLUTION_METERS}m")

def get_modis_lst(roi, start_date, end_date):
    logging.info(f"Fetching MODIS LST data for {start_date} to {end_date}...")
    try:
        lst_collection = ee.ImageCollection('MODIS/061/MOD11A2') \
                           .filterDate(start_date, end_date) \
                           .filterBounds(roi) \
                           .select('LST_Day_1km')

        mean_lst_kelvin = lst_collection.mean()
        
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
    logging.info(f"Fetching Landsat 8 Surface Temperature data for {start_date} to {end_date}...")
    try:
        landsat_collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                               .filterDate(start_date, end_date) \
                               .filterBounds(roi) \
                               .select('ST_B10', 'QA_PIXEL')

        def mask_clouds_landsat8_c2_thermal(image):
            qa_pixel = image.select('QA_PIXEL')
            cloud_mask = qa_pixel.bitwiseAnd(1 << 1).eq(0) \
                          .And(qa_pixel.bitwiseAnd(1 << 3).eq(0)) \
                          .And(qa_pixel.bitwiseAnd(1 << 4).eq(0))
            
            st_kelvin = image.select('ST_B10').multiply(0.00341802).add(149.0)
            st_celsius = st_kelvin.subtract(273.15)
            return st_celsius.updateMask(cloud_mask).rename('LST_Landsat_Celsius')

        landsat_lst_processed = landsat_collection.map(mask_clouds_landsat8_c2_thermal)
        
        mean_lst_landsat = landsat_lst_processed.mean().clip(roi)
        
        logging.info("Landsat LST data processed successfully.")
        return mean_lst_landsat
    except Exception as e:
        logging.error(f"Error processing Landsat LST data: {e}")
        return None

def get_landsat_indices(roi, start_date, end_date):
    logging.info(f"Fetching Landsat 8 data for {start_date} to {end_date}...")
    try:
        landsat_collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                               .filterDate(start_date, end_date) \
                               .filterBounds(roi)

        def mask_clouds_landsat8_c2_optical(image):
            qa_pixel = image.select('QA_PIXEL')
            cloud_mask = qa_pixel.bitwiseAnd(1 << 1).eq(0) \
                          .And(qa_pixel.bitwiseAnd(1 << 3).eq(0)) \
                          .And(qa_pixel.bitwiseAnd(1 << 4).eq(0))

            optical_bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
            image = image.select(optical_bands).multiply(0.0000275).add(-0.2)
            return image.updateMask(cloud_mask)

        def add_indices(image):
            ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
            ndbi = image.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
            return image.addBands([ndvi, ndbi])

        landsat_processed = landsat_collection.map(mask_clouds_landsat8_c2_optical)
        landsat_with_indices = landsat_processed.map(add_indices)

        mean_ndvi = landsat_with_indices.select('NDVI').mean().clip(roi)
        mean_ndbi = landsat_with_indices.select('NDBI').mean().clip(roi)

        logging.info("Landsat NDVI and NDBI data processed successfully.")
        return mean_ndvi, mean_ndbi
    except Exception as e:
        logging.error(f"Error processing Landsat data: {e}")
        return None, None

def get_worldcover_landcover(roi, year):
    logging.info(f"Fetching ESA WorldCover data for {year}...")
    try:
        if year == 2020:
            landcover_image = ee.Image('ESA/WorldCover/v100/2020').select('Map')
        elif year == 2021:
            landcover_image = ee.Image('ESA/WorldCover/v200/2021').select('Map')
        else:
            logging.warning(f"WorldCover data not available for {year}. Using 2021 as default.")
            landcover_image = ee.Image('ESA/WorldCover/v200/2021').select('Map')

        clipped_landcover = landcover_image.clip(roi).rename('Land_Cover')
        logging.info("ESA WorldCover data processed successfully.")
        return clipped_landcover
    except Exception as e:
        logging.error(f"Error processing WorldCover data: {e}")
        return None

def get_topographic_features(roi):
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
    logging.info(f"Sampling {num_pixels} pixels at {int(scale)}m resolution...")
    try:
        sample_points_fc = image.sample(
            region=roi,
            scale=scale,
            numPixels=num_pixels,
            seed=RANDOM_STATE,
            geometries=True # Crucial for getting geometry
        )
        
        # Add geometry as properties to ensure it's included in the DataFrame
        sample_points_fc = sample_points_fc.map(lambda f: f.set({'longitude': f.geometry().coordinates().get(0),
                                                                    'latitude': f.geometry().coordinates().get(1)}))
        
        # Convert to DataFrame using geemap.ee_to_df which should now include geometry properties
        df = geemap.ee_to_df(sample_points_fc)
        
        if df.empty:
            raise ValueError("No data was sampled even after GEE-side filtering. Check ROI, dates, and masking strictness.")
        
        # Explicitly check for and extract latitude and longitude
        if 'latitude' in df.columns and 'longitude' in df.columns:
            logging.info(f"Successfully extracted {len(df)} pixels into a DataFrame with latitude/longitude.")
            return df
        else:
            # This case should ideally not be reached if the map function above works.
            # If it still fails, it means something went wrong with the geometry mapping.
            logging.error("Latitude or Longitude columns not found in DataFrame after explicit mapping.")
            return None
            
    except Exception as e:
        logging.error(f"Error during sampling: {e}")
        return None

# --- Main Execution Logic ---
all_data_dfs = []

for year in ANALYSIS_YEARS:
    for season, (start_month, end_month) in SEASONS_TO_ANALYZE.items():
        logging.info(f"\n--- Processing {year} - {season} ---")

        start_date_season = f'{year}-{start_month:02d}-01'
        
        if start_month > end_month:
            end_year = year + 1
            end_date_season = f'{end_year}-{end_month:02d}-{pd.Timestamp(f"{end_year}-{end_month:02d}-01").days_in_month}'
            logging.warning(f"Season '{season}' spans across years. Date range: {start_date_season} to {end_date_season}.")
        else:
            end_date_season = f'{year}-{end_month:02d}-{pd.Timestamp(f"{year}-{end_month:02d}-01").days_in_month}'

        logging.info(f"Fetching data for {start_date_season} to {end_date_season}...")

        if PRIMARY_LST_SOURCE == 'MODIS':
            lst_image = get_modis_lst(ROI, start_date_season, end_date_season)
            lst_band_name_in_df = 'LST_Celsius'
        elif PRIMARY_LST_SOURCE == 'LANDSAT':
            lst_image = get_landsat_lst(ROI, start_date_season, end_date_season)
            lst_band_name_in_df = 'LST_Landsat_Celsius'
        else:
            logging.error(f"Invalid PRIMARY_LST_SOURCE: {PRIMARY_LST_SOURCE}. Must be 'MODIS' or 'LANDSAT'. Exiting.")
            exit()

        ndvi_image, ndbi_image = get_landsat_indices(ROI, start_date_season, end_date_season)

        landcover_image = get_worldcover_landcover(ROI, WORLDCOVER_YEAR)

        topographic_image = get_topographic_features(ROI)

        if any(img is None for img in [lst_image, ndvi_image, ndbi_image, landcover_image, topographic_image]):
            logging.error(f"ERROR: Failed to acquire one or more datasets for {year}-{season}. Skipping this period.")
            continue

        combined_image = lst_image.addBands([ndvi_image, ndbi_image, landcover_image, topographic_image])
        logging.info("All data layers combined successfully for the current period.")

        df_pixels = sample_to_dataframe(combined_image, ROI, NUM_PIXELS_FOR_ML, COMMON_RESOLUTION_METERS)

        if df_pixels is None or df_pixels.empty:
            logging.error(f"ERROR: Failed to create DataFrame for {year}-{season}. Skipping this period.")
            continue

        if lst_band_name_in_df != 'LST_Celsius':
            if lst_band_name_in_df in df_pixels.columns:
                df_pixels.rename(columns={lst_band_name_in_df: 'LST_Celsius'}, inplace=True)
                logging.info(f"Renamed LST column from '{lst_band_name_in_df}' to 'LST_Celsius'.")
            else:
                logging.error(f"LST band '{lst_band_name_in_df}' not found in DataFrame columns. Cannot proceed for {year}-{season}.")
                continue

        logging.info("\n--- Data Cleaning ---")
        initial_rows = len(df_pixels)

        df_pixels.dropna(inplace=True)
        logging.info(f"Removed {initial_rows - len(df_pixels)} rows with NaN values.")

        valid_lc_classes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
        df_pixels = df_pixels[df_pixels['Land_Cover'].isin(valid_lc_classes)]
        logging.info(f"Filtered to {len(df_pixels)} rows with valid land cover classes.")

        land_cover_mapping = {
            10: 'Trees',
            20: 'Shrubland', 
            30: 'Grassland',
            40: 'Cropland',
            50: 'Built-up',
            60: 'Bare_Vegetation',
            70: 'Snow_Ice',
            80: 'Water',
            90: 'Wetland',
            95: 'Mangroves'
        }

        df_pixels['Land_Cover_Type'] = df_pixels['Land_Cover'].map(land_cover_mapping)
        df_pixels.drop(columns=['Land_Cover'], inplace=True)

        df_pixels['Year'] = year
        df_pixels['Season'] = season
        
        all_data_dfs.append(df_pixels)

if not all_data_dfs:
    logging.error("No data was successfully processed for any year/season. Exiting.")
    exit()

df_combined_all_periods = pd.concat(all_data_dfs, ignore_index=True)

df_pixels = df_combined_all_periods
logging.info(f"\n--- Combined Data Summary (All Periods) ---")
logging.info(f"Total DataFrame shape after combining all periods: {df_pixels.shape}")

logging.info("\n--- Data Summary ---")
logging.info(f"\nDataFrame shape: {df_pixels.shape}")
logging.info("\nLand Cover Distribution:\n" + str(df_pixels['Land_Cover_Type'].value_counts()))
logging.info("\nDescriptive Statistics:\n" + str(df_pixels.describe()))
logging.info("\nFirst 5 rows of data:\n" + str(df_pixels.head()))

logging.info("\n--- Enhanced EDA and Visualization ---")

def create_enhanced_plots(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    sns.histplot(data=df, x='LST_Celsius', kde=True, ax=axes[0,0])
    axes[0,0].set_title('Land Surface Temperature Distribution')
    axes[0,0].set_xlabel('Temperature (°C)')
    
    sns.histplot(data=df, x='NDVI', kde=True, ax=axes[0,1])
    axes[0,1].set_title('NDVI Distribution')
    axes[0,1].set_xlabel('NDVI')
    
    sns.histplot(data=df, x='NDBI', kde=True, ax=axes[1,0])
    axes[1,0].set_title('NDBI Distribution')
    axes[1,0].set_xlabel('NDBI')
    
    sns.boxplot(data=df, x='Land_Cover_Type', y='LST_Celsius', ax=axes[1,1])
    axes[1,1].set_title('LST by Land Cover Type')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
    logging.info(f"Saved feature distributions plot to {OUTPUTS_DIR}/feature_distributions.png")
    plt.close()
    
    plt.figure(figsize=(10, 8))
    numerical_cols = ['LST_Celsius', 'NDVI', 'NDBI', 'Elevation', 'Slope', 'Aspect']
    numerical_cols_present = [col for col in numerical_cols if col in df.columns]
    correlation_matrix = df[numerical_cols_present].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Environmental Variables')
    plt.savefig(os.path.join(OUTPUTS_DIR, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    logging.info(f"Saved correlation matrix plot to {OUTPUTS_DIR}/correlation_matrix.png")
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.scatterplot(data=df, x='NDVI', y='LST_Celsius', hue='Land_Cover_Type', 
                   alpha=0.6, ax=axes[0], legend='full')
    axes[0].set_title('LST vs NDVI by Land Cover')
    
    sns.scatterplot(data=df, x='NDBI', y='LST_Celsius', hue='Land_Cover_Type',
                   alpha=0.6, ax=axes[1], legend='full')
    axes[1].set_title('LST vs NDBI by Land Cover')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'scatter_relationships.png'), dpi=300, bbox_inches='tight')
    logging.info(f"Saved scatter relationships plot to {OUTPUTS_DIR}/scatter_relationships.png")
    plt.close()

create_enhanced_plots(df_pixels)

logging.info("\n--- Enhanced UHI Detection ---")

optimal_n_clusters = 3

if UHI_CLUSTERS == 'auto':
    logging.info("Determining optimal number of clusters using Silhouette Score...")
    silhouette_scores = []
    
    min_samples_for_silhouette = max(UHI_CLUSTER_RANGE) + 1
    if len(df_pixels) < min_samples_for_silhouette:
        logging.warning(f"Not enough samples ({len(df_pixels)}) for requested cluster range {UHI_CLUSTER_RANGE}. Minimum required: {min_samples_for_silhouette}. Skipping auto-detection.")
    else:
        valid_k_values_for_plot = []
        for n_c in UHI_CLUSTER_RANGE:
            if n_c <= 1 or n_c > len(df_pixels): continue
            try:
                kmeans_temp = KMeans(n_clusters=n_c, random_state=RANDOM_STATE, n_init=10)
                cluster_labels_temp = kmeans_temp.fit_predict(df_pixels[['LST_Celsius']])
                if len(np.unique(cluster_labels_temp)) < 2:
                    logging.warning(f"  Only 1 cluster formed for {n_c} clusters, skipping silhouette score.")
                    continue
                score = silhouette_score(df_pixels[['LST_Celsius']], cluster_labels_temp)
                silhouette_scores.append(score)
                valid_k_values_for_plot.append(n_c)
                logging.info(f"  Clusters: {n_c}, Silhouette Score: {score:.3f}")
            except Exception as e:
                logging.warning(f"  Could not calculate silhouette score for {n_c} clusters: {e}")

        if silhouette_scores and valid_k_values_for_plot:
            optimal_n_clusters = valid_k_values_for_plot[np.argmax(silhouette_scores)]
            logging.info(f"Optimal number of clusters (Silhouette Score): {optimal_n_clusters}")

            plt.figure(figsize=(8, 5))
            plt.plot(valid_k_values_for_plot, silhouette_scores, marker='o')
            plt.xlabel("Number of Clusters (k)")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette Score for Optimal K")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(OUTPUTS_DIR, 'silhouette_score.png'), dpi=300, bbox_inches='tight')
            logging.info(f"Saved Silhouette Score plot to {OUTPUTS_DIR}/silhouette_score.png")
            plt.close()
        else:
            logging.warning(f"Failed to auto-detect optimal clusters. Using default {optimal_n_clusters}. Data or K-means behavior might be an issue.")
else:
    optimal_n_clusters = int(UHI_CLUSTERS) 
    logging.info(f"Using fixed number of clusters: {optimal_n_clusters}")

if optimal_n_clusters < 2:
    logging.error("Not enough samples or clusters to perform meaningful UHI detection. Exiting script.")
    exit()

if optimal_n_clusters > len(df_pixels):
    logging.warning(f"Optimal clusters ({optimal_n_clusters}) > number of samples ({len(df_pixels)}). Adjusting clusters to number of samples.")
    optimal_n_clusters = len(df_pixels)
    if optimal_n_clusters < 2:
        logging.error("Not enough samples for clustering after adjustment. Exiting script.")
        exit()

kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=RANDOM_STATE, n_init=10)
df_pixels['LST_Cluster'] = kmeans.fit_predict(df_pixels[['LST_Celsius']])

cluster_centers = kmeans.cluster_centers_.flatten()
cluster_order = np.argsort(cluster_centers)

cluster_labels = {}
if optimal_n_clusters == 2:
    zone_names = ['Cool_Zone', 'Hot_Zone']
elif optimal_n_clusters == 3:
    zone_names = ['Cool_Zone', 'Mild_Zone', 'Hot_Zone']
elif optimal_n_clusters == 4:
    zone_names = ['Cool_Zone', 'Mild_Zone_1', 'Mild_Zone_2', 'Hot_Zone']
else:
    zone_names = [f'Zone_{i+1}' for i in range(optimal_n_clusters)]

for i, cluster_id in enumerate(cluster_order):
    if i < len(zone_names):
        cluster_labels[cluster_id] = zone_names[i]
    else:
        cluster_labels[cluster_id] = f'Zone_{i+1}'

df_pixels['LST_Zone'] = df_pixels['LST_Cluster'].map(cluster_labels)

logging.info("UHI Zones Distribution:\n" + str(df_pixels['LST_Zone'].value_counts()))
logging.info(f"Cluster Centers (°C): {sorted(cluster_centers)}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
ordered_zones = [cluster_labels[idx] for idx in cluster_order if cluster_labels[idx] in df_pixels['LST_Zone'].unique()]
sns.boxplot(data=df_pixels, x='LST_Zone', y='LST_Celsius', order=ordered_zones)
plt.title('LST Distribution by UHI Zone')
plt.ylabel('Temperature (°C)')

plt.subplot(1, 2, 2)
zone_counts = df_pixels['LST_Zone'].value_counts().reindex(ordered_zones, fill_value=0)
zone_colors = {
    'Cool_Zone': '#0000FF',
    'Mild_Zone': '#FFFF00',
    'Hot_Zone': '#FF0000',
    'Mild_Zone_1': '#ADD8E6',
    'Mild_Zone_2': '#FFD700'
}
colors_for_pie = [zone_colors.get(zone, 'grey') for zone in zone_counts.index]

plt.pie(zone_counts.values, labels=zone_counts.index, autopct='%1.1f%%', 
        colors=colors_for_pie, startangle=90)
plt.title('UHI Zone Distribution')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'uhi_zones_analysis.png'), dpi=300, bbox_inches='tight')
logging.info(f"Saved UHI zones analysis plot to {OUTPUTS_DIR}/uhi_zones_analysis.png")
plt.close()

urban_land_cover_types = ['Built-up']
rural_land_cover_types = ['Trees', 'Shrubland', 'Grassland', 'Cropland', 'Water']

urban_lst = df_pixels[df_pixels['Land_Cover_Type'].isin(urban_land_cover_types)]['LST_Celsius'].mean()
rural_lst = df_pixels[df_pixels['Land_Cover_Type'].isin(rural_land_cover_types)]['LST_Celsius'].mean()

if not pd.isna(urban_lst) and not pd.isna(rural_lst):
    uhi_intensity = urban_lst - rural_lst
    logging.info(f"\n--- UHI Intensity Calculation ---")
    logging.info(f"Mean LST in Urban Areas ({', '.join(urban_land_cover_types)}): {urban_lst:.2f}°C")
    logging.info(f"Mean LST in Rural/Natural Areas ({', '.join(rural_land_cover_types)}): {rural_lst:.2f}°C")
    logging.info(f"Urban Heat Island Intensity: {uhi_intensity:.2f}°C")
else:
    logging.warning("Could not calculate UHI intensity. Check if urban/rural land cover types are present in data.")
    uhi_intensity = None

logging.info("\nMean LST by Land Cover Type:")
mean_lst_by_lc = df_pixels.groupby('Land_Cover_Type')['LST_Celsius'].mean().sort_values(ascending=False)
logging.info("\n" + str(mean_lst_by_lc))

logging.info("\n--- Performing Hot Spot Analysis (Getis-Ord Gi*) ---")
if 'latitude' in df_pixels.columns and 'longitude' in df_pixels.columns:
    coords = df_pixels[['longitude', 'latitude']].values
    if len(coords) > 1:
        try:
            wq = weights.Distance(coords, threshold=COMMON_RESOLUTION_METERS * 2, p=2, alpha=-1.0, island_weight=0)
            if wq.n_components > 1:
                logging.warning(f"Spatial weights matrix has {wq.n_components} disconnected components. Hot spot analysis results might be fragmented.")
            elif wq.n_components == 0:
                 logging.warning("Spatial weights matrix is empty. No neighbors found for any points.")

            if wq.cardinalities.sum() == 0:
                logging.warning("No neighbors found for any points with the specified distance threshold. Hot spot analysis skipped.")
                df_pixels['Gi_ZScore'] = np.nan
                df_pixels['Gi_PValue'] = np.nan
                df_pixels['Hot_Spot_Class'] = 'No Data'
            else:
                lisa = G_Local(df_pixels['LST_Celsius'].values, wq, transform='R')
                df_pixels['Gi_ZScore'] = lisa.z_sim
                df_pixels['Gi_PValue'] = lisa.p_sim

                df_pixels['Hot_Spot_Class'] = 'Not Significant'
                df_pixels.loc[(df_pixels['Gi_ZScore'] > 1.96) & (df_pixels['Gi_PValue'] < 0.05), 'Hot_Spot_Class'] = 'Hot Spot'
                df_pixels.loc[(df_pixels['Gi_ZScore'] < -1.96) & (df_pixels['Gi_PValue'] < 0.05), 'Hot_Spot_Class'] = 'Cold Spot'

                logging.info(f"Hot Spot Analysis completed. Found {df_pixels[df_pixels['Hot_Spot_Class'] == 'Hot Spot'].shape[0]} Hot Spots and {df_pixels[df_pixels['Hot_Spot_Class'] == 'Cold Spot'].shape[0]} Cold Spots.")
                logging.info("Hot Spot Class Distribution:\n" + str(df_pixels['Hot_Spot_Class'].value_counts()))

                plt.figure(figsize=(10, 8))
                sns.scatterplot(
                    data=df_pixels,
                    x='longitude',
                    y='latitude',
                    hue='Hot_Spot_Class',
                    palette={'Hot Spot': 'red', 'Cold Spot': 'blue', 'Not Significant': 'lightgray', 'No Data': 'black'},
                    s=50, alpha=0.7,
                    legend='full'
                )
                plt.title('LST Hot Spot Analysis (Getis-Ord Gi*)')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.gca().set_aspect('equal', adjustable='box')
                plt.savefig(os.path.join(OUTPUTS_DIR, 'hot_spot_analysis.png'), dpi=300, bbox_inches='tight')
                logging.info(f"Saved Hot Spot Analysis plot to {OUTPUTS_DIR}/hot_spot_analysis.png")
                plt.close()

        except Exception as e:
            logging.error(f"Error during Hot Spot Analysis: {e}")
            df_pixels['Gi_ZScore'] = np.nan
            df_pixels['Gi_PValue'] = np.nan
            df_pixels['Hot_Spot_Class'] = 'Analysis Error'
    else:
        logging.warning("Not enough sampled points for Hot Spot Analysis (requires at least 2 points).")
        df_pixels['Gi_ZScore'] = np.nan
        df_pixels['Gi_PValue'] = np.nan
        df_pixels['Hot_Spot_Class'] = 'Not Enough Data'
else:
    logging.warning("Skipping Hot Spot Analysis: Latitude/Longitude columns not found in DataFrame.")
    df_pixels['Gi_ZScore'] = np.nan
    df_pixels['Gi_PValue'] = np.nan
    df_pixels['Hot_Spot_Class'] = 'No Coords'

logging.info("\n--- Creating Enhanced Interactive Map ---")

def create_enhanced_folium_map(roi, lst_img, ndvi_img, ndbi_img, lc_img, topo_img, df_sampled_points, kmeans_model, ordered_zone_names, zone_colors_map):
    
    roi_centroid = roi.centroid().coordinates().getInfo()
    map_center = [roi_centroid[1], roi_centroid[0]]
    
    m = folium.Map(location=map_center, zoom_start=9, tiles='OpenStreetMap')
    
    lst_vis = {
        'min': df_sampled_points['LST_Celsius'].quantile(0.02), 
        'max': df_sampled_points['LST_Celsius'].quantile(0.98),
        'palette': ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000', '#800000']
    }
    
    ndvi_vis = {'min': -0.2, 'max': 0.8, 'palette': ['#8B4513', '#FFFFFF', '#00FF00']}
    ndbi_vis = {'min': -0.5, 'max': 0.5, 'palette': ['#00FF00', '#FFFF00', '#FF0000']}
    
    lc_vis_palette = [
        '#006400', # Trees
        '#ffbb22', # Shrubland
        '#ffff4c', # Grassland
        '#f096ff', # Cropland
        '#fa0000', # Built-up
        '#b4b4b4', # Bare_Vegetation
        '#f0f0f0', # Snow_Ice
        '#0064c8', # Water
        '#0096ff', # Wetland
        '#6e6e6e', # Mangroves
    ]
    lc_vis = {
        'min': 10, 'max': 95,
        'palette': lc_vis_palette
    }

    elevation_vis = {'min': 0, 'max': 1000, 'palette': ['#006633', '#E6E699', '#B36B00']}
    slope_vis = {'min': 0, 'max': 30, 'palette': ['#FFFFFF', '#B3B3B3', '#000000']}

    lst_band_only = lst_img.select('LST_Celsius')
    
    remapped_uhi_zone_image = None # Initialize to None
    if 'kmeans' in locals() and 'ordered_zones' in locals():
        try:
            cluster_centers_numpy = kmeans.cluster_centers_.flatten()
            ee_cluster_centers = ee.List(cluster_centers_numpy.tolist())
            
            classified_image_with_distance = lst_band_only.spectralDistance(ee_cluster_centers, 'L2').select('reducer_index')
            
            uhi_zone_ee_image = classified_image_with_distance.rename('UHI_Zone_Class')

            from_values = ee.List(cluster_order.tolist())
            to_values = ee.List.sequence(0, optimal_n_clusters - 1)

            remapped_uhi_zone_image = uhi_zone_ee_image.remap(from_values, to_values)
            
            uhi_zone_palette = [zone_colors_map.get(name, 'grey') for name in ordered_zones]
            
            uhi_vis = {
                'min': 0, 
                'max': len(uhi_zone_palette) - 1, 
                'palette': uhi_zone_palette
            }
        except Exception as e:
            logging.error(f"Error generating UHI Zone EE Image for map: {e}")

    def add_ee_layer(folium_map, ee_image, vis_params, name, opacity=1.0):
        try:
            map_id_dict = ee.Image(ee_image).getMapId(vis_params)
            folium.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                overlay=True,
                control=True,
                name=name,
                opacity=opacity
            ).add_to(folium_map)
        except Exception as e:
            logging.error(f"Failed to add GEE layer '{name}': {e}")
            logging.debug(f"Error details for '{name}': {e}", exc_info=True)

    if remapped_uhi_zone_image:
        add_ee_layer(m, remapped_uhi_zone_image, uhi_vis, 'Urban Heat Island Zones', opacity=0.7)
    add_ee_layer(m, lst_image, lst_vis, f'Land Surface Temperature ({START_DATE} to {END_DATE})')
    add_ee_layer(m, ndvi_image, ndvi_vis, 'NDVI (Vegetation Index)')
    add_ee_layer(m, ndbi_image, ndbi_vis, 'NDBI (Built-up Index)')
    add_ee_layer(m, lc_img, lc_vis, 'Land Cover (ESA WorldCover)')
    add_ee_layer(m, topo_img.select('Elevation'), elevation_vis, 'Elevation')
    add_ee_layer(m, topo_img.select('Slope'), slope_vis, 'Slope')
    
    roi_geojson = roi.getInfo()
    folium.GeoJson(
        roi_geojson,
        name='Study Area Boundary',
        style_function=lambda x: {
            'color': 'yellow', 
            'fillOpacity': 0, 
            'weight': 3,
            'dashArray': '10, 10'
        }
    ).add_to(m)

    if 'latitude' in df_sampled_points.columns and 'longitude' in df_sampled_points.columns:
        if len(df_sampled_points) > 2000:
            display_points_df = df_sampled_points.sample(n=2000, random_state=RANDOM_STATE)
        else:
            display_points_df = df_sampled_points
            
        sampled_points_group = folium.FeatureGroup(name='Sampled Data Points (UHI Zones)')
        for idx, row in display_points_df.iterrows():
            point_color = zone_colors_map.get(row['LST_Zone'], 'gray')
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=2,
                color='black',
                fill=True,
                fill_color=point_color,
                fill_opacity=0.7,
                tooltip=f"LST: {row['LST_Celsius']:.1f}°C, Zone: {row['LST_Zone']}, LC: {row['Land_Cover_Type']}"
            ).add_to(sampled_points_group)
        sampled_points_group.add_to(m)
        
        if 'Hot_Spot_Class' in df_sampled_points.columns and not df_sampled_points['Hot_Spot_Class'].isnull().all():
            hot_spot_group = folium.FeatureGroup(name='LST Hot/Cold Spots')
            hot_spot_colors = {
                'Hot Spot': 'red',
                'Cold Spot': 'blue',
                'Not Significant': 'lightgray',
                'No Data': 'black',
                'Analysis Error': 'purple',
                'Not Enough Data': 'black',
                'No Coords': 'black'
            }
            display_hot_spots_df = df_sampled_points[df_sampled_points['Hot_Spot_Class'].isin(['Hot Spot', 'Cold Spot'])]
            if len(display_hot_spots_df) == 0 and len(df_sampled_points) > 0:
                display_hot_spots_df = df_sampled_points.sample(min(200, len(df_sampled_points)), random_state=RANDOM_STATE)

            for idx, row in display_hot_spots_df.iterrows():
                hs_color = hot_spot_colors.get(row['Hot_Spot_Class'], 'gray')
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,
                    color='black',
                    weight=1,
                    fill=True,
                    fill_color=hs_color,
                    fill_opacity=0.8,
                    tooltip=f"LST: {row['LST_Celsius']:.1f}°C, Hot Spot: {row['Hot_Spot_Class']}, Z-Score: {row['Gi_ZScore']:.2f}"
                ).add_to(hot_spot_group)
            hot_spot_group.add_to(m)

    else:
        logging.warning("Skipping addition of sampled points to map: Latitude/Longitude columns not found.")

    folium.LayerControl(position='topright').add_to(m)
    
    uhi_zone_legend_items = ""
    # Ensure ordered_zones is defined before using it
    if 'ordered_zones' in locals() and 'uhi_zone_palette' in locals():
        for i, zone_name in enumerate(ordered_zones):
            # Ensure index is within bounds for uhi_zone_palette
            if i < len(uhi_zone_palette):
                color = uhi_zone_palette[i] 
                uhi_zone_legend_items += f'<p style="margin-top: 2px; margin-bottom: 2px;"><i style="background:{color}; opacity:0.7; width:20px; height:12px; float:left; margin-right:5px; border:1px solid black;"></i> {zone_name}</p>'
            else:
                logging.warning(f"Index {i} out of bounds for uhi_zone_palette.")
    else:
        logging.warning("UHI zone legend items could not be generated due to missing ordered_zones or uhi_zone_palette.")


    hot_spot_legend_items = """
    <p style="margin-top: 5px; margin-bottom: 2px;"><b>Hot/Cold Spots:</b></p>
    <p style="margin-top: 2px; margin-bottom: 2px;"><i style="background:red; width:20px; height:12px; float:left; margin-right:5px; border:1px solid black;"></i> Hot Spot</p>
    <p style="margin-top: 2px; margin-bottom: 2px;"><i style="background:blue; width:20px; height:12px; float:left; margin-right:5px; border:1px solid black;"></i> Cold Spot</p>
    <p style="margin-top: 2px; margin-bottom: 2px;"><i style="background:lightgray; width:20px; height:12px; float:left; margin-right:5px; border:1px solid black;"></i> Not Significant</p>
    """

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
    <p style="margin-top: 2px; margin-bottom: 2px;"><i style="background:{lc_vis_palette[4]}; width:20px; height:12px; float:left; margin-right:5px; border:1px solid black;"></i> Built-up Area (Land Cover)</p>
    <p style="margin-top: 2px; margin-bottom: 2px;"><i style="background:white; border:1px solid black; width:20px; height:12px; float:left; margin-right:5px;"></i> Sampled Points</p>
    </div>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# Check if df_pixels is populated before proceeding with map creation
if df_pixels is not None and not df_pixels.empty:
    # Prepare map layers using the overall START_DATE and END_DATE
    logging.info(f"Preparing map layers using overall period: {START_DATE} to {END_DATE}")

    lst_image_map = None
    if PRIMARY_LST_SOURCE == 'MODIS':
        lst_image_map = get_modis_lst(ROI, START_DATE, END_DATE)
    elif PRIMARY_LST_SOURCE == 'LANDSAT':
        lst_image_map = get_landsat_lst(ROI, START_DATE, END_DATE)
    else:
        logging.error(f"Invalid PRIMARY_LST_SOURCE: {PRIMARY_LST_SOURCE}. Cannot prepare map layers.")

    ndvi_image_map, ndbi_image_map = get_landsat_indices(ROI, START_DATE, END_DATE)
    landcover_image_map = get_worldcover_landcover(ROI, WORLDCOVER_YEAR)
    topographic_image_map = get_topographic_features(ROI)

    if any(img is None for img in [lst_image_map, ndvi_image_map, ndbi_image_map, landcover_image_map, topographic_image_map]):
        logging.warning("One or more layers for the map could not be fetched. The map might be incomplete.")

    if 'kmeans' in locals() and 'ordered_zones' in locals() and 'zone_colors' in locals():
        enhanced_map = create_enhanced_folium_map(
            ROI, lst_image_map, ndvi_image_map, ndbi_image_map, landcover_image_map, 
            topographic_image_map,
            df_pixels, kmeans, ordered_zones, zone_colors
        )

        map_path = os.path.join(OUTPUTS_DIR, f'{CITY_NAME.replace(" ", "_").lower()}_enhanced_uhi_map.html')
        enhanced_map.save(map_path)
        logging.info(f"Enhanced interactive map saved to: {map_path}")
    else:
        logging.error("Could not generate the enhanced interactive map due to missing UHI clustering information or data layers. Exiting map generation.")
else:
    logging.error("DataFrame `df_pixels` is empty or None. Cannot create map. Exiting.")

logging.info("\n--- Enhanced Machine Learning Pipeline ---")

cols_to_drop_ml = ['LST_Cluster', 'LST_Zone']
if 'latitude' in df_pixels.columns:
    cols_to_drop_ml.append('latitude')
if 'longitude' in df_pixels.columns:
    cols_to_drop_ml.append('longitude')
if 'Gi_ZScore' in df_pixels.columns:
    cols_to_drop_ml.append('Gi_ZScore')
if 'Gi_PValue' in df_pixels.columns:
    cols_to_drop_ml.append('Gi_PValue')
if 'Hot_Spot_Class' in df_pixels.columns:
    cols_to_drop_ml.append('Hot_Spot_Class')
if 'Year' in df_pixels.columns:
    cols_to_drop_ml.append('Year')
if 'Season' in df_pixels.columns:
    cols_to_drop_ml.append('Season')

cols_to_drop_ml = [col for col in cols_to_drop_ml if col in df_pixels.columns]

df_ml = pd.get_dummies(df_pixels.drop(columns=cols_to_drop_ml), 
                       columns=['Land_Cover_Type'], prefix='LC')

X = df_ml.drop(columns=['LST_Celsius'])
y = df_ml['LST_Celsius']

logging.info(f"Features shape: {X.shape}")
logging.info(f"Target shape: {y.shape}")
logging.info(f"Features: {list(X.columns)}")

if X.empty or y.empty:
    logging.error("DataFrame for machine learning is empty after processing. Cannot proceed with model training. Exiting.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=None
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logging.info("Training Enhanced Random Forest model...")

if PERFORM_HYPERPARAMETER_TUNING:
    logging.info(f"Performing hyperparameter tuning with RandomizedSearchCV ({TUNING_ITERATIONS} iterations)...")
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_features': ['sqrt', 'log2', 1.0],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        estimator=rf, 
        param_distributions=param_dist, 
        n_iter=TUNING_ITERATIONS, 
        cv=5,
        verbose=1, 
        random_state=RANDOM_STATE, 
        n_jobs=-1,
        scoring='neg_root_mean_squared_error'
    )
    
    random_search.fit(X_train_scaled, y_train)
    rf_model = random_search.best_estimator_
    logging.info(f"Hyperparameter tuning completed. Best parameters: {random_search.best_params_}")
    logging.info(f"Best RMSE from tuning: {-random_search.best_score_:.2f}°C")
else:
    logging.info("Skipping hyperparameter tuning. Training with default/configured parameters.")
    rf_model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)

logging.info("Model training completed.")

y_pred = rf_model.predict(X_test_scaled)

logging.info("\n--- Enhanced Model Evaluation ---")

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

logging.info(f"Model Performance Metrics:")
logging.info(f"Mean Absolute Error (MAE): {mae:.2f}°C")
logging.info(f"Root Mean Square Error (RMSE): {rmse:.2f}°C") 
logging.info(f"R² Score: {r2:.3f}")

logging.info("\n--- Cross-Validation Performance ---")
cv_scores_rmse = -cross_val_score(rf_model, X_train_scaled, y_train, 
                                  cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
logging.info(f"Cross-validated RMSE: {np.mean(cv_scores_rmse):.2f}°C (Std: {np.std(cv_scores_rmse):.2f}°C)")

cv_scores_r2 = cross_val_score(rf_model, X_train_scaled, y_train, 
                               cv=5, scoring='r2', n_jobs=-1)
logging.info(f"Cross-validated R²: {np.mean(cv_scores_r2):.3f} (Std: {np.std(cv_scores_r2):.3f})")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0,0].scatter(y_test, y_pred, alpha=0.6, s=30)
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0,0].set_xlabel('Actual LST (°C)')
axes[0,0].set_ylabel('Predicted LST (°C)')
axes[0,0].set_title(f'Actual vs Predicted LST (R² = {r2:.3f})')
axes[0,0].grid(True, alpha=0.3)

residuals = y_test - y_pred
axes[0,1].scatter(y_pred, residuals, alpha=0.6, s=30)
axes[0,1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0,1].set_xlabel('Predicted LST (°C)')
axes[0,1].set_ylabel('Residuals (°C)')
axes[0,1].set_title('Residuals Plot')
axes[0,1].grid(True, alpha=0.3)

feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = feature_importance.nlargest(10)
top_features.plot(kind='barh', ax=axes[1,0])
axes[1,0].set_title('Top 10 Feature Importances')
axes[1,0].set_xlabel('Importance')
axes[1,0].invert_yaxis()

axes[1,1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
axes[1,1].set_xlabel('Prediction Error (°C)')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Distribution of Prediction Errors')
axes[1,1].axvline(x=0, color='r', linestyle='--', lw=2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'enhanced_model_evaluation.png'), dpi=300, bbox_inches='tight')
logging.info(f"Saved model evaluation plot to {OUTPUTS_DIR}/enhanced_model_evaluation.png")
plt.close()

logging.info(f"\nTop 10 Most Important Features:")
for feature, importance in top_features.items():
    logging.info(f"{feature}: {importance:.4f}")

logging.info("\n--- SHAP Value Analysis for Model Interpretability ---")
try:
    shap_sample_size = min(2000, X_test_scaled.shape[0])
    if shap_sample_size == 0:
        logging.warning("No data available for SHAP analysis after splitting.")
        shap_values = None
    else:
        shap_X = pd.DataFrame(X_test_scaled[:shap_sample_size], columns=X.columns)

        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(shap_X)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, shap_X, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance (Mean Absolute SHAP Value)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, 'shap_summary_bar.png'), dpi=300, bbox_inches='tight')
        logging.info(f"Saved SHAP summary bar plot to {OUTPUTS_DIR}/shap_summary_bar.png")
        plt.close()

        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, shap_X, show=False)
        plt.title('SHAP Summary Plot (Feature Impact on Model Output)')
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


logging.info("\n" + "="*80)
logging.info("ENHANCED URBAN HEAT ISLAND ANALYSIS - RESULTS SUMMARY")
logging.info("="*80)

logging.info(f"\n📍 Study Area: {CITY_NAME}")
logging.info(f"📅 Analysis Period (for map layers): {START_DATE} to {END_DATE}") 
logging.info(f"📊 Total Pixels Analyzed (across all periods): {len(df_pixels):,}")

logging.info(f"\n🌡️  TEMPERATURE ANALYSIS (Combined Data):")
logging.info(f"   • Mean LST: {df_pixels['LST_Celsius'].mean():.1f}°C")
logging.info(f"   • Min LST: {df_pixels['LST_Celsius'].min():.1f}°C")
logging.info(f"   • Max LST: {df_pixels['LST_Celsius'].max():.1f}°C")
logging.info(f"   • Temperature Range: {df_pixels['LST_Celsius'].max() - df_pixels['LST_Celsius'].min():.1f}°C")

if uhi_intensity is not None:
    logging.info(f"   • Urban Heat Island Intensity: {uhi_intensity:.2f}°C (Urban vs. Rural/Natural LST)")
else:
    logging.warning("   • UHI Intensity could not be calculated (missing urban/rural land cover data).")

logging.info(f"\n🏙️  UHI ZONES DISTRIBUTION ({optimal_n_clusters} clusters):")
if 'ordered_zones' in locals() and 'zone_colors' in locals():
    for zone, count in df_pixels['LST_Zone'].value_counts(normalize=True).reindex(ordered_zones, fill_value=0).items():
        logging.info(f"   • {zone}: {count*100:.1f}%")
else:
    logging.warning("   • UHI Zone distribution cannot be reported as zone ordering or colors are not available.")

logging.info(f"\n🗺️  LST HOT SPOT ANALYSIS:")
if 'Hot_Spot_Class' in df_pixels.columns and not df_pixels['Hot_Spot_Class'].isnull().all():
    hot_spot_counts = df_pixels['Hot_Spot_Class'].value_counts()
    for hs_class, count in hot_spot_counts.items():
        logging.info(f"   • {hs_class}: {count:,} pixels")
else:
    logging.warning("   • Hot Spot Analysis results not available.")


logging.info(f"\n🤖 MACHINE LEARNING MODEL PERFORMANCE:")
logging.info(f"   • Model Type: Random Forest Regressor")
if PERFORM_HYPERPARAMETER_TUNING:
    logging.info(f"   • Hyperparameter Tuning: Performed with RandomizedSearchCV")
    if 'random_search' in locals() and hasattr(random_search, 'best_params_') and random_search.best_params_:
        logging.info(f"   • Best Parameters: {random_search.best_params_}")
else:
    logging.info(f"   • Model Parameters: n_estimators={RF_N_ESTIMATORS}, max_depth={RF_MAX_DEPTH}, min_samples_split={RF_MIN_SAMPLES_SPLIT}, min_samples_leaf={RF_MIN_SAMPLES_LEAF}")

logging.info(f"   • R² Score: {r2:.3f} (Variance explained)")
logging.info(f"   • Mean Absolute Error: {mae:.2f}°C")
logging.info(f"   • Root Mean Square Error: {rmse:.2f}°C")
if 'cv_scores_rmse' in locals() and len(cv_scores_rmse) > 0:
    logging.info(f"   • Cross-validated R² (Mean): {np.mean(cv_scores_r2):.3f} (Std: {np.std(cv_scores_r2):.3f})")
    logging.info(f"   • Cross-validated RMSE (Mean): {np.mean(cv_scores_rmse):.2f}°C (Std: {np.std(cv_scores_rmse):.2f}°C)")
else:
    logging.warning("   • Cross-validation scores could not be computed or are not available.")


logging.info(f"\n📁 OUTPUT FILES GENERATED:")
# Check if map_path was defined before logging
if 'map_path' in locals() and map_path:
    logging.info(f"   • Interactive Map: {map_path}")
else:
    logging.warning("   • Interactive Map: Not generated.")

logging.info(f"   • Feature Distributions: {os.path.join(OUTPUTS_DIR, 'feature_distributions.png')}")
logging.info(f"   • Correlation Matrix: {os.path.join(OUTPUTS_DIR, 'correlation_matrix.png')}")
if 'silhouette_score.png' in os.listdir(OUTPUTS_DIR):
    logging.info(f"   • Silhouette Score Plot: {os.path.join(OUTPUTS_DIR, 'silhouette_score.png')}")
logging.info(f"   • UHI Zones Analysis: {os.path.join(OUTPUTS_DIR, 'uhi_zones_analysis.png')}")
if 'hot_spot_analysis.png' in os.listdir(OUTPUTS_DIR):
    logging.info(f"   • Hot Spot Analysis Plot: {os.path.join(OUTPUTS_DIR, 'hot_spot_analysis.png')}")
logging.info(f"   • Model Evaluation: {os.path.join(OUTPUTS_DIR, 'enhanced_model_evaluation.png')}")
if shap_values is not None:
    logging.info(f"   • SHAP Summary Bar Plot: {os.path.join(OUTPUTS_DIR, 'shap_summary_bar.png')}")
    logging.info(f"   • SHAP Summary Dot Plot: {os.path.join(OUTPUTS_DIR, 'shap_summary_dot.png')}")

logging.info(f"\n🔍 KEY INSIGHTS:")
if not top_features.empty:
    most_important_feature = feature_importance.idxmax()
    logging.info(f"   • Most influential factor in LST prediction: {most_important_feature}")
else:
    logging.warning("   • No features for importance analysis (likely due to empty dataset after cleaning).")

if 'NDVI' in df_pixels.columns and 'LST_Celsius' in df_pixels.columns:
    try:
        ndvi_corr = df_pixels[['NDVI', 'LST_Celsius']].corr().iloc[0,1]
        logging.info(f"   • NDVI correlation with LST: {ndvi_corr:.3f}")
    except Exception as e:
        logging.warning(f"   • Could not calculate NDVI correlation: {e}")
if 'NDBI' in df_pixels.columns and 'LST_Celsius' in df_pixels.columns:
    try:
        ndbi_corr = df_pixels[['NDBI', 'LST_Celsius']].corr().iloc[0,1]
        logging.info(f"   • NDBI correlation with LST: {ndbi_corr:.3f}")
    except Exception as e:
        logging.warning(f"   • Could not calculate NDBI correlation: {e}")
if 'Elevation' in df_pixels.columns and 'LST_Celsius' in df_pixels.columns:
    try:
        elevation_corr = df_pixels[['Elevation', 'LST_Celsius']].corr().iloc[0,1]
        logging.info(f"   • Elevation correlation with LST: {elevation_corr:.3f}")
    except Exception as e:
        logging.warning(f"   • Could not calculate Elevation correlation: {e}")

if not df_pixels.empty:
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

logging.info(f"\n✅ ANALYSIS COMPLETE!")
logging.info(f"All results saved in '{OUTPUTS_DIR}' directory.")
logging.info("="*80)
