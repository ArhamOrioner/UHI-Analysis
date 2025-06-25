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

# -----------------------------------------------------------------------------
# SECTION 0: Import Libraries
# -----------------------------------------------------------------------------
import ee # Google Earth Engine Python API
import geemap # Simplifies GEE operations and interactive mapping
import folium # For robust interactive mapping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # For creating directories

# Machine Learning specific imports
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor # Better than Linear Regression for this task
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set aesthetic style for Seaborn plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Create outputs directory
OUTPUTS_DIR = 'outputs'
os.makedirs(OUTPUTS_DIR, exist_ok=True)
print(f"Output directory created at: {OUTPUTS_DIR}\n")

# -----------------------------------------------------------------------------
# SECTION 1: Configuration & GEE Initialization
# -----------------------------------------------------------------------------

# Initialize Google Earth Engine FIRST (this was the main issue in your new script)
try:
    ee.Initialize()
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print(f"Error initializing Google Earth Engine: {e}")
    print("ACTION REQUIRED: Please ensure you have authenticated by running 'earthengine authenticate' in your terminal.")
    print("Exiting script.")
    exit()

# Define Study Area (NOW it's safe to create geometry objects)
CITY_NAME = "New Delhi"
# Enhanced bounding box for New Delhi with better coverage
ROI_BOUNDS = [76.5, 28.1, 78.5, 29.5]  # [min_lon, min_lat, max_lon, max_lat]
ROI = ee.Geometry.Rectangle(ROI_BOUNDS)

# Define Time Period for Analysis
START_DATE = '2023-06-01'
END_DATE = '2023-08-31'

# Define common spatial resolution for analysis
COMMON_RESOLUTION_METERS = 1000  # 1km resolution for better processing speed

# Number of pixels to sample for ML
NUM_PIXELS_FOR_ML = 5000

# Random state for reproducibility
RANDOM_STATE = 42

print(f"\nTargeting: {CITY_NAME} from {START_DATE} to {END_DATE}")
print(f"Analysis Resolution: {COMMON_RESOLUTION_METERS}m")

# -----------------------------------------------------------------------------
# SECTION 2: Enhanced Data Acquisition Functions
# -----------------------------------------------------------------------------

def get_modis_lst(roi, start_date, end_date):
    """
    Enhanced MODIS Land Surface Temperature acquisition with better error handling.
    """
    print(f"Fetching MODIS LST data for {start_date} to {end_date}...")
    try:
        lst_collection = ee.ImageCollection('MODIS/061/MOD11A2') \
                           .filterDate(start_date, end_date) \
                           .filterBounds(roi) \
                           .select('LST_Day_1km')

        # Calculate mean and convert to Celsius
        mean_lst_kelvin = lst_collection.mean()
        mean_lst_celsius = mean_lst_kelvin.multiply(0.02).subtract(273.15) \
                                         .rename('LST_Celsius')
        
        clipped_lst = mean_lst_celsius.clip(roi)
        print("MODIS LST data processed successfully.")
        return clipped_lst
    except Exception as e:
        print(f"Error processing MODIS LST data: {e}")
        return None

def get_landsat_indices(roi, start_date, end_date):
    """
    Enhanced Landsat 8 processing for NDVI and NDBI with improved cloud masking.
    """
    print(f"Fetching Landsat 8 data for {start_date} to {end_date}...")
    try:
        # Use Collection 2 for better cloud masking
        landsat_collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                               .filterDate(start_date, end_date) \
                               .filterBounds(roi)

        def mask_clouds_landsat8_c2(image):
            """Enhanced cloud masking for Landsat 8 Collection 2"""
            qa_pixel = image.select('QA_PIXEL')
            cloud_mask = qa_pixel.bitwiseAnd(1 << 1).eq(0) \
                          .And(qa_pixel.bitwiseAnd(1 << 3).eq(0)) \
                          .And(qa_pixel.bitwiseAnd(1 << 4).eq(0)) \
                          .And(qa_pixel.bitwiseAnd(1 << 5).eq(0))

            # Apply scaling factors
            optical_bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
            image = image.select(optical_bands).multiply(0.0000275).add(-0.2)
            return image.updateMask(cloud_mask)

        def add_indices(image):
            """Calculate NDVI and NDBI"""
            ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
            ndbi = image.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
            return image.addBands([ndvi, ndbi])

        # Process collection
        landsat_processed = landsat_collection.map(mask_clouds_landsat8_c2)
        landsat_with_indices = landsat_processed.map(add_indices)

        # Calculate means
        mean_ndvi = landsat_with_indices.select('NDVI').mean().clip(roi)
        mean_ndbi = landsat_with_indices.select('NDBI').mean().clip(roi)

        print("Landsat NDVI and NDBI data processed successfully.")
        return mean_ndvi, mean_ndbi
    except Exception as e:
        print(f"Error processing Landsat data: {e}")
        return None, None

def get_worldcover_landcover(roi, year=2021):
    """
    Enhanced WorldCover data acquisition with validation.
    """
    print(f"Fetching ESA WorldCover data for {year}...")
    try:
        if year == 2020:
            landcover_image = ee.Image('ESA/WorldCover/v100/2020').select('Map')
        elif year == 2021:
            landcover_image = ee.Image('ESA/WorldCover/v200/2021').select('Map')
        else:
            print(f"WARNING: WorldCover data not available for {year}. Using 2021 as default.")
            landcover_image = ee.Image('ESA/WorldCover/v200/2021').select('Map')

        clipped_landcover = landcover_image.clip(roi).rename('Land_Cover')
        print("ESA WorldCover data processed successfully.")
        return clipped_landcover
    except Exception as e:
        print(f"Error processing WorldCover data: {e}")
        return None

def sample_to_dataframe(image, roi, num_pixels, scale):
    """
    Enhanced sampling with better error handling and validation.
    """
    print(f"Sampling {num_pixels} pixels at {int(scale)}m resolution...")
    try:
        sample_points_fc = image.sample(
            region=roi,
            scale=scale,
            numPixels=num_pixels,
            seed=RANDOM_STATE,
            geometries=False
        )
        
        df = geemap.ee_to_df(sample_points_fc)
        
        if df.empty:
            raise ValueError("No data was sampled. Check your ROI and date range.")
        
        print(f"Successfully extracted {len(df)} pixels into a DataFrame.")
        return df
    except Exception as e:
        print(f"Error during sampling: {e}")
        return None

# -----------------------------------------------------------------------------
# SECTION 3: Main Data Processing Workflow
# -----------------------------------------------------------------------------

print("\n--- Starting Enhanced Data Acquisition ---")

# Get all satellite data
lst_image = get_modis_lst(ROI, START_DATE, END_DATE)
ndvi_image, ndbi_image = get_landsat_indices(ROI, START_DATE, END_DATE)
landcover_image = get_worldcover_landcover(ROI, 2021)

# Validate that all data was acquired successfully
if any(img is None for img in [lst_image, ndvi_image, ndbi_image, landcover_image]):
    print("ERROR: Failed to acquire one or more datasets. Exiting.")
    exit()

# Combine all images into a single stack
combined_image = lst_image.addBands([ndvi_image, ndbi_image, landcover_image])
print("All data layers combined successfully.")

# Sample data to DataFrame
df_pixels = sample_to_dataframe(combined_image, ROI, NUM_PIXELS_FOR_ML, COMMON_RESOLUTION_METERS)

if df_pixels is None or df_pixels.empty:
    print("ERROR: Failed to create DataFrame. Exiting.")
    exit()

# Enhanced data cleaning
print("\n--- Enhanced Data Cleaning ---")
initial_rows = len(df_pixels)

# Remove NaN values
df_pixels.dropna(inplace=True)
print(f"Removed {initial_rows - len(df_pixels)} rows with NaN values.")

# Filter valid land cover types (ESA WorldCover classes)
valid_lc_classes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
df_pixels = df_pixels[df_pixels['Land_Cover'].isin(valid_lc_classes)]
print(f"Filtered to {len(df_pixels)} rows with valid land cover classes.")

# Map land cover codes to descriptive names
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

print("\n--- Data Summary ---")
print(df_pixels.head())
print(f"\nDataFrame shape: {df_pixels.shape}")
print("\nLand Cover Distribution:")
print(df_pixels['Land_Cover_Type'].value_counts())
print(f"\nDescriptive Statistics:")
print(df_pixels.describe())

# -----------------------------------------------------------------------------
# SECTION 4: Enhanced Exploratory Data Analysis
# -----------------------------------------------------------------------------

print("\n--- Enhanced EDA and Visualization ---")

def create_enhanced_plots(df):
    """Create comprehensive visualization plots"""
    
    # 1. Feature Distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # LST Distribution
    sns.histplot(data=df, x='LST_Celsius', kde=True, ax=axes[0,0])
    axes[0,0].set_title('Land Surface Temperature Distribution')
    axes[0,0].set_xlabel('Temperature (°C)')
    
    # NDVI Distribution
    sns.histplot(data=df, x='NDVI', kde=True, ax=axes[0,1])
    axes[0,1].set_title('NDVI Distribution')
    axes[0,1].set_xlabel('NDVI')
    
    # NDBI Distribution
    sns.histplot(data=df, x='NDBI', kde=True, ax=axes[1,0])
    axes[1,0].set_title('NDBI Distribution')
    axes[1,0].set_xlabel('NDBI')
    
    # LST by Land Cover
    sns.boxplot(data=df, x='Land_Cover_Type', y='LST_Celsius', ax=axes[1,1])
    axes[1,1].set_title('LST by Land Cover Type')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Correlation Analysis
    plt.figure(figsize=(10, 8))
    numerical_cols = ['LST_Celsius', 'NDVI', 'NDBI']
    correlation_matrix = df[numerical_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Environmental Variables')
    plt.savefig(os.path.join(OUTPUTS_DIR, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Scatter Plot Matrix
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # NDVI vs LST
    sns.scatterplot(data=df, x='NDVI', y='LST_Celsius', hue='Land_Cover_Type', 
                   alpha=0.6, ax=axes[0])
    axes[0].set_title('LST vs NDVI by Land Cover')
    
    # NDBI vs LST
    sns.scatterplot(data=df, x='NDBI', y='LST_Celsius', hue='Land_Cover_Type',
                   alpha=0.6, ax=axes[1])
    axes[1].set_title('LST vs NDBI by Land Cover')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'scatter_relationships.png'), dpi=300, bbox_inches='tight')
    plt.show()

create_enhanced_plots(df_pixels)

# -----------------------------------------------------------------------------
# SECTION 5: Enhanced UHI Detection using K-Means
# -----------------------------------------------------------------------------

print("\n--- Enhanced UHI Detection ---")

# Perform K-Means clustering on LST
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
df_pixels['LST_Cluster'] = kmeans.fit_predict(df_pixels[['LST_Celsius']])

# Get cluster centers and create meaningful labels
cluster_centers = kmeans.cluster_centers_.flatten()
cluster_order = np.argsort(cluster_centers)

# Create mapping from cluster ID to descriptive label
cluster_labels = {}
zone_names = ['Cool_Zone', 'Mild_Zone', 'Hot_Zone']
for i, cluster_id in enumerate(cluster_order):
    cluster_labels[cluster_id] = zone_names[i]

df_pixels['LST_Zone'] = df_pixels['LST_Cluster'].map(cluster_labels)

print("UHI Zones Distribution:")
print(df_pixels['LST_Zone'].value_counts())
print(f"\nCluster Centers (°C): {sorted(cluster_centers)}")

# Visualize clusters
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(data=df_pixels, x='LST_Zone', y='LST_Celsius')
plt.title('LST Distribution by UHI Zone')
plt.ylabel('Temperature (°C)')

plt.subplot(1, 2, 2)
zone_counts = df_pixels['LST_Zone'].value_counts()
plt.pie(zone_counts.values, labels=zone_counts.index, autopct='%1.1f%%', 
        colors=['blue', 'orange', 'red'])
plt.title('UHI Zone Distribution')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'uhi_zones_analysis.png'), dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------------------------
# SECTION 6: Enhanced Interactive Map Creation
# -----------------------------------------------------------------------------

print("\n--- Creating Enhanced Interactive Map ---")

def create_enhanced_folium_map(roi, lst_img, ndvi_img, ndbi_img, lc_img):
    """Create an enhanced interactive map with better styling and controls"""
    
    # Calculate map center
    roi_centroid = roi.centroid().coordinates().getInfo()
    map_center = [roi_centroid[1], roi_centroid[0]]
    
    # Create base map
    m = folium.Map(location=map_center, zoom_start=9, tiles='OpenStreetMap')
    
    # Enhanced visualization parameters
    lst_vis = {
        'min': df_pixels['LST_Celsius'].quantile(0.02), 
        'max': df_pixels['LST_Celsius'].quantile(0.98),
        'palette': ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000', '#800000']
    }
    
    ndvi_vis = {'min': -0.2, 'max': 0.8, 'palette': ['#8B4513', '#FFFFFF', '#00FF00']}
    ndbi_vis = {'min': -0.5, 'max': 0.5, 'palette': ['#00FF00', '#FFFF00', '#FF0000']}
    
    # WorldCover visualization
    lc_vis = {
        'min': 10, 'max': 95,
        'palette': ['#006400', '#ffbb22', '#ffff4c', '#f096ff', '#fa0000', 
                   '#b4b4b4', '#f0f0f0', '#0064c8', '#0096ff', '#6e6e6e']
    }
    
    # Helper function to add EE layers
    def add_ee_layer(folium_map, ee_image, vis_params, name):
        map_id_dict = ee.Image(ee_image).getMapId(vis_params)
        folium.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr='Google Earth Engine',
            overlay=True,
            control=True,
            name=name
        ).add_to(folium_map)
    
    # Add layers
    add_ee_layer(m, lst_img, lst_vis, f'Land Surface Temperature ({START_DATE} to {END_DATE})')
    add_ee_layer(m, ndvi_img, ndvi_vis, 'NDVI (Vegetation Index)')
    add_ee_layer(m, ndbi_img, ndbi_vis, 'NDBI (Built-up Index)')
    add_ee_layer(m, lc_img, lc_vis, 'Land Cover (ESA WorldCover)')
    
    # Add ROI boundary
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
    
    # Add enhanced controls
    folium.LayerControl(position='topright').add_to(m)
    
    # Add custom legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; background-color: rgba(255, 255, 255, 0.8);
                border:2px solid grey; z-index:9999; font-size:14px;
                padding: 10px">
    <p><b>UHI Analysis Legend</b></p>
    <p><i class="fa fa-thermometer" style="color:red"></i> LST: Land Surface Temperature</p>
    <p><i class="fa fa-leaf" style="color:green"></i> NDVI: Vegetation Index</p>
    <p><i class="fa fa-building" style="color:blue"></i> NDBI: Built-up Index</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# Create the enhanced map
enhanced_map = create_enhanced_folium_map(ROI, lst_image, ndvi_image, ndbi_image, landcover_image)

# Save the map
map_path = os.path.join(OUTPUTS_DIR, f'{CITY_NAME.replace(" ", "_").lower()}_enhanced_uhi_map.html')
enhanced_map.save(map_path)
print(f"Enhanced interactive map saved to: {map_path}")

# -----------------------------------------------------------------------------
# SECTION 7: Enhanced Machine Learning Model
# -----------------------------------------------------------------------------

print("\n--- Enhanced Machine Learning Pipeline ---")

# Prepare data for ML
df_ml = pd.get_dummies(df_pixels.drop(columns=['LST_Cluster', 'LST_Zone']), 
                       columns=['Land_Cover_Type'], prefix='LC')

# Define features and target
X = df_ml.drop(columns=['LST_Celsius'])
y = df_ml['LST_Celsius']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Features: {list(X.columns)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=None
)

# Feature scaling (important for some algorithms, though Random Forest is less sensitive)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Enhanced Random Forest Model
print("Training Enhanced Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=200,  # More trees for better performance
    max_depth=15,      # Prevent overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1          # Use all CPU cores
)

rf_model.fit(X_train_scaled, y_train)
print("Model training completed.")

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# -----------------------------------------------------------------------------
# SECTION 8: Enhanced Model Evaluation
# -----------------------------------------------------------------------------

print("\n--- Enhanced Model Evaluation ---")

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}°C")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}°C") 
print(f"R² Score: {r2:.3f}")

# Enhanced visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Actual vs Predicted
axes[0,0].scatter(y_test, y_pred, alpha=0.6, s=30)
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0,0].set_xlabel('Actual LST (°C)')
axes[0,0].set_ylabel('Predicted LST (°C)')
axes[0,0].set_title(f'Actual vs Predicted LST (R² = {r2:.3f})')
axes[0,0].grid(True, alpha=0.3)

# 2. Residuals plot
residuals = y_test - y_pred
axes[0,1].scatter(y_pred, residuals, alpha=0.6, s=30)
axes[0,1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0,1].set_xlabel('Predicted LST (°C)')
axes[0,1].set_ylabel('Residuals (°C)')
axes[0,1].set_title('Residuals Plot')
axes[0,1].grid(True, alpha=0.3)

# 3. Feature importance
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = feature_importance.nlargest(10)
top_features.plot(kind='barh', ax=axes[1,0])
axes[1,0].set_title('Top 10 Feature Importances')
axes[1,0].set_xlabel('Importance')

# 4. Prediction error distribution
axes[1,1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
axes[1,1].set_xlabel('Prediction Error (°C)')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Distribution of Prediction Errors')
axes[1,1].axvline(x=0, color='r', linestyle='--', lw=2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'enhanced_model_evaluation.png'), dpi=300, bbox_inches='tight')
plt.show()

# Feature importance analysis
print(f"\nTop 10 Most Important Features:")
for feature, importance in top_features.items():
    print(f"{feature}: {importance:.4f}")

# -----------------------------------------------------------------------------
# SECTION 9: Enhanced Results Summary
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("ENHANCED URBAN HEAT ISLAND ANALYSIS - RESULTS SUMMARY")
print("="*80)

print(f"\n📍 Study Area: {CITY_NAME}")
print(f"📅 Analysis Period: {START_DATE} to {END_DATE}")
print(f"🔬 Spatial Resolution: {COMMON_RESOLUTION_METERS}m")
print(f"📊 Total Pixels Analyzed: {len(df_pixels):,}")

print(f"\n🌡️  TEMPERATURE ANALYSIS:")
print(f"   • Mean LST: {df_pixels['LST_Celsius'].mean():.1f}°C")
print(f"   • Min LST: {df_pixels['LST_Celsius'].min():.1f}°C")
print(f"   • Max LST: {df_pixels['LST_Celsius'].max():.1f}°C")
print(f"   • Temperature Range: {df_pixels['LST_Celsius'].max() - df_pixels['LST_Celsius'].min():.1f}°C")

print(f"\n🏙️  UHI ZONES DISTRIBUTION:")
for zone, count in df_pixels['LST_Zone'].value_counts().items():
    percentage = (count / len(df_pixels)) * 100
    print(f"   • {zone}: {count:,} pixels ({percentage:.1f}%)")

print(f"\n🤖 MACHINE LEARNING MODEL PERFORMANCE:")
print(f"   • Model Type: Random Forest Regressor")
print(f"   • R² Score: {r2:.3f} (Variance explained)")
print(f"   • Mean Absolute Error: {mae:.2f}°C")
print(f"   • Root Mean Square Error: {rmse:.2f}°C")

print(f"\n📁 OUTPUT FILES GENERATED:")
print(f"   • Interactive Map: {map_path}")
print(f"   • Feature Distributions: {os.path.join(OUTPUTS_DIR, 'feature_distributions.png')}")
print(f"   • Correlation Matrix: {os.path.join(OUTPUTS_DIR, 'correlation_matrix.png')}")
print(f"   • UHI Zones Analysis: {os.path.join(OUTPUTS_DIR, 'uhi_zones_analysis.png')}")
print(f"   • Model Evaluation: {os.path.join(OUTPUTS_DIR, 'enhanced_model_evaluation.png')}")

print(f"\n🔍 KEY INSIGHTS:")
most_important_feature = feature_importance.idxmax()
print(f"   • Most influential factor: {most_important_feature}")
print(f"   • NDVI correlation with LST: {df_pixels[['NDVI', 'LST_Celsius']].corr().iloc[0,1]:.3f}")
print(f"   • NDBI correlation with LST: {df_pixels[['NDBI', 'LST_Celsius']].corr().iloc[0,1]:.3f}")

hottest_lc = df_pixels.groupby('Land_Cover_Type')['LST_Celsius'].mean().idxmax()
coolest_lc = df_pixels.groupby('Land_Cover_Type')['LST_Celsius'].mean().idxmin()
print(f"   • Hottest land cover type: {hottest_lc}")
print(f"   • Coolest land cover type: {coolest_lc}")

print(f"\n✅ ANALYSIS COMPLETE!")
print(f"All results saved in '{OUTPUTS_DIR}' directory.")
print("="*80)