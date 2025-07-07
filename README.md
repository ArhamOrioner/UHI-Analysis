🏙️🔥 Urban Heat Island (UHI) Detection & Prediction with Satellite Data and Machine Learning

Project Overview

This project provides a comprehensive and robust framework for detecting, analyzing, and predicting Urban Heat Islands (UHIs) in any configurable city using satellite imagery and advanced machine learning techniques. It leverages the power of Google Earth Engine (GEE) for efficient cloud-based geospatial data acquisition and preprocessing, combined with Python's scientific computing stack for in-depth analysis, visualization, and predictive modeling.

The primary script, uhi_analysis.py, is now fully self-sufficient, containing all configuration parameters directly within the file for ease of modification. A new utility script, find_city_roi.py, has been introduced to help users easily define the geographic boundaries for their target city.

Urban Heat Islands (UHIs)

Urban Heat Islands are metropolitan areas that experience significantly higher temperatures compared to their surrounding rural landscapes. This phenomenon is primarily driven by changes in land cover (e.g., replacement of vegetation with concrete and asphalt), reduced evapotranspiration, and heat-generating human activities. This project analyzes Land Surface Temperature (LST) patterns in relation to various environmental factors, enabling a deeper understanding of urban thermal dynamics.

Key Features & Methodology

The uhi_analysis.py script orchestrates a complete geospatial analysis and machine learning pipeline:

⚙️ Configurable Parameters

All analysis settings, including CITY_NAME, ROI_BOUNDS, ANALYSIS_YEARS, SEASONS_TO_ANALYZE, PRIMARY_LST_SOURCE, and machine learning hyperparameters, are defined directly at the top of uhi_analysis.py for easy customization.

🛰️ Google Earth Engine (GEE) Data Acquisition

The script programmatically acquires and processes multi-temporal satellite data:

Land Surface Temperature (LST): Fetched from either MODIS (1km resolution) or Landsat 8 (30m resolution), configurable via PRIMARY_LST_SOURCE. Includes data scaling and smoothing.

Spectral Indices:

Normalized Difference Vegetation Index (NDVI): Derived from Landsat 8 optical bands, indicating vegetation health and density.

Normalized Difference Built-up Index (NDBI): Derived from Landsat 8 optical bands, highlighting urban built-up areas.

Land Cover/Land Use (LULC): From ESA WorldCover (10m resolution), providing detailed classification of surface types.

Topographic Features: Derived from SRTM Digital Elevation Model (DEM), including Elevation, Slope, and Aspect.

Multi-Temporal Analysis: Supports analysis across specified ANALYSIS_YEARS and SEASONS_TO_ANALYZE (e.g., Summer, Winter), automatically handling cross-year seasons.

All acquired datasets are resampled to a common COMMON_RESOLUTION_METERS (defaulting to 1000m) for consistent analysis.

📊 Data Preprocessing & Exploratory Data Analysis (EDA)

Data Sampling: Samples a defined number of pixels (NUM_PIXELS_FOR_ML) from the combined GEE layers into a Pandas DataFrame, retaining geographic coordinates.

Cleaning & Engineering: Handles missing values, filters for valid land cover classes, maps numerical IDs to descriptive names, and performs one-hot encoding for categorical features for ML readiness.

Descriptive Statistics: Provides summary statistics of all features.

Visualizations: Generates enhanced plots including histograms for feature distributions, a comprehensive correlation heatmap, and scatter plots illustrating relationships (e.g., LST vs. NDVI/NDBI), saved as PNG files.

🌡️ Urban Heat Island Detection & Analysis

K-Means Clustering: Applies unsupervised K-Means clustering on LST values to categorize the study area into distinct temperature zones (e.g., "Cool_Zone", "Mild_Zone", "Hot_Zone").

Optimal Cluster Determination: Automatically determines the optimal_n_clusters using the Silhouette Score method, or allows for fixed UHI_CLUSTERS input.

UHI Intensity Calculation: Quantifies the Urban Heat Island intensity by comparing mean LST in urban (Built-up) areas against mean LST in rural/natural areas.

Spatial Hot Spot Analysis (Getis-Ord Gi):* Identifies statistically significant clusters of high (hot spots) or low (cold spots) LST values using local spatial autocorrelation. The method dynamically adapts to dataset size for robust neighbor identification (KNN or distance-based).

🤖 LST Prediction with Machine Learning

Random Forest Regressor: Trains a powerful Random Forest Regressor model to predict LST based on all relevant environmental features (spectral indices, land cover types, topographic features).

Hyperparameter Tuning: Optionally performs RandomizedSearchCV to find optimal model hyperparameters, significantly enhancing predictive performance.

Cross-Validation: Evaluates model robustness using k-fold cross-validation.

Performance Metrics: Reports standard regression metrics including Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and R-squared (R²) score.

Model Evaluation Plots: Generates plots for Actual vs. Predicted LST, Residuals, and Feature Importances.

🧠 Model Interpretability (SHAP Analysis)

Integrates SHAP (SHapley Additive exPlanations) to provide deep insights into how each feature influences the model's LST predictions, explaining complex relationships in an understandable way. Generates SHAP summary plots.

🗺️ Interactive Map Output

Generates a dynamic and interactive HTML map ([city_name]_enhanced_uhi_map.html) showcasing all key layers:

Mean Land Surface Temperature (LST)

NDVI (Vegetation Index)

NDBI (Built-up Index)

ESA WorldCover Land Cover

Elevation and Slope

Identified UHI Zones (Cool, Mild, Hot)

LST Hot/Cold Spots from Getis-Ord Gi* analysis

The map includes a comprehensive legend and allows users to zoom, pan, and toggle layers for detailed exploration.

Visuals of Results

Examples of generated plots and the interactive map (actual outputs will vary based on city and parameters):

![alt text](https://github.com/user-attachments/assets/eb3b7d4e-0d04-4116-9602-90948f135e89)

Figure 1: Distributions of LST, NDVI, NDBI, and LST by Land Cover Type.

![alt text](https://github.com/user-attachments/assets/8b5a3310-5e75-4b8b-b478-1e935ffa9a18)

Figure 2: Model performance metrics including Actual vs Predicted, Residuals, and Feature Importances.

![alt text](https://github.com/user-attachments/assets/baf294b7-9d70-4578-85eb-b63b78c6758f)

Figure 3: Interactive map showcasing various layers like LST, Land Cover, and UHI Zones (GIF representation).

Getting Started
1. Google Earth Engine Setup

Sign up for a free GEE account: earthengine.google.com/signup

Authenticate your GEE account in your terminal (one-time setup):

Generated bash
earthengine authenticate

2. Clone the Repository
Generated bash
git clone https://github.com/your-github-username/urban-heat-island-prediction-ml.git
cd urban-heat-island-prediction-ml
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
3. Install Dependencies

It's highly recommended to use a Python virtual environment to manage project dependencies:

Generated bash
python -m venv venv
# Activate the virtual environment:
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages:
pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
4. Configure for Your City

Before running the main analysis, you need to define the geographic boundaries (ROI_BOUNDS) for your city.

Option A: Using the find_city_roi.py helper script (Recommended for ease)
This script uses the Nominatim API to find approximate bounding box coordinates for a given city name.

Generated bash
python find_city_roi.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Follow the prompts. The script will output a line like ROI_BOUNDS = [west_lon, south_lat, east_lon, north_lat]. Copy this line.

Option B: Using Google Earth Engine Code Editor (For precise boundaries)

Go to code.earthengine.google.com.

Search for your city.

Use the "Draw a rectangle" tool on the map to define your precise Area of Interest.

The coordinates will appear in the "Imports" section or the "Console". Copy these [west_lon, south_lat, east_lon, north_lat] values.

Once you have your ROI_BOUNDS:

Open uhi_analysis.py in a text editor.

Locate the "PARAMETER SCRIPT SECTION" at the top of the file.

Update the CITY_NAME string (e.g., "Mumbai").

Replace the existing ROI_BOUNDS list with the coordinates you obtained (e.g., ROI_BOUNDS = [72.768652, 18.887258, 73.018978, 19.300587]).

(Optional but Recommended) Adjust ANALYSIS_YEARS and SEASONS_TO_ANALYZE to match the specific climate patterns and periods you want to study for your chosen city.

5. Run the Main Analysis Script

With your parameters configured, execute the main script:

Generated bash
python uhi_analysis.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

This will:

Fetch and process satellite data.

Perform UHI detection, intensity calculation, and hot spot analysis.

Train and evaluate the machine learning model.

Generate various plots (PNG files) in the outputs/ directory.

Create an interactive HTML map ([city_name]_enhanced_uhi_map.html) in the outputs/ directory.

Key Insights & Results (Example - Adapt based on your actual output)

Significant UHI Effect: Analysis confirms a measurable Urban Heat Island effect, with mean LST in built-up areas typically X.XX°C higher than surrounding natural areas.

Spatial Hotspots Identified: Getis-Ord Gi* analysis pinpoints specific "Hot Spot" and "Cold Spot" clusters of LST, providing granular detail on the thermal landscape.

NDVI as a Key Mitigator: Consistent strong negative correlation between LST and NDVI highlights the crucial role of urban green spaces in moderating temperatures.

Built-up Areas as Heat Contributors: NDBI shows a positive correlation with LST, and Built-up land cover types are consistently among the hottest identified by both UHI clustering and feature importance.

Highly Predictive Model: The Random Forest Regressor demonstrates strong predictive capabilities with an average cross-validated R² of 0.YY and RMSE of Z.ZZ °C, proving its utility for LST prediction.

Top Influencing Factors: SHAP analysis reinforces the importance of land cover types (especially Built-up and Trees), NDVI, and NDBI as the primary drivers of LST variations in the study area.

Future Enhancements

Advanced Feature Integration: Incorporate additional layers such as population density, building height, impervious surface area, and wind patterns.

Temporal Trend Analysis: Expand the analysis to study UHI evolution over multi-decade periods or diurnal cycles.

Climate Change Projections: Integrate climate model outputs to predict future UHI scenarios.

Advanced ML Models: Explore deep learning models (e.g., Convolutional Neural Networks for direct image processing) for more complex LST prediction.

Interactive Dashboard Development: Build a web-based dashboard (e.g., using Dash, Streamlit, or Voilà) for dynamic exploration of UHI results by non-technical users.

Local Climate Zone (LCZ) Integration: Classify the urban area into LCZs and analyze UHI characteristics specific to each zone.
