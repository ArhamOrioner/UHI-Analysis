# 🛰️ Mapping Urban Heat Stress with Satellite Data and Python

## Project Overview
This project focuses on detecting and predicting Urban Heat Islands (UHIs) using satellite imagery and machine learning techniques. It leverages Google Earth Engine (GEE) for efficient data acquisition and preprocessing of geospatial data, followed by analysis, visualization, and predictive modeling using Python's scientific stack.

## Urban Heat Islands (UHIs)
Urban Heat Islands are metropolitan areas that are significantly warmer than their surrounding rural areas due to human activities. This project analyzes the Land Surface Temperature (LST) patterns in **New Delhi**, India, in relation to vegetation cover and land use.

## Key Steps & Features:
1.  **Google Earth Engine (GEE) Data Acquisition:**
    *   Programmatically pulls and processes:
        *   **Land Surface Temperature (LST):** from MODIS (1km resolution).
        *   **Normalized Difference Vegetation Index (NDVI):** derived from Landsat 8 (30m resolution), including basic cloud masking.
        *   **Land Cover/Land Use (LULC):** from ESA WorldCover (10m resolution).
    *   All datasets are resampled to a common `100m` resolution for consistent analysis.
2.  **Data Preparation:**
    *   Samples pixel data (LST, NDVI, LULC) into a Pandas DataFrame.
    *   Converts LST from Kelvin to Celsius.
    *   Maps numerical Land Cover IDs to descriptive names.
    *   Performs one-hot encoding for categorical land cover data, ready for machine learning.
3.  **Exploratory Data Analysis (EDA):**
    *   Provides descriptive statistics of the extracted features.
    *   Visualizes feature distributions (histograms).
    *   Analyzes relationships between LST and other features (e.g., NDVI) using scatter plots.
    *   Generates a correlation heatmap to understand inter-feature relationships.
4.  **UHI Detection (Unsupervised Learning):**
    *   Applies **K-Means Clustering** to `LST_Celsius` to categorize urban areas into distinct temperature zones (e.g., "Cool", "Mild", "Hot").
    *   Visualizes these identified UHI zones on an interactive map.
5.  **LST Prediction (Supervised Learning):**
    *   Trains a **Random Forest Regressor** model to predict `LST_Celsius` based on `NDVI` and one-hot encoded `Land_Cover_Type` features.
    *   Splits data into training and testing sets (80/20 ratio).
    *   Evaluates model performance using Mean Absolute Error (MAE) and R-squared (R² score).
    *   Visualizes model predictions against actual values with an "Actual vs. Predicted" scatter plot.
6.  **Feature Importance:**
    *   Identifies the most influential features contributing to LST predictions, providing insights into factors driving urban heat.
7.  **Interactive Map Output:**
    *   Generates a dynamic HTML map showcasing the mean LST, NDVI, Land Cover, and identified UHI zones, allowing users to zoom, pan, and toggle layers.

Visuals of Result
![feature_distributions](https://github.com/user-attachments/assets/eb3b7d4e-0d04-4116-9602-90948f135e89)
![enhanced_model_evaluation](https://github.com/user-attachments/assets/8b5a3310-5e75-4b8b-b478-1e935ffa9a18)
![ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/da4be6f3-6a5b-4c6b-9093-e135ce342aac)


## How to Run the Project:
1.  **Google Earth Engine Setup:**
    *   Sign up for a free GEE account: [earthengine.google.com/signup](https://earthengine.google.com/signup/)
    *   Authenticate your GEE account in your terminal: `earthengine authenticate`
2.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-github-username/urban-heat-island-prediction-ml.git
    cd urban-heat-island-prediction-ml
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the script:**
    ```bash
    python urban_heat_island_project.py
    ```
    This will generate various plots and an interactive HTML map file (`UHI_Map_New Delhi_2023-06-01_to_2023-08-31.html`) in your project directory.

## Insights & Results (Example - Adapt based on your actual output):
*   **High LST in Built-up Areas:** The interactive map clearly shows higher surface temperatures concentrated in areas classified as 'Built-up' land cover, validating the UHI effect.
*   **NDVI's Cooling Effect:** A strong negative correlation between LST and NDVI suggests that greener areas tend to be cooler, highlighting the importance of urban green spaces.
*   **Predictive Model Performance:** The Random Forest Regressor achieved an R² score of around `0.XX` and an MAE of `Y.YY` °C, indicating a reasonable ability to predict LST variations.
*   **Key Drivers of Heat:** Feature importance analysis revealed that `NDVI` and `Built-up` land cover (from one-hot encoding) were the most significant predictors of LST, underscoring their role in urban heat patterns.

## Future Enhancements:
*   Incorporate more features: Population density, building height, road network density, albedo.
*   Analyze temporal trends: Study UHI evolution over multiple years or different seasons.
*   Spatial Autocorrelation: Account for spatial dependency in temperature data.
*   Advanced ML Models: Explore deep learning models for image-based LST prediction.
*   Interactive Dashboard: Build a web-based dashboard (e.g., with Dash or Streamlit) for dynamic exploration.
