# 🏙️ Urban Heat Island (UHI) Detection & Prediction with Satellite Data and Machine Learning

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/your-github-username/urban-heat-island-prediction-ml?style=social)](https://github.com/ArhamOrioner/UHI-Analysis)
[![Issues](https://img.shields.io/github/issues/your-github-username/urban-heat-island-prediction-ml)](https://github.com/ArhamOrioner/UHI-Analysis/issues))

---

## 🚀 Project Overview

This project offers a **comprehensive and robust framework** for detecting, analyzing, and predicting **Urban Heat Islands (UHIs)** in _any configurable city_ using cutting-edge satellite imagery and advanced machine learning techniques. It harnesses the power of [Google Earth Engine (GEE)](https://earthengine.google.com/) for efficient cloud-based geospatial data acquisition and preprocessing, seamlessly integrated with Python's scientific computing stack for in-depth analysis, visualization, and predictive modeling.

The core script, `uhi_analysis.py`, is designed for maximum user-friendliness, embedding all configuration parameters directly within the file. A new, helpful utility script, `find_city_roi.py`, simplifies the process of defining geographic boundaries for your target city.

---

## 🌡️ Understanding Urban Heat Islands (UHIs)

Urban Heat Islands are metropolitan areas that experience significantly higher temperatures compared to their surrounding rural landscapes. This critical phenomenon is primarily driven by:

*   **Changes in Land Cover:** Replacement of natural vegetation with heat-absorbing concrete, asphalt, and buildings.
*   **Reduced Evapotranspiration:** Less green space leads to less natural cooling from water evaporation.
*   **Heat-Generating Human Activities:** Energy consumption from vehicles, industries, and air conditioning.

This project analyzes **Land Surface Temperature (LST)** patterns in relation to various environmental factors, enabling a deeper understanding of urban thermal dynamics and contributing to data-driven climate resilience strategies.

---

## ✨ Key Features & Methodology

The `uhi_analysis.py` script orchestrates a complete geospatial analysis and machine learning pipeline, providing powerful insights into urban thermal environments.

### ⚙️ Highly Configurable Parameters

All analysis settings are transparently defined at the top of `uhi_analysis.py` for easy customization:

*   `CITY_NAME`: Your target city (e.g., "Mumbai").
*   `ROI_BOUNDS`: Geographic coordinates defining your Area of Interest.
*   `ANALYSIS_YEARS`: Specific years for data collection.
*   `SEASONS_TO_ANALYZE`: Focus on specific seasons (e.g., "Summer", "Winter").
*   `PRIMARY_LST_SOURCE`: Choose between MODIS (1km) or Landsat 8 (30m) for LST.
*   Machine Learning hyperparameters for fine-tuning.

### 🛰️ Google Earth Engine (GEE) Data Acquisition

The script programmatically acquires and processes multi-temporal satellite data directly from GEE:

*   **Land Surface Temperature (LST):**
    *   Fetched from either **MODIS** (1km resolution) or **Landsat 8** (30m resolution), configurable via `PRIMARY_LST_SOURCE`.
    *   Includes automated data scaling and smoothing for consistency.
*   **Spectral Indices (from Landsat 8 optical bands):**
    *   **Normalized Difference Vegetation Index (NDVI):** Crucial for indicating vegetation health and density.
    *   **Normalized Difference Built-up Index (NDBI):** Highlights urban built-up areas.
*   **Land Cover/Land Use (LULC):**
    *   Utilizes **ESA WorldCover** (10m resolution), providing detailed classification of surface types (e.g., Built-up, Trees, Water).
*   **Topographic Features (from SRTM Digital Elevation Model - DEM):**
    *   Includes **Elevation**, **Slope**, and **Aspect** to account for terrain influence on temperature.
*   **Multi-Temporal Analysis:**
    *   Supports analysis across specified `ANALYSIS_YEARS` and `SEASONS_TO_ANALYZE` (e.g., Summer, Winter), automatically handling cross-year seasons.

**Note:** All acquired datasets are intelligently resampled to a common `COMMON_RESOLUTION_METERS` (defaulting to 1000m) for consistent analysis across different data sources.

### 📊 Data Preprocessing & Exploratory Data Analysis (EDA)

*   **Data Sampling:** Samples a defined number of pixels (`NUM_PIXELS_FOR_ML`) from the combined GEE layers into a Pandas DataFrame, preserving geographic coordinates.
*   **Cleaning & Engineering:** Robust handling of missing values, filtering for valid land cover classes, mapping numerical IDs to descriptive names, and one-hot encoding for categorical features, preparing data for ML readiness.
*   **Descriptive Statistics:** Provides essential summary statistics of all features, giving an initial data overview.
*   **Visualizations:** Generates insightful plots saved as PNG files:
    *   Histograms for feature distributions.
    *   A comprehensive correlation heatmap to understand inter-feature relationships.
    *   Scatter plots illustrating key relationships (e.g., LST vs. NDVI/NDBI).

### 🌡️ Urban Heat Island Detection & Analysis

*   **K-Means Clustering:** Applies unsupervised K-Means clustering on LST values to categorize the study area into distinct temperature zones (e.g., "Cool_Zone", "Mild_Zone", "Hot_Zone").
*   **Optimal Cluster Determination:** Automatically determines `optimal_n_clusters` using the Silhouette Score method, or allows for a fixed `UHI_CLUSTERS` input.
*   **UHI Intensity Calculation:** Quantifies the Urban Heat Island intensity by comparing mean LST in urban (Built-up) areas against mean LST in rural/natural areas.
*   **Spatial Hot Spot Analysis (Getis-Ord Gi\*):** Identifies statistically significant clusters of high (hot spots) or low (cold spots) LST values using local spatial autocorrelation. The method dynamically adapts to dataset size for robust neighbor identification (KNN or distance-based).

### 🤖 LST Prediction with Machine Learning

*   **Random Forest Regressor:** Trains a powerful Random Forest Regressor model to predict LST based on all relevant environmental features (spectral indices, land cover types, topographic features).
*   **Hyperparameter Tuning:** Optionally performs `RandomizedSearchCV` to find optimal model hyperparameters, significantly enhancing predictive performance.
*   **Cross-Validation:** Evaluates model robustness using k-fold cross-validation.
*   **Performance Metrics:** Reports standard regression metrics including Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and R-squared (R²) score.
*   **Model Evaluation Plots:** Generates plots for Actual vs. Predicted LST, Residuals, and Feature Importances, providing a visual assessment of model performance.

### 🧠 Model Interpretability (SHAP Analysis)

Integrates **SHAP (SHapley Additive exPlanations)** to provide deep insights into how each feature influences the model's LST predictions, explaining complex relationships in an understandable way. Generates clear SHAP summary plots.

### 🗺️ Interactive Map Output

Generates a dynamic and interactive HTML map (`[city_name]_enhanced_uhi_map.html`) showcasing all key layers, allowing for immersive exploration:

*   Mean Land Surface Temperature (LST)
*   NDVI (Vegetation Index)
*   NDBI (Built-up Index)
*   ESA WorldCover Land Cover
*   Elevation and Slope
*   Identified UHI Zones (Cool, Mild, Hot)
*   LST Hot/Cold Spots from Getis-Ord Gi\* analysis

The map includes a comprehensive legend and allows users to zoom, pan, and toggle layers for detailed exploration.

---

## 📈 Visuals of Results - Mumbai Area Test Results

*Actual outputs will vary based on your chosen city and parameters.*

### Figure 1: Feature Distributions

![Distributions Plot](https://github.com/user-attachments/assets/1ed9cf63-bee5-4761-8fc5-6c73997ac44e)

### Figure 2: Model Performance Metrics

![Model Performance Plot](https://github.com/user-attachments/assets/7903159e-2374-4ace-a85a-91b3bfce2e50)

### Figure 3: Interactive Map Showcasing Various Layers (GIF Representation)

![Interactive Map GIF](https://github.com/user-attachments/assets/baf294b7-9d70-4578-85eb-b63b78c6758f)

---

## 🚀 Getting Started

Follow these steps to set up and run the UHI analysis for your city:

### 1. Google Earth Engine Setup

*   **Sign up** for a free GEE account: [earthengine.google.com/signup](https://earthengine.google.com/signup)
*   **Authenticate** your GEE account in your terminal (a one-time setup):

    ```bash
    earthengine authenticate
    ```

### 2. Clone the Repository

Clone this project to your local machine:

```bash
git clone https://github.com/your-github-username/urban-heat-island-prediction-ml.git
cd urban-heat-island-prediction-ml
```

### 3. Install Dependencies

It's **highly recommended** to use a Python virtual environment to manage project dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment:
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages:
pip install -r requirements.txt
```

### 4. Configure for Your City

Before running the main analysis, you need to define the geographic boundaries (`ROI_BOUNDS`) for your city.

#### Option A: Using the `find_city_roi.py` helper script (Recommended for ease)

This script uses the Nominatim API to find approximate bounding box coordinates for a given city name.

```bash
python find_city_roi.py
```

Follow the prompts. The script will output a line like `ROI_BOUNDS = [west_lon, south_lat, east_lon, north_lat]`. **Copy this line.**

#### Option B: Using Google Earth Engine Code Editor (For precise boundaries)

1.  Go to [code.earthengine.google.com](https://code.earthengine.google.com).
2.  Search for your city.
3.  Use the "Draw a rectangle" tool on the map to define your precise Area of Interest (ROI).
4.  The coordinates will appear in the "Imports" section or the "Console". **Copy these `[west_lon, south_lat, east_lon, north_lat]` values.**

#### Once you have your `ROI_BOUNDS`:

1.  Open `uhi_analysis.py` in a text editor.
2.  Locate the "**PARAMETER SCRIPT SECTION**" at the top of the file.
3.  Update the `CITY_NAME` string (e.g., `"Mumbai"`).
4.  Replace the existing `ROI_BOUNDS` list with the coordinates you obtained (e.g., `ROI_BOUNDS = [72.768652, 18.887258, 73.018978, 19.300587]`).
5.  *(Optional but Recommended)* Adjust `ANALYSIS_YEARS` and `SEASONS_TO_ANALYZE` to match the specific climate patterns and periods you want to study for your chosen city.

### 5. Run the Main Analysis Script

With your parameters configured, execute the main script:

```bash
python uhi_analysis.py
```

This will perform the entire pipeline:

*   Fetch and process satellite data from GEE.
*   Perform UHI detection, intensity calculation, and hot spot analysis.
*   Train and evaluate the machine learning model.
*   Generate various plots (PNG files) in the `outputs/` directory.
*   Create an interactive HTML map (`[city_name]_enhanced_uhi_map.html`) in the `outputs/` directory.

---

## 💡 Key Insights & Results (Example - Adapt based on your actual output)

Based on typical results from this framework, you can expect to uncover insights like:

*   **Significant UHI Effect:** Analysis consistently confirms a measurable Urban Heat Island effect, with mean LST in built-up areas typically **X.XX°C higher** than surrounding natural areas.
*   **Spatial Hotspots Identified:** Getis-Ord Gi\* analysis pinpoints specific "Hot Spot" and "Cold Spot" clusters of LST, providing granular detail on the thermal landscape within the city.
*   **NDVI as a Key Mitigator:** Consistent strong negative correlation between LST and NDVI highlights the crucial role of urban green spaces in moderating temperatures. Increased vegetation cover directly correlates with cooler surface temperatures.
*   **Built-up Areas as Heat Contributors:** NDBI shows a clear positive correlation with LST, and Built-up land cover types are consistently among the hottest identified by both UHI clustering and feature importance analysis.
*   **Highly Predictive Model:** The Random Forest Regressor demonstrates strong predictive capabilities with an average cross-validated R² of **0.YY** and RMSE of **Z.ZZ °C**, proving its utility for accurate LST prediction.
*   **Top Influencing Factors:** SHAP analysis reinforces the importance of land cover types (especially Built-up and Trees), NDVI, and NDBI as the primary drivers of LST variations in the study area, offering actionable insights for urban planning.

---

## 🔮 Future Enhancements

We envision several exciting advancements for this project:

*   **Advanced Feature Integration:** Incorporate additional layers such as population density, building height, impervious surface area, and wind patterns for a more holistic analysis.
*   **Temporal Trend Analysis:** Expand the analysis to study UHI evolution over multi-decade periods or diurnal cycles (e.g., day vs. night LST).
*   **Climate Change Projections:** Integrate climate model outputs to predict future UHI scenarios under various climate change projections.
*   **Advanced ML Models:** Explore deep learning models (e.g., Convolutional Neural Networks for direct image processing) for more complex LST prediction and pattern recognition.
*   **Interactive Dashboard Development:** Build a web-based dashboard (e.g., using Dash, Streamlit, or Voilà) for dynamic exploration of UHI results by non-technical users, enhancing accessibility.
*   **Local Climate Zone (LCZ) Integration:** Classify the urban area into Local Climate Zones (LCZs) and analyze UHI characteristics specific to each zone, providing more nuanced insights for targeted interventions.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

---

## 📧 Contact

For questions or collaborations, please reach out via GitHub issues.
```
