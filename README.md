##🛰️ Advanced Urban Heat Island Analysis with GEE & Machine Learning
![alt text](https://img.shields.io/badge/Python-3.9%2B-blue.svg) ![alt text](https://img.shields.io/badge/License-MIT-yellow.svg) ![alt text](https://img.shields.io/github/stars/ArhamOrioner/UHI-Analysis?style=social) ![alt text](https://img.shields.io/github/issues/ArhamOrioner/UHI-Analysis)

This repository provides a powerful, modular, and command-line-driven framework for detecting, analyzing, and predicting Urban Heat Islands (UHIs) in any city worldwide. It leverages the planetary-scale data catalog of Google Earth Engine (GEE) for efficient satellite imagery acquisition and uses a robust Python stack (Scikit-learn, GeoPandas, SHAP) to deliver deep, actionable insights.
The entire analysis is now orchestrated through a clean command-line interface (app.py), making it simple to configure and run complex geospatial workflows for different cities and years.

✨ Showcase: Uncovering the Las Vegas Heat Island
To demonstrate the power of this tool, here are the final results from a full annual analysis of Las Vegas for the year 2024.
Final Summary Report
The analysis provides a clear, human-readable summary of the key findings, including the quantified UHI effect and the performance of the predictive model.
<img width="1234" height="344" alt="Screenshot 2025-08-19 220943" src="https://github.com/user-attachments/assets/d828b1ed-a5b5-43b7-85d3-dfc307c22837" />

Interactive Map
An enhanced interactive map is generated, featuring multiple professional baselayers (including dark mode and satellite), a fullscreen button, and a comprehensive legend.
![ScreenRecording2025-08-19221122-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/2d51ac7d-91ac-49d8-921e-794f1e55306e)

Key Visual Insights
The analysis produces a suite of publication-quality infographics that visually explain the UHI phenomenon and the factors driving it.
### Hot/Cold Spot Analysis & Model Performance
<div align="center">
  <img src="https://github.com/user-attachments/assets/5a785621-f866-44c4-8263-b6cd503143ad" alt="2024_hot_spots" width="48%"/>
  <img src="https://github.com/user-attachments/assets/b1a5f546-a2b4-4075-b6a9-f646dc92c3df" alt="correlation_matrix" width="48%"/>
</div>
<div align="center">
  <img src="https://github.com/user-attachments/assets/acefb1fe-5445-4389-ade7-11e916958766" alt="2024_uhi_zones" width="80%"/>
</div>

### UHI Zone Distribution & Feature Correlation Matrix
<div align="center">
  <img src="https://github.com/user-attachments/assets/f82b3db6-3c90-4126-a19d-2db0dc53e20b" alt="feature_distributions" width="48%"/>
  <img src="https://github.com/user-attachments/assets/3260ded5-f839-4d6b-b672-0da038f07bf9" alt="scatter_relationships" width="48%"/>
</div>
<div align="center">
  <img src="https://github.com/user-attachments/assets/8dff639e-51da-4d4a-81e1-61331c030497" alt="model_evaluation" width="80%"/>
</div>


⚙️ Key Features & Methodology
This project is a complete, end-to-end pipeline for UHI analysis.
Powerful Command-Line Interface: Run the entire analysis for any city with a single command. app.py handles all configuration through intuitive arguments.
Automated Data Acquisition (GEE): Programmatically fetches and preprocesses key satellite datasets:
Land Surface Temperature (LST): From MODIS or Landsat 8.
Spectral Indices: NDVI (Vegetation) & NDBI (Built-up).
Land Cover: High-resolution ESA WorldCover.
Topography: Elevation, Slope, and Aspect from SRTM DEM.
Annual Analysis Engine: The logic is now built to perform a complete, self-contained analysis for each specified year, handling past years and the current year (up to the latest available data) automatically.

Advanced Analytics:
UHI Intensity: Quantifies the temperature difference between urban and rural areas.
K-Means Clustering: Automatically identifies distinct temperature zones (Cool, Mild, Hot) within the city.
Hot Spot Analysis (Getis-Ord Gi*): Pinpoints statistically significant clusters of extreme heat and cold.
Predictive Modeling & Interpretability:
Trains a Random Forest Regressor to predict temperature based on landscape features.
Includes hyperparameter tuning (RandomizedSearchCV) for optimal model performance.
Integrates SHAP (SHapley Additive exPlanations) to explain why the model makes its predictions, identifying the most influential factors.
Professional Outputs:
Generates a suite of enhanced, clearly labeled PNG infographics.
Produces a polished, multi-layer interactive HTML map.
Logs a final, easy-to-understand summary report to the console.

🚀 Getting Started
Follow these steps to set up the project and run your first analysis.
Prerequisites
Python 3.9+
Git

Step 1: Clone the repository:
```bash
git clone https://github.com/ArhamOrioner/UHI-Analysis.git
cd UHI-Analysis
```
Step 2: Install Dependencies
It's highly recommended to use a Python virtual environment.

```bash
Copy
Edit
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install required packages from the requirements file
pip install -r requirements.txt
```
Step 3: Run an Analysis
Execute the analysis using app.py. The script will automatically find the geographic boundaries for the specified city.

Basic Example (Las Vegas, 2023)

```bash
Copy
Edit
python app.py --city "Las Vegas" --years 2023
Multi-Year Example (Dubai, MODIS data)
```
```bash
Copy
Edit
python app.py --city "Dubai" --years 2022 2023 --source MODIS
```
Note: A full analysis for a single year can take 5–10 minutes to complete, depending on the size of the city and the number of available satellite images. The hyperparameter tuning step is the most time-intensive part.
All results, including plots, logs, and the interactive map, will be saved in the outputs/ directory.


🔮 Future Enhancements
Advanced Feature Integration: Incorporate population density, building height, and wind patterns.
Temporal Trend Analysis: Study UHI evolution over multi-decade periods.
Climate Change Projections: Integrate climate models to predict future UHI scenarios.
Interactive Dashboard: Build a web-based dashboard (e.g., using Streamlit or Dash) for dynamic exploration of results.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
