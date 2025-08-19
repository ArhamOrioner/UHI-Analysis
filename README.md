# Urban Heat Island (UHI) Analysis – Modular Pipeline

This is a clean, modular rewrite of your UHI script. It fetches data from Google Earth Engine (GEE), samples pixels to a DataFrame, performs EDA, detects UHI zones, runs hot spot analysis (Gi*), trains a Random Forest regressor for LST prediction, generates plots, and creates an interactive Folium map.

## Quick start

1) Install dependencies (prefer a virtual environment):

```
pip install -r requirements.txt
```

2) Authenticate Earth Engine (one-time):

```
earthengine authenticate
```

3) Run the pipeline (examples):

```
python app.py --city "Mumbai" --years 2022 2023 --source MODIS --resolution 1000 --pixels 5000

# or use explicit bounds (west south east north)
python app.py --roi-bounds 72.7 18.8 73.2 19.3 --years 2022 2023 --source MODIS
```

Outputs are saved to the `outputs` directory by default.

## Project layout

```
Project/
  app.py                 # CLI entrypoint
  requirements.txt       # Dependencies
  README.md              # This file
  uhi/
    __init__.py
    config.py            # Dataclass config
    utils.py             # Logging, ROI lookup, helpers
    gee.py               # Earth Engine data access
    sampling.py          # Sampling to DataFrame
    eda.py               # EDA plots
    clustering.py        # KMeans, zone labeling, plots
    spatial.py           # Getis-Ord Gi* hot/cold spots + plot
    map.py               # Folium map creation
    ml.py                # Feature prep, RF training, eval, SHAP
    pipeline.py          # Orchestration
```

## Notes

- By default, the pipeline uses KNN spatial weights (k=8) for Gi* to avoid mixing meters with degrees.
- RandomForest does not require feature scaling, so none is applied.
- You can adjust behavior via CLI flags; see `python app.py -h`.


