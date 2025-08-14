
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

@dataclass
class AnalysisConfig:
    project_name: str = "Urban Heat Island (UHI) Analysis"
    author: str = "Arham Ansari"

    # Region of Interest
    city_name: str = "Mumbai"
    roi_bounds: List[float] = field(default_factory=lambda: [72.7, 18.8, 73.2, 19.3])  # [west, south, east, north]

    # Temporal
    analysis_years: List[int] = field(default_factory=lambda: [2022, 2023])
    seasons_to_analyze: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {"Summer": (6, 8), "Winter": (12, 2)}
    )
    start_date: str = "2022-01-01"  # used for overall map layers
    end_date: str = "2023-12-31"

    # Data sources and resolution
    primary_lst_source: str = "MODIS"  # 'MODIS' or 'LANDSAT'
    common_resolution_m: int = 1000
    num_pixels_for_ml: int = 5000

    # Reproducibility
    random_state: int = 42

    # Clustering
    uhi_clusters: Union[str, int] = "auto"  # 'auto' or integer
    uhi_cluster_range: List[int] = field(default_factory=lambda: list(range(2, 6)))

    # Random Forest defaults (used when tuning is disabled)
    rf_n_estimators: int = 200
    rf_max_depth: int = 15
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2

    # Hyperparameter tuning
    perform_hyperparameter_tuning: bool = True
    tuning_iterations: int = 30

    # Data products
    worldcover_year: int = 2021

    # IO
    outputs_dir: str = "outputs"
    log_level: str = "INFO"

    # Map
    initial_zoom: int = 9
