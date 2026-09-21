import math
from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class AnalysisConfig:
    project_name: str = "Urban Heat Island (UHI) Analysis"
    author: str = "Arham Ansari"

    # Region of Interest
    city_name: str = "Mumbai"
    roi_bounds: Optional[List[float]] = None

    # Temporal
    analysis_years: List[int] = field(default_factory=lambda: [2022, 2023])
    ee_project: Optional[str] = None
    index_denominator_epsilon: float = 1e-6
    index_diagnostics: bool = False
    spatial_block_size_m: int = 5000

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

    def validate(self) -> None:
        if not self.analysis_years or any(
            y < 2013 or y > 9998 for y in self.analysis_years
        ):
            raise ValueError(
                "Analysis years must be >=2013 (Landsat 8 availability) and <=9998"
            )
        self.analysis_years = sorted(set(self.analysis_years))
        self.primary_lst_source = self.primary_lst_source.upper()
        if self.primary_lst_source not in ("MODIS", "LANDSAT"):
            raise ValueError("LST source must be MODIS or LANDSAT")
        minimum_scale = 1000 if self.primary_lst_source == "MODIS" else 30
        if not minimum_scale <= self.common_resolution_m <= 2000:
            raise ValueError(
                f"Resolution must be {minimum_scale}..2000 m; avoid false precision and excessive aggregation"
            )
        if self.num_pixels_for_ml < 10:
            raise ValueError("Request at least 10 sample pixels")
        if self.worldcover_year not in (2020, 2021):
            raise ValueError("WorldCover year must be 2020 or 2021")
        if (
            not math.isfinite(self.index_denominator_epsilon)
            or self.index_denominator_epsilon <= 0
        ):
            raise ValueError("Index denominator epsilon must be finite and positive")
        if self.spatial_block_size_m < self.common_resolution_m:
            raise ValueError(
                "Spatial blocks must be at least as large as the analysis grid"
            )
        if self.uhi_clusters != "auto" and (
            not isinstance(self.uhi_clusters, int) or self.uhi_clusters < 2
        ):
            raise ValueError("Clusters must be 'auto' or an integer >=2")
        if self.tuning_iterations < 1:
            raise ValueError("Tuning iterations must be positive")
        if self.roi_bounds is not None:
            if len(self.roi_bounds) != 4 or not all(
                math.isfinite(v) for v in self.roi_bounds
            ):
                raise ValueError("ROI needs four finite bounds: west south east north")
            west, south, east, north = self.roi_bounds
            if not (-180 <= west < east <= 180 and -56 <= south < north <= 60):
                raise ValueError(
                    "ROI must be ordered, not cross the antimeridian, and lie within SRTM coverage (-56..60 latitude)"
                )
