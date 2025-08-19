import argparse
from typing import List

from config import AnalysisConfig
from pipeline import run_analysis


def parse_args() -> AnalysisConfig:
    p = argparse.ArgumentParser(description="UHI Analysis Pipeline")
    p.add_argument("--city", type=str, default="Mumbai", help="City name (for ROI lookup)")
    p.add_argument(
        "--roi-bounds",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="ROI bounds as west south east north",
    )
    p.add_argument("--years", nargs="+", type=int, default=[2022, 2023], help="Years to analyze")
    p.add_argument("--source", choices=["MODIS", "LANDSAT"], default="MODIS", help="Primary LST source")
    p.add_argument("--resolution", type=int, default=1000, help="Common resolution (meters)")
    p.add_argument("--pixels", type=int, default=5000, help="#pixels to sample for ML per period")
    p.add_argument("--clusters", default="auto", help="'auto' or integer")
    p.add_argument("--outputs", type=str, default="outputs", help="Outputs directory")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    p.add_argument("--no-tune", action="store_true", help="Disable hyperparameter tuning")

    args = p.parse_args()

    cfg = AnalysisConfig()
    cfg.city_name = args.city
    if args.roi_bounds:
        cfg.roi_bounds = list(args.roi_bounds)
    cfg.analysis_years = list(args.years)
    cfg.primary_lst_source = args.source
    cfg.common_resolution_m = args.resolution
    cfg.num_pixels_for_ml = args.pixels
    cfg.uhi_clusters = int(args.clusters) if args.clusters != "auto" else "auto"
    cfg.outputs_dir = args.outputs
    cfg.log_level = args.log_level
    cfg.perform_hyperparameter_tuning = not args.no_tune
    return cfg


if __name__ == "__main__":
    run_analysis(parse_args())