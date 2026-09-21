import argparse

from config import AnalysisConfig


def parse_args() -> AnalysisConfig:
    p = argparse.ArgumentParser(description="UHI Analysis Pipeline")
    p.add_argument(
        "--city", type=str, default="Mumbai", help="City name (for ROI lookup)"
    )
    p.add_argument(
        "--roi-bounds",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="ROI bounds as west south east north",
    )
    p.add_argument(
        "--years", nargs="+", type=int, default=[2022, 2023], help="Years to analyze"
    )
    p.add_argument(
        "--source",
        choices=["MODIS", "LANDSAT"],
        default="MODIS",
        help="Primary LST source",
    )
    p.add_argument(
        "--resolution", type=int, default=1000, help="Common resolution (meters)"
    )
    p.add_argument(
        "--pixels", type=int, default=5000, help="#pixels to sample for ML per period"
    )
    p.add_argument("--clusters", default="auto", help="'auto' or integer")
    p.add_argument("--outputs", type=str, default="outputs", help="Outputs directory")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    p.add_argument(
        "--no-tune", action="store_true", help="Disable hyperparameter tuning"
    )
    p.add_argument("--project", help="Earth Engine Cloud project (or set EE_PROJECT)")
    p.add_argument("--worldcover-year", type=int, choices=[2020, 2021], default=2021)
    p.add_argument(
        "--index-epsilon",
        type=float,
        default=1e-6,
        help="Minimum absolute index denominator in reflectance units",
    )
    p.add_argument(
        "--index-diagnostics",
        action="store_true",
        help="Estimate observations affected by negative-input masking",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=5000,
        help="Spatial validation block size in meters",
    )

    args = p.parse_args()

    cfg = AnalysisConfig()
    cfg.city_name = args.city
    if args.roi_bounds:
        cfg.roi_bounds = list(args.roi_bounds)
    cfg.analysis_years = list(args.years)
    cfg.primary_lst_source = args.source
    cfg.common_resolution_m = args.resolution
    cfg.num_pixels_for_ml = args.pixels
    try:
        cfg.uhi_clusters = int(args.clusters) if args.clusters != "auto" else "auto"
    except ValueError:
        p.error("--clusters must be 'auto' or an integer >=2")
    cfg.outputs_dir = args.outputs
    cfg.log_level = args.log_level
    cfg.perform_hyperparameter_tuning = not args.no_tune
    cfg.ee_project = args.project
    cfg.worldcover_year = args.worldcover_year
    cfg.index_denominator_epsilon = args.index_epsilon
    cfg.index_diagnostics = args.index_diagnostics
    cfg.spatial_block_size_m = args.block_size
    try:
        cfg.validate()
    except ValueError as exc:
        p.error(str(exc))
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    from pipeline import run_analysis

    raise SystemExit(0 if run_analysis(cfg) else 1)
