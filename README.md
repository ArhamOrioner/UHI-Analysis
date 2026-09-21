# Urban Heat Island Analysis with Earth Engine

Annual daytime land-surface-temperature analysis using MODIS or Landsat 8, Landsat NDVI/NDBI, ESA WorldCover and SRTM. The CLI produces sampled surface-temperature contrasts, relative temperature zones, exploratory Gi* hot/cold spots, a spatially held-out Random Forest evaluation, plots and annual maps.

This is an exploratory **surface UHI** workflow. It does not measure near-surface air temperature, establish causal effects, or forecast future heat. Review the study boundary, reference land cover, observation coverage and spatial scale before interpreting the results.

## Setup

Use Python 3.11 or newer; local regression checks use Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
earthengine authenticate
```

On macOS/Linux, activate with `source .venv/bin/activate`. Use a Google Cloud project enabled and registered for Earth Engine. Supply `--project` or set the `EE_PROJECT` environment variable. See [Earth Engine authentication](https://developers.google.com/earth-engine/guides/auth).

## Run

```powershell
python app.py --city "Mumbai" --years 2023 --project YOUR_CLOUD_PROJECT --no-tune --index-diagnostics --outputs outputs/mumbai_2023
```

Explicit bounds avoid a geocoder request and make the study area reproducible:

```powershell
python app.py --city "Mumbai" --roi-bounds 72.75 18.85 73.05 19.35 --years 2022 2023 --project YOUR_CLOUD_PROJECT --outputs outputs/mumbai_multi
```

Landsat temperature analysis at a 100 m output grid:

```powershell
python app.py --city "Las Vegas" --years 2024 --source LANDSAT --resolution 100 --project YOUR_CLOUD_PROJECT --no-tune --outputs outputs/vegas_2024
```

Useful options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--years` | 2022 2023 | Annual intervals; current year ends at today, exclusive |
| `--source` | MODIS | MODIS Terra daytime or Landsat 8 surface temperature |
| `--resolution` | 1000 | Output grid spacing in meters |
| `--pixels` | 5000 | Approximate complete samples requested per year |
| `--clusters` | auto | Silhouette-selected clusters or an integer >=2 |
| `--worldcover-year` | 2021 | Static reference map: 2020 or 2021 |
| `--block-size` | 5000 | Spatial holdout/tuning block size in meters |
| `--index-epsilon` | 0.000001 | Mask index denominators with absolute sum <= this value |
| `--index-diagnostics` | off | Estimate the negative-input masking condition; adds EE work |
| `--no-tune` | off | Skip grouped hyperparameter search |

Use `python app.py --help` for all options. Configuration is validated before accessing Earth Engine. Supported analysis years start at 2013 because Landsat 8 indices are required with either LST source. The ROI must lie within the workflow's SRTM coverage limits, latitude -56 to 60, and cannot cross the antimeridian. Resolution is restricted to 1000–2000 m for MODIS and 30–2000 m for Landsat; 30 m product spacing does not imply 30 m independent thermal detail.

## Processing changes

- **NDVI/NDBI:** apply Collection 2 scale and offset, then explicit ratio expressions. Negative reflectances remain eligible; near-zero denominators are masked. Existing QA masks survive. No automatic clipping to [-1, 1].
- **Quality:** Landsat rejects fill, dilated cloud, cirrus, cloud, shadow, snow, radiometric saturation and terrain occlusion. Temperature scenes require L2SP. MODIS uses a strict QC_Day policy before averaging, with no focal smoothing.
- **Dates and grid:** complete years use January 1 to the following January 1, exclusive. Continuous bands are aggregated by mean and land cover by mode on an explicit EPSG:6933 grid. Temporal composites receive a source projection before spatial aggregation.
- **Surface contrast:** built-up mean LST minus the mean for trees, shrubland, grassland and cropland within the same ROI. Water is excluded from that reference. Reports include both sample sizes and return null if either group is absent.
- **Statistics:** Gi* includes each focal observation, uses projected distances, two-sided normal-approximation p-values and Benjamini–Hochberg adjusted q-values. Sparse and constant samples are handled explicitly.
- **Model validation:** hold out spatial blocks before fitting; use grouped folds inside training data for tuning. Repeated years in a location stay in the same split. Evaluation never refits the model. Coordinates, years, observation counts and target-derived labels are excluded from predictors.
- **Consistency:** retain year labels, all WorldCover classes and circular aspect components. Each map uses that year's raster layers, points and cluster colors.

The original warning came from [eo-workflow-lint](https://github.com/orbseekr-labs/eo-workflow-lint). Google confirms the negative-input masking behavior in its [normalizedDifference documentation](https://developers.google.com/earth-engine/apidocs/ee-image-normalizeddifference).

## Outputs

Use a new output directory per run; files with the same names are overwritten.

- `summary.json`: run status, configuration, dependency versions, date intervals, annual contrasts and sample counts, diagnostics, model metrics and errors.
- `YYYY_samples.csv` and `samples_all_years.csv`: sampled variables, year, coordinates, observation counts and analysis labels.
- `model_split.csv`: spatial block IDs and train/test assignments.
- `YYYY_uhi_zones.png`, `YYYY_hot_spots.png`: annual exploratory results.
- `CITY_YYYY_interactive_map.html`: annual map. Internet access is required; Earth Engine tile URLs expire and may need regeneration.
- Distribution, correlation, scatter and model-evaluation PNGs; optional SHAP figures.
- `uhi_analysis.log`: run messages.

A partial or failed run returns a nonzero exit code. Completed annual samples are retained when later plotting or modeling fails. Check `summary.json` instead of assuming that existing files all came from a complete run.

## Interpretation and limitations

The negative-input fix changes which observations are eligible, not their scientific validity. Negative reflectance and large ratios need review; the epsilon is a numerical guard, not a calibrated physical threshold. Annual means can remain sensitive to near-zero sums.

The optional diagnostic compares the new eligible mask with the same mask plus a nonnegative-input requirement. It isolates the negative-input condition under the **new QA and denominator policy**; it is not a comparison of every change against the original workflow. It sums valid scene-pixel observations on the requested grid. It is a coarse-grid estimate, not a unique-pixel count, area estimate, or measured change in UHI intensity.

MODIS output is an unweighted mean of accepted 8-day composites, not a day-weighted annual mean. Composite QA describes the majority quality of contributing daily observations. Landsat means are based on available clear observations. Neither guarantees unbiased seasonal coverage; the two sensors also have different observation times. Count bands describe valid composites/scenes per source pixel; spatial aggregation makes them fractional. Sampling requires all analysis bands to be valid, so the shared coverage mask can affect the population used for the surface contrast. No minimum observation-count threshold is imposed.

WorldCover is a static 2020/2021 reference, not contemporaneous land cover for each analysis year. Dominant-class labeling loses subpixel mixtures. Aggregation uses the available unmasked source pixels and does not require a minimum valid-area fraction. For Landsat composites crossing scene projections, the first scene's grid is the native working grid. Coarse aggregation over large or high-latitude regions can hit Earth Engine pixel/memory limits.

A city bounding box may lack a suitable rural reference. The within-ROI vegetated contrast is not a matched urban-versus-surrounding-rural experimental design; terrain, climate and irrigation can confound it. Temperature clusters are relative within each year and their thresholds may differ across years.

Hotspots are exploratory and sensitive to sample density and neighbor choice; normal approximation and BH do not remove all effects of spatial dependence. Spatial blocks reduce train/test proximity but impose no buffer between adjacent blocks. Block size needs sensitivity analysis. A pooled multi-year spatial holdout is not temporal forecasting validation. Small ROIs may have too few blocks for modeling.

## Verification

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover numerical preprocessing, masks, actual Earth Engine API graph construction, dates, sparse inputs, model split separation and an offline synthetic end-to-end run. They require no credentials. They do not validate server execution or measured satellite outcomes.

See [the review notes](docs/REVIEW.md) for the audit and sources, and [the follow-up draft](docs/FOLLOW_UP_DRAFT.md) for a message to Takahito. [Original screenshots](docs/LEGACY_SHOWCASE.md) are retained as historical outputs and have not been regenerated.

MIT license — see [LICENSE](LICENSE).

