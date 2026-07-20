# ERA5 streaming data card

## Source and retained scope

- Source: ECMWF ERA5 redistributed through the public WeatherBench2 cloud store.
- Endpoint: `gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr`.
- Cadence and grid: 6 hours, 64 longitudes × 32 latitudes, conservative regridding.
- Study interval: 1959–2022 inclusive.
- Retained data: selected point cells only; the global archive is never downloaded or materialized.

The repository preserves three experiment stages. The pilot retained 16 climate anchors. Locked v2 retained 64 fitting, 16 development, and 32 confirmation cells while excluding every pilot cell. Fresh v3 adds 32 cells that exclude every pilot and v2 coordinate. Large source-derived arrays are local, checksummed artifacts and are excluded from Git.

## Variables and transformations

| Source field | Retained unit | Modeling transform |
|---|---:|---|
| `2m_temperature` | °C | Fourier seasonal mean; robust scale |
| `mean_sea_level_pressure` | hPa | Fourier seasonal mean; robust scale |
| `10m_u_component_of_wind` | m/s | Fourier seasonal mean; robust scale |
| `10m_v_component_of_wind` | m/s | Fourier seasonal mean; robust scale |
| `total_precipitation_6hr` | mm/6 h | nonnegative clip, `log1p`, seasonal mean, robust scale |

Wind speed is derived as `sqrt(u² + v²)`. The four prediction targets are temperature, pressure, wind speed, and six-hour precipitation at 6, 24, 72, and 168 hours.

## Split-specific information policy

- Model fitting: 1959–1994 at the 64 fitting cells.
- Hyperparameter calibration: 1995–2004 at fitting cells.
- Development selection: 2005–2016 at 16 development cells.
- Fresh confirmation: 2017–2022 at 32 v3 cells.
- Every admitted origin has its entire 56-step context and all targets inside its split.
- Online memory inserts an outcome only after the 168-hour target has matured.

V2 interpolates normalization statistics from fitting cells and uses no held-out-cell values for those statistics. That strict spatial setting produced large climate bias and is retained as a negative persistence gate. V3 instead permits each fresh cell's 1959–1994 values to estimate its own seasonal normals and scales; no 2017–2022 label contributes. V3 is therefore historical-normal adaptation at a new location, not strict zero-shot spatial transfer.

## Intended and prohibited uses

Intended use is reproducible research on bounded retrieval memory for multivariate geophysical time series. The cache is not suitable for operational weather prediction, hazard warnings, local decision support, or claims about a general time-series foundation model. Cell identifiers are coarse grid coordinates, not observations from people or private infrastructure.

## Known limitations

- ERA5 is reanalysis, not direct observation.
- Coarse conservative cells blur topography, coastlines, fronts, and extremes.
- Pointwise models omit spatial fields, physical conservation, and data assimilation.
- The sampled cells and 2017–2022 interval do not establish universal geographic or temporal generality.
- Historical-normal adaptation requires decades of prior data at a new cell.
- Source licensing and attribution remain governed by ECMWF/Copernicus and WeatherBench2.

## References

- Hersbach et al. (2020), *The ERA5 global reanalysis*. DOI: `10.1002/qj.3803`.
- Rasp et al. (2024), *WeatherBench 2: A benchmark for the next generation of data-driven global weather models*. DOI: `10.1029/2023MS004019`.
- Copernicus Climate Data Store ERA5 single-level dataset. DOI: `10.24381/cds.adbb2d47`.
