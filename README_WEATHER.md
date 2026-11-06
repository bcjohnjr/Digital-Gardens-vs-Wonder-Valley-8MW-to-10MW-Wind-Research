# Hourly ERA5 reanalysis resource data, 17 British Columbia location proxies

17 compressed CSV files — each is the original complete user-downloaded hourly dataset, 2010-01-01 00:00 through 2025-12-31 23:00 UTC (140,256 timestamped hours per site). Town-centre proxies are **not** wind farm, solar farm or data-centre property measurements. ERA5 is observationally constrained atmospheric reanalysis.

The files include wind speed/direction at 100 m, incoming shortwave radiation, direct normal irradiance, air temperature at 2 m, cloud cover, precipitation, optional snow depth, and surface pressure. Snow depth is blank where not returned by the API; required wind, irradiation and temperature values are present. `provenance_manifest.json` lists the original API queries and source provenance. `locations.json` lists municipal coordinates; `data_download_script.py` is the downloader. The original raw yearly JSON responses referenced by the manifest are not in this submission; only the consolidated CSV data were uploaded.

The site series have leap days in 2012, 2016, 2020 and 2024. The reproducible cross-year energy-dispatch analysis excludes those four leap days within each comparable annual 8,760-hour run; they remain intact in this repository.

The file hashes of each embedded CSV are also listed in the top-level SHA256SUMS.txt. Attribution: Open-Meteo historical API / ECMWF Copernicus ERA5 (`models=era5`), retrieval October 2026. Original provider data are governed by their respective reuse/licensing terms.
