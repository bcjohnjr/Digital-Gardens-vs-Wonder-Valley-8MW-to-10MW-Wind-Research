# Wind data, independent ridge-site allocation, and 16-year replay

This folder documents the provenance and limits of the British Columbia model, and supplies the code for five independently selectable wind-site capacities.

1. `INPUT/GWA_BC_5_RIDGE_RESULTS.csv` and audit: Global Wind Atlas 100-m modelled mean speeds at five BC named or proxy coordinates. Countrywide raster is not supplied.
2. `INPUT/BC_monthly_observed_wind_generation_2016_2026.csv`: Statistics Canada observed BC **provincial monthly** wind production. Year 2020 remains in the canonical input and is omitted ONLY in a sensitivity.
3. `INPUT/aeso_wind_hourly_utc_2015_2025.csv`: AESO Alberta hourly reported fleet data, not directly measured at BC proposal sites. Check third-party redistribution terms before rehosting this source.
4. `optimize_sites_cap_cf.py`: bounded search over five wind capacities, solar, battery and HVO limit, with a site-modelled 40% CF screen (and optional alternate caps). Default high-resolution search is computationally demanding. Use `--help` for options.
5. `build_site_free_outputs.py`: constructs selected model outputs, 16-year dispatch stress replays, BC observed-month comparison and source audits. Use only after the appropriate search-result JSON exists.
6. `figure_site_free.py`: updated figures for independent wind allocations and identical-layout wind cap stresses.

Stored source data and model outputs are never evidence of wind-mast observations, transmission deliverability, turbine permits or global numerical optimality. Independent 35%-CF optimisation is a different installed portfolio; the **same-design** 35% cap stress is a more restrictive reliability check.
