#!/usr/bin/env python3
"""Digital Gardens historical ERA5 downloader — no external Python packages needed.

Windows / PowerShell:
    python digitalgardens.py --years 2025             # fast one-year test (17 places)
    python digitalgardens.py                          # full 2010–2025 download
    python digitalgardens.py --years 2020:2025         # shorter historical range
    python digitalgardens.py --dry-run                 # inspect sites without downloading

Downloads UTC hourly ERA5 reanalysis via Open-Meteo Historical Weather API.
Coordinates represent town centers / municipality proxies, NOT turbine or data-center
sites. Precisely located wind/PV development needs separate project coordinates and
resource validation.  Water flow for hydro requires river-gauge records separately.

The downloader writes each successful year to disk immediately. Restarting it resumes
from the cached annual JSON files, and a complete run creates one ZIP for upload.
Missing optional variables (e.g., snow depth) are recorded as missing values,
not silently converted to zero or treated as a fatal download failure.
Attribution: ECMWF/Copernicus Climate Change Service ERA5; Open-Meteo.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

API = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = [
    "wind_speed_100m", "wind_direction_100m", "shortwave_radiation",
    "direct_normal_irradiance", "temperature_2m", "cloud_cover",
    "precipitation", "snow_depth", "surface_pressure",
]
# These three fields are mandatory for the energy model. Optional weather
# fields may legitimately be unavailable for a model/grid/year and stay blank.
REQUIRED_VARIABLES = frozenset(("wind_speed_100m", "shortwave_radiation", "temperature_2m"))
OPTIONAL_VARIABLES = tuple(v for v in VARIABLES if v not in REQUIRED_VARIABLES)

# First 14 locations: coordinates logged from the user's previous Open-Meteo
# geocoding run (as town-centre proxies). Last 3: official BC geographical
# names: https://apps.gov.bc.ca/pub/bcgnws/names/{752,38229,5536}.html
# They are NOT coordinates of proposed generation installations.
SITES = [
    ("Lillooet",       50.68560, -121.94200),
    ("Cache Creek",    50.81011, -121.32460),
    ("Kamloops",       50.66648, -120.31920),
    ("Chase",          50.81650, -119.68571),
    ("Salmon Arm",     50.69981, -119.26907),
    ("Vernon",         50.26729, -119.27337),
    ("Kelowna",        49.88307, -119.48568),
    ("West Kelowna",   49.86250, -119.58333),
    ("Penticton",      49.48062, -119.58584),
    ("Oliver",         49.18306, -119.55240),
    ("Osoyoos",        49.03200, -119.46800),
    ("Hope",           49.38299, -121.44144),
    ("Agassiz / Kent", 49.23298, -121.76926),
    ("Chilliwack",     49.16638, -121.95257),
    ("Abbotsford",     49.05194, -122.32917),
    ("Mission",        49.15889, -122.28389),
    ("Maple Ridge",    49.21972, -122.59861),
]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    temp.replace(path)


def api_url(latitude: float, longitude: float, year: int) -> str:
    params = {
        "latitude": f"{latitude:.5f}", "longitude": f"{longitude:.5f}",
        "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
        "hourly": ",".join(VARIABLES),
        "models": "era5",           # never use blended best_match
        "timezone": "GMT",          # UTC, no daylight savings
        "wind_speed_unit": "ms",   # wind speeds in m/s
    }
    return API + "?" + urllib.parse.urlencode(params)


def fetch_json(url: str, attempts: int = 6) -> dict:
    for n in range(attempts):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "DigitalGardensHistoricalWeather/2.0 (research)",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=180) as r:
                value = json.load(r)
            if not isinstance(value, dict):
                raise ValueError("API returned a non-object response")
            if value.get("error"):
                raise ValueError("Weather API rejected request: " + str(value.get("reason", value)))
            return value
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                try:
                    detail = e.read(500).decode("utf-8", errors="replace")
                except Exception:
                    detail = str(e)
                raise RuntimeError(f"API HTTP {e.code}: {detail}") from e
            message = f"HTTP {e.code} (rate-limited or server unavailable)"
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            message = str(e)
        if n == attempts - 1:
            raise RuntimeError(f"Weather request failed after {attempts} attempts: {message}\n{url}")
        pause = min(4 * 2 ** n, 60)
        print(f"  Temporary network error: {message}; retrying in {pause}s...", flush=True)
        time.sleep(pause)
    raise RuntimeError("Unreachable retry state")


def expected_hours(year: int) -> int:
    return 8784 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 8760


def validate(data: dict, year: int, site: str) -> dict:
    """Validate the essential timeseries; leave unavailable optional values missing.

    This intentionally returns the unmodified API hourly dictionary: cached raw
    JSON remains faithful to the provider, including null optional variables.
    """
    hourly = data.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError(f"No hourly data found for {site} {year}")
    times = hourly.get("time")
    count = expected_hours(year)
    if not isinstance(times, list) or len(times) != count:
        raise ValueError(f"{site} {year}: received {len(times) if isinstance(times,list) else 'no'} hours; expected {count}")
    if times[0] != f"{year}-01-01T00:00" or times[-1] != f"{year}-12-31T23:00":
        raise ValueError(f"{site} {year}: unexpected UTC timestamp bounds {times[0]} / {times[-1]}")
    if data.get("utc_offset_seconds", 0) != 0:
        raise ValueError(f"{site} {year}: returned timestamps are not UTC")
    for k in VARIABLES:
        col = hourly.get(k)
        valid_shape = isinstance(col, list) and len(col) == count
        if k in REQUIRED_VARIABLES:
            if not valid_shape:
                raise ValueError(f"{site} {year}: missing/short essential '{k}' column")
            if any(v is None for v in col):
                raise ValueError(f"{site} {year}: essential '{k}' has missing values")
        # Optional variables can have missing/short columns or all-null series.
        # They stay blank in the CSV and their data completeness is logged.
    return hourly


def missing_coverage(hourly: dict, year: int) -> dict[str, int]:
    """Number of missing hours per variable; malformed optionals count as 100% missing."""
    count = expected_hours(year)
    return {
        k: (sum(v is None for v in col) if isinstance(col, list) and len(col) == count else count)
        for k in VARIABLES
        for col in (hourly.get(k),)
    }


def read_overrides(path: Path) -> dict[str, tuple[float, float]]:
    overrides = {}
    if path.exists():
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not {"community", "latitude", "longitude"}.issubset(reader.fieldnames or []):
                raise ValueError(f"{path}: expected CSV columns community,latitude,longitude")
            for r in reader:
                name = r["community"].strip()
                if name not in {s[0] for s in SITES}:
                    raise ValueError(f"Unknown community in {path}: {name}")
                lat, lon = float(r["latitude"]), float(r["longitude"])
                if not (48 <= lat <= 61 and -140 <= lon <= -114):
                    raise ValueError(f"Coordinates for {name} are outside reasonable B.C. bounds")
                overrides[name] = (lat, lon)
    return overrides


def load_annual(raw: Path, year: int, name: str) -> dict | None:
    if not raw.exists():
        return None
    try:
        with gzip.open(raw, "rt", encoding="utf-8") as f:
            cached = json.load(f)
        validate(cached, year, name)
        return cached
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"  Ignoring corrupt or incomplete cached file {raw}: {e}", flush=True)
        raw.rename(raw.with_name(raw.name + ".invalid"))
        return None


def save_annual(raw: Path, data: dict) -> None:
    raw.parent.mkdir(parents=True, exist_ok=True)
    temp = raw.with_name(raw.name + ".part")
    with gzip.open(temp, "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump(data, f, separators=(",", ":"))
    temp.replace(raw)


def make_zip(folder: Path, first: int, last: int) -> Path:
    archive = folder.parent / f"Digital_Gardens_ERA5_{first}_{last}_DATA.zip"
    tmp = archive.with_suffix(".zip.part")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5, allowZip64=True) as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file() and not f.name.endswith((".part", ".invalid")):
                z.write(f, f.relative_to(folder.parent))
    tmp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        error = z.testzip()
        if error:
            raise RuntimeError(f"ZIP integrity error in {error}")
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", default="2010:2025", help="Year or inclusive range e.g. 2025 or 2010:2025")
    parser.add_argument("--output", default="Digital_Gardens_ERA5_historical", help="Destination folder")
    parser.add_argument("--site", action="append", help="Only download this community (repeat --site for more)")
    parser.add_argument("--dry-run", action="store_true", help="Show locations without network calls")
    args = parser.parse_args()
    try:
        bits = args.years.split(":")
        if len(bits) not in (1, 2):
            raise ValueError
        first, last = int(bits[0]), int(bits[-1])
    except ValueError:
        parser.error("--years must be YYYY or YYYY:YYYY")
    if not (1940 <= first <= last <= dt.datetime.now(dt.timezone.utc).year):
        parser.error("Invalid year range")

    out = Path(args.output).resolve()
    override_path = Path("site_coordinates_override.csv")
    overrides = read_overrides(override_path)
    locations = []
    for name, lat, lon in SITES:
        modified = name in overrides
        if modified:
            lat, lon = overrides[name]
        locations.append({
            "community": name, "geocoded_name": name, "latitude": lat, "longitude": lon,
            "location_kind": "user supplied location override" if modified else "town/municipal proxy, not project-site coordinate",
            "coordinate_source": "site_coordinates_override.csv" if modified else (
                "BC Geographical Names" if name in {"Abbotsford", "Mission", "Maple Ridge"}
                else "Open-Meteo geocoder response captured in user's earlier run"),
        })
    selected = [r for r in locations if args.site is None or r["community"] in args.site]
    if not selected or (args.site and set(args.site) - {r["community"] for r in locations}):
        parser.error("Unknown or unselected --site; use full community names from --dry-run")
    print(f"ERA5 hourly reanalysis, {first}–{last}, {len(selected)} location(s)", flush=True)
    for row in selected:
        print(f"  {row['community']}: {row['latitude']:.5f}, {row['longitude']:.5f}", flush=True)
    if args.dry_run:
        print("DRY RUN: no requests or files created.")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    rawdir = out / "raw_open_meteo_era5"
    rawdir.mkdir(exist_ok=True)
    # Keep a manifest as the run proceeds; even failed downloads leave useful state.
    atomic_json(out / "locations.json", locations)
    manifest = {
        "dataset": "Hourly ERA5 atmospheric reanalysis (not synthetic weather)",
        "provider": "Open-Meteo historical weather API", "model": "era5 (explicit)",
        "underlying_provider": "ECMWF / Copernicus Climate Change Service",
        "era5_doi": "https://doi.org/10.24381/cds.adbb2d47",
        "api_documentation": "https://open-meteo.com/en/docs/historical-weather-api",
        "retrieval_started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "first_year": first, "last_year": last, "timezone": "UTC", "variables": VARIABLES,
        "required_variables": sorted(REQUIRED_VARIABLES), "optional_variables": list(OPTIONAL_VARIABLES),
        "locations": selected, "downloads": [], "status": "running",
        "limitations": [
            "Coordinates are community proxies; replace with actual proposed generation coordinates.",
            "ERA5 is reanalysis based on observations, not measured mast data at these sites.",
            "ERA5 resolution is about 0.25 degrees; neighboring communities may share weather grids.",
            "Optional variables can be blank; missing never means zero and is logged per site-year.",
            "Wind turbine and solar generation require device-specific conversion and losses.",
            "River turbines require measured discharge and environmental constraints separately.",
        ],
    }
    manifest_path = out / "provenance_manifest.json"
    atomic_json(manifest_path, manifest)
    print(f"\nSaving files immediately into: {out}", flush=True)
    total = len(selected) * (last - first + 1)
    successful = 0
    reported_optional = set()
    try:
        for place in selected:
            name, lat, lon = place["community"], place["latitude"], place["longitude"]
            site_slug = slug(name)
            raw_site = rawdir / site_slug
            raw_site.mkdir(exist_ok=True)
            combined = out / f"{site_slug}_hourly_era5.csv.gz"
            tmp_csv = combined.with_name(combined.name + ".part")
            with gzip.open(tmp_csv, "wt", encoding="utf-8", newline="", compresslevel=6) as f:
                writer = csv.writer(f)
                writer.writerow(["time_utc", "community", "requested_latitude", "requested_longitude",
                                 "era5_grid_latitude", "era5_grid_longitude"] + VARIABLES)
                for year in range(first, last + 1):
                    raw = raw_site / f"{year}.json.gz"
                    url = api_url(lat, lon, year)
                    value = load_annual(raw, year, name)
                    status = "cached" if value is not None else "downloaded"
                    if value is None:
                        print(f"[{successful+1}/{total}] Downloading {name} {year}...", flush=True)
                        value = fetch_json(url)
                        validate(value, year, name)
                        save_annual(raw, value)
                        time.sleep(0.15)
                    hours = validate(value, year, name)
                    rows = len(hours["time"])
                    missing = missing_coverage(hours, year)
                    missing_optional = [v for v in OPTIONAL_VARIABLES if missing[v] > 0]
                    for variable in missing_optional:
                        if (name, variable) not in reported_optional:
                            print(f"  NOTE: {name} optional '{variable}' has missing data "
                                  f"({missing[variable]:,}/{rows:,} hours in {year}); "
                                  "leaving values blank, continuing.", flush=True)
                            reported_optional.add((name, variable))
                    # This also handles entirely absent optional columns without
                    # pretending missing snow depth or cloud data are zero.
                    columns = [hours.get(v) for v in VARIABLES]
                    for i, timestamp in enumerate(hours["time"]):
                        writer.writerow([timestamp, name, lat, lon,
                                         value.get("latitude"), value.get("longitude")]
                                        + [col[i] if isinstance(col, list) and len(col) == rows
                                           else None for col in columns])
                    manifest["downloads"].append({
                        "site": name, "year": year, "request_url": url, "source": status,
                        "raw_file": str(raw.relative_to(out)).replace(os.sep, "/"),
                        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                        "rows": rows, "units": value.get("hourly_units", {}),
                        "missing_hours_by_variable": missing,
                        "optional_variables_with_gaps": missing_optional,
                    })
                    successful += 1
                    manifest["completed_years"] = successful
                    atomic_json(manifest_path, manifest)
                    print(f"[{successful}/{total}] SAVED {name} {year}: {rows:,} UTC hours ({status})", flush=True)
            tmp_csv.replace(combined)
            print(f"  COMPLETED {name}: {combined.name}", flush=True)
    except (ValueError, RuntimeError, OSError) as e:
        manifest["status"] = "interrupted; run same command to resume from cached years"
        manifest["error"] = str(e)
        atomic_json(manifest_path, manifest)
        print("\nDOWNLOAD STOPPED. Any completed annual files remain safely saved.", file=sys.stderr)
        print("Run the SAME command again to resume from saved years.", file=sys.stderr)
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    manifest["status"] = "complete"
    manifest["retrieval_completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_json(manifest_path, manifest)
    archive = make_zip(out, first, last)
    print(f"\nSUCCESS — {successful} site-year files, {sum(r['rows'] for r in manifest['downloads']):,} hourly records", flush=True)
    print(f"ZIP FOR CHATGPT: {archive}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
