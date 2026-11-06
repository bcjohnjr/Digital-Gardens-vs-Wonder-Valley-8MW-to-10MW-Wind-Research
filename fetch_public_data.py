#!/usr/bin/env python3
"""Download public comparison evidence without inventing wind speeds or observations.

Official sources:
- Statistics Canada monthly electricity generation, table 25-10-0015-01.
- European Commission JRC PVGIS v5.3 hourly solar API.
- AESO Historical Generation Data (CSD), downloaded using an official URL supplied
  from the live AESO portal, which may refer to its third-party file host.

This script never edits the main manuscript or modelling inputs.
"""
import argparse, csv, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
OUT = HERE / 'INPUT'
STATS_URL = 'https://www150.statcan.gc.ca/n1/tbl/csv/25100015-eng.zip'
PVGIS_API = 'https://re.jrc.ec.europa.eu/api/v5_3/seriescalc'
AESO_LANDING = 'https://www.aeso.ca/market/market-and-system-reporting/data-requests/historical-generation-data/'


def metadata(path, url, note=''):
    file = Path(path)
    info = {'source_url': url, 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
            'file': file.name, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
            'bytes': file.stat().st_size, 'notes': note}
    file.with_suffix(file.suffix + '.source.json').write_text(json.dumps(info, indent=2) + '\n')
    return info


def download(url, path, timeout=120):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    req = Request(url, headers={'User-Agent': 'ResearchProvenanceAudit/1.0 (+published source URLs)', 'Accept': '*/*'})
    with urlopen(req, timeout=timeout) as response, tmp.open('wb') as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk: break
            output.write(chunk)
    tmp.replace(path)
    return metadata(path, url)


def pvgis(args):
    nodes = list(csv.DictReader((HERE / 'solar_nodes.csv').open()))
    start, end = args.startyear, args.endyear
    if not (2005 <= start <= end <= 2025):
        raise SystemExit('Invalid years. PVGIS availability is data-source and region dependent.')
    target = OUT / 'pvgis'; target.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for i, node in enumerate(nodes):
        params = {'lat': node['latitude'], 'lon': node['longitude'], 'startyear': start,
                  'endyear': end, 'pvcalculation': 1, 'peakpower': 1, 'loss': args.loss,
                  'angle': args.tilt, 'aspect': args.aspect, 'outputformat': 'json'}
        url = PVGIS_API + '?' + urlencode(params)
        key = ''.join(c if c.isalnum() else '_' for c in node['community'].lower()).strip('_')
        dest = target / (key + '.json')
        try:
            source = download(url, dest, timeout=90)
            payload = json.loads(dest.read_text())
            hours = payload.get('outputs', {}).get('hourly', [])
            if not hours or 'P' not in hours[0]:
                raise ValueError('PVGIS returned no hourly P (W per 1 kWp) values')
            valid = sum(1 for r in hours if isinstance(r.get('P'), (int, float)))
            outcomes.append([node['community'], 'OK', valid, dest.name, source['sha256'], ''])
            print('OK', node['community'], valid, 'hourly records', flush=True)
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError) as e:
            # Never substitute ERA5 or a forecast for missing independent satellite data.
            dest.unlink(missing_ok=True)
            outcomes.append([node['community'], 'FAILED', 0, '', '', str(e)[:300]])
            print('FAILED', node['community'], str(e)[:160], flush=True)
        if i < len(nodes) - 1 and args.delay: time.sleep(args.delay)
    with (target / 'fetch_status.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['community','status','hourly_rows','filename','sha256','error']); writer.writerows(outcomes)
    print('PVGIS status:', sum(r[1] == 'OK' for r in outcomes), '/', len(nodes))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    sub.add_parser('statcan', help='Download complete official Statistics Canada monthly generation cube')
    s = sub.add_parser('pvgis', help='Attempt PVGIS 5.3 hourly PV benchmark at all 17 node proxies')
    s.add_argument('--startyear', type=int, default=2015)
    s.add_argument('--endyear', type=int, default=2020)
    s.add_argument('--loss', type=float, default=14, help='Percent system losses, not a measured value')
    s.add_argument('--tilt', type=float, default=30)
    s.add_argument('--aspect', type=float, default=0)
    s.add_argument('--delay', type=float, default=1.0)
    a = sub.add_parser('aeso', help='Download the AESO CSD archive from URL shown in official portal')
    a.add_argument('--url', required=True, help='Paste the actual 1-hour data file URL from the AESO portal')
    a.add_argument('--name', default='aeso_hourly_original.csv', help='Destination filename; .zip supported')
    args = ap.parse_args(); OUT.mkdir(exist_ok=True)
    if args.action == 'statcan':
        print(download(STATS_URL, OUT / '25100015-eng.zip'))
    elif args.action == 'aeso':
        if not args.url.startswith('https://'):
            raise SystemExit('Provide an HTTPS URL copied from ' + AESO_LANDING)
        if '/' in args.name or '\\' in args.name:
            raise SystemExit('--name must be a filename, not a path')
        print(download(args.url, OUT / args.name))
    else:
        pvgis(args)

if __name__ == '__main__': main()
