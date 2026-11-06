#!/usr/bin/env python3
"""Extract 100 m mean wind speed at original 5 BC ERA5 proxy coordinates.

INPUT is a user-downloaded Global Wind Atlas 4.0 100-m MEAN WIND SPEED GeoTIFF
from https://globalwindatlas.info/download/gis-files (or GIS area export).
There is no claimed stable or bulk-download API endpoint here; GWA download terms
must be respected. The five coordinates are screening locations, not turbine sites.
"""
import argparse, csv, json, hashlib
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tif', required=True, help='Local GWA 4.0 100-m mean-speed GeoTIFF')
    p.add_argument('--out', default=str(HERE/'INPUT/gwa_100m_means.csv'))
    p.add_argument('--source-url', default='https://globalwindatlas.info/download/gis-files')
    a = p.parse_args()
    try:
        import rasterio
        from rasterio.warp import transform
    except ImportError:
        raise SystemExit('Install rasterio: python -m pip install rasterio')
    sites = list(csv.DictReader((HERE/'ridge_sites.csv').open()))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    tif = Path(a.tif)
    if not tif.is_file(): raise SystemExit('Missing GeoTIFF: '+str(tif))
    rows = []
    with rasterio.open(tif) as ds:
        if ds.count < 1: raise SystemExit('Raster has no bands')
        if ds.crs is None: raise SystemExit('Raster missing coordinate reference system')
        for site in sites:
            xs, ys = transform('EPSG:4326', ds.crs, [float(site['longitude'])], [float(site['latitude'])])
            val = next(ds.sample([(xs[0], ys[0])], masked=True))[0]
            if getattr(val, 'mask', False): raise SystemExit('No GWA data at '+site['site_key']+'; export area covering all points')
            v = float(val)
            if not (0.5 < v < 35):
                raise SystemExit(f'GWA speed {v:g} at {site["site_key"]} outside 0.5-35 m/s; inspect units, scale, or nodata')
            rows.append([site['site_key'],f'{v:.5f}','GWA 4.0',a.source_url,datetime.now(timezone.utc).isoformat(),
                         f'GeoTIFF={tif.name}; band1 CRS={ds.crs}; spatial sample; not a mast observation'])
    with out.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['site_key','gwa_mean_100m_ms','gwa_version','source_url','retrieved_utc','notes']);w.writerows(rows)
    prov={'raster_path':str(tif.resolve()),'raster_sha256':hashlib.sha256(tif.read_bytes()).hexdigest(),
          'point_count':len(rows),'source_url':a.source_url,'disclaimer':'atlas mean is not independent hourly site wind measurement'}
    out.with_suffix('.provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    print('Wrote',out,'with',len(rows),'GWA 100m mean wind speeds')

if __name__ == '__main__': main()
