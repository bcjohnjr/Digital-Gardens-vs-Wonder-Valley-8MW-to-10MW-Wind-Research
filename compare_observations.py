#!/usr/bin/env python3
"""Compare 5-site ERA5/GWA-wind profiles with supplied observed electricity generation.

Requires actual, independently sourced measured data. Provincial monthly wind
or Alberta hourly wind do not prove wind resources at any five BC proposed sites.
A capacity factor requires correct contemporaneous commissioned capacity data.
"""
import argparse, csv, io, json, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from replay_wind import import_atlas
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'MODEL_CORE'))
from dispatch_reference import read_inputs, wind_cf, PICK
WEIGHTS=np.array([.35,.25,.18,.12,.10])


def wind_profiles(gwa):
    dates,solar,wind=read_inputs(ROOT/'DATA',ROOT/'DATA/RIDGE_WIND_ERA5_6_POINTS_2010_2025.zip')
    atlas=import_atlas(Path(gwa), PICK)
    series={'raw_era5':np.average(np.stack([wind_cf(w,1) for w in wind]),axis=0,weights=WEIGHTS)}
    if atlas:
        mult=[atlas[k][0]/float(w.mean()) for k,w in zip(PICK,wind)]
        series['gwa_mean_corrected_hourly_unvalidated']=np.average(np.stack([wind_cf(w,m) for w,m in zip(wind,mult)]),axis=0,weights=WEIGHTS)
    return pd.DataFrame({'time_utc':pd.to_datetime(dates,utc=True),**series})


def longest_lulls(arr, threshold=.10):
    a=np.asarray(arr,float)
    binary=np.isfinite(a)&(a<threshold)
    stretches=[];current=0
    for flag in binary:
        if flag: current+=1
        elif current: stretches.append(current); current=0
    if current:stretches.append(current)
    return {'hours_under_10pct':int(np.sum(binary)),
            'frac_hours_under_10pct':float(binary.mean()),
            'longest_lull_hours':max(stretches,default=0),
            'lull_count_ge_24h':sum(v>=24 for v in stretches),
            'hourly_ramp_abs_p95':float(np.nanquantile(np.abs(np.diff(a)),.95)),
            'hourly_ramp_abs_p99':float(np.nanquantile(np.abs(np.diff(a)),.99)),
            'cf_p10':float(np.nanquantile(a,.1)), 'cf_p50':float(np.nanmedian(a)),
            'cf_p90':float(np.nanquantile(a,.9))}


def statcan(args, modeled):
    fp=Path(args.statcan)
    if fp.suffix.lower()=='.zip':
        with zipfile.ZipFile(fp) as z:
            names=[n for n in z.namelist() if n.endswith('.csv') and 'MetaData' not in n and 'Meta' not in n]
            if len(names)!=1: raise SystemExit('Specify extracted actual data CSV; archive CSV candidates '+repr(names))
            df=pd.read_csv(io.BytesIO(z.read(names[0])),low_memory=False)
    else:df=pd.read_csv(fp,low_memory=False)
    print('Statistics Canada fields:',list(df.columns))
    if args.inspect_statcan:
        for c in df.columns:
            if c not in ['VALUE','REF_DATE','VECTOR','COORDINATE','DGUID','DECIMALS','TERMINATED']:
                vals=[str(v) for v in df[c].dropna().unique() if any(s in str(v).lower() for s in ('wind','electric','producer','british columbia','total'))]
                print(c,':',vals[:40])
        return
    for key in ['REF_DATE','GEO','VALUE']:
        if key not in df:raise SystemExit('Missing required StatsCan column '+key)
    df=df[df.GEO.astype(str).str.fullmatch('British Columbia',case=False,na=False)].copy()
    # The data cube has multiple producer-type series; user must choose an unambiguous one.
    dims=[c for c in df.columns if c not in ['REF_DATE','GEO','DGUID','UOM','UOM_ID','SCALAR_FACTOR','SCALAR_ID','VECTOR','COORDINATE','VALUE','STATUS','SYMBOL','TERMINATED','DECIMALS']]
    windcols=[c for c in dims if df[c].astype(str).str.contains(r'wind',case=False).any()]
    if len(windcols)!=1:raise SystemExit('Wind dimension ambiguous. Use --inspect-statcan and select a CSV series manually.')
    wc=windcols[0]
    df=df[df[wc].astype(str).str.fullmatch(args.statcan_wind_label,case=False,na=False)]
    for selection in args.statcan_filter:
        if '=' not in selection:raise SystemExit('Use --statcan-filter "Dimension=exact value"')
        col,val=selection.split('=',1)
        if col not in df.columns:raise SystemExit('Unknown StatsCan dimension '+col)
        df=df[df[col].astype(str).eq(val)]
    if df.empty:raise SystemExit('No StatsCan BC wind rows after filtering. Inspect the source labels.')
    if df.duplicated('REF_DATE').any():
        print('Multiple records per month. Filter dimensions to one unique province-wide wind series. Values:')
        for c in dims:print(c,df[c].dropna().unique()[:20])
        raise SystemExit('Refusing to sum producer categories or double-count wind generation')
    x=df[['REF_DATE','VALUE']].copy();x['month']=pd.to_datetime(x.REF_DATE.astype(str).str.slice(0,7)+'-01')
    x['observed_mwh']=pd.to_numeric(x.VALUE,errors='coerce');x=x.dropna(subset=['observed_mwh'])
    x['days_in_month']=x.month.dt.days_in_month
    x['observed_mean_mw']=x.observed_mwh/(x.days_in_month*24) # units must be MWh: check before interpretation
    x['observed_normalized']=x.observed_mean_mw/x.observed_mean_mw.mean()
    m=modeled.copy();m['month']=m.time_utc.dt.to_period('M').dt.to_timestamp()
    mod=m.groupby('month').agg({col:'mean' for col in modeled.columns if col!='time_utc'}).reset_index()
    match=x.merge(mod,on='month',how='inner')
    summary={'source':'Statistics Canada 25-10-0015-01','matched_months':len(match),
             'warning':'Mean monthly output is NOT per-site, nor an observed capacity factor without verified contemporaneous installed MW; verify StatsCan units.'}
    for c in modeled.columns[1:]:
        norm=match[c]/match[c].mean()
        summary[c+'_monthly_shape_correlation']=float(match.observed_normalized.corr(norm)) if len(match)>=3 else None
        summary[c+'_monthly_shape_rmse_normalized']=float(np.sqrt(np.mean((match.observed_normalized-norm)**2))) if len(match)>=3 else None
    match.to_csv(HERE/'OUTPUT/statcan_monthly_wind_comparison.csv',index=False)
    return summary


def hourly(args, modeled):
    obs=pd.read_csv(args.hourly)
    for col in [args.time_col,args.generation_col]:
        if col not in obs:raise SystemExit('Hourly observation CSV missing '+col+'; available: '+str(list(obs.columns)))
    obs['time_utc']=pd.to_datetime(obs[args.time_col],utc=True,errors='coerce')
    obs['generation_mw']=pd.to_numeric(obs[args.generation_col],errors='coerce')
    if args.capacity_col:
        if args.capacity_col not in obs:raise SystemExit('Missing capacity column: '+args.capacity_col)
        obs['capacity_mw']=pd.to_numeric(obs[args.capacity_col],errors='coerce')
    elif args.capacity_mw and args.capacity_mw>0:obs['capacity_mw']=args.capacity_mw
    else:raise SystemExit('A sourced, commissioned hourly installed MW capacity is mandatory (--capacity-col or --capacity-mw).')
    obs=obs.dropna(subset=['time_utc','generation_mw','capacity_mw']).sort_values('time_utc')
    if (obs.capacity_mw<=0).any():raise SystemExit('Nonpositive observed generating capacity')
    if obs.time_utc.duplicated().any():raise SystemExit('Duplicate UTC hours: source must be aggregated to fleet MW per hour')
    if ((obs.generation_mw<0)|(obs.generation_mw>obs.capacity_mw*1.05)).any():
        raise SystemExit('Generation outside 0..1.05*capacity; check units, metering and capacity commissioning dates')
    obs['observed_cf']=obs.generation_mw/obs.capacity_mw
    match=obs[['time_utc','observed_cf']].merge(modeled,on='time_utc',how='inner').sort_values('time_utc')
    if len(match)<24*30:raise SystemExit('Less than 30d overlapping modeled and observed hours')
    # Missing hours break runs; print gaps as information, do not mistake concatenated periods for continuous.
    intervals=match.time_utc.diff().dt.total_seconds().div(3600)
    gaps=int((intervals>1.01).sum()); print('Hourly overlap',len(match),'hours, gaps',gaps)
    summary={'hourly_observations':len(match),'gaps_in_utc_hours':gaps,
             'observation_source':args.source_description,'capacity_mw_definition':args.capacity_definition,
             'warning':'Observed Alberta fleet temporal shape cannot validate a distinct BC ridge location; cross-site exact-hour correlation is not meaningful.'}
    # Interpret distributions/ramp percentiles and contiguous lull statistics only on longest complete block.
    block=(intervals.fillna(1)>1.01).cumsum()
    largest=match.groupby(block).size().idxmax();cont=match[block==largest]
    summary['longest_contiguous_observed_span_h']=len(cont)
    for c in ['observed_cf']+list(modeled.columns[1:]):
        summary[c]=longest_lulls(cont[c].to_numpy())
    match.to_csv(HERE/'OUTPUT/aeso_hourly_wind_shape_comparison.csv',index=False)
    return summary


def bc_farms(path):
    df=pd.read_csv(path)
    df=df.dropna(subset=['year','net_generation_mwh','installed_capacity_mw'])
    if df.empty:return {'usable_records':0,'warning':'No observed annual per-project generation and commissioned MW entered'}
    for col in ['net_generation_mwh','installed_capacity_mw']:
        df[col]=pd.to_numeric(df[col],errors='raise')
    df['year']=df.year.astype(int)
    df['hours']=df.year.map(lambda y:8784 if (y%4==0 and (y%100!=0 or y%400==0)) else 8760)
    df['capacity_factor']=df.net_generation_mwh/(df.installed_capacity_mw*df.hours)
    if ((df.capacity_factor<0)|(df.capacity_factor>1)).any():raise ValueError('BC farm calculated CF out of bounds: check units')
    if not df.generation_source_url.notna().all() or not df.capacity_source_url.notna().all():
        raise ValueError('Every BC farm observation requires generation and capacity URLs')
    df.to_csv(HERE/'OUTPUT/bc_farm_annual_observed_capacity_factors.csv',index=False)
    return {'usable_records':len(df),'mean_capacity_factor':float(df.capacity_factor.mean()),
            'warning':'Farm fleet CF does not validate proposed ridge coordinates or within-hour lulls'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gwa',default=str(HERE/'INPUT/gwa_100m_means.csv'))
    ap.add_argument('--statcan',help='Downloaded 25100015-eng.zip or extracted CSV')
    ap.add_argument('--inspect-statcan',action='store_true')
    ap.add_argument('--statcan-wind-label',default='Wind')
    ap.add_argument('--statcan-filter',action='append',default=[],help='Repeat as --statcan-filter "Dimension=exact value"')
    ap.add_argument('--hourly',help='AESO hourly aggregated total generation MW in UTC (user-normalised CSV)')
    ap.add_argument('--time-col',default='time_utc')
    ap.add_argument('--generation-col',default='generation_mw')
    ap.add_argument('--capacity-mw',type=float)
    ap.add_argument('--capacity-col')
    ap.add_argument('--source-description',default='AESO historical CSD hourly generation (user-prepared subset)')
    ap.add_argument('--capacity-definition',default='Must reflect commissioning changes, report source')
    ap.add_argument('--bc-farms',default=str(HERE/'INPUT/bc_wind_farms_TEMPLATE.csv'))
    args=ap.parse_args();(HERE/'OUTPUT').mkdir(exist_ok=True)
    profile=wind_profiles(args.gwa)
    summary={}
    if args.statcan:
        r=statcan(args,profile)
        if args.inspect_statcan:return
        summary['statcan']=r
    if args.hourly:summary['aeso']=hourly(args,profile)
    if args.bc_farms:summary['bc_farms']=bc_farms(args.bc_farms)
    summary['scientific_status']='INDEPENDENT SITE-LEVEL HOURLY WIND VALIDATION NOT ESTABLISHED'
    summary['time_of_comparison_note']='Monthly BC aggregate and hourly Alberta generation are contextual stress tests, not BC ridge observations.'
    (HERE/'OUTPUT/observation_comparison_summary.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
