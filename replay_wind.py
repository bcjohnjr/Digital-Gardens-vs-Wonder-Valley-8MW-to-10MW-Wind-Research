#!/usr/bin/env python3
"""Reanalyse 16 historic years using raw ERA5 and, if supplied, GWA 100-m mean corrections.

Mean-only GWA corrections do not validate hourly ERA5 variability. No GWA values
are inferred, extrapolated, or manufactured. Outputs are segregated and NOT
written over the manuscript's originally published outputs.
"""
from pathlib import Path
import argparse, csv, json, sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'MODEL_CORE'))
import storage_fuel_sweetspot as sf
from land_constrained_optimizer import make_data, dispatch
from dispatch_reference import read_inputs, wind_cf, HVO_L_PER_MWH, LOSS_WIND


def import_atlas(path, expected):
    if not path.exists(): return None
    rows = list(csv.DictReader(path.open(encoding='utf-8-sig')))
    got = {}
    for r in rows:
        if not r.get('gwa_mean_100m_ms','').strip(): continue
        k = r['site_key'].strip()
        if k in got: raise ValueError('Duplicated GWA site: '+k)
        if k not in expected: raise ValueError('Unknown GWA site: '+k)
        v = float(r['gwa_mean_100m_ms'])
        if not (0.5 < v < 35): raise ValueError('Nonphysical GWA 100m speed for '+k)
        if not r.get('source_url','').strip() or not r.get('gwa_version','').strip():
            raise ValueError('Need source_url and gwa_version for '+k)
        got[k] = (v,r)
    if not got: return None
    missing = set(expected) - set(got)
    if missing: raise ValueError('Partial GWA input (no extrapolation): '+', '.join(sorted(missing)))
    return got


def summarize_fuel(groups, caps, solar_gw, battery_gwh, hvo_ml, name):
    from storage_fuel_sweetspot import P
    rows = []
    for g in groups:
        generation = (caps @ g['windcf'])*1000*(1-LOSS_WIND) + solar_gw*g['solar']+g['hydro']
        b,h,gas,unserved,grid_import,export,curtailed = dispatch(
            generation,g['load'],battery_gwh*1000,0.,hvo_ml*1e6/HVO_L_PER_MWH,P['backup_mw'])
        rows.append({'case':name, 'weather_year':int(g['year']), 'gas_gwh':float(gas/1000),
                     'unserved_gwh':float(unserved/1000), 'hvo_ml':float(h*HVO_L_PER_MWH/1e6),
                     'curtail_twh':float(curtailed/1e6), 'generation_twh':float(generation.sum()/1e6),
                     'load_twh':float(g['load'].sum()/1e6)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gwa',default=str(HERE/'INPUT/gwa_100m_means.csv'))
    ap.add_argument('--out',default=str(HERE/'OUTPUT'))
    args = ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    groups, wind = make_data(ROOT/'DATA', ROOT/'DATA/RIDGE_WIND_ERA5_6_POINTS_2010_2025.zip')
    dates, _, original = read_inputs(ROOT/'DATA', ROOT/'DATA/RIDGE_WIND_ERA5_6_POINTS_2010_2025.zip')
    from dispatch_reference import PICK
    assert wind.shape[0] == len(PICK) == 5 and original.shape == wind.shape
    atlas = import_atlas(Path(args.gwa), PICK)
    sel=json.loads((ROOT/'OUTPUT/optimization_provenance.json').read_text())
    s=json.loads((ROOT/'OUTPUT/storage_fuel_sweetspot.json').read_text())['sweet_spot']
    caps=np.asarray(s.get('wind_GW_per_site', np.minimum(np.asarray(sel['wind_GW'])*s.get('wind_scale',1.),2.0)),dtype=float)
    solo=s['solar_GW']; battery=s['battery_GWh']; fuel=s['fuel_cap_ML']
    speedmeans=wind.mean(axis=1)
    reports=[]; yearly=[]; cases=[('raw_era5',np.ones(5))]
    (out/'ridge_resource_comparison.csv').unlink(missing_ok=True)
    if atlas:
        corrections=np.array([atlas[k][0]/float(speedmeans[j]) for j,k in enumerate(PICK)])
        cases.append(('gwa_mean_corrected_hourly_unvalidated',corrections))
    for name,scales in cases:
        cfs=np.stack([wind_cf(w,sc) for w,sc in zip(wind,scales)]).astype('float64')
        for g in groups:
            mask=((dates.dt.year == g['year']) & ~((dates.dt.month==2)&(dates.dt.day==29))).to_numpy()
            g['windcf']=cfs[:,mask]
        peryr=summarize_fuel(groups,caps,solo,battery,fuel,name)
        yearly.extend(peryr)
        stats=sf.run_case(groups,caps,solo,battery,fuel)
        cost=sf.cost_25y(caps.sum(),solo,battery,stats,fuel)
        reports.append({'case':name,'wind_capacity_factor_five_site_unweighted':float(cfs.mean()),
                        'wind_capacity_factor_selected_installed_capacity_weighted':float(np.average(cfs.mean(axis=1),weights=caps)),
                        'max_gas_gwh_16yr':float(stats['gas_gwh_max']),
                        'max_unserved_gwh_16yr':float(stats['unserved_gwh_max']),
                        'mean_hvo_ml_year':float(stats['hvo_ml_mean']),
                        'max_hvo_ml_year':float(stats['hvo_ml_max']),
                        'renewable_generation_twh_year':float(stats['gen_twh']),
                        'capital_bn_CAD':float(cost['capex_bn']),
                        'lcoe_proxy_cad_mwh_excludes_emergency_gas_cost':float(cost['lcoe_25y_cad_mwh']),
                        'gas_free_in_all_16_historical_years':bool(stats['gas_gwh_max']<1e-3 and stats['unserved_gwh_max']<1e-3),
                        'is_validated_hourly_resource':False})
        for j,k in enumerate(PICK):
            r={'case':name,'site_key':k,'raw_era5_mean_speed_ms':float(speedmeans[j]),
               'model_speed_multiplier':float(scales[j]),
               'model_annual_cf':float(cfs[j].mean()),'terrain_corrected_mean_speed_ms':float(speedmeans[j]*scales[j]),
               'not_an_hourly_validation':True}
            if atlas: r['gwa_100m_mean_ms']=atlas[k][0]
            # No statement of model validation is written on GWA mean alone.
            with (out/'ridge_resource_comparison.csv').open('a' if (out/'ridge_resource_comparison.csv').exists() and (name!='raw_era5' or j>0) else 'w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(r.keys()))
                if f.tell()==0:w.writeheader()
                w.writerow(r)
        print(name, 'worst gas GWh=',round(stats['gas_gwh_max'],2),
              'installed-capacity-weighted CF=',round(reports[-1]['wind_capacity_factor_selected_installed_capacity_weighted'],4),flush=True)
    pd.DataFrame(yearly).to_csv(out/'annual_16_year_wind_replays.csv',index=False)
    (out/'wind_replay_summary.json').write_text(json.dumps({
        'cases':reports,
        'selected_original_capacity':{'wind_gw_per_site':caps.tolist(),'solar_gw':solo,
                                      'battery_gwh':battery,'hvo_cap_ml':fuel},
        'calibration_limits':['GWA mean does not validate ERA5 seasonal or hourly structure',
            'the original five model site points are proxies and not engineered turbine positions',
            'monthly provincial generation and Alberta hourly output are comparisons, not proof of individual BC ridges',
            'resource/dispatch scenarios exclude actual transmission interconnection and firm fuel supply',
            'cost proxy excludes emergency gas and pilot costs if gas_gwh > 0'],
        'validated_result':False,
        'data_provenance':'GWA national 100m mean-speed raster sampled at five site/proxy coordinates; ERA5 hourly wind shape retained and unvalidated; four other benchmark files separately sourced'
    },indent=2)+'\n')
    print('Outputs:',out/'wind_replay_summary.json',out/'annual_16_year_wind_replays.csv')

if __name__=='__main__': main()
