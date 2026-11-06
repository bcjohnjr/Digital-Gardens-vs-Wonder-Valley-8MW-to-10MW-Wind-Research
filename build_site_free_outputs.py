#!/usr/bin/env python3
"""Generate internally consistent constrained site-specific screening scenario inputs.
All feasibility and prices are conditional; does not validate high-voltage deliverability.
"""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;R=HERE.parent;OUT=R/'OUTPUT';O=HERE/'OUTPUT';sys.path[:0]=[str(R),str(R/'MODEL_CORE')]
import storage_fuel_sweetspot as sf
from land_constrained_optimizer import make_data
from dispatch_reference import read_inputs,wind_cf,PICK,HVO_L_PER_MWH,LOSS_WIND
from scipy.optimize import brentq

def get(cf,fuel=None):
 p=O/(f'site_free_cf{cf}_fuel{int(fuel)}_optimization.json' if fuel else f'site_free_cf{cf}_optimization.json')
 return json.loads(p.read_text())

def pack(d,groups):
 c=np.array([d['wind_gw_per_site'][k] for k in PICK]); solar=d['solar_gw'];bat=d['battery_gwh'];fuel=d['fuel_limit_ml']
 st=sf.run_case(groups,c,solar,bat,fuel);cp=sf.cost_25y(c.sum(),solar,bat,st,fuel)
 assert abs(cp['lcoe_25y_cad_mwh']-d['cost']['lcoe_25y_cad_mwh'])<.01
 return dict(wind_GW=float(sum(c)),wind_GW_per_site=c.tolist(),wind_scale=None,
      solar_GW=solar,battery_GWh=bat,fuel_cap_ML=fuel,qualifies=bool(st['gas_gwh_max']<.001 and st['unserved_gwh_max']<.001),**st,**cp)

def main():
 data=get(40);ref=get(40,300);low=get(40,600);s35=get(35)
 groups,_=sf.load_years();assert len(groups)==16
 sw=pack(data,groups);rr=pack(ref,groups);ll=pack(low,groups)
 sw35cp=s35['cost']['lcoe_25y_cad_mwh']
 fuel_allow=sf.P['supply_cap_ml']-sf.P['farm_hvo_ml']
 assert sw['hvo_ml_max']+sf.P['farm_hvo_ml']<=sf.P['supply_cap_ml']
 # no false assertion that fuel allocated exactly equals full national supply.
 group_select={'reference_selected':rr,'sweet_spot':sw,'low_fuel_sweet_spot':ll,
    'tradeoff_curve':[dict(fuel_cap_ML=z['fuel_cap_ML'],battery_GWh=z['battery_GWh'],solar_GW=z['solar_GW'],wind_GW=z['wind_GW'], lcoe_25y_cad_mwh=z['lcoe_25y_cad_mwh'],hvo_ml_mean=z['hvo_ml_mean'],hvo_ml_max=z['hvo_ml_max'],fuel_lifecycle_mt=z['fuel_lifecycle_mt'],capex_bn=z['capex_bn']) for z in [rr,ll,sw]],
    'data_centre_fuel_allowance_ml':fuel_allow, 'wind_benchmark_cf':float(np.average([data['wind_cf_after_ceiling'][k] for k in PICK],weights=sw['wind_GW_per_site'])),
    'n_cases':sum(z['n_evaluations'] for z in [data,ref,low,s35]),
    'n_qualifying':3,'sweet_spot_at_30pct_wind':{'not_computed':'Variable per-site CF and mixture: uniform 30% target case not a direct comparable site model'},
    'scenario_status':'CONDITIONAL 40%-CF CEILING / FREE SITE CAPACITIES; optimizers hit iteration limit; NOT A VERIFIED GLOBAL OPTIMUM',
    'optimization_not_global_proof':True, 'per_site_wind_caps':data['wind_gw_per_site'],
    'comparison_no_gas_300ml': 'Separately optimized 300ML HVO cap, both site-free under identical 40% cap',
    'params':sf.P}
 (OUT/'storage_fuel_sweetspot.json').write_text(json.dumps(group_select,indent=2)+'\n')
 pd.DataFrame(group_select['tradeoff_curve']).to_csv(OUT/'storage_fuel_sweetspot_grid.csv',index=False)
 # raw ERA5 and stronger terrain-screen sensitivity for THIS selected portfolio
 gr,raw=make_data(R/'DATA',R/'DATA/RIDGE_WIND_ERA5_6_POINTS_2010_2025.zip')
 dates,_,_=read_inputs(R/'DATA',R/'DATA/RIDGE_WIND_ERA5_6_POINTS_2010_2025.zip')
 caps=np.array(sw['wind_GW_per_site']);annual=[];cases={}
 allcfs={}
 allcfs['base40']=np.stack([wind_cf(raw[i],data['wind_speed_multiplier_used'][PICK[i]]) for i in range(5)])
 allcfs['cap35_same_design']=np.stack([wind_cf(raw[i],s35['wind_speed_multiplier_used'][PICK[i]]) for i in range(5)])
 allcfs['raw_era5']=np.stack([wind_cf(raw[i],1.) for i in range(5)])
 for case,cfs in allcfs.items():
    for g in gr:
        mask=((dates.dt.year==g['year'])&~((dates.dt.month==2)&(dates.dt.day==29))).to_numpy()
        g['windcf']=cfs[:,mask]
    res=sf.run_case(gr,caps,sw['solar_GW'],sw['battery_GWh'],sw['fuel_cap_ML'])
    cases[case]=res
    for g in gr:
        generation=(caps@g['windcf'])*1000*(1-LOSS_WIND)+sw['solar_GW']*g['solar']+g['hydro']
        from land_constrained_optimizer import dispatch
        b,h,gas,unserved,grid,ex,cur=dispatch(generation,g['load'],sw['battery_GWh']*1000,0,sw['fuel_cap_ML']*1e6/HVO_L_PER_MWH,sf.P['backup_mw'])
        annual.append(dict(case=case,year=g['year'],gas_gwh=gas/1000,unserved_gwh=unserved/1000,hvo_ml=h*HVO_L_PER_MWH/1e6,wind_gwh=(caps@g['windcf']).sum()*1000*(1-LOSS_WIND)/1000, curtail_twh=cur/1e6))
 pd.DataFrame(annual).to_csv(O/'site_free_16year_dispatch_all_cases.csv',index=False)
 (O/'site_free_16year_dispatch_summary.json').write_text(json.dumps({'same_selected_site_allocation':sw['wind_GW_per_site'],'cases':cases,'best_35_cap_own_allocation_cost_cad_mwh':sw35cp,'is_transmission_deliverability_validated':False,'is_independent_hourly_wind_validated':False},indent=2)+'\n')
 # BC monthly 2020 outlier: retain in raw file and explicitly analyze both specifications.
 b=pd.read_csv(HERE/'INPUT/BC_monthly_observed_wind_generation_2016_2026.csv')
 b['month']=pd.to_datetime(b.month); b=b[(b.month.dt.year>=2016)&(b.month.dt.year<=2025)].copy()
 b['year']=b.month.dt.year
 b['obs_norm']=b.observed_wind_generation_MWh/b.groupby('year').observed_wind_generation_MWh.transform('mean')
 cfs=allcfs['base40'];fcf=np.average(cfs,axis=0,weights=caps)
 md=pd.DataFrame({'month':dates.dt.tz_localize(None).dt.to_period('M').dt.to_timestamp(),'cf':fcf})
 md=md.groupby('month',as_index=False).cf.mean();md['year']=md.month.dt.year
 md['sim_norm']=md.cf/md.groupby('year').cf.transform('mean')
 j=b.merge(md,on=['month','year'],how='inner');assert len(j)==120
 j['excluded_2020_sensitivity']=j.year==2020
 j.to_csv(O/'site_free_bc_observed_monthly_comparison.csv',index=False)
 rp={'all_2016_2025':float(j.obs_norm.corr(j.sim_norm)), 'excluding_2020':float(j.loc[j.year!=2020,'obs_norm'].corr(j.loc[j.year!=2020,'sim_norm'])),
    'year_2020_gwh':float(b[b.year==2020].observed_wind_generation_MWh.sum()/1000),
    'bc_2021_2025_annual_mean_gwh':float(b[b.year.between(2021,2025)].observed_wind_generation_MWh.sum()/5/1000),
    'installed_bc_wind_capacity_mw_external_CER':702,
    'capacity_factor_2021_2025_if_installed_capacity_702mw_unchanged':float(b[b.year.between(2021,2025)].observed_wind_generation_MWh.sum()/(5*702*8760)),
    'caveat':'702MW is a recent snapshot not verified yearly historical capacity; NOT a historical observed wind plant CF; 2020 excluded from sensitivity ONLY, not deleted.'}
 (O/'site_free_bc_monthly_validation.json').write_text(json.dumps(rp,indent=2)+'\n')
 print('Selected',sw['wind_GW'],sw['solar_GW'],sw['battery_GWh'],sw['lcoe_25y_cad_mwh'])
 print('Ref cost',rr['lcoe_25y_cad_mwh'],'cases',cases,'BC',rp)
if __name__=='__main__':main()
