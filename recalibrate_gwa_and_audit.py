#!/usr/bin/env python3
"""Independent mean-atlas correction, ERA5 hourly scenario, BC monthly shape, and AESO transfer stress.

All assumptions are explicitly scenario-based: GWA means are NOT onsite measurements;
monthly BC output cannot be converted to CF without the commissioning timeline;
Alberta output is NOT the local BC temporal pattern. Reuse dispatch kernel, unchanged.
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'MODEL_CORE'))
import storage_fuel_sweetspot as sf
from land_constrained_optimizer import make_data,dispatch
from dispatch_reference import read_inputs,wind_cf,PICK,LOSS_WIND,HVO_L_PER_MWH
from replay_wind import import_atlas

WEATHER=ROOT/'DATA/RIDGE_WIND_ERA5_6_POINTS_2010_2025.zip'
GWA=HERE/'INPUT/gwa_100m_means.csv'
BC=HERE/'INPUT/BC_monthly_observed_wind_generation_2016_2026.csv'
AESO=HERE/'INPUT/aeso_wind_hourly_utc_2015_2025.csv'
OUT=HERE/'OUTPUT';OUT.mkdir(exist_ok=True)


def yrshape_comparison(dates,wind_cf_fleet):
    b=pd.read_csv(BC)
    b['month']=pd.to_datetime(b['month'])
    b=b[b.month.dt.year.between(2016,2025)].copy()
    b['mean_mw']=b.observed_wind_generation_MWh/(b.month.dt.days_in_month*24)
    b['year']=b.month.dt.year
    b['obs_norm']=b.mean_mw/b.groupby('year').mean_mw.transform('mean')
    data=pd.DataFrame({'date':pd.to_datetime(dates,utc=True),'sim_cf':wind_cf_fleet})
    data=data[(data.date.dt.year>=2016)&(data.date.dt.year<=2025)]
    data['month']=data.date.dt.tz_localize(None).dt.to_period('M').dt.to_timestamp()
    m=data.groupby('month',as_index=False).sim_cf.mean()
    m['year']=m.month.dt.year
    m['sim_norm']=m.sim_cf/m.groupby('year').sim_cf.transform('mean')
    joined=b.merge(m,on=['month','year'],how='inner',validate='one_to_one')
    assert len(joined)==120, f'Expected 120 matched months, found {len(joined)}'
    corr=float(joined.obs_norm.corr(joined.sim_norm))
    rmse=float(np.sqrt(np.mean((joined.obs_norm-joined.sim_norm)**2)))
    joined.to_csv(OUT/'bc_observed_vs_gwa_era5_monthly_seasonality.csv',index=False)
    seasonal=joined.assign(month_number=joined.month.dt.month).groupby('month_number',as_index=False).agg(
        observed_normalized=('obs_norm','mean'),modeled_normalized=('sim_norm','mean'))
    seasonal.to_csv(OUT/'bc_observed_vs_gwa_era5_seasonal_shape.csv',index=False)
    return {'matched_months':len(joined),'mean_annual_normalized_shape_pearson_r':corr,
            'mean_annual_normalized_shape_rmse':rmse,
            'BC_observation':'StatsCan 25-10-0015-01 monthly provincial MWh; NOT plant CF',
            'caution':'Normalising each year partly mitigates installed-capacity changes, but does not control within-year commissioning, curtailment, or farm composition.'}


def continuity_and_lulls(obs):
    obs=obs.sort_values('time_utc').copy()
    v=obs.observed_cf.to_numpy()
    dt=obs.time_utc.diff().dt.total_seconds().div(3600).fillna(1).to_numpy()
    inl=(v<.1);streak=0;worst=0;over24=0
    for k,b in enumerate(inl):
        if dt[k]>1.01:
            if streak>=24:over24+=1
            worst=max(streak,worst);streak=0
        if b:streak+=1
        else:
            if streak>=24:over24+=1
            worst=max(streak,worst);streak=0
    if streak>=24:over24+=1
    worst=max(worst,streak)
    return {'observed_hours':len(v),'fraction_under_10pct':float(inl.mean()),
            'longest_lull_below_10pct_hours':int(worst),'number_lulls_ge_24h':int(over24),
            'gaps_in_record_hours':int(np.sum(dt>1.01)),
            'p95_absolute_hourly_ramp_cf':float(np.quantile(np.abs(np.diff(v)),.95))}


def transfer_stress(groups,wind_caps,solar,batt,fuel,mean_by_year):
    obs=pd.read_csv(AESO,parse_dates=['time_utc'])
    obs['time_utc']=pd.to_datetime(obs.time_utc,utc=True)
    obs=obs[(obs.time_utc.dt.year>=2016)&(obs.time_utc.dt.year<=2025)].copy()
    obs['observed_cf']=obs.generation_mw/obs.capacity_mw
    metrics=continuity_and_lulls(obs)
    rows=[]
    yeargroups={g['year']:g for g in groups}
    for y in range(2016,2026):
        a=obs[obs.time_utc.dt.year==y].copy()
        a=a[~((a.time_utc.dt.month==2)&(a.time_utc.dt.day==29))]
        assert len(a)==8760, (y,len(a))
        assert (a.time_utc.diff().dt.total_seconds().dropna()==3600).all() or y in (2016,2020,2024)
        # An aggregate Alberta fleet is substituted for a hypothetical *perfectly synchronized*
        # BC wind portfolio, with each year's CF mean matched to BC GWA scenario output.
        # Bisection avoids truncation bias at 0.85 availability bound.
        v=np.clip(a.observed_cf.to_numpy(),0,1)
        target=float(mean_by_year[y]);lo,hi=0.,100.
        for _ in range(60):
            mid=(lo+hi)/2
            if np.minimum(v*mid,.85).mean()<target:lo=mid
            else:hi=mid
        standin=np.minimum(v*(lo+hi)/2,.85)
        g=yeargroups[y]
        generation=wind_caps.sum()*1000*(1-LOSS_WIND)*standin + solar*g['solar']+g['hydro']
        b,h,gas,u,gr,ex,cur=dispatch(generation,g['load'],batt*1000,0.,fuel*1e6/HVO_L_PER_MWH,sf.P['backup_mw'])
        rows.append(dict(year=y,mean_matched_cf=float(standin.mean()),
             hvo_ml=h*HVO_L_PER_MWH/1e6,gas_gwh=gas/1000,unserved_gwh=u/1000,
             generation_twh=generation.sum()/1e6,case='AESO_hourly_shape_transfer_stress_NOT_BC_validation'))
    pd.DataFrame(rows).to_csv(OUT/'aeso_hourly_shape_transfer_2016_2025.csv',index=False)
    return {**metrics,'n_tested_years':len(rows),'max_gas_gwh':float(max(r['gas_gwh'] for r in rows)),
            'max_unserved_gwh':float(max(r['unserved_gwh'] for r in rows)),
            'max_hvo_ml':float(max(r['hvo_ml'] for r in rows)),
            'source':'AESO CSD observed generation/reported capability, hourly UTC',
            'interpretation':'Out-of-region aggregate fleet pattern substituted to stress diversification, NOT BC site validation or probability forecast'}


def optimize(groups,basewind):
    sel=json.loads((ROOT/'OUTPUT/optimization_provenance.json').read_text())
    sweet=json.loads((ROOT/'OUTPUT/storage_fuel_sweetspot.json').read_text())['sweet_spot']
    w0=np.asarray(sel['wind_GW'],dtype=float)
    dc_fuel_limit=float(sf.P['supply_cap_ml']-sf.P['farm_hvo_ml'])
    winds=[.55,.7,.85,1.0,1.0435]
    solars=[10.,15.,20.,25.,30.,40.,50.,60.]
    batts=[10.,20.,30.,45.,60.,89.,120.]
    fuels=[300.,500.,700.,900.,dc_fuel_limit]
    rows=[]
    for m in winds:
        cap=np.minimum(w0*m,2.)
        for so in solars:
            for ba in batts:
                for fuel in fuels:
                    st=sf.run_case(groups,cap,so,ba,fuel)
                    c=sf.cost_25y(float(cap.sum()),so,ba,st,fuel)
                    ok=st['gas_gwh_max']<1e-3 and st['unserved_gwh_max']<1e-3 and st['hvo_ml_max']+sf.P['farm_hvo_ml']<=sf.P['supply_cap_ml']+1e-4
                    rows.append(dict(wind_multiplier=m,wind_gw=float(cap.sum()),solar_gw=so,battery_gwh=ba,
                        hvo_cap_ml=fuel,gas_free_all_years=bool(ok),**st,
                        lcoe_25y_cad_mwh=c['lcoe_25y_cad_mwh'],capex_cad_bn=c['capex_bn'],
                        cost_after_payoff_cad_mwh=c['cost_after_payoff_cad_mwh']))
        print('Completed candidates wind',m,flush=True)
    grid=pd.DataFrame(rows)
    grid.to_csv(OUT/'GWA_16year_storage_fuel_candidate_grid.csv',index=False,float_format='%.7f')
    q=grid[grid.gas_free_all_years].sort_values('lcoe_25y_cad_mwh')
    def rec(row):
        if row is None:return None
        return {k:(float(v) if isinstance(v,(float,np.floating)) else bool(v) if isinstance(v,(bool,np.bool_)) else int(v) if isinstance(v,(int,np.integer)) else v)
                for k,v in row.to_dict().items()}
    spot_cap=np.minimum(w0*float(sweet['wind_scale']),2.)
    chosen_stats=sf.run_case(groups,spot_cap,float(sweet['solar_GW']),float(sweet['battery_GWh']),dc_fuel_limit)
    chosen_cost=sf.cost_25y(float(spot_cap.sum()),sweet['solar_GW'],sweet['battery_GWh'],chosen_stats,dc_fuel_limit)
    batt_cap=w0; batt_stats=sf.run_case(groups,batt_cap,sel['solar_GW'],sel['battery_GWh'],300.)
    batt_cost=sf.cost_25y(float(batt_cap.sum()),sel['solar_GW'],sel['battery_GWh'],batt_stats,300.)
    report={'n_candidates_evaluated':len(grid),'n_qualifying':len(q),
        'candidate_ranges':{'wind_alloc_multiplier':winds,'solar_gw':solars,'battery_gwh':batts,'hvo_cap_ml':fuels},
        'discrete_grid_best':rec(q.iloc[0]) if not q.empty else None,
        'existing_sweet_spot_gwa':{'capacities':{'wind_gw':float(spot_cap.sum()),'solar_gw':sweet['solar_GW'],
                             'battery_gwh':sweet['battery_GWh'],'hvo_cap_ml':dc_fuel_limit},**chosen_stats,
                             'cost_25y_cad_mwh':chosen_cost['lcoe_25y_cad_mwh'],'capex_bn':chosen_cost['capex_bn']},
        'existing_battery_heavy_gwa':{'capacities':{'wind_gw':float(batt_cap.sum()),'solar_gw':sel['solar_GW'],
                             'battery_gwh':sel['battery_GWh'],'hvo_cap_ml':300.},**batt_stats,
                             'cost_25y_cad_mwh':batt_cost['lcoe_25y_cad_mwh'],'capex_bn':batt_cost['capex_bn']},
        'caveats':['No per-site SCADA/mast measurements','GWA corrects mean, not ERA5 hour-to-hour wind shape',
                   'No electrical network interconnection proof','Costs are modeled screening estimates',
                   'No capital optimization beyond the stated discrete candidate grid'],
        'gas_free_is_modeled_not_validated':True}
    (OUT/'GWA_storage_fuel_search_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Qualifying',len(q),'best',json.dumps(report['discrete_grid_best'],indent=2)[:900],flush=True)
    return report


def main():
    groups,wind=make_data(ROOT/'DATA',WEATHER)
    dates,_,orig=read_inputs(ROOT/'DATA',WEATHER)
    atlas=import_atlas(GWA,PICK)
    means=orig.mean(axis=1)
    mult=np.asarray([atlas[k][0]/float(means[i]) for i,k in enumerate(PICK)])
    cfs=np.stack([wind_cf(orig[i],mult[i]) for i in range(len(PICK))])
    sel=json.loads((ROOT/'OUTPUT/optimization_provenance.json').read_text())
    sweet=json.loads((ROOT/'OUTPUT/storage_fuel_sweetspot.json').read_text())['sweet_spot']
    cap=np.minimum(np.asarray(sel['wind_GW'])*sweet['wind_scale'],2.)
    portfolio_cf=(cap[:,None]*cfs).sum(axis=0)/cap.sum()
    for g in groups:
        idx=((dates.dt.year==g['year']) &~((dates.dt.month==2)&(dates.dt.day==29))).to_numpy()
        g['windcf']=cfs[:,idx]
    r_bc=yrshape_comparison(dates,portfolio_cf)
    means_by_year={g['year']:float(np.average(g['windcf'].mean(axis=1),weights=cap)) for g in groups}
    r_aeso=transfer_stress(groups,cap,sweet['solar_GW'],sweet['battery_GWh'],sweet['fuel_cap_ML'],means_by_year)
    report=optimize(groups,orig)
    combined={'bc_monthly':r_bc,'aeso_transfer':r_aeso,'wind_capacity_factor_selected_capacity_weighted':float(portfolio_cf.mean()),
              'results_status':'Mean-corrected ERA5 simulation and externally benchmarked scenarios, NOT site-level validation',
              'calibration_multipliers':{p:float(m) for p,m in zip(PICK,mult)},'search':report}
    (OUT/'GWA_BC_AESO_integrated_summary.json').write_text(json.dumps(combined,indent=2)+'\n')
    print('BC monthly r',r_bc['mean_annual_normalized_shape_pearson_r'],'AESO stress max gas GWh',r_aeso['max_gas_gwh'])

if __name__=='__main__':main()
