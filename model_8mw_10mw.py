#!/usr/bin/env python3
"""Independent 8 MW / 10 MW onshore turbine comparison for Digital Gardens vs Wonder Valley.

Not manufacturer-certified power curves or a project yield certification. 8-MW model uses
manufacturer stated rated speed of 11.5 m/s; 10-MW model's rated 11.0 m/s is an
EXPLICIT HYPOTHESIS (maker's public data omit the rated speed). IEC-S site-specific.
Wind shapes: ERA5 100 m scaled to five point GWA mean speeds (not local measurements).
The 40%-cap model attenuates speeds at any site exceeding 40% modelled mean, to
avoid pretending a whole multi-GW wind farm sits in a single 250 m summit pixel.
"""
from pathlib import Path
import sys, json, csv, argparse, time
from functools import lru_cache
import numpy as np
from scipy.optimize import differential_evolution, brentq
from numba import njit

HERE=Path(__file__).resolve().parent
RC=HERE.parent
sys.path[:0] = [str(RC),str(RC/'MODEL_CORE'),str(RC/'OBSERVATIONAL_VALIDATION')]
import storage_fuel_sweetspot as sf
from dispatch_reference import PICK, read_inputs, LOSS_WIND, HVO_L_PER_MWH
from land_constrained_optimizer import make_data, make_fast_inputs
from optimize_sites_cap_cf import run_fast

TURBINES = {
    'SANY_SI19580_8MW': {'rated_mw':8.0,'rotor_m':195.,'cutin_ms':3.,'rated_ms':11.5,'cutout_ms':25.,'curve':'engineering cubic proxy; rated speed is manufacturer stated'},
    'SANY_SI230100_10MW': {'rated_mw':10.0,'rotor_m':230.,'cutin_ms':3.,'rated_ms':11.0,'cutout_ms':25.,'curve':'engineering cubic proxy; 11.0 m/s rated speed ASSUMED, NOT manufacturer verified'},
}

# Align baseline 0.85 model multiplier and additional 6% resource loss with earlier model.
AVAILABILITY_ENVELOPE=.85

def turbine_cf(v,rated_ms,avail=AVAILABILITY_ENVELOPE):
    v=np.asarray(v,dtype=np.float64)
    z=(np.maximum(v,0)**3-3**3)/(rated_ms**3-3**3)
    return np.where((v<3)|(v>=25),0.,np.minimum(np.maximum(z,0.),1.))*avail

@lru_cache(maxsize=1)
def load_weather():
    groups,raw=make_data(RC/'DATA',RC/'DATA/RIDGE_WIND_ERA5_6_POINTS_2010_2025.zip')
    dates,_,_=read_inputs(RC/'DATA',RC/'DATA/RIDGE_WIND_ERA5_6_POINTS_2010_2025.zip')
    return groups,raw,dates

def build_inputs(name,cf_ceiling=.4,wind_mean_multiplier=1.0,rated_override=None):
    m=TURBINES[name]
    rated=rated_override or m['rated_ms']
    basegroups,raw,dates=load_weather()
    groups=[dict(g) for g in basegroups]  # isolate CF scenarios; avoid shared mutable cache across downstream reruns
    inp=RC/'OBSERVATIONAL_VALIDATION'/'INPUT'/'gwa_100m_means.csv'
    vals={r['site_key']:float(r['gwa_mean_100m_ms']) for r in csv.DictReader(inp.open())}
    assert set(vals)==set(PICK)
    site=[]; cf=[]
    for i,k in enumerate(PICK):
        corr=vals[k]*wind_mean_multiplier/raw[i].mean()
        rawcf=turbine_cf(raw[i]*corr,rated).mean()
        eff_corr=corr
        if rawcf>cf_ceiling:
            eff_corr=brentq(lambda q:turbine_cf(raw[i]*q,rated).mean()-cf_ceiling,.2,corr)
        v=turbine_cf(raw[i]*eff_corr,rated)
        site.append({'site_key':k,'GWA_100m_mean_ms':vals[k], 'target_mean_ratio':wind_mean_multiplier,
            'initial_speed_multiplier':float(corr),'applied_speed_multiplier':float(eff_corr),
            'raw_mean_ERA5_ms':float(raw[i].mean()),'uncapped_cf':float(rawcf),
            'screened_cf':float(v.mean()),'cap_activated':bool(rawcf>cf_ceiling)})
        cf.append(v)
    cc=np.stack(cf)
    for g in groups:
        mask=((dates.dt.year==g['year'])&~((dates.dt.month==2)&(dates.dt.day==29))).to_numpy()
        g['windcf']=cc[:,mask]
    W,S,H,L=make_fast_inputs(groups)
    return groups,W,S,H,L,site

def run_case_fast(x,W,S,H,L):
    return run_fast(np.asarray(x,dtype=np.float64),W,S,H,L,0.)

def summarize(x,W,S,H,L,groups,m):
    cap=x[:5];r=run_case_fast(x,W,S,H,L)
    stats={'hvo_ml_mean':float(r[0]),'hvo_ml_max':float(r[1]),'gas_gwh_max':float(r[2]),
        'unserved_gwh_max':float(r[3]),'gen_twh':float(r[4]),'load_twh':float(r[5]),'curtail_twh':float(r[6])}
    cost=sf.cost_25y(cap.sum(),x[5],x[6],stats,x[7])
    cost={k:float(v) for k,v in cost.items() if k!='annual_path_bn'}
    return {'wind_gw_by_site':{k:float(v) for k,v in zip(PICK,cap)},'wind_gw':float(cap.sum()),
            'solar_gw':float(x[5]),'battery_gwh':float(x[6]),'fuel_cap_ml':float(x[7]),
            'stats':stats,'cost':cost,'qualified_dispatch':bool(stats['gas_gwh_max']<.001 and stats['unserved_gwh_max']<.001 and stats['hvo_ml_max']<=x[7]+.001),
            'turbine_count_ceil_by_site':{k:int(np.ceil(v*1000/m['rated_mw'])) for k,v in zip(PICK,cap)},
            'turbine_count_total_ceil':int(np.ceil(cap*1000/m['rated_mw']).sum()),
            'approx_rotor_spacing_envelope_km2':float(np.ceil(cap*1000/m['rated_mw']).sum()*35*m['rotor_m']**2/1e6)}

def optimization(name,cf_ceiling=.4,maxiter=15,popsize=6,seed=2026):
    from scipy.optimize import differential_evolution
    m=TURBINES[name]
    groups,W,S,H,L,sites=build_inputs(name,cf_ceiling)
    fuel_max=sf.P['supply_cap_ml']-sf.P['farm_hvo_ml']
    bounds=[(0.,2.) if k!='nicola_wind_project_regional_proxy' else (0.,0.2) for k in PICK]+[(10.,60.),(10.,120.),(250.,fuel_max)]
    counter=0
    def obj(x):
        nonlocal counter
        counter+=1
        r=run_case_fast(x,W,S,H,L)
        st={'hvo_ml_mean':r[0],'hvo_ml_max':r[1],'gas_gwh_max':r[2],
            'unserved_gwh_max':r[3],'curtail_twh':r[6],'gen_twh':r[4],'load_twh':r[5]}
        cost=sf.cost_25y(np.sum(x[:5]),x[5],x[6],st,x[7])
        return cost['lcoe_25y_cad_mwh'] + 3*r[2] + 45*r[3] + 5*max(0.,r[1]-x[7])
    start=time.monotonic()
    rs=differential_evolution(obj,bounds,seed=seed,maxiter=maxiter,popsize=popsize,
           updating='immediate',workers=1,polish=False,tol=.004,disp=True)
    x=np.array(rs.x)
    out={'turbine':name,'manufacturer_params':m,'site_cf_limit':cf_ceiling,'nicola_capacity_upper_bound_GW':0.2,
         'sites':sites,'candidate':summarize(x,W,S,H,L,groups,m),
         'search':{'maxiter':maxiter,'popsize':popsize,'seed':seed,'evaluations':counter,
                   'success':bool(rs.success),'message':str(rs.message),'elapsed_seconds':time.monotonic()-start},
         'case_warnings':['No proprietary manufacturer power curve supplied or used.','SANY 10MW assumed rated speed.','5 proxied GWA mean wind locations, no farm layouts.','GWA-corrected ERA5 hourly shape not validated in BC.','Transmission, transport, permits, procurement and farm wakes not validated.']}
    return out

def replay_stresses(s):
    name=s['turbine'];m=TURBINES[name];c=s['candidate'];x=np.array(list(c['wind_gw_by_site'].values())+[c['solar_gw'],c['battery_gwh'],c['fuel_cap_ml']])
    ret={}
    for label,kwargs in [('central_40pct',dict(cf_ceiling=.4)),('cap35_same_installation',dict(cf_ceiling=.35)),
                          ('GWA90pct_cap40',dict(cf_ceiling=.4,wind_mean_multiplier=.9)),
                          ('raw_ERA5',dict(cf_ceiling=1.0,wind_mean_multiplier=1.0))]:
        if label=='raw_ERA5':
            # Actual raw ERA5 site resource without speed multipliers.
            basegroups,raw,dates=load_weather()
            groups=[dict(g) for g in basegroups]
            for g in groups: g['windcf']=turbine_cf(g['wind'],m['rated_ms'])
            W,S,H,L=make_fast_inputs(groups)
        else:
            groups,W,S,H,L,sites=build_inputs(name,**kwargs)
        ret[label]=summarize(x,W,S,H,L,groups,m)['stats']
    # 10MW rated-speed uncertainty: 10.5–12 m/s assumed range, same layout
    if m['rated_mw']==10:
        for rated in (10.5,11.5,12.0):
            groups,W,S,H,L,_=build_inputs(name,.4,rated_override=rated)
            ret[f'10MW_assumed_rated_{rated:.1f}ms']=summarize(x,W,S,H,L,groups,m)['stats']
    return ret

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--maxiter',type=int,default=15);ap.add_argument('--popsize',type=int,default=6);ap.add_argument('--out',default=str(HERE/'OUTPUT'));args=ap.parse_args()
    out=Path(args.out);out.mkdir(exist_ok=True)
    allout=[]
    for name in TURBINES:
        print('OPTIMIZING',name,flush=True)
        s=optimization(name,maxiter=args.maxiter,popsize=args.popsize,seed=313 if '8MW' in name else 317)
        s['stress']=replay_stresses(s)
        (out/(name+'_optimization.json')).write_text(json.dumps(s,indent=2)+'\n')
        print('RESULT',name,s['candidate']['cost']['lcoe_25y_cad_mwh'],s['candidate']['stats'],'COUNT',s['candidate']['turbine_count_total_ceil'],flush=True)
        allout.append(s)
    with open(out/'large_turbine_summary.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['turbine','power_MW','rotor_m','wind_GW','solar_GW','battery_GWh','fuel_cap_ML','lcoe_CAD_MWh','mean_HVO_ML','worst_year_HVO_ML','worst_year_gas_GWh','worst_year_gas_35cap_GWh','worst_year_gas_GWA90_GWh','turbine_count','rotor_spacing_km2'])
        for a in allout:
            m=a['manufacturer_params'];c=a['candidate'];w.writerow([a['turbine'],m['rated_mw'],m['rotor_m'],c['wind_gw'],c['solar_gw'],c['battery_gwh'],c['fuel_cap_ml'],c['cost']['lcoe_25y_cad_mwh'],c['stats']['hvo_ml_mean'],c['stats']['hvo_ml_max'],c['stats']['gas_gwh_max'],a['stress']['cap35_same_installation']['gas_gwh_max'],a['stress']['GWA90pct_cap40']['gas_gwh_max'],c['turbine_count_total_ceil'],c['approx_rotor_spacing_envelope_km2']])
if __name__=='__main__':main()
