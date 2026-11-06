#!/usr/bin/env python3
"""Site-free 16-year energy/storage/fuel optimisation for Joule screening.

All 5 wind site capacities are independent [0, 2] GW (2 GW/site envelope is
itself unverified); mean CF from GWA-adjusted ERA5 is limited to an assumed 40%
by reducing the wind-speed correction, NEVER by multiplying generated energy.
Weather chronology from ERA5; no claim of site validation or network deliverability.
"""
import json, sys, argparse
from pathlib import Path
import numpy as np
from numba import njit
from scipy.optimize import differential_evolution, brentq

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(ROOT),str(ROOT/'MODEL_CORE')]
from dispatch_reference import read_inputs,wind_cf,PICK,HVO_L_PER_MWH,LOSS_WIND,B100_MWh
from land_constrained_optimizer import make_data,make_fast_inputs
import storage_fuel_sweetspot as sf
from land_constrained_optimizer import dispatch

@njit(cache=True)
def run_fast(x,W,S,H,L,farm_fuel_ml):
    # W = 16 years x 5 sites x 8760 hours in MW per GW of site capacity
    cap=x[:5]; solar=x[5]; batt=x[6];fuel=x[7]
    hvo_limit=fuel*1.e6/HVO_L_PER_MWH
    hi_hvo=0.;gas_max=0.; unserved_max=0.;sum_hvo=0.;sum_g=0.;sum_load=0.;sum_curt=0.
    for yi in range(L.shape[0]):
        gen=np.zeros(L.shape[1],dtype=np.float64)
        for t in range(L.shape[1]):
            v=solar*S[yi,t]+H[yi,t]
            for si in range(5): v+=cap[si]*W[yi,si,t]
            gen[t]=v
        b,h,g,u,gi,ex,cu=dispatch(gen,L[yi],batt*1000.,0.,hvo_limit,sf_P_BACKUP)
        h_ml=h*HVO_L_PER_MWH/1e6
        if h_ml>hi_hvo:hi_hvo=h_ml
        if g/1000>gas_max:gas_max=g/1000
        if u/1000>unserved_max:unserved_max=u/1000
        sum_hvo+=h_ml
        sum_g+=gen.sum()/1e6
        sum_load+=L[yi].sum()/1e6
        sum_curt+=cu/1e6
    return (sum_hvo/L.shape[0],hi_hvo,gas_max,unserved_max,sum_g/L.shape[0],sum_load/L.shape[0],sum_curt/L.shape[0])

sf_P_BACKUP = 5100. # numba global literal

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--maxiter',type=int,default=18)
    ap.add_argument('--popsize',type=int,default=10)
    ap.add_argument('--cf-ceiling',type=float,default=.4)
    ap.add_argument('--seed',type=int,default=1204)
    ap.add_argument('--fuel-cap',type=float,default=None)
    args=ap.parse_args()
    groups, raw=make_data(ROOT/'DATA',ROOT/'DATA/RIDGE_WIND_ERA5_6_POINTS_2010_2025.zip')
    dates,_,_=read_inputs(ROOT/'DATA',ROOT/'DATA/RIDGE_WIND_ERA5_6_POINTS_2010_2025.zip')
    source=json.loads((HERE/'INPUT/GWA_BC_PROVENANCE.json').read_text())
    import csv
    means={r['site_key']:float(r['gwa_mean_100m_ms']) for r in csv.DictReader((HERE/'INPUT/gwa_100m_means.csv').open())}
    if set(means)!=set(PICK):raise ValueError('Missing GWA sites')
    speeds=[]; cfs=[]; cfraw=[]
    for si,k in enumerate(PICK):
        wind=raw[si]
        mult=means[k]/np.mean(wind)
        uncap=wind_cf(wind,mult).mean()
        if uncap>args.cf_ceiling:
            mult=brentq(lambda q:wind_cf(wind,q).mean()-args.cf_ceiling,0.25,mult)
        speeds.append(mult);cfraw.append(uncap)
        cfs.append(wind_cf(wind,mult))
    cf=np.stack(cfs)
    for g in groups:
        mask=((dates.dt.year==g['year'])&~((dates.dt.month==2)&(dates.dt.day==29))).to_numpy()
        g['windcf']=cf[:,mask]
    W,S,H,L=make_fast_inputs(groups)
    farm=float(sf.P['farm_hvo_ml'])
    limit=sf.P['supply_cap_ml']-farm
    bounds=[(0.,2.)]*5+[(5.,60.),(5.,120.),(100.,limit)]
    if args.fuel_cap is not None: bounds[-1]=(args.fuel_cap,args.fuel_cap)
    n_calls=[0]
    def obj(x):
        r=run_fast(np.asarray(x,dtype=np.float64),W,S,H,L,farm)
        stat={'hvo_ml_mean':r[0],'hvo_ml_max':r[1],'gas_gwh_max':r[2],
              'unserved_gwh_max':r[3],'gen_twh':r[4],'load_twh':r[5],'curtail_twh':r[6]}
        c=sf.cost_25y(sum(x[:5]),x[5],x[6],stat,x[7])
        n_calls[0]+=1
        # Annual fuel ≤ plant-wide supply cap; no gas/no unserved required.
        score=c['lcoe_25y_cad_mwh'] + 2.5*r[2] + 25.*r[3] + 4.*max(0,r[1]-x[7])
        return score
    res=differential_evolution(obj,bounds,seed=args.seed,maxiter=args.maxiter,
               popsize=args.popsize,polish=True,updating='immediate',workers=1,
               tol=.004,disp=True)
    x=np.array(res.x)
    st=sf.run_case(groups,x[:5],x[5],x[6],x[7])
    cc=sf.cost_25y(x[:5].sum(),x[5],x[6],st,x[7])
    result={'site_keys':list(PICK),'wind_gw_per_site':dict(zip(PICK,[float(q) for q in x[:5]])),
      'wind_speed_multiplier_used':dict(zip(PICK,[float(q) for q in speeds])),
      'wind_cf_before_ceiling':dict(zip(PICK,[float(q) for q in cfraw])),
      'wind_cf_after_ceiling':dict(zip(PICK,[float(q) for q in cf.mean(axis=1)])),
      'solar_gw':float(x[5]),'battery_gwh':float(x[6]),'fuel_limit_ml':float(x[7]),
      'total_wind_gw':float(sum(x[:5])),'solar_land_envelope_ha_at_1_4ha_per_mwp':float(x[5]*1000*1.4),
      'wind_spacing_land_km2_at_3mw_km2':float(sum(x[:5])*1000/3),
      'n_evaluations':n_calls[0], 'optimization':{'seed':args.seed,'maxiter':args.maxiter,
      'popsize':args.popsize,'objective':float(res.fun),'success':bool(res.success),'message':str(res.message)},
      'dispatch':st,'cost':{k:v for k,v in cc.items() if k!='annual_path_bn'},
      'physical':{'max_site_gw':2.,'cf_ceiling':args.cf_ceiling,'terrain_discount':True,
        'assumes_no_transmission_constraints':True,'site_positions_are_proxies':True,
        'no_annual_or_hourly_validation_from_GWA_only':True},
      'comparators':'Wonder Valley remains a matched-load model; gas twin economics recalculated in companion model',
      'gas_free_condition_in_model':bool(st['gas_gwh_max']<.001 and st['unserved_gwh_max']<.001 and st['hvo_ml_max']<=x[7]+.0001),
      'requires_independent_validation':True}
    p=HERE/'OUTPUT';p.mkdir(exist_ok=True)
    o=p/(f'site_free_cf{int(args.cf_ceiling*100)}_fuel{int(args.fuel_cap)}_optimization.json' if args.fuel_cap else f'site_free_cf{int(args.cf_ceiling*100)}_optimization.json');o.write_text(json.dumps(result,indent=2)+'\n')
    print('SAVED',o,'\n',json.dumps(result,indent=2)[:3500],flush=True)
if __name__=='__main__':main()
