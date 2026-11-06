"""Conditional least annualized cost capacity search with full 16-year historical verification.
Historical source: self-contained V20 preservation vault; inferred 30/34.79/40% wind yield
is a portfolio-level expectation sensitivity, not observed hub-height output.
"""
from pathlib import Path
import sys, json, time
import numpy as np, pandas as pd
from scipy.optimize import differential_evolution,brentq
HERE=Path(__file__).resolve().parent
BASE=HERE/'DATA'
sys.path.insert(0,str(HERE/'MODEL_CORE'))
from dispatch_reference import read_inputs,wind_cf,PICK,HVO_L_PER_MWH
from land_constrained_optimizer import make_data,make_fast_inputs,fast_objective,evaluate
from numba import njit
OUT=Path(__file__).resolve().parent/'OUTPUT';OUT.mkdir(exist_ok=True)
print('loading weather...',flush=True)
groups,wind=make_data(BASE,BASE/'RIDGE_WIND_ERA5_6_POINTS_2010_2025.zip')
# Source benchmark as in reference model
ref_cf=(4_830_000 -104*8760*.18)/(1531*8760)
print('source benchmark',ref_cf,flush=True)
windcache={}
for mode,cf in [('raw_ERA5',None),('assumed_30pct',.30),('BC_Hydro_fleet_expected_34p79pct',ref_cf),('assumed_40pct',.40)]:
 scales=[]
 for w in wind:
  if cf is None:scales.append(1.)
  else:
   candidates=np.arange(1.,8.01,.05)
   pairs=[(candidates[k],candidates[k+1]) for k in range(len(candidates)-1) if wind_cf(w,candidates[k]).mean()<cf<=wind_cf(w,candidates[k+1]).mean()]
   if not pairs:raise ValueError('Cannot bracket wind calibration '+str(cf))
   scales.append(float(brentq(lambda s:wind_cf(w,s).mean()-cf,*pairs[0])))
 windcache[mode]=np.stack([wind_cf(w,s) for w,s in zip(wind,scales)]).astype('float64')
 print('wind mode',mode,'scales',np.round(scales,3),flush=True)
# groups store only year fractions of original globally calibrated wind CF; make_data gives sliced wind
from dispatch_reference import read_inputs
dates,_,_=read_inputs(BASE,BASE/'RIDGE_WIND_ERA5_6_POINTS_2010_2025.zip')
mask_by_year={y:((dates.dt.year==y)&~((dates.dt.month==2)&(dates.dt.day==29))).to_numpy() for y in range(2010,2026)}
for g in groups:g['windcf']=windcache['BC_Hydro_fleet_expected_34p79pct'][:,mask_by_year[g['year']]]
W,S,H,L=make_fast_inputs(groups)
# real annualized cost model. Units cost CAD B; bounds GW, GWh.
# cost x: region wind GW, solar GW, battery GWh
# 0.085 recovery, fixed O&M assumptions from reference model, 5.95GW engines already included in 26.94B fixed network/generation
@njit(cache=True)
def score(x,W,S,H,L,penalty):
    capex=(x[0]+x[1]+x[2]+x[3]+x[4])*2.6+x[5]*1.35+x[6]*.31+26.94
    wind_total=x[0]+x[1]+x[2]+x[3]+x[4]
    fixed_om=wind_total*.045+x[5]*.020+x[6]*.008+.119
    r=fast_objective(x,W,S,H,L,0.)
    # HVO price CAD 1.8/L; assume 18 ML B100 with 1.6/L, no exports, no grid
    hvo=r[5]*.0018
    out=capex*.085+fixed_om+hvo+18*.0016
    # strong gas and unserved penalty, annual gas max GWh; steady zero gas requirement
    return out+penalty*(r[1]+r[2]*10.0),out,r[1],r[2],r[3],r[4],r[5],capex

# sample subset climate extremes from 2010-25; 4 years for search, validate all 16
sel=[2012,2015,2020,2025];ind=[i for i,g in enumerate(groups) if g['year'] in sel]
ws,ss,hs,ls=W[ind],S[ind],H[ind],L[ind]
bounds=[(0,2)]*5+[(0,80),(5,160)]
print('optimizing full dispatch on four training years',flush=True)
res=differential_evolution(lambda x:score(np.asarray(x,dtype=np.float64),ws,ss,hs,ls,.30)[0],bounds,
  maxiter=25,popsize=7,seed=20261029,polish=False,updating='immediate',workers=1,tol=.006)
print('search complete',np.round(res.x,3),flush=True)
# local neighborhood perturbations: widen checked feasibility, plus prior unoptimized baseline and conservative solar
candidate=[('search_result',res.x),('published_50GW_screen',np.array([1.8776680892225666,1.304012949275942,1.7420588634193999,1.9345211864728717,1.4852965820686421,50.,66.41339673731622])),('conservative_10GW_reference',np.array([2,2,2,2,2,47.355843316,108.70785546])),
 ('solar_40_batt_110',np.array([2,2,2,2,2,40,110.])),('solar_45_batt_110',np.array([2,2,2,2,2,45,110.])),
 ('solar_47_batt_110',np.array([2,2,2,2,2,47,110.])),('solar_50_batt_120',np.array([2,2,2,2,2,50,120.])),
 ('solar_40_batt_140',np.array([2,2,2,2,2,40,140.])),('solar_45_batt_140',np.array([2,2,2,2,2,45,140.])),
 ('solar_35_batt_140',np.array([2,2,2,2,2,35,140.])),('solar_30_batt_140',np.array([2,2,2,2,2,30,140.])),
 ('solar_60_batt_80',np.array([1.8776680892225666,1.304012949275942,1.7420588634193999,1.9345211864728717,1.4852965820686421,60.,80.])),('solar_70_batt_75',np.array([1.8776680892225666,1.304012949275942,1.7420588634193999,1.9345211864728717,1.4852965820686421,70.,75.])),('solar_80_batt_70',np.array([1.8776680892225666,1.304012949275942,1.7420588634193999,1.9345211864728717,1.4852965820686421,80.,70.])),('solar_50_batt_100',np.array([2,2,2,2,2,50,100.])),('solar_40_batt_160',np.array([2,2,2,2,2,40,160.]))]
# search complement: perturb scale globally to see cost before full verification
for sw in [.85,1.,1.15]:
 for ssf in [.85,1.,1.1]:
  for sb in [.80,1,1.2]:
   xc=res.x.copy();xc[:5]=np.minimum(xc[:5]*sw,2);xc[5]=min(xc[5]*ssf,80);xc[6]=min(xc[6]*sb,160)
   candidate.append((f'search_grid_{sw}_{ssf}_{sb}',xc))
scored=[]
for name,x in candidate:
 r=score(np.asarray(x,float),W,S,H,L,0.)
 scored.append((name,np.array(x),r))
# qualified no gas full all 16; choose min annual cost, else minimize penalty
qualified=[s for s in scored if s[2][2]<1e-5 and s[2][3]<1e-5]
chosen=min(qualified,key=lambda s:s[2][1]) if qualified else min(scored,key=lambda s:s[2][2]+10*s[2][3])
print('qualified',len(qualified),'chosen',chosen[0],np.round(chosen[1],3),chosen[2],flush=True)
rows=[];annual=[]
# validate several portfolio screens full years across all wind modes, 0 award
shortlist=[('expanded_screen',chosen[1],None),('solar_50GW_sensitivity',candidate[1][1],None),('conservative_10GW',np.array([2,2,2,2,2,47.355843316,108.70785546]),None)]
for name,x,_ in shortlist:
 for mode,cfmat in windcache.items():
  for g in groups:g['windcf']=cfmat[:,mask_by_year[g['year']]]
  ys=evaluate(x,groups,grid=0,return_years=True)
  ys.insert(0,'case',name);ys.insert(1,'wind_mode',mode);annual.append(ys)
  capex=float(sum(x[:5])*2.6+x[5]*1.35+x[6]*.31+26.94)
  fixed=sum(x[:5])*.045+x[5]*.020+x[6]*.008+.119
  hvo=ys.hvo_ML.mean()*.0018
  b100=.0288
  gas_fuel_CAD_B=(ys.gas_GWh.mean()*1000)*(.90*3.6/.52*8+.10*3600/(36.51*.52)*1.8)/1e9
  annual_CAD_B=capex*.085+fixed+hvo+b100+gas_fuel_CAD_B
  cost=annual_CAD_B*1000/(ys.load_TWh.mean()-ys.unserved_GWh.mean()/1000)
  rows.append(dict(portfolio=name,wind_mode=mode,grid_import_MW=0,wind_GW=sum(x[:5]),solar_GW=x[5],battery_GWh=x[6],annual_load_TWh=ys.load_TWh.mean(),wind_gen_TWh=ys.wind_TWh.mean(),solar_gen_TWh=ys.solar_TWh.mean(),curtailment_TWh=ys.curtail_TWh.mean(),HVO_ML_mean=ys.hvo_ML.mean(),HVO_ML_max=ys.hvo_ML.max(),emergency_gas_GWh_mean=ys.gas_GWh.mean(),emergency_gas_GWh_max=ys.gas_GWh.max(),unserved_GWh_max=ys.unserved_GWh.max(),capex_CAD_B=capex,annual_cost_CAD_B=annual_CAD_B,emergency_gas_cost_CAD_B=gas_fuel_CAD_B,screen_cost_CAD_MWh=cost,zero_gas_all_16=bool((ys.gas_GWh<1e-4).all()),zero_unserved_all_16=bool((ys.unserved_GWh<1e-4).all())))
  print(name,mode,'cost',round(cost,2),'gasMax',round(ys.gas_GWh.max(),2),flush=True)
pd.DataFrame(rows).to_csv(OUT/'capacity_optimization_scenario_comparison.csv',index=False,float_format='%.8f')
pd.concat(annual).to_csv(OUT/'capacity_optimization_all_annual_results.csv',index=False,float_format='%.8f')
with open(OUT/'optimization_candidate_search.csv','w') as f:
 f.write('name,annual_CAD_B,max_gas_GWh,max_unserved_GWh,wind_GW,solar_GW,battery_GWh,capex_CAD_B\n')
 for name,x,r in scored:f.write(f'{name},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{sum(x[:5]):.6f},{x[5]:.6f},{x[6]:.6f},{r[-1]:.6f}\n')
(OUT/'optimization_provenance.json').write_text(json.dumps(dict(algorithm='scipy differential_evolution heuristic + expanded solar candidates, 4 training years, all 16 verification',not_a_global_proof=True,wind_benchmark_CF=ref_cf,wind_resource_status='unmeasured wind farm extrapolation',penalty='0.30 CAD billion per GWh maximum gas or 10x unserved on training years; final selection requires zero gas and unserved across all years',gas_case='gas backup assumed with no legal siting or fuel contract',grid='0 MW, no allocation',wind_region_capacity_ceilings_GW=[2]*5,solar_cap_GW=80,battery_cap_GWh=160,hvo_cap_ML_per_year=300,wind_selection=chosen[0],wind_GW=list(map(float,chosen[1][:5])),solar_GW=float(chosen[1][5]),battery_GWh=float(chosen[1][6]),constraints=['Pooled transmission not designed','HVO not procured','Solar land not mapped','No electrical N-1 system study','Food thermal hourly not validated'],related_public_project_repository='https://github.com/bcjohnjr/Digital-Gardens-vs.-Wonder-Valley',analysis_specific_publication_status='definitive analysis files supplied in the supplementary research archive; public repository is a related project resource',wind_regional_cap_qualification='Five 2-GW region limits restrict searched design space; selected portfolio totals 9.5665 GW of the 10 GW allowed. Caps are indicative; search not rerun beyond these caps.'),indent=2))
