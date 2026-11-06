#!/usr/bin/env python3
"""Recompute only central contracted drying with active 8MW hourly portfolio.
Does not pretend unrelated high-scale sensitivity cases were rerun.
"""
import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
import industrial_drying as dr
import profit_community_model as pcm
import pandas as pd
T=pcm.load_temps()
surplus,served,months,_=dr.base_heat(T)
curtail=dr.curtailment_series()
assert len(curtail)==len(months)
d=dr.scenario('Central contracted drying with 8MW wind portfolio',surplus,months,curtail,served,0,1.,1.)
O=ROOT/'OUTPUT'
file=O/'industrial_drying_results.json'
old=json.loads(file.read_text())
old['central']=d
old['curtailment_twh']=float(curtail.sum()/1e6/(len(months)/8760))
old['noncentral_scenarios_status']='Archived reference scenarios not recalculated for SANY 8MW and must not be compared to the updated central.'
file.write_text(json.dumps(old,indent=1,default=float)+'\n')
pd.DataFrame(d['markets']).to_csv(O/'industrial_drying_markets_central.csv',index=False)
(HERE/'OUTPUT/8MW_central_drying.json').write_text(json.dumps({k:v for k,v in d.items() if k!='electricity'},indent=2,default=float)+'\n')
print('CENTRAL_DRYING',d['utilisation_pct'],d['erf_pct'],d['economics']['water_mt'],d['economics']['capex_bn'],d['economics']['margin_m_cad_mean'],flush=True)
