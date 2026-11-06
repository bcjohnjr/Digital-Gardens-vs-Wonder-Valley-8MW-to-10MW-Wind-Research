#!/usr/bin/env python3
"""Independently search same 8-MW turbine screen at 300 ML data-centre HVO allowance."""
from model_8mw_10mw import build_inputs,run_case_fast,summarize,TURBINES
from pathlib import Path
from scipy.optimize import differential_evolution
import storage_fuel_sweetspot as sf
import numpy as np,json
name='SANY_SI19580_8MW'; m=TURBINES[name]
groups,W,S,H,L,site=build_inputs(name, .4)
count=0
bounds=[(0,2)]*2+[(0,.2)]+[(0,2)]*2+[(10,65),(10,120)]
def obj(x):
 global count
 count+=1
 full=np.r_[x,300.]
 r=run_case_fast(full,W,S,H,L)
 st={'hvo_ml_mean':r[0],'hvo_ml_max':r[1],'gas_gwh_max':r[2],'unserved_gwh_max':r[3],'curtail_twh':r[6],'gen_twh':r[4],'load_twh':r[5]}
 return sf.cost_25y(x[:5].sum(),x[5],x[6],st,300.)['lcoe_25y_cad_mwh']+3*r[2]+45*r[3]+5*max(0,r[1]-300)
res=differential_evolution(obj,bounds,seed=456,maxiter=45,popsize=8,polish=False,updating='immediate',tol=.004,workers=1)
x=np.r_[res.x,300.]
c=summarize(x,W,S,H,L,groups,m)
out={'turbine':name,'fuel_cap_ml':300,'search':{'evaluations':count,'success':bool(res.success),'message':str(res.message)},'candidate':c}
p=Path(__file__).resolve().parent/'OUTPUT'/'SANY_SI19580_8MW_300ML_comparator.json';p.write_text(json.dumps(out,indent=2)+'\n')
print('300ML comparator',c['cost']['lcoe_25y_cad_mwh'],c['qualified_dispatch'],c['stats'],flush=True)
