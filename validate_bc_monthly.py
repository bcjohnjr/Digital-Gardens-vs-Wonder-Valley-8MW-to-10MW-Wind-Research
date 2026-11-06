#!/usr/bin/env python3
"""Observed BC monthly shapes vs modeled 8MW and 10MW portfolio (NOT site-specific validation)."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));from model_8mw_10mw import build_inputs
obs=pd.read_csv(HERE.parent/'OBSERVATIONAL_VALIDATION/INPUT/BC_monthly_observed_wind_generation_2016_2026.csv')
obs['month']=pd.to_datetime(obs['month']);obs=obs[obs.month.dt.year.between(2016,2025)].copy();obs['year']=obs.month.dt.year
obs['observed_normalized']=obs.observed_wind_generation_MWh/obs.groupby('year').observed_wind_generation_MWh.transform('mean')
for name in ('SANY_SI19580_8MW','SANY_SI230100_10MW'):
 a=json.loads((HERE/'OUTPUT'/(name+'_optimization.json')).read_text());wg=np.array(list(a['candidate']['wind_gw_by_site'].values()))
 groups,*_=build_inputs(name,.4)
 months=[]
 for g in groups:
  if g['year'] not in range(2016,2026):continue
  hourly=wg@g['windcf']
  d=pd.DataFrame({'month':pd.date_range(f'{g["year"]}-01-01',periods=8760,freq='h').to_period('M').to_timestamp(),'cf':hourly/wg.sum()})
  months.append(d.groupby('month',as_index=False).cf.mean())
 mod=pd.concat(months);mod['year']=mod.month.dt.year;mod['model_normalized']=mod.cf/mod.groupby('year').cf.transform('mean')
 joint=obs.merge(mod,on=['month','year'],validate='one_to_one')
 out={'matched_months':len(joint),'all_2016_2025_r':float(joint.observed_normalized.corr(joint.model_normalized)),
      'excluding_2020_r':float(joint[joint.year!=2020].observed_normalized.corr(joint[joint.year!=2020].model_normalized)),
      'note':'Province-wide wind output divided by each year mean vs scaled ridge model; not project-level or capacity-corrected validation.'}
 (HERE/'OUTPUT'/(name+'_BC_monthly_validation.json')).write_text(json.dumps(out,indent=2)+'\n')
 joint.to_csv(HERE/'OUTPUT'/(name+'_BC_monthly_comparison.csv'),index=False)
 print(name,out)
