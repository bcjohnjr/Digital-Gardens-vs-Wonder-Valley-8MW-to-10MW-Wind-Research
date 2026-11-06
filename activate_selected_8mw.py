#!/usr/bin/env python3
"""Synchronize the 8MW scenario with common heat, NOx and financial inputs.
Run once before downstream codes (a file marker selects the 8MW power proxy).
Does not change fixed load, heat capture assumptions, gas comparator or BC public records.
"""
from pathlib import Path
import json
import numpy as np
import sys
HERE=Path(__file__).resolve().parent;RC=HERE.parent;O=RC/'OUTPUT'
sys.path.insert(0,str(RC));import storage_fuel_sweetspot as sf
new=json.loads((HERE/'OUTPUT/SANY_SI19580_8MW_optimization.json').read_text())['candidate']
ref=json.loads((HERE/'OUTPUT/SANY_SI19580_8MW_300ML_comparator.json').read_text())['candidate']
old=json.loads((O/'storage_fuel_sweetspot.json').read_text())

def convert(c):
    s=c['stats'];co=sf.cost_25y(c['wind_gw'],c['solar_gw'],c['battery_gwh'],s,c['fuel_cap_ml'])
    out={ 'wind_GW':c['wind_gw'],'wind_GW_per_site':list(c['wind_gw_by_site'].values()),
       'solar_GW':c['solar_gw'],'battery_GWh':c['battery_gwh'],'fuel_cap_ML':c['fuel_cap_ml'],
       'qualifies':c['qualified_dispatch'],**s,**co}
    return out
nw=convert(new);rf=convert(ref)
cf=json.loads((HERE/'OUTPUT/SANY_SI19580_8MW_optimization.json').read_text())['sites'];cw=np.array(new['wind_gw_by_site'].values() if False else list(new['wind_gw_by_site'].values()));windcf=float(np.dot(cw,[s['screened_cf'] for s in cf])/sum(cw))
old['reference_selected']=rf;old['sweet_spot']=nw;old['low_fuel_sweet_spot']=rf
old['tradeoff_curve']=[{k:row[k] for k in ['fuel_cap_ML','battery_GWh','solar_GW','wind_GW','lcoe_25y_cad_mwh','hvo_ml_mean','hvo_ml_max','fuel_lifecycle_mt','capex_bn']} for row in (rf,nw)]
old['wind_benchmark_cf']=windcf;old['data_centre_fuel_allowance_ml']=sf.P['supply_cap_ml']-sf.P['farm_hvo_ml']
old['n_cases']=sum(json.loads((HERE/'OUTPUT'/(n+'_optimization.json')).read_text())['search']['evaluations'] for n in ['SANY_SI19580_8MW','SANY_SI230100_10MW'])
old['n_qualifying']=2;old['sweet_spot_at_30pct_wind']=None
old['scenario_status']='SANY 8MW main portfolio with manufacturer-stated rated wind speed, but engineering proxy power curve; 10MW scenario separate; all hypothetical'
old['optimization_not_global_proof']=True
old['per_site_wind_caps']='<=2.0 GW/site and <=0.2 GW at Nicola proxy, unverified siting'
(O/'storage_fuel_sweetspot.json').write_text(json.dumps(old,indent=2)+'\n')
(O/'optimization_provenance.json').write_text(json.dumps({'algorithm':'independent bounded site-capacity differential-evolution screening','active_turbine':'SANY SI-19580 8 MW engineering power-curve proxy','wind_GW':nw['wind_GW_per_site'],'solar_GW':nw['solar_GW'],'battery_GWh':nw['battery_GWh'],'nicola_cap_gw':0.2,'maximum_ridge_cap_gw':2.,'not_independent_validation':True},indent=2)+'\n')
(HERE/'ACTIVE_8MW.json').write_text(json.dumps({'active_turbine':'SANY_SI19580_8MW','cf_cap':.4,'hub_height_data_m':100,'power_curve_certified':False},indent=2)+'\n')
print('ACTIVE 8MW',nw['wind_GW'],nw['lcoe_25y_cad_mwh'],'300ML',rf['lcoe_25y_cad_mwh'],'weighted_cf',windcf)
