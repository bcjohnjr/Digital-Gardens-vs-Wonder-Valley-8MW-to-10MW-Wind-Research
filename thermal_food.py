#!/usr/bin/env python3
"""Hourly heat accounting: customer-useful heat = min(eta*source,demand).
Directly runnable without the preserved archive. All site-grade limitations remain.
"""
from pathlib import Path
import numpy as np, pandas as pd
OUT=Path(__file__).resolve().parent/'OUTPUT'; OUT.mkdir(exist_ok=True)
t=pd.date_range('2025-01-01',periods=8760,freq='h');months=t.month.to_numpy();hours=t.hour.to_numpy()
season=np.array([1.6,1.55,1.35,1.12,.85,.55,.43,.48,.70,1.03,1.35,1.6])[months-1].astype(float)
season*=1+.08*np.cos((hours-5)*2*np.pi/24);season/=season.mean()
demand=4.819e6/8760*season
it=32.861982366e6/8760
result=[];hourly=[]
for capture in [.10,.15,.25,14.57/32.861982366]:
 for eta in [.65,.85]:
  available=np.full(8760,it*capture)
  supplied=available*eta
  delivered=np.minimum(supplied,demand)
  label=f'capture_{capture:.3f}_eta_{eta:.2f}'
  assert np.all(delivered<=demand+1e-8)
  assert np.all(delivered<=supplied+1e-8)
  assert np.isclose((demand-delivered).sum()+delivered.sum(),demand.sum())
  result.append(dict(case=label,IT_electricity_TWh=32.861982366,thermal_need_TWh=4.819,heat_capture_fraction_IT=capture,available_server_heat_TWh=available.sum()/1e6,transfer_efficiency=eta,heat_delivered_TWh=delivered.sum()/1e6,heat_demand_coverage_pct=100*delivered.sum()/demand.sum(),peak_thermal_demand_MW=demand.max(),annual_unmet_at_demand_MWh=(demand-delivered).sum(),is_matched_hourly=True,heat_network_siting_validated=False,heat_grade_validated=False,accounting_rule='min(eta * available_server_heat, customer_heat_demand)'))
  if eta==.85: hourly.append(pd.DataFrame(dict(timestamp=t.astype(str),case=label,heat_demand_MWh=demand,heat_available_MWh=available,delivered_MWh=delivered,source_after_losses_MWh=supplied)))
pd.DataFrame(result).to_csv(OUT/'hourly_thermal_heat_matching_summary.csv',index=False)
pd.concat(hourly).to_csv(OUT/'hourly_heat_matching_2025_scenarios.csv.gz',index=False,compression='gzip')
print(pd.DataFrame(result)[['case','heat_delivered_TWh','heat_demand_coverage_pct']].to_string(index=False))
