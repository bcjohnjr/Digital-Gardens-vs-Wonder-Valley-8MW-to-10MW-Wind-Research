#!/usr/bin/env python3
"""Gas electricity-service sensitivity, consistent CCS fuel and CO2 mass accounting.
Non-inventory upstream 20% adder, not a certified carbon price or Wonder Valley report.
"""
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parent/'OUTPUT';P.mkdir(exist_ok=True)
MWh=36.00486e6; eta=.57; co2_per_GJ=.0561; co2_per_MWh=(3.6/eta)*co2_per_GJ
rows=[]
for p in [4,8,12]:
 base=9e9*.085/MWh+6000*25000/MWh+12+(3.6/eta)*p
 for charge in [0,50,100]:
  for capture in [0,.9]:
   penalty=(.15 if capture else 0.)
   burn=co2_per_MWh*(1+penalty)
   captured=burn*capture
   residual=burn*(1-capture)
   extra_gas=(3.6/eta)*p*penalty
   variable_ccs=captured*75 if capture else 0.
   row=dict(case=('90% combustion CCS sensitivity (partial cost only)' if capture else 'unabated gas'),gas_CAD_per_GJ=p,carbon_charge_CAD_per_tCO2=charge,gas_power_CAD_per_MWh=base,carbon_charge_CAD_per_MWh=residual*charge,total_CAD_per_MWh_excluding_pipeline_heat=base+extra_gas+variable_ccs+residual*charge,combustion_CO2_Mt=MWh*residual/1e6,upstream_proxy_Mt=.20*MWh*burn/1e6,combustion_capture_fraction=capture,CCS_included='variable capture OPEX + fuel penalty only' if capture else False,additional_fuel_fraction=penalty,CO2_captured_Mt=MWh*captured/1e6,pre_capture_combustion_CO2_Mt=MWh*burn/1e6,CCS_CAPEX_included=False,pipeline_CAPEX_included=False)
   assert abs(row['combustion_CO2_Mt']+row['CO2_captured_Mt']-row['pre_capture_combustion_CO2_Mt'])<1e-9
   rows.append(row)
pd.DataFrame(rows).to_csv(P/'gas_and_carbon_comparison_screen.csv',index=False)
print('Gas comparison rows:',len(rows))
