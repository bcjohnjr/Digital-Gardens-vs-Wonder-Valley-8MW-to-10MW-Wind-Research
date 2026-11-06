#!/usr/bin/env python3
"""Build physically transparent site-specific figures from generated constrained outputs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd,numpy as np
HERE=Path(__file__).resolve().parent;R=HERE.parent;F=R/'FIGURES_EXTENSION';F.mkdir(exist_ok=True)
I=HERE/'OUTPUT'
read=lambda n:json.loads((I/n).read_text())
a=read('site_free_cf40_optimization.json');b=read('site_free_cf40_fuel300_optimization.json');c=read('site_free_cf40_fuel600_optimization.json');s=read('site_free_16year_dispatch_summary.json');v=read('site_free_bc_monthly_validation.json')
site_order=['pennask_mountain_summit','mount_mabel_summit','cache_wind_northern_area_proxy','highland_valley_wind_project_area','nicola_wind_project_regional_proxy']
names=['Pennask','Mt Mabel','Cache*','Highland','Nicola*']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'savefig.dpi':300})

def save(fig,name):
 fig.savefig(F/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(8.3,3.8),gridspec_kw={'width_ratios':[1.,1.15]})
fuels=[300,600,a['fuel_limit_ml']];costs=[b['cost']['lcoe_25y_cad_mwh'],c['cost']['lcoe_25y_cad_mwh'],a['cost']['lcoe_25y_cad_mwh']]
axs[0].plot(fuels,costs,'o-',color='#1e4265',lw=2)
for f,cv in zip(fuels,costs):axs[0].annotate(f'C${cv:.0f}',(f,cv),xytext=(5,8),textcoords='offset points',fontsize=9)
axs[0].set(xlabel='Maximum data-centre HVO (million L/yr)',ylabel='Modelled 25-year electricity cost (C$/MWh)',title='a  Site-free fuel-allocation solutions')
axs[0].grid(alpha=.15)
x=np.arange(3);bw=.24
for j,(k,label) in enumerate([('total_wind_gw','Wind (GW)'),('solar_gw','Solar (GW)'),('battery_gwh','Battery (GWh)')]):
 vals=[t[k] for t in [b,c,a]]
 axs[1].bar(x+(j-1)*bw,vals,bw,label=label)
axs[1].set_xticks(x,['300 ML','600 ML',f'{a["fuel_limit_ml"]:.0f} ML'])
axs[1].set(title='b  Independently selected capacities',ylabel='Installed GW or GWh')
axs[1].legend(frameon=False,fontsize=8)
fig.text(.03,.01,'Each candidate has independent site capacities (0–2 GW per site); Pennask 16-year modelled CF capped at 40%.\nAll results are scenario optimisations; no transmission power-flow, permitting, or site mast verification.',fontsize=7,color='#555')
fig.tight_layout(rect=[0,.09,1,1]);save(fig,'Figure_2_storage_fuel_sweet_spot')
fig,axs=plt.subplots(1,3,figsize=(10.0,3.7),gridspec_kw={'width_ratios':[1.18,1.0,.8]})
xx=np.arange(len(names));barw=.38
pre=[a['wind_cf_before_ceiling'][q]*100 for q in site_order]
post=[a['wind_cf_after_ceiling'][q]*100 for q in site_order]
axs[0].bar(xx-barw/2,pre,barw,color='#8c98aa',label='GWA mean corrected')
axs[0].bar(xx+barw/2,post,barw,color='#3877aa',label='40%-ceiling screen')
axs[0].axhline(40,color='#bc594f',ls='--',lw=1)
axs[0].set_xticks(xx,names,rotation=27,ha='right');axs[0].set(ylabel='Modelled site wind CF (%)',title='a  Site-capacity-factor screen')
axs[0].legend(fontsize=6.8,frameon=False)
cap=[a['wind_gw_per_site'][q] for q in site_order]
axs[1].bar(names,cap,color='#4a9366');axs[1].tick_params(axis='x',rotation=27)
axs[1].set(title='b  Free wind capacity allocation',ylabel='Installed wind capacity (GW)',ylim=(0,2.25))
for t,z in enumerate(cap):axs[1].text(t,z+.06,f'{z:.2f}',ha='center',fontsize=8)
rows=s['cases'];worst=[rows[k]['gas_gwh_max']/1000 for k in ['base40','cap35_same_design','raw_era5']]
axs[2].bar(['40%-cap','35%-cap','Raw ERA5'],worst,color=['#40855e','#da9a34','#b54b4d'])
axs[2].set(ylabel='Worst-year emergency fossil gas (TWh)',title='c  Same selected portfolio')
axs[2].tick_params(axis='x',rotation=24)
for t,z in enumerate(worst):axs[2].text(t,z+.15,f'{z:.2f}',ha='center',fontsize=8)
fig.text(.02,.01,f'Annual BC monthly-shape correlation: r={v["all_2016_2025"]:.2f}; excluding anomalous 2020: r={v["excluding_2020"]:.2f}.\n*Region-scale site proxies; hourly ERA5 shape is not independently validated. 35%-cap stress applies to 40%-cap design.',fontsize=7,color='#555')
fig.tight_layout(rect=[0,.09,1,1]);save(fig,'Figure_3_wind_sensitivity')
print('Saved Figures 2 and 3')
