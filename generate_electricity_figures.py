#!/usr/bin/env python3
"""Regenerate published electricity cost, wind-sensitivity, and generation figures.

Run from any directory with: python generate_electricity_figures.py
Outputs go to ../JOURNAL_UPLOAD/FIGURES relative to RESEARCH_CODE/.
Figure 1 is a strictly matched 34.79%-wind-factor comparison; the 30% test
appears separately in the results tables and the wind uncertainty figure.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
F=R.parent/'JOURNAL_UPLOAD'/'FIGURES';F.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(R/'OUTPUT'/'capacity_optimization_scenario_comparison.csv')
central='BC_Hydro_fleet_expected_34p79pct'
def get(a,b):
 q=d[(d.portfolio==a)&(d.wind_mode==b)]
 assert len(q)==1,(a,b)
 return q.iloc[0]
a=get('expanded_screen',central)
b=get('solar_50GW_sensitivity',central)
c=get('conservative_10GW',central)
fig,ax=plt.subplots(figsize=(9.2,4.4))
labels=['Expanded selected\n9.57 GW wind','50-GW solar test\n8.34 GW wind','10-GW wind reference\n10.00 GW wind']
vals=[a.screen_cost_CAD_MWh,b.screen_cost_CAD_MWh,c.screen_cost_CAD_MWh]
bars=ax.bar(np.arange(3),vals,width=.58)
ax.set_ylim(0,475);ax.set_ylabel('Screened electricity cost (C$/MWh delivered)')
ax.set_title('Matched 34.79% wind-factor comparison: zero modeled gas (16 years)',pad=14)
ax.set_xticks(np.arange(3),labels)
for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+6,f'{v:.2f}',ha='center',fontsize=11,fontweight='bold')
ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
ax.text(.5,-.19,'All three cases use the same modeled wind factor. Land, fuel and grid siting remain unverified.',transform=ax.transAxes,ha='center',fontsize=9)
fig.subplots_adjust(bottom=.23,top=.89,left=.11,right=.98)
fig.savefig(F/'Figure_1_Cost_Screens.png',dpi=240);plt.close(fig)
fig,ax=plt.subplots(figsize=(9.3,4.5));scens=['raw_ERA5','assumed_30pct',central,'assumed_40pct'];names=['Raw ERA5','30% fleet','34.8% fleet','40% fleet'];idx=np.arange(4);bw=.34
for i,(case,label) in enumerate([('expanded_screen','Expanded screen'),('conservative_10GW','10-GW reference')]):
 vals=[get(case,m).emergency_gas_GWh_max/1000 for m in scens]
 ax.bar(idx+(i-.5)*bw,vals,bw,label=label)
ax.set_xticks(idx,names);ax.set_ylabel('Maximum gas generation in a historical year (TWh)');ax.set_title('Wind yield uncertainty: all cases have zero contracted grid supply');ax.legend();ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(F/'Figure_2_Wind_Uncertainty_Gas.png',dpi=250);plt.close(fig)
fig,ax=plt.subplots(figsize=(9.3,4.5));cases=[get(k,central) for k in ('expanded_screen','solar_50GW_sensitivity','conservative_10GW')];labs=['Expanded screen','50-GW solar test','10-GW wind test'];t=np.arange(3)
w=np.array([p.wind_gen_TWh for p in cases]);s=np.array([p.solar_gen_TWh for p in cases]);cu=np.array([p.curtailment_TWh for p in cases]);bw=.34
ax.bar(t-bw/2,w,bw,label='Wind generation');ax.bar(t-bw/2,s,bw,bottom=w,label='Solar generation');ax.bar(t+bw/2,cu,bw,label='Curtailment (part of generation)',color='lightgray',edgecolor='black',hatch='///')
ax.set_ylabel('Mean annual energy (TWh)');ax.set_xticks(t,labs);ax.set_title('Gross renewable output and curtailed share at 34.79% wind yield');ax.legend(loc='upper left');ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(F/'Figure_3_Generation_Curtailment.png',dpi=250);plt.close(fig)
print('Regenerated Figures 1–3 from archived numerical results.')
