#!/usr/bin/env python3
"""Figures for independently bounded large-turbine portfolio screens."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'OUTPUT';P=ROOT.parent.parent/'FIGURES';P.mkdir(exist_ok=True)
a=json.loads((OUT/'SANY_SI19580_8MW_optimization.json').read_text());b=json.loads((OUT/'SANY_SI230100_10MW_optimization.json').read_text())
plt.rcParams.update({'font.size':9,'figure.dpi':160,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False})
names=['Pennask','Mount Mabel','Nicola','Cache Creek','Highland Valley']
fig,axs=plt.subplots(1,2,figsize=(9,3.2),gridspec_kw={'width_ratios':[1.15,1.0]})
x=np.arange(5)
axs[0].bar(x-.18,list(a['candidate']['wind_gw_by_site'].values()),.36,label='SANY 8 MW')
axs[0].bar(x+.18,list(b['candidate']['wind_gw_by_site'].values()),.36,label='SANY 10 MW')
axs[0].set_xticks(x,names,rotation=20,ha='right');axs[0].set_ylabel('Wind capacity (GW)');axs[0].set_ylim(0,2.25)
axs[0].set_title('a  Site-free allocations; Nicola ≤0.2 GW');axs[0].legend(fontsize=8,frameon=False)
keys=['central_40pct','cap35_same_installation','GWA90pct_cap40','raw_ERA5'];labs=['40% cap\ncentral','35% cap\nsame build','Atlas means\n−10%','Raw ERA5']
y1=np.array([a['stress'][k]['gas_gwh_max'] for k in keys])/1000;y2=np.array([b['stress'][k]['gas_gwh_max'] for k in keys])/1000
x=np.arange(4)
axs[1].bar(x-.18,y1,.36,label='8 MW');axs[1].bar(x+.18,y2,.36,label='10 MW')
axs[1].set_xticks(x,labs,fontsize=8);axs[1].set_ylabel('Worst-year emergency gas (TWh)');axs[1].set_title('b  Selected designs: 16-year weather replay')
axs[1].legend(fontsize=8,frameon=False)
fig.suptitle('Digital Gardens vs. Wonder Valley — large onshore turbine screening',fontsize=11,fontweight='bold')
fig.tight_layout()
for root in [P,ROOT.parent/'FIGURES_EXTENSION']:
 fig.savefig(root/'Figure_3_wind_sensitivity.png');fig.savefig(root/'Figure_3_wind_sensitivity.pdf')
plt.close(fig)

fig,ax=plt.subplots(figsize=(6.4,3.35))
temp=[10.5,11.0,11.5,12.0]
vals=[b['stress'].get(f'10MW_assumed_rated_{s:.1f}ms',b['stress']['central_40pct'])['gas_gwh_max'] for s in temp]
ax.bar([str(s) for s in temp],vals)
for i,v in enumerate(vals):ax.text(i,v+15,f'{v:.0f} GWh',ha='center',fontsize=8)
ax.set_title('10 MW turbine: assumed rated-speed uncertainty (same installation)')
ax.set_ylabel('Worst-year emergency gas (GWh)')
ax.set_xlabel('Assumed rated wind speed (m/s): NOT manufacturer-certified')
ax.set_ylim(0,max(vals)*1.25)
fig.tight_layout()
fig.savefig(ROOT/'Supplemental_Figure_S5_10MW_RatedSpeed.png');fig.savefig(ROOT/'Supplemental_Figure_S5_10MW_RatedSpeed.pdf');plt.close(fig)
print('Created Figures 3 and S5.')
