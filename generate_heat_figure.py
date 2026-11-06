#!/usr/bin/env python3
"""Regenerate publication Figure 5 directly from the reproducible thermal CSV."""
from pathlib import Path
import pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
x=pd.read_csv(P/'OUTPUT/hourly_thermal_heat_matching_summary.csv')
x=x[x.transfer_efficiency==.85]
fig,ax=plt.subplots(figsize=(8,4.5))
ax.plot(x.heat_capture_fraction_IT*100,x.heat_delivered_TWh,marker='o',lw=2)
ax.axhline(4.819,ls='--',lw=1,color='gray',label='Modelled annual customer heat demand (4.819 TWh)')
ax.set_xlabel('Assumed source heat as % of IT electricity')
ax.set_ylabel('Demand-capped useful heat (TWh/year)')
ax.set_title('Hourly heat accounting; useful temperature grade unverified')
ax.legend(loc='lower right');ax.set_xlim(8,46);ax.set_ylim(0,5.15);fig.tight_layout()
fig.savefig(P.parent/'JOURNAL_UPLOAD/FIGURES/Figure_5_Hourly_Thermal_Matching.png',dpi=300)
plt.close(fig)
