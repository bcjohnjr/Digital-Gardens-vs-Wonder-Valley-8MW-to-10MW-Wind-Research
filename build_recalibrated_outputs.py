#!/usr/bin/env python3
"""Reconstruct model scenario snapshot from explicit GWA-corrected 16-year screening grid.

This does not assert site validation. Assumptions and limitations are in integrated summary.
Run from RESEARCH_CODE directory after `recalibrate_gwa_and_audit.py`.
"""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'MODEL_CORE'))
import storage_fuel_sweetspot as sf
from land_constrained_optimizer import make_data
from dispatch_reference import HVO_L_PER_MWH

OUT=ROOT/'OUTPUT'

def make_values(cap,sol,batt,fuel,groups):
    st=sf.run_case(groups,cap,sol,batt,fuel)
    cost=sf.cost_25y(cap.sum(),sol,batt,st,fuel)
    return dict(wind_GW=float(cap.sum()),wind_scale=float(cap.sum()/np.asarray(json.loads((OUT/'optimization_provenance.json').read_text())['wind_GW']).sum()),wind_GW_per_site=cap.tolist(),solar_GW=float(sol),
                battery_GWh=float(batt),fuel_cap_ML=float(fuel),qualifies=bool(st['gas_gwh_max']<1e-3 and st['unserved_gwh_max']<1e-3),
                **st,**cost)

def main():
    grid=pd.read_csv(HERE/'OUTPUT/GWA_16year_storage_fuel_candidate_grid.csv')
    q=grid[grid.gas_free_all_years].copy()
    if q.empty:raise SystemExit('No qualifying candidates; not replacing scenario JSON')
    provenance=json.loads((OUT/'optimization_provenance.json').read_text())
    base=np.asarray(provenance['wind_GW'])
    groups,_=make_data(ROOT/'DATA',ROOT/'DATA/RIDGE_WIND_ERA5_6_POINTS_2010_2025.zip')
    g,_=sf.load_years()
    assert len(g)==16
    farm=sf.P['farm_hvo_ml']; lim=sf.P['supply_cap_ml']-farm
    def best(f):
        z=q[np.isclose(q.hvo_cap_ml,f)].sort_values('lcoe_25y_cad_mwh')
        assert not z.empty, f
        r=z.iloc[0];cap=np.minimum(base*r.wind_multiplier,2.)
        return make_values(cap,r.solar_gw,r.battery_gwh,r.hvo_cap_ml,g)
    selected=best(lim); ref=best(300.); low=best(700.)
    assert selected['gas_gwh_max']<.001 and ref['gas_gwh_max']<.001
    assert selected['hvo_ml_max']+farm<=sf.P['supply_cap_ml']+.0001
    # keep a bona fide zero-gas comparator at 300 ML rather than the previous infeasible one
    gg30,_=sf.load_years(.30)
    ws=np.asarray(selected['wind_GW_per_site'])
    st30=sf.run_case(gg30,ws,selected['solar_GW'],selected['battery_GWh'],lim)
    # Support code historically expects some of the old top-level keys.
    trade=[]
    for fuel in sorted(q.hvo_cap_ml.unique()):
        r=q[q.hvo_cap_ml==fuel].sort_values('lcoe_25y_cad_mwh').iloc[0]
        trade.append(dict(fuel_cap_ML=float(fuel),battery_GWh=float(r.battery_gwh),solar_GW=float(r.solar_gw),
                          wind_GW=float(r.wind_gw),lcoe_25y_cad_mwh=float(r.lcoe_25y_cad_mwh),
                          hvo_ml_mean=float(r.hvo_ml_mean),hvo_ml_max=float(r.hvo_ml_max),
                          fuel_lifecycle_mt=float(r.hvo_ml_mean*1e6*sf.P['hvo_lhv_mj_l']*sf.P['hvo_ci_g_mj']/1e12),
                          capex_bn=float(r.capex_cad_bn)))
    out=dict(reference_selected=ref,sweet_spot=selected,low_fuel_sweet_spot=low,
             tradeoff_curve=trade,sweet_spot_at_30pct_wind=st30,
             n_cases=len(grid),n_qualifying=len(q),
             data_centre_fuel_allowance_ml=lim,
             wind_benchmark_cf=json.loads((HERE/'OUTPUT/GWA_BC_AESO_integrated_summary.json').read_text())['wind_capacity_factor_selected_capacity_weighted'],
             scenario_status='CONDITIONAL MODEL SCREEN - NOT SITE VALIDATED',
             reference_definition='Re-optimised qualifying 300ML annual HVO cap design, not the earlier unqualified legacy reference',
             params=sf.P)
    (OUT/'storage_fuel_sweetspot.json').write_text(json.dumps(out,indent=2)+'\n')
    pd.DataFrame(trade).to_csv(OUT/'storage_fuel_sweetspot_grid.csv',index=False)
    print('stored gas free candidate',selected['lcoe_25y_cad_mwh'],'HVO',selected['hvo_ml_mean'],selected['hvo_ml_max'])
    print('reference qualifying',ref['lcoe_25y_cad_mwh'],'HVO',ref['hvo_ml_mean'],ref['hvo_ml_max'])
    print('GWA-relative savings percent',100*(1-selected['lcoe_25y_cad_mwh']/ref['lcoe_25y_cad_mwh']))

if __name__=='__main__':main()
