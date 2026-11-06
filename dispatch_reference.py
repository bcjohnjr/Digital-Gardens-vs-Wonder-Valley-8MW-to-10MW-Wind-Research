"""Prospective 17-site portfolio screen from uploaded ERA5 reanalysis.

ERA5 wind at 5 distinct mountainous grid cells is NOT a turbine-level wind measurement.
Ridge speed-up multipliers are unvalidated sensitivities; not a forecast or entitlement.
BC Hydro 300 MW is competitive province-wide limit, not project allocation;
$74/MWh is 2024 awarded projects' average, NOT spot-price or export offer.
This is a pooled-transmission UPPER BOUND requiring transmission development.
"""
from pathlib import Path
import argparse, io, json, gzip, zipfile, itertools, math, hashlib
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent
DEFAULT_BASE=Path(__file__).resolve().parent.parent/'DATA'
PICK=['pennask_mountain_summit','mount_mabel_summit','nicola_wind_project_regional_proxy','cache_wind_northern_area_proxy','highland_valley_wind_project_area']
WEIGHTS=np.array([.35,.25,.18,.12,.10]); assert abs(sum(WEIGHTS)-1)<1e-10
N=17; MW_DEMAND=36004.86*1000/8760
GRID_CAP=300.; HYDRO=8.205; HYDRO_CF=.45
SOLAR_PF=.82; LOSS_WIND=.06; LOSS_SOLAR=.03
WINDFARM_AVAIL=.85
B100_OIL_L=20e6; B100_YIELD=.90; B100_LHV=35.4; HVO_LHV=36.51; ENG_NET_EFF=.54
B100_MWh=B100_OIL_L*B100_YIELD*B100_LHV/3600*ENG_NET_EFF
HVO_L_PER_MWH=3600/(HVO_LHV*ENG_NET_EFF)
HVO_COST_CAD_L=1.8
BATTERY_CHARGE_EFF=.94; BATTERY_DISCH_EFF=.94
RATED=12.;CUTIN=3.;CUTOUT=25.
GAS_REF=14.631*1.2 # Mt lifetime proxy from earlier model
GAS_CI_g_MJ=1966/36*1.2+.005*1000/50*29.8

def read_inputs(base, ridge_zip):
    municipal=base/'WEATHER';site_names=['lillooet','cache_creek','kamloops','chase','salmon_arm','vernon','kelowna','west_kelowna','penticton','oliver','osoyoos','hope','agassiz_kent','chilliwack','abbotsford','mission','maple_ridge']
    dfs=[]
    for s in site_names:
        df=pd.read_csv(municipal/(s+'_hourly_era5.csv.gz'),compression='gzip',usecols=['time_utc','shortwave_radiation','temperature_2m']);dfs.append(df)
    dates=pd.to_datetime(dfs[0].time_utc)
    assert all(df.time_utc.equals(dfs[0].time_utc) for df in dfs)
    solar_r=np.stack([df.shortwave_radiation.to_numpy(dtype=np.float32) for df in dfs]);temps=np.stack([df.temperature_2m.to_numpy(dtype=np.float32) for df in dfs])
    ghi=solar_r; tcell=temps+.03*ghi
    solar_cf=np.clip((ghi/1000)*SOLAR_PF*(1-.004*(tcell-25)),0,1).mean(axis=0)
    del dfs, solar_r,temps
    z=zipfile.ZipFile(ridge_zip)
    ridge=[]
    for pick in PICK:
        n=f'combined_hourly/era5/{pick}_hourly.csv.gz'
        raw=z.read(n)
        df=pd.read_csv(io.BytesIO(gzip.decompress(raw)),usecols=['time_utc','wind_speed_100m'])
        assert df.time_utc.equals(pd.Series(dates.dt.strftime('%Y-%m-%dT%H:%M').to_numpy()))
        ridge.append(df.wind_speed_100m.to_numpy(dtype=np.float32))
    wind=np.stack(ridge)
    assert np.isfinite(wind).all() and np.isfinite(solar_cf).all()
    return dates,solar_cf,wind

def wind_cf(wind, speedup):
    v=wind*speedup
    return WINDFARM_AVAIL*np.where((v>=CUTOUT)|(v<CUTIN),0,np.minimum(np.maximum((v**3-CUTIN**3)/(RATED**3-CUTIN**3),0),1))

def demand(dates):
    months=dates.dt.month.to_numpy();hr=dates.dt.hour.to_numpy(); hours=len(dates)
    annual=36004.86*1000;it=32861.982366*1000;farm=annual-it
    factor=np.array([1.48,1.48,1.27,1.10,.93,.73,.68,.67,.83,1.05,1.32,1.48])[months-1].astype(float)
    factor*=1+.12*np.cos((hr-4)*2*np.pi/24)
    factor/=factor.mean()
    return it/hours + farm/hours*factor

def simulate(generation,load,*,battery_gwh,grid_mw=300, backup_mw=5100, hvo_cap_ml=300, record=False):
    """Energy ledger in MWh per hourly slot, independent yearly battery and fuel accounting.
    HVO procurement assumed available as needed until annual cap; no pipeline claims.
    Last resort NG substitutes only AFTER contracted liquid energy is exhausted;
    a real engine still needs liquid pilot. Gas is modeled with 90/10 energy split.
    The N+1 generator available power is 5,100 MW (17x300), NOT unlimited. Seven 50-MW engines per node with one unavailable are screened.
    """
    soc=battery_gwh*1000*.5; cap=battery_gwh*1000; pmax=4000 if battery_gwh<=48 else min(battery_gwh*1000/8,12000)
    b100_left=B100_MWh;hvo_left=hvo_cap_ml*1e6/HVO_L_PER_MWH;gas_left=1e30
    out=dict(renew_generated_MWh=float(generation.sum()),renew_used_MWh=0.,load_MWh=float(load.sum()),grid_import_MWh=0.,battery_discharge_MWh=0.,battery_charge_MWh=0.,b100_MWh=0.,hvo_MWh=0.,gas_MWh=0.,gas_MJ=0.,unserved_MWh=0.,export_MWh=0.,curtailed_MWh=0.,gas_hours=0,unserved_hours=0, max_hourly_deficit_MW=0)
    hourly=[]
    for i in range(len(load)):
        d=float(load[i]);g=float(generation[i]); used=min(d,g); d-=used; surplus=g-used
        out['renew_used_MWh']+=used
        if surplus>0:
            room=min(pmax, (cap-soc)/BATTERY_CHARGE_EFF)
            ch=min(surplus,max(0,room));soc+=ch*BATTERY_CHARGE_EFF;surplus-=ch;out['battery_charge_MWh']+=ch
            ex=min(surplus,grid_mw);out['export_MWh']+=ex;out['curtailed_MWh']+=surplus-ex
        if d>0:
            dis=min(d,pmax,soc*BATTERY_DISCH_EFF);soc-=dis/BATTERY_DISCH_EFF;d-=dis;out['battery_discharge_MWh']+=dis
        if d>0:
            gr=min(d,grid_mw);out['grid_import_MWh']+=gr;d-=gr
        if d>0:
            candidate=min(backup_mw,d);out['max_hourly_deficit_MW']=max(out['max_hourly_deficit_MW'],candidate)
            b=min(candidate,b100_left);candidate-=b;d-=b;b100_left-=b;out['b100_MWh']+=b
            h=min(candidate,hvo_left);candidate-=h;d-=h;hvo_left-=h;out['hvo_MWh']+=h
            if candidate>1e-9:
                # Normal liquid fuel unavailable; emergency 10% liquid pilot must be in secured inventory.
                # With no pilot fuel after exhausting HVO, gas switch cannot start. Separate gas pathway case
                # assumes a dedicated 10% pilot reserve, quantified as incremental HVO below.
                ng=candidate;d-=ng;out['gas_MWh']+=ng;out['gas_MJ']+=ng*3600/.52*.90;out['gas_hours']+=1
            if d>1e-7: out['unserved_MWh']+=d;out['unserved_hours']+=1
        if record:
            hourly.append(dict(hour=i,renew_MWh=g,load_MWh=load[i],soc_MWh=soc,short_MWh=max(0,d),gas_MWh=out['gas_MWh'],export_MWh=out['export_MWh']))
    # emergency pilot fuel requirement not counted as pre-existing HVO supply - explicit accounting
    out['emergency_pilot_HVO_l']=out['gas_MWh']*.10*3600/(.52*HVO_LHV)
    out['hvo_l']=out['hvo_MWh']*HVO_L_PER_MWH
    out['b100_l']=out['b100_MWh']*3600/(ENG_NET_EFF*B100_LHV)
    # WTW screening: renewable intensity charged to all GROSS generated electricity, including curtailed
    out['carbon_Mt']=(out['renew_generated_MWh']*10*1000+out['grid_import_MWh']*100*1000+out['b100_l']*B100_LHV*21+out['hvo_l']*HVO_LHV*36+out['emergency_pilot_HVO_l']*HVO_LHV*36+out['gas_MJ']*GAS_CI_g_MJ)/1e12
    out['reduction_pct']=100*(1-out['carbon_Mt']/GAS_REF)
    out['HVO_ML']=out['hvo_l']/1e6
    out['gross_exports_revenue_CAD_M']=out['export_MWh']*74/1e6
    out['hvo_cost_CAD_M']=out['hvo_l']*HVO_COST_CAD_L/1e6
    out['gas_Mm3']=out['gas_MJ']/36/1e6
    # important: 2010-style 'no-gas shortfall' upper bound in same hourly model
    out['need_without_gas_MWh']=out['gas_MWh']+out['unserved_MWh']
    out['hours']=len(load)
    return out,hourly

def run(base, ridge_zip, out,quick=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dates,solar,wind=read_inputs(Path(base),Path(ridge_zip))
    years=[2015,2020,2025] if quick else list(range(2010,2026))
    input_stats=[]
    for j,n in enumerate(PICK):
        for boost in [1,1.3,1.6,2.0]:
            input_stats.append(dict(hub=n,wind_multiplier=boost,modeled_CF=float(wind_cf(wind[j],boost).mean()),mean_ERA5_wind_m_s=float(wind[j].mean())))
    pd.DataFrame(input_stats).to_csv(out/'ridge_wind_resource_sensitivity.csv',index=False)
    cases=[(1.0,18.625,9.249,48),(1.3,25,16,48),(1.6,25,16,48),(1.6,35,16,96),(1.6,35,24,192),(2.0,25,16,96),(2.0,35,16,96),(1.3,45,24,192),(1.6,45,24,192),(2.0,45,24,192)]
    results=[]
    for y in years:
        idx=((dates.dt.year==y)&~((dates.dt.month==2)&(dates.dt.day==29))).to_numpy()
        dt=dates.loc[idx].reset_index(drop=True)
        d=demand(dt); assert len(d)==8760
        months=dt.dt.month.to_numpy();hyd=np.array([.55,.50,.65,1.10,1.50,1.55,1.20,.80,.80,.75,.65,.55])[months-1];hyd=hyd/hyd.mean()*HYDRO_CF*HYDRO
        for speed,wind_gw,solar_gw,batt_gwh in cases:
            gen=(wind_cf(wind[:,idx],speed)*WEIGHTS[:,None]).sum(axis=0)*wind_gw*1000*(1-LOSS_WIND)+solar[idx]*solar_gw*1000*(1-LOSS_SOLAR)+hyd
            r,h=simulate(gen,d,battery_gwh=batt_gwh,hvo_cap_ml=300,record=(y==2025 and speed==1.6 and wind_gw==35 and solar_gw==16 and batt_gwh==96))
            r.update(year=y,terrain_speedup=speed,wind_GW=wind_gw,solar_GW=solar_gw,battery_GWh=batt_gwh,wind_CF=float((wind_cf(wind[:,idx],speed)*WEIGHTS[:,None]).sum(axis=0).mean()),solar_CF=float(solar[idx].mean()))
            results.append(r)
            if h:pd.DataFrame(h).to_csv(out/'2025_example_hourly_cumulative.csv.gz',index=False,compression='gzip')
        print('year',y,'cases',len(cases),flush=True)
    df=pd.DataFrame(results);df.to_csv(out/'16year_portfolio_hourly_results.csv',index=False,float_format='%.7f')
    agg=df.groupby(['terrain_speedup','wind_GW','solar_GW','battery_GWh']).agg(mean_wind_CF=('wind_CF','mean'),max_gas_Mm3=('gas_Mm3','max'),mean_gas_Mm3=('gas_Mm3','mean'),mean_HVO_ML=('HVO_ML','mean'),max_unserved_GWh=('unserved_MWh',lambda x:x.max()/1000),mean_export_GWh=('export_MWh',lambda x:x.mean()/1000),mean_curtail_GWh=('curtailed_MWh',lambda x:x.mean()/1000),mean_reduction_pct=('reduction_pct','mean'),min_reduction_pct=('reduction_pct','min'),mean_carbon_Mt=('carbon_Mt','mean'),mean_hvo_cost_CAD_M=('hvo_cost_CAD_M','mean'),mean_export_revenue_CAD_M=('gross_exports_revenue_CAD_M','mean'),mean_renew_GWh=('renew_generated_MWh',lambda x:x.mean()/1000)).reset_index()
    agg.to_csv(out/'portfolio_comparison_summary.csv',index=False,float_format='%.6f')
    # Explicit contract and development economic screen, not overnight quotes or LCOE
    capex={'wind_CAD_M_per_MW':2.6,'solar_CAD_M_per_MW':1.35,'battery_CAD_M_per_MWh':.31,'regional_network_CAD_billion':18,'generator_CAD_billion':8.94}
    agg['capex_CAD_billion']=agg.wind_GW*capex['wind_CAD_M_per_MW']+agg.solar_GW*capex['solar_CAD_M_per_MW']+agg.battery_GWh*capex['battery_CAD_M_per_MWh']+18+8.94
    agg['annualized_capex_CAD_billion']=agg['capex_CAD_billion']*.085
    agg['net_annual_operating_proxy_CAD_billion']=(agg.mean_hvo_cost_CAD_M-agg.mean_export_revenue_CAD_M)/1000
    agg['total_annual_cost_proxy_CAD_billion']=agg.annualized_capex_CAD_billion+agg.net_annual_operating_proxy_CAD_billion
    # Find candidates with zero gas even if HVO procurement up to 300ML/a; does not establish fuel availability.
    agg.to_csv(out/'portfolio_financial_screen.csv',index=False,float_format='%.6f')
    selected=agg.sort_values(['max_unserved_GWh','mean_gas_Mm3','total_annual_cost_proxy_CAD_billion']).iloc[0]
    result={'assumptions':{'years':years,'grid_import_cap_MW':GRID_CAP,'grid_export_cap_MW':GRID_CAP,'price_CAD_MWh_2024_awards':74,'B100_UCO_ML_per_year':20,'HVO_contract_upper_bound_ML_per_year':300,'land_and_PPA_unverified':True,'terrain_speedups_are_not_observed':True,'transmission_is_idealized_pooled_upper_bound':True,'wind_sites_are_weather_proxies':True,'hydro_flow_not_observed':True,'historical_energy_supply_not_future_uptime_guarantee':True,'capex_inputs':capex},'selected_by_gas_minimum_then_cost':selected.to_dict(),'other_candidates':agg.to_dict('records')}
    (out/'SUMMARY.json').write_text(json.dumps(result,indent=2,default=float))
    print(agg[['terrain_speedup','wind_GW','solar_GW','battery_GWh','mean_wind_CF','mean_HVO_ML','max_gas_Mm3','max_unserved_GWh','mean_export_GWh','mean_reduction_pct','capex_CAD_billion']].round(2).to_string(index=False),flush=True)
    return agg,df
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',default=str(DEFAULT_BASE));p.add_argument('--ridge-zip',default=str(DEFAULT_BASE/'RIDGE_WIND_ERA5_6_POINTS_2010_2025.zip'));p.add_argument('--out',default=str(HERE/'RESULTS'));p.add_argument('--quick',action='store_true');a=p.parse_args()
 run(a.base,a.ridge_zip,a.out,a.quick)
