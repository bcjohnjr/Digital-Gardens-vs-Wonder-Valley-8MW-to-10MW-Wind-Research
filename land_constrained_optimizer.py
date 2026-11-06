"""Reproducible portfolio screen: optimize resource hubs, wind-farm envelope and annual HVO.
Does not demonstrate actual constructible ridge sites. Weather uses observation-informed ERA5 reanalysis.
All 16 historical years, hourly, cap 300 MW competitively awarded BC Hydro (also zero award screen).
"""
from pathlib import Path
import argparse, json, zipfile, hashlib, sys
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from numba import njit
from dispatch_reference import read_inputs, demand, wind_cf, PICK, SOLAR_PF, LOSS_WIND, LOSS_SOLAR, HYDRO, HYDRO_CF, B100_MWh, HVO_L_PER_MWH, GAS_REF

# Units: MW, MWh, GW, GWh, million litres (ML)
@njit(cache=True)
def dispatch(generation,load,capacity_mwh, grid, hvo_limit_mwh, backup_mw):
    soc=capacity_mwh*.5
    # assumptions from reference model inherited, except ability to permit smaller batteries P=E/8 up to 12 GW
    pmax=min(capacity_mwh/8., 12000.)
    b100=B100_MWh
    hvo=hvo_limit_mwh
    use_b100=0.; use_hvo=0.; unserved=0.; used_grid=0.; exports=0.; curtail=0.; gas=0.
    for k in range(len(generation)):
        g=generation[k];d=load[k];net=g-d
        if net>=0:
            c=min(net,pmax,max((capacity_mwh-soc)/.94,0.))
            soc+=c*.94
            extra=net-c
            ex=min(extra,grid)
            exports+=ex;curtail+=extra-ex
        else:
            deficit=-net
            dis=min(deficit,pmax,soc*.94)
            soc-=dis/.94;deficit-=dis
            gi=min(deficit,grid);deficit-=gi;used_grid+=gi
            if deficit>0:
                cap=min(deficit,backup_mw)
                b=min(cap,b100);cap-=b;b100-=b;use_b100+=b
                h=min(cap,hvo);cap-=h;hvo-=h;use_hvo+=h
                gas+=cap
                unserved+=max(0.,deficit-backup_mw)
    return use_b100,use_hvo,gas,unserved,used_grid,exports,curtail


def make_data(base,ridge_zip):
    dates,solar,wind=read_inputs(base,ridge_zip)
    groups=[]
    for y in range(2010,2026):
        idx=((dates.dt.year==y)&~((dates.dt.month==2)&(dates.dt.day==29))).to_numpy()
        dt=dates.loc[idx].reset_index(drop=True)
        dm=demand(dt).astype('float64')
        months=dt.dt.month.to_numpy();h=np.array([.55,.50,.65,1.10,1.50,1.55,1.20,.80,.80,.75,.65,.55])[months-1];h=h/h.mean()*HYDRO_CF*HYDRO
        groups.append(dict(year=y, solar=solar[idx].astype('float64')*(1-LOSS_SOLAR)*1000, wind=wind[:,idx],load=dm,hydro=h.astype('float64')))
    return groups,wind

def evaluate(x,groups,speedup=1.6,grid=300,return_years=False):
    # x [wind GW by hub 1..5, solar GW, battery GWh]
    cap=np.maximum(np.array(x[:5]),0); solar=max(x[5],0);batt=max(x[6],0)
    worst_unserved=0;worst_gas=0;max_hvo=0;annual=[]
    for grp in groups:
        # wind values cached as windcf matrix below
        generation=(cap@grp['windcf'])*1000*(1-LOSS_WIND)+solar*grp['solar']+grp['hydro']
        b,h,g,u,gr,ex,curtail=dispatch(generation,grp['load'],batt*1000,grid,300e6/HVO_L_PER_MWH,5100)
        row=dict(year=grp['year'],wind_TWh=((cap@grp['windcf'])*1000*(1-LOSS_WIND)).sum()/1e6,solar_TWh=(solar*grp['solar']).sum()/1e6,hydro_GWh=grp['hydro'].sum()/1000,generated_TWh=generation.sum()/1e6,load_TWh=grp['load'].sum()/1e6,b100_GWh=b/1000,hvo_ML=h*HVO_L_PER_MWH/1e6,gas_GWh=g/1000,unserved_GWh=u/1000,grid_GWh=gr/1000,export_GWh=ex/1000,curtail_TWh=curtail/1e6)
        annual.append(row)
        worst_unserved=max(worst_unserved,u/1000);worst_gas=max(worst_gas,g/1000);max_hvo=max(max_hvo,row['hvo_ML'])
    if return_years:return pd.DataFrame(annual)
    # same reference model capex assumptions, explicit area implication, includes 18B network & 8.94B engines
    capex=float(cap.sum()*2.6 + solar*1.35 + batt*.31+18+8.94)
    return capex,worst_gas,worst_unserved,max_hvo,np.mean([r['curtail_TWh'] for r in annual]),np.mean([r['hvo_ML'] for r in annual])

@njit(cache=True)
def fast_objective(x, windblock,solarblock,hydroblock,loadblock,grid):
    maxgas=0.; maxunserved=0.; maxhvo=0.; sumcur=0.;sumhvo=0.
    batt=x[6]*1000.; solarGW=x[5]
    for y in range(loadblock.shape[0]):
        gen=np.zeros(loadblock.shape[1],dtype=np.float64)
        for t in range(loadblock.shape[1]):
            v=solarGW*solarblock[y,t]+hydroblock[y,t]
            for hub in range(5):v+=x[hub]*windblock[y,hub,t]
            gen[t]=v
        b,h,g,u,gr,ex,cur=dispatch(gen,loadblock[y],batt,grid,300e6/HVO_L_PER_MWH,5100.)
        maxgas=max(maxgas,g/1000.)
        maxunserved=max(maxunserved,u/1000.)
        maxhvo=max(maxhvo,h*HVO_L_PER_MWH/1e6)
        sumcur+=cur/1e6
        sumhvo+=h*HVO_L_PER_MWH/1e6
    capex=(x[0]+x[1]+x[2]+x[3]+x[4])*2.6+solarGW*1.35+x[6]*.31+26.94
    return capex,maxgas,maxunserved,maxhvo,sumcur/loadblock.shape[0],sumhvo/loadblock.shape[0]

def make_fast_inputs(groups):
    W=np.stack([np.ascontiguousarray(g['windcf']*1000*(1-LOSS_WIND),dtype='float64') for g in groups])
    S=np.stack([np.ascontiguousarray(g['solar'],dtype='float64') for g in groups])
    H=np.stack([np.ascontiguousarray(g['hydro'],dtype='float64') for g in groups])
    L=np.stack([np.ascontiguousarray(g['load'],dtype='float64') for g in groups])
    return W,S,H,L

def run(args):
    print('Loading full 17-site+ridge ERA5; preparing 2010-2025 hourly models',flush=True)
    groups,wind=make_data(args.base,args.ridge_zip)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    output=[]; yearly=[]; land=[];history=[]
    for speedup in [1.3,1.6,2.0]:
        for g in groups:g['windcf']=wind_cf(g['wind'],speedup).astype('float64')
        W,S,H,L=make_fast_inputs(groups)
        # 5 sampled *distinct* ridge cells, 3MW/km² gross farm envelope capacity benchmark
        for cap_hub in [2.,5.,10.,20.]:
            # optimize no-gas case. BESS 0..300 GWh; solar 0..50GW
            bounds=[(0.,cap_hub)]*5+[(0,50),(8,300)]
            counter=[0]
            def objective(x):
                r=fast_objective(np.asarray(x,dtype=np.float64),W,S,H,L,300.)
                counter[0]+=1
                # firm constraint; heavily penalize gas and unserved, but allow infeasible screen
                return r[0]+r[1]*5+r[2]*50+max(0,r[3]-300)*20
            res=differential_evolution(objective,bounds,popsize=6,maxiter=14,seed=272+round(speedup*100)+int(cap_hub),polish=True,tol=.015,updating='immediate',workers=1)
            x=res.x;r=evaluate(x,groups,speedup,grid=300)
            result=dict(speedup=speedup,per_hub_cap_GW=cap_hub,land_ceiling_km2=5*cap_hub*1000/3,
                wind_GW=float(sum(x[:5])),solar_GW=float(x[5]),battery_GWh=float(x[6]),
                wind_land_envelope_km2=float(sum(x[:5])*1000/3),
                capex_CAD_B=r[0],max_gas_GWh=r[1],max_unserved_GWh=r[2],max_HVO_ML=r[3],
                mean_curtail_TWh=r[4],mean_HVO_ML=r[5],qualified_no_gas=bool(r[1]<.001 and r[2]<.001),**{f'wind_{p}_GW':float(x[j]) for j,p in enumerate(PICK)})
            output.append(result)
            yy=evaluate(x,groups,speedup,grid=300,return_years=True)
            yy['speedup']=speedup;yy['per_hub_cap_GW']=cap_hub;yearly.append(yy)
            print('screen speedup',speedup,'hub cap',cap_hub,'wind',round(result['wind_GW'],2),'solar',round(result['solar_GW'],2),'battery',round(result['battery_GWh'],2),'CAPEX',round(r[0],1),'GASmax',round(r[1],1),'HVOmax',round(r[3],1),flush=True)
    results=pd.DataFrame(output)
    results.to_csv(out/'land_capacity_optimization.csv',index=False)
    pd.concat(yearly,ignore_index=True).to_csv(out/'all_16year_optimizer_validation.csv',index=False)
    # selected scenario: globally cheapest no-gas from 1.6 speedup and 20GW/hub max, otherwise report infeasible
    qualified=results[(results.speedup==1.6)&results.qualified_no_gas]
    chosen=qualified.sort_values('capex_CAD_B').iloc[0] if len(qualified) else results[results.speedup==1.6].sort_values('max_gas_GWh').iloc[0]
    x=np.array([chosen[f'wind_{p}_GW'] for p in PICK]+[chosen.solar_GW,chosen.battery_GWh]);
    for speed in [1.,1.3,1.6,2.0]:
        for g in groups:g['windcf']=wind_cf(g['wind'],speed).astype('float64')
        for grid in [0,150,300]:
            yy=evaluate(x,groups,speed,grid=grid,return_years=True)
            history.append(dict(case='chosen_speed_robustness',speedup=speed,grid_MW=grid,max_gas_GWh=yy.gas_GWh.max(),max_unserved_GWh=yy.unserved_GWh.max(),max_HVO_ML=yy.hvo_ML.max(),mean_export_GWh=yy.export_GWh.mean(),mean_curtail_TWh=yy.curtail_TWh.mean(),mean_generation_TWh=yy.generated_TWh.mean()))
    pd.DataFrame(history).to_csv(out/'selected_portfolio_robustness.csv',index=False)
    # region-level wind envelope sensitivity, total capacity density 3MW/km2 not turbine-pad footprint
    for row in output:
        for density in [2,3,5]:land.append(dict(speedup=row['speedup'],hub_capacity_GW=row['per_hub_cap_GW'],optimized_wind_GW=row['wind_GW'],farm_density_MW_km2=density,spacing_envelope_km2=row['wind_GW']*1000/density,actual_permitted_area_verified=False))
    pd.DataFrame(land).to_csv(out/'wind_land_area_density_sensitivity.csv',index=False)
    report=dict(selected_1_6_screen=chosen.to_dict(),optimized_screens=len(results),historical_years=list(range(2010,2026)),sources=['User supplied 17 municipal ERA5 2010–2025','User supplied six-point ridge ERA5 archive'],limitations=['Wind speed uplift unmeasured','5 ERA5 coarse grid cells cannot spatially verify capacity buildability','No GIS exclusion polygons or actual available land','No feeder/transmission power-flow constraints','5 ridge cells are screening proxies not permits','BC Hydro 300MW unawarded','HVO 300ML/yr uncontracted','Hydro flow 8.2MW hypothetical','Export capped at awarded grid MW but a separate export agreement would be required','Solar land or transmission construction not modeled','Battery duration and multi-year degradation not tested'])
    (out/'portfolio_results_and_limits.json').write_text(json.dumps(report,indent=2,default=float))
    return results,pd.DataFrame(history)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',default=str(Path(__file__).resolve().parents[1]/'DATA'))
    ap.add_argument('--ridge-zip',default='')
    ap.add_argument('--out',default=str(Path(__file__).resolve().parent/'portfolio_results'))
    a=ap.parse_args()
    a.base=Path(a.base)
    if not a.ridge_zip:a.ridge_zip=str(a.base/'RIDGE_WIND_ERA5_6_POINTS_2010_2025.zip')
    run(a)
