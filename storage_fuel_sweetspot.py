#!/usr/bin/env python3
"""Battery versus stored renewable liquid fuel: the 25-year least-cost "sweet spot".

The hourly dispatch kernel (battery, then B100, then HVO, then emergency gas within the 5.1-GW N+1 engine envelope) is
replayed over all 16 ERA5 weather years for a grid of battery sizes, solar sizes, wind scales and annual stored-fuel
allowances. A configuration qualifies only if it needs zero gas and serves every hour in all 16 years. Each qualifying
configuration is costed over 25 years: capital financed over 20 years (paid off afterwards), fixed O&M, battery
replacement in year 15, fuel, fuel tanks sized to the worst-year burn, and fuel inventory carrying cost. Life-cycle
emissions of the stored fuel are reported so the cheaper-storage option is not mistaken for zero-carbon.
"""
from pathlib import Path
import sys, json, itertools
import numpy as np, pandas as pd
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
BASE = HERE / "DATA"
sys.path.insert(0, str(HERE / "MODEL_CORE"))
from dispatch_reference import read_inputs, wind_cf, HVO_L_PER_MWH, B100_MWh, LOSS_WIND, HYDRO, HYDRO_CF  # noqa: E402
from land_constrained_optimizer import make_data, dispatch  # noqa: E402

OUT = HERE / "OUTPUT"; OUT.mkdir(exist_ok=True)

P = dict(
    capex_wind_bn_gw=2.6, capex_solar_bn_gw=1.35, capex_batt_bn_gwh=0.31, capex_network_engines_bn=26.94,
    om_wind_bn_gw=0.045, om_solar_bn_gw=0.020, om_batt_bn_gwh=0.008, om_engines_bn=0.119,
    b100_bn=0.0288, hvo_cad_l=1.8, finance_rate=0.06, finance_years=20, horizon=25, battery_life=15,
    battery_replacement_cost_fraction=0.60,      # year-15 replacement at 60% of today's cost (price decline; assumption)
    tank_cad_per_m3=250.0, tank_om_frac=0.02, inventory_carry_rate=0.06,
    hvo_ci_g_mj=25.0,                             # waste-lipid HVO life-cycle intensity (CFR/LCFS range ~15-35)
    hvo_lhv_mj_l=36.51, backup_mw=5100.0,
    supply_cap_ml=1000.0,
    farm_hvo_ml=(49159 * 400.0 + 38848 * 100.0) / 1e6,   # host-community farm diesel replaced by HVO (23.55 ML/yr), drawn from the same ceiling   # procurement ceiling: about the output of Canada's largest renewable-diesel plant (Strathcona, ~1 billion L/yr)
)


def crf(r, n):
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def calibrated_wind(wind, target=None):
    from dispatch_reference import read_inputs as _r  # noqa
    ref_cf = (4_830_000 - 104 * 8760 * .18) / (1531 * 8760) if target is None else target
    out = []
    for w in wind:
        c = np.arange(1., 8.01, .05)
        pairs = [(c[k], c[k + 1]) for k in range(len(c) - 1) if wind_cf(w, c[k]).mean() < ref_cf <= wind_cf(w, c[k + 1]).mean()]
        s = brentq(lambda s: wind_cf(w, s).mean() - ref_cf, *pairs[0])
        out.append(wind_cf(w, s))
    return np.stack(out).astype("float64"), ref_cf


def load_years(target_cf=None):
    active=HERE/'LARGE_TURBINE_STUDY'/'ACTIVE_8MW.json'
    if active.exists():
        # Same hourly ERA5/GWA data as the manufacturer's 8-MW engineering proxy study;
        # power curve is not a certified SANY manufacturer performance guarantee.
        from LARGE_TURBINE_STUDY.model_8mw_10mw import build_inputs
        groups,_,_,_,_,site=build_inputs('SANY_SI19580_8MW',cf_ceiling=.40 if target_cf is None else float(target_cf))
        return groups,float(np.mean([s['screened_cf'] for s in site]))
    groups, wind = make_data(BASE, BASE / "RIDGE_WIND_ERA5_6_POINTS_2010_2025.zip")
    dates, _, _ = read_inputs(BASE, BASE / "RIDGE_WIND_ERA5_6_POINTS_2010_2025.zip")
    # When supplied, the independently modelled GWA ridge MEANS replace an arbitrary
    # fleet-CF target. Hourly structure still comes from ERA5 and is unvalidated.
    atlasfile=HERE/'OBSERVATIONAL_VALIDATION'/'INPUT'/'gwa_100m_means.csv'
    if target_cf is None and atlasfile.exists():
        import csv
        from dispatch_reference import PICK
        vals={r['site_key']:float(r['gwa_mean_100m_ms']) for r in csv.DictReader(atlasfile.open(encoding='utf-8-sig'))}
        if set(vals)!=set(PICK): raise ValueError('GWA site set must exactly match 5 ridge proxies')
        speeds=[vals[k]/float(wind[j].mean()) for j,k in enumerate(PICK)]
        # Independent screening constraint: no unverified site is allowed a modeled
        # 16-year turbine capacity factor above 40% on the GWA/ERA5 shape.
        for j in range(len(PICK)):
            if wind_cf(wind[j],speeds[j]).mean() > .40:
                speeds[j]=brentq(lambda m:wind_cf(wind[j],m).mean()-.40, .25, speeds[j])
        wcf=np.stack([wind_cf(wind[j],speeds[j]) for j in range(len(PICK))]).astype('float64')
        ref=float(wcf.mean())  # unweighted five-site mean, not installed portfolio CF
    else:
        wcf,ref=calibrated_wind(wind,target_cf)
    for g in groups:
        m = ((dates.dt.year == g["year"]) & ~((dates.dt.month == 2) & (dates.dt.day == 29))).to_numpy()
        g["windcf"] = wcf[:, m]
    return groups, ref


def run_case(groups, wind_gw, solar_gw, batt_gwh, fuel_ml):
    """Dispatch all years; return worst-year and mean statistics."""
    hvo_limit = fuel_ml * 1e6 / HVO_L_PER_MWH
    rows = []
    for g in groups:
        gen = (np.asarray(wind_gw) @ g["windcf"]) * 1000 * (1 - LOSS_WIND) + solar_gw * g["solar"] + g["hydro"]
        b, h, gas, u, gr, ex, cur = dispatch(gen, g["load"], batt_gwh * 1000, 0., hvo_limit, P["backup_mw"])
        rows.append((h * HVO_L_PER_MWH / 1e6, gas / 1000, u / 1000, cur / 1e6, gen.sum() / 1e6, g["load"].sum() / 1e6))
    a = np.array(rows)
    return dict(hvo_ml_mean=a[:, 0].mean(), hvo_ml_max=a[:, 0].max(), gas_gwh_max=a[:, 1].max(), unserved_gwh_max=a[:, 2].max(),
                curtail_twh=a[:, 3].mean(), gen_twh=a[:, 4].mean(), load_twh=a[:, 5].mean())


def cost_25y(wind_total, solar, batt, stats, fuel_cap_ml):
    """Annual cash costs (CAD bn) by year and the 25-year present-value levelized cost."""
    p = P
    capex = wind_total * p["capex_wind_bn_gw"] + solar * p["capex_solar_bn_gw"] + batt * p["capex_batt_bn_gwh"] + p["capex_network_engines_bn"]
    tanks = stats["hvo_ml_max"] * 1e3 * p["tank_cad_per_m3"] / 1e9            # tanks for the worst-year volume
    debt = (capex + tanks) * crf(p["finance_rate"], p["finance_years"])
    om = wind_total * p["om_wind_bn_gw"] + solar * p["om_solar_bn_gw"] + batt * p["om_batt_bn_gwh"] + p["om_engines_bn"] + tanks * p["tank_om_frac"]
    fuel = stats["hvo_ml_mean"] * 1e6 * p["hvo_cad_l"] / 1e9 + p["b100_bn"]
    carry = 0.5 * stats["hvo_ml_max"] * 1e6 * p["hvo_cad_l"] * p["inventory_carry_rate"] / 1e9
    years = []
    for y in range(1, p["horizon"] + 1):
        c = om + fuel + carry + (debt if y <= p["finance_years"] else 0.0)
        if y == p["battery_life"]:
            c += batt * p["capex_batt_bn_gwh"] * p["battery_replacement_cost_fraction"]
        years.append(c)
    r = p["finance_rate"]
    pv = sum(c / (1 + r) ** y for y, c in enumerate(years, 1))
    pv_mwh = sum(stats["load_twh"] * 1e6 / (1 + r) ** y for y in range(1, p["horizon"] + 1))
    co2 = stats["hvo_ml_mean"] * 1e6 * p["hvo_lhv_mj_l"] * p["hvo_ci_g_mj"] / 1e12      # Mt/yr
    return dict(capex_bn=capex, tanks_bn=tanks, annual_debt_bn=debt, om_bn=om, fuel_bn=fuel, carry_bn=carry,
                cash_yr1_bn=years[0], cash_yr21_bn=years[20], pv_cost_bn=pv, lcoe_25y_cad_mwh=pv * 1e9 / pv_mwh,
                cost_after_payoff_cad_mwh=years[20] * 1e3 / stats["load_twh"], fuel_lifecycle_mt=co2, annual_path_bn=years)


def main():
    groups, ref = load_years()
    sel = json.load(open(OUT / "optimization_provenance.json"))
    w0 = np.array(sel["wind_GW"])
    rows = []
    dc_cap = P["supply_cap_ml"] - P["farm_hvo_ml"]          # renewable-fuel allowance left for the data centres
    if "--reuse-grid" in sys.argv and (OUT / "storage_fuel_sweetspot_grid.csv").exists():
        grid = []
        df = pd.read_csv(OUT / "storage_fuel_sweetspot_grid.csv")
    else:
        df = None
    grid = grid if df is not None else itertools.product([0.5, 0.65, 0.8, 1.0, 1.0435], [10, 15, 20, 25, 30, 40, 50, 60], [5, 10, 15, 20, 25, 30, 45, 60, 89, 120],
                             [300, 450, 600, 800, 1000, 1250, 1500, 2000, 3000])
    for ws, solar, batt, fuel in grid:
        wind = np.minimum(w0 * ws, 2.0)
        st = run_case(groups, wind, solar, batt, fuel)
        ok = st["gas_gwh_max"] < 1e-3 and st["unserved_gwh_max"] < 1e-3
        c = cost_25y(wind.sum(), solar, batt, st, fuel)
        rows.append(dict(wind_scale=ws, wind_GW=wind.sum(), solar_GW=solar, battery_GWh=batt, fuel_cap_ML=fuel, qualifies=ok, **st,
                         **{k: v for k, v in c.items() if k != "annual_path_bn"}))
    if df is None:
        df = pd.DataFrame(rows)
    df.to_csv(OUT / "storage_fuel_sweetspot_grid.csv", index=False, float_format="%.6f")
    q = df[df.qualifies]
    best_unconstrained = q.sort_values("lcoe_25y_cad_mwh").iloc[0]
    # robust sweet spot: cheapest configuration that is also gas-free at the 30% fleet wind yield
    g30, _ = load_years(0.30)
    robust = []
    # the data-centre allowance is the procurement ceiling less the farm HVO; a case qualifies only if it needs no gas at
    # both the calibrated and the 30% wind yield with that allowance
    cand = q[q.fuel_cap_ML <= P["supply_cap_ml"]].sort_values("lcoe_25y_cad_mwh")
    for _, r in cand.iterrows():
        alw = min(r.fuel_cap_ML, dc_cap); wv = np.minimum(w0 * r.wind_scale, 2.0)
        st30_ = run_case(g30, wv, r.solar_GW, r.battery_GWh, alw)
        if st30_["gas_gwh_max"] < 1e-3 and st30_["unserved_gwh_max"] < 1e-3:
            stc = run_case(groups, wv, r.solar_GW, r.battery_GWh, alw)
            if stc["gas_gwh_max"] < 1e-3 and stc["unserved_gwh_max"] < 1e-3:
                robust.append((r.name, alw, stc)); break
    idx, alw, stc = robust[0]
    best = q.loc[idx].copy()
    best["fuel_cap_ML"] = alw
    best.update(pd.Series({**stc, **{k: v for k, v in cost_25y(best.wind_GW, best.solar_GW, best.battery_GWh, stc, alw).items() if k != "annual_path_bn"}}))
    # reference: the published selected portfolio (9.57 GW, 40.48 GW, 88.92 GWh, 300 ML)
    st_ref = run_case(groups, w0, sel["solar_GW"], sel["battery_GWh"], 300)
    ref_cost = cost_25y(w0.sum(), sel["solar_GW"], sel["battery_GWh"], st_ref, 300)
    best_cost = cost_25y(best.wind_GW, best.solar_GW, best.battery_GWh, stc, alw)
    # least-cost per fuel cap (the trade-off curve) and a fuel-capped "low-carbon" sweet spot (<=600 ML)
    curve = q.sort_values("lcoe_25y_cad_mwh").groupby("fuel_cap_ML").head(1).sort_values("fuel_cap_ML")
    low = q[q.fuel_cap_ML <= 600].sort_values("lcoe_25y_cad_mwh").iloc[0]
    # 30% wind stress test of the sweet spot
    st30 = run_case(g30, np.minimum(w0 * best.wind_scale, 2.0), best.solar_GW, best.battery_GWh, best.fuel_cap_ML)
    res = dict(reference_selected=dict(wind_GW=float(w0.sum()), solar_GW=sel["solar_GW"], battery_GWh=sel["battery_GWh"], fuel_cap_ML=300,
                                       **{k: float(v) for k, v in st_ref.items()}, **{k: (v if k == "annual_path_bn" else float(v)) for k, v in ref_cost.items()}),
               sweet_spot=dict(**{k: (bool(v) if k == "qualifies" else float(v)) for k, v in best.items()},
                               annual_path_bn=best_cost["annual_path_bn"]),
               unconstrained_cheapest=dict(**{k: (bool(v) if k == "qualifies" else float(v)) for k, v in best_unconstrained.items()},
                                           at_30pct_wind=run_case(g30, np.minimum(w0 * best_unconstrained.wind_scale, 2.0), best_unconstrained.solar_GW,
                                                                  best_unconstrained.battery_GWh, best_unconstrained.fuel_cap_ML)),
               low_fuel_sweet_spot=dict(**{k: (bool(v) if k == "qualifies" else float(v)) for k, v in low.items()}),
               tradeoff_curve=curve[["fuel_cap_ML", "battery_GWh", "solar_GW", "wind_GW", "lcoe_25y_cad_mwh", "hvo_ml_mean", "hvo_ml_max",
                                     "fuel_lifecycle_mt", "capex_bn"]].to_dict("records"),
               sweet_spot_at_30pct_wind=st30, data_centre_fuel_allowance_ml=dc_cap, n_cases=len(df), n_qualifying=int(df.qualifies.sum()), wind_benchmark_cf=ref, params=P)
    (OUT / "storage_fuel_sweetspot.json").write_text(json.dumps(res, indent=1, default=float))
    print("reference", round(ref_cost["lcoe_25y_cad_mwh"], 1), "sweet spot", best[["wind_GW", "solar_GW", "battery_GWh", "fuel_cap_ML", "lcoe_25y_cad_mwh",
          "hvo_ml_mean", "hvo_ml_max", "fuel_lifecycle_mt"]].round(2).to_dict())
    print("low-fuel", low[["battery_GWh", "solar_GW", "fuel_cap_ML", "lcoe_25y_cad_mwh", "hvo_ml_mean"]].round(2).to_dict())
    print(curve[["fuel_cap_ML", "battery_GWh", "solar_GW", "lcoe_25y_cad_mwh", "hvo_ml_mean"]].round(1).to_string(index=False))
    print("30% wind:", {k: round(v, 2) for k, v in st30.items()})


if __name__ == "__main__":
    main()
