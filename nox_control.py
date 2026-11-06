#!/usr/bin/env python3
"""NOx (and N2O) from the renewable-diesel engines, and what combustion and after-treatment controls achieve.

The hourly engine output of the selected portfolio is taken from the same dispatch rules as the published kernel. Engine-out
NOx is built up multiplicatively: a fossil-diesel baseline for large medium-speed engines, the fuel effect of HVO (and B100),
in-cylinder measures (retarded injection timing, exhaust-gas recirculation, water-in-fuel emulsion) and selective catalytic
reduction (SCR). In-cylinder measures cost fuel; that penalty is checked against the 1,000 ML/yr renewable-fuel ceiling by
re-running the selected 40%-per-site screening dispatch with the reduced effective allowance. Every factor is a screening assumption with its range
and source in NOX below.
"""
from pathlib import Path
import json
import numpy as np, pandas as pd
from numba import njit
import storage_fuel_sweetspot as sfs
from dispatch_reference import HVO_L_PER_MWH, LOSS_WIND, B100_MWh  # noqa: E402

HERE = Path(__file__).resolve().parent; OUT = HERE / "OUTPUT"
SS = json.load(open(OUT / "storage_fuel_sweetspot.json")); sw = SS["sweet_spot"]
sel = json.load(open(OUT / "optimization_provenance.json")); W = np.asarray(sw["wind_GW_per_site"],dtype=float)
ALW = sw["fuel_cap_ML"]; FARM = SS["params"]["farm_hvo_ml"]
LOAD_TWH = 36.00486

NOX = dict(
    baseline_g_kwh=9.0, baseline_range=(8.0, 10.5),      # large medium-speed diesel engines; IMO Tier II limit 9.7-10.5 g/kWh at 500-750 rpm
    hvo_effect=-0.05, hvo_range=(0.0, -0.10),            # high load: 0-6% lower (Duva et al. 2025); >50% at low load with tuning (Imperato et al. 2011)
    b100_effect=+0.10,                                   # biodiesel B100 about +10% NOx (US EPA 2002)
    timing_retard=dict(nox=-0.15, fuel=+0.02),           # retarded injection: lower peak flame temperature (Doppalapudi et al. 2023)
    egr_10pct=dict(nox=-0.20, fuel=+0.02),               # 10% EGR: ~20% NOx cut, 1-4% efficiency loss (Bansod et al. 2026; ranges 5-25%)
    emulsion=dict(nox=-0.15, fuel=0.0),                  # water-in-fuel emulsion, 10-15% water (Lif and Holmberg 2006)
    scr={"90%": 0.90, "95%": 0.95, "97%": 0.97},          # SCR: >90% typical, near 100% theoretical (US EPA Cost Manual ch. 2)
    n2o_scr_g_kwh=0.03, n2o_range=(0.01, 0.10),          # N2O formed over SCR/oxidation catalysts (SAE 2015-01-1030, 2017-01-0987); assumption
    gwp_n2o=265.0,                                       # IPCC AR5 GWP100
    def_kg_per_kg_nox=2.0, def_cad_per_kg=0.75,           # 32.5% urea solution: ~0.65 kg urea per kg NOx (1.05 NH3:NO) -> ~2 kg DEF
    scr_capex_cad_kw={"90%": 80.0, "95%": 100.0, "97%": 130.0}, egr_capex_cad_kw=25.0, emulsion_capex_cad_kw=10.0,
    om_frac=0.03, crf=0.06 * 1.06 ** 20 / (1.06 ** 20 - 1), engine_mw=5100.0, hvo_cad_l=1.80,
    gas_twin_g_kwh=0.025,                                # SCR-equipped combined cycle (~2 ppm at 15% O2)
)

PACKAGES = {
    "Uncontrolled (fossil-diesel calibration)": dict(),
    "HVO only": dict(hvo=True),
    "HVO + SCR 90%": dict(hvo=True, scr="90%"),
    "HVO + SCR 95%": dict(hvo=True, scr="95%"),
    "HVO + emulsion + SCR 95%": dict(hvo=True, emulsion=True, scr="95%"),
    "HVO + emulsion + SCR 97%": dict(hvo=True, emulsion=True, scr="97%"),
    "HVO + retarded timing + EGR + SCR 95%": dict(hvo=True, timing=True, egr=True, scr="95%"),
    "HVO + retarded timing + EGR + emulsion + SCR 97%": dict(hvo=True, timing=True, egr=True, emulsion=True, scr="97%"),
}


@njit(cache=True)
def dispatch_engines(generation, load, capacity_mwh, hvo_limit_mwh, backup_mw, b100):
    """Published dispatch order; returns hourly B100 and HVO engine output (MWh)."""
    soc = capacity_mwh * .5; pmax = min(capacity_mwh / 8., 12000.); hvo = hvo_limit_mwh
    eb = np.zeros(len(generation)); eh = np.zeros(len(generation))
    for k in range(len(generation)):
        net = generation[k] - load[k]
        if net >= 0:
            c = min(net, pmax, max((capacity_mwh - soc) / .94, 0.)); soc += c * .94
        else:
            d = -net; dis = min(d, pmax, soc * .94); soc -= dis / .94; d -= dis
            if d > 0:
                cap = min(d, backup_mw); b = min(cap, b100); cap -= b; b100 -= b; eb[k] = b
                h = min(cap, hvo); hvo -= h; eh[k] = h
    return eb, eh


def engine_series(groups, allowance_ml):
    eb, eh = [], []
    for g in groups:
        gen = (W @ g["windcf"]) * 1000 * (1 - LOSS_WIND) + sw["solar_GW"] * g["solar"] + g["hydro"]
        b, h = dispatch_engines(gen, g["load"], sw["battery_GWh"] * 1000, allowance_ml * 1e6 / HVO_L_PER_MWH, 5100., B100_MWh)
        eb.append(b); eh.append(h)
    return np.concatenate(eb), np.concatenate(eh)


def factor(pk):
    """NOx g/kWh for HVO and for B100 under a package, and its fuel penalty."""
    f = 1.0; pen = 0.0
    for key, flag in (("timing_retard", "timing"), ("egr_10pct", "egr"), ("emulsion", "emulsion")):
        if pk.get(flag):
            f *= 1 + NOX[key]["nox"]; pen += NOX[key]["fuel"]
    scr = NOX["scr"][pk["scr"]] if pk.get("scr") else 0.0
    base = NOX["baseline_g_kwh"] * f * (1 - scr)
    g_hvo = base * (1 + (NOX["hvo_effect"] if pk.get("hvo") else 0.0))
    g_b100 = base * (1 + NOX["b100_effect"])
    return g_hvo, g_b100, pen, scr


def main():
    groups, _ = sfs.load_years(); g30, _ = sfs.load_years(0.30)
    eb, eh = engine_series(groups, ALW)
    ny = len(eh) / 8760
    eng = eb + eh
    months = np.concatenate([pd.date_range(f"{g['year']}-01-01", periods=8760, freq="h").month.to_numpy() for g in groups])
    prof = dict(engine_twh=eng.sum() / 1e6 / ny, hvo_twh=eh.sum() / 1e6 / ny, b100_twh=eb.sum() / 1e6 / ny,
                hours_per_year=float((eng > 1).sum() / ny), peak_mw=float(eng.max()),
                mean_mw_when_running=float(eng[eng > 1].mean()),
                share_nov_feb=float(eng[np.isin(months, [11, 12, 1, 2])].sum() / eng.sum()),
                monthly_twh=(pd.Series(eng).groupby(months).sum() / 1e6 / ny).tolist())
    rows = []
    for name, pk in PACKAGES.items():
        g_h, g_b, pen, scr = factor(pk)
        nox_t = (eh.sum() * g_h + eb.sum() * g_b) / 1e3 / ny                 # MWh x g/kWh = kg; /1e3 -> t
        peak_kg_h = eng.max() * max(g_h, g_b)                                  # MWh/h x g/kWh = kg/h
        # fuel-ceiling check: the in-cylinder fuel penalty shrinks the energy a litre delivers
        eff_alw = ALW / (1 + pen)
        st30 = sfs.run_case(groups, W, sw["solar_GW"], sw["battery_GWh"], eff_alw)
        worst_l = st30["hvo_ml_max"] * (1 + pen) + FARM if pen > 0 else st30["hvo_ml_max"] + FARM
        n2o_t = (eng.sum() * NOX["n2o_scr_g_kwh"] / 1e3 / ny) if scr > 0 else 0.0
        kw = NOX["engine_mw"] * 1e3
        capex = (kw * NOX["scr_capex_cad_kw"][pk["scr"]] if scr else 0) + (kw * NOX["egr_capex_cad_kw"] if (pk.get("egr") or pk.get("timing")) else 0) \
            + (kw * NOX["emulsion_capex_cad_kw"] if pk.get("emulsion") else 0)
        uncontrolled_t = (eh.sum() * NOX["baseline_g_kwh"] * (1 + (NOX["hvo_effect"] if pk.get("hvo") else 0)) + eb.sum() * NOX["baseline_g_kwh"] * 1.1) / 1e3 / ny
        reduced_t = uncontrolled_t - nox_t
        def_cost = reduced_t * 1e3 * NOX["def_kg_per_kg_nox"] * NOX["def_cad_per_kg"] if scr else 0.0
        fuel_cost = sw["hvo_ml_mean"] * 1e6 * pen * NOX["hvo_cad_l"]
        annual = capex * (NOX["crf"] + NOX["om_frac"]) + def_cost + fuel_cost
        base_t = (eh.sum() * NOX["baseline_g_kwh"] + eb.sum() * NOX["baseline_g_kwh"] * 1.1) / 1e3 / ny
        rows.append(dict(package=name, g_kwh_hvo=g_h, nox_t_yr=nox_t, vs_gas_twin=nox_t / (LOAD_TWH * 1e6 * NOX["gas_twin_g_kwh"] / 1e3),
                         reduction_vs_uncontrolled_pct=100 * (1 - nox_t / base_t), peak_kg_h=peak_kg_h, fuel_penalty_pct=100 * pen,
                         worst_year_fuel_ml_selected_40pctcap=worst_l, gas_gwh_selected_40pctcap=st30["gas_gwh_max"], within_ceiling=bool(st30["gas_gwh_max"] < 1e-3 and worst_l <= 1000 + 1e-6),
                         n2o_t_yr=n2o_t, n2o_kt_co2e=n2o_t * NOX["gwp_n2o"] / 1e3, capex_m_cad=capex / 1e6, annual_cost_m_cad=annual / 1e6,
                         cad_per_t_nox_avoided=annual / max(base_t - nox_t, 1e-9),
                         cad_per_mwh_load=annual / (LOAD_TWH * 1e6)))
    df = pd.DataFrame(rows)
    gas_t = LOAD_TWH * 1e6 * NOX["gas_twin_g_kwh"] / 1e3
    # sensitivity of the recommended package to the engine-out baseline, HVO effect and SCR efficiency
    rec = "HVO + emulsion + SCR 97%"
    sens = []
    for b in NOX["baseline_range"] + (NOX["baseline_g_kwh"],):
        for he in NOX["hvo_range"] + (NOX["hvo_effect"],):
            for s in ("90%", "95%", "97%"):
                g = b * (1 + NOX["emulsion"]["nox"]) * (1 - NOX["scr"][s]) * (1 + he)   # emulsion + SCR at 90/95/97%
                sens.append(dict(baseline=b, hvo_effect=he, scr=s, nox_t_yr=(eh.sum() * g + eb.sum() * g / (1 + he) * 1.1) / 1e3 / ny))
    S = pd.DataFrame(sens)
    # what extra storage or solar would let the fuel-penalising packages stay gas-free at 30% wind
    fix = []
    pen = NOX["timing_retard"]["fuel"] + NOX["egr_10pct"]["fuel"]
    for solar, batt in ((20, 60), (25, 45), (25, 60), (30, 45)):
        st = sfs.run_case(g30, W, solar, batt, ALW / (1 + pen))
        stc = sfs.run_case(groups, W, solar, batt, ALW / (1 + pen))
        c = sfs.cost_25y(W.sum(), solar, batt, stc, ALW / (1 + pen))["lcoe_25y_cad_mwh"]
        fix.append(dict(solar_GW=solar, battery_GWh=batt, gas_gwh_30pct=st["gas_gwh_max"], gas_free=bool(st["gas_gwh_max"] < 1e-3 and stc["gas_gwh_max"] < 1e-3),
                        lcoe=c, extra_cad_mwh=c - sw["lcoe_25y_cad_mwh"]))
    print(pd.DataFrame(fix).round(2).to_string())
    res = dict(engine_profile=prof, timing_egr_repair=fix, packages=df.to_dict("records"), gas_twin_nox_t=gas_t, recommended=rec,
               recommended_range_t=[float(S.nox_t_yr.min()), float(S.nox_t_yr.max())], sensitivity=sens, params=NOX)
    (OUT / "nox_control_results.json").write_text(json.dumps(res, indent=1, default=float))
    df.to_csv(OUT / "nox_control_packages.csv", index=False, float_format="%.4f")
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in prof.items() if k != "monthly_twh"})
    print("gas twin NOx t/yr", round(gas_t))
    print(df.drop(columns=["cad_per_mwh_load"]).round(3).to_string())
    print("recommended range", res["recommended_range_t"])


if __name__ == "__main__":
    main()
