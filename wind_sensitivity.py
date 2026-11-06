#!/usr/bin/env python3
"""Wind-yield sensitivity of the storage-fuel sweet spot.

(a) The selected portfolio is replayed over all 16 weather years at fleet capacity factors of 25%, 30%, the calibrated
34.79% and 40%, and with the raw (unscaled) ERA5 100 m wind speeds. (b) The portfolio is re-optimised on a reduced grid at
25%, 30% and 40% with the same rules: no gas and no unserved energy in any year, and a data-centre renewable-fuel allowance
equal to the 1,000 ML/yr procurement ceiling less the host-community farm HVO.
"""
from pathlib import Path
import json, itertools
import numpy as np, pandas as pd
import storage_fuel_sweetspot as s
from dispatch_reference import wind_cf  # noqa: E402

HERE = Path(__file__).resolve().parent; OUT = HERE / "OUTPUT"
SS = json.load(open(OUT / "storage_fuel_sweetspot.json")); sw = SS["sweet_spot"]
sel = json.load(open(OUT / "optimization_provenance.json")); w0 = np.array(sel["wind_GW"])
ALW = SS["data_centre_fuel_allowance_ml"]


def raw_years():
    groups, wind = s.make_data(s.BASE, s.BASE / "RIDGE_WIND_ERA5_6_POINTS_2010_2025.zip")
    dates, _, _ = s.read_inputs(s.BASE, s.BASE / "RIDGE_WIND_ERA5_6_POINTS_2010_2025.zip")
    wcf = np.stack([wind_cf(w, 1.0) for w in wind]).astype("float64")
    for g in groups:
        m = ((dates.dt.year == g["year"]) & ~((dates.dt.month == 2) & (dates.dt.day == 29))).to_numpy()
        g["windcf"] = wcf[:, m]
    return groups, float(wcf.mean())


def evaluate(groups, ws, solar, batt, fuel):
    st = s.run_case(groups, np.minimum(w0 * ws, 2.0), solar, batt, fuel)
    c = s.cost_25y(np.minimum(w0 * ws, 2.0).sum(), solar, batt, st, fuel)
    return {**st, "lcoe_25y_cad_mwh": c["lcoe_25y_cad_mwh"], "capex_bn": c["capex_bn"]}


def main():
    res = {"fixed_portfolio": [], "reoptimised": []}
    cases = [(0.25, "25%"), (0.30, "30%"), (None, "34.79% (scaled assumption)"), (0.40, "40%")]
    loaded = {}
    for cf, lab in cases:
        g, ref = s.load_years(cf); loaded[lab] = g
        r = evaluate(g, sw["wind_scale"], sw["solar_GW"], sw["battery_GWh"], sw["fuel_cap_ML"])
        res["fixed_portfolio"].append(dict(wind_cf=lab, **r)); print(lab, {k: round(v, 2) for k, v in r.items()})
    graw, cfraw = raw_years()
    r = evaluate(graw, sw["wind_scale"], sw["solar_GW"], sw["battery_GWh"], sw["fuel_cap_ML"])
    res["fixed_portfolio"].append(dict(wind_cf=f"raw ERA5 ({100 * cfraw:.1f}%)", **r)); print("raw", round(cfraw, 3), {k: round(v, 2) for k, v in r.items()})
    grid = list(itertools.product([0.8, 1.0, 1.0435], [15, 20, 25, 30, 40, 50], [20, 30, 45, 60, 89, 120], [600, 800, ALW]))
    for lab in ("25%", "30%", "40%"):
        g = loaded[lab]; rows = []
        for ws, solar, batt, fuel in grid:
            r = evaluate(g, ws, solar, batt, fuel)
            rows.append(dict(wind_scale=ws, wind_GW=float(np.minimum(w0 * ws, 2.0).sum()), solar_GW=solar, battery_GWh=batt, fuel_cap_ML=fuel,
                             ok=r["gas_gwh_max"] < 1e-3 and r["unserved_gwh_max"] < 1e-3, **r))
        df = pd.DataFrame(rows); q = df[df.ok]
        best = q.sort_values("lcoe_25y_cad_mwh").iloc[0].to_dict() if len(q) else None
        res["reoptimised"].append(dict(wind_cf=lab, n_qualifying=int(len(q)), best=best))
        print(lab, "qualifying", len(q), None if best is None else {k: (round(v, 2) if isinstance(v, float) else v) for k, v in best.items()})
    (OUT / "wind_sensitivity.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
