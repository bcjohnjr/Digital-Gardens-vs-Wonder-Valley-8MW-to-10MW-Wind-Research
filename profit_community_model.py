#!/usr/bin/env python3
"""Digital Gardens 25-year profit, gas comparison, emissions per GW of IT, community GHG and hourly thermal balance.

Reads the storage/fuel sweet-spot results (storage_fuel_sweetspot.py) and the 17-centre ERA5 temperatures. Every input
is a declared screening assumption in PARAM below; sources are given beside each value. US$ for compute economics,
CAD for energy and community accounts (1.40 CAD/USD). All values are real 2026 money.
"""
from pathlib import Path
import json, gzip, itertools
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "OUTPUT"
WX = HERE / "DATA" / "WEATHER"
SS = json.load(open(OUT / "storage_fuel_sweetspot.json"))
FX = 1.40
YEARS = 25
LOAD_TWH = 36.00486
IT_TWH = 32.861982366
IT_GW = IT_TWH / 8.76
PUE = 1.10

PARAM = {
    # ---------------- gas comparator (matched 36.0 TWh/yr electricity) ----------------
    "gas": dict(gw=6.0, capex_cad_kw=1500.0, fom_cad_kw_yr=25.0, vom_cad_mwh=12.0, efficiency=0.57, co2_t_per_gj=0.0561,
                upstream_share=0.20, finance_rate=0.06, finance_years=20,
                price_paths={"low_flat_3": (3.0, 0.0), "central_4_rising_3pct": (4.0, 0.03), "high_6_rising_4pct": (6.0, 0.04)},
                # Alberta TIER headline price (Canada-Alberta implementation agreement, May 2026): 95 (2026), 100 (2027-29),
                # 115 (2030) ... 130 (2035), +1.5%/yr to 140 (2040); held flat in real terms afterwards
                tier_price={2026: 95, 2027: 100, 2028: 100, 2029: 100, 2030: 115, 2031: 118, 2032: 121, 2033: 124, 2034: 127,
                            2035: 130, 2036: 132, 2037: 134, 2038: 136, 2039: 138, 2040: 140},
                # TIER electricity high-performance benchmark (t CO2e/MWh): Alberta's published schedule gives 0.3404 for 2026 and
                # 0.3330 for 2027; the Canada-Alberta implementation agreement tightens it further each year. Policy scenario used
                # here: -1%/yr from 2027 to 2040, then held; a stricter -2%/yr case is reported as a sensitivity.
                tier_benchmark={2026: 0.3404, 2027: 0.3330}, tier_benchmark_decline=0.01, tier_benchmark_hold_from=2040,
                ccs=dict(start=2035, capture=0.90, capex_cad_kw=2000.0, finance_years=15, fuel_penalty=0.15, var_cad_t=75.0,
                         transport_storage_cad_t=20.0)),
    # ---------------- emissions factors (life cycle, g CO2e/kWh or g/MJ) ----------------
    "lca": dict(wind_g_kwh=11.0, solar_g_kwh=35.0, battery_kg_per_kwh=70.0, battery_life=15, hvo_g_mj=25.0, hvo_lhv_mj_l=36.51,
                b100_mt=0.02 * 0.9 * 35.4 * 30.0 / 1000),
    # ---------------- compute economics (US$) ----------------
    "compute": dict(
        products={
            "GB300 NVL72": dict(kw_per_gpu=135.0 / 72 * 1.10, capex_per_gpu=6.0e6 / 72, price_factor=1.10,
                                note="135 kW and US$6.0 M per 72-GPU rack incl. core network; +10% non-rack IT; rack-scale premium +10% (assumption)"),
            "HGX B300": dict(kw_per_gpu=14.0 / 8 * 1.10, capex_per_gpu=0.54e6 / 8, price_factor=1.00,
                             note="~14 kW per 8-GPU server; US$0.54 M per server incl. networking and storage (assumption)")},
        price_start={"spot (conservative)": 5.87, "market on-demand": 8.98, "premium tier": 12.0, "top tier": 17.80},
        price_step=-1.0, cycle_years=4, generation_decline={"prices held each generation": 0.0, "prices compress 15% per generation": 0.15}, residual=0.20, sold=0.85,
        label_premium=[0.0, 0.25, 0.5, 1.0, 2.0], label_eligible_share=0.5, label_admin_usd=3.0e6,
        facility_capex_usd_mw=10.0e6, maintenance_frac=0.025, insurance_frac=0.005, fte_per_mw=0.25, usd_per_fte=90000.0,
        network_ops_usd_mw=50000.0, tax=0.27, wacc=0.09),
    # ---------------- host communities (2021 Census populations, rounded) ----------------
    "population": {"lillooet": 2302, "cache_creek": 969, "kamloops": 97902, "chase": 2399, "salmon_arm": 19432, "vernon": 44519,
                   "kelowna": 144576, "west_kelowna": 36078, "penticton": 36885, "oliver": 5094, "osoyoos": 5556, "hope": 6686,
                   "agassiz_kent": 6300, "chilliwack": 93203, "abbotsford": 153524, "mission": 41519, "maple_ridge": 90990},
    "ev": dict(vehicles=5000, cad_per_vehicle=35000.0, km_yr=15000.0, l_per_100km=8.9, kg_co2_per_l=2.31, upstream_share=0.20,
               kwh_per_100km=18.0, vehicles_per_person=0.68),     # programme as in the companion manuscript; gasoline 2.31 kg CO2/L
    "fraser_valley_nodes": ["hope", "agassiz_kent", "chilliwack", "abbotsford", "mission", "maple_ridge"],
    "heat": dict(persons_per_household=2.4, connected_share=0.60, gas_heated_share=0.60, ua_kw_per_k=0.20, base_c=15.5,
                 dhw_l_person_day=50.0, dhw_c=45.0, mains_c=8.0, furnace_eff=0.92, water_heater_eff=0.62, network_loss=0.15,
                 boost_below_c=-5.0, boost_share=0.30, booster_cop=3.5, gas_t_per_gj=0.0499, gas_upstream_share=0.10,
                 capex_cad_household=18000.0, om_frac=0.015, life=40, gas_price_cad_gj=13.0, tariff_share_of_avoided_bill=0.70,
                 recoverable_heat_share_it=0.85),
    # ---------------- livestock (BC Ag in Brief, 2021 Census of Agriculture) ----------------
    "livestock": dict(fv_dairy_cows=49159, fv_pigs=66965, tnrd_beef_cows=38848,
                      enteric_kg_ch4=dict(dairy=141.0, beef=113.0), gwp_ch4=28.0,
                      manure_kg_ch4_baseline=dict(dairy=45.0, pig=10.0),
                      ad_storage_reduction=0.80, fugitive_baseline=0.025, fugitive_with_sensors=0.005,
                      biogas_m3_ch4_yr=dict(dairy=394.0, pig=44.0), plant_waste_t=100000.0, plant_waste_m3_ch4_t=60.0,
                      ch4_mj_m3=35.8, digester_heat_share=0.20, rng_cad_gj=25.0,
                      ad_capex_cad_per_gj_yr=60.0, ad_om_frac=0.05, sensor_sites=20, sensor_capex_cad=50000.0, sensor_om_cad=10000.0,
                      farm_diesel_l=dict(dairy=400.0, beef=100.0), diesel_kg_co2e_l=3.2, hvo_reduction=0.75),
    "asparagopsis": dict(dmi_kg_day=dict(dairy=21.0, beef=12.0), dose_dm=0.005, bromoform_retention=0.70,
                         coverage=dict(dairy=0.90, beef=150 / 365), reduction=dict(dairy=0.30, beef=0.68),
                         reduction_best=dict(dairy=0.50, beef=0.68), productivity_g_dw_m2_day=12.0, capex_cad_m2=250.0, om_frac=0.08,
                         milk_kg_day=33.0, milk_loss=0.07, milk_cad_kg=0.95, regulatory="not yet registered as a livestock feed in Canada (CFIA)"),
    "drying": dict(water_t={"digestate_solids": 107000.0, "poultry_litter": 70000.0, "hay_forage": 50000.0, "fruit_berries": 40000.0},
                   kwh_th_per_kg_water=0.68 / 0.50, displaced_fossil_share=0.50, service_value_cad_t_water=60.0),
}


PARAM_DRY_RATE = 0.06


def crf(r, n):
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


# ============================================================ 25-year electricity: Digital Gardens vs gas
def tier_benchmark(yr):
    g = PARAM["gas"]; b = g["tier_benchmark"]
    if yr in b:
        return b[yr]
    n = min(yr, g["tier_benchmark_hold_from"]) - 2027
    return b[2027] * (1 - g["tier_benchmark_decline"]) ** n


def gas_paths():
    g = PARAM["gas"]; hr = 3.6 / g["efficiency"]
    mwh = LOAD_TWH * 1e6
    co2_mwh = hr * g["co2_t_per_gj"]
    capex = g["gw"] * 1e6 * g["capex_cad_kw"]
    debt = capex * crf(g["finance_rate"], g["finance_years"])
    fom = g["gw"] * 1e6 * g["fom_cad_kw_yr"]
    out = {}
    for pname, (p0, gr) in g["price_paths"].items():
        for carbon in ("tier_output_based", "full_headline_price", "ccs_from_2035"):
            years = []
            for i in range(YEARS):
                yr = 2027 + i; price = p0 * (1 + gr) ** i
                cp = g["tier_price"].get(yr, 140)
                fuel_gj = mwh * hr; direct = mwh * co2_mwh; extra = 0.0
                if carbon == "ccs_from_2035" and yr >= g["ccs"]["start"]:
                    c = g["ccs"]
                    fuel_gj *= 1 + c["fuel_penalty"]; captured = direct * (1 + c["fuel_penalty"]) * c["capture"]
                    direct = direct * (1 + c["fuel_penalty"]) * (1 - c["capture"])
                    if yr < c["start"] + c["finance_years"]:
                        extra += g["gw"] * 1e6 * c["capex_cad_kw"] * crf(g["finance_rate"], c["finance_years"])
                    extra += captured * (c["var_cad_t"] + c["transport_storage_cad_t"])
                    carbon_cost = 0.0
                elif carbon == "full_headline_price":
                    carbon_cost = direct * cp
                else:   # TIER output-based: pays only on intensity above the benchmark
                    carbon_cost = max(0.0, direct / mwh - tier_benchmark(yr)) * mwh * cp
                cost = (debt if i < g["finance_years"] else 0.0) + fom + mwh * g["vom_cad_mwh"] + fuel_gj * price + carbon_cost + extra
                years.append(dict(year=yr, gas_cad_gj=price, carbon_cad_t=cp, tier_benchmark_t_mwh=tier_benchmark(yr), cost_bn=cost / 1e9, cad_mwh=cost / mwh,
                                  direct_mt=direct / 1e6, upstream_mt=fuel_gj * g["co2_t_per_gj"] * g["upstream_share"] / 1e6))
            df = pd.DataFrame(years)
            r = g["finance_rate"]
            pv = sum(c / (1 + r) ** (i + 1) for i, c in enumerate(df.cost_bn))
            pvm = sum(LOAD_TWH * 1e6 / (1 + r) ** (i + 1) for i in range(YEARS))
            out[f"{pname}|{carbon}"] = dict(lcoe_25y_cad_mwh=pv * 1e9 / pvm, yr1_cad_mwh=df.cad_mwh.iloc[0], yr25_cad_mwh=df.cad_mwh.iloc[-1],
                                            mean_direct_mt=df.direct_mt.mean(), mean_upstream_mt=df.upstream_mt.mean(), path=df.to_dict("records"))
    return out


def dg_paths():
    out = {}
    for key in ("reference_selected", "sweet_spot", "low_fuel_sweet_spot"):
        d = SS[key]
        path = d.get("annual_path_bn")
        if path is None:   # low-fuel case: rebuild the path from its components
            path = [d["om_bn"] + d["fuel_bn"] + d["carry_bn"] + (d["annual_debt_bn"] if y <= 20 else 0.0)
                    + (d["battery_GWh"] * 0.31 * 0.60 if y == 15 else 0.0) for y in range(1, YEARS + 1)]
        out[key] = dict(lcoe_25y_cad_mwh=d["lcoe_25y_cad_mwh"], yr1_cad_mwh=path[0] * 1e3 / LOAD_TWH, yr21_cad_mwh=path[20] * 1e3 / LOAD_TWH,
                        path_cad_mwh=[c * 1e3 / LOAD_TWH for c in path], config={k: d[k] for k in ("wind_GW", "solar_GW", "battery_GWh", "fuel_cap_ML",
                                                                                                  "hvo_ml_mean", "hvo_ml_max", "capex_bn")})
    return out


def breakeven_gas_flat(dg_lcoe, carbon="tier_output_based"):
    """Constant real gas price (C$/GJ) over 25 years that makes the gas cost equal Digital Gardens'."""
    lo, hi = 0.0, 200.0
    g = PARAM["gas"]
    for _ in range(60):
        m = (lo + hi) / 2
        g["price_paths"]["_probe"] = (m, 0.0)
        v = gas_paths()[f"_probe|{carbon}"]["lcoe_25y_cad_mwh"]
        lo, hi = (m, hi) if v < dg_lcoe else (lo, m)
    g["price_paths"].pop("_probe")
    return (lo + hi) / 2


def breakeven_gas_growth(dg_lcoe, p0=4.0, carbon="tier_output_based"):
    """Real annual gas-price growth from C$4/GJ that makes the 25-year gas cost equal Digital Gardens'."""
    lo, hi = -0.05, 0.25
    g = PARAM["gas"]
    for _ in range(60):
        m = (lo + hi) / 2
        g["price_paths"]["_probe"] = (p0, m)
        v = gas_paths()[f"_probe|{carbon}"]["lcoe_25y_cad_mwh"]
        lo, hi = (m, hi) if v < dg_lcoe else (lo, m)
    g["price_paths"].pop("_probe")
    return (lo + hi) / 2


# ============================================================ emissions per GW of IT
def emissions_per_gw_it(gas):
    L = PARAM["lca"]
    facility_twh_per_gw_it = 8.76 * PUE
    res = {}
    g = PARAM["gas"]; hr = 3.6 / g["efficiency"]
    direct = facility_twh_per_gw_it * 1e6 * hr * g["co2_t_per_gj"] / 1e6
    res["Wonder Valley-type gas, unabated"] = dict(direct_mt=direct, upstream_mt=direct * g["upstream_share"], total_mt=direct * (1 + g["upstream_share"]))
    c = g["ccs"]; burn = direct * (1 + c["fuel_penalty"])
    res["Wonder Valley-type gas, 90% capture"] = dict(direct_mt=burn * (1 - c["capture"]), upstream_mt=burn * g["upstream_share"],
                                                       total_mt=burn * (1 - c["capture"]) + burn * g["upstream_share"])
    for key, label in (("reference_selected", "Digital Gardens, published portfolio"), ("sweet_spot", "Digital Gardens, storage-fuel sweet spot")):
        d = SS[key]
        wind_twh = d["wind_GW"] * SS["wind_benchmark_cf"] * 8.76
        solar_twh = max(d["gen_twh"] - wind_twh, 0.0)          # all built output, curtailed energy included
        mt = (wind_twh * L["wind_g_kwh"] + solar_twh * L["solar_g_kwh"]) / 1000 \
            + d["battery_GWh"] * 1e6 * L["battery_kg_per_kwh"] / L["battery_life"] / 1e9 \
            + d["hvo_ml_mean"] * 1e6 * L["hvo_lhv_mj_l"] * L["hvo_g_mj"] / 1e12 + L["b100_mt"]
        per_mwh = mt * 1e6 / (LOAD_TWH * 1e6)
        res[label] = dict(system_mt=mt, g_per_kwh=per_mwh * 1000, total_mt=per_mwh * facility_twh_per_gw_it)
    return res


# ============================================================ hourly heat: communities, digesters, aquaculture, drying
def load_temps():
    T = {}
    for k in PARAM["population"]:
        df = pd.read_csv(WX / f"{k}_hourly_era5.csv.gz", usecols=["time_utc", "temperature_2m"])
        t = pd.to_datetime(df.time_utc)
        m = ~((t.dt.month == 2) & (t.dt.day == 29))
        T[k] = (t[m].reset_index(drop=True), df.temperature_2m[m].to_numpy(dtype=float))
    return T


def community_and_heat(T):
    H = PARAM["heat"]; LV = PARAM["livestock"]; AS = PARAM["asparagopsis"]; DR = PARAM["drying"]
    nodes = list(PARAM["population"]); n = len(nodes)
    it_mw_node = IT_GW * 1000 / n                             # equal IT per node (notional siting)
    avail_mw = it_mw_node * H["recoverable_heat_share_it"] * (1 - H["network_loss"])
    # agricultural / aquaculture heat (manuscript's 4.819 TWh, winter-weighted) split equally
    t0, _ = T[nodes[0]]
    months = t0.dt.month.to_numpy(); hours = t0.dt.hour.to_numpy()
    season = np.array([1.6, 1.55, 1.35, 1.12, .85, .55, .43, .48, .70, 1.03, 1.35, 1.6])[months - 1]
    season = season * (1 + .08 * np.cos((hours - 5) * 2 * np.pi / 24)); season = season / season.mean()
    nyears = len(t0) / 8760
    # biogas and digester heat (Fraser Valley nodes)
    ch4_m3 = LV["fv_dairy_cows"] * LV["biogas_m3_ch4_yr"]["dairy"] + LV["fv_pigs"] * LV["biogas_m3_ch4_yr"]["pig"] + LV["plant_waste_t"] * LV["plant_waste_m3_ch4_t"]
    biogas_gj = ch4_m3 * LV["ch4_mj_m3"] / 1000
    digester_mwh = biogas_gj * LV["digester_heat_share"] / 3.6
    drying_mwh = sum(DR["water_t"].values()) * DR["kwh_th_per_kg_water"]
    fv = PARAM["fraser_valley_nodes"]
    rows = []; total = dict(avail=0, space=0, dhw=0, digester=0, aqua=0, drying=0, booster_el=0, gas_gj_avoided=0, demand_space_dhw=0)
    for k in nodes:
        t, temp = T[k]
        pop = PARAM["population"][k]; hh = pop / H["persons_per_household"]; conn = hh * H["connected_share"]
        space = conn * H["ua_kw_per_k"] * np.maximum(0, H["base_c"] - temp) / 1000                       # MW_th
        dhw_mwh_yr = conn * H["persons_per_household"] * H["dhw_l_person_day"] * 365 * 4.186 * (H["dhw_c"] - H["mains_c"]) / 3.6e6
        dhw = np.full(len(temp), dhw_mwh_yr / 8760)
        boost = np.where(temp < H["boost_below_c"], space * H["boost_share"], 0.0)                         # needs heat-pump lift
        dig = np.full(len(temp), (digester_mwh / len(fv)) / 8760 if k in fv else 0.0)
        aqua = 4.819e6 / n / 8760 * season
        avail = np.full(len(temp), avail_mw)
        d1 = space + dhw; s1 = np.minimum(avail, d1); rem = avail - s1
        s2 = np.minimum(rem, dig); rem -= s2
        s3 = np.minimum(rem, aqua); rem -= s3
        dry_target = drying_mwh * (0.7 / len(fv) if k in fv else 0.3 / (n - len(fv))) * nyears
        s4 = min(rem.sum(), dry_target)                                                                     # storable products dried in surplus hours
        gas_avoided_gj = (s1.sum() * (space.sum() / max(d1.sum(), 1e-9)) / H["furnace_eff"]
                          + s1.sum() * (dhw.sum() / max(d1.sum(), 1e-9)) / H["water_heater_eff"]) * 3.6 * H["gas_heated_share"]
        r = dict(node=k, population=pop, connected_households=conn, it_mw=it_mw_node, avail_twh=avail.sum() / 1e6 / nyears,
                 space_dhw_demand_twh=d1.sum() / 1e6 / nyears, space_dhw_served_twh=s1.sum() / 1e6 / nyears,
                 digester_twh=s2.sum() / 1e6 / nyears, aqua_twh=s3.sum() / 1e6 / nyears, drying_twh=s4 / 1e6 / nyears,
                 booster_el_twh=np.minimum(boost, s1).sum() / H["booster_cop"] / 1e6 / nyears, gas_gj_avoided=gas_avoided_gj / nyears,
                 home_heat_gj_per_household=(space.sum() + dhw.sum()) * 3.6 / max(conn, 1) / nyears * 1000 / 1000)
        rows.append(r)
    df = pd.DataFrame(rows)
    useful = df.space_dhw_served_twh.sum() + df.digester_twh.sum() + df.aqua_twh.sum() + df.drying_twh.sum()
    recoverable = IT_TWH * H["recoverable_heat_share_it"]
    thermal = dict(it_heat_twh=IT_TWH, recoverable_twh=recoverable, delivered_to_network_twh=df.avail_twh.sum(),
                   useful_twh=useful, space_dhw_twh=df.space_dhw_served_twh.sum(), digester_twh=df.digester_twh.sum(),
                   aquaculture_greenhouse_twh=df.aqua_twh.sum(), drying_twh=df.drying_twh.sum(), booster_el_twh=df.booster_el_twh.sum(),
                   utilisation_of_recoverable_pct=100 * useful / recoverable, erf_pct=100 * useful / LOAD_TWH,
                   space_dhw_coverage_pct=100 * df.space_dhw_served_twh.sum() / df.space_dhw_demand_twh.sum())
    # ---- community GHG (t CO2e/yr) ----
    gwp = LV["gwp_ch4"]
    g_home = df.gas_gj_avoided.sum() * H["gas_t_per_gj"] * (1 + H["gas_upstream_share"])
    booster_t = 0.0                                                                                      # booster power from Digital Gardens supply
    cows, beef = LV["fv_dairy_cows"], LV["tnrd_beef_cows"]
    def enteric(case):
        red = AS["reduction" if case == "central" else "reduction_best"]
        return (cows * LV["enteric_kg_ch4"]["dairy"] * red["dairy"] * AS["coverage"]["dairy"]
                + beef * LV["enteric_kg_ch4"]["beef"] * red["beef"] * AS["coverage"]["beef"]) * gwp / 1000
    manure_base = (cows * LV["manure_kg_ch4_baseline"]["dairy"] + LV["fv_pigs"] * LV["manure_kg_ch4_baseline"]["pig"]) * gwp / 1000
    manure_red = manure_base * LV["ad_storage_reduction"]
    ch4_kg = ch4_m3 * 0.668
    fug_no_sensor = ch4_kg * LV["fugitive_baseline"] * gwp / 1000
    fug_sensor = ch4_kg * LV["fugitive_with_sensors"] * gwp / 1000
    rng_gj = biogas_gj * (1 - LV["digester_heat_share"]) if True else 0.0
    rng_gj = biogas_gj * (1 - LV["fugitive_with_sensors"])                                             # digester heat comes from server heat
    rng_t = rng_gj * H["gas_t_per_gj"] * (1 + H["gas_upstream_share"])
    diesel_l = cows * LV["farm_diesel_l"]["dairy"] + beef * LV["farm_diesel_l"]["beef"]
    diesel_t = diesel_l * LV["diesel_kg_co2e_l"] / 1000 * LV["hvo_reduction"]
    drying_t = drying_mwh * DR["displaced_fossil_share"] * 3.6 / 0.85 * H["gas_t_per_gj"]
    ghg = {"Home heating and hot water from server heat (gas displaced)": g_home,
           "Asparagopsis for dairy and beef cattle (enteric methane)": enteric("central"),
           "Manure removed to digesters (storage methane avoided)": manure_red,
           "Renewable natural gas displacing fossil gas": rng_t,
           "Methane-leak sensors on digesters and upgraders (fugitive avoided vs no sensors)": fug_no_sensor - fug_sensor,
           "Farm diesel replaced by HVO": diesel_t,
           "Drying with server heat (fossil dryers displaced)": drying_t}
    ghg_best = dict(ghg); ghg_best["Asparagopsis for dairy and beef cattle (enteric methane)"] = enteric("best")
    baseline = {"Home heating and hot water (connected gas-heated homes)": g_home,
                "Enteric methane (Fraser Valley dairy cows, Thompson-Nicola beef cows)": (cows * LV["enteric_kg_ch4"]["dairy"] + beef * LV["enteric_kg_ch4"]["beef"]) * gwp / 1000,
                "Manure storage methane (Fraser Valley dairy and pigs)": manure_base,
                "Farm diesel (dairy and beef)": diesel_l * LV["diesel_kg_co2e_l"] / 1000,
                "Fossil drying heat displaced": drying_t}
    # Asparagopsis supply
    dw_t = (cows * AS["dmi_kg_day"]["dairy"] * AS["coverage"]["dairy"] + beef * AS["dmi_kg_day"]["beef"] * AS["coverage"]["beef"]) * 365 \
        * AS["dose_dm"] / AS["bromoform_retention"] / 1000
    area_m2 = dw_t * 1e6 / (AS["productivity_g_dw_m2_day"] * 365)
    milk_comp = cows * AS["coverage"]["dairy"] * AS["milk_kg_day"] * 365 * AS["milk_loss"] * AS["milk_cad_kg"] * 0.6   # 60% of herd lactating
    # costs and revenues (CAD/yr)
    conn_total = df.connected_households.sum()
    hn_capex = conn_total * H["capex_cad_household"]
    heat_rev = df.gas_gj_avoided.sum() / H["gas_heated_share"] * H["gas_price_cad_gj"] * H["tariff_share_of_avoided_bill"]
    ad_capex = rng_gj * LV["ad_capex_cad_per_gj_yr"]
    rng_rev = rng_gj * LV["rng_cad_gj"]
    asp_capex = area_m2 * AS["capex_cad_m2"]
    sensors = LV["sensor_sites"] * LV["sensor_capex_cad"]
    econ = dict(hot_water_capex_bn=hn_capex / 1e9, hot_water_annual_cost_bn=(hn_capex * crf(0.06, H["life"]) + hn_capex * H["om_frac"]) / 1e9,
                heat_revenue_bn=heat_rev / 1e9, ad_capex_bn=ad_capex / 1e9,
                ad_annual_cost_bn=(ad_capex * crf(0.06, 20) + ad_capex * LV["ad_om_frac"]) / 1e9, rng_revenue_bn=rng_rev / 1e9,
                asparagopsis_capex_bn=asp_capex / 1e9, asparagopsis_annual_cost_bn=(asp_capex * crf(0.06, 15) + asp_capex * AS["om_frac"] + milk_comp) / 1e9,
                sensors_annual_cost_bn=(sensors * crf(0.06, 10) + LV["sensor_sites"] * LV["sensor_om_cad"]) / 1e9,
                drying_service_value_bn=sum(DR["water_t"].values()) * DR["service_value_cad_t_water"] / 1e9,
                hvo_for_farms_cost_bn=diesel_l * 0.6 / 1e9)          # HVO premium over diesel ~C$0.60/L, paid by the project
    econ["net_annual_bn"] = (econ["heat_revenue_bn"] + econ["rng_revenue_bn"] - econ["hot_water_annual_cost_bn"] - econ["ad_annual_cost_bn"]
                             - econ["asparagopsis_annual_cost_bn"] - econ["sensors_annual_cost_bn"] - econ["hvo_for_farms_cost_bn"])
    community = dict(population=int(df.population.sum()), connected_households=conn_total, ghg_reduction_t=ghg, ghg_reduction_best_t=ghg_best,
                     total_reduction_mt=sum(ghg.values()) / 1e6, total_reduction_best_mt=sum(ghg_best.values()) / 1e6, baseline_covered_t=baseline,
                     baseline_covered_mt=sum(baseline.values()) / 1e6, per_capita_reduction_t=sum(ghg.values()) / df.population.sum(),
                     bc_average_per_capita_t=61.1e6 / 5.52e6, asparagopsis_dw_t=dw_t, asparagopsis_area_ha=area_m2 / 1e4,
                     biogas_gj=biogas_gj, rng_gj=rng_gj, economics=econ)
    return df, thermal, community


# ============================================================ GPU profit over 25 years
def gpu_npv(product, tier, label, elec_cad_mwh_path, community_net_bn=0.0, community_capex_bn=0.0, decline=0.0,
            label_share=None, sold=None, capex_mult=1.0, price_mult=1.0, build_years=1):
    """25-year NPV of the GPU rental business. label_share: share of sold hours that carry the label premium.
    build_years > 1 phases the IT fleet (and its electricity) in equal steps; capex_mult scales GPU capital; price_mult scales prices."""
    C = PARAM["compute"]; pr = C["products"][product]
    label_share = C["label_eligible_share"] if label_share is None else label_share
    sold = C["sold"] if sold is None else sold
    gpus_full = IT_GW * 1e6 / pr["kw_per_gpu"]
    it_mw = IT_GW * 1000
    gpu_capex_full = gpus_full * pr["capex_per_gpu"] * capex_mult
    fac_capex = it_mw * C["facility_capex_usd_mw"]
    r = C["wacc"]
    ramp = [min(1.0, (k + 1) / build_years) for k in range(YEARS + 1)]       # fleet share installed in year k (0 = construction)
    cash = [-(fac_capex + gpu_capex_full * ramp[0] + community_capex_bn * 1e9 / FX)]
    rows = []
    gpus = gpus_full
    for y in range(1, YEARS + 1):
        frac = ramp[y - 1]; gpu_capex = gpu_capex_full * frac
        cy = (y - 1) % C["cycle_years"]
        price = max(C["price_start"][tier] * price_mult * (1 - decline) ** ((y - 1) // C["cycle_years"]) + C["price_step"] * cy, 0.5) * pr["price_factor"]
        hours = gpus_full * frac * 8760 * sold
        rev = hours * price + hours * label_share * label
        elec = elec_cad_mwh_path[y - 1] * LOAD_TWH * 1e6 / FX * frac
        opex = (gpu_capex * C["maintenance_frac"] + (gpu_capex + fac_capex) * C["insurance_frac"] + it_mw * C["fte_per_mw"] * C["usd_per_fte"]
                + it_mw * C["network_ops_usd_mw"] + (C["label_admin_usd"] if label > 0 else 0.0))
        ebitda = rev - elec - opex + community_net_bn * 1e9 / FX
        end_cycle = cy == C["cycle_years"] - 1 or y == YEARS
        frac_used = (cy + 1) / C["cycle_years"]
        resid = gpu_capex * (1 - (1 - C["residual"]) * frac_used) if end_cycle else 0.0
        capex = gpu_capex if (end_cycle and y < YEARS) else 0.0                 # refresh bought at the end of each cycle
        if build_years > 1 and y < build_years:
            capex += gpu_capex_full * (ramp[y] - ramp[y - 1])                    # next tranche of the build-out
        dep = gpu_capex * (1 - C["residual"]) / C["cycle_years"] + fac_capex / YEARS
        tax = max(0.0, (ebitda - dep) * C["tax"])
        cf = ebitda - tax - capex + resid
        cash.append(cf)
        rows.append(dict(year=y, price=price, revenue_bn=rev / 1e9, electricity_bn=elec / 1e9, ebitda_bn=ebitda / 1e9, capex_bn=capex / 1e9, cash_bn=cf / 1e9))
    npv = sum(c / (1 + r) ** i for i, c in enumerate(cash))
    try:
        irr = np.irr(cash)  # type: ignore[attr-defined]
    except Exception:
        lo, hi = -0.9, 20.0
        f = lambda x: sum(c / (1 + x) ** i for i, c in enumerate(cash))
        if f(lo) * f(hi) > 0:
            irr = float("nan")
        else:
            for _ in range(200):
                m = (lo + hi) / 2
                lo, hi = (m, hi) if f(m) > 0 else (lo, m)
            irr = (lo + hi) / 2
    df = pd.DataFrame(rows)
    payback = next((i for i in range(1, len(cash)) if sum(cash[:i + 1]) > 0), None)
    return dict(product=product, tier=tier, label=label, gpus=gpus, payback_years=payback, elec_usd_per_gpu_h=df.electricity_bn.mean() * 1e9 / (gpus * 8760 * sold), gpu_capex_bn=gpu_capex / 1e9, npv_bn=npv / 1e9, irr=irr,
                mean_revenue_bn=df.revenue_bn.mean(), mean_ebitda_bn=df.ebitda_bn.mean(), mean_electricity_bn=df.electricity_bn.mean())


def main():
    tier_sens = {}
    for d_ in (0.0, 0.01, 0.02):
        PARAM["gas"]["tier_benchmark_decline"] = d_
        tier_sens[f"benchmark -{int(d_ * 100)}%/yr after 2027"] = gas_paths()["central_4_rising_3pct|tier_output_based"]["lcoe_25y_cad_mwh"]
    PARAM["gas"]["tier_benchmark_decline"] = 0.01
    gas = gas_paths(); dg = dg_paths()
    T = load_temps()
    nodes_df, thermal, community = community_and_heat(T)
    # commercial drying contracts (industrial_drying.py) replace the simple drying-barter screen
    DRY = json.load(open(OUT / "industrial_drying_results.json"))
    dce = DRY["central"]["economics"]
    OLD = "Drying with server heat (fossil dryers displaced)"; NEW = "Commercial drying contracts with server heat (gas and propane displaced)"
    for key in ("ghg_reduction_t", "ghg_reduction_best_t"):
        community[key] = {k: v for k, v in community[key].items() if k != OLD}; community[key][NEW] = dce["ghg_reduction_t"]
    community["baseline_covered_t"]["Fossil drying heat displaced"] = dce["ghg_reduction_t"]
    community["baseline_covered_mt"] = sum(community["baseline_covered_t"].values()) / 1e6
    community["total_reduction_mt"] = sum(community["ghg_reduction_t"].values()) / 1e6
    community["total_reduction_best_mt"] = sum(community["ghg_reduction_best_t"].values()) / 1e6
    # electric vehicles: the 5,000-vehicle programme with free charging from the network (and a 10%-of-fleet sensitivity)
    EVP = PARAM["ev"]; dg_g_kwh = emissions_per_gw_it(gas)["Digital Gardens, storage-fuel sweet spot"]["g_per_kwh"]
    ice_t = EVP["km_yr"] * EVP["l_per_100km"] / 100 * EVP["kg_co2_per_l"] * (1 + EVP["upstream_share"]) / 1000
    ev_t = EVP["km_yr"] * EVP["kwh_per_100km"] / 100 * dg_g_kwh / 1e6
    ev_net = ice_t - ev_t
    KEV = "Electric vehicles charged from Digital Gardens (5,000-vehicle programme)"
    for key in ("ghg_reduction_t", "ghg_reduction_best_t"):
        community[key][KEV] = EVP["vehicles"] * ev_net
    fleet10 = community["population"] * EVP["vehicles_per_person"] * 0.10
    community["ev"] = dict(net_t_per_vehicle=ev_net, ice_t=ice_t, ev_t=ev_t, programme_vehicles=EVP["vehicles"], programme_t=EVP["vehicles"] * ev_net,
                           fleet10_vehicles=fleet10, fleet10_t=fleet10 * ev_net, charging_gwh=EVP["vehicles"] * EVP["km_yr"] * EVP["kwh_per_100km"] / 100 / 1e6,
                           fleet10_charging_gwh=fleet10 * EVP["km_yr"] * EVP["kwh_per_100km"] / 100 / 1e6, capex_bn=EVP["vehicles"] * EVP["cad_per_vehicle"] / 1e9)
    community["total_reduction_mt"] = sum(community["ghg_reduction_t"].values()) / 1e6
    community["total_reduction_best_mt"] = sum(community["ghg_reduction_best_t"].values()) / 1e6
    community["per_capita_reduction_t"] = community["total_reduction_mt"] * 1e6 / community["population"]
    # host-community emissions baseline: provincial gross emissions per resident (61.1 Mt / 5.52 million, 2023)
    community["baseline_total_mt"] = community["population"] * 61.1 / 5.52 / 1e6
    # deployment scenarios: the central case assumes full deployment of every measure
    g0 = community["ghg_reduction_t"]
    k_home = "Home heating and hot water from server heat (gas displaced)"; k_asp = "Asparagopsis for dairy and beef cattle (enteric methane)"
    half_dry = next(x for x in DRY["scale_sweep"] if x["scale"] == 0.5)["ghg_reduction_t"]
    limited = {k: v * 0.5 for k, v in g0.items()}; limited[k_asp] = 0.0; limited[NEW] = half_dry
    community["deployment_scenarios_mt"] = {
        "Full deployment (all measures, seaweed feed approved)": sum(g0.values()) / 1e6,
        "Full deployment without seaweed-feed approval": (sum(g0.values()) - g0[k_asp]) / 1e6,
        "Limited adoption (half of homes, farms and drying customers; no seaweed approval)": sum(limited.values()) / 1e6}
    community["deployment_limited_t"] = limited
    em = emissions_per_gw_it(gas)
    # electricity cost gap per GPU-hour and label premium needed
    kwh_per_gpu_h = {p: v["kw_per_gpu"] * LOAD_TWH / IT_TWH for p, v in PARAM["compute"]["products"].items()}
    gaps = {}
    for dk in ("reference_selected", "sweet_spot"):
        for gk in ("central_4_rising_3pct|tier_output_based", "central_4_rising_3pct|full_headline_price", "low_flat_3|tier_output_based",
                   "high_6_rising_4pct|full_headline_price", "central_4_rising_3pct|ccs_from_2035"):
            gap = dg[dk]["lcoe_25y_cad_mwh"] - gas[gk]["lcoe_25y_cad_mwh"]
            gaps[f"{dk}|{gk}"] = dict(gap_cad_mwh=gap, premium_usd_per_gpu_h={p: gap / 1000 * k / FX for p, k in kwh_per_gpu_h.items()})
    # profit grid: product x tier x label x electricity source
    sources = {"Digital Gardens sweet spot": dg["sweet_spot"]["path_cad_mwh"], "Digital Gardens published portfolio": dg["reference_selected"]["path_cad_mwh"],
               "Gas (central price, TIER)": [p["cad_mwh"] for p in gas["central_4_rising_3pct|tier_output_based"]["path"]],
               "Gas (central price, full carbon price)": [p["cad_mwh"] for p in gas["central_4_rising_3pct|full_headline_price"]["path"]],
               "Gas (central price, CCS from 2035)": [p["cad_mwh"] for p in gas["central_4_rising_3pct|ccs_from_2035"]["path"]]}
    ce = community["economics"]
    comm_capex = ce["hot_water_capex_bn"] + ce["ad_capex_bn"] + ce["asparagopsis_capex_bn"] + dce["capex_bn"] + community["ev"]["capex_bn"]
    comm_net_ex_capital = (ce["heat_revenue_bn"] + ce["rng_revenue_bn"] - ce["hot_water_capex_bn"] * PARAM["heat"]["om_frac"]
                           - ce["ad_capex_bn"] * PARAM["livestock"]["ad_om_frac"]
                           - (ce["asparagopsis_annual_cost_bn"] - ce["asparagopsis_capex_bn"] * crf(0.06, 15))
                           - ce["sensors_annual_cost_bn"] - ce["hvo_for_farms_cost_bn"]
                           + (dce["margin_m_cad_mean"] / 1e3 + dce["capex_bn"] * crf(PARAM_DRY_RATE, 20))   # drying margin before its capital charge
                           - community["ev"]["charging_gwh"] * 1e3 * dg["sweet_spot"]["lcoe_25y_cad_mwh"] / 1e9)          # free charging at system cost
    grid = []
    for (src, path), prod, tier, (mk, dec) in itertools.product(sources.items(), PARAM["compute"]["products"], PARAM["compute"]["price_start"],
                                                                PARAM["compute"]["generation_decline"].items()):
        labels = PARAM["compute"]["label_premium"] if src.startswith("Digital") else [0.0]
        for lab in labels:
            dgflag = src.startswith("Digital")
            r = gpu_npv(prod, tier, lab, path, comm_net_ex_capital if dgflag else 0.0, comm_capex if dgflag else 0.0, dec)
            r["electricity"] = src; r["market"] = mk
            grid.append(r)
    G = pd.DataFrame(grid)
    G.to_csv(OUT / "gpu_profit_grid.csv", index=False, float_format="%.6f")
    best = G[G.electricity.str.startswith("Digital")].sort_values("npv_bn", ascending=False).iloc[0]
    # ---- break-even label premium, solved from NPV (NPV is linear in the premium)
    dgp = sources["Digital Gardens sweet spot"]
    be = []
    for prod, tier, (mk, dec), gsrc in itertools.product(PARAM["compute"]["products"], PARAM["compute"]["price_start"], PARAM["compute"]["generation_decline"].items(),
                                                        [k for k in sources if k.startswith("Gas")]):
        gv = gpu_npv(prod, tier, 0.0, sources[gsrc], decline=dec)["npv_bn"]
        for share in (0.5, 1.0):
            n0 = gpu_npv(prod, tier, 1e-9, dgp, comm_net_ex_capital, comm_capex, dec, label_share=share)["npv_bn"]
            n1 = gpu_npv(prod, tier, 1.0, dgp, comm_net_ex_capital, comm_capex, dec, label_share=share)["npv_bn"]
            be.append(dict(product=prod, tier=tier, market=mk, gas_case=gsrc, labelled_share=share, npv_gap_bn=gv - n0, premium_usd_per_labelled_hour=(gv - n0) / (n1 - n0),
                           premium_equivalent_on_all_hours=(gv - n0) / (n1 - n0) * share))
    BE = pd.DataFrame(be); BE.to_csv(OUT / "label_breakeven.csv", index=False, float_format="%.5f")
    # ---- GPU business sensitivity (market price, prices held, label US$0.50 on all sold hours, sweet-spot power)
    sens = []
    cases = {"Central": {}, "GPU capital +38% (US$8.3 M per NVL72 rack, B300 pro rata)": dict(capex_mult=8.3 / 6.0), "Sold utilisation 70%": dict(sold=0.70),
             "Sold utilisation 60%": dict(sold=0.60), "Build-out over 3 years": dict(build_years=3), "Contract prices -20%": dict(price_mult=0.80),
             "All four adverse together": dict(capex_mult=8.3 / 6.0, sold=0.70, build_years=3, price_mult=0.80)}
    for prod in PARAM["compute"]["products"]:
        for tier in ("spot (conservative)", "market on-demand"):
            for name, kw in cases.items():
                for mk, dec in PARAM["compute"]["generation_decline"].items():
                    d = gpu_npv(prod, tier, 0.5, dgp, comm_net_ex_capital, comm_capex, dec, label_share=1.0, **kw)
                    gv = gpu_npv(prod, tier, 0.0, sources["Gas (central price, TIER)"], decline=dec, **kw)
                    sens.append(dict(product=prod, tier=tier, case=name, market=mk, npv_bn=d["npv_bn"], irr=d["irr"], gas_twin_npv_bn=gv["npv_bn"]))
    SENS = pd.DataFrame(sens); SENS.to_csv(OUT / "gpu_sensitivity.csv", index=False, float_format="%.5f")
    # ---- Digital Gardens versus Wonder Valley: energy, water and carbon ledger for the same host communities
    WVU_ = em["Wonder Valley-type gas, unabated"]; DGS_ = em["Digital Gardens, storage-fuel sweet spot"]
    n2o_mt = json.load(open(OUT / "nox_control_results.json"))["packages"][5]["n2o_kt_co2e"] / 1e3 if (OUT / "nox_control_results.json").exists() else 0.0
    gas_pj = LOAD_TWH * 3.6 / PARAM["gas"]["efficiency"]                          # PJ/yr of gas for the same 36.0 TWh
    full_it = 7.5 / PUE                                                            # Wonder Valley's announced 7.5 GW, as IT
    base = community["baseline_total_mt"]; red = community["deployment_scenarios_mt"]
    comparison = dict(
        it_gw=IT_GW, wonder_valley_full_it_gw=full_it, gas_pj_equal_it=gas_pj, gas_bcm_equal_it=gas_pj * 1e15 / 37.9e6 / 1e9,
        gas_pj_full_scale=7.5 * 8.76 * 3.6 / PARAM["gas"]["efficiency"], gas_bcm_full_scale=7.5 * 8.76 * 3.6 / PARAM["gas"]["efficiency"] * 1e15 / 37.9e6 / 1e9,
        water_gl_wonder_valley=6.0, water_gl_wonder_valley_certificate=24.0, water_gl_digital_gardens_cooling=0.0,
        community_baseline_mt=base,
        wv_facility_mt=WVU_["total_mt"] * IT_GW, wv_direct_mt=WVU_["direct_mt"] * IT_GW,
        dg_facility_mt=DGS_["total_mt"] * IT_GW, dg_n2o_mt=n2o_mt,
        reductions_full_mt=red["Full deployment (all measures, seaweed feed approved)"],
        reductions_no_seaweed_mt=red["Full deployment without seaweed-feed approval"],
        reductions_limited_mt=red["Limited adoption (half of homes, farms and drying customers; no seaweed approval)"])
    c_ = comparison
    c_["ledger_wv_mt"] = base + c_["wv_facility_mt"]
    c_["ledger_dg_full_mt"] = base + c_["dg_facility_mt"] + n2o_mt - c_["reductions_full_mt"]
    c_["ledger_dg_limited_mt"] = base + c_["dg_facility_mt"] + n2o_mt - c_["reductions_limited_mt"]
    c_["net_added_wv_mt"] = c_["wv_facility_mt"]
    c_["net_added_dg_full_mt"] = c_["dg_facility_mt"] + n2o_mt - c_["reductions_full_mt"]
    c_["net_added_dg_limited_mt"] = c_["dg_facility_mt"] + n2o_mt - c_["reductions_limited_mt"]
    c_["per_gw_it_wv_mt"] = WVU_["total_mt"]; c_["per_gw_it_dg_net_full_mt"] = c_["net_added_dg_full_mt"] / IT_GW
    c_["wv_full_scale_mt"] = WVU_["total_mt"] * full_it
    c_["dg_at_wv_scale_net_full_mt"] = c_["per_gw_it_dg_net_full_mt"] * full_it
    # ---- label disclosure: fossil, biogenic and life-cycle carbon
    sw = SS["sweet_spot"]; hvo_l = sw["hvo_ml_mean"] * 1e6
    # ---- air quality: renewable-diesel engines run in low-wind hours; gas twin runs all year
    eng_mwh = sw["hvo_ml_mean"] * 1e6 / 182.6 + 18e6 * 0.9 * 35.4 * 0.54 / 3600   # HVO plus B100 (MWh)
    AQ = dict(engine_twh=eng_mwh / 1e6,
              engine_nox_t=eng_mwh * 0.67e-3, engine_pm_t=eng_mwh * 0.04e-3,          # US EPA Tier 4 final, gensets >900 kW (g/kWh)
              gas_nox_t=LOAD_TWH * 1e6 * 0.025e-3, gas_pm_t=LOAD_TWH * 1e6 * 0.019e-3,  # CCGT with SCR (~2 ppm NOx); AP-42 PM ~0.0066 lb/MMBtu
              notes="engine factors are regulatory limits, not measurements; gas factors are typical for SCR-equipped combined cycle")
    label = dict(fossil_co2_operation_t=0.0, gas_hours_in_16_years=0, biogenic_co2_hvo_mt=hvo_l * 0.78 * 0.848 * 44 / 12 / 1e9,
                 biogenic_co2_b100_mt=18e6 * 0.88 * 0.77 * 44 / 12 / 1e9, lifecycle_g_kwh=em["Digital Gardens, storage-fuel sweet spot"]["g_per_kwh"])
    res = dict(gas_25y={k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in gas.items()},
               gas_paths={k: v["path"] for k, v in gas.items()}, digital_gardens_25y=dg,
               breakeven_gas_growth={k: breakeven_gas_growth(dg[k]["lcoe_25y_cad_mwh"]) for k in ("reference_selected", "sweet_spot")},
               breakeven_gas_growth_full_carbon={k: breakeven_gas_growth(dg[k]["lcoe_25y_cad_mwh"], carbon="full_headline_price") for k in ("reference_selected", "sweet_spot")},
               breakeven_gas_flat={c: {k: breakeven_gas_flat(dg[k]["lcoe_25y_cad_mwh"], c) for k in ("reference_selected", "sweet_spot")}
                                   for c in ("tier_output_based", "full_headline_price", "ccs_from_2035")},
               emissions_per_gw_it=em, label_premium_needed=gaps, kwh_per_gpu_hour=kwh_per_gpu_h, thermal=thermal, community=community,
               community_net_ex_capital_bn=comm_net_ex_capital, community_capex_bn=comm_capex,
               best_profit=best.to_dict(), it_gw=IT_GW, pue=PUE, label_carbon=label, air_quality=AQ, comparison=comparison, drying_central=dce,
               tier_benchmark_path={y: tier_benchmark(y) for y in range(2026, 2052)}, tier_benchmark_sensitivity=tier_sens, params=PARAM)
    (OUT / "profit_community_results.json").write_text(json.dumps(res, indent=1, default=float))
    nodes_df.to_csv(OUT / "community_heat_by_node.csv", index=False, float_format="%.6f")
    # printing
    print("DG 25y:", {k: round(v["lcoe_25y_cad_mwh"], 1) for k, v in dg.items()}, "yr21:", {k: round(v["yr21_cad_mwh"], 1) for k, v in dg.items()})
    print("gas 25y:", {k: (round(v["lcoe_25y_cad_mwh"], 1), round(v["yr25_cad_mwh"], 1)) for k, v in gas.items()})
    print("breakeven growth:", res["breakeven_gas_growth"], res["breakeven_gas_growth_full_carbon"])
    print("flat breakeven:", res["breakeven_gas_flat"])
    print("emissions per GW IT:", {k: round(v["total_mt"], 3) for k, v in em.items()})
    print("thermal:", {k: round(v, 3) for k, v in thermal.items()})
    print("community:", round(community["total_reduction_mt"], 3), round(community["total_reduction_best_mt"], 3), {k: round(v / 1e6, 3) for k, v in community["ghg_reduction_t"].items()})
    print("baseline covered", round(community["baseline_covered_mt"], 3), "asparagopsis", round(community["asparagopsis_dw_t"]), "t", round(community["asparagopsis_area_ha"], 1), "ha")
    print("econ:", {k: round(v, 3) for k, v in community["economics"].items()})
    print("home heat GJ/hh:", nodes_df[["node", "home_heat_gj_per_household"]].round(1).to_dict("records")[:4])
    print("label gap:", {k: (round(v["gap_cad_mwh"], 1), {p: round(x, 3) for p, x in v["premium_usd_per_gpu_h"].items()}) for k, v in gaps.items()})
    print(G.sort_values("npv_bn", ascending=False).head(8)[["electricity", "product", "tier", "label", "npv_bn", "irr", "mean_ebitda_bn"]].round(3).to_string(index=False))
    print(BE[(BE.tier == "market on-demand") & (BE.market == "prices held each generation")].round(3).to_string())
    print(SENS[SENS.market == "prices held each generation"].round(2).to_string())
    print("label carbon", {k: round(v, 3) for k, v in label.items()}, "scenarios", community["deployment_scenarios_mt"])
    piv = G[((G.label == 0) | (G.label == 0.5))].pivot_table(index=["market", "electricity", "label"], columns=["product", "tier"], values="npv_bn").round(1)
    print(piv.to_string())


if __name__ == "__main__":
    main()
