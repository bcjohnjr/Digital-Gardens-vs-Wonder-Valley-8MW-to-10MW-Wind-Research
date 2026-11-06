#!/usr/bin/env python3
"""Commercial and industrial drying with surplus server heat, sold below the cost of gas-fired drying.

Hour by hour over the 2010-2025 ERA5 years, recoverable IT heat at each of the 17 nodes is first given to homes (space heat
and hot water), digesters and aquaculture/greenhouses, as in profit_community_model.py. Whatever is left in that hour goes
to drying hubs built at the nodes: perishable farm products first (in their harvest windows), then year-round industrial
contracts whose wet feedstock can be stockpiled (lumber, sawdust, wood residue, biosolids, feed by-products, laundry).
Each contract has an annual tonnage, a dryer capacity and a share of its heat that must be lifted above the 45-55 C loop
by a booster heat pump. Booster and fan electricity comes first from renewable output that would otherwise be curtailed
in that hour (from the sweet-spot dispatch), and is otherwise bought at the 25-year system cost.

Every market size below is a screening assumption for the catchments of the 17 nodes; sources are noted beside each.
"""
from pathlib import Path
import json
import numpy as np, pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
OUT = HERE / "OUTPUT"
import profit_community_model as pcm                                    # noqa: E402
import storage_fuel_sweetspot as sfs                                    # noqa: E402
from dispatch_reference import HVO_L_PER_MWH, LOSS_WIND                     # noqa: E402  (path added by storage_fuel_sweetspot)

PARAM = pcm.PARAM
SS = json.load(open(OUT / "storage_fuel_sweetspot.json"))

# ------------------------------------------------------------------ prices
PRICE = dict(
    gas_commodity_cad_gj=4.0, gas_growth=0.03,          # central path used for electricity (C$4/GJ, +3%/yr real)
    gas_delivery_cad_gj=4.0,                            # FortisBC large-commercial delivery and storage charges (approx.)
    gas_dryer_efficiency=0.80,                          # direct-fired or boiler-fed convective dryer
    discount_to_gas=0.25,                               # heat sold at 25% below the gas-fired cost of the same useful heat
    carbon_cad_t=0.0,                                   # B.C. consumer carbon tax ended April 2025; most dryers are below OBPS thresholds
    electricity_cad_mwh=SS["sweet_spot"]["lcoe_25y_cad_mwh"],   # non-curtailed booster/fan power charged at the system's 25-year cost
    fan_kwh_e_per_kwh_th=0.04,
    booster_cop=3.0,
    schedule_boost_to_curtailment=True,                 # stockpiled loads run their high-temperature stage only when renewables are curtailed
    curtail_threshold_mw=500.0,                                    # 50 C loop -> 80-85 C kiln air (35 K lift)
    capex_gas_dryer_cad_kw_th=300.0,                   # equivalent gas-fired dryer the customer would otherwise own
    capex_dryer_cad_kw_th=450.0,                        # belt / batch dryers and dehumidification kilns at the hubs
    capex_booster_cad_kw_th=700.0,                      # high-temperature heat pump per kW of boosted heat
    capex_pipe_cad_kw_th=120.0,                         # loop extension from the data hall to the hub
    om_frac=0.03, life=20, rate=0.06, handling_cad_t_water=8.0,  # labour, loaders, stockpile management per tonne of water
    gas_t_co2_per_gj=0.0499, gas_upstream_share=0.10, horizon=25,
    truck_cad_per_t_km=0.15,                            # loaded tonne-km, empty return included; wet in plus dry out
    pipe_cad_per_km=2.5e6,                              # buried district-heat main to an off-site mill or plant
)

# ------------------------------------------------------------------ markets
# water_t: tonnes of water removed per year; kwh_per_kg: dryer heat per kg water; boost: share of heat lifted by heat pump;
# months: operating window (None = year-round, feedstock stockpiled); region: descriptive only (hub nodes are set in SITING);
# gas_share: share of the displaced drying that would otherwise burn gas or propane (rest is biomass, sun or no drying)
MARKETS = {
    # --- farm drying in the warm months (the user's question)
    "Hay and alfalfa (barn-dried bales and cubes for export)": dict(
        water_t=300_000 * 0.229 * 2, kwh_per_kg=1.30, boost=0.0, months=range(5, 10), region="INT", gas_share=0.30, kind="farm",
        note="600 kt hay dried from 30% to 14% moisture in May-Sep; Thompson-Nicola, Okanagan and Shuswap hay land (assumption)"),
    "Berries, cherries and apples (dried, infused, powders)": dict(
        water_t=(10_000 + 9_000 + 3_000 + 25_000) * 0.80, kwh_per_kg=1.40, boost=0.0, months=range(7, 11), region="FV", gas_share=0.80, kind="farm",
        note="15% of B.C. highbush blueberries (67 kt), 20% of cranberries (46 kt), cherry culls, 25 kt apples and culls (AAFC 2024)"),
    "Fruit pomace and grape marc (feed and flour)": dict(
        water_t=40_000 * 0.55, kwh_per_kg=1.40, boost=0.0, months=range(8, 12), region="INT", gas_share=0.50, kind="farm",
        note="apple juice pomace and Okanagan winery marc, 40 kt at 65% to 10% moisture (assumption)"),
    "Grain, seed and corn (Fraser Valley and Shuswap)": dict(
        water_t=60_000 * 0.12, kwh_per_kg=1.30, boost=0.0, months=range(8, 12), region="FV", gas_share=0.90, kind="farm",
        note="60 kt grain corn, cereals and seed dried 12 points (assumption; gas or propane bin dryers)"),
    "Vegetable and herb dehydration": dict(
        water_t=20_000 * 0.85, kwh_per_kg=1.40, boost=0.0, months=range(6, 11), region="FV", gas_share=0.80, kind="farm",
        note="20 kt onions, garlic, herbs, greenhouse seconds (assumption)"),
    # --- year-round industrial contracts (wet feedstock stockpiled, so they absorb the summer surplus)
    "Lumber kiln drying (dehumidification and boosted kilns)": dict(
        water_t=4_000_000 * 1.0 / 3.6 * 1e3 / 1.6 / 1e3,      # GJ -> MWh -> t water (1 MWh / 1.6 kWh/kg = 625 kg)
        kwh_per_kg=1.60, boost=0.50, months=None, region="INT", gas_share=0.35, kind="industry",
        note="4.0 million m3/yr of SPF lumber from mills within reach of Kamloops, Vernon, Salmon Arm, Okanagan and Lillooet; 1.0 GJ/m3 "
             "kiln heat (FPInnovations / Wood Business); half lifted to 80 C"),
    "Wood pellets (sawdust and shavings belt dryers)": dict(
        water_t=600_000 * 0.90, kwh_per_kg=1.40, boost=0.20, months=None, region="INT", gas_share=0.10, kind="industry",
        note="600 kt/yr pellets (North Okanagan / Thompson plants), 0.9 t water per t pellets"),
    "Biomass residue drying for biochar, boiler fuel and district energy": dict(
        water_t=1_000_000 * 0.286, kwh_per_kg=1.35, boost=0.0, months=None, region="INT", gas_share=0.0, kind="industry",
        note="1 Mt/yr green hog fuel and roadside residue dried from 50% to 30% (new product; no fossil credit taken)"),
    "Municipal biosolids (belt-dried to 90% solids)": dict(
        water_t=788_000 * 25 / 1000 * (1 / 0.22 - 1 / 0.90), kwh_per_kg=1.30, boost=0.0, months=None, region="FV", gas_share=0.50, kind="industry",
        note="25 kg dry solids per resident per year, 22% cake to 90% (Metro/FVRD and Interior plants)"),
    "Fish-processing by-products and insect meal": dict(
        water_t=48_000 * 0.62 + 20_000 * 0.65, kwh_per_kg=1.40, boost=0.30, months=None, region="FV", gas_share=0.80, kind="industry",
        note="by-products of the Digital Gardens aquaculture (48 kt) and black-soldier-fly larvae (20 kt)"),
    "Brewers' and distillers' spent grain (dried feed)": dict(
        water_t=40_000 * 0.72, kwh_per_kg=1.40, boost=0.0, months=None, region="FV", gas_share=0.20, kind="industry",
        note="40 kt wet spent grain from Fraser Valley and Okanagan breweries and distilleries (assumption)"),
    "Digestate solids and poultry litter (fertiliser pellets)": dict(
        water_t=107_000 + 70_000, kwh_per_kg=1.30, boost=0.0, months=None, region="FV", gas_share=0.0, kind="industry",
        note="the drying barter of the community programme: digester solids and Fraser Valley poultry litter (no fossil credit taken)"),
    "Malting and craft-grain kilning": dict(
        water_t=20_000 * 0.75, kwh_per_kg=1.50, boost=0.30, months=None, region="FV", gas_share=0.95, kind="industry",
        note="20 kt/yr malt for B.C. brewers and distillers (now largely imported from the Prairies); kilning 50-85 C"),
    "Hemp, cannabis and medicinal herbs (low-temperature curing)": dict(
        water_t=10_000 * 0.75, kwh_per_kg=1.50, boost=0.0, months=None, region="FV", gas_share=0.90, kind="industry",
        note="10 kt/yr wet greenhouse hemp and cannabis biomass, 20-50 C curing"),
    "Commercial laundry drying (hospitals, hotels, care homes)": dict(
        water_t=40_000 * 0.55, kwh_per_kg=1.50, boost=0.60, months=None, region="FV", gas_share=1.00, kind="industry",
        note="40 kt/yr linen for Fraser Health and Interior Health facilities and hotels; gas tumble dryers"),
}

# ------------------------------------------------------------------ siting and delivery
# nodes: hubs at these data-centre nodes share the contract equally ("pop" = every node, split by population).
# delivery: ("pipe", km of heat main per hub) when the customer stays on its own site (mills, pellet plants), or
# ("truck", average one-way km) when wet product is hauled to a hub at the node; wet: tonnes of wet input per tonne of water.
SITING = {
    "Hay and alfalfa (barn-dried bales and cubes for export)": (["kamloops", "salmon_arm", "vernon", "chase", "cache_creek", "lillooet"], ("truck", 30), 5.4),
    "Berries, cherries and apples (dried, infused, powders)": (["abbotsford", "chilliwack", "mission", "maple_ridge", "kelowna", "oliver"], ("truck", 25), 1.25),
    "Fruit pomace and grape marc (feed and flour)": (["kelowna", "west_kelowna", "penticton", "oliver", "osoyoos"], ("truck", 20), 1.8),
    "Grain, seed and corn (Fraser Valley and Shuswap)": (["chilliwack", "agassiz_kent", "abbotsford", "salmon_arm"], ("truck", 25), 8.3),
    "Vegetable and herb dehydration": (["abbotsford", "chilliwack", "agassiz_kent"], ("truck", 20), 1.2),
    "Lumber kiln drying (dehumidification and boosted kilns)": (["kamloops", "chase", "salmon_arm", "vernon", "lillooet", "west_kelowna"], ("pipe", 5.0), 4.3),
    "Wood pellets (sawdust and shavings belt dryers)": (["vernon", "salmon_arm", "kamloops"], ("pipe", 4.0), 2.0),
    "Biomass residue drying for biochar, boiler fuel and district energy": (["kamloops", "lillooet", "cache_creek", "chase", "hope"], ("truck", 20), 3.5),
    "Municipal biosolids (belt-dried to 90% solids)": ("pop", ("truck", 15), 1.32),
    "Fish-processing by-products and insect meal": (["chilliwack", "abbotsford", "agassiz_kent"], ("truck", 10), 1.6),
    "Brewers' and distillers' spent grain (dried feed)": (["abbotsford", "kelowna", "kamloops", "penticton"], ("truck", 25), 1.39),
    "Digestate solids and poultry litter (fertiliser pellets)": (["abbotsford", "chilliwack", "agassiz_kent", "mission"], ("truck", 15), 1.5),
    "Malting and craft-grain kilning": (["abbotsford", "kelowna"], ("truck", 30), 2.3),
    "Hemp, cannabis and medicinal herbs (low-temperature curing)": (["hope", "mission", "maple_ridge", "oliver"], ("truck", 20), 1.33),
    "Commercial laundry drying (hospitals, hotels, care homes)": ("pop", ("truck", 10), 4.6),
}
for _k, (_n, _d, _w) in SITING.items():
    MARKETS[_k].update(nodes=_n, delivery=_d, wet_per_water=_w)

# ------------------------------------------------------------------ heat rejection
# Surplus loop heat that no customer takes in a given hour is rejected through a closed-loop heat exchanger to a large
# river or lake (no evaporation), or through dry coolers where the receiving water is small. No cooling water is consumed.
WATER_BODY = {
    "lillooet": ("Fraser River", "river exchanger"), "cache_creek": ("Bonaparte River (small)", "dry coolers"),
    "kamloops": ("Thompson River", "river exchanger"), "chase": ("South Thompson River / Little Shuswap Lake", "river exchanger"),
    "salmon_arm": ("Shuswap Lake", "lake exchanger"), "vernon": ("Okanagan / Kalamalka lakes", "lake exchanger"),
    "kelowna": ("Okanagan Lake", "lake exchanger"), "west_kelowna": ("Okanagan Lake", "lake exchanger"),
    "penticton": ("Okanagan / Skaha lakes", "lake exchanger"), "oliver": ("Okanagan River (regulated, small)", "dry coolers"),
    "osoyoos": ("Osoyoos Lake (small)", "dry coolers"), "hope": ("Fraser River", "river exchanger"),
    "agassiz_kent": ("Fraser River", "river exchanger"), "chilliwack": ("Fraser River", "river exchanger"),
    "abbotsford": ("Fraser River", "river exchanger"), "mission": ("Fraser River", "river exchanger"),
    "maple_ridge": ("Fraser River", "river exchanger"),
}


@njit(cache=True)
def dispatch_curtail(generation, load, capacity_mwh, hvo_limit_mwh, backup_mw, b100):
    """Same order as the published kernel, returning the hourly curtailed energy (MWh)."""
    soc = capacity_mwh * .5; pmax = min(capacity_mwh / 8., 12000.); hvo = hvo_limit_mwh
    cur = np.zeros(len(generation))
    for k in range(len(generation)):
        net = generation[k] - load[k]
        if net >= 0:
            c = min(net, pmax, max((capacity_mwh - soc) / .94, 0.)); soc += c * .94; cur[k] = net - c
        else:
            d = -net; dis = min(d, pmax, soc * .94); soc -= dis / .94; d -= dis
            if d > 0:
                cap = min(d, backup_mw); b = min(cap, b100); cap -= b; b100 -= b; h = min(cap, hvo); hvo -= h
    return cur


def curtailment_series():
    groups, _ = sfs.load_years()
    sel = json.load(open(OUT / "optimization_provenance.json"))
    sw = SS["sweet_spot"]; w = np.asarray(sw["wind_GW_per_site"],dtype=float)
    from dispatch_reference import B100_MWh
    out = []
    for g in groups:
        gen = (w @ g["windcf"]) * 1000 * (1 - LOSS_WIND) + sw["solar_GW"] * g["solar"] + g["hydro"]
        out.append(dispatch_curtail(gen, g["load"], sw["battery_GWh"] * 1000, sw["fuel_cap_ML"] * 1e6 / HVO_L_PER_MWH, 5100., B100_MWh))
    return np.concatenate(out)                 # MWh per hour, 16 x 8760


def base_heat(T):
    """Hourly surplus heat (MW_th) by region after homes, digesters and aquaculture, plus the base-case served totals."""
    H = PARAM["heat"]; LV = PARAM["livestock"]
    nodes = list(PARAM["population"]); n = len(nodes); fv = PARAM["fraser_valley_nodes"]
    avail_mw = pcm.IT_GW * 1000 / n * H["recoverable_heat_share_it"] * (1 - H["network_loss"])
    t0, _ = T[nodes[0]]; months = t0.dt.month.to_numpy(); hours = t0.dt.hour.to_numpy()
    season = np.array([1.6, 1.55, 1.35, 1.12, .85, .55, .43, .48, .70, 1.03, 1.35, 1.6])[months - 1]
    season = season * (1 + .08 * np.cos((hours - 5) * 2 * np.pi / 24)); season = season / season.mean()
    ch4 = LV["fv_dairy_cows"] * LV["biogas_m3_ch4_yr"]["dairy"] + LV["fv_pigs"] * LV["biogas_m3_ch4_yr"]["pig"] + LV["plant_waste_t"] * LV["plant_waste_m3_ch4_t"]
    dig_mwh = ch4 * LV["ch4_mj_m3"] / 1000 * LV["digester_heat_share"] / 3.6
    surplus = {}
    served = dict(homes=0.0, digester=0.0, aqua=0.0)
    for k in nodes:
        _, temp = T[k]; conn = PARAM["population"][k] / H["persons_per_household"] * H["connected_share"]
        space = conn * H["ua_kw_per_k"] * np.maximum(0, H["base_c"] - temp) / 1000
        dhw = np.full(len(temp), conn * H["persons_per_household"] * H["dhw_l_person_day"] * 365 * 4.186 * (H["dhw_c"] - H["mains_c"]) / 3.6e6 / 8760)
        rem = np.full(len(temp), avail_mw)
        s1 = np.minimum(rem, space + dhw); rem -= s1
        s2 = np.minimum(rem, dig_mwh / len(fv) / 8760 if k in fv else 0.0); rem -= s2
        s3 = np.minimum(rem, 4.819e6 / n / 8760 * season); rem -= s3
        served["homes"] += s1.sum(); served["digester"] += s2.sum(); served["aqua"] += s3.sum()
        surplus[k] = rem
    return surplus, served, months, avail_mw * n


def run_drying(surplus, months, curtail, markets, scale=1.0):
    """Allocate each node's surplus heat hour by hour: farm products in season first, then stockpiled industrial feedstock.
    Each contract is split into hubs at its assigned nodes; a hub can only use heat from its own node."""
    nyears = len(months) / 8760
    rows = []; boost_el = np.zeros(len(months))
    rem = {r: s_.copy() for r, s_ in surplus.items()}
    pop = PARAM["population"]
    curt_mask = curtail > PRICE["curtail_threshold_mw"]
    block_y = np.repeat(np.arange(len(months) // 8760), 8760)
    block_m = block_y * 12 + (months - 1)
    order = sorted(markets.items(), key=lambda kv: 0 if kv[1]["kind"] == "farm" else 1)
    for name, m in order:
        nodes = list(pop) if m["nodes"] == "pop" else m["nodes"]
        wts = np.array([pop[n] for n in nodes], float) if m["nodes"] == "pop" else np.ones(len(nodes))
        wts = wts / wts.sum()
        heat_yr = m["water_t"] * scale * m["kwh_per_kg"]                       # MWh_th per year
        if m["months"] == "curtailed":
            mask = curt_mask
        else:
            mask = np.ones(len(months), bool) if m["months"] is None else np.isin(months, list(m["months"]))
        cap_mw = heat_yr / (mask.sum() / nyears * 0.85)                        # hubs sized to finish the season / year at 85% load
        loop_share = 1 - m["boost"] / PRICE["booster_cop"]                     # heat pump: loop supplies output minus compressor work
        block, nb = (block_m, 12) if m["months"] is None else (block_y, 1)
        take_all = np.zeros(len(months))
        for n, wgt in zip(nodes, wts):
            want = np.where(mask, cap_mw * wgt * loop_share, 0.0)
            take = np.minimum(rem[n], want)
            lim = heat_yr * wgt * loop_share / nb
            out = np.empty_like(take)
            for b in np.unique(block):
                idx = block == b; cs = np.minimum(np.cumsum(take[idx]), lim)
                out[idx] = np.diff(np.concatenate([[0.0], cs]))
            rem[n] -= out; take_all += out
        heat_delivered = take_all / loop_share
        el_m = heat_delivered * m["boost"] / PRICE["booster_cop"]
        boost_el += el_m
        water = heat_delivered / m["kwh_per_kg"]
        rows.append(dict(market=name, kind=m["kind"], nodes=nodes, n_hubs=len(nodes), delivery=m["delivery"], wet_per_water=m["wet_per_water"],
                         capacity_mw_th=cap_mw, target_twh=heat_yr / 1e6,
                         loop_heat_twh=take_all.sum() / 1e6 / nyears, delivered_twh=heat_delivered.sum() / 1e6 / nyears,
                         share_of_target=heat_delivered.sum() / nyears / heat_yr if heat_yr > 0 else 0.0, water_kt=water.sum() / 1e3 / nyears,
                         boost_share=m["boost"], gas_share=m["gas_share"], note=m["note"], hourly=take_all, el_hourly=el_m))
    el = boost_el
    from_curtail = np.minimum(el, curtail)
    frac_bought = np.where(el > 0, (el - from_curtail) / np.maximum(el, 1e-12), 0.0)
    for x in rows:
        x["bought_el_twh"] = (x.pop("el_hourly") * frac_bought).sum() / 1e6 / nyears
    return rows, rem, dict(el_twh=el.sum() / 1e6 / nyears, curtail_used_twh=from_curtail.sum() / 1e6 / nyears,
                           bought_twh=(el - from_curtail).sum() / 1e6 / nyears)


def economics(rows, el):
    """Per market: what drying would cost the customer with gas, the price Digital Gardens charges (a fixed discount to that),
    and Digital Gardens' own cost (hot-water dryers, loop extension, boosters, booster power not covered by curtailment).
    Fans, labour and handling are the same with either heat source and are left out of both sides."""
    P = PRICE; r, life = P["rate"], P["life"]
    crf = r * (1 + r) ** life / ((1 + r) ** life - 1)
    fix = crf + P["om_frac"]
    gas_heat = [(P["gas_commodity_cad_gj"] * (1 + P["gas_growth"]) ** i + P["gas_delivery_cad_gj"] + P["carbon_cad_t"] * P["gas_t_co2_per_gj"])
                / P["gas_dryer_efficiency"] * 3.6 for i in range(P["horizon"])]                 # C$/MWh of useful heat
    per = []; tot_rev = np.zeros(P["horizon"]); tot_cost = np.zeros(P["horizon"]); tot_gas = np.zeros(P["horizon"])
    for x in rows:
        mwh = x["delivered_twh"] * 1e6; kw = x["capacity_mw_th"] * 1e3
        if mwh <= 0:
            continue
        dg_fixed = kw * (P["capex_dryer_cad_kw_th"] + P["capex_pipe_cad_kw_th"] + x["boost_share"] * P["capex_booster_cad_kw_th"]) * fix
        dg_el = x["bought_el_twh"] * 1e6 * P["electricity_cad_mwh"]
        mode, km = x["delivery"]
        if mode == "truck":
            wet = x["water_kt"] * 1e3 * x["wet_per_water"]
            delivery = (2 * wet - x["water_kt"] * 1e3) * km * P["truck_cad_per_t_km"]
        else:
            delivery = x["n_hubs"] * km * P["pipe_cad_per_km"] * fix
        dg_fixed += delivery
        gas_fixed = kw * P["capex_gas_dryer_cad_kw_th"] * fix
        gas = np.array([mwh * g + gas_fixed for g in gas_heat])
        rev = gas * (1 - P["discount_to_gas"])
        cost = np.full(P["horizon"], dg_fixed + dg_el)
        tot_rev += rev; tot_cost += cost; tot_gas += gas
        per.append(dict(market=x["market"], heat_twh=x["delivered_twh"], water_kt=x["water_kt"], full_load_hours=mwh / kw,
                        gas_cost_cad_t_water_yr1=gas[0] / (x["water_kt"] * 1e3), price_cad_t_water_yr1=rev[0] / (x["water_kt"] * 1e3),
                        dg_cost_cad_t_water=cost[0] / (x["water_kt"] * 1e3), gas_cost_cad_mwh_yr1=gas[0] / mwh, dg_cost_cad_mwh=cost[0] / mwh,
                        margin_m_cad_yr1=(rev[0] - cost[0]) / 1e6, margin_m_cad_mean=(rev - cost).mean() / 1e6,
                        customer_saving_m_cad_yr1=(gas[0] - rev[0]) / 1e6, bought_el_gwh=x["bought_el_twh"] * 1e3,
                        delivery=f"{mode} {km:g} km", delivery_m_cad=delivery / 1e6, dg_margin_cad_t_water=(rev[0] - cost[0]) / (x["water_kt"] * 1e3)))
    margin = tot_rev - tot_cost
    npv = sum(m / (1 + r) ** (i + 1) for i, m in enumerate(margin)) / 1e9
    heat = sum(x["delivered_twh"] for x in rows) * 1e6
    cap_kw = sum(x["capacity_mw_th"] for x in rows) * 1e3
    capex = sum(x["capacity_mw_th"] * 1e3 * (P["capex_dryer_cad_kw_th"] + P["capex_pipe_cad_kw_th"] + x["boost_share"] * P["capex_booster_cad_kw_th"]) for x in rows)
    gj_gas = sum(x["delivered_twh"] * 1e6 * 3.6 / P["gas_dryer_efficiency"] * x["gas_share"] for x in rows)
    return dict(capex_bn=capex / 1e9, capacity_mw_th=cap_kw / 1e3, heat_twh=heat / 1e6, water_mt=sum(x["water_kt"] for x in rows) / 1e3,
                gas_heat_cad_mwh_yr1=gas_heat[0], gas_heat_cad_mwh_yr25=gas_heat[-1],
                revenue_m_cad_yr1=tot_rev[0] / 1e6, cost_m_cad_yr1=tot_cost[0] / 1e6, margin_m_cad_yr1=margin[0] / 1e6, margin_m_cad_mean=margin.mean() / 1e6,
                customer_saving_m_cad_yr1=(tot_gas[0] - tot_rev[0]) / 1e6, customer_saving_m_cad_mean=(tot_gas - tot_rev).mean() / 1e6, npv_25y_bn=npv,
                gas_gj_displaced=gj_gas, ghg_reduction_t=gj_gas * P["gas_t_co2_per_gj"] * (1 + P["gas_upstream_share"]), per_market=per)


def scenario(name, surplus, months, curtail, served, delivered_total, scale_farm=1.0, scale_ind=1.0, extra=None,
             predry_only=False, exclude=()):
    """predry_only: customers needing 70-85 C keep finishing in their own kilns/dryers and buy only the low-temperature stage.
    exclude: contracts left out (e.g. those whose haul cost exceeds the margin)."""
    mk = {}
    for k, v0 in MARKETS.items():
        if k in exclude:
            continue
        v = dict(v0); v["water_t"] *= scale_farm if v["kind"] == "farm" else scale_ind
        if v["boost"] > 0 and v["months"] is None and predry_only:
            mk[k + " | low-temperature stage"] = dict(v, boost=0.0, water_t=v["water_t"] * (1 - v["boost"]))
        elif PRICE["schedule_boost_to_curtailment"] and v["boost"] > 0 and v["months"] is None:
            lo = dict(v, boost=0.0, water_t=v["water_t"] * (1 - v["boost"]))
            hi = dict(v, boost=1.0, water_t=v["water_t"] * v["boost"], months="curtailed",
                      delivery=("pipe", 0.0) if v["delivery"][0] == "pipe" else v["delivery"])     # the heat main is shared with the low-temperature stage
            mk[k + " | low-temperature stage"] = lo; mk[k + " | boosted stage in curtailed hours"] = hi
        else:
            mk[k] = v
    if extra:
        mk.update(extra)
    rows, rem, el = run_drying(surplus, months, curtail, mk)
    ec = economics(rows, el)
    nyears = len(months) / 8760
    base_useful = (served["homes"] + served["digester"] + served["aqua"]) / 1e6 / nyears
    useful = base_useful + ec["heat_twh"] - sum(x["delivered_twh"] - x["loop_heat_twh"] for x in rows)   # count loop heat only
    recoverable = pcm.IT_TWH * PARAM["heat"]["recoverable_heat_share_it"]
    z = np.zeros(len(months))
    monthly = pd.DataFrame({"month": months, "dry": sum(x["hourly"] for x in rows),
                            "farm": sum((x["hourly"] for x in rows if x["kind"] == "farm"), z), "ind": sum((x["hourly"] for x in rows if x["kind"] != "farm"), z),
                            "rem": sum(rem.values())}).groupby("month").sum() / 1e6 / nyears
    return dict(name=name, markets=[{k: v for k, v in x.items() if k != "hourly"} for x in rows], electricity=el, economics=ec,
                useful_loop_heat_twh=useful, utilisation_pct=100 * useful / recoverable, erf_pct=100 * useful / pcm.LOAD_TWH,
                rejected_twh=sum(r.sum() for r in rem.values()) / 1e6 / nyears, monthly_drying_twh=monthly.dry.tolist(), monthly_farm_twh=monthly.farm.tolist(),
                monthly_industry_twh=monthly.ind.tolist(), monthly_rejected_twh=monthly.rem.tolist(),
                node_rejection={n: dict(mean_mw=float(r.mean()), peak_mw=float(r.max()), p99_mw=float(np.percentile(r, 99)), twh=float(r.sum() / 1e6 / nyears),
                                        flow_m3s_for_1C=float(r.max() / (4.186 * 1.0)), flow_m3s_for_0p5C=float(r.max() / (4.186 * 0.5)),
                                        water_body=WATER_BODY[n][0], rejection=WATER_BODY[n][1]) for n, r in rem.items()})


def main():
    T = pcm.load_temps()
    surplus, served, months, _ = base_heat(T)
    curtail = curtailment_series()
    assert len(curtail) == len(months), (len(curtail), len(months))
    nyears = len(months) / 8760
    base_useful = (served["homes"] + served["digester"] + served["aqua"]) / 1e6 / nyears
    hrs = pd.Series(months).value_counts().sort_index().to_numpy() / nyears
    res = {"available_twh_by_month": (hrs * pcm.IT_GW * 1000 * PARAM["heat"]["recoverable_heat_share_it"] * (1 - PARAM["heat"]["network_loss"]) / 1e6).tolist(),
           "base_without_drying": dict(useful_twh=base_useful, utilisation_pct=100 * base_useful / (pcm.IT_TWH * 0.85),
                                       erf_pct=100 * base_useful / pcm.LOAD_TWH, surplus_twh=sum(s.sum() for s in surplus.values()) / 1e6 / nyears,
                                       monthly_surplus_twh=(pd.DataFrame({"m": months, "s": sum(surplus.values())}).groupby("m").s.sum() / 1e6 / nyears).tolist()),
           "curtailment_twh": curtail.sum() / 1e6 / nyears}
    PRICE["schedule_boost_to_curtailment"] = False
    res["central_boost_unscheduled"] = scenario("Central, boosters run whenever needed", surplus, months, curtail, served, 0, 1.0, 1.0)
    res["central_boost_no_curtailment"] = scenario("Central, boosters buy all power at system cost (no access to curtailment)", surplus, months,
                                                   np.zeros_like(curtail), served, 0, 1.0, 1.0)
    PRICE["schedule_boost_to_curtailment"] = True
    res["water_mt_needed_for_60pct"] = (0.60 * pcm.IT_TWH * 0.85 - base_useful) / 1.40
    res["water_mt_needed_for_80pct"] = (0.80 * pcm.IT_TWH * 0.85 - base_useful) / 1.40
    res["curtailed_hours_per_year"] = float((curtail > PRICE["curtail_threshold_mw"]).sum() / nyears)
    res["profit_max"] = scenario("Profit-maximising contract set (pre-drying for high-temperature customers; hay and grain left on-farm)",
                                 surplus, months, curtail, served, 0, 1.0, 1.0, predry_only=True,
                                 exclude=("Hay and alfalfa (barn-dried bales and cubes for export)", "Grain, seed and corn (Fraser Valley and Shuswap)"))
    res["farm_only"] = scenario("Farm drying in the warm months only", surplus, months, curtail, served, 0, 1.0, 0.0)
    res["central"] = scenario("Farm and industrial drying (central)", surplus, months, curtail, served, 0, 1.0, 1.0)
    res["high"] = scenario("Regional drying hub (2x industrial contracts, 1.5x farm)", surplus, months, curtail, served, 0, 1.5, 2.0)
    # how far drying contracts can push heat use: scale every contract together
    sweep = []
    for sc in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
        x = scenario("sweep", surplus, months, curtail, served, 0, sc, sc)
        sweep.append(dict(scale=sc, water_mt=x["economics"].get("water_mt", 0.0), utilisation_pct=x["utilisation_pct"], erf_pct=x["erf_pct"],
                          npv_bn=x["economics"].get("npv_25y_bn", 0.0), ghg_reduction_t=x["economics"].get("ghg_reduction_t", 0.0), served_share=sum(m["delivered_twh"] for m in x["markets"]) / max(sum(m["target_twh"] for m in x["markets"]), 1e-9)))
    res["scale_sweep"] = sweep
    print(pd.DataFrame(sweep).round(3).to_string())
    # price sensitivity of the central case
    sens = {}
    for d in (0.10, 0.25, 0.40, 0.50):
        PRICE["discount_to_gas"] = d
        sens[f"{int(d * 100)}% below gas"] = scenario("x", surplus, months, curtail, served, 0)["economics"]["npv_25y_bn"]
    PRICE["discount_to_gas"] = 0.25
    res["npv_by_discount_bn"] = sens
    res["price_params"] = PRICE; res["markets_params"] = {k: {kk: (list(vv) if isinstance(vv, range) else vv) for kk, vv in v.items()} for k, v in MARKETS.items()}
    (OUT / "industrial_drying_results.json").write_text(json.dumps(res, indent=1, default=float))
    pd.DataFrame(res["central"]["markets"]).to_csv(OUT / "industrial_drying_markets_central.csv", index=False, float_format="%.5f")
    b = res["base_without_drying"]
    print("curtailed hours/yr", res["curtailed_hours_per_year"]); print(f"base: useful {b['useful_twh']:.2f} TWh, util {b['utilisation_pct']:.1f}%, ERF {b['erf_pct']:.1f}%, surplus {b['surplus_twh']:.2f}, curtail {res['curtailment_twh']:.2f}")
    print("monthly surplus", np.round(b["monthly_surplus_twh"], 2))
    for k in ("central_boost_unscheduled", "farm_only", "central", "profit_max", "high"):
        s = res[k]; e = s["economics"]
        print(f"\n{s['name']}: util {s['utilisation_pct']:.1f}% ERF {s['erf_pct']:.1f}% useful {s['useful_loop_heat_twh']:.2f} rejected {s['rejected_twh']:.2f}")
        print({kk: round(v, 3) for kk, v in e.items() if kk != "per_market"}, {kk: round(v, 3) for kk, v in s["electricity"].items()})
        if k == "central":
            print(pd.DataFrame(e["per_market"]).drop(columns=["market"]).assign(m=[x["market"][:30] for x in e["per_market"]]).round(2).to_string())
        for m in s["markets"]:
            print(f"  {m['market'][:55]:55s} {m['n_hubs']:2d} hubs target {m['target_twh']:.3f} deliv {m['delivered_twh']:.3f} ({100 * m['share_of_target']:.0f}%) water {m['water_kt']:.0f} kt cap {m['capacity_mw_th']:.0f} MW")
    print(sens)


if __name__ == "__main__":
    main()
