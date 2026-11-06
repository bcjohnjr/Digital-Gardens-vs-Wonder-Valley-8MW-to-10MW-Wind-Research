#!/usr/bin/env python3
"""Figures for the storage-fuel sweet spot, 25-year electricity cost, emissions per GW of IT, community GHG, heat and GPU profit."""
from pathlib import Path
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "OUTPUT"; FIG = HERE / "FIGURES_EXTENSION"; FIG.mkdir(exist_ok=True)
SS = json.load(open(OUT / "storage_fuel_sweetspot.json"))
R = json.load(open(OUT / "profit_community_results.json"))
D = json.load(open(OUT / "industrial_drying_results.json")); DC = D["central"]
GHG = R["community"]["ghg_reduction_t"]
COMM_MT = R["community"]["total_reduction_mt"]
NAVY, GREEN, BLUE, GREY, ORANGE, RED = "#00264E", "#4C9A2A", "#2F6DB5", "#8A8F98", "#D9822B", "#B23A3A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "savefig.dpi": 300})


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png"); fig.savefig(FIG / f"{name}.pdf"); plt.close(fig)


# 1. storage-fuel trade-off
tc = pd.DataFrame(SS["tradeoff_curve"])
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax[0].plot(tc.fuel_cap_ML, tc.lcoe_25y_cad_mwh, "o-", color=NAVY)
ref, sw = SS["reference_selected"], SS["sweet_spot"]
ax[0].scatter([ref["fuel_cap_ML"]], [ref["lcoe_25y_cad_mwh"]], s=70, color=GREY, zorder=3, label="Reoptimised 300-ML case")
ax[0].scatter([sw["fuel_cap_ML"]], [sw["lcoe_25y_cad_mwh"]], s=90, color=GREEN, zorder=3, label="GWA mean-corrected (gas-free in 16 ERA5 years)")
ax[0].axvline(1000, color=ORANGE, ls="--", lw=1); ax[0].text(1030, ax[0].get_ylim()[1] * 0.98, "1 billion L/yr ceiling\n(data centres get\n976 ML after farm HVO)", va="top", fontsize=7, color=ORANGE)
ax[0].set_xlabel("Annual HVO allowance (ML)"); ax[0].set_ylabel("25-year cost (C$/MWh)"); ax[0].set_title("a  Cost falls as stored fuel replaces batteries"); ax[0].legend(fontsize=7, frameon=False)
cats = ["Wind\nGW", "Solar\nGW", "Battery\nGWh", "Capital\nC$ bn"]
a = [ref["wind_GW"], ref["solar_GW"], ref["battery_GWh"], ref["capex_bn"]]; b = [sw["wind_GW"], sw["solar_GW"], sw["battery_GWh"], sw["capex_bn"]]
x = np.arange(4); ax[1].bar(x - 0.2, a, 0.4, color=GREY, label="Reoptimised"); ax[1].bar(x + 0.2, b, 0.4, color=GREEN, label="Sweet spot")
for i in range(4):
    ax[1].text(x[i] - 0.2, a[i], f"{a[i]:.0f}", ha="center", va="bottom", fontsize=7); ax[1].text(x[i] + 0.2, b[i], f"{b[i]:.0f}", ha="center", va="bottom", fontsize=7)
ax[1].set_xticks(x, cats, fontsize=7.5); ax[1].set_title("b  Smaller solar and battery build"); ax[1].legend(fontsize=7, frameon=False)
save(fig, "Figure_2_storage_fuel_sweet_spot")

# 2. 25-year annual electricity cost paths
fig, ax = plt.subplots(figsize=(7.2, 3.3))
yrs = np.arange(2027, 2052)
ax.plot(yrs, R["digital_gardens_25y"]["sweet_spot"]["path_cad_mwh"], color=GREEN, lw=2.2, label="Digital Gardens, sweet spot")
ax.plot(yrs, R["digital_gardens_25y"]["reference_selected"]["path_cad_mwh"], color=GREY, lw=1.6, label="Digital Gardens, 8 MW / 300 ML comparison")
for k, c, ls, lab in (("central_4_rising_3pct|tier_output_based", BLUE, "-", "Gas C$4 +3%/yr, output-based carbon (TIER)"),
                      ("central_4_rising_3pct|full_headline_price", BLUE, "--", "Gas C$4 +3%/yr, carbon price on every tonne"),
                      ("high_6_rising_4pct|full_headline_price", RED, "--", "Gas C$6 +4%/yr, carbon price on every tonne"),
                      ("central_4_rising_3pct|ccs_from_2035", ORANGE, ":", "Gas C$4 +3%/yr, 90% capture from 2035")):
    ax.plot(yrs, [p["cad_mwh"] for p in R["gas_paths"][k]], color=c, ls=ls, lw=1.4, label=lab)
ax.annotate("Battery\nreplacement", xy=(2041, 520), xytext=(2037.5, 650), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7))
ax.axvspan(2046.5, 2051.5, color=GREEN, alpha=0.08); ax.text(2049, 330, "Renewable\ndebt repaid", ha="center", fontsize=7, color=GREEN)
ax.set_ylabel("Annual cost (C$/MWh, real 2026)"); ax.set_xlim(2027, 2051); ax.legend(fontsize=6.8, frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.0)); ax.set_ylim(0, 1050)
ax.set_title("Electricity cost by year: renewable capital up front, gas fuel and carbon every year")
save(fig, "Figure_4_25_year_electricity_cost")

# 3. emissions per GW IT and the community offset
em = R["emissions_per_gw_it"]; it = R["it_gw"]
labels = ["Wonder Valley-type\ngas, unabated", "Wonder Valley-type\ngas, 90% capture", "Digital Gardens\n300-ML comparator", "Digital Gardens\nsweet spot"]
keys = ["Wonder Valley-type gas, unabated", "Wonder Valley-type gas, 90% capture", "Digital Gardens, published portfolio", "Digital Gardens, storage-fuel sweet spot"]
vals = [em[k]["total_mt"] for k in keys]
comm = COMM_MT / it
fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.bar(range(4), vals, color=[RED, ORANGE, GREY, GREEN], width=0.6)
ax.bar([2, 3], [-comm, -comm], color=BLUE, width=0.6, alpha=0.75, label="Potential community reductions (not offsets)")
for i, v in enumerate(vals):
    ax.text(i, v + 0.06, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
for i in (2, 3):
    ax.text(i + 0.33, -comm + 0.02, f"net {vals[i] - comm:.2f}", ha="left", fontsize=7.5, color=BLUE)
ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(4), labels, fontsize=7.8)
ax.set_ylabel("Mt CO$_2$e per GW of IT per year"); ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.set_title("Life-cycle emissions per GW of IT (PUE 1.10, 9.64 TWh per GW-year)")
save(fig, "Figure_5_emissions_per_GW_IT")

# 4. community GHG reductions vs Wonder Valley increase
g = GHG
short = {"Home heating and hot water from server heat (gas displaced)": "Home heat and hot water",
         "Asparagopsis for dairy and beef cattle (enteric methane)": "Red seaweed feed (cattle methane)",
         "Manure removed to digesters (storage methane avoided)": "Manure to digesters",
         "Renewable natural gas displacing fossil gas": "Renewable natural gas",
         "Methane-leak sensors on digesters and upgraders (fugitive avoided vs no sensors)": "Methane-leak sensors",
         "Farm diesel replaced by HVO": "Farm diesel → HVO",
         "Commercial drying contracts with server heat (gas and propane displaced)": "Commercial drying contracts",
         "Electric vehicles charged from Digital Gardens (5,000-vehicle programme)": "Electric vehicles (5,000)"}
s = pd.Series({short[k]: v / 1e6 for k, v in g.items()}).sort_values()
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax[0].barh(s.index, s.values, color=GREEN)
for i, v in enumerate(s.values):
    ax[0].text(v + 0.004, i, f"{v * 1000:.0f} kt", va="center", fontsize=7)
ax[0].set_xlabel("Mt CO$_2$e avoided per year"); ax[0].set_title(f"a  Potential reductions, full deployment: {COMM_MT:.2f} Mt/yr", fontsize=9)
wv_same = em[keys[0]]["total_mt"] * it; wv_full = em[keys[0]]["total_mt"] * 7.5 / 1.10
bars = [-COMM_MT, em[keys[3]]["total_mt"] * it, wv_same, wv_full]
ax[1].bar(range(4), bars, color=[BLUE, GREEN, RED, RED], alpha=0.9)
ax[1].set_xticks(range(4), ["Host\ncommunities", "Digital\nGardens", f"Gas,\n{it:.2f} GW IT", "Wonder\nValley\n7.5 GW"], fontsize=6.6)
for i, v in enumerate(bars):
    ax[1].text(i, v + (0.5 if v >= 0 else -1.6), f"{v:+.1f}", ha="center", fontsize=8, fontweight="bold")
ax[1].axhline(0, color="k", lw=0.6); ax[1].set_ylabel("Mt CO$_2$e per year"); ax[1].set_title("b  Change in emissions")
save(fig, "Figure_9_community_GHG_vs_Wonder_Valley")

# 5. heat balance with commercial drying (central contracts)
th = R["thermal"]
farm = sum(m["loop_heat_twh"] for m in DC["markets"] if m["kind"] == "farm"); ind = sum(m["loop_heat_twh"] for m in DC["markets"] if m["kind"] != "farm")
fig, ax = plt.subplots(figsize=(7.2, 2.7))
parts = [("Homes: space heat and hot water", th["space_dhw_twh"], NAVY), ("Aquaculture and greenhouses", th["aquaculture_greenhouse_twh"], BLUE),
         ("Digester heating", th["digester_twh"], GREEN), ("Farm drying, May–Nov", farm, ORANGE), ("Industrial drying contracts", ind, "#8C5A2B"),
         ("Network losses", th["recoverable_twh"] - th["delivered_to_network_twh"], GREY), ("Rejected (no contract in that hour)", DC["rejected_twh"], "#D5D8DC")]
left = 0
for lab, v, c in parts:
    ax.barh([0], [v], left=left, color=c, label=f"{lab} ({v:.2f} TWh)"); left += v
ax.set_yticks([]); ax.set_xlabel("TWh per year (recoverable IT heat = 85% of 32.86 TWh)")
ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.35))
ax.set_title(f"Hourly heat matching with drying contracts: {DC['utilisation_pct']:.1f}% of recoverable heat used; ERF {DC['erf_pct']:.1f}%")
save(fig, "Figure_7_heat_balance")

# 7. drying by month and how far contracts can push heat use
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw=dict(width_ratios=[1.3, 1]))
ax = axs[0]; mth = np.arange(1, 13)
avail = np.array(D["available_twh_by_month"]); base_surplus = np.array(D["base_without_drying"]["monthly_surplus_twh"])
base_used = avail - base_surplus
f_ = np.array(DC["monthly_farm_twh"]); i_ = np.array(DC["monthly_industry_twh"]); rj = np.array(DC["monthly_rejected_twh"])
ax.bar(mth, base_used, color=NAVY, label="Homes, digesters, aquaculture")
ax.bar(mth, f_, bottom=base_used, color=ORANGE, label="Farm drying")
ax.bar(mth, i_, bottom=base_used + f_, color="#8C5A2B", label="Industrial drying")
ax.bar(mth, rj, bottom=base_used + f_ + i_, color="#D5D8DC", label="Still rejected")
ax.set_xticks(mth, list("JFMAMJJASOND")); ax.set_ylabel("TWh per month"); ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
ax.set_title("a  Heat use by month, central contracts", fontsize=8.5)
ax = axs[1]; sw_ = pd.DataFrame(D["scale_sweep"])
ax.plot(sw_.water_mt, sw_.utilisation_pct, "o-", color=NAVY, label="Recoverable heat used")
ax.plot(sw_.water_mt, sw_.erf_pct, "s--", color=GREEN, label="Energy reuse factor")
c1 = sw_[sw_.scale == 1.0].iloc[0]
ax.annotate("Central\ncontracts", xy=(c1.water_mt, c1.utilisation_pct), xytext=(c1.water_mt + 1.2, c1.utilisation_pct - 14), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7))
ax.set_xlabel("Water removed by drying contracts (Mt/yr)"); ax.set_ylabel("%"); ax.legend(fontsize=7, frameon=False, loc="upper left")
ax.set_title("b  Heat use rises with contracted tonnage", fontsize=8.5)
save(fig, "Figure_8_drying_by_month_and_scale")

# 6. GPU profit
G = pd.read_csv(OUT / "gpu_profit_grid.csv")
tiers = ["spot (conservative)", "market on-demand", "premium tier", "top tier"]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
ax = axs[0]; w = 0.2
for j, (prod, mk, c, h) in enumerate((("GB300 NVL72", "prices held each generation", NAVY, None), ("HGX B300", "prices held each generation", GREEN, None),
                                      ("GB300 NVL72", "prices compress 15% per generation", NAVY, "///"), ("HGX B300", "prices compress 15% per generation", GREEN, "///"))):
    q = G[(G.electricity == "Digital Gardens sweet spot") & (G.label == 0.5) & (G["product"] == prod) & (G.market == mk)].set_index("tier").loc[tiers]
    ax.bar(np.arange(4) + (j - 1.5) * w, q.npv_bn, w, color=c if h is None else "white", edgecolor=c, hatch=h,
           label=f"{'NVL72' if prod.startswith('GB') else 'B300'}, {'prices held' if 'held' in mk else 'prices compress 15%/gen'}")
ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(4), ["Spot\n$5.87", "Market\n$8.98", "Premium\n$12.00", "Top\n$17.80"], fontsize=7.5)
ax.set_ylabel("25-year NPV at 9% (US$ bn)"); ax.legend(fontsize=6.5, frameon=False); ax.set_title("a  Digital Gardens, +$0.50 on 50% of hours", fontsize=8.5)
ax = axs[1]
BE = pd.read_csv(OUT / "label_breakeven.csv")
q = BE[(BE.tier == "market on-demand") & (BE.market == "prices held each generation") & (BE.gas_case == "Gas (central price, TIER)")]
xs = np.linspace(0, 1.2, 25)
for prod, c in (("GB300 NVL72", NAVY), ("HGX B300", GREEN)):
    for share, ls in ((1.0, "-"), (0.5, "--")):
        r_ = q[(q["product"] == prod) & (q.labelled_share == share)].iloc[0]
        slope = r_.npv_gap_bn / r_.premium_usd_per_labelled_hour
        ax.plot(xs, slope * xs - r_.npv_gap_bn, color=c, ls=ls, lw=1.3,
                label=f"{'NVL72' if prod.startswith('GB') else 'B300'}, premium on {int(share * 100)}% of hours (break-even ${r_.premium_usd_per_labelled_hour:.2f})")
ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("Label premium (US$ per labelled GPU-hour)"); ax.set_ylabel("NPV gain over gas twin (US$ bn)")
ax.legend(fontsize=6.2, frameon=False, loc="upper left"); ax.set_title("b  Premium needed to match a gas-powered twin", fontsize=8.5)
fig.suptitle(f"GPU rental profit, {R['it_gw']:.2f} GW of IT, 85% of hours sold", fontsize=10, fontweight="bold")
save(fig, "Figure_6_GPU_profit")
# Site-specific wind-resource sensitivity is regenerated by FIGURE_SITE_FREE.py.
# 9. NOx control
NX = json.load(open(OUT / "nox_control_results.json")); pkg = pd.DataFrame(NX["packages"])
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw=dict(width_ratios=[1.6, 1]))
ax = axs[0]; q = pkg.iloc[2:]
short = [p.replace("HVO + ", "").replace("retarded timing", "timing").replace(" + ", " +\n") for p in q.package]
cols = [GREEN if (w and v <= 1.0) else (ORANGE if w else GREY) for w, v in zip(q.within_ceiling, q.vs_gas_twin)]
ax.bar(range(len(q)), q.nox_t_yr, color=cols, hatch=None)
for i_, (v, w) in enumerate(zip(q.nox_t_yr, q.within_ceiling)):
    ax.text(i_, v + 40, f"{v:,.0f}" + ("" if w else "\n(needs\n+5 GW PV)"), ha="center", fontsize=6.5)
ax.axhline(NX["gas_twin_nox_t"], color=RED, ls="--", lw=1); ax.text(-0.45, NX["gas_twin_nox_t"] + 60, "gas twin with SCR", color=RED, fontsize=7, ha="left")
ax.set_xticks(range(len(q)), short, fontsize=6.3); ax.set_ylabel("NOx, t/yr (17 nodes)"); ax.set_ylim(0, 3400)
ax.set_title(f"a  Engine NOx by control package (uncontrolled {pkg.nox_t_yr.iloc[1]:,.0f} t)", fontsize=8.5)
ax = axs[1]; mth = np.arange(1, 13)
ax.bar(mth, NX["engine_profile"]["monthly_twh"], color=NAVY); ax.set_xticks(mth, list("JFMAMJJASOND"))
ax.set_ylabel("Engine output, TWh/month"); ax.set_title(f"b  When the engines run ({NX['engine_profile']['hours_per_year']:,.0f} h/yr)", fontsize=8.5)
save(fig, "Figure_10_NOx_control")
# 1. Digital Gardens versus Wonder Valley: carbon ledger, water and gas
CMP = R["comparison"]
fig, axs = plt.subplots(1, 3, figsize=(7.8, 4.0), gridspec_kw=dict(width_ratios=[1.6, 0.8, 0.8]))
ax = axs[0]
base = CMP["community_baseline_mt"]
ax.bar(0, base, color=GREY, width=0.55, label="Host-community emissions (17 municipalities)")
ax.bar(0, CMP["wv_facility_mt"], bottom=base, color=RED, width=0.55, label="Gas campus (Wonder Valley-type), life cycle")
ax.bar(1, base, color=GREY, width=0.55)
ax.bar(1, CMP["dg_facility_mt"] + CMP["dg_n2o_mt"], bottom=base, color=GREEN, width=0.55, label="Digital Gardens, life cycle")
top_dg = base + CMP["dg_facility_mt"] + CMP["dg_n2o_mt"]
ax.bar(1.32, -CMP["reductions_full_mt"], bottom=top_dg, color="white", edgecolor=BLUE, hatch="////", width=0.22,
       label="Community reductions (heat, seaweed, biogas, EVs, drying…)")
ax.hlines(CMP["ledger_dg_full_mt"], 0.72, 1.45, color=BLUE, lw=1.2)
ax.text(0, CMP["ledger_wv_mt"] + 0.5, f"{CMP['ledger_wv_mt']:.1f} Mt", ha="center", fontsize=8, fontweight="bold")
ax.text(1, top_dg + 0.5, f"net {CMP['ledger_dg_full_mt']:.1f} Mt", ha="center", fontsize=8, fontweight="bold", color=BLUE)
ax.set_xticks([0, 1], ["Communities +\ngas campus", "Communities +\nDigital Gardens"], fontsize=7.5)
ax.set_ylabel("Mt CO$_2$e per year"); ax.set_ylim(0, 27); ax.set_xlim(-0.5, 1.6)
ax.legend(fontsize=6.2, frameon=False, loc="upper center", bbox_to_anchor=(0.6, -0.16), ncol=1)
ax.set_title(f"a  Carbon ledger, {CMP['it_gw']:.2f} GW of IT", fontsize=8.5)
ax = axs[1]
ax.bar([0, 1], [CMP["water_gl_wonder_valley"], 0.0], color=[RED, GREEN], width=0.55)
ax.text(0, CMP["water_gl_wonder_valley"] + 0.15, f"{CMP['water_gl_wonder_valley']:.0f} GL/yr", ha="center", fontsize=7.5)
ax.text(1, 0.15, "0", ha="center", fontsize=7.5)
ax.set_xticks([0, 1], ["Wonder\nValley", "Digital\nGardens"], fontsize=7.5); ax.set_ylabel("Cooling and process water, GL/yr"); ax.set_ylim(0, 7.5)
ax.set_title("b  Water withdrawn", fontsize=8.5)
ax = axs[2]
ax.bar([0, 1], [CMP["gas_pj_equal_it"], 0.0], color=[RED, GREEN], width=0.55)
ax.text(0, CMP["gas_pj_equal_it"] + 4, f"{CMP['gas_pj_equal_it']:.0f} PJ\n({CMP['gas_bcm_equal_it']:.1f} bcm)", ha="center", fontsize=7)
ax.text(1, 4, "0", ha="center", fontsize=7.5)
ax.set_xticks([0, 1], ["Gas\ncampus", "Digital\nGardens"], fontsize=7.5); ax.set_ylabel("Fossil gas burned, PJ/yr"); ax.set_ylim(0, 280)
ax.set_title("c  Fossil gas", fontsize=8.5)
save(fig, "Figure_1_Digital_Gardens_vs_Wonder_Valley")
print("figures written to", FIG)
