#!/usr/bin/env python3
"""Recomputed large-turbine stored-output consistency checks.
Scientific field validation and global-optimizer convergence are NOT implied.
"""
from pathlib import Path
import json,math
from docx import Document
P=Path(__file__).resolve().parent;L=P/'LARGE_TURBINE_STUDY'/'OUTPUT';O=P/'OUTPUT';SUB=P.parent/'SUBMISSION'
J=lambda p:json.loads(Path(p).read_text())
a=J(L/'SANY_SI19580_8MW_optimization.json');b=J(L/'SANY_SI230100_10MW_optimization.json');s=J(O/'storage_fuel_sweetspot.json');pc=J(O/'profit_community_results.json');nx=J(O/'nox_control_results.json');d=J(O/'industrial_drying_results.json');
checks=[]
def chk(name,ok):
 assert ok,name
 checks.append(name)
for n,j,mw in [('8 MW',a,8),('10 MW',b,10)]:
 c=j['candidate']; cap=list(c['wind_gw_by_site'].values())
 chk(n+' independent five-site allocation',len(cap)==5)
 chk(n+' wind total',abs(sum(cap)-c['wind_gw'])<1e-8)
 chk(n+' Nicola at most 0.2 GW',c['wind_gw_by_site']['nicola_wind_project_regional_proxy']<=0.2)
 chk(n+' each site at most 2 GW',max(cap)<=2.000001)
 chk(n+' screened site CF at most 40%',max(x['screened_cf'] for x in j['sites'])<.400001)
 chk(n+' turbine count consistent',c['turbine_count_total_ceil']==sum(math.ceil(q*1000/mw) for q in cap))
 chk(n+' baseline conditional gas zero',c['stats']['gas_gwh_max']<.001)
 chk(n+' baseline unserved load zero',c['stats']['unserved_gwh_max']<.001)
 chk(n+' worst-year supply <= fuel ceiling',c['stats']['hvo_ml_max']<c['fuel_cap_ml']+.01)
 chk(n+' 35% same-design cap not gas-free',j['stress']['cap35_same_installation']['gas_gwh_max']>100)
 chk(n+' 10% GWA speed reduction not gas-free',j['stress']['GWA90pct_cap40']['gas_gwh_max']>300)
 chk(n+' benchmark raw ERA5 requires substantial gas',j['stress']['raw_ERA5']['gas_gwh_max']>1000)
 chk(n+' optimization not falsely certified',j['search']['success'] is False)
chk('8 MW manufacturer rated speed cited correctly',a['manufacturer_params']['rated_ms']==11.5)
chk('10 MW assumed speed explicitly hypothetical', 'ASSUMED' in b['manufacturer_params']['curve'])
chk('10 MW 11.5 rated proxy loses gas-free condition',b['stress']['10MW_assumed_rated_11.5ms']['gas_gwh_max']>200)
chk('8 MW selected matches common model',abs(s['sweet_spot']['wind_GW']-a['candidate']['wind_gw'])<1e-6)
chk('8 MW selected solar',abs(s['sweet_spot']['solar_GW']-a['candidate']['solar_gw'])<1e-6)
chk('8 MW selected battery',abs(s['sweet_spot']['battery_GWh']-a['candidate']['battery_gwh'])<1e-6)
chk('8 MW selected cost',abs(s['sweet_spot']['lcoe_25y_cad_mwh']-a['candidate']['cost']['lcoe_25y_cad_mwh'])<1e-6)
chk('8 MW reference independently feasible',s['reference_selected']['qualifies'] and s['reference_selected']['gas_gwh_max']<.01)
chk('8 MW linked drying central',35<d['central']['utilisation_pct']<37)
chk('8 MW linked drying updated capex',.6<d['central']['economics']['capex_bn']<.7)
selected=next(z for z in nx['packages'] if z['package']==nx['recommended'])
chk('8 MW NOx present',900<selected['nox_t_yr']<950)
chk('Gas NOx reference present',abs(nx['gas_twin_nox_t']-900)<1)
chk('Facility emissions include N2O',1.8<pc['comparison']['dg_facility_mt']+pc['comparison']['dg_n2o_mt']<2.0)
chk('Gas comparator emissions',15.3<pc['comparison']['wv_facility_mt']<15.5)
chk('Community avoidance separate',pc['comparison']['reductions_limited_mt']<pc['comparison']['reductions_full_mt'])
for name in ['Joule_Research_Article','Joule_Supplemental_Information','Joule_Cover_Letter','Joule_Highlights_and_eTOC','Joule_Key_Resources_Table']:
 chk(name+' Word and PDF supplied',(SUB/(name+'.docx')).exists() and (SUB/(name+'.pdf')).exists())
article=Document(SUB/'Joule_Research_Article.docx');text='\n'.join(p.text for p in article.paragraphs)
chk('Wonder Valley explicitly retained','Wonder Valley' in text)
chk('10 MW turbine discussed','SI-230100' in text)
chk('manuscript has six main figures',len(article.inline_shapes)==6)
chk('only one main table',len(article.tables)==1)
chk('unpublished DOI not fabricated','DOI is not yet registered' in text or 'DOI has not been registered' in text or 'permanent DOI for this exact research package has yet been registered' in text)
print('PASS',len(checks),'large-turbine archived-output checks. No proof of field performance, bankability or global optimality.')
for k in checks:print('-',k)
