# -*- coding: utf-8 -*-
"""Export supplementary-table data (English) from the R xlsx to JSON for docx build."""
import pandas as pd, numpy as np, json, os

CANDS=["/Users/guns/Desktop/도시_농촌_분석","/sessions/affectionate-lucid-bohr/mnt/도시_농촌_분석"]
ROOT=next((p for p in CANDS if os.path.isdir(p)),CANDS[0])
IJ=f"{ROOT}/results/B_cereus/IJEHR_B_cereus"
MAIN=f"{IJ}/B_cereus_MAIN_2011to2024_260620_2016.xlsx"
SUPP=f"{IJ}/B_cereus_SUPP_2011to2024_260620_2016.xlsx"
OUT=f"{ROOT}/results/B_cereus/SUBMISSION_IJEHR/_supp_data.json"
os.makedirs(os.path.dirname(OUT),exist_ok=True)

VAR_EN={"성비":"Sex ratio","고령인구비율":"Elderly rate","도시지역인구비율":"Urban pop. rate",
"상수도보급률":"Piped-water","하수도보급률":"Sewerage","고도처리인구보급률":"Adv. treatment",
"정화조인구":"Septic-tank pop.","가정용지하수개소수":"Household groundwater","민방위용지하수이용량":"Civil-defense groundwater",
"유지":"Reservoir","하천":"Stream/river","목장용지":"Pasture","답":"Paddy field","임야":"Forest","의사수":"Physicians",
"관내진료비전체":"In-area medical cost","관내진료비_전체":"In-area medical cost","인플루엔자예방접종률":"Influenza vaccination",
"건강생활실천율":"Healthy-lifestyle","걷기실천율":"Walking practice","재정자립도":"Fiscal independence","재정자주도":"Fiscal autonomy",
"사육두수합계":"Livestock total","농가수합계":"Farms total","패류생산총합":"Shellfish production","갑각류생산총합":"Crustacean production"}
def en(v):
    s=str(v).strip(); return VAR_EN.get(s,s)
CAT_EN={"⑦ 인구학적 특성":"Demographic","③ 수질 및 하수 인프라":"Water & sanitation","④ 수계 및 토지이용":"Land use",
"⑧ 의료접근":"Healthcare","⑥ 사회경제 및 취약성":"Socioeconomic","① 병원소 및 축산":"Livestock/zoonotic","① 식품원 및 축산":"Food source"}
def cat(v): return CAT_EN.get(str(v).strip(),str(v).strip())

data={}

# S: stratified IRR (Urban / Rural) from Table2_*
def strata(sheet):
    t=pd.read_excel(MAIN,sheet,header=None); h=t.index[t[0]=="Category"][0]; rows=[]
    for i in range(h+1,len(t)):
        if pd.isna(t.iloc[i,1]): break
        rows.append([cat(t.iloc[i,0]),en(t.iloc[i,1]),str(t.iloc[i,2]),
                     f"{float(t.iloc[i,3]):.2f}",f"{float(t.iloc[i,4]):.2f}–{float(t.iloc[i,5]):.2f}",
                     "*" if str(t.iloc[i,6])=="★" else ""])
    return rows
data["urban"]=strata("Table2_Urban")
data["rural"]=strata("Table2_Rural")

# S: period sweep
s1=pd.read_excel(SUPP,"S1_PeriodSweep",header=None); h=s1.index[s1[0]=="start"][0]
per=[]
for i in range(h+1,len(s1)):
    r=s1.iloc[i]
    if pd.isna(r[0]) or not str(r[0]).strip().isdigit(): continue
    per.append([str(int(r[0])),f"{int(r[1])}",f"{int(r[2])}",f"{int(r[3])}",
                f"{float(r[4]):+.3f}",f"{float(r[5]):.3f}",str(r[6]),
                f"{float(r[7]):.1f}",str(int(r[10])),str(int(r[11])),str(int(r[12]))])
data["period"]=per

# S: model comparison
mc=pd.read_excel(MAIN,"Fig_ModelComparison",header=None); mc.columns=mc.iloc[0]; mc=mc.iloc[1:]
data["modelcomp"]=[[str(r["model"]),f"{float(r['DIC']):.2f}",f"{float(r['WAIC']):.2f}",
                    f"{float(r['postMoranI']):+.4f}",f"{float(r['postMoranP']):.4f}",f"{float(r['dDIC']):.2f}"] for _,r in mc.iterrows()]

# S: full interaction
fi=pd.read_excel(SUPP,"S3_Interaction_full",header=None)
inter=[]
for i in range(1,len(fi)):
    if pd.isna(fi.iloc[i,0]): break
    r=fi.iloc[i]
    inter.append([en(r[0]),f"{float(r[1]):.2f}",str(r[2]),f"{float(r[3]):.2f}",
                  f"{float(r[4]):.2f}",str(r[5]),"*" if str(r[6])=="★" else ""])
data["interaction"]=inter

# S: calibration
s4=pd.read_excel(SUPP,"S4_CPO_PIT",header=None)
cal=[]
for i in range(1,len(s4)):
    if pd.isna(s4.iloc[i,0]): break
    m=str(s4.iloc[i,0]); m=m.split("(")[0].strip()
    cal.append([m,str(s4.iloc[i,1])])
data["calib"]=cal

# S: COVID excluded
s5=pd.read_excel(SUPP,"S5_COVIDexcluded",header=None); s5.columns=s5.iloc[0]; s5=s5.iloc[1:]
cov=[]
for _,r in s5.iterrows():
    if pd.isna(r["Variable"]): continue
    cov.append([en(r["Variable"]),f"{float(r['IRR']):.2f}",str(r["CrI"]),"*" if str(r["Sig"])=="★" else ""])
data["covid"]=cov

# S: food pathway
s7=pd.read_excel(SUPP,"S7_FoodSensitivity_2020_2024",header=None); h=s7.index[s7[0]=="tier"][0]
food=[]
for i in range(h+1,len(s7)):
    if pd.isna(s7.iloc[i,2]): break
    r=s7.iloc[i]
    try: irr,lo,hi=float(r[4]),float(r[5]),float(r[6])
    except: continue
    food.append([cat(r[1]),en(r[2]),str(r[3]),f"{irr:.2f}",f"{lo:.2f}–{hi:.2f}","*" if str(r[7])=="★" else ""])
data["food"]=food

json.dump(data,open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=1)
print("exported",OUT)
for k,v in data.items(): print(f"  {k}: {len(v)} rows")
