# -*- coding: utf-8 -*-
"""
BC_build_covariate_panel.py
---------------------------------------------------------------
Reproduces the 21 PRE-SPECIFIED covariates used by the B. cereus
urban/rural ZINB analysis (config: 00_config_공통.R / TV_PRESPEC),
purely for DESCRIPTIVE Figure 1 panels (distributions / transforms).

Faithful to build_cor_merged():
  - region cleaning (strip spaces; 인천시미추홀구 -> 인천시남구)
  - region x year mean aggregation
  - whole-year-all-zero column -> NA (KCHS unsurveyed-year coding)
  - period 2011-2024
Region-level value = mean over available (non-NA) years.

NOTE: This file only feeds DESCRIPTIVE covariate distributions in
Figure 1 (d-f). All inferential numbers (IRR, DIC, incidence) come
straight from the R-generated xlsx and are never recomputed here.
Output: BC_covariate_panel_region.csv (cached for reproducibility).
"""
import pandas as pd, numpy as np, re, os

# Path auto-detection (host machine vs. analysis sandbox)
_CANDS = ["/Users/guns/Desktop/도시_농촌_분석",
          "/sessions/affectionate-lucid-bohr/mnt/도시_농촌_분석"]
ROOT = next((p for p in _CANDS if os.path.isdir(p)), _CANDS[0])
BASE = f"{ROOT}/FBD_DATA_ZIP"
OUT  = f"{ROOT}/results/B_cereus/IJEHR_B_cereus/BC_covariate_panel_region.csv"
YS, YE = 2011, 2024

# code -> (English label, theory direction) from TV_PRESPEC
COVS = [
    ("성비",                          "Sex ratio",            "neutral", "parquet"),
    ("고령인구비율",                   "Elderly rate (%)",     "neutral", "parquet"),
    ("도시지역인구비율",               "Urban pop. rate (%)",  "risk",    "parquet"),
    ("상수도보급률",                   "Piped-water (%)",      "protect", "parquet"),
    ("하수도보급률",                   "Sewerage (%)",         "protect", "parquet"),
    ("고도처리인구보급률(%)",          "Adv. treatment (%)",   "protect", "sewer"),
    ("하수처리구역외_정화조인구",      "Septic-tank pop.",     "risk",    "sewer"),
    ("가정용_개소수",                  "Household groundwater wells", "risk", "gw"),
    ("민방위용_이용량",                "Civil-defense groundwater use", "risk", "gw"),
    ("유지",                          "Reservoir (area)",     "risk",    "land"),
    ("하천",                          "Stream/river (area)",  "risk",    "land"),
    ("목장용지",                       "Pasture (area)",       "risk",    "land"),
    ("답",                            "Paddy field (area)",   "risk",    "land"),
    ("임야",                          "Forest (area)",        "neutral", "land"),
    ("의사수",                         "Physicians",           "protect", "parquet"),
    ("관내진료비_전체",                "In-area medical cost", "protect", "parquet"),
    ("연간인플루엔자예방접종률_표준화율","Influenza vaccination (%)","protect","parquet"),
    ("건강생활실천율_조율",            "Healthy-lifestyle (%)","protect", "parquet"),
    ("걷기실천율_표준화율",            "Walking practice (%)", "protect", "parquet"),
    ("재정자립도",                     "Fiscal independence",  "protect", "parquet"),
    ("재정자주도",                     "Fiscal autonomy",      "protect", "parquet"),
]

def clean_region(df):
    df = df.copy()
    df["region"] = df["region"].astype(str).str.replace(r"\s+", "", regex=True)
    df.loc[df.region == "인천시미추홀구", "region"] = "인천시남구"
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df[(df.year >= YS) & (df.year <= YE)]

def agg_year(df, cols):
    df = clean_region(df)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.groupby(["region", "year"], as_index=False)[cols].mean()

# ---- assemble region x year panel ----
panel = None
def merge_in(tab):
    global panel
    panel = tab if panel is None else panel.merge(tab, on=["region", "year"], how="outer")

# parquet
pq = pd.read_parquet(f"{BASE}/국민건강결과_최종.parquet")
pq_codes = [c for c, *_ , src in COVS if src == "parquet"]
merge_in(agg_year(pq, pq_codes))
# sewer csv
sw = pd.read_csv(f"{BASE}/merged_하수도보급률.csv")
merge_in(agg_year(sw, [c for c, *_ , src in COVS if src == "sewer"]))
# groundwater csv
gw = pd.read_csv(f"{BASE}/생활용지하수이용현황_전처리.csv")
merge_in(agg_year(gw, [c for c, *_ , src in COVS if src == "gw"]))
# land-use csv
ld = pd.read_csv(f"{BASE}/국토이용현황_전처리_수정.csv")
merge_in(agg_year(ld, [c for c, *_ , src in COVS if src == "land"]))

# whole-year-all-zero -> NA (per build_cor_merged finalize step)
codes = [c for c, *_ in COVS]
for c in codes:
    for y, idx in panel.groupby("year").groups.items():
        v = panel.loc[idx, c]
        fin = v[np.isfinite(v)]
        if len(fin) > 0 and (fin == 0).all():
            panel.loc[idx, c] = np.nan

# region-level mean over available years
reg = panel.groupby("region")[codes].mean().reset_index()
reg = clean_region(reg.assign(year=YE)).drop(columns="year")  # standardize region names

# urban/rural classification (classify_region)
metro = re.compile(r"^(서울시|부산시|대구시|인천시|광주시|대전시|울산시)")
def classify(r):
    if "세종" in r: return "Urban"
    if metro.match(r) and not r.endswith("군"): return "Urban"
    if r.endswith("군"): return "Rural"
    return "Urban"  # 중소도시 -> Urban (도시)
reg["UR"] = reg["region"].map(classify)

# rename to English labels
ren = {c: lab for c, lab, _d, _s in COVS}
reg = reg.rename(columns=ren)
reg.to_csv(OUT, index=False, encoding="utf-8-sig")
print("✅ covariate panel written:", OUT)
print("   regions:", len(reg), "| Urban:", (reg.UR=="Urban").sum(), "Rural:", (reg.UR=="Rural").sum())
print("   columns:", [c for c in reg.columns if c not in ('region','UR')][:6], "...")
