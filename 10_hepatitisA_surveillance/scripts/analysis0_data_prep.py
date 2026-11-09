"""Hepatitis A reanalysis (2011-2024) — data preparation
================================================================
Produces canonical CSVs used by analyses 1-9:
  - tbl_annual_rates.csv:   year, total, domestic, imported, pop_total
  - tbl_sido_year.csv:      year, sido_kr, sido_en, cases (domestic), pop
  - tbl_sido_rate.csv:      sido_en, total_cases, dom_cases, imp_cases,
                            mean_pop, mean_annual_rate, dom_rate, imp_rate
  - weekly_total.csv:       date (ISO Mon), cases (national domestic weekly)
  - df2_long.csv:           year, sido, sex, age_group, cases (clean, long form)
  - pop_sido_year.csv:      year, sido_kr, sido_en, pop_total

Population source: population_final_harmonized.csv. Eleven 일반시 (mid-tier
cities with 자치구) were 2× duplicated in the source (확인: 수원·성남·안양·
안산·고양·용인·창원·청주·천안·전주·포항). This script halves those records
so the resulting sigungu and sido populations align with KOSIS 주민등록인구.
"""
import pandas as pd
import numpy as np
import os

DATA = '/sessions/elegant-quirky-hamilton/mnt/논문_이건상/A형간염/data'
OUT  = '/sessions/elegant-quirky-hamilton/mnt/outputs/hepa_work'
os.makedirs(OUT, exist_ok=True)

# Sigungu names (city-level, no sido qualifier) where the population data is 2x
DOUBLED_CITIES = {
    '수원시': '경기도',
    '성남시': '경기도',
    '안양시': '경기도',
    '안산시': '경기도',
    '고양시': '경기도',
    '용인시': '경기도',
    '창원시': '경상남도',
    '청주시': '충청북도',
    '천안시': '충청남도',
    '전주시': '전라북도',
    '포항시': '경상북도',
}

# Sido normalisation (data uses mixed forms): map all to a 17-sido EN palette
SIDO_EN = {
    '강원도':'Gangwon', '경기도':'Gyeonggi', '경상남도':'Gyeongnam',
    '경상북도':'Gyeongbuk', '광주시':'Gwangju', '대구시':'Daegu',
    '대전시':'Daejeon', '부산시':'Busan', '서울시':'Seoul',
    '세종시':'Sejong', '울산시':'Ulsan', '인천시':'Incheon',
    '전라남도':'Jeonnam', '전라북도':'Jeonbuk', '제주도':'Jeju',
    '충청남도':'Chungnam', '충청북도':'Chungbuk',
    # Variants that may appear
    '광주광역시':'Gwangju', '대구광역시':'Daegu', '대전광역시':'Daejeon',
    '부산광역시':'Busan', '서울특별시':'Seoul', '세종특별자치시':'Sejong',
    '울산광역시':'Ulsan', '인천광역시':'Incheon',
    '제주특별자치도':'Jeju', '강원특별자치도':'Gangwon',
    '전북특별자치도':'Jeonbuk',
}

# ─── 1. Population: load + correct ────────────────────────────────────────
pop = pd.read_csv(f'{DATA}/population_final_harmonized.csv', encoding='utf-8-sig')
pop.columns = pop.columns.str.strip()

print('Original population_final rows:', len(pop))
# Halve the 11 doubled cities
mask = pop['sigungu'].isin(DOUBLED_CITIES.keys())
# Verify that sido of the row matches DOUBLED_CITIES[sg] (avoid halving 광주시 in 경기도)
mask &= pop.apply(lambda r: DOUBLED_CITIES.get(r['sigungu']) == r['sido'], axis=1)
n_double = mask.sum()
print(f'Halving {n_double} rows for {len(DOUBLED_CITIES)} doubled cities')
pop.loc[mask, 'population'] = pop.loc[mask, 'population'] / 2

# National total per year (sex='계', age='계') — also add Sejong correction below
pop_year = pop[(pop.sex=='계') & (pop.age_group=='계')].groupby('year')['population'].sum()
SEJONG_POP_FOR_NATIONAL = {2011:0, 2012:113117, 2013:122153, 2014:156125, 2015:210884,
                          2016:243048, 2017:280100, 2018:314126, 2019:340575, 2020:355831,
                          2021:371895, 2022:383591, 2023:386525, 2024:388927}
for y, p in SEJONG_POP_FOR_NATIONAL.items():
    if y in pop_year.index:
        pop_year[y] = pop_year[y] + p
print('\nCorrected national total population by year:')
for y in range(2011, 2025):
    print(f'  {y}: {pop_year[y]:,.0f}')

# Sido × year population (sex='계', age='계'): sum sigungu within each sido
pop_sido_year = (
    pop[(pop.sex=='계') & (pop.age_group=='계')]
    .groupby(['year','sido'])['population'].sum()
    .reset_index()
)
pop_sido_year['sido_en'] = pop_sido_year['sido'].map(SIDO_EN)
pop_sido_year = pop_sido_year.rename(columns={'sido':'sido_kr','population':'pop_total'})

# Sejong is missing from population_final — KOSIS 주민등록인구 (mid-year) approximations
SEJONG_POP = {
    2011: 0,           # not yet established (Jul 2012)
    2012: 113117,
    2013: 122153,
    2014: 156125,
    2015: 210884,
    2016: 243048,
    2017: 280100,
    2018: 314126,
    2019: 340575,
    2020: 355831,
    2021: 371895,
    2022: 383591,
    2023: 386525,
    2024: 388927,
}
sejong_rows = pd.DataFrame([
    {'year': y, 'sido_kr': '세종시', 'sido_en': 'Sejong', 'pop_total': p}
    for y, p in SEJONG_POP.items()
])
pop_sido_year = pd.concat([pop_sido_year, sejong_rows], ignore_index=True)
pop_sido_year = pop_sido_year[pop_sido_year['pop_total'] > 0].reset_index(drop=True)
pop_sido_year.to_csv(f'{OUT}/pop_sido_year.csv', index=False)
print(f'\npop_sido_year.csv: {len(pop_sido_year)} rows')

# ─── 2. df1 (weekly × sido, domestic) ─────────────────────────────────────
df1 = pd.read_csv(f'{DATA}/df1_A형간염_harmonized.csv', encoding='utf-8-sig')
df1.columns = df1.columns.str.strip()
df1 = df1[df1['year'].between(2011, 2024)]
df1['sido_en'] = df1['sido'].map(SIDO_EN)
print(f'\ndf1: {len(df1)} weekly×sido rows; total domestic = {df1.cases.sum():,}')

# Build weekly national total (domestic)
weekly = df1.groupby(['year','week'])['cases'].sum().reset_index()
weekly['date'] = pd.to_datetime(
    weekly['year'].astype(str) + '-W' + weekly['week'].astype(str).str.zfill(2) + '-1',
    format='%G-W%V-%u', errors='coerce')
weekly = weekly.dropna(subset=['date']).groupby('date', as_index=True)['cases'].sum().sort_index()
full_range = pd.date_range(weekly.index.min(), weekly.index.max(), freq='W-MON')
weekly = weekly.reindex(full_range).fillna(0.0)
weekly.index.name = 'date'
weekly.to_csv(f'{OUT}/weekly_total.csv')
print(f'weekly_total.csv: {len(weekly)} weeks')

# Sido × year cases (domestic)
sy = df1.groupby(['year','sido','sido_en'])['cases'].sum().reset_index()
sy = sy.rename(columns={'cases':'dom_cases','sido':'sido_kr'})
sy = sy.merge(pop_sido_year, on=['year','sido_kr','sido_en'], how='left')
sy.to_csv(f'{OUT}/tbl_sido_year.csv', index=False)
print(f'tbl_sido_year.csv: {len(sy)} rows')

# ─── 3. df2_clean (yearly × sido × sex × age, all cases) ─────────────────
df2 = pd.read_csv(f'{DATA}/df2_A형간염_clean.csv')
df2.columns = df2.columns.str.strip()
df2 = df2[df2['year'].between(2011, 2024)]
df2['sido_en'] = df2['sido'].map(SIDO_EN)
df2.to_csv(f'{OUT}/df2_long.csv', index=False)
print(f'df2_long.csv: {len(df2)} rows; total (sex=계,age=계) = '
      f'{df2[(df2.sex=="계") & (df2.age_group=="계")].cases.sum():,}')

# ─── 4. Annual rates: total, domestic, imported, pop ─────────────────────
yr_total = df2[(df2.sex=='계') & (df2.age_group=='계')].groupby('year').cases.sum()
yr_dom   = df1.groupby('year').cases.sum()
# Imported = total - domestic; tiny negative values in 3 years are reporting
# artefacts (df1 has marginally more after corrections) — kept as-is so that
# the cumulative imported = 263 matches the published difference.
yr_imp   = (yr_total - yr_dom.reindex(yr_total.index, fill_value=0))
ann = pd.DataFrame({
    'year':     yr_total.index,
    'total':    yr_total.values,
    'domestic': yr_dom.reindex(yr_total.index, fill_value=0).values,
    'imported': yr_imp.values,
    'pop_total': pop_year.reindex(yr_total.index).values,
})
ann.to_csv(f'{OUT}/tbl_annual_rates.csv', index=False)
print(f'\ntbl_annual_rates.csv: {len(ann)} rows')
print(ann.to_string(index=False))
print(f'\nGrand totals: total={ann.total.sum():,}, domestic={ann.domestic.sum():,}, '
      f'imported={ann.imported.sum():,}')

# ─── 5. Sido rate aggregate ──────────────────────────────────────────────
years = list(range(2011, 2025))
N_YR = len(years)

agg_dom = sy.groupby(['sido_kr','sido_en']).agg(
    dom_cases=('dom_cases','sum'),
    mean_pop=('pop_total','mean'),
).reset_index()

# Total cases by sido (from df2 sex=계 age=계 across years)
agg_tot = (df2[(df2.sex=='계') & (df2.age_group=='계')]
           .groupby(['sido','sido_en']).cases.sum()
           .reset_index().rename(columns={'sido':'sido_kr','cases':'total_cases'}))

sido_rate = agg_dom.merge(agg_tot, on=['sido_kr','sido_en'], how='outer')
sido_rate['imp_cases'] = (sido_rate['total_cases'] - sido_rate['dom_cases']).clip(lower=0)
sido_rate['dom_rate']  = sido_rate['dom_cases']  / sido_rate['mean_pop'] / N_YR * 1e5
sido_rate['mean_annual_rate'] = sido_rate['total_cases'] / sido_rate['mean_pop'] / N_YR * 1e5
sido_rate['imported_pct'] = (sido_rate['imp_cases'] / sido_rate['total_cases'] * 100)
sido_rate = sido_rate.sort_values('mean_annual_rate', ascending=False).reset_index(drop=True)
sido_rate.to_csv(f'{OUT}/tbl_sido_rate.csv', index=False)
print(f'\ntbl_sido_rate.csv: {len(sido_rate)} rows')
print(sido_rate[['sido_en','total_cases','dom_cases','imp_cases',
                  'mean_pop','mean_annual_rate']].to_string(index=False))
