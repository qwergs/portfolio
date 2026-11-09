"""Analysis 3 — Sex × age stratification.

Computes:
  - Pooled M:F IRR (exact Poisson) over 2011-2024
  - Annual M:F IRR series + Hamed-Rao MK trend test
  - Age-specific IRRs (reference 0-9세) using sex-pooled person-time
  - 70+ share of cases over time (Hamed-Rao MK)

Outputs:
  tbl_sex_age.csv          — pooled IRR, age-specific IRRs
  tbl_mf_yearly.csv        — year, M cases, F cases, IRR, lo, hi
  tbl_age70_share.csv      — year, all cases, 70+ cases, share, pop 70+
"""
import numpy as np
import pandas as pd
from scipy import stats
import pymannkendall as mk

OUT = '/sessions/elegant-quirky-hamilton/mnt/outputs/hepa_work'

df2 = pd.read_csv(f'{OUT}/df2_long.csv')
pop_path = '/sessions/elegant-quirky-hamilton/mnt/논문_이건상/A형간염/data/population_final_harmonized.csv'
pop = pd.read_csv(pop_path, encoding='utf-8-sig')
pop.columns = pop.columns.str.strip()

# Halve doubled cities to align with data prep
DOUBLED_CITIES = {'수원시':'경기도','성남시':'경기도','안양시':'경기도',
                  '안산시':'경기도','고양시':'경기도','용인시':'경기도',
                  '창원시':'경상남도','청주시':'충청북도','천안시':'충청남도',
                  '전주시':'전라북도','포항시':'경상북도'}
mask = pop.apply(lambda r: DOUBLED_CITIES.get(r['sigungu']) == r['sido'], axis=1)
pop.loc[mask, 'population'] = pop.loc[mask, 'population'] / 2

# Map age groups to canonical 8-bin scheme
AGE_TO_BIN = {
    '0~9세':'0-9', '10~19세':'10-19', '20~29세':'20-29',
    '30~39세':'30-39', '40~49세':'40-49', '50~59세':'50-59',
    '60~69세':'60-69', '70세 이상':'70+',
}
df2 = df2[df2.age_group.isin(AGE_TO_BIN.keys())].copy()
df2['agebin'] = df2.age_group.map(AGE_TO_BIN)

# Build population × sex × agebin for each year (national, sex 남/여)
def pop_to_bin(ag):
    """Map KOSIS 5-yr to 10-yr bins."""
    if ag in ('0-4세','5-9세'): return '0-9'
    if ag in ('10-14세','15-19세'): return '10-19'
    if ag in ('20-24세','25-29세'): return '20-29'
    if ag in ('30-34세','35-39세'): return '30-39'
    if ag in ('40-44세','45-49세'): return '40-49'
    if ag in ('50-54세','55-59세'): return '50-59'
    if ag in ('60-64세','65-69세'): return '60-69'
    if ag in ('70-74세','75-79세','80-84세','80세 이상','85-89세','85세 이상',
              '90-94세','95-99세','95세 이상','100세 이상'): return '70+'
    return None

pop = pop[(pop.sex.isin(['남','여'])) & (pop.year.between(2011, 2024))].copy()
pop['agebin'] = pop['age_group'].map(pop_to_bin)
pop_sex_age = pop.dropna(subset=['agebin']).groupby(['year','sex','agebin'])['population'].sum().reset_index()

# ─── 1. Pooled M:F IRR (sex-only, all ages) ────────────────────────────
sex_yr = df2[df2.sex.isin(['남','여'])].groupby(['year','sex'])['cases'].sum().reset_index()
sex_pop_yr = pop_sex_age.groupby(['year','sex'])['population'].sum().reset_index()
sex = sex_yr.merge(sex_pop_yr, on=['year','sex'])
m_total = sex[sex.sex=='남']['cases'].sum()
f_total = sex[sex.sex=='여']['cases'].sum()
m_pop   = sex[sex.sex=='남']['population'].sum()
f_pop   = sex[sex.sex=='여']['population'].sum()
mf_irr = (m_total/m_pop) / (f_total/f_pop)
# 95% CI via exact Poisson on M cases conditional on M+F
ratio = m_pop / f_pop
total_mf = m_total + f_total
# binomial exact CI
lo, hi = stats.beta.ppf([0.025, 0.975], m_total, f_total + 1)
# convert proportion p to IRR: IRR = (p/(1-p)) * (f_pop/m_pop)
def prop_to_irr(p):
    return (p / (1-p)) * (f_pop / m_pop)
mf_irr_lo = prop_to_irr(lo)
mf_irr_hi = prop_to_irr(hi)
print(f'Pooled M:F IRR = {mf_irr:.3f} (95% CI {mf_irr_lo:.3f}-{mf_irr_hi:.3f})')
print(f'  M = {m_total:,}, F = {f_total:,}; M-pop = {m_pop:,.0f}, F-pop = {f_pop:,.0f}')

# ─── 2. Annual M:F IRR series + MK trend ────────────────────────────────
yearly = []
for yr in sorted(sex.year.unique()):
    sub = sex[sex.year==yr]
    if len(sub) != 2: continue
    m = sub[sub.sex=='남'].iloc[0]
    f = sub[sub.sex=='여'].iloc[0]
    if f.cases < 5 or m.cases < 5:
        irr = np.nan; lo = np.nan; hi = np.nan
    else:
        irr = (m.cases/m.population) / (f.cases/f.population)
        # exact CI using binomial p
        plo, phi = stats.beta.ppf([0.025, 0.975], m.cases, f.cases + 1)
        lo = (plo/(1-plo)) * (f.population/m.population)
        hi = (phi/(1-phi)) * (f.population/m.population)
    yearly.append({'year':yr, 'M':m.cases, 'F':f.cases, 'IRR_MF':irr, 'IRR_lo':lo, 'IRR_hi':hi})
mf = pd.DataFrame(yearly)
mf.to_csv(f'{OUT}/tbl_mf_yearly.csv', index=False)
print('\nAnnual M:F IRR:')
print(mf.to_string(index=False))

# Hamed-Rao MK on annual M:F IRR
valid = mf.dropna()
res = mk.hamed_rao_modification_test(valid.IRR_MF.values)
print(f'\nMK on annual M:F IRR: tau = {res.Tau:.3f}, p = {res.p:.4f}')

# ─── 3. Age-specific IRRs (sex-pooled, reference 0-9) ─────────────────
age_cases = df2[df2.sex.isin(['남','여'])].groupby('agebin')['cases'].sum()
age_pop   = pop_sex_age.groupby('agebin')['population'].sum()  # person-years (sum across years)
age_rate  = age_cases / age_pop * 1e5  # per 100k person-years
print('\nAge-stratified rates (per 100k):')
for ab in ['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70+']:
    print(f'  {ab}: cases={age_cases[ab]:,}, pop-years={age_pop[ab]:,.0f}, rate={age_rate[ab]:.2f}')

ref_rate = age_rate['0-9']
age_rows = []
for ab in ['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70+']:
    n_a = int(age_cases[ab]); n_r = int(age_cases['0-9'])
    irr = age_rate[ab] / ref_rate
    # CI via Poisson exact ratio
    if n_r > 0 and n_a > 0:
        plo, phi = stats.beta.ppf([0.025,0.975], n_a, n_r+1)
        ratio_pop = age_pop['0-9'] / age_pop[ab]
        lo = (plo/(1-plo)) * ratio_pop
        hi = (phi/(1-phi)) * ratio_pop
    else:
        lo = np.nan; hi = np.nan
    age_rows.append({'agebin':ab, 'cases':n_a, 'pop_years':float(age_pop[ab]),
                     'rate_per_100k':float(age_rate[ab]),
                     'IRR_vs_0_9':float(irr), 'IRR_lo':lo, 'IRR_hi':hi})

# ─── 4. 70+ share trend ────────────────────────────────────────────────
total_yr = df2[df2.sex.isin(['남','여'])].groupby('year')['cases'].sum()
age70_yr = df2[(df2.sex.isin(['남','여'])) & (df2.agebin=='70+')].groupby('year')['cases'].sum()
share = (age70_yr / total_yr * 100).reset_index().rename(columns={'cases':'share_70plus_pct'})
pop70_yr = pop_sex_age[pop_sex_age.agebin=='70+'].groupby('year')['population'].sum()
pop_total = pop_sex_age.groupby('year')['population'].sum()
share['pop70_pct'] = ((pop70_yr / pop_total) * 100).values
share.to_csv(f'{OUT}/tbl_age70_share.csv', index=False)
print('\n70+ share of cases vs 70+ pop share:')
print(share.to_string(index=False))
res2 = mk.hamed_rao_modification_test(share.share_70plus_pct.values)
print(f'\nMK on 70+ case share: tau = {res2.Tau:.3f}, p = {res2.p:.4f}')

# Save summary table
sum_table = pd.DataFrame({
    'metric': ['Pooled M:F IRR (95% CI)',
               'MK tau on annual M:F IRR',
               'MK p on annual M:F IRR',
               'MK tau on 70+ share',
               'MK p on 70+ share',
               'M cases', 'F cases', 'M person-years', 'F person-years'],
    'value':  [f'{mf_irr:.3f} ({mf_irr_lo:.3f}-{mf_irr_hi:.3f})',
               f'{res.Tau:.3f}', f'{res.p:.4f}',
               f'{res2.Tau:.3f}', f'{res2.p:.4f}',
               f'{m_total:,}', f'{f_total:,}',
               f'{m_pop:,.0f}', f'{f_pop:,.0f}'],
})
sum_table.to_csv(f'{OUT}/tbl_sex_summary.csv', index=False)
pd.DataFrame(age_rows).to_csv(f'{OUT}/tbl_sex_age.csv', index=False)
print('\nSaved tbl_sex_age.csv, tbl_mf_yearly.csv, tbl_age70_share.csv, tbl_sex_summary.csv')
