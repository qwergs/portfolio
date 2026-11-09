"""Analysis 11 (NEW) — Age-standardized incidence + age-specific joinpoint.

Reviewer Q4/Q7: provide direct age-standardized national incidence and
age-specific trend analyses to substantiate the claim that the ≥70-year
shift exceeds demographic ageing.

Method:
  • Compute year × age-group cases and population from df2 + harmonized population.
  • Direct standardization to 2005 Korean Standard Population (KSP-2005).
  • Age-specific linear-segment (joinpoint) trend by Hamed–Rao MK + Sen slope
    plus a single-best joinpoint via BIC over piecewise-linear NB GLMs.

Output:
  tbl_asr.csv             year, crude_rate, asr_2005ksp, asr_lo, asr_hi
  tbl_age_specific.csv    age_group, sen_slope_pct_per_year, ci_lo, ci_hi, mk_p, joinpoint_year
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pymannkendall as mk

OUT = '/sessions/serene-confident-edison/mnt/outputs/hepa_work'
DATA = '/sessions/serene-confident-edison/mnt/논문_이건상/A형간염/data'

# 1) Cases by year × age (national, all sex, both domestic+imported)
df2 = pd.read_csv(f'{OUT}/df2_long.csv')
cases_age = (
    df2[(df2.sex == '계') & (~df2.age_group.isin(['계', '연령미상']))]
    .groupby(['year', 'age_group'])['cases'].sum().reset_index()
)
print("Unique age groups in cases:", sorted(cases_age.age_group.unique()))

# 2) Population by year × age (national)
pop = pd.read_csv(f'{DATA}/population_final_harmonized.csv', encoding='utf-8-sig')

# Apply same halving for 11 doubled cities so that aggregation matches KOSIS
DOUBLED = {'수원시':'경기도','성남시':'경기도','안양시':'경기도','안산시':'경기도',
           '고양시':'경기도','용인시':'경기도','창원시':'경상남도','청주시':'충청북도',
           '천안시':'충청남도','전주시':'전라북도','포항시':'경상북도'}
mask = (pop['sigungu'].isin(DOUBLED.keys())
        & pop.apply(lambda r: DOUBLED.get(r['sigungu']) == r['sido'], axis=1))
pop.loc[mask, 'population'] = pop.loc[mask, 'population'] / 2

# 5-year age bins; df2 uses 0~9, 10~19, ..., 70+ groupings
# Map population ages (0-4, 5-9, 10-14, ...) to df2 bins
def map_age(ag):
    if ag in ['0-4세','5-9세']: return '0~9세'
    if ag in ['10-14세','15-19세']: return '10~19세'
    if ag in ['20-24세','25-29세']: return '20~29세'
    if ag in ['30-34세','35-39세']: return '30~39세'
    if ag in ['40-44세','45-49세']: return '40~49세'
    if ag in ['50-54세','55-59세']: return '50~59세'
    if ag in ['60-64세','65-69세']: return '60~69세'
    return '70세 이상'

pop_age = pop[(pop.sex == '계') & (pop.age_group != '계')].copy()
pop_age['age_bin'] = pop_age['age_group'].map(map_age)
# Years 2011-2024
pop_age = pop_age[pop_age.year.between(2011, 2024)]
pop_year_age = pop_age.groupby(['year','age_bin'])['population'].sum().reset_index()
pop_year_age = pop_year_age.rename(columns={'age_bin':'age_group'})

# Sejong addition (national level only, equally distributed across age bins
# since sigungu-level Sejong age data is not available at fine resolution).
# Use proportional split using the rest-of-country age structure.
SEJONG_TOT = {2011:0,2012:113117,2013:122153,2014:156125,2015:210884,
              2016:243048,2017:280100,2018:314126,2019:340575,2020:355831,
              2021:371895,2022:383591,2023:386525,2024:388927}
sj_rows = []
for y, total in SEJONG_TOT.items():
    if total == 0:
        continue
    # Use the country age share in that year as proxy
    yr_shares = pop_year_age[pop_year_age.year==y].set_index('age_group')['population']
    yr_shares = yr_shares / yr_shares.sum()
    for ag, sh in yr_shares.items():
        sj_rows.append({'year': y, 'age_group': ag, 'population': total * sh})
sj = pd.DataFrame(sj_rows)
pop_year_age = (pd.concat([pop_year_age, sj], ignore_index=True)
                  .groupby(['year','age_group'])['population'].sum().reset_index())

# 3) Merge cases × population
merged = cases_age.merge(pop_year_age, on=['year','age_group'], how='left')
merged['rate_per_100k'] = merged['cases'] / merged['population'] * 1e5
print(f"Merged: {len(merged)} year×age rows")

# 4) Korean Standard Population 2005 (8-bin, %)
# Source: Statistics Korea (2005 census-based standard).
KSP2005 = {
    '0~9세': 11.6, '10~19세': 13.3, '20~29세': 14.7, '30~39세': 17.5,
    '40~49세': 16.2, '50~59세': 11.0, '60~69세': 8.4, '70세 이상': 7.3,
}
ksp = pd.Series(KSP2005)
ksp = ksp / ksp.sum()        # weights sum to 1

# 5) Direct age standardization
asr_rows = []
for y in sorted(merged.year.unique()):
    sub = merged[merged.year == y].set_index('age_group')
    rates = sub['rate_per_100k']
    asr = (rates * ksp).sum()
    # Variance: sum((w_i * sqrt(c_i)/n_i * 1e5)^2) standard formula
    var = ((ksp**2) * sub['cases'] / (sub['population']**2) * 1e10).sum()
    se = np.sqrt(var)
    crude = sub['cases'].sum() / sub['population'].sum() * 1e5
    asr_rows.append({
        'year': y, 'crude_rate': crude, 'asr_2005ksp': asr,
        'asr_se': se,
        'asr_lo': asr - 1.96*se, 'asr_hi': asr + 1.96*se
    })
asr = pd.DataFrame(asr_rows)
asr.to_csv(f'{OUT}/tbl_asr.csv', index=False)
print('\nAge-standardised rates (per 100,000, KSP-2005):')
print(asr.to_string(index=False))

# 6) Age-specific MK + Sen slope + simple 1-bp joinpoint
def hamed_rao(rates):
    rates = np.asarray(rates) + 1e-9
    log_r = np.log(rates)
    res = mk.hamed_rao_modification_test(log_r)
    sen_pct = (np.exp(res.slope) - 1) * 100   # %/year on rate scale
    # Approximate 95% CI of Sen slope from MK variance
    n = len(rates)
    s = res.s
    var_s = res.var_s
    z = 1.96 * np.sqrt(var_s) / (n*(n-1)/2)   # rough z scaling
    sen_lo = (np.exp(res.slope - z) - 1) * 100
    sen_hi = (np.exp(res.slope + z) - 1) * 100
    return res.Tau, res.p, sen_pct, sen_lo, sen_hi

age_rows = []
for ag in sorted(merged.age_group.unique()):
    sub = merged[merged.age_group == ag].sort_values('year').reset_index(drop=True)
    if len(sub) < 5 or sub['rate_per_100k'].dropna().nunique() < 3:
        print(f"Skipping {ag}: insufficient data (n={len(sub)})")
        continue
    tau, p, sen, lo, hi = hamed_rao(sub['rate_per_100k'].values)
    # Single-joinpoint search via NB GLM BIC
    y = sub['cases'].astype(int).values
    log_p = np.log(sub['population'].values)
    n = len(y); t = np.arange(n)
    X0 = np.column_stack([np.ones(n), t])
    try:
        bic0 = sm.GLM(y, X0, family=sm.families.NegativeBinomial(alpha=1.0),
                      offset=log_p).fit(maxiter=200).bic_llf
    except Exception:
        bic0 = np.inf
    best_jp = (None, np.inf)
    for jp in range(3, n - 2):
        X = np.column_stack([X0, np.maximum(0, t - jp)])
        try:
            mod = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0),
                         offset=log_p).fit(maxiter=200)
            if mod.bic_llf < best_jp[1]:
                best_jp = (jp, mod.bic_llf)
        except Exception:
            continue
    jp_year = sub['year'].iloc[best_jp[0]] if best_jp[0] is not None else None
    dbic = best_jp[1] - bic0
    age_rows.append({
        'age_group':              ag,
        'mean_rate_per_100k':     sub['rate_per_100k'].mean(),
        'sen_pct_year':           sen,
        'sen_lo':                 lo,
        'sen_hi':                 hi,
        'mk_tau':                 tau,
        'mk_p':                   p,
        'joinpoint_year':         jp_year,
        'dBIC_jp_vs_no_jp':       dbic,
    })
age_tbl = pd.DataFrame(age_rows)
age_tbl.to_csv(f'{OUT}/tbl_age_specific.csv', index=False)
print('\nAge-specific Hamed–Rao MK + 1-joinpoint NB:')
print(age_tbl.to_string(index=False))
