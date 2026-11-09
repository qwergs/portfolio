"""Table 1, Supplementary Tables S3, S4, S5 and S8, and the 2010 denominator check.

Run: python scripts/01_trend.py
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from common import DATA, OUT, load_annual, nb_trend, fmt_ci

a = load_annual()
pop = a.population

# Table 1
t1 = a[['ehec_total', 'ehec_domestic', 'ehec_imported', 'population']].copy()
t1['rate_per_100000'] = (a.ehec_total / pop * 1e5).round(3)
t1.to_csv(os.path.join(OUT, 'table1.csv'))
print('Table 1 written; 2001 rate', t1.rate_per_100000.iloc[0], '2024 rate', t1.rate_per_100000.iloc[-1])

# Supplementary Table S3: dispersion and overdispersion tests
rows = []
X = sm.add_constant(np.arange(len(a), dtype=float))
for label, col in [('total', 'ehec_total'), ('domestic', 'ehec_domestic'), ('imported', 'ehec_imported')]:
    y = a[col].values.astype(float)
    nb = sm.NegativeBinomial(y, X, offset=np.log(pop.values)).fit(disp=0, maxiter=500)
    po = sm.GLM(y, X, family=sm.families.Poisson(), offset=np.log(pop.values)).fit()
    la, se_la = np.log(nb.params[-1]), nb.bse[-1] / nb.params[-1]
    lr = 2 * (nb.llf - po.llf)
    rows.append({'series': label, 'events': int(y.sum()), 'alpha': round(nb.params[-1], 3),
                 'alpha_lo': round(np.exp(la - 1.96 * se_la), 3), 'alpha_hi': round(np.exp(la + 1.96 * se_la), 3),
                 'pearson_chi2_df_poisson': round(po.pearson_chi2 / po.df_resid, 2),
                 'LR_vs_poisson': round(lr, 1), 'p_one_sided': 0.5 * stats.chi2.sf(lr, 1)})
s3 = pd.DataFrame(rows)
s3.to_csv(os.path.join(OUT, 'tableS3_dispersion.csv'), index=False)
print(s3.to_string(index=False))

# Supplementary Table S5: negative controls
s5 = []
for label, col in [('EHEC', 'ehec_total'), ('Shigellosis', 'shigellosis'), ('Typhoid fever', 'typhoid_fever')]:
    r = nb_trend(a[col], pop)
    s5.append({'infection': label, 'cases_2001': int(a[col].iloc[0]), 'cases_2024': int(a[col].iloc[-1]),
               'total': int(a[col].sum()), 'change_per_year': fmt_ci(r)})
s5 = pd.DataFrame(s5)
s5.to_csv(os.path.join(OUT, 'tableS5_negative_controls.csv'), index=False)
print(s5.to_string(index=False))

# Supplementary Table S8: sensitivity analyses
def sub(lo=2001, hi=2024, drop=None, col='ehec_total'):
    d = a.loc[lo:hi]
    if drop:
        d = d.drop(index=drop)
    r = nb_trend(d[col], d.population)
    return len(d), int(d[col].sum()), fmt_ci(r)

s8 = []
for label, args in [('Full series, 2001-2024 (primary)', {}),
                    ('Modern diagnostic era, 2010-2024', {'lo': 2010}),
                    ('Excluding the 2020 outbreak year', {'drop': 2020}),
                    ('Domestic cases only, 2001-2024', {'col': 'ehec_domestic'}),
                    ('2005-2024', {'lo': 2005}),
                    ('2007-2024 (post hoc)', {'lo': 2007})]:
    n, c, ci = sub(**args)
    s8.append({'analysis': label, 'n_years': n, 'cases': c, 'change_per_year': ci})
s8 = pd.DataFrame(s8)
s8.to_csv(os.path.join(OUT, 'tableS8_sensitivity.csv'), index=False)
print(s8.to_string(index=False))

# Supplementary Table S4: direct age standardisation (2005 Korean population weights)
age = pd.read_csv(os.path.join(DATA, 'ehec_annual_by_age.csv'))
c = age.pivot(index='year', columns='age_band', values='cases')
p = age.pivot(index='year', columns='age_band', values='population')
w = p.loc[2005] / p.loc[2005].sum()
asr = (c / p * 1e5 * w).sum(axis=1)
se = np.sqrt((w ** 2 * c / p ** 2 * 1e10).sum(axis=1))
s4 = pd.DataFrame({'asr': asr.round(3), 'lo': (asr - 1.96 * se).round(3), 'hi': (asr + 1.96 * se).round(3)})
s4.to_csv(os.path.join(OUT, 'tableS4_age_standardised.csv'))
print('S4: ASR 2001', s4.asr.iloc[0], '2024', s4.asr.iloc[-1])

# Denominator check: one-off step in the 2010 resident-registration total
growth = (pop.pct_change() * 100).round(2)
adj = pop.copy().astype(float)
adj.loc[:2009] *= 1 + 466205 / pop.loc[2009]
r0, r1 = nb_trend(a.ehec_total, pop), nb_trend(a.ehec_total, adj)
print(f"2009-2010 growth {growth.loc[2010]}%; trend {r0['pct']:.2f} -> {r1['pct']:.2f} %/yr after adjustment")
