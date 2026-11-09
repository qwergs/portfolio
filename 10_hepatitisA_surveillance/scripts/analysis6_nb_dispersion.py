"""Analysis 6 — NB2 dispersion alpha with 95% Wald (log-scale) CI for the
total / domestic / imported annual series.

Output: tbl_nb_dispersion.csv
"""
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.discrete_model import NegativeBinomial

OUT = '/sessions/elegant-quirky-hamilton/mnt/outputs/hepa_work'
df = pd.read_csv(f'{OUT}/tbl_annual_rates.csv').sort_values('year').reset_index(drop=True)
df['yearc'] = df['year'] - df['year'].mean()

rows = []
for series in ['total', 'domestic', 'imported']:
    y = df[series].astype(int).values
    X = sm.add_constant(df[['yearc']])
    try:
        mod = NegativeBinomial(y, X, loglike_method='nb2', exposure=df['pop_total'].values)
        fit = mod.fit(disp=0, maxiter=400)
        a = fit.params['alpha']
        se = fit.bse['alpha']
        log_a = np.log(max(a, 1e-9))
        log_a_se = se / max(a, 1e-9)
        lo = float(np.exp(log_a - 1.96*log_a_se))
        hi = float(np.exp(log_a + 1.96*log_a_se))
    except Exception as e:
        a = np.nan; se = np.nan; lo = np.nan; hi = np.nan
    pois = sm.GLM(y, X, family=sm.families.Poisson(),
                  exposure=df['pop_total'].values).fit()
    pearson = pois.pearson_chi2 / pois.df_resid
    if not np.isnan(a):
        lr = 2 * (fit.llf - pois.llf)
        p_lr = 0.5 * stats.chi2.sf(lr, 1) if lr > 0 else 0.5
    else:
        lr = np.nan; p_lr = np.nan
    rows.append({
        'series': series,
        'n_years': len(y),
        'total_events': int(y.sum()),
        'alpha_hat': a,
        'alpha_lo': lo, 'alpha_hi': hi,
        'pearson_chi2_over_df': pearson,
        'LR_poisson_vs_NB': lr,
        'p_LR_one_sided': p_lr,
    })
res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/tbl_nb_dispersion.csv', index=False)
print(res.to_string(index=False))
