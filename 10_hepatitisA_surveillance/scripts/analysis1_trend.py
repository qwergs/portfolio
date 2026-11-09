"""Analysis 1 — Annual trend: Hamed-Rao modified Mann-Kendall + Sen's slope
+ Negative-binomial GLM (log-pop offset).

Output: tbl_trend.csv with rows for total / domestic / imported.
"""
import numpy as np
import pandas as pd
import pymannkendall as mk
import statsmodels.api as sm

OUT = '/sessions/elegant-quirky-hamilton/mnt/outputs/hepa_work'

ann = pd.read_csv(f'{OUT}/tbl_annual_rates.csv')
ann = ann.sort_values('year').reset_index(drop=True)
log_pop = np.log(ann['pop_total'].values)

rows = []
for series in ['total', 'domestic', 'imported']:
    y = ann[series].astype(int).values
    rate = y / ann['pop_total'].values * 1e5
    log_rate = np.log(rate + 1e-9)

    # Mann-Kendall on log-rate (per Sen on log scale)
    res = mk.hamed_rao_modification_test(log_rate)
    sen_pct = (np.exp(res.slope) - 1) * 100   # %/year on rate scale
    tau = res.Tau
    p_mk = res.p

    # NB GLM (log-pop offset, linear year)
    X = sm.add_constant(np.arange(len(y)))
    try:
        nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0),
                    offset=log_pop).fit(maxiter=200)
        irr_year = float(np.exp(nb.params[1]))
        irr_se   = float(nb.bse[1])
        irr_lo   = float(np.exp(nb.params[1] - 1.96*irr_se))
        irr_hi   = float(np.exp(nb.params[1] + 1.96*irr_se))
        p_nb     = float(nb.pvalues[1])
    except Exception as e:
        irr_year = np.nan; irr_lo = np.nan; irr_hi = np.nan; p_nb = np.nan

    rows.append({
        'series':       series,
        'mean_rate':    rate.mean(),
        'tau':          tau,
        'p_MK':         p_mk,
        'sen_pct_year': sen_pct,
        'IRR_NB_year':  irr_year,
        'IRR_lo':       irr_lo,
        'IRR_hi':       irr_hi,
        'p_NB':         p_nb,
    })

df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/tbl_trend.csv', index=False)
print(df.to_string(index=False))
