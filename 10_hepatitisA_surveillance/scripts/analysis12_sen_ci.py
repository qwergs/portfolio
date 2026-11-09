"""Analysis 12 (NEW) — Sen slope confidence intervals for total / domestic /
imported / M:F IRR series (reviewer Q2).

Output: tbl_sen_ci.csv
"""
import numpy as np, pandas as pd
import pymannkendall as mk

OUT = '/sessions/serene-confident-edison/mnt/outputs/hepa_work'
ann = pd.read_csv(f'{OUT}/tbl_annual_rates.csv').sort_values('year').reset_index(drop=True)
mf  = pd.read_csv(f'{OUT}/tbl_mf_yearly.csv').sort_values('year').reset_index(drop=True)

def sen_ci(values, alpha=0.05):
    """Hollander–Wolfe Theil–Sen 95% CI: ranks of pairwise slopes."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    slopes = []
    for i in range(n-1):
        for j in range(i+1, n):
            slopes.append((v[j] - v[i]) / (j - i))
    slopes = np.sort(slopes)
    N = len(slopes)
    # Standard normal-based bounds
    var_s = n*(n-1)*(2*n+5)/18
    z = 1.96
    Calpha = z * np.sqrt(var_s)
    M1 = int(np.floor((N - Calpha) / 2))
    M2 = int(np.ceil((N + Calpha) / 2))
    lo = slopes[max(M1, 0)]
    hi = slopes[min(M2, N-1)]
    med = np.median(slopes)
    return med, lo, hi

rows = []
for series, vec in [
    ('total_log_rate',    np.log(ann['total']/ann['pop_total']*1e5 + 1e-9)),
    ('domestic_log_rate', np.log(ann['domestic']/ann['pop_total']*1e5 + 1e-9)),
    ('imported_log_rate', np.log((ann['imported'].clip(lower=0))/ann['pop_total']*1e5 + 1e-9)),
    ('mf_irr',            mf['IRR_MF'].values),
]:
    res = mk.hamed_rao_modification_test(np.asarray(vec))
    med, lo, hi = sen_ci(np.asarray(vec))
    if 'log_rate' in series:
        # convert to %/year on rate scale
        med_pct = (np.exp(med) - 1) * 100
        lo_pct  = (np.exp(lo)  - 1) * 100
        hi_pct  = (np.exp(hi)  - 1) * 100
    else:
        med_pct = med * 100      # interpreted as approx % change/year for IRR
        lo_pct  = lo * 100
        hi_pct  = hi * 100
    rows.append({
        'series':         series,
        'tau':            res.Tau,
        'p_MK':           res.p,
        'sen_slope':      med,
        'sen_slope_lo':   lo,
        'sen_slope_hi':   hi,
        'sen_pct_year':   med_pct,
        'sen_pct_lo':     lo_pct,
        'sen_pct_hi':     hi_pct,
    })

out = pd.DataFrame(rows)
out.to_csv(f'{OUT}/tbl_sen_ci.csv', index=False)
print(out.to_string(index=False))
