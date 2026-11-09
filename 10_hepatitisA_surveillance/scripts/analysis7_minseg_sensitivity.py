"""Analysis 7 — Minimum-segment-length sensitivity for breakpoint search.

Same continuous piecewise-linear NB GLM spec as analysis4_breakpoint.py.
Only h_min ∈ {2, 3, 4} varies.

Output: tbl_minseg_sensitivity.csv
"""
import numpy as np, pandas as pd
from itertools import combinations
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import NegativeBinomial as NBfam

OUT = '/sessions/elegant-quirky-hamilton/mnt/outputs/hepa_work'
ann = pd.read_csv(f'{OUT}/tbl_annual_rates.csv').sort_values('year').reset_index(drop=True)
ann['t'] = ann['year'] - ann['year'].min()
ann['logpop'] = np.log(ann['pop_total'])
YEAR_MIN = int(ann['year'].min())
T_MAX = int(ann['t'].max())

def design(t, bps):
    X = np.column_stack([np.ones_like(t), t])
    for bp in sorted(bps):
        X = np.column_stack([X, np.maximum(0, t - bp)])
    return X

def fit_nb(cases, t, lp, bps):
    X = design(t, bps)
    for a in [1.0, 0.5, 2.0, 0.1, 5.0]:
        try:
            m = GLM(cases, X, family=NBfam(alpha=a), offset=lp).fit(maxiter=200, disp=False)
            if m.converged: return m.bic_llf, m.llf, X.shape[1]
        except Exception: continue
    return np.inf, np.nan, X.shape[1]

def feasible(n_bp, h, t_max):
    cands = list(range(h, t_max - h + 2))
    for c in combinations(cands, n_bp):
        prev = 0; ok = True
        for b in c:
            if b - prev < h: ok = False; break
            prev = b
        if ok and (t_max - c[-1] + 1) >= h:
            yield c

def best_for(cases, t, lp, n_bp, h):
    if n_bp == 0:
        bic, ll, k = fit_nb(cases, t, lp, [])
        return bic, (), ll, k
    best = (np.inf, None, np.nan, None)
    for c in feasible(n_bp, h, T_MAX):
        bic, ll, k = fit_nb(cases, t, lp, list(c))
        if bic < best[0]:
            best = (bic, c, ll, k)
    return best

rows = []
for label, y in [('total', ann.total), ('domestic', ann.domestic), ('imported', ann.imported)]:
    for h in [2, 3, 4]:
        bic_0, _, _, _ = best_for(y.values, ann.t.values, ann.logpop.values, 0, h)
        per = [(0, bic_0, ())]
        for nbp in [1, 2, 3]:
            b = best_for(y.values, ann.t.values, ann.logpop.values, nbp, h)
            if b[1] is not None:
                per.append((nbp, b[0], b[1]))
        per.sort(key=lambda x: x[1])
        best_n, best_bic, best_combo = per[0]
        runner_n, runner_bic, runner_combo = per[1] if len(per) > 1 else (None, np.inf, ())
        def desc(n, c):
            if n == 0 or not c: return 'no break'
            return f'{n}bp@' + ','.join(str(YEAR_MIN+i) for i in c)
        rows.append({
            'series': label,
            'h_min': h,
            'best_model': desc(best_n, best_combo),
            'best_bic':   round(best_bic, 2),
            'best_n_bp':  best_n,
            'dBIC_vs_0bp': round(best_bic - bic_0, 2),
            'runner_up':  desc(runner_n, runner_combo),
            'runner_bic': round(runner_bic, 2),
            'delta_bic_to_runner': round(runner_bic - best_bic, 2),
        })

df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/tbl_minseg_sensitivity.csv', index=False)
print(df.to_string(index=False))
