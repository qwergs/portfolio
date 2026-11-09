"""Analysis 4 — BIC-based breakpoint search (NB GLM, log-pop offset).

Continuous piecewise-linear log-rate model:
    log(rate_t) = β0 + β1·t + Σ_b βb·max(0, t - bp_b)
Exhaustive search over 0-3 breakpoints, h_min=3 by default.

Output: tbl_breakpoint.csv
"""
import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import NegativeBinomial as NBfam

OUT = '/sessions/elegant-quirky-hamilton/mnt/outputs/hepa_work'

ann = pd.read_csv(f'{OUT}/tbl_annual_rates.csv').sort_values('year').reset_index(drop=True)
ann['t'] = ann['year'] - ann['year'].min()
ann['logpop'] = np.log(ann['pop_total'])
YEAR_MIN = int(ann['year'].min())
T_MAX = int(ann['t'].max())
H_MIN = 3

def design_matrix(t, bps):
    X = np.column_stack([np.ones_like(t), t])
    for bp in sorted(bps):
        X = np.column_stack([X, np.maximum(0, t - bp)])
    return X

def fit_nb(cases, t, logpop, bps):
    X = design_matrix(t, bps)
    for alpha in [1.0, 0.5, 2.0, 0.1, 5.0]:
        try:
            mod = GLM(cases, X, family=NBfam(alpha=alpha), offset=logpop).fit(maxiter=200, disp=False)
            if mod.converged:
                return mod.bic_llf, mod.llf, X.shape[1]
        except Exception:
            continue
    return np.inf, np.nan, X.shape[1]

def feasible(n_bp, h, t_max):
    cands = list(range(h, t_max - h + 2))
    for combo in combinations(cands, n_bp):
        prev = 0; ok = True
        for b in combo:
            if (b - prev) < h: ok=False; break
            prev = b
        if ok and (t_max - combo[-1] + 1) >= h:
            yield combo

def best_for(cases, t, logpop, n_bp, h):
    if n_bp == 0:
        bic, ll, k = fit_nb(cases, t, logpop, [])
        return bic, (), ll, k
    best = (np.inf, None, np.nan, None)
    for combo in feasible(n_bp, h, T_MAX):
        bic, ll, k = fit_nb(cases, t, logpop, list(combo))
        if bic < best[0]:
            best = (bic, combo, ll, k)
    return best

rows = []
for label, y in [('total', ann.total), ('domestic', ann.domestic), ('imported', ann.imported)]:
    per = []
    for n_bp in [0,1,2,3]:
        b = best_for(y.values, ann.t.values, ann.logpop.values, n_bp, H_MIN)
        if b[1] is not None:
            per.append((n_bp, b[0], b[1], b[2], b[3]))
    per.sort(key=lambda x: x[1])
    bic_0 = next((p[1] for p in per if p[0] == 0), np.inf)
    if not per: continue
    best_n, best_bic, best_combo, best_ll, best_k = per[0]
    runner_n, runner_bic, runner_combo = (None, np.inf, ()) if len(per) < 2 else (per[1][0], per[1][1], per[1][2])
    def desc(n, c):
        if n == 0 or not c: return 'no break'
        return f'{n}bp@' + ','.join(str(YEAR_MIN+i) for i in c)
    rows.append({
        'series': label,
        'best_model': desc(best_n, best_combo),
        'best_bic':   round(best_bic, 2),
        'dBIC_vs_0bp': round(best_bic - bic_0, 2),
        'runner_up':  desc(runner_n, runner_combo),
        'runner_bic': round(runner_bic, 2),
    })

df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/tbl_breakpoint.csv', index=False)
print(df.to_string(index=False))
