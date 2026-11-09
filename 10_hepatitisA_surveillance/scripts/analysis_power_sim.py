"""Power simulation: detect a single NB breakpoint at year 12 in a 14-yr series.

For each per-year additional rate change ∈ {0, 1, 5, 15, 35, 65, 120} %,
generate 200 NB-distributed series with α=2 dispersion, fit 0-bp vs 1-bp NB
GLM (continuous piecewise-linear, h_min = 3), and count fraction with ΔBIC<-6.

Output: tbl_power_sim.csv (effect_pct, detection_prob, n_sim)
"""
import numpy as np, pandas as pd
from itertools import combinations
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import NegativeBinomial as NBfam

OUT = '/sessions/elegant-quirky-hamilton/mnt/outputs/hepa_work'
ann = pd.read_csv(f'{OUT}/tbl_annual_rates.csv').sort_values('year').reset_index(drop=True)
N = len(ann)
T = np.arange(N)
LOGPOP = np.log(ann['pop_total'].values)
H_MIN = 3
ALPHA_NB = 2.0       # dispersion of simulated noise
N_SIM = 200
BREAK_T = 8          # breakpoint year index (year 8 ≈ 2019)
BASELINE_RATE = ann['total'].sum() / ann['pop_total'].sum() * 1e5

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
            if m.converged:
                return m.bic_llf
        except Exception:
            continue
    return np.inf

def best_1bp(cases, t, lp):
    best = np.inf
    for bp in range(H_MIN, N - H_MIN + 1):
        b = fit_nb(cases, t, lp, [bp])
        if b < best: best = b
    return best

def simulate_one(rng, slope_pct):
    """Generate one NB series with breakpoint slope change at BREAK_T."""
    log_rate0 = np.log(BASELINE_RATE / 1e5)
    log_rate = log_rate0 + np.zeros(N)
    # post-break additional slope of slope_pct % per year
    slope = np.log(1 + slope_pct/100)
    for i in range(BREAK_T, N):
        log_rate[i] = log_rate0 + slope * (i - BREAK_T)
    mu = np.exp(log_rate + LOGPOP)
    # NB2 sample: mean=mu, var=mu+α·mu²; equivalent to Gamma-Poisson mixture
    p = 1 / (1 + ALPHA_NB * mu)
    n = 1 / ALPHA_NB
    cases = rng.negative_binomial(n, p).astype(int)
    return cases

results = []
rng = np.random.default_rng(42)
for pct in [0, 1, 5, 15, 35, 65, 120]:
    detected = 0
    for s in range(N_SIM):
        cases = simulate_one(rng, pct)
        bic_0 = fit_nb(cases, T, LOGPOP, [])
        bic_1 = best_1bp(cases, T, LOGPOP)
        if (bic_1 - bic_0) < -6:
            detected += 1
    results.append({'effect_pct': pct, 'detection_prob': detected/N_SIM, 'n_sim': N_SIM})
    print(f'  effect = +{pct:3d}%/yr  →  power = {detected/N_SIM:.3f}')

df = pd.DataFrame(results)
df.to_csv(f'{OUT}/tbl_power_sim.csv', index=False)
print(f'\nSaved tbl_power_sim.csv')
