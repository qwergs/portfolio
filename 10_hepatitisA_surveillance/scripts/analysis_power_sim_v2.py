"""Analysis power_sim (revised) — Breakpoint detection power at empirical alpha.

Reviewer Q10: redo with alpha ≈ 0.68 (empirical) instead of alpha = 2.
Add step-change scenarios (mimicking 2019 jolt).

Output: tbl_power_sim_v2.csv (continuous slopes); tbl_power_sim_step.csv (step changes)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import product

OUT = '/sessions/serene-confident-edison/mnt/outputs/hepa_work'
RNG = np.random.default_rng(42)

# --- Calibration --------------------------------------------------------
ann = pd.read_csv(f'{OUT}/tbl_annual_rates.csv').sort_values('year').reset_index(drop=True)
mu_baseline = ann['total'].mean()       # ≈ 3,903 cases/year
ALPHA_EMPIRICAL = 0.68                  # from analysis6_nb_dispersion.py
N = 14                                  # 14-year series
N_SIM = 1000                            # reviewer Q6: ≥1,000 sims
DBIC_THRESHOLD = -6                     # detection rule

def nb_sample(rng, mu, alpha, size):
    """NB2 sample with mean mu, dispersion alpha (var = mu + alpha*mu^2)."""
    p = 1 / (1 + alpha * mu)
    n_param = mu * p / (1 - p)
    return rng.negative_binomial(n_param, p, size=size)

def fit_bp(y, n_bp, hmin=3):
    """Best BIC over piecewise-linear NB GLMs with n_bp breakpoints."""
    n = len(y)
    t = np.arange(n)
    X0 = sm.add_constant(t)
    if n_bp == 0:
        try:
            mod = sm.GLM(y, X0, family=sm.families.NegativeBinomial(alpha=1.0)).fit(maxiter=200)
            return mod.bic_llf
        except Exception:
            return np.inf
    candidates = list(range(hmin, n - hmin + 1))
    if n_bp == 1:
        best = np.inf
        for bp in candidates:
            X = np.column_stack([X0, np.maximum(0, t - bp)])
            try:
                mod = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0)).fit(maxiter=200)
                if mod.bic_llf < best:
                    best = mod.bic_llf
            except Exception:
                continue
        return best
    return np.inf

def run_sim(scenarios, simulator_fn, label):
    """Generic simulator: simulator_fn(rng, scenario, n) -> y vector."""
    out = []
    for sc in scenarios:
        n_detect = 0
        for _ in range(N_SIM):
            y = simulator_fn(RNG, sc, N)
            bic0 = fit_bp(y, 0)
            bic1 = fit_bp(y, 1)
            dbic = bic1 - bic0
            if dbic < DBIC_THRESHOLD:
                n_detect += 1
        det_prob = n_detect / N_SIM
        out.append({**{label: sc}, 'detection_prob': det_prob, 'n_sim': N_SIM,
                    'alpha': ALPHA_EMPIRICAL, 'dBIC_threshold': DBIC_THRESHOLD})
        print(f"  {label}={sc}: detection prob = {det_prob:.3f}")
    return pd.DataFrame(out)

# --- (a) Continuous slope scenarios -------------------------------------
# Effect = % change per year after midpoint (centered at t=7)
def slope_simulator(rng, eff_pct, n):
    bp = n // 2
    t = np.arange(n)
    eff = np.where(t < bp, 0, (t - bp) * eff_pct / 100)
    mu = mu_baseline * (1 + eff)
    mu = np.maximum(mu, 1)
    return nb_sample(rng, mu, ALPHA_EMPIRICAL, n)

print(f"\n=== Continuous-slope power ({N_SIM} sims, alpha={ALPHA_EMPIRICAL}) ===")
slope_eff = [0, 5, 15, 35, 65, 120, 200, 300]
df_slope = run_sim(slope_eff, slope_simulator, 'effect_pct_per_year')
df_slope.to_csv(f'{OUT}/tbl_power_sim_v2.csv', index=False)

# --- (b) Step-change scenarios (mimicking 2019 jolt) --------------------
# Single one-year multiplier at year 9 (≈2019 in 2011-2024) then return to baseline
def step_simulator(rng, multiplier, n):
    mu = np.full(n, mu_baseline)
    bp = 9                                       # year index of step
    mu[bp] = mu[bp] * multiplier
    mu = np.maximum(mu, 1)
    return nb_sample(rng, mu, ALPHA_EMPIRICAL, n)

print(f"\n=== Step-change power ({N_SIM} sims) ===")
step_mults = [1.0, 1.5, 2.0, 3.0, 5.0, 7.0]
df_step = run_sim(step_mults, step_simulator, 'step_multiplier')
df_step.to_csv(f'{OUT}/tbl_power_sim_step.csv', index=False)

print('\nSaved tbl_power_sim_v2.csv and tbl_power_sim_step.csv')
print('\n--- Slope power ---')
print(df_slope.to_string(index=False))
print('\n--- Step power ---')
print(df_step.to_string(index=False))
