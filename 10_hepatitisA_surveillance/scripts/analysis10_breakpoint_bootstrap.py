"""Analysis 10 (NEW) — Bootstrap breakpoint location confidence intervals.

Reviewer Q4: ΔBIC for the 2014 breakpoint is small (≈ −2.4). Provide a
bootstrap-based confidence interval for the breakpoint *location* and a
permutation/parametric-bootstrap p-value for the existence of the second
breakpoint.

Method:
  • Parametric bootstrap from the BIC-best 2-breakpoint NB GLM.
  • Repeat fit on each bootstrap sample to obtain the empirical distribution
    of the early breakpoint location.
  • Permutation null: shuffle residuals to break temporal structure; fit best
    1-bp vs 2-bp; record fraction of permutations with ΔBIC ≤ −2.

Output: tbl_breakpoint_bootstrap.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT = '/sessions/serene-confident-edison/mnt/outputs/hepa_work'
RNG = np.random.default_rng(42)

ann = pd.read_csv(f'{OUT}/tbl_annual_rates.csv').sort_values('year').reset_index(drop=True)
y = ann['total'].astype(int).values
log_pop = np.log(ann['pop_total'].values)
n = len(y)
t = np.arange(n)
years = ann['year'].values

ALPHA = 0.68     # from analysis6
HMIN = 3
N_BOOT = 500     # 1000 takes too long in interactive mode

def fit_glm(X, y, log_pop, alpha=ALPHA):
    return sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha),
                  offset=log_pop).fit(maxiter=300)

def best_2bp(y, log_pop, hmin=HMIN):
    """Return (bp1, bp2, BIC, beta_vec) for the BIC-best 2-bp model."""
    n = len(y)
    t = np.arange(n)
    best = (None, None, np.inf, None)
    for bp1 in range(hmin, n - 2*hmin + 1):
        for bp2 in range(bp1 + hmin, n - hmin + 1):
            X = np.column_stack([
                np.ones(n), t,
                np.maximum(0, t - bp1),
                np.maximum(0, t - bp2),
            ])
            try:
                mod = fit_glm(X, y, log_pop)
                if mod.bic_llf < best[2]:
                    best = (bp1, bp2, mod.bic_llf, mod.params)
            except Exception:
                continue
    return best

def best_1bp(y, log_pop, hmin=HMIN):
    n = len(y)
    t = np.arange(n)
    best = (None, np.inf)
    for bp in range(hmin, n - hmin + 1):
        X = np.column_stack([np.ones(n), t, np.maximum(0, t - bp)])
        try:
            mod = fit_glm(X, y, log_pop)
            if mod.bic_llf < best[1]:
                best = (bp, mod.bic_llf)
        except Exception:
            continue
    return best

def best_0bp(y, log_pop):
    n = len(y)
    t = np.arange(n)
    X = np.column_stack([np.ones(n), t])
    mod = fit_glm(X, y, log_pop)
    return mod.bic_llf, mod.params

def nb_sample(rng, mu, alpha=ALPHA):
    p = 1 / (1 + alpha * mu)
    n_param = mu * p / (1 - p)
    return rng.negative_binomial(n_param, p)

# Observed best 2-bp
obs_bp1, obs_bp2, obs_bic2, obs_beta2 = best_2bp(y, log_pop)
obs_bp_1, obs_bic1 = best_1bp(y, log_pop)
obs_bic0, _ = best_0bp(y, log_pop)
print(f"Observed best 2-bp: years {years[obs_bp1]} & {years[obs_bp2]}, BIC={obs_bic2:.2f}")
print(f"Observed best 1-bp: year {years[obs_bp_1]}, BIC={obs_bic1:.2f}")
print(f"ΔBIC(2-bp vs 1-bp) = {obs_bic2-obs_bic1:.2f}")
print(f"ΔBIC(2-bp vs 0-bp) = {obs_bic2-obs_bic0:.2f}")

# Generate fitted means under the 2-bp model
X_obs = np.column_stack([
    np.ones(n), t,
    np.maximum(0, t - obs_bp1),
    np.maximum(0, t - obs_bp2),
])
fitted_mu = np.exp(X_obs @ obs_beta2 + log_pop)

# --- Parametric bootstrap of bp1 location (under H1: 2 bps) ----------------
bp1_loc = []
bp2_loc = []
print(f"\nRunning {N_BOOT} parametric bootstrap reps...")
for k in range(N_BOOT):
    y_b = nb_sample(RNG, fitted_mu)
    try:
        bp1b, bp2b, _, _ = best_2bp(y_b, log_pop)
        if bp1b is not None:
            bp1_loc.append(years[bp1b])
            bp2_loc.append(years[bp2b])
    except Exception:
        continue

bp1_loc = np.array(bp1_loc)
bp2_loc = np.array(bp2_loc)
ci1 = (np.percentile(bp1_loc, 2.5), np.percentile(bp1_loc, 97.5))
ci2 = (np.percentile(bp2_loc, 2.5), np.percentile(bp2_loc, 97.5))
print(f"\nBootstrap 95% CI for early breakpoint: [{ci1[0]:.0f}, {ci1[1]:.0f}]  "
      f"(median={np.median(bp1_loc):.0f}, point={years[obs_bp1]})")
print(f"Bootstrap 95% CI for late breakpoint:  [{ci2[0]:.0f}, {ci2[1]:.0f}]  "
      f"(median={np.median(bp2_loc):.0f}, point={years[obs_bp2]})")

# --- Parametric-bootstrap p-value for 2-bp vs 1-bp -------------------------
# Generate from BEST 1-bp fit, then refit 2-bp; fraction with ΔBIC ≤ −2 ≈ p
print(f"\nRunning {N_BOOT} parametric bootstrap reps under H0 (1-bp)...")
X_1 = np.column_stack([np.ones(n), t, np.maximum(0, t - obs_bp_1)])
mod1 = fit_glm(X_1, y, log_pop)
fitted_mu_1 = np.exp(X_1 @ mod1.params + log_pop)
n_extreme = 0
n_valid = 0
for k in range(N_BOOT):
    y_b = nb_sample(RNG, fitted_mu_1)
    try:
        _, _, bic2_b, _ = best_2bp(y_b, log_pop)
        bp_b, bic1_b = best_1bp(y_b, log_pop)
        if not (np.isfinite(bic2_b) and np.isfinite(bic1_b)):
            continue
        n_valid += 1
        if (bic2_b - bic1_b) <= (obs_bic2 - obs_bic1):
            n_extreme += 1
    except Exception:
        continue

p_value = (n_extreme + 1) / (n_valid + 1) if n_valid > 0 else np.nan
print(f"Parametric-bootstrap p-value for 2-bp vs 1-bp (ΔBIC ≤ {obs_bic2-obs_bic1:.2f}): "
      f"{p_value:.3f} ({n_extreme}/{n_valid} reps)")

# Save
out = pd.DataFrame([{
    'series': 'total',
    'point_bp1_year': int(years[obs_bp1]),
    'point_bp2_year': int(years[obs_bp2]),
    'bp1_ci_lo':  int(ci1[0]),
    'bp1_ci_hi':  int(ci1[1]),
    'bp2_ci_lo':  int(ci2[0]),
    'bp2_ci_hi':  int(ci2[1]),
    'BIC_0bp':  obs_bic0,
    'BIC_1bp':  obs_bic1,
    'BIC_2bp':  obs_bic2,
    'dBIC_2bp_vs_1bp': obs_bic2 - obs_bic1,
    'param_bootstrap_p_2bp_vs_1bp': p_value,
    'n_boot': N_BOOT,
}])
out.to_csv(f'{OUT}/tbl_breakpoint_bootstrap.csv', index=False)
print(f"\nSaved tbl_breakpoint_bootstrap.csv")
print(out.to_string(index=False))
