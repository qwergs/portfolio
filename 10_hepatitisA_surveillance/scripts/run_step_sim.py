"""Step-change power sim only (continues v2 work). 500 sims for speed."""
import numpy as np, pandas as pd
import statsmodels.api as sm

OUT = '/sessions/serene-confident-edison/mnt/outputs/hepa_work'
RNG = np.random.default_rng(42)

ann = pd.read_csv(f'{OUT}/tbl_annual_rates.csv').sort_values('year').reset_index(drop=True)
mu_baseline = ann['total'].mean()
ALPHA = 0.68
N = 14
N_SIM = 500
DBIC = -6

def nb_sample(rng, mu, alpha, size):
    p = 1/(1+alpha*mu); n_p = mu*p/(1-p)
    return rng.negative_binomial(n_p, p, size=size)

def fit_bp(y, n_bp, hmin=3):
    n = len(y); t = np.arange(n)
    X0 = sm.add_constant(t)
    if n_bp == 0:
        try: return sm.GLM(y, X0, family=sm.families.NegativeBinomial(alpha=1.0)).fit(maxiter=200).bic_llf
        except: return np.inf
    if n_bp == 1:
        best = np.inf
        for bp in range(hmin, n-hmin+1):
            X = np.column_stack([X0, np.maximum(0, t-bp)])
            try:
                m = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0)).fit(maxiter=200)
                if m.bic_llf < best: best = m.bic_llf
            except: pass
        return best
    return np.inf

def step_sim(rng, mult, n):
    mu = np.full(n, mu_baseline); mu[9] = mu[9]*mult
    return nb_sample(rng, np.maximum(mu,1), ALPHA, n)

mults = [1.0, 1.5, 2.0, 3.0, 5.0, 7.0]
out = []
for m in mults:
    n_d = 0
    for _ in range(N_SIM):
        y = step_sim(RNG, m, N)
        b0 = fit_bp(y, 0); b1 = fit_bp(y, 1)
        if (b1-b0) < DBIC: n_d += 1
    pr = n_d / N_SIM
    out.append({'step_multiplier': m, 'detection_prob': pr, 'n_sim': N_SIM,
                'alpha': ALPHA, 'dBIC_threshold': DBIC})
    print(f'  step={m}x: detection prob = {pr:.3f}')
df = pd.DataFrame(out)
df.to_csv(f'{OUT}/tbl_power_sim_step.csv', index=False)
print('Saved.')
print(df.to_string(index=False))
