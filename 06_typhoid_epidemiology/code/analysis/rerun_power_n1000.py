"""Run continuous-slope and step-shock power simulations with n=1000 sims at empirical alpha=0.154."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import statsmodels.api as sm
from concurrent.futures import ProcessPoolExecutor
import os, sys

DATA = '/sessions/epic-determined-feynman/mnt/논문_이건상/장티푸스/data'
TBL = '/sessions/epic-determined-feynman/mnt/논문_이건상/장티푸스/FBPD_R2/tables_new'
FIG = '/sessions/epic-determined-feynman/mnt/논문_이건상/장티푸스/FBPD_R2/figures'

df2 = pd.read_csv(f'{DATA}/df2_장티푸스.csv')
df2.columns = [c.strip().replace('\ufeff','') for c in df2.columns]
df2['year'] = df2['year'].astype(int)
df2 = df2[(df2.year>=2001)&(df2.year<=2024)]
ann = df2[(df2.sex=='계')&(df2.age_group=='계')].groupby('year', as_index=False)['cases'].sum()
mu_baseline = float(ann['cases'].mean())
print(f'mu_baseline={mu_baseline:.1f}', flush=True)

ALPHA = 0.154
n_yr = 24

def nb_sample(rng, mu, a, size):
    p_ = 1.0/(1.0 + a*mu); n_p = mu*p_/(1.0-p_)
    return rng.negative_binomial(n_p, p_, size=size)

def fit_bp_quick(y, n_bp, alpha=ALPHA, hmin=4):
    n_l = len(y); t = np.arange(n_l)
    X0 = sm.add_constant(t)
    if n_bp == 0:
        try: return sm.GLM(y, X0, family=sm.families.NegativeBinomial(alpha=alpha)).fit(maxiter=50, disp=False).bic_llf
        except: return np.inf
    if n_bp == 1:
        best = np.inf
        for bp in range(hmin, n_l-hmin+1):
            X = np.column_stack([X0, np.maximum(0, t-bp)])
            try:
                m = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha)).fit(maxiter=50, disp=False)
                if m.bic_llf < best: best = m.bic_llf
            except: pass
        return best
    return np.inf

def gen_series(rng, eff_pct, half=12):
    base = (1 - 0.073) ** np.arange(n_yr)
    rates = mu_baseline * base
    extra = np.ones(n_yr)
    for t in range(half, n_yr):
        extra[t] = (1.0 + eff_pct/100.0) ** (t - half + 1)
    rates = rates * extra
    return np.maximum(nb_sample(rng, rates, ALPHA, n_yr), 0)

def gen_step(rng, mult, half=12):
    base = (1 - 0.073) ** np.arange(n_yr)
    rates = mu_baseline * base
    rates_step = rates.copy(); rates_step[half] = rates[half]*mult
    return np.maximum(nb_sample(rng, rates_step, ALPHA, n_yr), 0)

def cont_scenario(args):
    eff, n_sim, seed = args
    rng = np.random.default_rng(seed); cnt = 0
    for s in range(n_sim):
        y = gen_series(rng, eff)
        if (fit_bp_quick(y,1) - fit_bp_quick(y,0)) < -6: cnt += 1
    return eff, cnt/n_sim

def step_scenario(args):
    mult, n_sim, seed = args
    rng = np.random.default_rng(seed); cnt = 0
    for s in range(n_sim):
        y = gen_step(rng, mult)
        if (fit_bp_quick(y,1) - fit_bp_quick(y,0)) < -6: cnt += 1
    return mult, cnt/n_sim

effects = [0, 25, 50, 75, 100, 150, 200, 300]
mults = [1.0, 1.5, 2.0, 3.0, 5.0]
N_SIM = 1000

print('Continuous slope, n=1000 ...', flush=True)
with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
    res_c = list(ex.map(cont_scenario, [(eff, N_SIM, 42 + 1000*i) for i, eff in enumerate(effects)]))
res_c.sort()
out_c = pd.DataFrame({'effect_pct_per_year': [e for e,_ in res_c],
                      'detection_prob': [d for _,d in res_c],
                      'n_sim': N_SIM, 'alpha': ALPHA})
out_c.to_csv(f'{TBL}/tbl_power_sim_v2.csv', index=False)
print(out_c.to_string(index=False), flush=True)

print('Step shock, n=1000 ...', flush=True)
with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
    res_s = list(ex.map(step_scenario, [(m, N_SIM, 42 + 7000*i) for i, m in enumerate(mults)]))
res_s.sort()
out_s = pd.DataFrame({'step_multiplier': [m for m,_ in res_s],
                      'detection_prob': [d for _,d in res_s],
                      'n_sim': N_SIM, 'alpha': ALPHA})
out_s.to_csv(f'{TBL}/tbl_power_sim_step.csv', index=False)
print(out_s.to_string(index=False), flush=True)

# Plot Figure 4
detect = [d for _,d in res_c]
fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
ax.plot(effects, [d*100 for d in detect], 'o-', color='steelblue',
        linewidth=2, markersize=8, markeredgecolor='black')
ax.axhline(80, color='red', linestyle='--', linewidth=1.0, label='80% power threshold')
ax.axhline(5, color='gray', linestyle=':', linewidth=1.0, label='5% nominal FPR')
ax.set_xlabel('Post-breakpoint additional rate change (% per year)')
ax.set_ylabel('Detection probability (%) with deltaBIC < -6')
ax.set_title(f'Figure 4.  Power to detect a single NB breakpoint\n(n=24, dispersion alpha={ALPHA}, {N_SIM} simulations per effect size)',
             fontsize=11, fontweight='bold')
ax.set_ylim(-2, 105); ax.legend(loc='lower right', framealpha=0.95); ax.grid(axis='y', alpha=0.3)
plt.savefig(f'{FIG}/Figure_4.png', dpi=180, bbox_inches='tight')
plt.close()
print('Saved Figure_4 with n=1000', flush=True)
