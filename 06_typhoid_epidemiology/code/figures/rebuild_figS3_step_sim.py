"""Rebuild Figure S3 (NB GLM fit overlay with proper rate/100k scaling)
   and recompute Supp S9 step-shock power simulation at empirical alpha."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import statsmodels.api as sm

DATA = '/sessions/epic-determined-feynman/mnt/논문_이건상/장티푸스/data'
TBL = '/sessions/epic-determined-feynman/mnt/논문_이건상/장티푸스/FBPD_R2/tables_new'
FIG = '/sessions/epic-determined-feynman/mnt/논문_이건상/장티푸스/FBPD_R2/figures'

# Annual rates from the existing manuscript table
ann = pd.read_csv(f'{TBL}/tbl_asr.csv').sort_values('year').reset_index(drop=True)
years = ann['year'].values
crude = ann['crude_rate'].values  # /100k

# Reload df1/df2 to get cases AND population
df2 = pd.read_csv(f'{DATA}/df2_장티푸스.csv')
df2.columns = [c.strip().replace('\ufeff','') for c in df2.columns]
df2['year'] = df2['year'].astype(int)
df2 = df2[(df2.year>=2001)&(df2.year<=2024)]
ann_total = df2[(df2.sex=='계')&(df2.age_group=='계')].groupby('year', as_index=False)['cases'].sum()

df1 = pd.read_csv(f'{DATA}/df1_장티푸스.csv')
df1.columns = [c.strip().replace('\ufeff','') for c in df1.columns]
df1['year'] = df1['year'].astype(int)
df1 = df1[(df1.year>=2001)&(df1.year<=2024)]
ann_dom = df1.groupby('year', as_index=False)['cases'].sum()
ann_dom.columns = ['year','dom']

# Population: derive from Table 1 (manuscript Table 1 populations)
TBL1_POP = {
    2001:47457211,2002:47957781,2003:48109404,2004:48264962,2005:48778235,
    2006:48978449,2007:49254055,2008:49792368,2009:50027977,2010:51314398,
    2011:51527552,2012:51740826,2013:51927682,2014:52183502,2015:52378325,
    2016:51696216,2017:51778544,2018:51826059,2019:51849861,2020:51829023,
    2021:51638809,2022:51439038,2023:51337778,2024:51985381
}
df = ann_total.copy()
df = df.merge(ann_dom, on='year', how='left')
df['pop'] = df['year'].map(TBL1_POP)
df['rate_total'] = df['cases']/df['pop']*1e5
df['rate_dom']   = df['dom']/df['pop']*1e5

# NB GLM (no breakpoint) fit on log-rate ~ year, return predicted rate /100k
def fit_nb_no_bp(y_count, pop, year):
    t = year - year.min()
    X = sm.add_constant(t.astype(float))
    offset = np.log(pop.astype(float))
    m = sm.GLM(y_count, X, family=sm.families.NegativeBinomial(alpha=0.154),
               offset=offset).fit(maxiter=200, disp=False)
    pred_rate = m.predict(X, offset=offset) / pop * 1e5
    return m, pred_rate

m_t, pred_t = fit_nb_no_bp(df['cases'].values, df['pop'].values, df['year'].values)
m_d, pred_d = fit_nb_no_bp(df['dom'].values, df['pop'].values, df['year'].values)
print('NB GLM total: BIC', round(m_t.bic_llf, 2), '   coef', m_t.params)
print('NB GLM domestic: BIC', round(m_d.bic_llf, 2), '  coef', m_d.params)

# Plot Figure S3
fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
fig.suptitle('Figure S3.  BIC-best negative-binomial GLM fit overlay (no breakpoint, log scale)',
             fontsize=11.5, fontweight='bold')
ax[0].plot(df['year'], df['rate_total'], 'ko-', mfc='white', label='Observed', lw=1.0, ms=5)
ax[0].plot(df['year'], pred_t, 'r-', lw=2, label='BIC-best NB GLM (no breakpoint)')
ax[0].set_yscale('log'); ax[0].set_xlabel('Year'); ax[0].set_ylabel('Incidence rate (per 100 000)')
ax[0].set_title('A.  Total cases', fontsize=10.5); ax[0].legend(loc='upper right'); ax[0].grid(alpha=0.3)

ax[1].plot(df['year'], df['rate_dom'], 'ko-', mfc='white', label='Observed', lw=1.0, ms=5)
ax[1].plot(df['year'], pred_d, 'r-', lw=2, label='BIC-best NB GLM (no breakpoint)')
ax[1].set_yscale('log'); ax[1].set_xlabel('Year'); ax[1].set_ylabel('Incidence rate (per 100 000)')
ax[1].set_title('B.  Domestic cases', fontsize=10.5); ax[1].legend(loc='upper right'); ax[1].grid(alpha=0.3)
plt.savefig(f'{FIG}/Figure_S3.png', dpi=180, bbox_inches='tight')
plt.close()
print('Saved Figure_S3 (corrected NB GLM scaling)')

# --- Step-shock power simulation, empirical alpha=0.154, n=300 (matches reviewer response) ---
ALPHA = 0.154; n_yr = 24
mu_baseline = float(df['cases'].mean())
print(f'mu_baseline={mu_baseline:.1f}')
def nb_sample(rng, mu, a, size):
    p_ = 1.0/(1.0 + a*mu); n_p = mu*p_/(1.0-p_)
    return rng.negative_binomial(n_p, p_, size=size)

def fit_bp_quick(y, n_bp, alpha=ALPHA, hmin=4):
    n_l = len(y); t_l = np.arange(n_l)
    X0 = sm.add_constant(t_l)
    if n_bp == 0:
        try: return sm.GLM(y, X0, family=sm.families.NegativeBinomial(alpha=alpha)).fit(maxiter=50, disp=False).bic_llf
        except: return np.inf
    if n_bp == 1:
        best = np.inf
        for bp in range(hmin, n_l-hmin+1):
            X = np.column_stack([X0, np.maximum(0, t_l-bp)])
            try:
                m = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha)).fit(maxiter=50, disp=False)
                if m.bic_llf < best: best = m.bic_llf
            except: pass
        return best
    return np.inf

def gen_step(rng, mult, mu0=mu_baseline, half=12):
    base_trend = (1 - 0.073) ** np.arange(n_yr)
    rates = mu0 * base_trend
    rates_step = rates.copy()
    rates_step[half] = rates[half] * mult
    return np.maximum(nb_sample(rng, rates_step, ALPHA, n_yr), 0)

mults = [1.0, 1.5, 2.0, 3.0, 5.0]
N_SIM = 300
rng = np.random.default_rng(42)
detect = []
for m_ in mults:
    cnt = 0
    for s in range(N_SIM):
        y = gen_step(rng, m_)
        b0 = fit_bp_quick(y, 0); b1 = fit_bp_quick(y, 1)
        if (b1 - b0) < -6: cnt += 1
    detect.append(cnt/N_SIM)
    print(f'  step mult {m_}x: detection {cnt/N_SIM:.3f}')

out = pd.DataFrame({'step_multiplier': mults, 'detection_prob': detect,
                    'n_sim': N_SIM, 'alpha': ALPHA})
out.to_csv(f'{TBL}/tbl_power_sim_step.csv', index=False)
print(out.to_string(index=False))
