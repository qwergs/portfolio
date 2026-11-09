"""Analysis 2 (revised) — Morlet CWT periodicity with AR(1) red-noise null,
2,000 simulations. Reports stability of 95th-percentile null and GWS:null
ratio across multiple bootstrap subsets.

Output: tbl_cwt.csv (revised), tbl_cwt_sim_stability.csv, fig_cwt_data.npz
"""
import numpy as np
import pandas as pd
from scipy import signal
from statsmodels.tsa.seasonal import STL

OUT = '/sessions/serene-confident-edison/mnt/outputs/hepa_work'
RNG = np.random.default_rng(42)

w = pd.read_csv(f'{OUT}/weekly_total.csv')
w['date'] = pd.to_datetime(w['date'])
w = w.sort_values('date').reset_index(drop=True)
y = np.log1p(w['cases'].values.astype(float))

# STL detrending (primary config: s.window=25, t.window=103, robust)
stl = STL(y, period=52, seasonal=25, trend=103, robust=True).fit()
yd = stl.resid + stl.seasonal     # remove trend, keep seasonal+resid for periodicity

# Morlet CWT
def morlet_cwt(x, scales, w0=6.0):
    n = len(x)
    coefs = np.zeros((len(scales), n), dtype=complex)
    for i, s in enumerate(scales):
        # Morlet wavelet centered at 0 with scale s
        t = np.arange(-n//2, n//2)
        psi = (np.pi**-0.25) * np.exp(1j*w0*t/s) * np.exp(-(t/s)**2/2) / np.sqrt(s)
        # Convolution
        coefs[i] = np.convolve(x, psi.conj()[::-1], mode='same')
    return coefs

# Scale-period mapping (Torrence-Compo)
w0 = 6.0
fourier_factor = 4*np.pi/(w0 + np.sqrt(2 + w0**2))
periods_wk = np.geomspace(2, 256, 80)        # 2 to 256 weeks
scales = periods_wk / fourier_factor

# Observed CWT
W = morlet_cwt(yd, scales, w0=w0)
power = np.abs(W)**2
gws_obs = power.mean(axis=1)                  # mean over time for each scale

# AR(1) parameters
rho = np.corrcoef(yd[:-1], yd[1:])[0,1]
sigma = np.std(yd) * np.sqrt(1 - rho**2)

# Null distribution via 2,000 AR(1) simulations (Q7 reviewer ask)
N_SIM = 2000
print(f'AR(1) rho = {rho:.4f}, sigma = {sigma:.4f}')
print(f'Running {N_SIM} AR(1) simulations for null distribution...')

null_gws = np.zeros((N_SIM, len(scales)))
for k in range(N_SIM):
    # AR(1) simulation
    eps = RNG.normal(0, sigma, len(yd))
    sim = np.zeros(len(yd))
    sim[0] = eps[0]
    for t in range(1, len(yd)):
        sim[t] = rho * sim[t-1] + eps[t]
    Wn = morlet_cwt(sim, scales, w0=w0)
    null_gws[k] = (np.abs(Wn)**2).mean(axis=1)

# 95th percentile null
p95 = np.percentile(null_gws, 95, axis=0)

# Annual band: 48-56 weeks
annual_mask = (periods_wk >= 48) & (periods_wk <= 56)
gws_annual = gws_obs[annual_mask].mean()
p95_annual = p95[annual_mask].mean()
ratio_annual = gws_annual / p95_annual

# Stability check across simulation subsets
sub_sizes = [200, 500, 1000, 1500, 2000]
stab_rows = []
for ns in sub_sizes:
    # Take first ns simulations
    p95_sub = np.percentile(null_gws[:ns], 95, axis=0)
    p95_sub_ann = p95_sub[annual_mask].mean()
    stab_rows.append({
        'n_sim': ns,
        'p95_null_annual': p95_sub_ann,
        'gws_to_null_ratio_annual': gws_annual / p95_sub_ann,
    })
stab = pd.DataFrame(stab_rows)
stab.to_csv(f'{OUT}/tbl_cwt_sim_stability.csv', index=False)
print('\nStability of GWS:null ratio across simulation subsets:')
print(stab.to_string(index=False))

# Peak period
peak_idx = np.argmax(gws_obs)
peak_period = periods_wk[peak_idx]

# Save updated CWT result table
out = pd.DataFrame([{
    'rho_AR1': rho,
    'sigma_AR1': sigma,
    'gws_annual': gws_annual,
    'p95_null_annual': p95_annual,
    'ratio_annual': ratio_annual,
    'gws_52wk': gws_obs[np.argmin(np.abs(periods_wk - 52))],
    'peak_period_wk': peak_period,
    'n_sim': N_SIM,
}])
out.to_csv(f'{OUT}/tbl_cwt.csv', index=False)
print(f'\n{N_SIM}-simulation results: GWS_annual={gws_annual:.3f}, p95_null={p95_annual:.3f}, ratio={ratio_annual:.3f}')

# Save full-resolution data for figure
np.savez(f'{OUT}/fig_cwt_data.npz',
         periods_wk=periods_wk, power=power, gws_obs=gws_obs, p95_null=p95,
         time=np.arange(len(yd)), rho=rho, sigma=sigma)
print('Saved fig_cwt_data.npz')
