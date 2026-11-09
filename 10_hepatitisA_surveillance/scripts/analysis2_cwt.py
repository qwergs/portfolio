"""Analysis 2 — Continuous wavelet transform of weekly national domestic series
+ Global wavelet spectrum + AR(1) red-noise null.

Output:
  tbl_cwt.csv         — annual band ratio, peak period, GWS at 52wk
  fig_cwt_data.npz    — power, periods, time, gws, p95_null (for plotting)
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

OUT = '/sessions/elegant-quirky-hamilton/mnt/outputs/hepa_work'

weekly = pd.read_csv(f'{OUT}/weekly_total.csv', parse_dates=['date']).set_index('date')['cases']
y_raw = weekly.values.astype(float)
log_y = np.log(y_raw + 1.0)

# STL (s.window=25, t.window=103, robust=True) — primary STL config
stl = STL(log_y, period=52, seasonal=25, trend=103, robust=True).fit()
detrended = log_y - stl.trend

def fit_ar1(x):
    x = x - x.mean()
    num = (x[:-1]*x[1:]).sum()
    den = (x[:-1]**2).sum()
    rho = num/den if den > 0 else 0.0
    resid = x[1:] - rho*x[:-1]
    return rho, resid.std(ddof=1)

def morlet_cwt(signal, dt, periods, omega0=6.0):
    n = len(signal)
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, d=dt) * 2 * np.pi
    scales = periods * (4*np.pi) / (omega0 + np.sqrt(2 + omega0**2))
    W = np.zeros((len(scales), n), dtype=complex)
    for i, s in enumerate(scales):
        expnt = -0.5 * (s*freqs - omega0)**2 * (freqs > 0)
        norm = np.sqrt(2*np.pi*s/dt) * np.pi**(-0.25)
        daughter = norm * np.exp(expnt) * (freqs > 0)
        W[i] = np.fft.ifft(fft * daughter)
    return np.abs(W)**2

periods = np.arange(8, 261, 2).astype(float)
power = morlet_cwt(detrended, 1.0, periods, omega0=6.0)
gws = power.mean(axis=1)

# AR(1) red-noise null (200 sims)
rho, sigma = fit_ar1(detrended)
rng = np.random.default_rng(42)
null_gws = np.zeros((200, len(periods)))
for i in range(200):
    x = np.zeros(len(detrended))
    e = rng.normal(0, sigma, len(x))
    for t in range(1, len(x)):
        x[t] = rho*x[t-1] + e[t]
    null_gws[i] = morlet_cwt(x, 1.0, periods, omega0=6.0).mean(axis=1)
p95 = np.percentile(null_gws, 95, axis=0)

# Annual band 48-56 weeks
mask52 = (periods >= 48) & (periods <= 56)
gws_annual = gws[mask52].mean()
p95_annual = p95[mask52].mean()
ratio = gws_annual / p95_annual

# Peak period in annual band
idx52 = np.argmin(np.abs(periods - 52))
peak_period = periods[gws.argmax()]

print(f'AR(1) rho = {rho:.4f}, sigma = {sigma:.4f}')
print(f'GWS annual band mean = {gws_annual:.4f}')
print(f'AR(1) null 95th percentile annual band = {p95_annual:.4f}')
print(f'Ratio (GWS:null) = {ratio:.3f}  → {"SIGNIFICANT" if ratio>1 else "NOT significant"} annual band')
print(f'GWS at 52wk = {gws[idx52]:.4f}, peak period = {peak_period:.0f} wk')

pd.DataFrame([{
    'rho_AR1': rho,
    'sigma_AR1': sigma,
    'gws_annual': gws_annual,
    'p95_null_annual': p95_annual,
    'ratio_annual': ratio,
    'gws_52wk': float(gws[idx52]),
    'peak_period_wk': float(peak_period),
}]).to_csv(f'{OUT}/tbl_cwt.csv', index=False)

# Save full time-frequency data for plotting
np.savez(f'{OUT}/fig_cwt_data.npz',
         power=power, periods=periods, time=np.arange(len(detrended)),
         dates=weekly.index.values, gws=gws, p95=p95,
         detrended=detrended, raw=y_raw, log_y=log_y, trend=stl.trend)
print(f'Saved fig_cwt_data.npz')
