"""Analysis 8 — STL × Morlet ω₀ sensitivity grid.

For each (s.window, t.window, robust) STL config and Morlet ω₀,
re-run STL → CWT → AR(1) red-noise null (200 sims) and report
annual-band (48-56 wk) GWS:null ratio.

Output: tbl_stl_sensitivity.csv
"""
import numpy as np, pandas as pd
from statsmodels.tsa.seasonal import STL

OUT = '/sessions/elegant-quirky-hamilton/mnt/outputs/hepa_work'
weekly = pd.read_csv(f'{OUT}/weekly_total.csv', parse_dates=['date']).set_index('date')['cases']
log_y = np.log(weekly.values + 1.0)

def ar1(x):
    x = x - x.mean()
    num = (x[:-1]*x[1:]).sum(); den = (x[:-1]**2).sum()
    rho = num/den if den>0 else 0.0
    return rho, (x[1:] - rho*x[:-1]).std(ddof=1)

def cwt(sig, dt, periods, omega0=6.0):
    n = len(sig)
    fft = np.fft.fft(sig)
    freqs = np.fft.fftfreq(n, d=dt) * 2 * np.pi
    scales = periods * (4*np.pi) / (omega0 + np.sqrt(2 + omega0**2))
    W = np.zeros((len(scales), n), dtype=complex)
    for i, s in enumerate(scales):
        e = -0.5 * (s*freqs - omega0)**2 * (freqs > 0)
        norm = np.sqrt(2*np.pi*s/dt) * np.pi**(-0.25)
        W[i] = np.fft.ifft(fft * norm * np.exp(e) * (freqs > 0))
    return np.abs(W)**2

def run(log_y, sw, tw, rb, om, nsim=200, seed=42):
    res = STL(log_y, period=52, seasonal=sw, trend=tw, robust=rb).fit()
    detr = log_y - res.trend
    periods = np.arange(8, 261, 2).astype(float)
    pw = cwt(detr, 1.0, periods, omega0=om)
    rho, sig = ar1(detr)
    rng = np.random.default_rng(seed)
    null_pk = np.zeros((nsim, len(periods)))
    for i in range(nsim):
        x = np.zeros(len(detr)); e = rng.normal(0, sig, len(x))
        for t in range(1, len(x)): x[t] = rho*x[t-1] + e[t]
        null_pk[i] = cwt(x, 1.0, periods, omega0=om).mean(axis=1)
    p95 = np.percentile(null_pk, 95, axis=0)
    gws = pw.mean(axis=1)
    m = (periods >= 48) & (periods <= 56)
    idx52 = np.argmin(np.abs(periods - 52))
    return {
        'gws_annual': float(gws[m].mean()),
        'p95_null_annual': float(p95[m].mean()),
        'gws_ratio': float(gws[m].mean() / p95[m].mean()),
        'pct_time_sig_52wk': float((pw[idx52] > p95[idx52]).mean()),
        'ar1_rho': rho,
    }

configs = []
for sw in [13, 25, 53]:
    for tw in [53, 103, 203]:
        for rb in [False, True]:
            configs.append({'s.window':sw,'t.window':tw,'robust':rb,'omega0':6.0})
for om in [5, 6, 8]:
    configs.append({'s.window':25,'t.window':103,'robust':True,'omega0':om})

rows = []
for c in configs:
    try:
        r = run(log_y, c['s.window'], c['t.window'], c['robust'], c['omega0'])
        rows.append({**c, **r})
    except Exception as e:
        rows.append({**c, 'error':str(e)})

df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/tbl_stl_sensitivity.csv', index=False)
print(df.to_string(index=False))
print(f'\nGWS-ratio range across configs: {df.gws_ratio.min():.3f} - {df.gws_ratio.max():.3f}')
print(f'Configs with ratio > 1: {(df.gws_ratio > 1).sum()} / {len(df)}')
