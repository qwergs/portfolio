"""Build the supplementary figures (S1–S3) — COLOUR version.

Numbering after the main paper was restructured (5-panel Fig 1 in main):
  S1  STL decomposition of weekly counts
  S2  Provincial rates + ≥70-year case-share trend (was S3)
  S3  Two-breakpoint NB GLM fit overlay (was S4)
The previous "S2 — Age-specific IRRs" was removed because main Figure 1E
already shows age-specific incidence trends.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'figure.dpi': 100,
})
OUT = '/sessions/elegant-quirky-hamilton/mnt/outputs/hepa_work'

# ── S1 — STL decomposition (colour) ──────────────────────────────────
weekly = pd.read_csv(f'{OUT}/weekly_total.csv', parse_dates=['date']).set_index('date')['cases']
log_y = np.log(weekly.values + 1)
stl = STL(log_y, period=52, seasonal=25, trend=103, robust=True).fit()
fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
axes[0].plot(weekly.index, log_y, color='black', linewidth=0.7); axes[0].set_ylabel('log(weekly cases + 1)\n(observed)')
axes[1].plot(weekly.index, stl.trend, color='tab:blue', linewidth=1.2); axes[1].set_ylabel('Trend')
axes[2].plot(weekly.index, stl.seasonal, color='tab:green', linewidth=0.7); axes[2].set_ylabel('Seasonal')
axes[3].plot(weekly.index, stl.resid, color='tab:red', linewidth=0.5); axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Year')
fig.suptitle('Figure S1.  STL decomposition of weekly hepatitis A counts (s.window = 25, t.window = 103, robust)',
             fontsize=11, fontweight='bold')
plt.savefig(f'{OUT}/Figure_S1.png', dpi=180, bbox_inches='tight')
plt.savefig(f'{OUT}/Figure_S1.tif', dpi=300, format='tiff',
            pil_kwargs={'compression':'tiff_lzw'}, bbox_inches='tight')
plt.close()
print('Saved Figure_S1')

# ── S2 — Provincial rates + 70+ share trend (colour) ────────────────
prov = pd.read_csv(f'{OUT}/tbl_provincial_rates.csv').sort_values('mean_annual_rate', ascending=True)
share = pd.read_csv(f'{OUT}/tbl_age70_share.csv')

fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
axA = axes[0]
axA.barh(prov.sido_en, prov.mean_annual_rate, color='steelblue',
         edgecolor='black', alpha=0.85)
axA.set_xlabel('Mean annual incidence rate (per 100 000)')
axA.set_title('A. Provincial hepatitis A incidence (2011–2024 average)',
              fontsize=10.5, fontweight='bold')
axA.axvline(prov.mean_annual_rate.mean(), color='red', linestyle='--',
            linewidth=1, alpha=0.8, label=f'Mean = {prov.mean_annual_rate.mean():.2f}')
axA.legend()

axB = axes[1]
axB.bar(share.year, share.share_70plus_pct, color='tab:orange', alpha=0.85,
        edgecolor='black', label='≥70 yr share of cases')
axB.plot(share.year, share.pop70_pct, color='tab:blue', marker='o',
         linewidth=2, markeredgecolor='black', markerfacecolor='white',
         label='≥70 yr share of population')
axB.set_xlabel('Year')
axB.set_ylabel('Per cent (%)')
axB.set_title('B. Hepatitis A cases vs population share aged ≥70 years',
              fontsize=10.5, fontweight='bold')
axB.legend(loc='upper left')
axB.set_xticks(np.arange(2011, 2025, 2))
plt.savefig(f'{OUT}/Figure_S2.png', dpi=180, bbox_inches='tight')
plt.savefig(f'{OUT}/Figure_S2.tif', dpi=300, format='tiff',
            pil_kwargs={'compression':'tiff_lzw'}, bbox_inches='tight')
plt.close()
print('Saved Figure_S2 (provincial rates + 70+ share trend)')

# ── S3 — Breakpoint NB GLM fit overlay ─────────────────────────────
import statsmodels.api as sm
ann = pd.read_csv(f'{OUT}/tbl_annual_rates.csv').sort_values('year').reset_index(drop=True)
t = (ann['year'] - ann['year'].min()).values
bp = [2014 - ann.year.min(), 2019 - ann.year.min()]
X = np.column_stack([np.ones_like(t), t,
                     np.maximum(0, t - bp[0]),
                     np.maximum(0, t - bp[1])])
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for axi, (label, y) in enumerate([('Total', ann.total), ('Domestic', ann.domestic)]):
    nb_full = sm.GLM(y.values, X,
                     family=sm.families.NegativeBinomial(alpha=1.0),
                     offset=np.log(ann.pop_total.values)).fit()
    pred = nb_full.predict(X)
    rate = y.values / ann.pop_total.values * 1e5
    pred_rate = pred / ann.pop_total.values * 1e5
    ax = axes[axi]
    ax.plot(ann.year, rate, 'o-', color='black', markersize=6,
            markerfacecolor='white', markeredgecolor='black', label='Observed')
    ax.plot(ann.year, pred_rate, '-', color='tab:red', linewidth=2,
            label='2-bp NB GLM (2014, 2019)')
    for b in [2014, 2019]:
        ax.axvline(b, color='tab:red', linestyle=':', alpha=0.6)
    ax.set_xlabel('Year')
    ax.set_ylabel('Incidence rate (per 100 000)')
    ax.set_title(f'{chr(65+axi)}. {label} cases', fontsize=10.5, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xticks(np.arange(2011, 2025, 2))
fig.suptitle('Figure S3.  Continuous piecewise-linear NB GLM with breakpoints at 2014 and 2019',
             fontsize=11, fontweight='bold')
plt.savefig(f'{OUT}/Figure_S3.png', dpi=180, bbox_inches='tight')
plt.savefig(f'{OUT}/Figure_S3.tif', dpi=300, format='tiff',
            pil_kwargs={'compression':'tiff_lzw'}, bbox_inches='tight')
plt.close()
print('Saved Figure_S3 (breakpoint NB GLM fit overlay)')
