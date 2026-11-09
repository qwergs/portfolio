"""Figures 1, 2 and 3 of the main text.

Requires the outputs of 01_trend.py and 03_spatial.py (run those first).

Run: python scripts/04_figures.py
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from common import OUT, load_annual

plt.rcParams.update({'axes.unicode_minus': True, 'font.family': 'DejaVu Sans', 'font.size': 8.5,
                     'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
                     'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
C_HH, C_LL, C_HL = '#C0392B', '#2E6DA4', '#E08A2E'
a = load_annual()
t1 = a.rename(columns={'ehec_total': 'total', 'ehec_domestic': 'domestic', 'ehec_imported': 'imported'})
pop = t1.population.astype(float)

# Figure 1: annual notification rate with the constant-slope NB fit
y = t1.total.values.astype(float); years = t1.index.values
X = sm.add_constant(np.arange(len(y), dtype=float))
nb = sm.NegativeBinomial(y, X, offset=np.log(pop.values)).fit(disp=0, maxiter=500)
beta = np.asarray(nb.params)[:2]; cov = np.asarray(nb.cov_params())[:2, :2]
eta = X @ beta; se = np.sqrt(np.einsum('ij,jk,ik->i', X, cov, X))
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=600)
plt.rcParams.update({'font.size': 11})
dom = t1.domestic.values / pop.values * 1e5; imp = t1.imported.values / pop.values * 1e5
ax.fill_between(years, np.exp(eta - 1.96 * se) * 1e5, np.exp(eta + 1.96 * se) * 1e5, color='#8A70C0', alpha=0.18, lw=0)
ax.bar(years, dom, color='#A6CEE3', edgecolor='#4E9CC4', lw=0.5, width=0.78)
ax.bar(years, imp, bottom=dom, color='#FDBF6F', edgecolor='#E08A2E', lw=0.5, width=0.78)
ax.plot(years, np.exp(eta) * 1e5, ls=(0, (6, 4)), color='#4A1D96', lw=1.6)
ax.set_xlabel('Year'); ax.set_ylabel('Notification rate (per 100,000)')
ax.set_xlim(2000.2, 2024.8); ax.set_ylim(0, None); ax.set_xticks(range(2001, 2025, 2))
for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
ax.legend(handles=[Patch(fc='#8A70C0', alpha=0.18, label='NB fit 95% CI'),
                   Line2D([0], [0], ls=(0, (6, 4)), color='#4A1D96', lw=1.6, label='Negative-binomial fitted trend'),
                   Patch(fc='#A6CEE3', ec='#4E9CC4', label='Domestic'), Patch(fc='#FDBF6F', ec='#E08A2E', label='Imported')],
          loc='upper left', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'Figure_1.tif'), dpi=600, pil_kwargs={'compression': 'tiff_lzw'}); plt.close(fig)
plt.rcParams.update({'font.size': 8.5})

# Figure 2: provincial pattern (values from Supplementary Table S7)
s7 = pd.read_csv(os.path.join(OUT, 'tableS7_lisa.csv'))
S7 = s7.rename(columns={'province': 'prov', 'rate_per_100000': 'rate', 'local_I': 'localI', 'quadrant': 'quad',
                        'p_raw': 'p', 'q_bh': 'q'}).set_index('prov')
ADJ = {'Seoul': ['Gyeonggi', 'Incheon'], 'Busan': ['Gyeongnam', 'Ulsan'], 'Daegu': ['Gyeongbuk', 'Gyeongnam'],
       'Incheon': ['Seoul', 'Gyeonggi'], 'Gwangju': ['Jeonnam', 'Jeju'], 'Daejeon': ['Chungnam', 'Chungbuk'],
       'Ulsan': ['Busan', 'Gyeongbuk', 'Gyeongnam'], 'Gyeonggi': ['Seoul', 'Incheon', 'Gangwon', 'Chungbuk', 'Chungnam'],
       'Gangwon': ['Gyeonggi', 'Chungbuk', 'Gyeongbuk'],
       'Chungbuk': ['Gangwon', 'Chungnam', 'Daejeon', 'Gyeongbuk', 'Jeonbuk', 'Gyeonggi'],
       'Chungnam': ['Gyeonggi', 'Chungbuk', 'Daejeon', 'Jeonbuk'],
       'Jeonbuk': ['Chungnam', 'Chungbuk', 'Gyeongbuk', 'Gyeongnam', 'Jeonnam'],
       'Jeonnam': ['Jeonbuk', 'Gyeongnam', 'Gwangju', 'Jeju'],
       'Gyeongbuk': ['Gangwon', 'Chungbuk', 'Jeonbuk', 'Gyeongnam', 'Daegu', 'Ulsan'],
       'Gyeongnam': ['Jeonnam', 'Jeonbuk', 'Gyeongbuk', 'Daegu', 'Busan', 'Ulsan'],
       'Jeju': ['Jeonnam', 'Gwangju']}  # Queen contiguity; Jeju linked to its two nearest provinces
names = list(ADJ); idx = {n: i for i, n in enumerate(names)}
W = np.zeros((16, 16))
for k, nbs in ADJ.items():
    for b in nbs:
        W[idx[k], idx[b]] = W[idx[b], idx[k]] = 1
W = W / W.sum(1, keepdims=True)
x = S7.reindex(names).rate.values
z = (x - x.mean()) / x.std(); lag = W @ z
I = (z @ lag) / (z @ z)
order = S7.sort_values('rate').index
col = {'HH': C_HH, 'LL': C_LL, 'HL': C_HL, 'LH': '#6A9F58'}
fig = plt.figure(figsize=(7.1, 3.5), dpi=600)
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.8, 1.25], wspace=0.12)
axA = fig.add_subplot(gs[0]); axC = fig.add_subplot(gs[1], sharey=axA); axB = fig.add_subplot(gs[2])
yy = np.arange(len(order))
axA.barh(yy, S7.loc[order, 'rate'], color=[col[q] for q in S7.loc[order, 'quad']], height=0.72)
axA.set_yticks(yy); axA.set_yticklabels(order)
axA.set_xlabel('Mean annual notification rate\nper 100,000 population')
axA.set_title('a  Notification rate')
axA.set_xlim(0, 1.3)
axC.barh(yy, S7.loc[order, 'localI'], color=[col[q] if p < 0.05 else '#B5B5B5'
                                              for q, p in zip(S7.loc[order, 'quad'], S7.loc[order, 'p'])], height=0.72)
sig = S7[S7.p < 0.05].sort_values('p')
axC.text(0.97, 0.36, "Raw p < 0.05:\n" + "\n".join(f"{k} {v:.3f}" for k, v in sig.p.items())
         + "\nFDR q ≥ {:.3f} for all\n(none significant\nafter adjustment)".format(S7.q.min()),
         transform=axC.transAxes, ha='right', va='center', fontsize=7, linespacing=1.15)
axC.axvline(0, color='k', lw=0.6)
axC.set_xlim(-0.3, 3.0); axC.set_xlabel("Local Moran's I")
axC.set_title('b  Local Moran\'s I'); plt.setp(axC.get_yticklabels(), visible=False); axC.tick_params(axis='y', length=0)
for ax in (axA, axC):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
# Moran scatter (panel b), placed at right
for pr in names:
    q = S7.loc[pr, 'quad']
    axB.scatter(z[idx[pr]], lag[idx[pr]], s=26 if q != 'LL' else 16, color=col[q],
                edgecolor='white', lw=0.4, zorder=3)
lab_off = {'Gwangju': (-0.15, 0.13, 'right'), 'Jeju': (0.12, 0.05, 'left'),
           'Jeonnam': (0.12, 0.05, 'left'), 'Ulsan': (0.12, -0.08, 'left')}
for pr, (dx, dy, ha) in lab_off.items():
    axB.text(z[idx[pr]] + dx, lag[idx[pr]] + dy, pr, fontsize=7.5, ha=ha)
b = np.polyfit(z, lag, 1)
xx = np.linspace(-1, 3.6, 10); axB.plot(xx, np.polyval(b, xx), color='#555555', lw=0.8, ls='--')
axB.axhline(0, color='k', lw=0.5); axB.axvline(0, color='k', lw=0.5)
axB.set_xlim(-1.0, 3.8); axB.set_ylim(-1.0, 2.2)
axB.set_xlabel('Standardised notification rate (z)'); axB.set_ylabel('Spatial lag of z')
axB.set_title("c  Moran scatter plot")
axB.text(0.98, 0.33, "Global Moran's I = {:.3f}\npermutation p = 0.0003\n(Queen contiguity;\nJeju linked to 2 nearest)".format(I),
         transform=axB.transAxes, ha='right', va='bottom', fontsize=7,
         bbox=dict(fc='white', ec='none', pad=1.5))
for s in ('top', 'right'): axB.spines[s].set_visible(False)
h = [Line2D([0], [0], marker='o', ls='', color=col[k], label=l, markersize=5)
     for k, l in [('HH', 'High-high'), ('HL', 'High-low'), ('LL', 'Low-low')]]
axB.legend(handles=h, loc='upper left', fontsize=7, frameon=False, handletextpad=0.2)
fig.subplots_adjust(left=0.125, right=0.985, top=0.92, bottom=0.17)
# move scatter slightly right for its own y label
pos = axB.get_position(); axB.set_position([pos.x0 + 0.06, pos.y0, pos.width - 0.06, pos.height])
fig.savefig(os.path.join(OUT, 'Figure_2.tif'), dpi=600, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)


# Figure 3: negative-control comparison
ser = {'EHEC': a.ehec_total.astype(float), 'Shigellosis': a.shigellosis.astype(float),
       'Typhoid fever': a.typhoid_fever.astype(float)}
est = {}
for k, y in ser.items():
    X = sm.add_constant(np.arange(24, dtype=float))
    m = sm.NegativeBinomial(y.values, X, offset=np.log(pop.values)).fit(disp=0, maxiter=500)
    b, se = m.params[1], m.bse[1]
    est[k] = [100 * np.expm1(v) for v in (b, b - 1.96 * se, b + 1.96 * se)]
    COL = {'EHEC': '#C8551B', 'Shigellosis': '#2E6DA4', 'Typhoid fever': '#1B9E77'}
MK = {'EHEC': 'o', 'Shigellosis': 's', 'Typhoid fever': '^'}
yrs = t1.index.values
fig = plt.figure(figsize=(7.1, 5.6), dpi=600)
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 0.8], hspace=0.5, wspace=0.28)
aA = fig.add_subplot(gs[0, 0]); aB = fig.add_subplot(gs[0, 1]); aC = fig.add_subplot(gs[1, :])
for k, y in ser.items():
    aA.plot(yrs, y.values / pop.values * 1e5, marker=MK[k], ms=3.2, lw=1.3, color=COL[k], label=k)
    aB.plot(yrs, 100 * y.values / y.values[0], marker=MK[k], ms=3.2, lw=1.3, color=COL[k], label=k)
aA.set_yscale('log'); aB.set_yscale('log')
aA.set_ylabel('Notification rate per 100,000\npopulation (log scale)')
aB.set_ylabel('Notifications, indexed\n(2001 = 100; log scale)')
aB.axhline(100, color='#777777', lw=0.7, ls='--')
aA.set_title('a  Annual notification rate'); aB.set_title('b  Relative trajectory')
for a in (aA, aB):
    a.set_xticks([2001, 2006, 2011, 2016, 2021]); a.set_xlabel('Year')
    for s in ('top', 'right'): a.spines[s].set_visible(False)
fig.legend(*aA.get_legend_handles_labels(), loc='upper center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.55, 1.0))
rows = ['EHEC', 'Typhoid fever', 'Shigellosis']
for i, k in enumerate(rows[::-1]):
    p, lo, hi = est[k]
    aC.errorbar(p, i, xerr=[[p - lo], [hi - p]], fmt=MK[k], color=COL[k], ms=6, capsize=3, lw=1.4)
    aC.text(14.5, i, (f'{p:+.1f}% per year ({lo:+.1f} to {hi:+.1f})').replace('-', '\u2212'), va='center', fontsize=7.5, color=COL[k])
aC.set_yticks(range(3)); aC.set_yticklabels(rows[::-1])
aC.axvline(0, color='k', lw=0.7); aC.set_xlim(-20, 35); aC.set_ylim(-0.6, 2.6)
aC.set_xlabel('Estimated change in notification rate per year (%, 95% CI; negative-binomial model)')
aC.set_title('c  Long-term trend, 2001–2024')
for s in ('top', 'right'): aC.spines[s].set_visible(False)
fig.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.09)
fig.savefig(os.path.join(OUT, 'Figure_3.tif'), dpi=600, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)
