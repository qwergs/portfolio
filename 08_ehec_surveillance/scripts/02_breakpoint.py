"""Breakpoint analysis: Supplementary Tables S6 and S9, Supplementary Figures S1 and S2.

Continuous piecewise-linear negative-binomial models with 0 to 3 breakpoints are compared by BIC,
with the dispersion fixed at the constant-slope estimate. dBIC = BIC(breakpoint) - BIC(constant
slope); a breakpoint model is accepted only if dBIC < -6. Each segment must span at least h_min
annual observations (h_min = 3 primary; 2 and 4 as sensitivity analyses).

Run: python scripts/02_breakpoint.py            (Table S9 and Figure S2, a few seconds)
     python scripts/02_breakpoint.py --power    (also Table S6 and Figure S1, about 20 minutes)
"""
import os, sys, itertools, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import OUT, SEED, load_annual, nb_trend

warnings.filterwarnings('ignore')
a = load_annual()


def design(n, bps):
    t = np.arange(n, dtype=float)
    return np.column_stack([np.ones(n), t] + [np.maximum(0, t - b) for b in bps])


def bic(y, lp, alpha, bps):
    m = sm.GLM(y, design(len(y), bps), family=sm.families.NegativeBinomial(alpha=alpha), offset=lp).fit()
    return m.bic_llf


def candidates(n, k, h):
    for bps in itertools.combinations(range(h, n - h + 1), k):
        if all(b2 - b1 >= h for b1, b2 in zip(bps, bps[1:])):
            yield bps


def best(y, lp, alpha, k, h):
    out = (np.inf, None)
    for bps in candidates(len(y), k, h):
        try:
            v = bic(y, lp, alpha, bps)
        except Exception:
            continue
        if v < out[0]:
            out = (v, bps)
    return out


# Supplementary Table S9
rows = []
for label, lo in [('2001-2024', 2001), ('2007-2024', 2007)]:
    d = a.loc[lo:2024]
    y, lp = d.ehec_total.values.astype(float), np.log(d.population.values.astype(float))
    alpha = nb_trend(d.ehec_total, d.population)['alpha']
    b0 = bic(y, lp, alpha, ())
    for h in (2, 3, 4):
        for k in (1, 2, 3):
            v, bps = best(y, lp, alpha, k, h)
            rows.append({'series': label, 'h_min': h, 'alpha': round(alpha, 4), 'n_breakpoints': k,
                         'best_years': ', '.join(str(lo + b) for b in bps), 'dBIC': round(v - b0, 2)})
s9 = pd.DataFrame(rows)
s9.to_csv(os.path.join(OUT, 'tableS9_breakpoint_search.csv'), index=False)
print(s9.to_string(index=False))

# Supplementary Figure S2: constant-slope and two-breakpoint (2004, 2007) fits
y, lp = a.ehec_total.values.astype(float), np.log(a.population.values.astype(float))
alpha = nb_trend(a.ehec_total, a.population)['alpha']
fit = lambda bps: sm.GLM(y, design(len(y), bps), family=sm.families.NegativeBinomial(alpha=alpha),
                         offset=lp).fit()
m0, m2 = fit(()), fit((3, 6))
b = m2.params
print('Two-breakpoint segment slopes (%/yr): 2001-04 {:+.1f}, 2004-07 {:+.1f}, 2007-24 {:+.1f}'.format(
    *(100 * np.expm1(v) for v in (b[1], b[1] + b[2], b[1] + b[2] + b[3]))))
yrs = a.index.values
fig, ax = plt.subplots(figsize=(7.9, 4.95), dpi=200)
ax.grid(True, color='#DDDDDD', lw=0.8); ax.set_axisbelow(True)
ax.plot(yrs, m0.fittedvalues, color='#3B4756', lw=2.0, label='Constant slope (2001-2024)')
ax.plot(yrs, m2.fittedvalues, color='#C8551B', lw=2.0, ls='--', label='Two breakpoints (2004, 2007)')
ax.scatter(yrs, y, s=55, color='#BF3B2B', zorder=3, label='Observed annual notifications')
ax.legend(loc='upper left', fontsize=10)
ax.set_xlabel('Year'); ax.set_ylabel('Annual EHEC notifications')
ax.set_xlim(2000, 2025); ax.set_ylim(0, None)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'Figure_S2.png'), dpi=200); plt.close(fig)

# Supplementary Table S6 and Figure S1: power of the single-breakpoint comparison
if '--power' in sys.argv:
    ALPHA_SIM, NSIM = 0.16, 1000
    rng = np.random.default_rng(SEED)
    n = len(y); tt = np.arange(n, dtype=float)
    base = sm.GLM(y, design(n, ()), family=sm.families.NegativeBinomial(alpha=ALPHA_SIM), offset=lp).fit().params

    def detect(mu):
        hits = 0
        for _ in range(NSIM):
            pr = 1 / (1 + ALPHA_SIM * mu)
            yb = rng.negative_binomial(mu * pr / (1 - pr), pr).astype(float)
            v, _ = best(yb, lp, ALPHA_SIM, 1, 3)
            if v - bic(yb, lp, ALPHA_SIM, ()) < -6:
                hits += 1
        return hits / NSIM

    res = []
    for e in [0, 25, 50, 75, 100, 150]:
        mu = np.exp(base[0] + base[1] * tt + np.log1p(e / 100) * np.maximum(0, tt - n // 2) + lp)
        res.append(('slope', e, detect(mu)))
    for k in [1, 1.5, 2, 3, 5]:
        mu = np.exp(base[0] + base[1] * tt + lp); mu[n // 2] *= k
        res.append(('step', k, detect(mu)))
    s6 = pd.DataFrame(res, columns=['scenario', 'effect', 'detection_probability'])
    s6.to_csv(os.path.join(OUT, 'tableS6_power.csv'), index=False)
    print(s6.to_string(index=False))

# Supplementary Figure S1, drawn from Table S6 if it has been produced
s6_path = os.path.join(OUT, 'tableS6_power.csv')
if os.path.exists(s6_path):
    s6 = pd.read_csv(s6_path)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)
    for ax, kind, xl, title in [
            (axs[0], 'slope', 'Post-breakpoint additional slope change (%/year)', 'A. Continuous slope change'),
            (axs[1], 'step', 'Single-year multiplicative step shock (x)', 'B. Single-year step shock')]:
        d = s6[s6.scenario == kind]
        ax.plot(d.effect, 100 * d.detection_probability, marker='o', color='#3B4756')
        ax.axhline(80, ls='--', color='grey', lw=0.8, label='80% power')
        ax.axhline(5, ls=':', color='grey', lw=0.8, label='5% nominal false-positive rate')
        ax.set_xlabel(xl); ax.set_ylabel('Detection probability (%)'); ax.set_ylim(0, 105)
        ax.set_title(title, loc='left', fontweight='bold')
    axs[0].legend(loc='center right', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'Figure_S1.png'), dpi=200); plt.close(fig)
