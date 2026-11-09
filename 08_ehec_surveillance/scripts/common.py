"""Shared helpers for the EHEC notification analyses."""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'output')
os.makedirs(OUT, exist_ok=True)
SEED = 42


def load_annual():
    return pd.read_csv(os.path.join(DATA, 'annual_national_series.csv'), index_col='year')


def nb_trend(cases, population):
    """NB2 regression of annual counts on calendar year with a log-population offset.

    Returns the percentage change per year with its Wald 95% CI, the dispersion alpha,
    and the fitted model.
    """
    y = np.asarray(cases, float)
    X = sm.add_constant(np.arange(len(y), dtype=float))
    m = sm.NegativeBinomial(y, X, offset=np.log(np.asarray(population, float))).fit(disp=0, maxiter=500)
    b, se = m.params[1], m.bse[1]
    pct = lambda v: 100 * np.expm1(v)
    return {'pct': pct(b), 'lo': pct(b - 1.96 * se), 'hi': pct(b + 1.96 * se),
            'alpha': m.params[-1], 'model': m}


def fmt_ci(r):
    return f"{r['pct']:+.1f} ({r['lo']:+.1f}, {r['hi']:+.1f})"
