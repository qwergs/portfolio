"""Spatial analysis: Supplementary Tables S1, S2 (Panel B) and S7, and the 2004 exclusion check.

Provincial rates are cumulative notifications divided by (mean annual population x 24 years),
from data/provincial_totals.csv (Supplementary Table S2, Panel A; Sejong excluded). Empirical
Bayes smoothing uses the domestic municipal series in data/ehec_yearly_sigungu.csv, the only
extract available at municipal level.

Permutation p values are Monte Carlo estimates and are reproducible with seed 42; the I statistics
are exact.

Run: python scripts/03_spatial.py
"""
import os
import numpy as np
import pandas as pd
from common import DATA, OUT

SEED = 42
PERM_MAIN, PERM_ALT = 9999, 999
R_EARTH = 6371.0

# Provincial capital coordinates (latitude, longitude)
CENT = {
    'Seoul': (37.5665, 126.9780), 'Busan': (35.1796, 129.0756), 'Daegu': (35.8714, 128.6014),
    'Incheon': (37.4563, 126.7052), 'Gwangju': (35.1595, 126.8526), 'Daejeon': (36.3504, 127.3845),
    'Ulsan': (35.5384, 129.3114), 'Gyeonggi': (37.4138, 127.5183), 'Gangwon': (37.8228, 128.1555),
    'Chungbuk': (36.6357, 127.4913), 'Chungnam': (36.5184, 126.8000), 'Jeonbuk': (35.7175, 127.1530),
    'Jeonnam': (34.8679, 126.9910), 'Gyeongbuk': (36.5760, 128.5056), 'Gyeongnam': (35.4606, 128.2132),
    'Jeju': (33.4996, 126.5312),
}
# Land contiguity (Jeju has none)
QUEEN_ADJ = {
    'Seoul': ['Gyeonggi', 'Incheon'], 'Busan': ['Gyeongnam', 'Ulsan'],
    'Daegu': ['Gyeongbuk', 'Gyeongnam'], 'Incheon': ['Seoul', 'Gyeonggi'],
    'Gwangju': ['Jeonnam'], 'Daejeon': ['Chungnam', 'Chungbuk'],
    'Ulsan': ['Busan', 'Gyeongbuk', 'Gyeongnam'],
    'Gyeonggi': ['Seoul', 'Incheon', 'Gangwon', 'Chungbuk', 'Chungnam'],
    'Gangwon': ['Gyeonggi', 'Chungbuk', 'Gyeongbuk'],
    'Chungbuk': ['Gangwon', 'Chungnam', 'Daejeon', 'Gyeongbuk', 'Jeonbuk', 'Gyeonggi'],
    'Chungnam': ['Gyeonggi', 'Chungbuk', 'Daejeon', 'Jeonbuk'],
    'Jeonbuk': ['Chungnam', 'Chungbuk', 'Gyeongbuk', 'Gyeongnam', 'Jeonnam'],
    'Jeonnam': ['Jeonbuk', 'Gyeongnam', 'Gwangju'],
    'Gyeongbuk': ['Gangwon', 'Chungbuk', 'Jeonbuk', 'Gyeongnam', 'Daegu', 'Ulsan'],
    'Gyeongnam': ['Jeonnam', 'Jeonbuk', 'Gyeongbuk', 'Daegu', 'Busan', 'Ulsan'],
    'Jeju': [],
}
SIDO_PREFIX = {
    '서울': 'Seoul', '부산': 'Busan', '대구': 'Daegu', '인천': 'Incheon', '광주': 'Gwangju',
    '대전': 'Daejeon', '울산': 'Ulsan', '세종': 'Sejong', '경기': 'Gyeonggi', '강원': 'Gangwon',
    '충청북': 'Chungbuk', '충북': 'Chungbuk', '충청남': 'Chungnam', '충남': 'Chungnam',
    '전라북': 'Jeonbuk', '전북': 'Jeonbuk', '전라남': 'Jeonnam', '전남': 'Jeonnam',
    '경상북': 'Gyeongbuk', '경북': 'Gyeongbuk', '경상남': 'Gyeongnam', '경남': 'Gyeongnam',
    '제주': 'Jeju',
}
NAMES = list(QUEEN_ADJ)
N = len(NAMES)
IDX = {m: i for i, m in enumerate(NAMES)}


def to_sido(region):
    for k in sorted(SIDO_PREFIX, key=len, reverse=True):
        if region.startswith(k):
            return SIDO_PREFIX[k]
    raise ValueError(region)


def banner(t):
    print('\n' + '=' * 74 + f'\n{t}\n' + '=' * 74)


def haversine(a, b):
    la1, lo1 = np.radians(CENT[a]); la2, lo2 = np.radians(CENT[b])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(h))


DIST = np.array([[0.0 if a == b else haversine(a, b) for b in NAMES] for a in NAMES])


def row_standardise(W):
    s = W.sum(axis=1, keepdims=True); s[s == 0] = 1
    return W / s


def w_queen(knn_jeju=2, binary=False):
    W = np.zeros((N, N))
    for a, nbs in QUEEN_ADJ.items():
        for b in nbs:
            W[IDX[a], IDX[b]] = 1.0
    if knn_jeju:
        j = IDX['Jeju']; d = DIST[j].copy(); d[j] = np.inf
        for k in np.argsort(d)[:knn_jeju]:
            W[j, k] = W[k, j] = 1.0
    W = np.maximum(W, W.T)
    return W if binary else row_standardise(W)


def w_knn(k):
    W = np.zeros((N, N))
    for i in range(N):
        d = DIST[i].copy(); d[i] = np.inf
        for j in np.argsort(d)[:k]:
            W[i, j] = 1.0
    return row_standardise(W)


def w_band(km):
    return row_standardise(((DIST > 0) & (DIST <= km)).astype(float))


SCHEMES = [
    ('Queen (Jeju isolated, row-std)', lambda: w_queen(0, False)),
    ('Queen (Jeju isolated, binary)',  lambda: w_queen(0, True)),
    ('Queen + KNN-2 Jeju (row-std)',   lambda: w_queen(2, False)),
    ('Queen + KNN-2 Jeju (binary)',    lambda: w_queen(2, True)),
    ('KNN k=1 (row-std)',              lambda: w_knn(1)),
    ('KNN k=3 (row-std)',              lambda: w_knn(3)),
    ('KNN k=4 (row-std)',              lambda: w_knn(4)),
    ('KNN k=5 (row-std)',              lambda: w_knn(5)),
    ('Distance band 100km (row-std)',  lambda: w_band(100)),
    ('Distance band 150km (row-std)',  lambda: w_band(150)),
    ('Distance band 200km (row-std)',  lambda: w_band(200)),
    ('Distance band 300km (row-std)',  lambda: w_band(300)),
]


def morans_I(x, W, nperm=PERM_MAIN, seed=SEED):
    """Two-sided permutation test (|I_perm| >= |I_obs|)."""
    z = x - x.mean(); S0 = W.sum()
    I = (N / S0) * (W * np.outer(z, z)).sum() / (z ** 2).sum()
    rng = np.random.default_rng(seed)
    perm = np.empty(nperm)
    for s in range(nperm):
        zp = rng.permutation(z)
        perm[s] = (N / S0) * (W * np.outer(zp, zp)).sum() / (z ** 2).sum()
    return I, ((np.abs(perm) >= abs(I)).sum() + 1) / (nperm + 1)


def local_morans(x, W, nperm=PERM_ALT, seed=SEED):
    """Conditional-permutation LISA. Returns (I_i, p_i, quadrant)."""
    z = x - x.mean(); m2 = (z ** 2).sum() / N
    Ii = z / m2 * (W @ z)
    lag = W @ z
    quad = np.where((z > 0) & (lag > 0), 'HH',
           np.where((z < 0) & (lag < 0), 'LL',
           np.where((z > 0) & (lag < 0), 'HL', 'LH')))
    rng = np.random.default_rng(seed)
    p = np.empty(N)
    for i in range(N):
        k = int((W[i] > 0).sum())
        others = np.delete(z, i)
        sim = np.empty(nperm)
        wi = W[i][W[i] > 0]
        for s in range(nperm):
            pick = rng.choice(others, size=k, replace=False)
            sim[s] = z[i] / m2 * float(wi @ pick)
        p[i] = ((np.abs(sim) >= abs(Ii[i])).sum() + 1) / (nperm + 1)
    return Ii, p, quad


def bh_fdr(p):
    n = len(p); order = np.argsort(p)[::-1]
    q = np.empty(n); prev = 1.0
    for rank, idx in enumerate(order):
        k = n - rank
        prev = min(prev, p[idx] * n / k)
        q[idx] = prev
    return q




prov = pd.read_csv(os.path.join(DATA, 'provincial_totals.csv')).set_index('province').reindex(NAMES)
rate = {k: (prov[k] / (prov.mean_population * 24) * 1e5).values for k in ('total', 'domestic')}
W0 = w_queen(2, False)

# Global Moran's I, baseline weights
for k in ('total', 'domestic'):
    I, p = morans_I(rate[k], W0)
    print(f"Moran's I ({k}) = {I:.3f}, p = {p:.4f}")

# Supplementary Table S1: 12 weighting schemes
rows = []
for name, fn in SCHEMES:
    W = fn()
    Id, pd_ = morans_I(rate['domestic'], W, nperm=PERM_ALT)
    It, pt = morans_I(rate['total'], W, nperm=PERM_ALT)
    rows.append({'scheme': name, 'I_domestic': round(Id, 3), 'p_domestic': round(pd_, 4),
                 'I_total': round(It, 3), 'p_total': round(pt, 4)})
s1 = pd.DataFrame(rows); s1.to_csv(os.path.join(OUT, 'tableS1_weight_schemes.csv'), index=False)
print(s1.to_string(index=False)); print('significant (total, p < 0.05):', int((s1.p_total < 0.05).sum()), 'of 12')

# Supplementary Table S2, Panel B: Marshall Empirical Bayes smoothing of municipal rates
df = pd.read_csv(os.path.join(DATA, 'ehec_yearly_sigungu.csv'))
df['sido'] = df.region.map(to_sido)
sg = df.groupby('region').agg(cases=('yearly_cases', 'sum'), pop=('pop', 'sum')).reset_index()
sg['sido'] = sg.region.map(to_sido)
n_i, x_i = sg['pop'].values, sg.cases.values
r_i = x_i / n_i; m = x_i.sum() / n_i.sum()
s2 = np.sum(n_i * (r_i - m) ** 2) / n_i.sum()
A = max(s2 - m / n_i.mean(), 0.0)
sg['raw_rate'] = r_i * 1e5
sg['eb_rate'] = (m + A / (A + m / n_i) * (r_i - m)) * 1e5
eb = sg.groupby('sido').apply(lambda d: pd.Series({
    'n_sigungu': len(d), 'cases': d.cases.sum(), 'raw_rate': d.cases.sum() / d['pop'].sum() * 1e5,
    'eb_rate': np.average(d.eb_rate, weights=d['pop'])}), include_groups=False)
eb.round(3).to_csv(os.path.join(OUT, 'tableS2_panelB_eb.csv'))
I_eb, p_eb = morans_I(eb.reindex(NAMES).eb_rate.values, W0)
print(f"EB-smoothed Moran's I = {I_eb:.3f}, p = {p_eb:.4f}")

# Supplementary Table S7: LISA with Benjamini-Hochberg adjustment
Ii, pi, quad = local_morans(rate['total'], W0)
s7 = pd.DataFrame({'province': NAMES, 'rate_per_100000': np.round(rate['total'], 3), 'local_I': np.round(Ii, 3),
                   'quadrant': quad, 'p_raw': np.round(pi, 3), 'q_bh': np.round(bh_fdr(pi), 3)})
s7 = s7.sort_values('p_raw'); s7.to_csv(os.path.join(OUT, 'tableS7_lisa.csv'), index=False)
print(s7.to_string(index=False))

# Check: provincial pattern without 2004 (domestic mean annual rates)
d = df[df.year != 2004]
ann = d.groupby(['sido', 'year']).agg(c=('yearly_cases', 'sum'), p=('pop', 'sum'))
x = (ann.c / ann.p * 1e5).groupby(level=0).mean().reindex(NAMES)
I, p = morans_I(x.values, W0)
print(f"Excluding 2004: Gwangju mean annual domestic rate {x['Gwangju']:.2f}; Moran's I = {I:.2f}, p = {p:.4f}")
