"""Analysis 5 — Provincial Moran's I + LISA (sido level, baseline weights)
+ Inputs go to analysis 9 for sensitivity over weights schemes.

Sejong is dropped from spatial analysis (population missing in original
denominator file; cumulative cases small at 588). 16 sido analysed.

Output:
  tbl_morans_main.csv  — 4 rows: (Queen+KNN2, Strict_Queen) × (df1, df2)
  tbl_lisa.csv         — 16 rows: per-sido LISA Iᵢ + permutation p
"""
import numpy as np, pandas as pd

OUT = '/sessions/elegant-quirky-hamilton/mnt/outputs/hepa_work'

sr = pd.read_csv(f'{OUT}/tbl_sido_rate.csv')
sr = sr[sr.sido_en != 'Sejong'].reset_index(drop=True)  # 16 sido

# Province centroids (decimal degrees)
CENT = {
    'Seoul':(37.5665,126.9780), 'Busan':(35.1796,129.0756),
    'Daegu':(35.8714,128.6014), 'Incheon':(37.4563,126.7052),
    'Gwangju':(35.1595,126.8526), 'Daejeon':(36.3504,127.3845),
    'Ulsan':(35.5384,129.3114),  'Gyeonggi':(37.4138,127.5183),
    'Gangwon':(37.8228,128.1555), 'Chungbuk':(36.6357,127.4913),
    'Chungnam':(36.5184,126.8000), 'Jeonbuk':(35.7175,127.1530),
    'Jeonnam':(34.8679,126.9910), 'Gyeongbuk':(36.5760,128.5056),
    'Gyeongnam':(35.4606,128.2132), 'Jeju':(33.4996,126.5312),
}
QUEEN_ADJ = {
    'Seoul':['Gyeonggi','Incheon'],
    'Busan':['Gyeongnam','Ulsan'],
    'Daegu':['Gyeongbuk','Gyeongnam'],
    'Incheon':['Seoul','Gyeonggi'],
    'Gwangju':['Jeonnam'],
    'Daejeon':['Chungnam','Chungbuk'],
    'Ulsan':['Busan','Gyeongbuk','Gyeongnam'],
    'Gyeonggi':['Seoul','Incheon','Gangwon','Chungbuk','Chungnam'],
    'Gangwon':['Gyeonggi','Chungbuk','Gyeongbuk'],
    'Chungbuk':['Gangwon','Chungnam','Daejeon','Gyeongbuk','Jeonbuk','Gyeonggi'],
    'Chungnam':['Gyeonggi','Chungbuk','Daejeon','Jeonbuk'],
    'Jeonbuk':['Chungnam','Chungbuk','Gyeongbuk','Gyeongnam','Jeonnam'],
    'Jeonnam':['Jeonbuk','Gyeongnam','Gwangju'],
    'Gyeongbuk':['Gangwon','Chungbuk','Jeonbuk','Gyeongnam','Daegu','Ulsan'],
    'Gyeongnam':['Jeonnam','Jeonbuk','Gyeongbuk','Daegu','Busan','Ulsan'],
    'Jeju':[],
}
names = list(QUEEN_ADJ.keys())  # 16 sido in fixed order
sr = sr.set_index('sido_en').reindex(names).reset_index()
n = len(sr)
idx = {nm:i for i, nm in enumerate(names)}
sr['lat'] = sr.sido_en.map(lambda x: CENT[x][0])
sr['lon'] = sr.sido_en.map(lambda x: CENT[x][1])

R = 6371.0
def hav(lat1, lon1, lat2, lon2):
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r; dlon = lon2r - lon1r
    a = np.sin(dlat/2)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

D = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if i != j:
            D[i,j] = hav(sr.lat.iat[i], sr.lon.iat[i], sr.lat.iat[j], sr.lon.iat[j])

def W_queen(jeju_knn=0):
    W = np.zeros((n,n))
    for a, nb in QUEEN_ADJ.items():
        for b in nb:
            W[idx[a], idx[b]] = 1.0
    if jeju_knn > 0:
        j = idx['Jeju']
        d = D[j].copy(); d[j] = np.inf
        knn = np.argsort(d)[:jeju_knn]
        for k in knn:
            W[j, k] = 1.0; W[k, j] = 1.0
    return np.maximum(W, W.T)

def row_std(W):
    rs = W.sum(axis=1, keepdims=True); rs[rs==0]=1
    return W / rs

def morans_I(x, W, nperm=9999, seed=42):
    n_ = len(x); z = x - x.mean()
    S0 = W.sum()
    num = (W * np.outer(z, z)).sum()
    den = (z**2).sum()
    I = (n_/S0) * num / den
    EI = -1.0/(n_-1)
    rng = np.random.default_rng(seed)
    perm = np.zeros(nperm)
    for s in range(nperm):
        zp = rng.permutation(z)
        perm[s] = (n_/S0) * (W * np.outer(zp, zp)).sum() / den
    p = (np.sum(np.abs(perm) >= np.abs(I)) + 1) / (nperm + 1)
    return I, EI, p

# ─── Global Moran's I across (df1=domestic, df2=total) × (Queen+KNN2, Strict_Queen)
WQ_KNN  = row_std(W_queen(2))
WQ_STR  = row_std(W_queen(0))
WQ_STR[idx['Jeju'], :] = 0  # truly isolated

rates_dom = sr['dom_rate'].values
rates_all = sr['mean_annual_rate'].values

print(f'Sido count: {n}; mean dom rate = {rates_dom.mean():.2f}, all = {rates_all.mean():.2f}')

rows = []
for sch_name, W in [('Queen+KNN2_baseline', WQ_KNN), ('Strict_Queen', WQ_STR)]:
    for series, x in [('domestic_df1', rates_dom), ('total_df2', rates_all)]:
        I, EI, p = morans_I(x, W, nperm=9999, seed=42)
        rows.append({'scheme':sch_name, 'series':series, 'I':I, 'EI':EI, 'p_perm':p})
mr = pd.DataFrame(rows)
mr.to_csv(f'{OUT}/tbl_morans_main.csv', index=False)
print('\n=== Global Moran\'s I (9,999 permutations) ===')
print(mr.to_string(index=False))

# ─── LISA (Local Moran's I) — baseline Queen+KNN2 ──────────────────────
def lisa(x, W, nperm=999, seed=42):
    z = x - x.mean()
    den = (z**2).sum() / len(x)  # variance estimate
    Ii = z * (W @ z) / den
    rng = np.random.default_rng(seed)
    perm_p = np.zeros(len(x))
    for k in range(len(x)):
        # conditional permutation on observation k
        others = [m for m in range(len(x)) if m != k]
        cnt = 0
        for s in range(nperm):
            shuffled = z.copy()
            shuf = rng.permutation(z[others])
            shuffled[others] = shuf
            Ii_perm = shuffled[k] * (W[k] @ shuffled) / den
            if abs(Ii_perm) >= abs(Ii[k]): cnt += 1
        perm_p[k] = (cnt + 1) / (nperm + 1)
    return Ii, perm_p

W = WQ_KNN
mean_rate = rates_all.mean()
Ii, p_lisa = lisa(rates_all, W)
lisa_rows = []
for i, name in enumerate(names):
    rate_i = rates_all[i]
    z_i = rate_i - mean_rate
    lag_i = (W[i] @ rates_all) - mean_rate * W[i].sum()
    if z_i > 0 and lag_i > 0: q = 'HH'
    elif z_i > 0 and lag_i < 0: q = 'HL'
    elif z_i < 0 and lag_i > 0: q = 'LH'
    else: q = 'LL'
    lisa_rows.append({'sido_en': name,
                      'rate_per_100k': rate_i,
                      'lag_rate': lag_i + mean_rate * W[i].sum() / max(W[i].sum(),1),
                      'I_local': Ii[i],
                      'p_perm': p_lisa[i],
                      'quadrant': q,
                      'sig_5pct': p_lisa[i] < 0.05})
ld = pd.DataFrame(lisa_rows).sort_values('p_perm').reset_index(drop=True)
ld.to_csv(f'{OUT}/tbl_lisa.csv', index=False)
print('\n=== LISA (baseline Queen+KNN2, 999 perm) ===')
print(ld.to_string(index=False))
