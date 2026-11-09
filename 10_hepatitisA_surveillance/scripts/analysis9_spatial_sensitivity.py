"""Analysis 9 — spatial weights sensitivity (12 schemes) + provincial table.

Output:
  tbl_weights_sensitivity.csv  — 12 schemes × {domestic, total} Moran's I
  tbl_provincial_rates.csv     — full provincial rates with imported %
"""
import numpy as np, pandas as pd

OUT = '/sessions/elegant-quirky-hamilton/mnt/outputs/hepa_work'
sr = pd.read_csv(f'{OUT}/tbl_sido_rate.csv')
sr_full = sr.copy()
sr = sr[sr.sido_en != 'Sejong'].reset_index(drop=True)

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
    'Seoul':['Gyeonggi','Incheon'], 'Busan':['Gyeongnam','Ulsan'],
    'Daegu':['Gyeongbuk','Gyeongnam'], 'Incheon':['Seoul','Gyeonggi'],
    'Gwangju':['Jeonnam'], 'Daejeon':['Chungnam','Chungbuk'],
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
names = list(QUEEN_ADJ.keys())
sr = sr.set_index('sido_en').reindex(names).reset_index()
n = len(sr); idx = {nm:i for i, nm in enumerate(names)}
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
        for b in nb: W[idx[a], idx[b]] = 1.0
    if jeju_knn > 0:
        j = idx['Jeju']
        d = D[j].copy(); d[j] = np.inf
        knn = np.argsort(d)[:jeju_knn]
        for k in knn: W[j, k] = 1.0; W[k, j] = 1.0
    return np.maximum(W, W.T)

def W_knn(k):
    W = np.zeros((n,n))
    for i in range(n):
        d = D[i].copy(); d[i] = np.inf
        nb = np.argsort(d)[:k]
        for j in nb: W[i, j] = 1.0
    return W

def W_dist(b):
    W = (D <= b).astype(float)
    np.fill_diagonal(W, 0)
    return W

def W_invd(threshold=500):
    W = np.where((D > 0) & (D <= threshold), 1.0/D.clip(min=1e-9), 0.0)
    np.fill_diagonal(W, 0)
    return W

def row_std(W):
    rs = W.sum(axis=1, keepdims=True); rs[rs==0]=1
    return W / rs

def morans_I(x, W, nperm=9999, seed=42):
    n_ = len(x); z = x - x.mean()
    S0 = W.sum()
    if S0 == 0: return np.nan, np.nan, np.nan
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

rates_dom = sr['dom_rate'].values
rates_all = sr['mean_annual_rate'].values

schemes = [
    ('Queen + KNN-2 Jeju (row-std)', row_std(W_queen(2))),
    ('Strict Queen (Jeju isolated, row-std)', row_std(W_queen(0))),
    ('KNN k=1 (row-std)', row_std(W_knn(1))),
    ('KNN k=2 (row-std)', row_std(W_knn(2))),
    ('KNN k=3 (row-std)', row_std(W_knn(3))),
    ('KNN k=4 (row-std)', row_std(W_knn(4))),
    ('KNN k=5 (row-std)', row_std(W_knn(5))),
    ('Distance band 100km (row-std)', row_std(W_dist(100))),
    ('Distance band 150km (row-std)', row_std(W_dist(150))),
    ('Distance band 200km (row-std)', row_std(W_dist(200))),
    ('Distance band 300km (row-std)', row_std(W_dist(300))),
    ('Inverse distance ≤500km (row-std)', row_std(W_invd(500))),
]
rows = []
for name, W in schemes:
    Id, _, pd_ = morans_I(rates_dom, W, nperm=9999, seed=42)
    Ia, _, pa_ = morans_I(rates_all, W, nperm=9999, seed=42)
    rows.append({'scheme':name, 'I_domestic':Id, 'p_domestic':pd_,
                 'I_total':Ia, 'p_total':pa_})

res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/tbl_weights_sensitivity.csv', index=False)
print('=== Spatial weights sensitivity (Moran\'s I, 9,999 perms) ===')
print(res.to_string(index=False))
print(f'\nSchemes significant (p<0.05): {(res.p_total < 0.05).sum()} / {len(res)}')

# Provincial rates table (full 17 sido including Sejong)
prov = sr_full.copy()
prov = prov[['sido_en','total_cases','dom_cases','imp_cases','mean_pop',
             'mean_annual_rate','dom_rate','imported_pct']]
prov.to_csv(f'{OUT}/tbl_provincial_rates.csv', index=False)
print(f'\ntbl_provincial_rates.csv: {len(prov)} sido')
print(prov.to_string(index=False))
