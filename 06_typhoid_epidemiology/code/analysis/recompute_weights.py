"""Recompute Moran's I across 12 weighting schemes for typhoid R2 (both domestic and total)."""
import numpy as np
import pandas as pd

OUT = '/sessions/epic-determined-feynman/mnt/논문_이건상/장티푸스/FBPD_R2/tables_new'
SRC = '/sessions/epic-determined-feynman/mnt/논문_이건상/장티푸스/reanalysis_2026_FoodbornePathogensDisease/tables/tbl_sido_rate.csv'
RNG = np.random.default_rng(42)
prov = pd.read_csv(SRC)
prov_spatial = prov[prov.sido_en != 'Sejong'].set_index('sido_en')

CENT = {
    'Seoul':(37.5665,126.9780),'Busan':(35.1796,129.0756),'Daegu':(35.8714,128.6014),
    'Incheon':(37.4563,126.7052),'Gwangju':(35.1595,126.8526),'Daejeon':(36.3504,127.3845),
    'Ulsan':(35.5384,129.3114),'Gyeonggi':(37.4138,127.5183),'Gangwon':(37.8228,128.1555),
    'Chungbuk':(36.6357,127.4913),'Chungnam':(36.5184,126.8000),'Jeonbuk':(35.7175,127.1530),
    'Jeonnam':(34.8679,126.9910),'Gyeongbuk':(36.5760,128.5056),'Gyeongnam':(35.4606,128.2132),
    'Jeju':(33.4996,126.5312),
}
QUEEN_ADJ = {
    'Seoul':['Gyeonggi','Incheon'],'Busan':['Gyeongnam','Ulsan'],'Daegu':['Gyeongbuk','Gyeongnam'],
    'Incheon':['Seoul','Gyeonggi'],'Gwangju':['Jeonnam'],'Daejeon':['Chungnam','Chungbuk'],
    'Ulsan':['Busan','Gyeongbuk','Gyeongnam'],'Gyeonggi':['Seoul','Incheon','Gangwon','Chungbuk','Chungnam'],
    'Gangwon':['Gyeonggi','Chungbuk','Gyeongbuk'],'Chungbuk':['Gangwon','Chungnam','Daejeon','Gyeongbuk','Jeonbuk','Gyeonggi'],
    'Chungnam':['Gyeonggi','Chungbuk','Daejeon','Jeonbuk'],'Jeonbuk':['Chungnam','Chungbuk','Gyeongbuk','Gyeongnam','Jeonnam'],
    'Jeonnam':['Jeonbuk','Gyeongnam','Gwangju'],'Gyeongbuk':['Gangwon','Chungbuk','Jeonbuk','Gyeongnam','Daegu','Ulsan'],
    'Gyeongnam':['Jeonnam','Jeonbuk','Gyeongbuk','Daegu','Busan','Ulsan'],'Jeju':[],
}
names = list(QUEEN_ADJ.keys())
prov_spatial = prov_spatial.reindex(names)
n = len(names); idx_map = {nm:i for i,nm in enumerate(names)}

R = 6371.0
def hav(la1, lo1, la2, lo2):
    la1,lo1,la2,lo2 = map(np.radians, [la1,lo1,la2,lo2])
    a = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
D = np.zeros((n,n))
for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        if i != j:
            D[i,j] = hav(*CENT[ni], *CENT[nj])

def W_queen():
    W = np.zeros((n,n))
    for a, nb in QUEEN_ADJ.items():
        for b in nb:
            W[idx_map[a], idx_map[b]] = 1.0
    return W

def W_queen_knn2():
    W = W_queen()
    j = idx_map['Jeju']
    d = D[j].copy(); d[j] = np.inf
    for k in np.argsort(d)[:2]:
        W[j, k] = 1.0; W[k, j] = 1.0
    return np.maximum(W, W.T)

def W_knn(k):
    W = np.zeros((n,n))
    for i in range(n):
        d = D[i].copy(); d[i] = np.inf
        nbrs = np.argsort(d)[:k]
        for nb in nbrs:
            W[i, nb] = 1.0
    return np.maximum(W, W.T)

def W_dist_band(km):
    W = (D <= km).astype(float)
    np.fill_diagonal(W, 0)
    return W

def W_inv_dist(power=1.0):
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                W[i,j] = 1.0/(D[i,j]**power)
    return W

def row_std(W):
    rs = W.sum(axis=1, keepdims=True); rs[rs==0]=1.0
    return W/rs

def morans_I(x, W):
    z = (x - x.mean())/x.std(ddof=0)
    return float((z*(W@z)).sum() / (z*z).sum())

def perm_p(x, W, n_perm=9999, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    obs = morans_I(x, W); null = []
    for _ in range(n_perm):
        x2 = rng.permutation(x); null.append(morans_I(x2, W))
    null = np.asarray(null)
    p = (np.abs(null) >= np.abs(obs)).mean()
    return obs, max(p, 1.0/(n_perm+1))

schemes = {
    'Queen_strict':       row_std(W_queen()),
    'Queen+KNN2_baseline':row_std(W_queen_knn2()),
    'KNN1':               row_std(W_knn(1)),
    'KNN2':               row_std(W_knn(2)),
    'KNN3':               row_std(W_knn(3)),
    'KNN4':               row_std(W_knn(4)),
    'KNN5':               row_std(W_knn(5)),
    'DistBand_100km':     row_std(W_dist_band(100)),
    'DistBand_150km':     row_std(W_dist_band(150)),
    'DistBand_200km':     row_std(W_dist_band(200)),
    'DistBand_300km':     row_std(W_dist_band(300)),
    'InvDist_p1':         row_std(W_inv_dist(1.0)),
}

x_total = prov_spatial['mean_annual_rate'].values
x_dom = prov_spatial['dom_rate'].values
rows=[]
for name, W in schemes.items():
    rng_t = np.random.default_rng(42)
    rng_d = np.random.default_rng(42)
    I_t, p_t = perm_p(x_total, W, 9999, rng_t)
    I_d, p_d = perm_p(x_dom, W, 9999, rng_d)
    rows.append({'scheme':name,
                 'I_domestic':round(I_d,4),'p_perm_domestic':round(p_d,5),
                 'I_total':round(I_t,4),'p_perm_total':round(p_t,5)})

df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/tbl_weights_sensitivity.csv', index=False)
print(df.to_string(index=False))
