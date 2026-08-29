"""Typhoid R2 part2: step sim, EB, LISA FDR, period spatial, ASR, age-specific."""
import numpy as np, pandas as pd, statsmodels.api as sm
import pymannkendall as mk
from statsmodels.stats.multitest import multipletests

WORK = '/sessions/serene-confident-edison/mnt/outputs/typhoid_work'
RNG = np.random.default_rng(42)
ann = pd.read_csv(f'{WORK}/tbl_annual_rates.csv').sort_values('year').reset_index(drop=True)
yearly_sigungu = pd.read_csv(f'{WORK}/typhoid_yearly_sigungu.csv')
df2 = pd.read_csv(f'{WORK}/df2_long.csv')

nbd = pd.read_csv(f'{WORK}/tbl_nb_dispersion.csv')
ALPHA = float(nbd[nbd.series == 'total'].alpha_hat.iloc[0])
mu_baseline = ann['total'].mean()

# Step sim only
def nb_sample(rng, mu, a, size):
    p_ = 1/(1+a*mu); n_p = mu*p_/(1-p_)
    return rng.negative_binomial(n_p, p_, size=size)
def fit_bp_power(y, n_bp, hmin=3):
    n_l = len(y); t_l = np.arange(n_l)
    X0 = sm.add_constant(t_l)
    if n_bp == 0:
        try: return sm.GLM(y, X0, family=sm.families.NegativeBinomial(alpha=1.0)).fit(maxiter=200).bic_llf
        except: return np.inf
    if n_bp == 1:
        best = np.inf
        for bp in range(hmin, n_l-hmin+1):
            X = np.column_stack([X0, np.maximum(0, t_l-bp)])
            try:
                m = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0)).fit(maxiter=200)
                if m.bic_llf < best: best = m.bic_llf
            except: pass
        return best
    return np.inf

print("Step-change power sim...")
N_SIM = 300
step_mults = [1.0, 2.0, 3.0, 5.0]
step_rows = []
for m in step_mults:
    n_d = 0
    for _ in range(N_SIM):
        mu = np.full(24, mu_baseline); mu[12] = mu[12]*m
        y = nb_sample(RNG, np.maximum(mu,1), ALPHA, 24)
        b0 = fit_bp_power(y, 0); b1 = fit_bp_power(y, 1)
        if (b1-b0) < -6: n_d += 1
    step_rows.append({'step_multiplier': m, 'detection_prob': n_d/N_SIM, 'n_sim': N_SIM, 'alpha': ALPHA})
    print(f"  step {m}x: power={n_d/N_SIM:.3f}")
pd.DataFrame(step_rows).to_csv(f'{WORK}/tbl_power_sim_step.csv', index=False)

# EB
print("\nEB shrinkage...")
SIDO_PREFIX = {
    '서울특별시':'Seoul','부산광역시':'Busan','대구광역시':'Daegu','인천광역시':'Incheon',
    '광주광역시':'Gwangju','대전광역시':'Daejeon','울산광역시':'Ulsan','세종특별자치시':'Sejong',
    '서울시':'Seoul','부산시':'Busan','대구시':'Daegu','인천시':'Incheon',
    '광주시':'Gwangju','대전시':'Daejeon','울산시':'Ulsan','세종시':'Sejong',
    '경기도':'Gyeonggi','강원도':'Gangwon','강원특별자치도':'Gangwon',
    '충청북도':'Chungbuk','충청남도':'Chungnam','전라북도':'Jeonbuk','전북특별자치도':'Jeonbuk',
    '전라남도':'Jeonnam','경상북도':'Gyeongbuk','경상남도':'Gyeongnam',
    '제주특별자치도':'Jeju','제주도':'Jeju'}
def get_sido(r):
    for k, v in SIDO_PREFIX.items():
        if r.startswith(k): return v
    return 'Unknown'
ys = yearly_sigungu.dropna(subset=['pop'])
cum = ys.groupby('region').agg(total_cases=('yearly_cases','sum'), mean_pop=('pop','mean')).reset_index()
cum['raw_rate_per_100k'] = cum['total_cases']/(cum['mean_pop']*24)*1e5
cum['sido_en'] = cum['region'].apply(get_sido)
pop_yrs = cum['mean_pop']*24
raw = cum['total_cases']/pop_yrs
g = cum['total_cases'].sum()/pop_yrs.sum()
s2 = max(raw.var(ddof=0) - raw.mean()/pop_yrs.mean(), 1e-12)
Bi = s2/(s2 + raw.mean()/pop_yrs)
cum['eb_rate_per_100k'] = (Bi*raw + (1-Bi)*g)*1e5
cum['eb_weight_Bi'] = Bi
cum.to_csv(f'{WORK}/tbl_eb_smoothed_sigungu.csv', index=False)

sido_eb = cum.groupby('sido_en').apply(lambda g_: (g_['eb_rate_per_100k']*g_['mean_pop']).sum()/g_['mean_pop'].sum() if g_['mean_pop'].sum()>0 else np.nan)
sido_raw = cum.groupby('sido_en').apply(lambda g_: g_['total_cases'].sum()/(g_['mean_pop'].sum()*24)*1e5 if g_['mean_pop'].sum()>0 else np.nan)
sido_n = cum.groupby('sido_en').size()
sido_tot = cum.groupby('sido_en')['total_cases'].sum()
sido_df = pd.DataFrame({'sido_en':sido_raw.index, 'n_sigungu':sido_n.values, 'total_cases':sido_tot.values,
                        'raw_rate':sido_raw.values, 'eb_rate':sido_eb.values})
sido_df.to_csv(f'{WORK}/tbl_eb_smoothed_sido.csv', index=False)

# Moran's I (raw vs EB)
CENT = {'Seoul':(37.5665,126.9780),'Busan':(35.1796,129.0756),'Daegu':(35.8714,128.6014),
    'Incheon':(37.4563,126.7052),'Gwangju':(35.1595,126.8526),'Daejeon':(36.3504,127.3845),
    'Ulsan':(35.5384,129.3114),'Gyeonggi':(37.4138,127.5183),'Gangwon':(37.8228,128.1555),
    'Chungbuk':(36.6357,127.4913),'Chungnam':(36.5184,126.8000),'Jeonbuk':(35.7175,127.1530),
    'Jeonnam':(34.8679,126.9910),'Gyeongbuk':(36.5760,128.5056),'Gyeongnam':(35.4606,128.2132),
    'Jeju':(33.4996,126.5312)}
QUEEN_ADJ = {'Seoul':['Gyeonggi','Incheon'],'Busan':['Gyeongnam','Ulsan'],
    'Daegu':['Gyeongbuk','Gyeongnam'],'Incheon':['Seoul','Gyeonggi'],
    'Gwangju':['Jeonnam'],'Daejeon':['Chungnam','Chungbuk'],
    'Ulsan':['Busan','Gyeongbuk','Gyeongnam'],
    'Gyeonggi':['Seoul','Incheon','Gangwon','Chungbuk','Chungnam'],
    'Gangwon':['Gyeonggi','Chungbuk','Gyeongbuk'],
    'Chungbuk':['Gangwon','Chungnam','Daejeon','Gyeongbuk','Jeonbuk','Gyeonggi'],
    'Chungnam':['Gyeonggi','Chungbuk','Daejeon','Jeonbuk'],
    'Jeonbuk':['Chungnam','Chungbuk','Gyeongbuk','Gyeongnam','Jeonnam'],
    'Jeonnam':['Jeonbuk','Gyeongnam','Gwangju'],
    'Gyeongbuk':['Gangwon','Chungbuk','Jeonbuk','Gyeongnam','Daegu','Ulsan'],
    'Gyeongnam':['Jeonnam','Jeonbuk','Gyeongbuk','Daegu','Busan','Ulsan'],
    'Jeju':[]}
names = list(QUEEN_ADJ.keys()); n_s = len(names); idx = {nm:i for i,nm in enumerate(names)}
R = 6371.0
def hav(la1, lo1, la2, lo2):
    la1r, lo1r, la2r, lo2r = map(np.radians, [la1, lo1, la2, lo2])
    dlat = la2r-la1r; dlon = lo2r-lo1r
    a = np.sin(dlat/2)**2+np.cos(la1r)*np.cos(la2r)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
D = np.zeros((n_s, n_s))
for i, a in enumerate(names):
    for j, b in enumerate(names):
        if i != j: D[i, j] = hav(CENT[a][0], CENT[a][1], CENT[b][0], CENT[b][1])
W = np.zeros((n_s, n_s))
for a, nb in QUEEN_ADJ.items():
    for b in nb: W[idx[a], idx[b]] = 1.0
j_jeju = idx['Jeju']; d_j = D[j_jeju].copy(); d_j[j_jeju] = np.inf; knn = np.argsort(d_j)[:2]
for k in knn: W[j_jeju, k] = 1.0; W[k, j_jeju] = 1.0
W = np.maximum(W, W.T)
rs = W.sum(axis=1, keepdims=True); rs[rs == 0] = 1; W = W / rs

def morans_I(x, W, nperm=9999, seed=42):
    z = x - x.mean(); n_ = len(x); S0 = W.sum()
    I = (n_/S0) * (W*np.outer(z,z)).sum() / (z**2).sum()
    rng = np.random.default_rng(seed); perm = np.zeros(nperm)
    for s in range(nperm):
        zp = rng.permutation(z)
        perm[s] = (n_/S0) * (W*np.outer(zp,zp)).sum() / (z**2).sum()
    p = (np.sum(np.abs(perm) >= np.abs(I)) + 1)/(nperm+1)
    return I, p

sr = sido_df.set_index('sido_en').reindex(names)
I_raw, p_raw = morans_I(sr['raw_rate'].fillna(0).values, W)
I_eb, p_eb = morans_I(sr['eb_rate'].fillna(0).values, W)
print(f"Moran's I: raw={I_raw:.3f} (p={p_raw:.4f}); EB={I_eb:.3f} (p={p_eb:.4f})")
pd.DataFrame([
    {'rate_type':'Raw rates','I':I_raw,'p_perm':p_raw},
    {'rate_type':'EB-smoothed (Marshall 1991)','I':I_eb,'p_perm':p_eb},
]).to_csv(f'{WORK}/tbl_morans_eb_comparison.csv', index=False)

# LISA FDR
print("\nLISA FDR...")
lisa = pd.read_csv(f'{WORK}/tbl_lisa.csv')
p_col = 'p_perm' if 'p_perm' in lisa.columns else 'p'
rej, q, _, _ = multipletests(lisa[p_col], alpha=0.05, method='fdr_bh')
lisa['q_fdr'] = q; lisa['sig_fdr'] = rej
lisa.to_csv(f'{WORK}/tbl_lisa_fdr.csv', index=False)
print(f"  LISA q<0.05: {rej.sum()} of {len(lisa)} provinces")

# Period-stratified
print("\nPeriod-stratified spatial...")
def sido_rate_period(years):
    sub = ys[ys.year.isin(years)]
    cum_p = sub.groupby('region').agg(cases=('yearly_cases','sum'), mean_pop=('pop','mean')).reset_index()
    cum_p['sido_en'] = cum_p['region'].apply(get_sido)
    sido_rate = cum_p.groupby('sido_en').apply(
        lambda g_: g_['cases'].sum()/(g_['mean_pop'].sum()*len(years))*1e5 if g_['mean_pop'].sum()>0 else 0
    ).reindex(names).fillna(0).values
    return sido_rate
early = sido_rate_period(range(2001, 2013))
late = sido_rate_period(range(2013, 2025))
I_e, p_e = morans_I(early, W)
I_l, p_l = morans_I(late, W)
print(f"Early (2001-2012): I={I_e:.3f} p={p_e:.4f}")
print(f"Late  (2013-2024): I={I_l:.3f} p={p_l:.4f}")
pd.DataFrame([
    {'period':'2001-2012','I':I_e,'p_perm':p_e},
    {'period':'2013-2024','I':I_l,'p_perm':p_l},
]).to_csv(f'{WORK}/tbl_morans_period.csv', index=False)

# ASR
print("\nASR + age-specific...")
df2y = df2[df2.year.between(2001, 2024)]
cases_age = (df2y[(df2y.sex == '계') & (~df2y.age_group.isin(['계', '연령미상']))]
             .groupby(['year', 'age_group']).cases.sum().reset_index())
pop_raw = pd.read_csv('/sessions/serene-confident-edison/mnt/논문_이건상/장출/data/population_final_harmonized.csv', encoding='utf-8-sig')
DOUBLED = {'수원시':'경기도','성남시':'경기도','안양시':'경기도','안산시':'경기도',
           '고양시':'경기도','용인시':'경기도','창원시':'경상남도','청주시':'충청북도',
           '천안시':'충청남도','전주시':'전라북도','포항시':'경상북도'}
mask = (pop_raw['sigungu'].isin(DOUBLED.keys())
        & pop_raw.apply(lambda r: DOUBLED.get(r['sigungu']) == r['sido'], axis=1))
pop_raw.loc[mask, 'population'] = pop_raw.loc[mask, 'population']/2
def map_age(ag):
    if ag in ['0-4세','5-9세']: return '0~9세'
    if ag in ['10-14세','15-19세']: return '10~19세'
    if ag in ['20-24세','25-29세']: return '20~29세'
    if ag in ['30-34세','35-39세']: return '30~39세'
    if ag in ['40-44세','45-49세']: return '40~49세'
    if ag in ['50-54세','55-59세']: return '50~59세'
    if ag in ['60-64세','65-69세']: return '60~69세'
    return '70세 이상'
pop_age = pop_raw[(pop_raw.sex == '계') & (pop_raw.age_group != '계')].copy()
pop_age['age_bin'] = pop_age['age_group'].map(map_age)
pop_age = pop_age[pop_age.year.between(2001, 2024)]
pop_year_age = (pop_age.groupby(['year','age_bin'])['population'].sum()
                  .reset_index().rename(columns={'age_bin':'age_group'}))
merged = cases_age.merge(pop_year_age, on=['year','age_group'], how='left')
merged['rate_per_100k'] = merged['cases']/merged['population']*1e5

KSP2005 = {'0~9세':11.6,'10~19세':13.3,'20~29세':14.7,'30~39세':17.5,
           '40~49세':16.2,'50~59세':11.0,'60~69세':8.4,'70세 이상':7.3}
ksp = pd.Series(KSP2005); ksp = ksp/ksp.sum()
asr_rows = []
for yr in sorted(merged.year.unique()):
    sub = merged[merged.year == yr].set_index('age_group')
    rates = sub['rate_per_100k']
    asr = (rates * ksp).sum()
    var = ((ksp**2)*sub['cases']/(sub['population']**2)*1e10).sum()
    se = np.sqrt(var)
    crude = sub['cases'].sum()/sub['population'].sum()*1e5
    asr_rows.append({'year':yr,'crude_rate':crude,'asr_2005ksp':asr,'asr_se':se,
                     'asr_lo':asr-1.96*se,'asr_hi':asr+1.96*se})
pd.DataFrame(asr_rows).to_csv(f'{WORK}/tbl_asr.csv', index=False)

def hr_test(rates):
    rates = np.asarray(rates)+1e-9
    log_r = np.log(rates)
    if len(rates) < 5 or pd.Series(rates).nunique() < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    res = mk.hamed_rao_modification_test(log_r)
    sen_pct = (np.exp(res.slope)-1)*100
    z = 1.96 * np.sqrt(res.var_s)/(len(rates)*(len(rates)-1)/2)
    return res.Tau, res.p, sen_pct, (np.exp(res.slope-z)-1)*100, (np.exp(res.slope+z)-1)*100

age_rows = []
for ag in sorted(merged.age_group.unique()):
    sub = merged[merged.age_group == ag].sort_values('year').reset_index(drop=True)
    if len(sub) < 5: continue
    tau, p, sen, lo, hi = hr_test(sub['rate_per_100k'].values)
    age_rows.append({'age_group':ag,'mean_rate_per_100k':sub['rate_per_100k'].mean(),
                     'sen_pct_year':sen,'sen_lo':lo,'sen_hi':hi,
                     'mk_tau':tau,'mk_p':p})
pd.DataFrame(age_rows).to_csv(f'{WORK}/tbl_age_specific.csv', index=False)
print(pd.DataFrame(age_rows).to_string(index=False))
print("\nDone.")
