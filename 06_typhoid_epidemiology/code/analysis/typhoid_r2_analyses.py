"""Typhoid R2 — combined new analyses (Sen CI, EB, age-std, period-stratified
spatial, FDR LISA, breakpoint bootstrap, imported reconstruction)."""
import numpy as np, pandas as pd, statsmodels.api as sm
import pymannkendall as mk
import os

DATA = '/sessions/serene-confident-edison/mnt/논문_이건상/장티푸스/data'
WORK = '/sessions/serene-confident-edison/mnt/outputs/typhoid_work'
RNG = np.random.default_rng(42)

ann = pd.read_csv(f'{WORK}/tbl_annual_rates.csv').sort_values('year').reset_index(drop=True)
df2 = pd.read_csv(f'{WORK}/df2_long.csv')
yearly_sigungu = pd.read_csv(f'{WORK}/typhoid_yearly_sigungu.csv')

# ─── 1) Sen 95% CIs ────────────────────────────────────────────────
def sen_ci(values):
    v = np.asarray(values, dtype=float)
    n = len(v); slopes = []
    for i in range(n-1):
        for j in range(i+1, n):
            slopes.append((v[j]-v[i])/(j-i))
    slopes = np.sort(slopes); N = len(slopes)
    var_s = n*(n-1)*(2*n+5)/18
    Calpha = 1.96 * np.sqrt(var_s)
    M1 = max(int(np.floor((N-Calpha)/2)), 0)
    M2 = min(int(np.ceil((N+Calpha)/2)), N-1)
    return np.median(slopes), slopes[M1], slopes[M2]

mf_y = pd.read_csv(f'{WORK}/tbl_mf_yearly.csv').sort_values('year').reset_index(drop=True)
mf_col = 'IRR_MF' if 'IRR_MF' in mf_y.columns else 'mf_irr'
sen_rows = []
for series, vec in [
    ('total_log_rate',    np.log(ann['total']/ann['pop_total']*1e5 + 1e-9)),
    ('domestic_log_rate', np.log(ann['domestic']/ann['pop_total']*1e5 + 1e-9)),
    ('imported_log_rate', np.log(ann['imported'].clip(lower=0)/ann['pop_total']*1e5 + 1e-9)),
    ('mf_irr',            mf_y[mf_col].values),
]:
    res = mk.hamed_rao_modification_test(np.asarray(vec))
    med, lo, hi = sen_ci(np.asarray(vec))
    if 'log_rate' in series:
        med_p, lo_p, hi_p = (np.exp(med)-1)*100, (np.exp(lo)-1)*100, (np.exp(hi)-1)*100
    else:
        med_p, lo_p, hi_p = med*100, lo*100, hi*100
    sen_rows.append({
        'series': series, 'tau': res.Tau, 'p_MK': res.p,
        'sen_slope': med, 'sen_slope_lo': lo, 'sen_slope_hi': hi,
        'sen_pct_year': med_p, 'sen_pct_lo': lo_p, 'sen_pct_hi': hi_p})
pd.DataFrame(sen_rows).to_csv(f'{WORK}/tbl_sen_ci.csv', index=False)
print("Sen CIs:")
print(pd.DataFrame(sen_rows).to_string(index=False))

# ─── 2) Bootstrap breakpoint CI for total series ───────────────────
print("\nBootstrap breakpoint CI...")
y_t = ann['total'].astype(int).values
log_pop = np.log(ann['pop_total'].values)
n = len(y_t); t = np.arange(n)
# Use empirical α from R1 NB dispersion
nbd = pd.read_csv(f'{WORK}/tbl_nb_dispersion.csv')
ALPHA = float(nbd[nbd.series == 'total'].alpha_hat.iloc[0])
print(f"  Empirical α = {ALPHA:.3f}")

def fit(X, y, lp): return sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=ALPHA), offset=lp).fit(maxiter=300)
def best1(y, lp, hmin=3):
    best = (None, np.inf, None)
    for bp in range(hmin, n-hmin+1):
        X = np.column_stack([np.ones(n), t, np.maximum(0, t-bp)])
        try:
            m = fit(X, y, lp)
            if m.bic_llf < best[1]: best = (bp, m.bic_llf, m.params)
        except: pass
    return best
def best0(y, lp): return fit(np.column_stack([np.ones(n), t]), y, lp).bic_llf

bp_pt, bic1, beta1 = best1(y_t, log_pop)
bic0 = best0(y_t, log_pop)
if bp_pt is not None:
    print(f"  Best 1-bp = {2001+bp_pt} BIC={bic1:.2f}; 0-bp BIC={bic0:.2f}; dBIC={bic1-bic0:.2f}")
    X1 = np.column_stack([np.ones(n), t, np.maximum(0, t-bp_pt)])
    mu1 = np.exp(X1 @ beta1 + log_pop)
    N_BOOT = 200
    bp_locs = []
    for k in range(N_BOOT):
        p_par = 1/(1+ALPHA*mu1); n_p = mu1*p_par/(1-p_par)
        yb = RNG.negative_binomial(n_p, p_par)
        bb, _, _ = best1(yb, log_pop)
        if bb is not None: bp_locs.append(2001+bb)
    bp_locs = np.array(bp_locs)
    ci = (np.percentile(bp_locs, 2.5), np.percentile(bp_locs, 97.5))
    print(f"  Bootstrap CI = [{ci[0]:.0f}, {ci[1]:.0f}] median={np.median(bp_locs):.0f} (n_boot={len(bp_locs)})")
    pd.DataFrame([{'series':'total','best_1bp_year':2001+bp_pt,'bp_ci_lo':int(ci[0]),'bp_ci_hi':int(ci[1]),
                   'bic_0bp':bic0,'bic_1bp':bic1,'dBIC_1bp_vs_0bp':bic1-bic0,'n_boot':N_BOOT}]
                ).to_csv(f'{WORK}/tbl_bp_bootstrap.csv', index=False)

# ─── 3) Power simulation at empirical α + step changes ─────────────
print(f"\nPower simulation at α={ALPHA:.3f}...")
mu_baseline = ann['total'].mean()
N_SIM = 500
def nb_sample(rng, mu, alpha, size):
    p_ = 1/(1+alpha*mu); n_p = mu*p_/(1-p_)
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

slope_eff = [0, 5, 15, 35, 65, 120, 200]
slope_rows = []
for eff in slope_eff:
    n_d = 0
    for _ in range(N_SIM):
        bp_idx = 12; t_l = np.arange(24)
        eff_v = np.where(t_l < bp_idx, 0, (t_l-bp_idx)*eff/100)
        mu = np.maximum(mu_baseline*(1+eff_v), 1)
        y = nb_sample(RNG, mu, ALPHA, 24)
        b0 = fit_bp_power(y, 0); b1 = fit_bp_power(y, 1)
        if (b1-b0) < -6: n_d += 1
    slope_rows.append({'effect_pct_per_year': eff, 'detection_prob': n_d/N_SIM, 'n_sim': N_SIM, 'alpha': ALPHA})
    print(f"  slope {eff}%/yr: power={n_d/N_SIM:.3f}")
pd.DataFrame(slope_rows).to_csv(f'{WORK}/tbl_power_sim_v2.csv', index=False)

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

# ─── 4) EB shrinkage + LISA FDR ────────────────────────────────────
print("\nMarshall EB shrinkage + LISA FDR...")
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
cum = (ys.groupby('region').agg(total_cases=('yearly_cases','sum'), mean_pop=('pop','mean')).reset_index())
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

# Moran's I on raw vs EB
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
    dlat = la2r - la1r; dlon = lo2r - lo1r
    a = np.sin(dlat/2)**2 + np.cos(la1r)*np.cos(la2r)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
D = np.zeros((n_s, n_s))
for i, a in enumerate(names):
    for j, b in enumerate(names):
        if i != j: D[i, j] = hav(CENT[a][0], CENT[a][1], CENT[b][0], CENT[b][1])
W = np.zeros((n_s, n_s))
for a, nb in QUEEN_ADJ.items():
    for b in nb: W[idx[a], idx[b]] = 1.0
j = idx['Jeju']; d_j = D[j].copy(); d_j[j] = np.inf; knn = np.argsort(d_j)[:2]
for k in knn: W[j, k] = 1.0; W[k, j] = 1.0
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

# LISA FDR — use existing LISA tbl + apply BH
lisa = pd.read_csv(f'{WORK}/tbl_lisa.csv')
print(f"\nLISA FDR adjustment ({len(lisa)} provinces)...")
from statsmodels.stats.multitest import multipletests
p_col = 'p_perm' if 'p_perm' in lisa.columns else 'p'
rej, q, _, _ = multipletests(lisa[p_col], alpha=0.05, method='fdr_bh')
lisa['q_fdr'] = q; lisa['sig_fdr'] = rej
lisa.to_csv(f'{WORK}/tbl_lisa_fdr.csv', index=False)
print(lisa[['sido_en' if 'sido_en' in lisa.columns else lisa.columns[0], p_col, 'q_fdr', 'sig_fdr']].to_string(index=False))

# ─── 5) Period-stratified spatial (early 2001-2012 vs late 2013-2024) ─
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

# ─── 6) Age-standardisation + age-specific ────────────────────────
print("\nAge standardisation + age-specific...")
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
print(f"  ASR table saved ({len(asr_rows)} years)")

def hr_test(rates):
    rates = np.asarray(rates) + 1e-9
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
                     'sen_pct_year':sen, 'sen_lo':lo, 'sen_hi':hi,
                     'mk_tau':tau, 'mk_p':p})
pd.DataFrame(age_rows).to_csv(f'{WORK}/tbl_age_specific.csv', index=False)
print(pd.DataFrame(age_rows).to_string(index=False))

print("\nDone.")
