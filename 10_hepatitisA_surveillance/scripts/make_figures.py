"""Build the 4 main-text figures (Figure 1, 2, 3, 4) — ZPH-MATCHED LAYOUT.

Figure 1 (5 panels): A stacked-area annual rate (domestic + imported), B NB
        regression overlay, C age composition stacked area (% by year), D M:F
        IRR with 95% CI, E age-specific incidence rate trends (selected strata).
Figure 2 (4 panels in 2 rows): full-period CWT scalogram + GWS, then
        pre-2019 subset CWT scalogram + GWS. Red-blue diverging colormap.
Figure 3 (3 horizontal panels): A province bar chart sorted by rate (coloured by
        quartile), B Moran scatter under Queen + KNN-2 fallback, C Local Moran's
        I bar chart by province (coloured by LISA quadrant).
Figure 4: Power to detect a single NB breakpoint vs effect-size %/yr.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
import geopandas as gpd
from statsmodels.tsa.seasonal import STL

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 100,
})

OUT = '/sessions/elegant-quirky-hamilton/mnt/outputs/hepa_work'

# ─── Common data ────────────────────────────────────────────────────
ann = pd.read_csv(f'{OUT}/tbl_annual_rates.csv').sort_values('year').reset_index(drop=True)
ann['rate_total'] = ann.total / ann.pop_total * 1e5
ann['rate_dom']   = ann.domestic / ann.pop_total * 1e5
ann['rate_imp']   = ann.imported.clip(lower=0) / ann.pop_total * 1e5
df2 = pd.read_csv(f'{OUT}/df2_long.csv')
mf = pd.read_csv(f'{OUT}/tbl_mf_yearly.csv')
sex_age_csv = pd.read_csv(f'{OUT}/tbl_sex_age.csv')
prov = pd.read_csv(f'{OUT}/tbl_provincial_rates.csv')
lisa = pd.read_csv(f'{OUT}/tbl_lisa.csv')
morans = pd.read_csv(f'{OUT}/tbl_morans_main.csv')
pwr = pd.read_csv(f'{OUT}/tbl_power_sim.csv')

# ════════════════════════════════════════════════════════════════════
# FIGURE 1 — 5 panels (matching ZPH layout)
# ════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(11, 11), constrained_layout=True)
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
fig.suptitle('Figure 1.  Temporal dynamics of hepatitis A in South Korea, 2011–2024',
             fontsize=13, fontweight='bold')

# ── Panel A (top, full width): stacked area annual rate by source ──
axA = fig.add_subplot(gs[0, :])
axA.fill_between(ann.year, 0, ann.rate_dom, color='steelblue', alpha=0.55,
                 label='Domestic')
axA.fill_between(ann.year, ann.rate_dom, ann.rate_dom + ann.rate_imp,
                 color='tab:orange', alpha=0.55, label='Imported')
axA.plot(ann.year, ann.rate_total, color='black', linewidth=1.6, marker='o',
         markersize=4, markerfacecolor='black', label='Total')
axA.set_xlabel('Year')
axA.set_ylabel('Incidence rate (per 100 000)')
axA.set_title('A.  Annual incidence rate of hepatitis A, 2011–2024, by source',
              fontsize=10.5, loc='center')
axA.set_xticks(np.arange(2011, 2025, 2))
axA.legend(loc='upper right', framealpha=0.95)

# ── Panel B (middle-left): NB regression fit ──
axB = fig.add_subplot(gs[1, 0])
X = sm.add_constant(np.arange(len(ann)))
log_pop = np.log(ann.pop_total.values)
nb_total = sm.GLM(ann.total.values, X, family=sm.families.NegativeBinomial(alpha=1.0),
                  offset=log_pop).fit()
nb_dom   = sm.GLM(ann.domestic.values, X, family=sm.families.NegativeBinomial(alpha=1.0),
                  offset=log_pop).fit()
mu_total = nb_total.predict(X) / ann.pop_total.values * 1e5
mu_dom   = nb_dom.predict(X)   / ann.pop_total.values * 1e5
axB.plot(ann.year, ann.rate_total, 'o-', color='steelblue', linewidth=1.0,
         markersize=4, label='Total observed')
axB.plot(ann.year, mu_total, '--', color='steelblue', linewidth=1.6,
         label='Total NB fit')
axB.plot(ann.year, ann.rate_dom, 's-', color='seagreen', linewidth=1.0,
         markersize=4, label='Domestic observed')
axB.plot(ann.year, mu_dom, '--', color='seagreen', linewidth=1.6,
         label='Domestic NB fit')
irr_t = float(np.exp(nb_total.params[1]))
irr_d = float(np.exp(nb_dom.params[1]))
pct_t = (irr_t - 1) * 100
pct_d = (irr_d - 1) * 100
axB.set_xlabel('Year')
axB.set_ylabel('Rate (per 100 000)')
axB.set_title(f'B.  NB regression (year-on-year fit)\n+{pct_t:.1f}% total, +{pct_d:.1f}% domestic /yr',
              fontsize=10.5)
axB.set_xticks(np.arange(2011, 2025, 2))
axB.legend(loc='upper right', fontsize=8, framealpha=0.95)

# ── Panel C (middle-right): age composition stacked area (% by year) ──
axC = fig.add_subplot(gs[1, 1])
AGE_ORDER = ['0~9세','10~19세','20~29세','30~39세','40~49세','50~59세','60~69세','70세 이상']
AGE_LABEL = ['0–9','10–19','20–29','30–39','40–49','50–59','60–69','70+']
age_yr = (df2[df2.sex=='계']
          .pivot_table(index='year', columns='age_group', values='cases', aggfunc='sum')
          .reindex(columns=AGE_ORDER, fill_value=0))
age_yr_pct = age_yr.div(age_yr.sum(axis=1), axis=0) * 100
viridis = mpl.colormaps['viridis'].resampled(8)
colors = [viridis(i) for i in range(8)]
axC.stackplot(age_yr_pct.index, age_yr_pct.values.T, labels=AGE_LABEL,
              colors=colors, alpha=0.85, edgecolor='white', linewidth=0.3)
axC.set_xlabel('Year')
axC.set_ylabel('Age composition (%)')
axC.set_title('C.  Age composition of reported cases', fontsize=10.5)
axC.set_xticks(np.arange(2011, 2025, 2))
axC.set_ylim(0, 100)
axC.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8,
           title='Age', frameon=False)

# ── Panel D (bottom-left): M:F IRR per year + pooled reference ──
axD = fig.add_subplot(gs[2, 0])
axD.errorbar(mf.year, mf.IRR_MF,
             yerr=[mf.IRR_MF - mf.IRR_lo, mf.IRR_hi - mf.IRR_MF],
             fmt='o-', color='black', markeredgecolor='black',
             markerfacecolor='white', markersize=5, capsize=3, linewidth=1)
axD.axhline(1.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
axD.axhline(1.31, color='tab:red', linestyle=':', linewidth=1.2, alpha=0.8,
            label='Pooled 1.31')
axD.axhspan(1.29, 1.33, color='tab:red', alpha=0.10, label='95% CI of pooled')
import pymannkendall as mk
res = mk.hamed_rao_modification_test(mf.IRR_MF.dropna().values)
axD.set_xlabel('Year')
axD.set_ylabel('Male-to-female IRR')
axD.set_title(f'D.  Male-to-female incidence rate ratio\n(overall 1.31, 95% CI 1.29–1.33; MK τ = {res.Tau:.3f}, p = {res.p:.4f})',
              fontsize=10.5)
axD.set_xticks(np.arange(2011, 2025, 2))
axD.legend(loc='upper right', fontsize=8, framealpha=0.95)

# ── Panel E (bottom-right): Age-specific rates over time ──
axE = fig.add_subplot(gs[2, 1])
# Compute rate per 100k per year per age stratum (sex-pooled)
pop_path = '/sessions/elegant-quirky-hamilton/mnt/논문_이건상/A형간염/data/population_final_harmonized.csv'
pop = pd.read_csv(pop_path, encoding='utf-8-sig'); pop.columns = pop.columns.str.strip()
DOUBLED_CITIES = {'수원시':'경기도','성남시':'경기도','안양시':'경기도','안산시':'경기도',
                  '고양시':'경기도','용인시':'경기도','창원시':'경상남도','청주시':'충청북도',
                  '천안시':'충청남도','전주시':'전라북도','포항시':'경상북도'}
mask = pop.apply(lambda r: DOUBLED_CITIES.get(r['sigungu']) == r['sido'], axis=1)
pop.loc[mask, 'population'] /= 2

def pop_to_bin(ag):
    if ag in ('0-4세','5-9세'): return '0~9세'
    if ag in ('10-14세','15-19세'): return '10~19세'
    if ag in ('20-24세','25-29세'): return '20~29세'
    if ag in ('30-34세','35-39세'): return '30~39세'
    if ag in ('40-44세','45-49세'): return '40~49세'
    if ag in ('50-54세','55-59세'): return '50~59세'
    if ag in ('60-64세','65-69세'): return '60~69세'
    if any(k in ag for k in ['70-74','75-79','80-84','80세 이상','85-89','85세 이상',
                              '90-94','95-99','95세 이상','100세 이상']): return '70세 이상'
    return None

pop = pop[(pop.sex.isin(['남','여'])) & (pop.year.between(2011, 2024))].copy()
pop['agebin'] = pop['age_group'].map(pop_to_bin)
pop_yr_age = pop.dropna(subset=['agebin']).groupby(['year','agebin'])['population'].sum().unstack()

cases_yr_age = (df2[df2.sex.isin(['남','여'])]
                .groupby(['year','age_group'])['cases'].sum().unstack(fill_value=0)
                .reindex(columns=AGE_ORDER, fill_value=0))
rate_yr_age = cases_yr_age.div(pop_yr_age.reindex(columns=AGE_ORDER), axis=1) * 1e5

# Plot 4 selected strata
SELECTED = [('30~39세', '30–39'), ('40~49세', '40–49'),
            ('50~59세', '50–59'), ('70세 이상', '70+')]
plot_colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
for (ag, lab), c in zip(SELECTED, plot_colors):
    axE.plot(rate_yr_age.index, rate_yr_age[ag], 'o-', color=c, linewidth=1.4,
             markersize=4, label=lab)
axE.set_xlabel('Year')
axE.set_ylabel('Rate (per 100 000)')
axE.set_title('E.  Age-specific incidence rate trends', fontsize=10.5)
axE.set_xticks(np.arange(2011, 2025, 2))
axE.set_yscale('log')
axE.legend(loc='upper right', fontsize=8, framealpha=0.95)

plt.savefig(f'{OUT}/Figure_1.tif', dpi=300, format='tiff',
            pil_kwargs={'compression':'tiff_lzw'}, bbox_inches='tight')
plt.savefig(f'{OUT}/Figure_1.png', dpi=180, format='png', bbox_inches='tight')
plt.close()
print('Saved Figure_1 (5 panels, ZPH-matched)')

# ════════════════════════════════════════════════════════════════════
# FIGURE 2 — Single full-period CWT scalogram + GWS sidecar (red-blue)
# ════════════════════════════════════════════════════════════════════
weekly = pd.read_csv(f'{OUT}/weekly_total.csv', parse_dates=['date']).set_index('date')['cases']
log_y_full = np.log(weekly.values + 1.0)

def fit_ar1(x):
    x = x - x.mean()
    num = (x[:-1]*x[1:]).sum(); den = (x[:-1]**2).sum()
    rho = num/den if den > 0 else 0.0
    return rho, (x[1:] - rho*x[:-1]).std(ddof=1)

def cwt(sig, periods, omega0=6.0):
    n = len(sig); fft = np.fft.fft(sig)
    freqs = np.fft.fftfreq(n, d=1.0) * 2 * np.pi
    scales = periods * (4*np.pi) / (omega0 + np.sqrt(2 + omega0**2))
    W = np.zeros((len(scales), n), dtype=complex)
    for i, s in enumerate(scales):
        e = -0.5 * (s*freqs - omega0)**2 * (freqs > 0)
        norm = np.sqrt(2*np.pi*s/1.0) * np.pi**(-0.25)
        W[i] = np.fft.ifft(fft * norm * np.exp(e) * (freqs > 0))
    return np.abs(W)**2

stl = STL(log_y_full, period=52, seasonal=25, trend=103, robust=True).fit()
detr = log_y_full - stl.trend
periods = np.arange(8, 261, 2).astype(float)
pw = cwt(detr, periods)
rho, sig = fit_ar1(detr)
rng = np.random.default_rng(42)
null_gws = np.zeros((200, len(periods)))
for i in range(200):
    x = np.zeros(len(detr)); e = rng.normal(0, sig, len(x))
    for t in range(1, len(x)): x[t] = rho*x[t-1] + e[t]
    null_gws[i] = cwt(x, periods).mean(axis=1)
p95 = np.percentile(null_gws, 95, axis=0)
gws = pw.mean(axis=1)
log_pw = np.log(pw + 1e-9)
vmax = max(abs(log_pw.min()), abs(log_pw.max()))

fig = plt.figure(figsize=(12, 5.5), constrained_layout=True)
gs2 = fig.add_gridspec(1, 4, width_ratios=[5, 5, 5, 2.5])
fig.suptitle('Figure 2.  Continuous wavelet transform of weekly hepatitis A incidence, 2011–2024\n(STL-detrended, Morlet ω₀ = 6, red-noise AR(1) null, 200 simulations)',
             fontsize=11.5, fontweight='bold')

axS = fig.add_subplot(gs2[0, :3])
im = axS.imshow(log_pw, aspect='auto',
                extent=[mpl.dates.date2num(weekly.index.min()),
                        mpl.dates.date2num(weekly.index.max()),
                        np.log2(periods.max()), np.log2(periods.min())],
                cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                interpolation='bilinear')
axS.xaxis_date()
axS.set_yticks(np.log2([16, 32, 52, 104, 208]))
axS.set_yticklabels(['16','32','52','104','208'])
axS.invert_yaxis()
axS.axhline(np.log2(52), color='black', linestyle='--', linewidth=0.8, alpha=0.7)
axS.text(mpl.dates.date2num(weekly.index.max()) - 200, np.log2(52) + 0.1,
         '52 wk (annual)', color='black', fontsize=9, ha='right')
axS.set_xlabel('Year')
axS.set_ylabel('Period (weeks)')
axS.set_title('A.  Wavelet power spectrum (full period 2011–2024)', fontsize=10.5)
plt.colorbar(im, ax=axS, label=r'$\log_e$ power', shrink=0.85)

axG = fig.add_subplot(gs2[0, 3])
axG.plot(gws, periods, color='black', linewidth=1.4, label='Observed GWS')
axG.plot(p95, periods, color='red', linestyle='--', linewidth=1.2,
         label='AR(1) 95% null')
axG.fill_betweenx(periods[(periods>=48)&(periods<=56)], 0,
                  max(gws.max(), p95.max())*1.1,
                  color='orange', alpha=0.15, label='Annual band')
axG.set_yscale('log')
axG.set_yticks([16, 32, 52, 104, 208])
axG.set_yticklabels(['16','32','52','104','208'])
axG.invert_yaxis()
axG.set_xlabel('Power')
axG.legend(fontsize=7.5, loc='lower right')
axG.set_title('B.  Global wavelet spectrum', fontsize=9.5)

plt.savefig(f'{OUT}/Figure_2.tif', dpi=300, format='tiff',
            pil_kwargs={'compression':'tiff_lzw'}, bbox_inches='tight')
plt.savefig(f'{OUT}/Figure_2.png', dpi=180, format='png', bbox_inches='tight')
plt.close()
print('Saved Figure_2 (single full-period CWT + GWS sidecar)')

# ════════════════════════════════════════════════════════════════════
# FIGURE 3 — 3 horizontal panels: province bar + Moran scatter + LISA bar
# ════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 6), constrained_layout=True)
gs3 = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
fig.suptitle('Figure 3.  Spatial pattern of hepatitis A incidence by province, 2011–2024',
             fontsize=12, fontweight='bold')

# Panel A: province bar (sorted, coloured by quartile)
axA = fig.add_subplot(gs3[0, 0])
prov_sorted = prov.sort_values('mean_annual_rate', ascending=False).reset_index(drop=True)
quartiles = pd.qcut(prov_sorted.mean_annual_rate, 4, labels=False)
quart_colors = ['tab:red', 'tab:orange', 'steelblue', 'navy']
bar_colors = [quart_colors[q] for q in quartiles]
axA.barh(prov_sorted.sido_en[::-1], prov_sorted.mean_annual_rate[::-1],
         color=bar_colors[::-1], edgecolor='black', linewidth=0.5)
axA.set_xlabel('Mean annual incidence rate (per 100 000)')
axA.set_title('A.  Province-level mean rate, 2011–2024', fontsize=10.5)

# Panel B: Moran scatter
# Need standardized rate (z-score) and spatial lag
prov_spatial = prov[prov.sido_en != 'Sejong'].set_index('sido_en')
CENT = {
    'Seoul':(37.5665,126.9780), 'Busan':(35.1796,129.0756),
    'Daegu':(35.8714,128.6014), 'Incheon':(37.4563,126.7052),
    'Gwangju':(35.1595,126.8526), 'Daejeon':(36.3504,127.3845),
    'Ulsan':(35.5384,129.3114), 'Gyeonggi':(37.4138,127.5183),
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
prov_spatial = prov_spatial.reindex(names)
n = len(names); idx_map = {nm: i for i, nm in enumerate(names)}
# Distance for Jeju KNN-2 fallback
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
W = np.zeros((n,n))
for a, nb in QUEEN_ADJ.items():
    for b in nb:
        W[idx_map[a], idx_map[b]] = 1.0
# Jeju fallback
j = idx_map['Jeju']
d = D[j].copy(); d[j] = np.inf
for k in np.argsort(d)[:2]:
    W[j, k] = 1.0; W[k, j] = 1.0
W = np.maximum(W, W.T)
W_rs = W / W.sum(axis=1, keepdims=True).clip(min=1)

rates = prov_spatial.mean_annual_rate.values
z = (rates - rates.mean()) / rates.std(ddof=0)
lag_z = W_rs @ z

axB = fig.add_subplot(gs3[0, 1])
# Color by LISA quadrant from tbl_lisa
lisa_idx = lisa.set_index('sido_en')
QUAD_COLOR = {'HH': 'tab:red', 'LL': 'steelblue', 'HL': 'tab:orange', 'LH': 'tab:green'}
for nm, zi, li in zip(names, z, lag_z):
    q = lisa_idx.loc[nm, 'quadrant'] if nm in lisa_idx.index else 'LL'
    sig = lisa_idx.loc[nm, 'sig_5pct'] == True if nm in lisa_idx.index else False
    color = QUAD_COLOR.get(q, 'gray')
    axB.scatter(zi, li, s=70 if sig else 40, color=color,
                edgecolor='black', linewidth=1.0 if sig else 0.4,
                alpha=0.95 if sig else 0.7, zorder=3)
    axB.annotate(nm, (zi, li), fontsize=7,
                 xytext=(4, 4), textcoords='offset points')
# Linear fit (Moran's I)
slope = (z * lag_z).sum() / (z * z).sum()
xx = np.array([z.min()*1.1, z.max()*1.1])
axB.plot(xx, slope * xx, '--', color='black', linewidth=1.0, alpha=0.7)
axB.axhline(0, color='gray', linewidth=0.5); axB.axvline(0, color='gray', linewidth=0.5)
moran_total = morans[(morans.scheme=='Queen+KNN2_baseline') & (morans.series=='total_df2')].iloc[0]
axB.set_xlabel('Standardised rate (z)')
axB.set_ylabel('Spatial lag (z)')
axB.set_title(f'B.  Moran scatter (Q+KNN-2 fallback)\nGlobal I = {moran_total["I"]:.3f}, p = {moran_total["p_perm"]:.4f}',
              fontsize=10.5)
# Legend
from matplotlib.patches import Patch
handles = [Patch(facecolor=c, edgecolor='black', label=q) for q, c in QUAD_COLOR.items()]
axB.legend(handles=handles, loc='upper left', fontsize=8, framealpha=0.95, title='LISA quadrant')

# Panel C: Local Moran's I bar
axC = fig.add_subplot(gs3[0, 2])
lisa_sorted = lisa.sort_values('I_local', ascending=False).reset_index(drop=True)
bar_colors_c = []
sig_marker = []
for _, r in lisa_sorted.iterrows():
    color = QUAD_COLOR.get(r.quadrant, 'lightgrey')
    bar_colors_c.append(color if r.sig_5pct == True else 'lightgrey')
    if r.p_perm < 0.01: sig_marker.append('**')
    elif r.p_perm < 0.05: sig_marker.append('*')
    else: sig_marker.append('')
axC.barh(lisa_sorted.sido_en[::-1], lisa_sorted.I_local[::-1],
         color=bar_colors_c[::-1], edgecolor='black', linewidth=0.5)
# Annotate stars
for i, (val, mk_) in enumerate(zip(lisa_sorted.I_local[::-1], sig_marker[::-1])):
    if mk_:
        axC.text(val + 0.05, i, mk_, va='center', fontsize=12, fontweight='bold')
axC.set_xlabel("Local Moran's Iᵢ")
axC.set_title("C.  Local Moran's I (LISA)\n* p < 0.05, ** p < 0.01", fontsize=10.5)
axC.axvline(0, color='gray', linewidth=0.5)

plt.savefig(f'{OUT}/Figure_3.tif', dpi=300, format='tiff',
            pil_kwargs={'compression':'tiff_lzw'}, bbox_inches='tight')
plt.savefig(f'{OUT}/Figure_3.png', dpi=180, format='png', bbox_inches='tight')
plt.close()
print('Saved Figure_3 (3 panels, ZPH-matched: bar + Moran scatter + LISA)')

# ════════════════════════════════════════════════════════════════════
# FIGURE 4 — Power simulation
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
ax.plot(pwr.effect_pct, pwr.detection_prob * 100, 'o-', color='steelblue',
        linewidth=2, markersize=8, markeredgecolor='black')
ax.axhline(80, color='red', linestyle='--', linewidth=1.0, label='80% power threshold')
ax.axhline(5, color='gray', linestyle=':', linewidth=1.0, label='5% nominal FPR')
ax.set_xlabel('Post-breakpoint additional rate change (% per year)')
ax.set_ylabel('Detection probability (%) with ΔBIC < −6')
ax.set_title('Figure 4.  Power to detect a single NB breakpoint\n(n=14, dispersion α=2, 200 simulations per effect size)',
             fontsize=11, fontweight='bold')
ax.set_ylim(0, 105)
ax.set_xlim(-5, 130)
ax.legend(loc='lower right', framealpha=0.95)
ax.grid(axis='y', alpha=0.3)
plt.savefig(f'{OUT}/Figure_4.tif', dpi=300, format='tiff',
            pil_kwargs={'compression':'tiff_lzw'}, bbox_inches='tight')
plt.savefig(f'{OUT}/Figure_4.png', dpi=180, format='png', bbox_inches='tight')
plt.close()
print('Saved Figure_4 (power simulation)')
