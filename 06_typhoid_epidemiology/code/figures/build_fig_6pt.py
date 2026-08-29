"""Rebuild composite figures freshly with matplotlib for clean panel labels."""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.size':7,'axes.titlesize':7.5,'axes.labelsize':7,
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
    'figure.titlesize':8.5,'axes.linewidth':0.6,
    'figure.constrained_layout.h_pad':0.03,'figure.constrained_layout.w_pad':0.03,
    'savefig.pad_inches':0.02,'pdf.fonttype':42,'ps.fonttype':42,
})

import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

DATA = '/sessions/hopeful-exciting-cannon/mnt/논문_이건상/장티푸스/data'
TBL = '/sessions/hopeful-exciting-cannon/mnt/논문_이건상/장티푸스/FBPD_R2/tables_new'
FIG = '/sessions/hopeful-exciting-cannon/mnt/논문_이건상/장티푸스/FBPD_R2/FINAL_SUBMISSION/figures_6pt_revised'

# Load data
df1 = pd.read_csv(f'{DATA}/df1_장티푸스.csv')
df1.columns = [c.strip().replace('\ufeff','') for c in df1.columns]
df1['year'] = df1['year'].astype(int); df1['week'] = df1['week'].astype(int)
df1 = df1[(df1.year>=2001)&(df1.year<=2024)]
df2 = pd.read_csv(f'{DATA}/df2_장티푸스.csv')
df2.columns = [c.strip().replace('\ufeff','') for c in df2.columns]
df2['year'] = df2['year'].astype(int)
df2 = df2[(df2.year>=2001)&(df2.year<=2024)]
TBL1_POP = {2001:47457211,2002:47957781,2003:48109404,2004:48264962,2005:48778235,
    2006:48978449,2007:49254055,2008:49792368,2009:50027977,2010:51314398,
    2011:51527552,2012:51740826,2013:51927682,2014:52183502,2015:52378325,
    2016:51696216,2017:51778544,2018:51826059,2019:51849861,2020:51829023,
    2021:51638809,2022:51439038,2023:51337778,2024:51985381}

# Annual totals
ann_dom = df1.groupby('year', as_index=False)['cases'].sum().rename(columns={'cases':'dom'})
ann_tot = df2[(df2.sex=='계')&(df2.age_group=='계')].groupby('year', as_index=False)['cases'].sum().rename(columns={'cases':'tot'})
ann = ann_tot.merge(ann_dom, on='year', how='left')
ann['imp'] = ann['tot'] - ann['dom']
ann['pop'] = ann['year'].map(TBL1_POP)
ann['rate_tot'] = ann['tot']/ann['pop']*1e5
ann['rate_dom'] = ann['dom']/ann['pop']*1e5

# Sex / Age data
sex = df2[(df2.sex.isin(['남','여']))&(df2.age_group=='계')]
mf = sex.groupby(['year','sex'], as_index=False)['cases'].sum()
mf_w = mf.pivot(index='year', columns='sex', values='cases').reset_index()
mf_w.columns = ['year','M','F']

age = df2[(df2.sex=='계')&(df2.age_group.isin(['0~9세','10~19세','20~29세','30~39세','40~49세','50~59세','60~69세','70세 이상']))]
age_y = age.groupby(['year','age_group'], as_index=False)['cases'].sum()
age_w = age_y.pivot(index='year', columns='age_group', values='cases').reset_index().fillna(0)
age_order = ['0~9세','10~19세','20~29세','30~39세','40~49세','50~59세','60~69세','70세 이상']
age_w_pct = age_w.copy()
totals = age_w[age_order].sum(axis=1)
for col in age_order:
    age_w_pct[col] = age_w[col] / totals * 100

# Power sim data
pwr = pd.read_csv(f'{TBL}/tbl_power_sim_v2.csv')

# === NEW FIGURE 1: 2x3 grid + bottom-row power sim ===
fig = plt.figure(figsize=(7.0, 9.2), constrained_layout=True)
gs = GridSpec(4, 4, figure=fig, height_ratios=[1.6, 1.5, 1.5, 1.6])
fig.suptitle('Figure 1.  Temporal dynamics and structural detectability of typhoid fever in South Korea, 2001–2024',
             fontsize=8.5, fontweight='bold')

# Panel A - stacked area total/imported
axA = fig.add_subplot(gs[0, :])
axA.fill_between(ann.year, 0, ann.rate_dom, color='#5b9bd5', alpha=0.5, label='Domestic')
axA.fill_between(ann.year, ann.rate_dom, ann.rate_tot, color='#ed7d31', alpha=0.5, label='Imported')
axA.plot(ann.year, ann.rate_tot, 'k-o', lw=1.4, ms=5, mfc='black', label='Total')
axA.set_xlabel('Year'); axA.set_ylabel('Incidence rate (per 100 000)')
axA.set_title('A.  Annual incidence rate of typhoid fever, 2001–2024, by source', fontsize=7.5)
axA.legend(loc='upper right', fontsize=7, framealpha=0.95)
axA.grid(alpha=0.3)

# Panel B - NB GLM
import statsmodels.api as sm
X = sm.add_constant(ann.year - ann.year.min())
m_t = sm.GLM(ann.tot, X, family=sm.families.NegativeBinomial(alpha=0.154),
             offset=np.log(ann['pop'])).fit(disp=False)
m_d = sm.GLM(ann.dom, X, family=sm.families.NegativeBinomial(alpha=0.154),
             offset=np.log(ann['pop'])).fit(disp=False)
ann['pred_t'] = m_t.predict(X, offset=np.log(ann['pop']))/ann['pop']*1e5
ann['pred_d'] = m_d.predict(X, offset=np.log(ann['pop']))/ann['pop']*1e5
axB = fig.add_subplot(gs[1, :2])
axB.plot(ann.year, ann.rate_tot, 'o', mfc='steelblue', mec='black', ms=5, label='Total observed')
axB.plot(ann.year, ann.pred_t, '-', color='steelblue', lw=2, label='Total NB fit')
axB.plot(ann.year, ann.rate_dom, 'D', mfc='seagreen', mec='black', ms=4, label='Domestic observed')
axB.plot(ann.year, ann.pred_d, '--', color='seagreen', lw=1.6, label='Domestic NB fit')
axB.set_xlabel('Year'); axB.set_ylabel('Rate (per 100 000)')
axB.set_title('B.  NB GLM fit (year-on-year)\n−7.6%/yr total, −9.1%/yr domestic', fontsize=7.5)
axB.legend(fontsize=7, loc='upper right'); axB.grid(alpha=0.3)

# Panel C - age composition
axC = fig.add_subplot(gs[1, 2:])
cmap = plt.cm.viridis(np.linspace(0, 1, len(age_order)))
axC.stackplot(age_w_pct['year'], [age_w_pct[c] for c in age_order],
              labels=['0–9','10–19','20–29','30–39','40–49','50–59','60–69','≥70'],
              colors=cmap, alpha=0.92)
axC.set_xlabel('Year'); axC.set_ylabel('Age composition (%)')
axC.set_title('C.  Age composition of reported cases', fontsize=7.5)
axC.legend(title='Age', loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=7)
axC.set_ylim(0, 100); axC.grid(alpha=0.3)

# Panel D - M:F IRR
mf_w['IRR'] = mf_w['M']/mf_w['F']
import scipy.stats as st
axD = fig.add_subplot(gs[2, :2])
axD.plot(mf_w['year'], mf_w['IRR'], 'o-', color='black', mfc='white', ms=5)
pooled = mf_w['M'].sum()/mf_w['F'].sum()
axD.axhline(pooled, color='red', linestyle=':', lw=1, alpha=0.6, label=f'Pooled {pooled:.2f}')
axD.axhspan(0.898, 1.025, color='red', alpha=0.10, label='95% CI of pooled')
axD.set_xlabel('Year'); axD.set_ylabel('Male-to-female IRR')
axD.set_title('D.  Male-to-female incidence rate ratio\n(overall 0.96, 95% CI 0.90–1.03; MK τ = 0.007, p = 0.9802)', fontsize=7.5)
axD.legend(fontsize=7); axD.grid(alpha=0.3)

# Panel E - age-specific incidence trends (selected)
sel_ages = ['20~29세','40~49세','50~59세','70세 이상']
axE = fig.add_subplot(gs[2, 2:])
# Use age population proxy = total pop * some weighting; we don't have age-stratified pop, so just use cases scaled
for ag, color in zip(sel_ages, ['#1f77b4','#2ca02c','#d62728','#ff7f0e']):
    series = age_y[age_y.age_group == ag].set_index('year')['cases']
    axE.plot(series.index, series.values, 'o-', label=ag.replace('세','').replace('이상','+'),
             color=color, ms=4, lw=1.2)
axE.set_yscale('log'); axE.set_xlabel('Year'); axE.set_ylabel('Cases (log scale)')
axE.set_title('E.  Age-specific case trends (selected age bands)', fontsize=7.5)
axE.legend(fontsize=7); axE.grid(alpha=0.3, which='both')

# Panel F - power simulation
axF = fig.add_subplot(gs[3, :])
axF.plot(pwr['effect_pct_per_year'], pwr['detection_prob']*100, 'o-', color='steelblue',
         lw=2, ms=8, mec='black')
axF.axhline(80, color='red', linestyle='--', lw=1, label='80% power threshold')
axF.axhline(5, color='gray', linestyle=':', lw=1, label='5% nominal FPR')
axF.set_xlabel('Post-breakpoint additional rate change (% per year)')
axF.set_ylabel('Detection probability (%) with ΔBIC < −6')
axF.set_title('F.  Power to detect a single NB breakpoint (n=24, α=0.154, 1,000 simulations per effect size)', fontsize=7.5)
axF.set_ylim(-2, 105); axF.legend(loc='lower right', fontsize=7); axF.grid(axis='y', alpha=0.3)

plt.savefig(f'{FIG}/Figure_1.png', dpi=600, bbox_inches='tight')
plt.savefig(f'{FIG}/Figure_1.pdf', bbox_inches='tight')
plt.close()
print('Saved new Figure 1 (6 panels A–F)')

# === NEW FIGURE 2: Wavelet (top, 2 panels) + Spatial (bottom, 3 panels) ===
from statsmodels.tsa.seasonal import STL

# Build weekly series
weekly = df1.groupby(['year','week'], as_index=False)['cases'].sum()
def iso_to_dt(y,w):
    try: return pd.to_datetime(f'{y}-W{w:02d}-1', format='%G-W%V-%u')
    except: return pd.NaT
weekly['date'] = [iso_to_dt(y,w) for y,w in zip(weekly['year'],weekly['week'])]
weekly = weekly.dropna(subset=['date']).set_index('date').sort_index()['cases']
log_y = np.log(weekly.values + 1.0)

def fit_ar1(x):
    x = x - x.mean(); num = (x[:-1]*x[1:]).sum(); den = (x[:-1]**2).sum()
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

stl = STL(log_y, period=52, seasonal=25, trend=103, robust=True).fit()
detr = log_y - stl.trend
periods = np.arange(8, 261, 2).astype(float)
pw = cwt(detr, periods)
rho, sig = fit_ar1(detr)
N_SIM = 1000
rng = np.random.default_rng(42)
null_gws = np.zeros((N_SIM, len(periods)))
for i in range(N_SIM):
    x = np.zeros(len(detr)); e = rng.normal(0, sig, len(x))
    for t in range(1, len(x)): x[t] = rho*x[t-1] + e[t]
    null_gws[i] = cwt(x, periods).mean(axis=1)
p95 = np.percentile(null_gws, 95, axis=0)
gws = pw.mean(axis=1)
log_pw = np.log(pw + 1e-9)
vmax = max(abs(log_pw.min()), abs(log_pw.max()))

# Spatial data
prov = pd.read_csv('/sessions/hopeful-exciting-cannon/mnt/논문_이건상/장티푸스/reanalysis_2026_FoodbornePathogensDisease/tables/tbl_sido_rate.csv')
lisa = pd.read_csv(f'{TBL}/tbl_lisa_fdr.csv')
prov_sorted = prov.sort_values('mean_annual_rate', ascending=True)

fig = plt.figure(figsize=(7.0, 7.2), constrained_layout=True)
gs2 = GridSpec(2, 3, figure=fig, height_ratios=[1.3, 1])
fig.suptitle('Figure 2.  Annual periodicity and spatial pattern of typhoid fever in South Korea, 2001–2024',
             fontsize=8.5, fontweight='bold')

# Top row: wavelet (2 sub-panels in same row)
axW = fig.add_subplot(gs2[0, :2])
im = axW.imshow(log_pw, aspect='auto',
                extent=[mpl.dates.date2num(weekly.index.min()),
                        mpl.dates.date2num(weekly.index.max()),
                        np.log2(periods.max()), np.log2(periods.min())],
                cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='bilinear')
axW.xaxis_date()
axW.set_yticks(np.log2([16, 32, 52, 104, 208]))
axW.set_yticklabels(['16','32','52','104','208'])
axW.invert_yaxis()
axW.axhline(np.log2(52), color='black', linestyle='--', linewidth=0.8, alpha=0.7)
axW.text(mpl.dates.date2num(weekly.index.max()) - 200, np.log2(52) + 0.1,
         '52 wk (annual)', color='black', fontsize=7, ha='right')
axW.set_xlabel('Year'); axW.set_ylabel('Period (weeks)')
axW.set_title('A.  Continuous wavelet power spectrum (Morlet ω₀ = 6, STL-detrended weekly series)', fontsize=7.5)
plt.colorbar(im, ax=axW, label=r'$\log_e$ power', shrink=0.85)

axGW = fig.add_subplot(gs2[0, 2])
axGW.plot(gws, periods, color='black', linewidth=1.4, label='Observed GWS')
axGW.plot(p95, periods, color='red', linestyle='--', linewidth=1.2, label='AR(1) 95% null')
axGW.fill_betweenx(periods[(periods>=48)&(periods<=56)], 0,
                   max(gws.max(), p95.max())*1.1,
                   color='orange', alpha=0.15, label='Annual band')
axGW.set_yscale('log'); axGW.set_yticks([16, 32, 52, 104, 208])
axGW.set_yticklabels(['16','32','52','104','208']); axGW.invert_yaxis()
axGW.set_xlabel('Power'); axGW.legend(fontsize=7, loc='lower right')
axGW.set_title('B.  Global wavelet spectrum\n(1,000-sim AR(1) null)', fontsize=7.5)

# Bottom row: 3 spatial panels
axR = fig.add_subplot(gs2[1, 0])
colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(prov_sorted)))
y_pos = np.arange(len(prov_sorted))
axR.barh(y_pos, prov_sorted['mean_annual_rate'], color=colors, edgecolor='gray', linewidth=0.4)
axR.set_yticks(y_pos); axR.set_yticklabels(prov_sorted['sido_en'], fontsize=7)
axR.set_xlabel('Mean annual incidence rate (per 100 000)')
axR.set_title('C.  Province-level mean rate, 2001–2024', fontsize=7.5); axR.grid(axis='x', alpha=0.3)

axM = fig.add_subplot(gs2[1, 1])
quad_color = {'HH':'red','LL':'steelblue','HL':'orange','LH':'green'}
for q, c in quad_color.items():
    sub = lisa[lisa.quadrant==q]
    axM.scatter(sub['z'], sub['lag_z'], color=c, s=60, edgecolor='black', label=q, zorder=3)
for _, r in lisa.iterrows():
    axM.annotate(r['sido_en'], (r['z'], r['lag_z']), fontsize=7, alpha=0.85,
                 xytext=(4,2), textcoords='offset points')
axM.axhline(0, color='gray', lw=0.5); axM.axvline(0, color='gray', lw=0.5)
slope = np.polyfit(lisa['z'], lisa['lag_z'], 1)[0]
ll = np.linspace(lisa['z'].min()-0.2, lisa['z'].max()+0.2, 50)
axM.plot(ll, slope*ll, 'k--', lw=0.8, alpha=0.6)
axM.set_xlabel('Standardised rate (z)'); axM.set_ylabel('Spatial lag (z)')
axM.set_title('D.  Moran scatter (Q+KNN-2)\nGlobal I = 0.403, p = 0.0083', fontsize=7.5)
axM.legend(title='Moran quadrant', fontsize=7, loc='upper left', framealpha=0.9)

axL = fig.add_subplot(gs2[1, 2])
lisa_sorted = lisa.sort_values('I_local', ascending=True).reset_index(drop=True)
bar_colors = [quad_color[q] for q in lisa_sorted['quadrant']]
axL.barh(np.arange(len(lisa_sorted)), lisa_sorted['I_local'], color=bar_colors, edgecolor='black', linewidth=0.5)
axL.set_yticks(np.arange(len(lisa_sorted))); axL.set_yticklabels(lisa_sorted['sido_en'], fontsize=7)
for i, r in lisa_sorted.iterrows():
    val = r['I_local']
    pad = 0.08 if val >= 0 else -0.08
    ha = 'left' if val >= 0 else 'right'
    marker = ''
    if r['sig_fdr']: marker = '†'
    if r['p_perm'] < 0.01: marker = ('** ' + marker).strip()
    elif r['p_perm'] < 0.05: marker = ('* ' + marker).strip()
    if marker:
        axL.text(val + pad, i, marker, va='center', ha=ha, fontsize=7.5, fontweight='bold')
axL.axvline(0, color='gray', lw=0.5); axL.set_xlabel("Local Moran's I_i")
axL.set_title("E.  Local Moran's I (LISA)\n* p<0.05, ** p<0.01, † q<0.05 (BH-FDR)", fontsize=7.5)

plt.savefig(f'{FIG}/Figure_2.png', dpi=600, bbox_inches='tight')
plt.savefig(f'{FIG}/Figure_2.pdf', bbox_inches='tight')
plt.close()
print('Saved new Figure 2 (5 panels A–E)')
