"""
교권침해 제도요인 정밀 분석
① 잠재요인(제도공백) 크기 정량화
② 전체 SEM 모델 R²/AUC (예측력)
③ D2 제도요인 4개 × 보호자민원/학생침해/교권침해 상관 (신규 분석)
"""
import pyreadstat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트
for fp in ['/System/Library/Fonts/Supplemental/AppleGothic.ttf','/Library/Fonts/NanumGothic.ttf']:
    try:
        fm.fontManager.addfont(fp)
        plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        break
    except Exception: pass
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────
# 0. 데이터 로드
# ─────────────────────────────────────────────
df_raw, meta = pyreadstat.read_sav(
    '/Users/baiohelseu/Desktop/Project/kossda/data/교원 인권상황 실태조사,2024/kor_data_20240073.sav'
)
N = len(df_raw)

# 변수 준비
D2_vars = {'D2_2':'아동학대\n신고보호', 'D2_5':'수업방해\n분리시스템', 'D2_10':'악성민원\n법적패널티', 'D2_16':'과밀학급\n해소'}
B_vars  = {'B2_1':'학생침해', 'B3_1':'보호자침해', 'B5_1':'관리자침해'}
C5_vars = {'C5_11':'보호자민원\n스트레스', 'C5_2':'교실질서\n스트레스', 'C5_3':'행정업무\n스트레스'}

df = pd.DataFrame()
for col in D2_vars: df[col] = df_raw[col].fillna(df_raw[col].median())
for col in B_vars:  df[col] = (df_raw[col]==1).astype(int)
for col in C5_vars: df[col] = (df_raw[col]==1).astype(int)
for i in range(1,6): df[f'C3_{i}'] = df_raw[f'C3_{i}'].fillna(df_raw[f'C3_{i}'].median())

df['침해지수']      = df[list(B_vars.keys())].sum(axis=1)
df['스트레스지수']  = df[list(C5_vars.keys())].sum(axis=1)
df['번아웃_총점']   = df[[f'C3_{i}' for i in range(1,6)]].mean(axis=1)
df['제도공백_총점'] = df[list(D2_vars.keys())].mean(axis=1)
df['정신건강역']    = 6 - df_raw['C1'].fillna(df_raw['C1'].median())
df['직업만족역']    = 6 - df_raw['A3_6'].fillna(df_raw['A3_6'].median())
df['이직고려']      = (df_raw['A4']==1).astype(int)
df['자살사고']      = (df_raw['C4']==1).astype(int)
df['초등더미']      = (df_raw['SQ2']==2).astype(int)
df['학교규모']      = df_raw['SQ5'].fillna(df_raw['SQ5'].median())

print(f"n={N:,} 로드 완료\n")

# ─────────────────────────────────────────────
# 1. 잠재요인(제도공백) 크기 정량화
#    ① 표준화 요인 부하량
#    ② 분산 설명률 (R²_indicator)
#    ③ 경로 표준화 β
# ─────────────────────────────────────────────
print("=" * 60)
print("① 잠재요인(제도공백) 크기 정량화")
print("=" * 60)

# 표준화 요인 부하량 = 상관계수 (CFA에서 단일요인일 때)
# 제도공백 = 4개 D2 항목의 공통 잠재요인
from factor_analyzer import FactorAnalyzer
inst_items = list(D2_vars.keys())
fa = FactorAnalyzer(n_factors=1, rotation=None)
fa.fit(df[inst_items])
std_loadings = fa.loadings_[:, 0]
communalities = fa.get_communalities()
ev, _ = fa.get_eigenvalues()
v_pct_inst = ev[0] / len(inst_items) * 100

print(f"\n제도공백 잠재요인 (1-factor CFA)")
print(f"  분산 설명률: {v_pct_inst:.1f}%  (eigenvalue={ev[0]:.3f})")
print(f"\n  {'항목':<16} {'레이블':<20} {'표준화 부하량':>12} {'공통분산 R²':>12}")
for item, lbl in D2_vars.items():
    i = inst_items.index(item)
    load = std_loadings[i]
    comm = communalities[i]
    print(f"  {item:<16} {lbl.replace(chr(10),' '):<20} {load:>12.3f} {comm:>12.3f}")

# 번아웃 잠재요인
burn_items = [f'C3_{i}' for i in range(1,6)]
fa_b = FactorAnalyzer(n_factors=1, rotation=None)
fa_b.fit(df[burn_items])
std_load_b = fa_b.loadings_[:, 0]
ev_b, _ = fa_b.get_eigenvalues()
v_pct_burn = ev_b[0] / len(burn_items) * 100
comm_b = fa_b.get_communalities()

burn_labels = ['고갈','탈진','피곤함','스트레스','소진']
print(f"\n번아웃 잠재요인 (1-factor CFA)")
print(f"  분산 설명률: {v_pct_burn:.1f}%  (eigenvalue={ev_b[0]:.3f})")
print(f"\n  {'항목':<10} {'레이블':<12} {'표준화 부하량':>12} {'공통분산 R²':>12}")
for item, lbl, load, comm in zip(burn_items, burn_labels, std_load_b, comm_b):
    print(f"  {item:<10} {lbl:<12} {abs(load):>12.3f} {comm:>12.3f}")

# 경로 표준화 β: 제도공백 → 침해지수
factor_scores_inst = fa.transform(df[inst_items])[:,0]
df['제도공백_FS'] = factor_scores_inst
r_inst_inj, p_inst = stats.pearsonr(df['제도공백_FS'], df['침해지수'])
print(f"\n  표준화 경로: 제도공백(FS) → 침해지수: r={r_inst_inj:.3f}  (r²={r_inst_inj**2:.3f}, p<0.001)")
print(f"  해석: 제도공백이 침해 분산의 {r_inst_inj**2*100:.1f}%를 설명")

# Cohen's d: 제도공백 고/저 그룹의 침해지수 차이
med_inst = np.median(df['제도공백_총점'])
hi_inst = df[df['제도공백_총점'] >= med_inst]['침해지수']
lo_inst = df[df['제도공백_총점'] < med_inst]['침해지수']
d_cohen = (hi_inst.mean() - lo_inst.mean()) / np.sqrt((hi_inst.std()**2 + lo_inst.std()**2)/2)
print(f"  Cohen's d (제도공백 높음 vs 낮음 → 침해지수): d={d_cohen:.3f}")

# ─────────────────────────────────────────────
# 2. 전체 모델 예측력 (R², AUC)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("② 전체 SEM 모델 예측력")
print("=" * 60)

predictors_endo = ['초등더미','학교규모','제도공백_총점']
model_specs = [
    # (종속변수, 예측변수 목록, 모델명)
    ('침해지수',   ['초등더미','학교규모','제도공백_총점'],          '침해지수 (OLS)'),
    ('스트레스지수',['침해지수','제도공백_총점'],                    '스트레스지수 (OLS)'),
    ('번아웃_총점', ['침해지수','스트레스지수','제도공백_총점'],     '번아웃 총점 (OLS)'),
    ('정신건강역',  ['번아웃_총점','침해지수'],                      '정신건강역 (OLS)'),
    ('직업만족역',  ['번아웃_총점','침해지수'],                      '직업만족역 (OLS)'),
]
r2_results = []
for y_var, x_vars, name in model_specs:
    X = sm.add_constant(df[x_vars])
    y = df[y_var]
    ols = sm.OLS(y, X).fit()
    r2_results.append((name, ols.rsquared, ols.rsquared_adj, ols.fvalue, ols.f_pvalue))
    print(f"  {name:<30} R²={ols.rsquared:.3f}  adj.R²={ols.rsquared_adj:.3f}")

# 로지스틱 AUC
all_predictors = ['번아웃_총점','침해지수','스트레스지수','제도공백_총점','초등더미','학교규모']
print()
auc_results = []
for y_bin, label in [('이직고려','이직고려 (Logit)'), ('자살사고','자살사고 (Logit)')]:
    X_log = sm.add_constant(df[all_predictors])
    lm = sm.Logit(df[y_bin], X_log).fit(disp=0)
    y_pred = lm.predict(X_log)
    auc = roc_auc_score(df[y_bin], y_pred)
    mcfadden = 1 - lm.llf / lm.llnull
    auc_results.append((label, auc, mcfadden))
    print(f"  {label:<30} AUC={auc:.3f}  McFadden R²={mcfadden:.3f}")

# ─────────────────────────────────────────────
# 3. D2 제도요인 4개 × 침해/민원 상세 분석 (신규)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("③ D2 제도요인 4개 × 교권침해 유형 × 스트레스 상관 [신규 분석]")
print("=" * 60)

# 결과 변수 정의
outcome_vars = {
    'B2_1':'학생침해', 'B3_1':'보호자침해', 'B5_1':'관리자침해',
    'C5_11':'보호자민원\n스트레스', 'C5_2':'교실질서\n스트레스',
    '침해지수':'침해지수\n(합산)',
}

print(f"\n  Spearman 상관계수 (D2 제도요인 × 침해/민원 유형)  [*** p<.001  ** p<.01  * p<.05]")
print(f"  {'':16}", end='')
for key, lbl in outcome_vars.items():
    print(f"  {lbl.replace(chr(10),' ')[:7]:>9}", end='')
print()

corr_matrix = {}
pval_matrix = {}
for d2_key, d2_lbl in D2_vars.items():
    corr_row = {}
    pval_row = {}
    print(f"  {d2_lbl.replace(chr(10),' '):<16}", end='')
    for out_key in outcome_vars:
        r, p = stats.spearmanr(df[d2_key], df[out_key])
        corr_row[out_key] = r
        pval_row[out_key] = p
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
        print(f"  {r:>5.3f}{sig:<3}", end='')
    print()
    corr_matrix[d2_key] = corr_row
    pval_matrix[d2_key] = pval_row

# 로지스틱: 각 D2 항목 → 보호자침해(B3_1) / 학생침해(B2_1) OR 비교
print(f"\n  Logistic OR: D2 제도요인 → 보호자침해(B3_1) / 학생침해(B2_1)")
print(f"  {'':20}  {'→ 보호자침해 OR':>16}  {'p':>8}  {'→ 학생침해 OR':>15}  {'p':>8}")

or_b3, or_b2 = {}, {}
for d2_key, d2_lbl in D2_vars.items():
    lbl_short = d2_lbl.replace('\n',' ')
    ors_row = {}
    for y_var, y_dict in [('B3_1', or_b3), ('B2_1', or_b2)]:
        X1 = sm.add_constant(df[[d2_key]])
        try:
            m = sm.Logit(df[y_var], X1).fit(disp=0)
            or_val = np.exp(m.params[d2_key])
            p_val  = m.pvalues[d2_key]
            ci     = np.exp(m.conf_int().loc[d2_key])
            y_dict[d2_key] = {'OR':or_val, 'p':p_val, 'lo':ci[0], 'hi':ci[1]}
        except Exception:
            y_dict[d2_key] = {'OR':np.nan, 'p':np.nan}

    sig_b3 = '***' if or_b3[d2_key]['p']<0.001 else '**' if or_b3[d2_key]['p']<0.01 else '*' if or_b3[d2_key]['p']<0.05 else 'n.s.'
    sig_b2 = '***' if or_b2[d2_key]['p']<0.001 else '**' if or_b2[d2_key]['p']<0.01 else '*' if or_b2[d2_key]['p']<0.05 else 'n.s.'
    print(f"  {lbl_short:<20}  {or_b3[d2_key]['OR']:>10.3f} {sig_b3:<4}  {or_b3[d2_key]['p']:>8.4f}  "
          f"{or_b2[d2_key]['OR']:>9.3f} {sig_b2:<4}  {or_b2[d2_key]['p']:>8.4f}")

# 침해 유형별 D2 평균 비교 (침해 있음 vs 없음)
print(f"\n  침해 경험 여부별 D2 정책 필요도 평균 차이 (Mann-Whitney)")
print(f"  {'':20}  {'보호자침해_있음':>14}  {'보호자침해_없음':>14}  {'차이':>8}  {'p':>10}  {'효과크기(r)':>10}")
for d2_key, d2_lbl in D2_vars.items():
    lbl_short = d2_lbl.replace('\n',' ')
    grp1 = df.loc[df['B3_1']==1, d2_key]
    grp0 = df.loc[df['B3_1']==0, d2_key]
    u, p = stats.mannwhitneyu(grp1, grp0, alternative='two-sided')
    r_eff = 1 - 2*u/(len(grp1)*len(grp0))  # rank-biserial correlation
    diff = grp1.mean() - grp0.mean()
    print(f"  {lbl_short:<20}  {grp1.mean():>14.3f}  {grp0.mean():>14.3f}  {diff:>8.3f}  {p:>10.4f}  {r_eff:>10.4f}")

# 과밀학급 × 학교규모 × 침해 추가
print(f"\n  학교 규모(SQ5)별 학생침해율 및 보호자침해율")
for sz in sorted(df['학교규모'].unique()):
    sz_lbl = {1:'5학급미만',2:'6-11학급',3:'12-17학급',4:'18-35학급',5:'36학급이상'}.get(sz,'?')
    sub = df[df['학교규모']==sz]
    r_stu = sub['B2_1'].mean()*100
    r_par = sub['B3_1'].mean()*100
    print(f"  {sz_lbl:<12}: 학생침해={r_stu:.1f}%  보호자침해={r_par:.1f}%  n={len(sub)}")

# ─────────────────────────────────────────────
# 4. 시각화
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(22, 16))
fig.suptitle('제도공백 잠재요인 크기 · 모델 예측력 · D2 요인 × 교권침해 상관 분석',
             fontsize=14, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

# ── P1: CFA 부하량 비교 (번아웃 vs 제도공백) ──
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor('#F8F9FA')
labels_inst = [l.replace('\n',' ') for l in D2_vars.values()]
labels_burn = ['고갈','탈진','피곤함','스트레스','소진']
x1 = np.arange(len(labels_inst))
x2 = np.arange(len(labels_burn))

ax1_twin = ax1.twinx()
b1 = ax1.bar(x1 - 0.2, np.abs(std_loadings), 0.35, color='#EF5350', alpha=0.85, label='제도공백 부하량')
b2 = ax1_twin.bar(x1 + 0.2, list(df[inst_items].var()), 0.35, color='#BDBDBD', alpha=0.5, label='분산(제도공백)')

for i, (v, c) in enumerate(zip(np.abs(std_loadings), communalities)):
    ax1.text(i-0.2, v+0.01, f'{v:.2f}', ha='center', fontsize=8, color='#C62828')

ax1.set_xticks(x1)
ax1.set_xticklabels(labels_inst, fontsize=8)
ax1.set_ylabel('표준화 부하량', color='#C62828', fontsize=9)
ax1.set_ylim(0, 1.1)
ax1_twin.set_ylabel('분산', fontsize=8, color='gray')
ax1.axhline(0.7, color='red', linestyle='--', alpha=0.4, linewidth=1)
ax1.set_title(f'제도공백 CFA 표준화 부하량\n(분산 설명률 {v_pct_inst:.0f}%)', fontsize=9, fontweight='bold')
ax1.legend(loc='upper left', fontsize=7)

# ── P2: R² / AUC 막대 ──
ax2 = fig.add_subplot(gs[0, 2:])
ax2.set_facecolor('#F8F9FA')
model_names = [x[0].replace(' (OLS)','').replace(' (Logit)','') for x in r2_results] + \
              [x[0].replace(' (Logit)','') for x in auc_results]
r2_vals = [x[1] for x in r2_results] + [None, None]
auc_vals = [None]*len(r2_results) + [x[1] for x in auc_results]
colors_r2 = []
r2_plot = []
bar_labels = []
for name, r2, adj_r2, f, fp in r2_results:
    r2_plot.append(r2)
    bar_labels.append(name.replace(' (OLS)',''))
    colors_r2.append('#5C6BC0')
for name, auc, mcf in auc_results:
    r2_plot.append(auc)
    bar_labels.append(name.replace(' (Logit)',''))
    colors_r2.append('#EF5350')

bars2 = ax2.barh(bar_labels, r2_plot, color=colors_r2, edgecolor='white', height=0.6)
ax2.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
ax2.axvline(0.3, color='orange', linestyle=':', alpha=0.5)
for bar, val in zip(bars2, r2_plot):
    ax2.text(val+0.005, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', fontsize=8)
ax2.set_xlim(0, 0.65)
ax2.set_xlabel('R² (OLS) / AUC (Logit)')
ax2.set_title('전체 모델 예측력\n파랑=R²(OLS), 빨강=AUC(Logit)', fontsize=9, fontweight='bold')
# 기준선 범례
from matplotlib.lines import Line2D
ax2.legend(handles=[
    Line2D([0],[0], color='gray', linestyle=':', label='AUC 0.50 (랜덤)'),
    Line2D([0],[0], color='orange', linestyle=':', label='R² 0.30 (보통)'),
], fontsize=7, loc='lower right')

# ── P3: 상관 히트맵 (D2 × 침해 유형) ──
ax3 = fig.add_subplot(gs[1, :2])
ax3.set_facecolor('#F8F9FA')
hmap_data = pd.DataFrame(corr_matrix).T
pmap_data = pd.DataFrame(pval_matrix).T
out_lbls = [l.replace('\n',' ') for l in outcome_vars.values()]
inst_lbls = [l.replace('\n',' ') for l in D2_vars.values()]
im = ax3.imshow(hmap_data.values, cmap='RdYlGn', vmin=-0.2, vmax=0.2, aspect='auto')
ax3.set_xticks(range(len(out_lbls)))
ax3.set_xticklabels(out_lbls, fontsize=7, rotation=20, ha='right')
ax3.set_yticks(range(len(inst_lbls)))
ax3.set_yticklabels(inst_lbls, fontsize=8)
for i in range(len(inst_lbls)):
    for j in range(len(out_lbls)):
        r_val = hmap_data.values[i,j]
        p_val = pmap_data.values[i,j]
        sig = '***' if p_val<0.001 else '**' if p_val<0.01 else '*' if p_val<0.05 else ''
        ax3.text(j, i, f'{r_val:.2f}{sig}', ha='center', va='center', fontsize=8,
                 fontweight='bold', color='black' if abs(r_val)<0.15 else 'white')
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
ax3.set_title('Spearman 상관계수\nD2 제도요인 × 교권침해 유형', fontsize=9, fontweight='bold')

# ── P4: OR 비교 (보호자침해 vs 학생침해) ──
ax4 = fig.add_subplot(gs[1, 2:])
ax4.set_facecolor('#F8F9FA')
d2_lbls = [l.replace('\n',' ') for l in D2_vars.values()]
or_b3_vals = [or_b3[k]['OR'] for k in D2_vars]
or_b2_vals = [or_b2[k]['OR'] for k in D2_vars]
or_b3_lo   = [or_b3[k]['lo'] for k in D2_vars]
or_b3_hi   = [or_b3[k]['hi'] for k in D2_vars]
or_b2_lo   = [or_b2[k]['lo'] for k in D2_vars]
or_b2_hi   = [or_b2[k]['hi'] for k in D2_vars]

x_pos = np.arange(len(d2_lbls))
ax4.bar(x_pos - 0.2, or_b3_vals, 0.35, color='#EF5350', alpha=0.85, label='보호자침해 OR')
ax4.bar(x_pos + 0.2, or_b2_vals, 0.35, color='#5C6BC0', alpha=0.85, label='학생침해 OR')

# 에러바
ax4.errorbar(x_pos - 0.2, or_b3_vals,
             yerr=[np.array(or_b3_vals)-np.array(or_b3_lo), np.array(or_b3_hi)-np.array(or_b3_vals)],
             fmt='none', color='black', capsize=3, linewidth=1.2)
ax4.errorbar(x_pos + 0.2, or_b2_vals,
             yerr=[np.array(or_b2_vals)-np.array(or_b2_lo), np.array(or_b2_hi)-np.array(or_b2_vals)],
             fmt='none', color='black', capsize=3, linewidth=1.2)

for i, (v3, v2) in enumerate(zip(or_b3_vals, or_b2_vals)):
    ax4.text(i-0.2, v3+0.01, f'{v3:.2f}', ha='center', fontsize=8, color='#C62828')
    ax4.text(i+0.2, v2+0.01, f'{v2:.2f}', ha='center', fontsize=8, color='#1A237E')

ax4.axhline(1.0, color='black', linestyle='--', linewidth=1.0, label='OR=1 (기준)')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(d2_lbls, fontsize=8)
ax4.set_ylabel('Odds Ratio')
ax4.set_title('제도요인별 침해 유형 예측력 (Logistic OR, 95% CI)\n빨강=보호자침해, 파랑=학생침해', fontsize=9, fontweight='bold')
ax4.legend(fontsize=7)
ax4.set_ylim(0.5, 3.0)

# ── P5: 침해 있음/없음별 D2 평균 비교 (보호자침해) ──
ax5 = fig.add_subplot(gs[2, :2])
ax5.set_facecolor('#F8F9FA')
grp1_means = [df.loc[df['B3_1']==1, k].mean() for k in D2_vars]
grp0_means = [df.loc[df['B3_1']==0, k].mean() for k in D2_vars]
diffs = [m1-m0 for m1, m0 in zip(grp1_means, grp0_means)]

bars5a = ax5.bar(x_pos - 0.2, grp1_means, 0.35, color='#EF5350', alpha=0.85, label='보호자침해 있음')
bars5b = ax5.bar(x_pos + 0.2, grp0_means, 0.35, color='#90CAF9', alpha=0.85, label='보호자침해 없음')
for i, (d, g1, g0) in enumerate(zip(diffs, grp1_means, grp0_means)):
    ax5.text(i, max(g1,g0)+0.01, f'Δ{d:+.3f}', ha='center', fontsize=8, color='#B71C1C', fontweight='bold')
    ax5.text(i-0.2, g1-0.02, f'{g1:.2f}', ha='center', va='top', fontsize=7)
    ax5.text(i+0.2, g0-0.02, f'{g0:.2f}', ha='center', va='top', fontsize=7)

ax5.set_xticks(x_pos)
ax5.set_xticklabels(d2_lbls, fontsize=8)
ax5.set_ylabel('D2 정책 필요도 평균 (1-5)')
ax5.set_ylim(4.5, 5.05)
ax5.set_title('보호자침해 경험 여부별 D2 정책 필요도 차이\n(Δ= 있음 - 없음)', fontsize=9, fontweight='bold')
ax5.legend(fontsize=8)

# ── P6: 학교규모 × 침해율 ──
ax6 = fig.add_subplot(gs[2, 2:])
ax6.set_facecolor('#F8F9FA')
sizes = sorted(df['학교규모'].unique())
size_lbls = {1:'5학급미만',2:'6-11학급',3:'12-17학급',4:'18-35학급',5:'36학급이상'}
stu_rates = [df[df['학교규모']==s]['B2_1'].mean()*100 for s in sizes]
par_rates = [df[df['학교규모']==s]['B3_1'].mean()*100 for s in sizes]
ns_grp    = [len(df[df['학교규모']==s]) for s in sizes]

x6 = np.arange(len(sizes))
ax6.plot(x6, stu_rates, 'o-', color='#5C6BC0', linewidth=2, markersize=8, label='학생침해율')
ax6.plot(x6, par_rates, 's-', color='#EF5350', linewidth=2, markersize=8, label='보호자침해율')
for i, (sr, pr, n) in enumerate(zip(stu_rates, par_rates, ns_grp)):
    ax6.text(i, sr+0.5, f'{sr:.0f}%', ha='center', fontsize=8, color='#1A237E')
    ax6.text(i, pr-1.5, f'{pr:.0f}%', ha='center', fontsize=8, color='#B71C1C')
ax6.set_xticks(x6)
ax6.set_xticklabels([size_lbls[s] for s in sizes], fontsize=8)
ax6.set_ylabel('침해율 (%)')
ax6.set_title('학교 규모(학급 수)별 침해율\n과밀학급 해소 정책의 근거', fontsize=9, fontweight='bold')
ax6.legend(fontsize=8)

# 우상단: 분산 설명 비율 파이
ax_inset = fig.add_axes([0.62, 0.64, 0.12, 0.12])
comm_vals = list(communalities)
ax_inset.pie([comm_vals[0], 1-comm_vals[0]], labels=['공통\n분산', '잔차'], startangle=90,
             colors=['#EF5350','#EEEEEE'], autopct='%1.0f%%', textprops={'fontsize':6})
ax_inset.set_title('D2_2\nR²', fontsize=6)

plt.tight_layout()
out_path = '/Users/baiohelseu/Desktop/Project/kossda/output/factor_impact.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print(f"\n저장 완료: {out_path}")
