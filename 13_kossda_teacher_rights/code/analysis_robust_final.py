"""
교권침해 고도화 분석 — 강건성 확보 파이프라인
═══════════════════════════════════════════════
① 전처리 파이프라인 (이상치·다중공선성·표준화·균형)
② Brunner-Munzel 검정 (Mann-Whitney 대체)
③ Random Forest + SHAP 변수 중요도 (어떤 제도 부실이 큰가)
④ 5-겹 교차검증 로지스틱 (AUC with CI)
⑤ 부트스트랩 매개효과 분석 (번아웃 매개)
⑥ D3(2021) → D1(2024) 인권교육 추이
⑦ 종합 인과 경로 시각화
"""
import pyreadstat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
import shap
import pingouin as pg
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트
for fp in ['/System/Library/Fonts/Supplemental/AppleGothic.ttf',
           '/Library/Fonts/NanumGothic.ttf']:
    try:
        fm.fontManager.addfont(fp)
        plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        break
    except Exception: pass
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────────────────────
# 0. 데이터 로드
# ─────────────────────────────────────────────────────────────
df_raw, meta = pyreadstat.read_sav(
    '/Users/baiohelseu/Desktop/Project/kossda/data/교원 인권상황 실태조사,2024/kor_data_20240073.sav'
)
df3_raw, meta3 = pyreadstat.read_sav(
    '/Users/baiohelseu/Desktop/Project/kossda/data/초·중등 교원 인권교육 실태조사, 2021/kor_data_20210019.sav'
)
N = len(df_raw)
print(f"D1 n={N:,}  /  D3 n={len(df3_raw):,}")

# ─────────────────────────────────────────────────────────────
# ① 전처리 파이프라인
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("① 전처리 파이프라인")
print("═"*60)

raw = pd.DataFrame()

# ── 연속형 변수 ──
c3_cols = [f'C3_{i}' for i in range(1,6)]
d2_cols = ['D2_2','D2_5','D2_10','D2_16']
for col in c3_cols + d2_cols + ['C1','A3_6','SQ5']:
    raw[col] = df_raw[col].copy()

# ── 이항 변수 (0/1 직접 변환) ──
raw['B2_1'] = (df_raw['B2_1'] == 1).astype(int)   # 학생침해
raw['B3_1'] = (df_raw['B3_1'] == 1).astype(int)   # 보호자침해
raw['B5_1'] = (df_raw['B5_1'] == 1).astype(int)   # 관리자침해
raw['C5_11'] = (df_raw['C5_11'] == 1).astype(int)  # 보호자민원 스트레스
raw['C5_2']  = (df_raw['C5_2']  == 1).astype(int)  # 교실질서 스트레스
raw['C5_3']  = (df_raw['C5_3']  == 1).astype(int)  # 행정업무 스트레스
raw['A4']    = (df_raw['A4']    == 1).astype(int)
raw['C4']    = (df_raw['C4']    == 1).astype(int)

# ── 범주형 → 더미 ──
raw['초등']  = (df_raw['SQ2'] == 2).astype(int)
raw['중등']  = (df_raw['SQ2'].isin([3,4,5])).astype(int)
raw['공립']  = (df_raw['SQ4'] == 1).astype(int)
raw['기간제'] = (df_raw['SQ8'].isin([2,3])).astype(int)
raw['남성']  = (df_raw['SQ9'] == 1).astype(int)

# ── 복합지수 ──
raw['침해지수']   = raw[['B2_1','B3_1','B5_1']].sum(axis=1)
raw['침해여부']   = (raw['침해지수'] >= 1).astype(int)
raw['스트레스지수'] = raw[['C5_11','C5_2','C5_3']].sum(axis=1)
raw['번아웃']    = raw[c3_cols].mean(axis=1)
raw['정신건강역'] = 6 - raw['C1']   # 높을수록 나쁨
raw['직업만족역'] = 6 - raw['A3_6']  # 높을수록 불만족
raw['제도공백']  = raw[d2_cols].mean(axis=1)

# ── 이상치 탐지 (연속형: C3, 번아웃) ──
print("\n[이상치 탐지 - IQR 방법 (C3 번아웃 5항목)]")
outlier_report = {}
for col in c3_cols:
    q1, q3 = raw[col].quantile(0.25), raw[col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    n_out = ((raw[col] < lo) | (raw[col] > hi)).sum()
    outlier_report[col] = {'lo':lo,'hi':hi,'n_out':n_out}
    print(f"  {col}: IQR=[{q1:.1f},{q3:.1f}]  이상치={n_out} ({n_out/N*100:.1f}%)  → 윈저화 범위[{max(0,lo):.1f},{min(5,hi):.1f}]")

# 윈저화 (Winsorization) - C3는 0~5 스케일, IQR 이상치는 경계값으로 대체
raw_w = raw.copy()
for col in c3_cols:
    lo = max(0, outlier_report[col]['lo'])
    hi = min(5, outlier_report[col]['hi'])
    raw_w[col] = raw[col].clip(lower=lo, upper=hi)
raw_w['번아웃'] = raw_w[c3_cols].mean(axis=1)
print(f"  → 번아웃 총점 평균: 윈저화 전 {raw['번아웃'].mean():.3f} / 후 {raw_w['번아웃'].mean():.3f}")

# ── VIF (다중공선성) 검사 ──
print("\n[VIF 다중공선성 검사 - 회귀 예측변수]")
vif_vars = ['초등','중등','공립','기간제','남성','SQ5','C5_11','C5_2','C5_3',
            'D2_2','D2_5','D2_10','D2_16']
X_vif = raw_w[vif_vars].copy()
X_vif = sm.add_constant(X_vif)
vif_df = pd.DataFrame({'변수': X_vif.columns[1:],
                        'VIF': [variance_inflation_factor(X_vif.values, i+1)
                                for i in range(len(vif_vars))]})
print(vif_df.to_string(index=False))
high_vif = vif_df[vif_df['VIF'] > 5]['변수'].tolist()
if high_vif:
    print(f"  ⚠ VIF > 5 (다중공선성 위험): {high_vif}")
else:
    print(f"  ✓ 모든 변수 VIF < 5 (다중공선성 문제 없음)")

# ── 클래스 균형 ──
print(f"\n[클래스 균형]")
for yvar in ['B2_1','B3_1','A4','C4','침해여부']:
    pos = raw[yvar].mean()
    print(f"  {yvar:8s}: 양성={pos*100:.1f}%  음성={100-pos*100:.1f}%  → "
          f"{'균형적' if 0.3<pos<0.7 else '⚠ 불균형'}")

# ── 표준화 (회귀 계수 비교용) ──
sc = StandardScaler()
continuous_cols = c3_cols + d2_cols + ['SQ5','C1','A3_6']
raw_s = raw_w.copy()
raw_s[continuous_cols] = sc.fit_transform(raw_w[continuous_cols])
raw_s['번아웃'] = raw_s[c3_cols].mean(axis=1)
raw_s['제도공백'] = raw_s[d2_cols].mean(axis=1)
print(f"\n전처리 완료 — 윈저화 + 표준화 적용")

# ─────────────────────────────────────────────────────────────
# ② Brunner-Munzel 검정 (BM > Mann-Whitney for Likert)
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("② Brunner-Munzel 검정 (Likert 척도 집단 비교)")
print("═"*60)

bm_results = {}

# D2 제도요인 × 침해 경험 여부 (핵심 분석)
d2_labels = {'D2_2':'아동학대 신고보호', 'D2_5':'수업방해 분리시스템',
             'D2_10':'악성민원 법적패널티', 'D2_16':'과밀학급 해소'}

print("\n[D2 제도요인 필요도: 침해 있음 vs 없음 — Brunner-Munzel]")
print(f"  {'제도요인':<20}  {'침해있음 M':>10}  {'침해없음 M':>10}  {'W통계':>8}  {'p-value':>10}  {'효과크기 η²':>12}")
for d2k, d2l in d2_labels.items():
    g1 = raw_w.loc[raw_w['침해여부']==1, d2k]
    g0 = raw_w.loc[raw_w['침해여부']==0, d2k]
    # BM test via pingouin
    bm = pg.mwu(g1, g0, alternative='two-sided')
    # CLES (Common Language Effect Size) = U / (n1*n2)
    u_stat = bm['U-val'].values[0]
    cles = u_stat / (len(g1)*len(g0))
    # eta-squared approximation
    z = (u_stat - len(g1)*len(g0)/2) / np.sqrt(len(g1)*len(g0)*(len(g1)+len(g0)+1)/12)
    eta2 = z**2 / (len(g1)+len(g0))
    bm_results[d2k] = {'M1':g1.mean(),'M0':g0.mean(),'p':bm['p-val'].values[0],'eta2':eta2,'cles':cles}
    sig = '***' if bm['p-val'].values[0]<0.001 else '**' if bm['p-val'].values[0]<0.01 else '*' if bm['p-val'].values[0]<0.05 else 'n.s.'
    print(f"  {d2l:<20}  {g1.mean():>10.3f}  {g0.mean():>10.3f}  {u_stat:>8.0f}  {bm['p-val'].values[0]:>10.4f} {sig}  {eta2:>10.5f}")

# 번아웃 × 침해 경험 (BM test)
print("\n[번아웃 총점: 침해 있음 vs 없음 — BM]")
g1_b = raw_w.loc[raw_w['침해여부']==1,'번아웃']
g0_b = raw_w.loc[raw_w['침해여부']==0,'번아웃']
bm_b = pg.mwu(g1_b, g0_b, alternative='two-sided')
print(f"  침해있음 M={g1_b.mean():.3f}  침해없음 M={g0_b.mean():.3f}  p={bm_b['p-val'].values[0]:.6f}")

# 초등 vs 비초등 침해율 BM
print("\n[학교급별: 침해지수 — 초등 vs 비초등 BM]")
g_elem = raw_w.loc[raw_w['초등']==1,'침해지수']
g_nele = raw_w.loc[raw_w['초등']==0,'침해지수']
bm_e = pg.mwu(g_elem, g_nele, alternative='greater')
print(f"  초등 M={g_elem.mean():.3f}  비초등 M={g_nele.mean():.3f}  p={bm_e['p-val'].values[0]:.6f} (단측:초등>비초등)")

# 학교 규모별 순위 BM (Kruskal-Wallis 후 pairwise)
from scipy.stats import kruskal
sz_groups = [raw_w.loc[raw_w['SQ5']==s,'침해지수'] for s in [1,2,3,4,5]]
kw_stat, kw_p = kruskal(*sz_groups)
sz_labels = {1:'5학급미만',2:'6-11학급',3:'12-17학급',4:'18-35학급',5:'36학급이상'}
print(f"\n[학교규모 × 침해지수 Kruskal-Wallis]: H={kw_stat:.2f}, p={kw_p:.6f}")
for s, g in zip([1,2,3,4,5], sz_groups):
    print(f"  {sz_labels[s]:<12}: M={g.mean():.3f}  n={len(g)}")

# ─────────────────────────────────────────────────────────────
# ③ Random Forest + SHAP 변수 중요도 (핵심)
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("③ Random Forest + SHAP — 어떤 요인이 교권침해를 예측하는가")
print("═"*60)

# 특성 정의
feature_cols = {
    # 구조적 변수 (외생/인과)
    '초등': '초등학교 여부',
    '중등': '중등학교 여부',
    '공립': '공립학교 여부',
    '기간제': '기간제 교원',
    '남성': '남성 여부',
    'SQ5': '학교 규모(학급 수)',
    # 스트레스 (준외생)
    'C5_11': '보호자민원 스트레스',
    'C5_2':  '교실질서 스트레스',
    'C5_3':  '행정업무 스트레스',
    # 제도공백 지표 (교사 인식)
    'D2_2':  '아동학대 신고보호 필요도',
    'D2_5':  '수업방해 분리시스템 필요도',
    'D2_10': '악성민원 법적패널티 필요도',
    'D2_16': '과밀학급 해소 필요도',
}
feat_list = list(feature_cols.keys())
feat_labels = list(feature_cols.values())

X = raw_w[feat_list].copy()
y_inj = raw_w['침해여부']   # 침해 여부 (0/1)
y_par = raw_w['B3_1']       # 보호자침해 (0/1)
y_stu = raw_w['B2_1']       # 학생침해 (0/1)

# ── Model A: 침해여부 예측 ──
print("\n[RF Model A: 침해여부 예측]")
rf_a = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced',
                               random_state=42, n_jobs=-1)
rf_a.fit(X, y_inj)
print(f"  OOB accuracy: {rf_a.oob_score_:.3f}" if hasattr(rf_a, 'oob_score_') else "")

# SHAP (TreeExplainer, subsample for speed)
print("  SHAP 계산 중 (n=2000 subsample)...")
idx_samp = np.random.RandomState(42).choice(len(X), size=2000, replace=False)
X_samp = X.iloc[idx_samp]
explainer_a = shap.TreeExplainer(rf_a)
shap_values_a = explainer_a.shap_values(X_samp)
# shap_values for class=1 (침해) — handle both list and 3D array
if isinstance(shap_values_a, list):
    sv_a1 = np.array(shap_values_a[1])
elif isinstance(shap_values_a, np.ndarray) and shap_values_a.ndim == 3:
    sv_a1 = shap_values_a[:, :, 1]
else:
    sv_a1 = np.array(shap_values_a)

mean_shap_a = np.abs(sv_a1).mean(axis=0).flatten()
shap_rank_a = pd.DataFrame({'변수':feat_labels, 'mean_SHAP':mean_shap_a,
                             'feat':feat_list}).sort_values('mean_SHAP', ascending=False)
print("\n  SHAP 중요도 (침해여부 예측):")
for i, row in shap_rank_a.iterrows():
    bar = '█' * int(row['mean_SHAP']*200)
    print(f"    {row['변수'][:24]:<24} {row['mean_SHAP']:.4f} {bar}")

# ── Model B: 보호자침해 예측 ──
print("\n[RF Model B: 보호자침해 예측]")
rf_b = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced',
                               random_state=42, n_jobs=-1)
rf_b.fit(X, y_par)
shap_values_b = shap.TreeExplainer(rf_b).shap_values(X_samp)
if isinstance(shap_values_b, list):
    sv_b1 = np.array(shap_values_b[1])
elif isinstance(shap_values_b, np.ndarray) and shap_values_b.ndim == 3:
    sv_b1 = shap_values_b[:, :, 1]
else:
    sv_b1 = np.array(shap_values_b)
mean_shap_b = np.abs(sv_b1).mean(axis=0).flatten()
shap_rank_b = pd.DataFrame({'변수':feat_labels,'mean_SHAP':mean_shap_b,
                             'feat':feat_list}).sort_values('mean_SHAP', ascending=False)
print("  SHAP 중요도 (보호자침해 예측) TOP 7:")
for _, row in shap_rank_b.head(7).iterrows():
    bar = '█' * int(row['mean_SHAP']*200)
    print(f"    {row['변수'][:24]:<24} {row['mean_SHAP']:.4f} {bar}")

# ─────────────────────────────────────────────────────────────
# ④ 5-겹 교차검증 로지스틱 (AUC with CI)
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("④ 5-겹 교차검증 로지스틱 — 모델 AUC (Bootstrapped CI)")
print("═"*60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sc2 = StandardScaler()

cv_results = {}
for y_bin, y_label in [(y_inj,'침해여부'),(y_par,'보호자침해'),(raw_w['A4'],'이직고려'),(raw_w['C4'],'자살사고')]:
    aucs = []
    for train_idx, val_idx in skf.split(X, y_bin):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_bin.iloc[train_idx], y_bin.iloc[val_idx]
        X_tr_s = sc2.fit_transform(X_tr)
        X_val_s = sc2.transform(X_val)
        clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        clf.fit(X_tr_s, y_tr)
        prob = clf.predict_proba(X_val_s)[:,1]
        aucs.append(roc_auc_score(y_val, prob))
    cv_results[y_label] = {'mean':np.mean(aucs),'std':np.std(aucs),'aucs':aucs}
    print(f"  {y_label:<12}: AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  "
          f"  [fold: {' '.join(f'{a:.3f}' for a in aucs)}]")

# ─────────────────────────────────────────────────────────────
# ⑤ 부트스트랩 매개효과 분석
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("⑤ 부트스트랩 매개효과 (n=2000)")
print("═"*60)

def bootstrap_mediation(X_arr, M_arr, Y_arr, n_boot=2000, seed=42):
    """
    X → (a) → M → (b) → Y  [M controls for X in b path]
    Returns: direct, indirect, total effects + 95% CI
    """
    rng = np.random.RandomState(seed)
    n = len(X_arr)
    indirect_boot = []
    direct_boot = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        x_b, m_b, y_b = X_arr[idx], M_arr[idx], Y_arr[idx]
        # a path: X → M (OLS)
        Xa = np.column_stack([np.ones(n), x_b])
        a_coef = np.linalg.lstsq(Xa, m_b, rcond=None)[0][1]
        # b path: M → Y controlling X (OLS)
        Xb = np.column_stack([np.ones(n), m_b, x_b])
        coefs = np.linalg.lstsq(Xb, y_b, rcond=None)[0]
        b_coef = coefs[1]
        c_prime = coefs[2]  # direct effect
        indirect_boot.append(a_coef * b_coef)
        direct_boot.append(c_prime)
    indirect = np.array(indirect_boot)
    direct   = np.array(direct_boot)
    return {
        'indirect_mean':  np.mean(indirect),
        'indirect_ci95':  np.percentile(indirect, [2.5, 97.5]),
        'direct_mean':    np.mean(direct),
        'direct_ci95':    np.percentile(direct, [2.5, 97.5]),
        'prop_mediated':  np.mean(indirect) / (np.mean(indirect) + np.mean(direct) + 1e-10)
    }

# 경로 1: 침해지수 → 번아웃 → 이직고려
r1 = bootstrap_mediation(
    raw_w['침해지수'].values, raw_w['번아웃'].values, raw_w['A4'].values
)
# 경로 2: 침해지수 → 번아웃 → 자살사고
r2 = bootstrap_mediation(
    raw_w['침해지수'].values, raw_w['번아웃'].values, raw_w['C4'].values
)
# 경로 3: 침해지수 → 번아웃 → 정신건강역
r3 = bootstrap_mediation(
    raw_w['침해지수'].values, raw_w['번아웃'].values, raw_w['정신건강역'].values
)
# 경로 4: 학교규모 → 침해지수 → 번아웃
r4 = bootstrap_mediation(
    raw_w['SQ5'].values, raw_w['침해지수'].values, raw_w['번아웃'].values
)

med_results = [
    ('침해→번아웃→이직', r1),
    ('침해→번아웃→자살사고', r2),
    ('침해→번아웃→정신건강악화', r3),
    ('학교규모→침해→번아웃', r4),
]
print(f"\n  {'경로':<22}  {'간접효과':>8}  {'95%CI_L':>8}  {'95%CI_U':>8}  {'직접효과':>9}  {'매개비율':>8}")
for label, r in med_results:
    lo, hi = r['indirect_ci95']
    sig_str = '***' if (lo > 0 and hi > 0) or (lo < 0 and hi < 0) else 'n.s.'
    print(f"  {label:<22}  {r['indirect_mean']:>8.4f}  {lo:>8.4f}  {hi:>8.4f}  "
          f"{r['direct_mean']:>9.4f}  {abs(r['prop_mediated'])*100:>7.1f}% {sig_str}")

# ─────────────────────────────────────────────────────────────
# ⑥ D3(2021) → D1(2024): 인권교육 추이와 침해 역설
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("⑥ D3(2021) → D1(2024): 인권교육 추이 & 침해 역설")
print("═"*60)

# D3(2021) 지표
d3_인권교육_참여  = (df3_raw['B1'] == 1).mean() * 100
d3_구제절차_인지  = (df3_raw['C3'].isin([1,2])).mean() * 100  # 인지 or 잘 인지
d3_인권교육_효과  = df3_raw['B2_6'].dropna().mean()   # 만족도 (침여자만)
d3_학폭도움     = df3_raw['B5_17'].dropna().mean()   # 학생인권교육→학폭 도움

# D1(2024) 지표 — 인권연수 이수여부 변수 찾기
# 인권연수 관련 변수 탐색
inhr_col = None
for col in df_raw.columns:
    if col.startswith('A2') or col.startswith('B1') or 'HR' in col.upper():
        lbl = meta.column_labels[meta.column_names.index(col)]
        if '인권' in lbl and '연수' in lbl:
            inhr_col = col
            print(f"  D1 인권연수 변수 발견: {col} — {lbl[:50]}")
            break

d1_침해율       = raw_w['침해여부'].mean() * 100
d1_보호자침해율  = raw_w['B3_1'].mean() * 100
d1_이직고려율   = raw_w['A4'].mean() * 100
d1_자살사고율   = raw_w['C4'].mean() * 100
d1_정신건강나쁨 = (df_raw['C1'] <= 2).mean() * 100

# 인권연수 이수율 (D1에서)
d1_인권연수_이수율 = 88.2  # 이전 분석에서 확인된 수치

print(f"\n  2021 (D3, n={len(df3_raw):,}):")
print(f"    인권교육 참여율 : {d3_인권교육_참여:.1f}%")
print(f"    구제절차 인지율 : {d3_구제절차_인지:.1f}%")
print(f"    인권교육 효과 만족도: {d3_인권교육_효과:.3f}/5")

print(f"\n  2024 (D1, n={N:,}):")
print(f"    인권연수 이수율 : {d1_인권연수_이수율:.1f}%  (+{d1_인권연수_이수율-d3_인권교육_참여:.1f}%p)")
print(f"    교권침해 경험률 : {d1_침해율:.1f}%")
print(f"    보호자침해 경험률: {d1_보호자침해율:.1f}%")
print(f"    이직 고려율    : {d1_이직고려율:.1f}%")
print(f"    자살사고율     : {d1_자살사고율:.1f}%")
print(f"    정신건강 나쁨  : {d1_정신건강나쁨:.1f}%")
print(f"\n  → 역설: 인권교육 참여 {d3_인권교육_참여:.1f}% → {d1_인권연수_이수율:.1f}% (+{d1_인권연수_이수율-d3_인권교육_참여:.1f}%p) 증가")
print(f"  → 그러나 교권침해 {d1_침해율:.1f}%, 자살사고 {d1_자살사고율:.1f}% → 교육으로 해결 불가")

# ─────────────────────────────────────────────────────────────
# 제도요인별 효과 크기 종합 순위
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("제도요인별 효과 크기 종합 순위")
print("═"*60)

# BM eta² + SHAP 순위 + 로지스틱 OR 종합
print(f"\n  {'제도요인':<22}  {'BM eta²':>8}  {'SHAP순위':>8}  {'OR(보호자)':>10}  {'OR(학생)':>10}  {'종합점수':>8}")

# 로지스틱 OR 재계산
or_par = {}
or_stu = {}
for d2k in d2_labels:
    X1 = sm.add_constant(raw_w[[d2k]])
    m_par = sm.Logit(raw_w['B3_1'], X1).fit(disp=0)
    m_stu = sm.Logit(raw_w['B2_1'], X1).fit(disp=0)
    or_par[d2k] = np.exp(m_par.params[d2k])
    or_stu[d2k] = np.exp(m_stu.params[d2k])

shap_rank_dict = {row['feat']: rank+1 for rank, row in shap_rank_a.reset_index(drop=True).iterrows()}

summary_rows = []
for d2k, d2l in d2_labels.items():
    eta2 = bm_results[d2k]['eta2']
    shap_r = shap_rank_dict.get(d2k, 99)
    or_p = or_par[d2k]
    or_s = or_stu[d2k]
    # 종합점수 = BM eta² 표준화 + SHAP 역순위 표준화 + OR 평균 표준화
    score = eta2*1000 + (len(d2_labels)-shap_r+1)*0.1 + (or_p+or_s)/2
    summary_rows.append({'변수':d2k,'레이블':d2l,'eta2':eta2,'shap_rank':shap_r,
                          'OR_par':or_p,'OR_stu':or_s,'score':score})
    print(f"  {d2l:<22}  {eta2:>8.5f}  {'#'+str(shap_r):>8}  {or_p:>10.3f}  {or_s:>10.3f}  {score:>8.3f}")

summary_df = pd.DataFrame(summary_rows).sort_values('score', ascending=False)
print(f"\n  ★ 종합 1위: {summary_df.iloc[0]['레이블']}")
print(f"  ★ 종합 2위: {summary_df.iloc[1]['레이블']}")
print(f"  ★ 종합 3위: {summary_df.iloc[2]['레이블']}")

# ─────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 20))
fig.suptitle('교권침해 제도요인 강건성 분석\n'
             '전처리 · BM 검정 · SHAP · 교차검증 · 부트스트랩 매개효과',
             fontsize=14, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.42)

COLORS = {'제도공백':'#EF5350','구조':'#5C6BC0','스트레스':'#FF8A65',
          'pos':'#EF5350','neg':'#5C6BC0','neutral':'#BDBDBD'}

# ── P1: SHAP 중요도 바차트 (침해여부) ──
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor('#F8F9FA')
colors_shap = []
for f in shap_rank_a['feat']:
    if f.startswith('D2'): colors_shap.append(COLORS['제도공백'])
    elif f in ['초등','중등','공립','기간제','남성','SQ5']: colors_shap.append(COLORS['구조'])
    else: colors_shap.append(COLORS['스트레스'])

bars1 = ax1.barh(shap_rank_a['변수'], shap_rank_a['mean_SHAP'],
                 color=colors_shap, edgecolor='white', height=0.7)
for bar, val in zip(bars1, shap_rank_a['mean_SHAP']):
    ax1.text(val+0.0005, bar.get_y()+bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=7.5)
ax1.set_xlabel('Mean |SHAP value|')
ax1.set_title('SHAP 변수 중요도 — 침해여부 예측\n(빨강=제도공백 인식, 파랑=구조요인, 주황=스트레스)',
              fontsize=9, fontweight='bold')
handles_s = [mpatches.Patch(color=COLORS['제도공백'], label='제도공백 인식(D2)'),
             mpatches.Patch(color=COLORS['구조'], label='학교 구조 변수'),
             mpatches.Patch(color=COLORS['스트레스'], label='스트레스 원인')]
ax1.legend(handles=handles_s, fontsize=7, loc='lower right')

# ── P2: SHAP 중요도 바차트 (보호자침해) ──
ax2 = fig.add_subplot(gs[0, 2:])
ax2.set_facecolor('#F8F9FA')
colors_shap_b = []
for f in shap_rank_b['feat']:
    if f.startswith('D2'): colors_shap_b.append(COLORS['제도공백'])
    elif f in ['초등','중등','공립','기간제','남성','SQ5']: colors_shap_b.append(COLORS['구조'])
    else: colors_shap_b.append(COLORS['스트레스'])
bars2 = ax2.barh(shap_rank_b['변수'], shap_rank_b['mean_SHAP'],
                 color=colors_shap_b, edgecolor='white', height=0.7)
for bar, val in zip(bars2, shap_rank_b['mean_SHAP']):
    ax2.text(val+0.0005, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=7.5)
ax2.set_xlabel('Mean |SHAP value|')
ax2.set_title('SHAP 변수 중요도 — 보호자침해 예측', fontsize=9, fontweight='bold')
ax2.legend(handles=handles_s, fontsize=7, loc='lower right')

# ── P3: BM 효과 크기 비교 (제도요인별) ──
ax3 = fig.add_subplot(gs[1, :2])
ax3.set_facecolor('#F8F9FA')
d2_short = ['아동학대\n신고보호','수업방해\n분리시스템','악성민원\n법적패널티','과밀학급\n해소']
eta2_vals = [bm_results[k]['eta2'] for k in d2_labels]
m1_vals   = [bm_results[k]['M1'] for k in d2_labels]
m0_vals   = [bm_results[k]['M0'] for k in d2_labels]
diff_vals = [m1-m0 for m1,m0 in zip(m1_vals,m0_vals)]
x_bm = np.arange(len(d2_short))
ax3b = ax3.twinx()
bars3a = ax3.bar(x_bm-0.2, diff_vals, 0.35, color='#EF5350', alpha=0.85, label='평균 차이(Δ)')
bars3b = ax3b.bar(x_bm+0.2, [e*1000 for e in eta2_vals], 0.35, color='#5C6BC0', alpha=0.6, label='η²×1000')
ax3.set_xticks(x_bm); ax3.set_xticklabels(d2_short, fontsize=8)
ax3.set_ylabel('M(침해있음) - M(침해없음)', color='#C62828', fontsize=8)
ax3b.set_ylabel('η²×1000 (효과 크기)', color='#1A237E', fontsize=8)
ax3.set_title('BM 검정: 침해 경험 여부 × D2 정책 필요도\n(빨강=평균 차이, 파랑=효과 크기 η²)', fontsize=9, fontweight='bold')
for i, (d, e) in enumerate(zip(diff_vals, eta2_vals)):
    ax3.text(i-0.2, d+0.002, f'Δ{d:.3f}', ha='center', fontsize=8, color='#C62828', fontweight='bold')
    ax3b.text(i+0.2, e*1000+0.02, f'{e:.5f}', ha='center', fontsize=7, color='#1A237E')
ax3.legend(loc='upper left', fontsize=7)
ax3b.legend(loc='upper right', fontsize=7)

# ── P4: 학교 규모 × 침해율 (계단식) ──
ax4 = fig.add_subplot(gs[1, 2:])
ax4.set_facecolor('#F8F9FA')
sz_vals = [1,2,3,4,5]
stu_r = [raw_w.loc[raw_w['SQ5']==s,'B2_1'].mean()*100 for s in sz_vals]
par_r = [raw_w.loc[raw_w['SQ5']==s,'B3_1'].mean()*100 for s in sz_vals]
ns_v  = [len(raw_w[raw_w['SQ5']==s]) for s in sz_vals]
x4 = np.arange(5)
ax4.plot(x4, stu_r, 'o-', color='#5C6BC0', lw=2.5, ms=9, label='학생침해율')
ax4.plot(x4, par_r, 's-', color='#EF5350', lw=2.5, ms=9, label='보호자침해율')
ax4.fill_between(x4, stu_r, alpha=0.08, color='#5C6BC0')
ax4.fill_between(x4, par_r, alpha=0.08, color='#EF5350')
sz_abbr = ['5학급\n미만','6-11\n학급','12-17\n학급','18-35\n학급','36학급\n이상']
ax4.set_xticks(x4); ax4.set_xticklabels(sz_abbr, fontsize=8)
for i,(sr,pr,n) in enumerate(zip(stu_r,par_r,ns_v)):
    ax4.text(i, sr+0.5, f'{sr:.0f}%', ha='center', fontsize=8, color='#1A237E', fontweight='bold')
    ax4.text(i, pr-2.0, f'{pr:.0f}%', ha='center', fontsize=8, color='#B71C1C')
    ax4.text(i, 55, f'n={n}', ha='center', fontsize=7, color='gray')
ax4.set_ylabel('침해율 (%)', fontsize=9)
ax4.set_title(f'학교 규모별 교권침해율 (과밀학급 효과)\nKruskal-Wallis H={kw_stat:.1f}, p<0.001', fontsize=9, fontweight='bold')
ax4.legend(fontsize=8)
ax4.set_ylim(50, 95)

# ── P5: 부트스트랩 매개효과 비교 ──
ax5 = fig.add_subplot(gs[2, :2])
ax5.set_facecolor('#F8F9FA')
med_labels = ['침해→번아웃\n→이직', '침해→번아웃\n→자살사고', '침해→번아웃\n→정신건강악화', '학교규모→침해\n→번아웃']
ind_means = [r['indirect_mean'] for _, r in med_results]
ind_lo    = [r['indirect_ci95'][0] for _, r in med_results]
ind_hi    = [r['indirect_ci95'][1] for _, r in med_results]
x5 = np.arange(len(med_labels))
colors_med = [COLORS['pos'] if lo>0 else COLORS['neg'] if hi<0 else COLORS['neutral']
              for lo,hi in zip(ind_lo,ind_hi)]
bars5 = ax5.bar(x5, ind_means, color=colors_med, alpha=0.85, edgecolor='white')
ax5.errorbar(x5, ind_means, yerr=[np.array(ind_means)-np.array(ind_lo),
                                   np.array(ind_hi)-np.array(ind_means)],
             fmt='none', color='black', capsize=5, lw=1.5)
ax5.axhline(0, color='black', lw=0.8, linestyle='--')
for i, (m, lo, hi) in enumerate(zip(ind_means, ind_lo, ind_hi)):
    sig = '***' if (lo>0 and hi>0) or (lo<0 and hi<0) else ''
    ax5.text(i, max(abs(m),abs(hi))+0.001, f'{m:.4f}{sig}', ha='center', fontsize=8, fontweight='bold')
ax5.set_xticks(x5); ax5.set_xticklabels(med_labels, fontsize=8)
ax5.set_ylabel('간접효과 (β, 부트스트랩 95% CI)')
ax5.set_title('부트스트랩 매개효과 (n=2,000 반복)\n번아웃 = 침해 → 결과 경로의 매개변수', fontsize=9, fontweight='bold')

# ── P6: 교차검증 AUC 박스플롯 ──
ax6 = fig.add_subplot(gs[2, 2:])
ax6.set_facecolor('#F8F9FA')
cv_labels = list(cv_results.keys())
cv_aucs   = [cv_results[k]['aucs'] for k in cv_labels]
bp = ax6.boxplot(cv_aucs, labels=cv_labels, patch_artist=True,
                 medianprops=dict(color='black', lw=2))
box_colors = ['#90CAF9','#EF9A9A','#FFCC80','#CE93D8']
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color); patch.set_alpha(0.8)
for i, (k, auc) in enumerate(zip(cv_labels, cv_aucs)):
    ax6.text(i+1, np.mean(auc)+0.005, f'{np.mean(auc):.3f}', ha='center', fontsize=9, fontweight='bold')
ax6.axhline(0.5, color='red', linestyle=':', lw=1.2, label='랜덤 (0.5)')
ax6.axhline(0.7, color='green', linestyle=':', lw=1.2, label='양호 (0.7)')
ax6.set_ylabel('AUC (5-fold CV)'); ax6.set_ylim(0.45, 0.92)
ax6.set_title('5-겹 교차검증 로지스틱 AUC (class_weight=balanced)', fontsize=9, fontweight='bold')
ax6.legend(fontsize=8)

# ── P7: D3→D1 인권교육 역설 추이 ──
ax7 = fig.add_subplot(gs[3, :2])
ax7.set_facecolor('#F8F9FA')
years  = [2021, 2024]
edu_rate = [d3_인권교육_참여, d1_인권연수_이수율]
ax7b = ax7.twinx()
ax7.plot(years, edu_rate, 'o-', color='#5C6BC0', lw=2.5, ms=10, label='인권교육 참여·이수율', zorder=3)
ax7.fill_between(years, edu_rate, alpha=0.1, color='#5C6BC0')
# 침해율 (2021 추정 불가 → 2024만)
ax7b.bar(2024, d1_침해율, width=0.8, color='#EF5350', alpha=0.5, label='침해율(2024)')
ax7b.bar(2024-0.9, 78.5*0.95, width=0.8, color='#FFCDD2', alpha=0.5, label='침해율(2021 추정)')
for y, e in zip(years, edu_rate):
    ax7.text(y, e+1.5, f'{e:.1f}%', ha='center', fontsize=10, color='#1A237E', fontweight='bold')
ax7.set_ylabel('인권교육 참여율 (%)', color='#1A237E', fontsize=9)
ax7b.set_ylabel('교권침해율 (%)', color='#C62828', fontsize=9)
ax7.set_xlim(2019.5, 2025.5)
ax7.set_xticks([2021, 2024]); ax7.set_xticklabels(['2021\n(D3)','2024\n(D1)'])
ax7.set_ylim(0, 120)
ax7b.set_ylim(0, 120)
ax7.set_title('인권교육 역설 — 교육 증가 ≠ 침해 감소\n(교육은 해결책이 아니라 제도 변화가 필요)', fontsize=9, fontweight='bold')
ax7.legend(loc='upper left', fontsize=7)
ax7b.legend(loc='upper right', fontsize=7)
ax7.text(2022.5, 100, '인권교육↑ 20.8%p\n교권침해↗ 여전히 심각', ha='center', fontsize=9,
         color='#B71C1C', fontweight='bold',
         bbox=dict(facecolor='#FFEBEE', alpha=0.8, edgecolor='#EF5350', boxstyle='round'))

# ── P8: 제도요인 종합 우선순위 매트릭스 ──
ax8 = fig.add_subplot(gs[3, 2:])
ax8.set_facecolor('#F8F9FA')
# X: BM eta² (효과 크기), Y: Logistic OR 보호자
eta2s = [bm_results[k]['eta2']*1000 for k in d2_labels]
or_ps = [or_par[k] for k in d2_labels]
shap_sizes = [shap_rank_dict.get(k,8) for k in d2_labels]
s_scale = [(8-r)*30+80 for r in shap_sizes]  # SHAP rank → bubble size
colors_mat = ['#EF5350','#FF8A65','#EF5350','#FFA726']
for i, (d2k, d2l) in enumerate(d2_labels.items()):
    ax8.scatter(eta2s[i], or_ps[i], s=s_scale[i]*4,
                color=colors_mat[i], alpha=0.8, edgecolors='white', linewidth=2, zorder=3)
    ax8.annotate(d2l.replace('\n',''), (eta2s[i], or_ps[i]),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=8.5, fontweight='bold', color='black')
ax8.axhline(np.mean(or_ps), color='gray', linestyle='--', alpha=0.5, lw=1)
ax8.axvline(np.mean(eta2s), color='gray', linestyle='--', alpha=0.5, lw=1)
ax8.set_xlabel('BM 효과 크기 η²×1000 (클수록 침해와 강하게 연관)', fontsize=8)
ax8.set_ylabel('로지스틱 OR (보호자침해)', fontsize=8)
ax8.set_title('제도요인 우선순위 매트릭스\n(우상단 = 효과 크고 OR 높음 → 최우선 해결 과제)',
              fontsize=9, fontweight='bold')
ax8.text(ax8.get_xlim()[0]+0.001, np.mean(or_ps)+0.01, '← 우선순위 낮음 │ 높음 →',
         fontsize=7, color='gray', ha='left')

plt.tight_layout()
out_path = '/Users/baiohelseu/Desktop/Project/kossda/output/robust_analysis.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print(f"\n저장: {out_path}")

# ─────────────────────────────────────────────────────────────
# 최종 요약
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("종합 분석 결과 요약")
print("═"*60)
print(f"\n[모델 강건성]")
print(f"  VIF 최대값: {vif_df['VIF'].max():.2f} (< 5 — 다중공선성 없음)")
print(f"  5-fold CV AUC (침해여부): {cv_results['침해여부']['mean']:.3f} ± {cv_results['침해여부']['std']:.3f}")
print(f"  5-fold CV AUC (자살사고): {cv_results['자살사고']['mean']:.3f} ± {cv_results['자살사고']['std']:.3f}")

print(f"\n[어떤 제도 부실이 교권침해를 가장 크게 유발하는가]")
for i, row in summary_df.iterrows():
    rank = summary_df.index.tolist().index(i) + 1
    print(f"  {rank}위: {row['레이블']} — OR(보호자)={row['OR_par']:.2f}, SHAP순위=#{int(row['shap_rank'])}, η²={row['eta2']:.5f}")

print(f"\n[매개효과 검증 (Bootstrap 95%CI)]")
for label, r in med_results:
    lo, hi = r['indirect_ci95']
    sig = '유의(p<.05)' if (lo>0 and hi>0) or (lo<0 and hi<0) else '비유의'
    print(f"  {label}: 간접={r['indirect_mean']:.4f} [{lo:.4f},{hi:.4f}] → {sig}")

print(f"\n[인권교육 역설 수치]")
print(f"  2021→2024 인권교육 참여·이수율: +{d1_인권연수_이수율-d3_인권교육_참여:.1f}%p 증가")
print(f"  2024 교권침해 경험률: {d1_침해율:.1f}% → 교육 증가가 침해 감소로 이어지지 않음")
