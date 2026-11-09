"""
교권침해 구조방정식 모델 (SEM)
─ 잠재요인(Latent Factors) + 인과 경로 추론
─ 이론 기반 4개 잠재변수: 번아웃, 제도공백, 침해압력(복합), 직업탈진
"""
import pyreadstat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─── 한글 폰트 ───
for fp in ['/System/Library/Fonts/Supplemental/AppleGothic.ttf',
           '/Library/Fonts/NanumGothic.ttf']:
    try:
        fm.fontManager.addfont(fp)
        plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────────────────
# 1. 데이터 로드 & 변수 준비
# ─────────────────────────────────────────────────────────
df_raw, meta = pyreadstat.read_sav(
    '/Users/baiohelseu/Desktop/Project/kossda/data/교원 인권상황 실태조사,2024/kor_data_20240073.sav'
)
N = len(df_raw)
print(f"D1 로드 완료: n={N:,}")

df = pd.DataFrame()

# 번아웃 indicators (0-5, 연속)
for i in range(1, 6):
    df[f'C3_{i}'] = df_raw[f'C3_{i}'].fillna(df_raw[f'C3_{i}'].median())

# 제도공백 indicators (1-5, 높을수록 필요도 큼 = 현재 공백 큼)
for col in ['D2_2', 'D2_5', 'D2_10', 'D2_16']:
    df[col] = df_raw[col].fillna(df_raw[col].median())

# 침해 경험 지수 (0-3 합산, 복합 관찰변수)
df['침해지수'] = (
    (df_raw['B2_1'] == 1).astype(int) +
    (df_raw['B3_1'] == 1).astype(int) +
    (df_raw['B5_1'] == 1).astype(int)
).fillna(0)

# 스트레스 원인 (0/1 binary, 주요 3개)
df['스트레스_보호자'] = (df_raw['C5_11'] == 1).astype(int)
df['스트레스_교실']   = (df_raw['C5_2']  == 1).astype(int)
df['스트레스_행정']   = (df_raw['C5_3']  == 1).astype(int)
df['스트레스지수'] = df[['스트레스_보호자','스트레스_교실','스트레스_행정']].sum(axis=1)

# 정신건강 (1=매우나쁨, 5=매우좋음 → 역코딩 → 높을수록 나쁨)
df['정신건강역'] = 6 - df_raw['C1'].fillna(df_raw['C1'].median())
# 직업만족도 (1-5 → 역코딩)
df['직업만족역'] = 6 - df_raw['A3_6'].fillna(df_raw['A3_6'].median())
# 이진 결과
df['이직고려'] = (df_raw['A4'] == 1).astype(int)
df['자살사고'] = (df_raw['C4'] == 1).astype(int)

# 학교 특성
df['초등더미'] = (df_raw['SQ2'] == 2).astype(int)  # 초등=2
df['학교규모'] = df_raw['SQ5'].fillna(df_raw['SQ5'].median())

# 번아웃 총점 (SEM 외 비교용)
df['번아웃_총점'] = df[[f'C3_{i}' for i in range(1,6)]].mean(axis=1)
# 제도공백 총점
df['제도공백_총점'] = df[['D2_2','D2_5','D2_10','D2_16']].mean(axis=1)

print("\n변수 기술통계:")
desc_vars = ['C3_1','C3_2','C3_3','C3_4','C3_5',
             'D2_2','D2_5','D2_10','D2_16',
             '침해지수','스트레스지수','정신건강역','직업만족역',
             '이직고려','자살사고','초등더미','학교규모']
print(df[desc_vars].describe().round(2).to_string())

# ─────────────────────────────────────────────────────────
# 2. EFA: 번아웃 5항목 요인 구조 확인
# ─────────────────────────────────────────────────────────
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity

burnout_items = [f'C3_{i}' for i in range(1,6)]
fa_data = df[burnout_items].copy()

# KMO & Bartlett
try:
    kmo_all, kmo_model = calculate_kmo(fa_data)
    chi2_b, p_b = calculate_bartlett_sphericity(fa_data)
    print(f"\n=== EFA: 번아웃 항목 ===")
    print(f"KMO = {kmo_model:.3f} (>0.6 = 적합)")
    print(f"Bartlett χ² = {chi2_b:.1f}, p = {p_b:.4f}")
except Exception as e:
    print(f"EFA 검정 오류: {e}")

# 1요인 EFA
try:
    fa = FactorAnalyzer(n_factors=1, rotation=None)
    fa.fit(fa_data)
    loadings = fa.loadings_
    ev, v = fa.get_eigenvalues()
    print(f"1요인 분산 설명률: {v[0]*100:.1f}%")
    print("요인 부하량:")
    for item, load in zip(burnout_items, loadings[:,0]):
        lbl = meta.column_labels[meta.column_names.index(item)].split(':')[-1].strip()[:30]
        print(f"  {item} [{lbl}]: {load:.3f}")
    burnout_efa_ok = True
except Exception as e:
    print(f"EFA 오류: {e}")
    burnout_efa_ok = False

# ─────────────────────────────────────────────────────────
# 3. Cronbach's Alpha for Burnout Scale
# ─────────────────────────────────────────────────────────
def cronbach_alpha(df_items):
    n_items = df_items.shape[1]
    item_vars = df_items.var(ddof=1)
    total_var = df_items.sum(axis=1).var(ddof=1)
    return (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)

alpha_burnout = cronbach_alpha(df[burnout_items])
alpha_inst    = cronbach_alpha(df[['D2_2','D2_5','D2_10','D2_16']])
print(f"\n신뢰도 (Cronbach α):")
print(f"  번아웃 5항목: α = {alpha_burnout:.3f}")
print(f"  제도공백 4항목: α = {alpha_inst:.3f}")

# ─────────────────────────────────────────────────────────
# 4. SEM (semopy)
# ─────────────────────────────────────────────────────────
import semopy

# 모델 사양
model_desc = """
# ─ 잠재변수 측정모델 (CFA) ─
번아웃 =~ C3_1 + C3_2 + C3_3 + C3_4 + C3_5
제도공백 =~ D2_2 + D2_5 + D2_10 + D2_16

# ─ 구조모델 (경로) ─
침해지수 ~ 초등더미 + 학교규모 + 제도공백
스트레스지수 ~ 침해지수 + 제도공백
번아웃 ~ 침해지수 + 스트레스지수 + 제도공백
정신건강역 ~ 번아웃 + 침해지수
직업만족역 ~ 번아웃 + 침해지수
"""

print("\n=== SEM 실행 ===")
sem_model = semopy.Model(model_desc)
sem_model.fit(df)
results = sem_model.inspect()
print(results.to_string())

# 모델 적합도
try:
    stats_fit = semopy.calc_stats(sem_model)
    print("\n=== 모델 적합도 ===")
    print(stats_fit.T)
except Exception as e:
    print(f"적합도 오류: {e}")

# ─────────────────────────────────────────────────────────
# 5. 이진 결과변수 로지스틱 회귀 (보완)
# ─────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

print("\n=== 로지스틱 회귀: 이직고려 ===")
X_logit = df[['번아웃_총점','침해지수','스트레스지수','제도공백_총점','초등더미','학교규모']].copy()
y_이직 = df['이직고려']
y_자살 = df['자살사고']

sc = StandardScaler()
X_s = sc.fit_transform(X_logit)

from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

X_sm = sm.add_constant(X_logit)
for label, y in [('이직고려', y_이직), ('자살사고', y_자살)]:
    try:
        lm = sm.Logit(y, X_sm).fit(disp=0)
        print(f"\n  [{label}] (표준화 아닌 원점수)")
        print(f"  {'변수':<16} {'OR':>6} {'95%CI_L':>8} {'95%CI_U':>8} {'p':>8}")
        ci = np.exp(lm.conf_int())
        for var, coef, p, lo, hi in zip(
            X_sm.columns[1:], lm.params[1:], lm.pvalues[1:], ci.values[1:,0], ci.values[1:,1]
        ):
            sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
            print(f"  {var:<16} {np.exp(coef):>6.3f} {lo:>8.3f} {hi:>8.3f} {p:>8.4f} {sig}")
    except Exception as e:
        print(f"  오류: {e}")

# ─────────────────────────────────────────────────────────
# 6. 간접효과(Indirect Effect) 계산 - Bootstrap
# ─────────────────────────────────────────────────────────
print("\n=== 간접효과 분석 (경로 곱) ===")

# 주요 간접 경로: 침해지수 → 번아웃 → 정신건강역
# OLS 기반으로 계산 (semopy 결과에서 경로 추출)
try:
    res_df = sem_model.inspect()

    def get_coef(lhs, op, rhs):
        mask = (res_df['lval']==lhs) & (res_df['op']==op) & (res_df['rval']==rhs)
        if mask.any():
            return float(res_df.loc[mask,'Estimate'].values[0])
        return np.nan

    # 경로 계수 추출
    # 번아웃 → 정신건강역
    a_burn_mental = get_coef('정신건강역','~','번아웃')
    # 침해지수 → 번아웃
    b_inj_burn = get_coef('번아웃','~','침해지수')
    # 간접효과 = a * b
    indirect_inj_mental = b_inj_burn * a_burn_mental

    # 제도공백 → 침해지수 → 번아웃
    c_inst_inj = get_coef('침해지수','~','제도공백')
    indirect_inst_burn = c_inst_inj * b_inj_burn
    indirect_inst_mental = c_inst_inj * b_inj_burn * a_burn_mental

    print(f"침해지수 → 번아웃: β={b_inj_burn:.3f}")
    print(f"번아웃 → 정신건강역: β={a_burn_mental:.3f}")
    print(f"침해지수 → 번아웃 → 정신건강역 (간접): {indirect_inj_mental:.3f}")
    print()
    print(f"제도공백 → 침해지수: β={c_inst_inj:.3f}")
    print(f"제도공백 → 침해지수 → 번아웃 (간접): {indirect_inst_burn:.3f}")
    print(f"제도공백 → 침해지수 → 번아웃 → 정신건강역 (총간접): {indirect_inst_mental:.3f}")

    # 직업만족역 경로도
    d_burn_job = get_coef('직업만족역','~','번아웃')
    e_inj_job  = get_coef('직업만족역','~','침해지수')
    indirect_inj_job = b_inj_burn * d_burn_job
    print()
    print(f"번아웃 → 직업만족역: β={d_burn_job:.3f}")
    print(f"침해지수 → 직업만족역 직접: β={e_inj_job:.3f}")
    print(f"침해지수 → 번아웃 → 직업만족역 (간접): {indirect_inj_job:.3f}")

except Exception as e:
    print(f"간접효과 계산 오류: {e}")

# ─────────────────────────────────────────────────────────
# 7. 시각화: SEM 경로도 + 지원 패널
# ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor('#FAFAFA')
fig.suptitle('교권침해 구조방정식 모델 (SEM)\n잠재요인 × 인과경로 분석',
             fontsize=15, fontweight='bold', y=0.99)

# ── 서브플롯 레이아웃 ──
gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.4)
ax_path  = fig.add_subplot(gs[0:2, 0:3])  # 경로도 (좌측 크게)
ax_cfa1  = fig.add_subplot(gs[0, 3])       # 번아웃 CFA
ax_cfa2  = fig.add_subplot(gs[1, 3])       # 제도공백 CFA
ax_med1  = fig.add_subplot(gs[2, 0])       # 번아웃 분포
ax_med2  = fig.add_subplot(gs[2, 1])       # 매개효과 비교
ax_log1  = fig.add_subplot(gs[2, 2])       # 이직 OR
ax_log2  = fig.add_subplot(gs[2, 3])       # 자살사고 OR

# ═══════════════════════════════════════
# Panel 1: SEM 경로도 (수동 그리기)
# ═══════════════════════════════════════
ax = ax_path
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_facecolor('#F8F9FA')

# SEM 경로 계수 추출 helper
def get_est(lhs, rhs):
    try:
        mask = (results['lval']==lhs) & (results['rval']==rhs)
        if mask.any():
            est = results.loc[mask,'Estimate'].values[0]
            pval = results.loc[mask,'p-value'].values[0]
            sig = '***' if pval<0.001 else '**' if pval<0.01 else '*' if pval<0.05 else ''
            return f"{est:.3f}{sig}"
        return "?"
    except Exception:
        return "?"

# ─ 노드 좌표 ─
nodes = {
    # 잠재변수 (타원)
    '번아웃\n(latent)':      (5.0, 6.0),
    '제도공백\n(latent)':    (1.5, 6.0),
    # 복합 관찰변수 (사각형)
    '침해지수':              (3.2, 4.0),
    '스트레스지수':          (5.5, 4.0),
    # 외생변수
    '초등더미':              (1.0, 4.0),
    '학교규모':              (2.0, 2.8),
    # 결과변수
    '정신건강역':            (7.8, 6.0),
    '직업만족역':            (7.8, 4.2),
}

# 번아웃 측정변수 (CFA)
burnout_meas = ['C3_1', 'C3_2', 'C3_3', 'C3_4', 'C3_5']
for i, var in enumerate(burnout_meas):
    nodes[var] = (3.0 + i*0.55, 7.6)

# 제도공백 측정변수 (CFA)
inst_meas = ['D2_2', 'D2_5', 'D2_10', 'D2_16']
for i, var in enumerate(inst_meas):
    nodes[var] = (0.2 + i*0.65, 7.6)

# ─ 노드 그리기 ─
LATENT_COLOR  = '#E3F2FD'  # 연파랑: 잠재
OBSERVED_COLOR= '#E8F5E9'  # 연녹: 관찰(복합)
EXOG_COLOR    = '#FFF3E0'  # 연오렌지: 외생
OUTCOME_COLOR = '#FCE4EC'  # 연분홍: 결과
MEAS_COLOR    = '#FFFDE7'  # 연노랑: 측정

def draw_node(ax, label, xy, shape='rect', color='white', fontsize=8, width=1.2, height=0.55):
    x, y = xy
    if shape == 'ellipse':
        ell = mpatches.Ellipse(xy, width=width, height=height+0.1,
                               facecolor=color, edgecolor='#555', linewidth=1.5, zorder=3)
        ax.add_patch(ell)
    else:
        rect = mpatches.FancyBboxPatch((x-width/2, y-height/2), width, height,
                                       boxstyle="round,pad=0.05",
                                       facecolor=color, edgecolor='#555', linewidth=1.2, zorder=3)
        ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', zorder=4, wrap=True)

# 잠재변수
draw_node(ax, '번아웃\n(Burnout)', nodes['번아웃\n(latent)'], 'ellipse', LATENT_COLOR, 9, 1.5, 0.7)
draw_node(ax, '제도공백\n(Inst. Gap)', nodes['제도공백\n(latent)'], 'ellipse', LATENT_COLOR, 9, 1.5, 0.7)

# 관찰 복합변수
draw_node(ax, '침해지수\n(複합)', nodes['침해지수'], 'rect', OBSERVED_COLOR, 8.5, 1.2, 0.6)
draw_node(ax, '스트레스\n지수', nodes['스트레스지수'], 'rect', OBSERVED_COLOR, 8.5, 1.2, 0.6)

# 외생변수
draw_node(ax, '초등\n더미', nodes['초등더미'], 'rect', EXOG_COLOR, 7.5, 0.85, 0.5)
draw_node(ax, '학교\n규모', nodes['학교규모'], 'rect', EXOG_COLOR, 7.5, 0.85, 0.5)

# 결과변수
draw_node(ax, '정신건강\n(역)', nodes['정신건강역'], 'rect', OUTCOME_COLOR, 8, 1.2, 0.6)
draw_node(ax, '직업만족\n(역)', nodes['직업만족역'], 'rect', OUTCOME_COLOR, 8, 1.2, 0.6)

# CFA 측정변수 (번아웃)
for var in burnout_meas:
    draw_node(ax, var, nodes[var], 'rect', MEAS_COLOR, 6.5, 0.48, 0.35)
# CFA 측정변수 (제도공백)
for var in inst_meas:
    draw_node(ax, var, nodes[var], 'rect', MEAS_COLOR, 6.5, 0.55, 0.35)

# ─ 화살표 그리기 helper ─
def draw_arrow(ax, from_node, to_node, label='', color='#1565C0', lw=1.5,
               label_offset=(0,0), fontsize=7.5, rad=0.0):
    x1, y1 = from_node
    x2, y2 = to_node
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'))
    if label:
        mx, my = (x1+x2)/2 + label_offset[0], (y1+y2)/2 + label_offset[1]
        ax.text(mx, my, label, fontsize=fontsize, ha='center', va='center',
                color=color, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))

# CFA 화살표 (잠재 → 측정)
bx, by = nodes['번아웃\n(latent)']
for var in burnout_meas:
    vx, vy = nodes[var]
    ax.annotate('', xy=(vx, vy-0.18), xytext=(bx, by+0.38),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.0, alpha=0.6))

ix, iy = nodes['제도공백\n(latent)']
for var in inst_meas:
    vx, vy = nodes[var]
    ax.annotate('', xy=(vx, vy-0.18), xytext=(ix, iy+0.38),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.0, alpha=0.6))

# 구조 경로 화살표
c_inst_chm = get_est('침해지수','제도공백')
c_elem_chm = get_est('침해지수','초등더미')
c_size_chm = get_est('침해지수','학교규모')
c_chm_str  = get_est('스트레스지수','침해지수')
c_inst_str = get_est('스트레스지수','제도공백')
c_chm_brn  = get_est('번아웃','침해지수')
c_str_brn  = get_est('번아웃','스트레스지수')
c_inst_brn = get_est('번아웃','제도공백')
c_brn_men  = get_est('정신건강역','번아웃')
c_chm_men  = get_est('정신건강역','침해지수')
c_brn_job  = get_est('직업만족역','번아웃')
c_chm_job  = get_est('직업만족역','침해지수')

# 제도공백 → 침해지수
draw_arrow(ax, (ix+0.4, iy-0.35), (nodes['침해지수'][0]-0.1, nodes['침해지수'][1]+0.31),
           c_inst_chm, '#C62828', 2.5, (0.3, 0.15))

# 제도공백 → 스트레스
draw_arrow(ax, (ix+0.6, iy-0.35), (nodes['스트레스지수'][0]-0.3, nodes['스트레스지수'][1]+0.31),
           c_inst_str, '#C62828', 1.8, (0.5, 0.2))

# 제도공백 → 번아웃 (직접)
draw_arrow(ax, (ix+0.75, iy), (nodes['번아웃\n(latent)'][0]-0.75, nodes['번아웃\n(latent)'][1]),
           c_inst_brn, '#E65100', 2.0, (0, 0.25), rad=0.15, fontsize=7.5)

# 초등 → 침해
draw_arrow(ax, nodes['초등더미'], (nodes['침해지수'][0]-0.3, nodes['침해지수'][1]-0.1),
           c_elem_chm, '#388E3C', 1.5, (-0.15, 0.15))

# 학교규모 → 침해
draw_arrow(ax, (nodes['학교규모'][0]+0.2, nodes['학교규모'][1]+0.15),
           (nodes['침해지수'][0]-0.4, nodes['침해지수'][1]-0.25),
           c_size_chm, '#388E3C', 1.5, (0, 0))

# 침해 → 스트레스
draw_arrow(ax, (nodes['침해지수'][0]+0.6, nodes['침해지수'][1]),
           (nodes['스트레스지수'][0]-0.6, nodes['스트레스지수'][1]),
           c_chm_str, '#5C6BC0', 2.0, (0, 0.2))

# 침해 → 번아웃
draw_arrow(ax, (nodes['침해지수'][0]+0.1, nodes['침해지수'][1]+0.31),
           (nodes['번아웃\n(latent)'][0]-0.5, nodes['번아웃\n(latent)'][1]-0.38),
           c_chm_brn, '#1565C0', 2.5, (-0.1, 0.1))

# 스트레스 → 번아웃
draw_arrow(ax, (nodes['스트레스지수'][0]+0.1, nodes['스트레스지수'][1]+0.31),
           (nodes['번아웃\n(latent)'][0]+0.2, nodes['번아웃\n(latent)'][1]-0.38),
           c_str_brn, '#1565C0', 2.5, (0.2, 0.1))

# 번아웃 → 정신건강
draw_arrow(ax, (nodes['번아웃\n(latent)'][0]+0.75, nodes['번아웃\n(latent)'][1]),
           (nodes['정신건강역'][0]-0.6, nodes['정신건강역'][1]),
           c_brn_men, '#AD1457', 2.5, (0, 0.2))

# 번아웃 → 직업만족
draw_arrow(ax, (nodes['번아웃\n(latent)'][0]+0.75, nodes['번아웃\n(latent)'][1]-0.2),
           (nodes['직업만족역'][0]-0.6, nodes['직업만족역'][1]+0.1),
           c_brn_job, '#AD1457', 2.5, (0, 0.1), rad=0.2)

# 침해 → 정신건강 (직접)
draw_arrow(ax, (nodes['침해지수'][0]+0.6, nodes['침해지수'][1]+0.1),
           (nodes['정신건강역'][0]-0.6, nodes['정신건강역'][1]-0.1),
           c_chm_men, '#6A1B9A', 1.5, (0.1, -0.2), rad=-0.15)

ax.set_title('SEM 인과 경로도  (*** p<.001  ** p<.01  * p<.05)\n파란/빨간 선=구조경로  회색 선=CFA 측정경로',
             fontsize=9, pad=4)

# 범례
legend_items = [
    mpatches.Patch(facecolor=LATENT_COLOR,   edgecolor='#555', label='잠재변수 (Latent)'),
    mpatches.Patch(facecolor=OBSERVED_COLOR,  edgecolor='#555', label='관찰 복합변수'),
    mpatches.Patch(facecolor=EXOG_COLOR,      edgecolor='#555', label='외생변수'),
    mpatches.Patch(facecolor=OUTCOME_COLOR,   edgecolor='#555', label='결과변수'),
    mpatches.Patch(facecolor=MEAS_COLOR,      edgecolor='#555', label='CFA 측정지표'),
]
ax.legend(handles=legend_items, loc='lower left', fontsize=7, framealpha=0.9)

# ═══════════════════════════════════════
# Panel 2: 번아웃 CFA 요인 부하량
# ═══════════════════════════════════════
ax = ax_cfa1
ax.set_facecolor('#F8F9FA')
labels_short = ['고갈', '탈진', '피곤함', '스트레스', '소진']
try:
    load_vals = loadings[:,0]
    colors_bar = ['#FF8A65' if v>0.7 else '#FFA726' for v in load_vals]
    bars = ax.barh(labels_short, load_vals, color=colors_bar, edgecolor='white')
    ax.set_xlim(0, 1.0)
    ax.axvline(0.7, color='red', linestyle='--', alpha=0.5, label='0.7 기준선')
    for bar, val in zip(bars, load_vals):
        ax.text(val+0.01, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', fontsize=8)
    ax.set_title(f'번아웃 CFA 부하량\nα={alpha_burnout:.3f}', fontsize=9, fontweight='bold')
    ax.set_xlabel('요인 부하량')
    ax.legend(fontsize=7)
except Exception:
    ax.text(0.5, 0.5, 'EFA 오류', transform=ax.transAxes, ha='center')

# ═══════════════════════════════════════
# Panel 3: 제도공백 항목 신뢰도
# ═══════════════════════════════════════
ax = ax_cfa2
ax.set_facecolor('#F8F9FA')
inst_labels = ['아동학대\n보호', '분리\n시스템', '악성민원\n패널티', '과밀학급\n해소']
inst_means  = [df_raw[c].mean() for c in ['D2_2','D2_5','D2_10','D2_16']]
inst_colors = ['#EF5350','#FF7043','#FF8A65','#FFA726']
bars = ax.bar(inst_labels, inst_means, color=inst_colors, edgecolor='white')
ax.set_ylim(4.5, 5.05)
ax.axhline(5.0, color='gray', linestyle='--', alpha=0.5)
for bar, val in zip(bars, inst_means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.3f}',
            ha='center', fontsize=8)
ax.set_title(f'제도공백 측정항목 (D2)\n평균 정책 필요도(1-5), α={alpha_inst:.3f}', fontsize=9, fontweight='bold')
ax.set_ylabel('평균 (1-5점)')

# ═══════════════════════════════════════
# Panel 4: 번아웃 × 침해여부 분포
# ═══════════════════════════════════════
ax = ax_med1
ax.set_facecolor('#F8F9FA')
mask_no = df_raw['B2_1'].isin([2, 2.0])
mask_ye = df_raw['B2_1'].isin([1, 1.0])
no_viol = df.loc[mask_no, 'C3_1'].dropna()
ye_viol = df.loc[mask_ye, 'C3_1'].dropna()
if len(no_viol) == 0: no_viol = pd.Series([0, 0])
if len(ye_viol) == 0: ye_viol = pd.Series([0, 0])
parts = ax.violinplot([no_viol.values, ye_viol.values], positions=[1,2], showmedians=True)
parts['bodies'][0].set_facecolor('#90CAF9')
parts['bodies'][1].set_facecolor('#FF8A65')
for pc in parts['bodies']:
    pc.set_alpha(0.7)
ax.set_xticks([1,2])
ax.set_xticklabels(['침해\n없음', '침해\n있음'])
t_stat, t_pval = stats.mannwhitneyu(ye_viol, no_viol)
ax.set_title(f'학생침해 × 번아웃\nMW p={t_pval:.4f}', fontsize=9, fontweight='bold')
ax.set_ylabel('번아웃 C3_1 (0-5)')

# ═══════════════════════════════════════
# Panel 5: 직접/간접/총 효과 비교
# ═══════════════════════════════════════
ax = ax_med2
ax.set_facecolor('#F8F9FA')
try:
    # 정신건강역 경로
    dir_inj_men = float(results.loc[(results['lval']=='정신건강역')&(results['rval']=='침해지수'),'Estimate'])
    indir_inj_men = indirect_inj_mental
    total_inj_men = dir_inj_men + indir_inj_men

    effect_labels = ['침해→정신건강\n직접', '침해→번아웃\n→정신건강\n(간접)', '총효과']
    effect_vals   = [dir_inj_men, indir_inj_men, total_inj_men]
    eff_colors    = ['#EF5350','#FF8A65','#B71C1C']
    bars = ax.bar(effect_labels, effect_vals, color=eff_colors, edgecolor='white')
    for bar, val in zip(bars, effect_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001, f'{val:.3f}',
                ha='center', fontsize=8, fontweight='bold')
    ax.set_title('침해지수 → 정신건강역\n직접/간접/총효과', fontsize=9, fontweight='bold')
    ax.set_ylabel('경로 계수 (β)')
    ax.axhline(0, color='gray', linewidth=0.8)
except Exception:
    ax.text(0.5, 0.5, '계산 오류', transform=ax.transAxes, ha='center')

# ═══════════════════════════════════════
# Panel 6 & 7: 이직고려 / 자살사고 OR 플롯
# ═══════════════════════════════════════
var_labels = ['번아웃\n총점', '침해\n지수', '스트레스\n지수', '제도공백\n총점', '초등\n더미', '학교\n규모']
for ax_i, (ax_out, y_bin, title) in enumerate([(ax_log1, y_이직, '이직고려'), (ax_log2, y_자살, '자살사고')]):
    ax_out.set_facecolor('#F8F9FA')
    try:
        X_sm2 = sm.add_constant(X_logit)
        lm2 = sm.Logit(y_bin, X_sm2).fit(disp=0)
        ors = np.exp(lm2.params[1:])
        ci2 = np.exp(lm2.conf_int())
        lows = ci2.values[1:,0]
        highs = ci2.values[1:,1]
        pvals = lm2.pvalues[1:]

        colors_or = ['#EF5350' if p<0.05 else '#BDBDBD' for p in pvals]
        ax_out.barh(var_labels, ors, xerr=[ors-lows, highs-ors],
                    color=colors_or, edgecolor='white', height=0.6,
                    error_kw=dict(ecolor='#555', capsize=3))
        ax_out.axvline(1.0, color='black', linestyle='--', linewidth=1.0)
        ax_out.set_title(f'[{title}] 로지스틱\nOR (95% CI)\n빨강=유의(p<.05)', fontsize=8.5, fontweight='bold')
        ax_out.set_xlabel('Odds Ratio')
    except Exception as e:
        ax_out.text(0.5, 0.5, f'오류:{e}', transform=ax_out.transAxes, ha='center', fontsize=7)

plt.tight_layout()
out_path = '/Users/baiohelseu/Desktop/Project/kossda/output/sem_analysis.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print(f"\n시각화 저장 완료: {out_path}")
