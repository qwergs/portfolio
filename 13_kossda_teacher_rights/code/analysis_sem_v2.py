"""
SEM v2 — 제도공백(잠재변수) × 인과경로
잠재변수: A1_12/A1_13/C6_8/B8_1 역코딩 → 제도공백
경로: 제도공백 + 학교구조 → 침해지수 → 정신건강역 → 이직/자살
"""
import pyreadstat
import pandas as pd
import numpy as np
import semopy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from scipy import stats
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# ── 폰트 ──
for fp in ['/Library/Fonts/NanumGothicBold.ttf',
           '/Library/Fonts/NanumGothic.ttf',
           '/System/Library/Fonts/Supplemental/AppleGothic.ttf']:
    try:
        fm.fontManager.addfont(fp)
        plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False

BASE = '/Users/baiohelseu/Desktop/Project/kossda/'

# ─────────────────────────────────────────
# 1. 데이터 로드 & 변수 구성
# ─────────────────────────────────────────
df_raw, meta = pyreadstat.read_sav(BASE + 'data/교원 인권상황 실태조사,2024/kor_data_20240073.sav')
print(f"D1 로드: n={len(df_raw):,}")

df = pd.DataFrame()

# 제도공백 지표 — 역코딩 (6-x: 낮은 원점수=제도 미작동 → 높은 역코딩값=제도공백 큼)
df['제도1_분리'] = 6 - df_raw['A1_12']   # 분리제도 미작동
df['제도2_민원'] = 6 - df_raw['A1_13']   # 민원응대 미작동
df['제도3_절차'] = 6 - df_raw['C6_8']    # 공식 절차 부재
df['제도4_신뢰'] = 6 - df_raw['B8_1']    # 교권보호위 불신

# 학교 구조 변수 (역인과 없음)
df['초등더미'] = (df_raw['SQ2'] == 2).astype(float)
df['학교규모'] = df_raw['SQ5'].fillna(df_raw['SQ5'].median())

# 침해지수 (0-3 복합)
df['침해지수'] = (
    (df_raw['B2_1'] == 1).astype(float) +
    (df_raw['B3_1'] == 1).astype(float) +
    (df_raw['B5_1'] == 1).astype(float)
)

# 정신건강역 (역코딩: 높을수록 나쁨)
df['정신건강역'] = 6 - df_raw['C1']

# 결과변수
df['이직고려'] = (df_raw['A4'] == 1).astype(float)
df['자살사고'] = (df_raw['C4'] == 1).astype(float)

# 결측 제거
inst_cols = ['제도1_분리','제도2_민원','제도3_절차','제도4_신뢰']
all_cols = inst_cols + ['초등더미','학교규모','침해지수','정신건강역','이직고려','자살사고']
df = df[all_cols].dropna()
print(f"분석 표본: n={len(df):,}")

# ─────────────────────────────────────────
# 2. 제도공백 잠재변수 신뢰도 확인
# ─────────────────────────────────────────
def cronbach_alpha(data):
    n = data.shape[1]
    item_var = data.var(ddof=1).sum()
    total_var = data.sum(axis=1).var(ddof=1)
    return (n / (n - 1)) * (1 - item_var / total_var)

alpha = cronbach_alpha(df[inst_cols])
print(f"\n제도공백 잠재변수 신뢰도 Cronbach α = {alpha:.3f}")
print(f"항목 간 상관:\n{df[inst_cols].corr().round(3)}")

# ─────────────────────────────────────────
# 3. SEM 실행
# ─────────────────────────────────────────
model_desc = """
제도공백 =~ 제도1_분리 + 제도2_민원 + 제도3_절차 + 제도4_신뢰

침해지수   ~ 초등더미 + 학교규모 + 제도공백
정신건강역 ~ 침해지수
이직고려   ~ 침해지수 + 정신건강역
자살사고   ~ 침해지수 + 정신건강역
"""

print("\n=== SEM 실행 중 ===")
model = semopy.Model(model_desc)
model.fit(df)
res = model.inspect()
print(res.to_string())

# 적합도
try:
    fit = semopy.calc_stats(model)
    print("\n=== 모델 적합도 ===")
    print(fit.T)
except Exception as e:
    print(f"적합도 계산 오류: {e}")

# ─────────────────────────────────────────
# 4. 경로계수 정리 (구조 경로만)
# ─────────────────────────────────────────
def get_path(lhs, rhs):
    mask = (res['lval'] == lhs) & (res['rval'] == rhs) & (res['op'] == '~')
    if not mask.any():
        mask = (res['lval'] == lhs) & (res['rval'] == rhs)
    if mask.any():
        row = res.loc[mask].iloc[0]
        est = float(row['Estimate'])
        pval = float(row['p-value'])
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
        return est, pval, sig
    return np.nan, np.nan, ''

paths = {
    ('침해지수',   '제도공백'):   get_path('침해지수',   '제도공백'),
    ('침해지수',   '초등더미'):   get_path('침해지수',   '초등더미'),
    ('침해지수',   '학교규모'):   get_path('침해지수',   '학교규모'),
    ('정신건강역', '침해지수'):   get_path('정신건강역', '침해지수'),
    ('이직고려',   '침해지수'):   get_path('이직고려',   '침해지수'),
    ('이직고려',   '정신건강역'): get_path('이직고려',   '정신건강역'),
    ('자살사고',   '침해지수'):   get_path('자살사고',   '침해지수'),
    ('자살사고',   '정신건강역'): get_path('자살사고',   '정신건강역'),
}

print("\n=== 구조 경로 요약 ===")
for (lhs, rhs), (est, p, sig) in paths.items():
    print(f"  {rhs} → {lhs}: β={est:.3f} {sig}")

# 간접효과 (침해 → 정신건강 → 이직/자살)
b_inj_men = paths[('정신건강역','침해지수')][0]
b_men_ij  = paths[('이직고려','정신건강역')][0]
b_men_su  = paths[('자살사고','정신건강역')][0]
b_inj_ij  = paths[('이직고려','침해지수')][0]
b_inj_su  = paths[('자살사고','침해지수')][0]

indir_ij = b_inj_men * b_men_ij
indir_su = b_inj_men * b_men_su
total_ij = b_inj_ij + indir_ij
total_su = b_inj_su + indir_su

med_pct_ij = abs(indir_ij) / abs(total_ij) * 100 if total_ij != 0 else 0
med_pct_su = abs(indir_su) / abs(total_su) * 100 if total_su != 0 else 0

print(f"\n=== 매개효과 (정신건강역 경유) ===")
print(f"  이직: 직접={b_inj_ij:.3f} 간접={indir_ij:.3f} 총={total_ij:.3f} 매개비율={med_pct_ij:.1f}%")
print(f"  자살: 직접={b_inj_su:.3f} 간접={indir_su:.3f} 총={total_su:.3f} 매개비율={med_pct_su:.1f}%")

# ─────────────────────────────────────────
# 5. 시각화 — SEM 경로도
# ─────────────────────────────────────────
DARK  = '#1A1A2E'
MUTED = '#6B6B7B'
C_LAT = '#DBEAFE'   # 잠재변수: 연파랑
C_OBS = '#DCFCE7'   # 관찰복합: 연녹
C_EXO = '#FEF9C3'   # 외생: 연노랑
C_OUT = '#FCE7F3'   # 결과: 연분홍
C_MED = '#F3E8FF'   # 매개: 연보라

fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
ax.set_facecolor('white')
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')

def node(ax, label, xy, shape='rect', color='white', w=1.6, h=0.55, fs=9):
    x, y = xy
    if shape == 'ellipse':
        e = mpatches.Ellipse((x, y), w, h+0.15, facecolor=color,
                             edgecolor='#555', linewidth=1.5, zorder=3)
        ax.add_patch(e)
    else:
        r = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
                                    boxstyle='round,pad=0.05',
                                    facecolor=color, edgecolor='#555',
                                    linewidth=1.3, zorder=3)
        ax.add_patch(r)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fs, fontweight='bold', color=DARK, zorder=4)

def arrow(ax, x1, y1, x2, y2, label='', color='#1565C0', lw=2.0, fs=8.5, rad=0.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, label, ha='center', va='center', fontsize=fs,
                color=color, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.5))

# ── 노드 배치 ──
# 제도공백 잠재변수 (왼쪽 상단)
node(ax, '제도공백\n(잠재)', (2.0, 5.2), 'ellipse', C_LAT, w=1.8, h=0.7)

# 측정변수 (제도공백 위)
inst_labels = ['분리제도\n미작동', '민원응대\n미작동', '공식절차\n부재', '교권보호위\n불신']
for i, lbl in enumerate(inst_labels):
    node(ax, lbl, (0.6 + i*1.05, 6.5), 'rect', '#EFF6FF', w=0.95, h=0.55, fs=7.5)
    ax.annotate('', xy=(0.6+i*1.05, 6.22), xytext=(2.0, 5.55),
                arrowprops=dict(arrowstyle='->', color='#93C5FD', lw=1.0, alpha=0.7))

# 학교구조 (왼쪽 하단)
node(ax, '초등학교\n더미', (0.9, 3.5), 'rect', C_EXO, w=1.3, h=0.55, fs=8.5)
node(ax, '학교\n규모', (0.9, 2.3), 'rect', C_EXO, w=1.3, h=0.55, fs=8.5)

# 침해지수 (중앙)
node(ax, '교권침해지수', (4.5, 4.0), 'rect', C_OBS, w=1.7, h=0.6, fs=9)

# 정신건강역 (중앙 우측)
node(ax, '정신건강 악화\n(C1 역코딩)', (7.5, 4.0), 'rect', C_MED, w=1.8, h=0.6, fs=8.5)

# 결과변수
node(ax, '이직 고려\n(A4)', (10.5, 5.0), 'rect', C_OUT, w=1.5, h=0.55, fs=8.5)
node(ax, '자살사고\n(C4)', (10.5, 3.0), 'rect', C_OUT, w=1.5, h=0.55, fs=8.5)

# ── 구조 경로 화살표 ──
def fmt(est, sig): return f'β={est:.3f}{sig}'

# 제도공백 → 침해지수
e, p, s = paths[('침해지수','제도공백')]
arrow(ax, 2.9, 5.0, 3.6, 4.25, fmt(e, s), '#DC2626', 2.5, rad=0.1)

# 초등더미 → 침해지수
e, p, s = paths[('침해지수','초등더미')]
arrow(ax, 1.55, 3.5, 3.6, 4.0, fmt(e, s), '#15803D', 1.8)

# 학교규모 → 침해지수
e, p, s = paths[('침해지수','학교규모')]
arrow(ax, 1.55, 2.4, 3.6, 3.75, fmt(e, s), '#15803D', 1.8)

# 침해지수 → 정신건강역
e, p, s = paths[('정신건강역','침해지수')]
arrow(ax, 5.35, 4.0, 6.6, 4.0, fmt(e, s), '#7C3AED', 2.5)

# 정신건강역 → 이직
e, p, s = paths[('이직고려','정신건강역')]
arrow(ax, 8.4, 4.3, 9.75, 4.85, fmt(e, s), '#B45309', 1.8)

# 정신건강역 → 자살
e, p, s = paths[('자살사고','정신건강역')]
arrow(ax, 8.4, 3.7, 9.75, 3.15, fmt(e, s), '#B45309', 1.8)

# 침해지수 → 이직 (직접)
e, p, s = paths[('이직고려','침해지수')]
arrow(ax, 5.35, 4.25, 9.75, 5.0, fmt(e, s), '#64748B', 1.3, rad=-0.25)

# 침해지수 → 자살 (직접)
e, p, s = paths[('자살사고','침해지수')]
arrow(ax, 5.35, 3.75, 9.75, 3.0, fmt(e, s), '#64748B', 1.3, rad=0.25)

# ── 매개 비율 텍스트 박스 ──
ax.text(7.5, 2.0,
        f'정신건강 매개 비율\n이직: {med_pct_ij:.0f}%  |  자살사고: {med_pct_su:.0f}%',
        ha='center', va='center', fontsize=9, color='#4C1D95',
        bbox=dict(facecolor='#F5F3FF', edgecolor='#7C3AED', linewidth=1.2, pad=5))

# ── 범례 ──
legend = [
    mpatches.Patch(facecolor=C_LAT, edgecolor='#555', label='잠재변수 (Latent)'),
    mpatches.Patch(facecolor=C_OBS, edgecolor='#555', label='관찰 복합변수'),
    mpatches.Patch(facecolor=C_EXO, edgecolor='#555', label='구조 변수 (외생)'),
    mpatches.Patch(facecolor=C_MED, edgecolor='#555', label='매개변수'),
    mpatches.Patch(facecolor=C_OUT, edgecolor='#555', label='결과변수'),
]
ax.legend(handles=legend, loc='lower left', fontsize=8, frameon=True,
          framealpha=0.9, edgecolor='#DDD')

# ── 적합도 텍스트 ──
try:
    fit_vals = fit.T
    cfi_val  = float(fit_vals.loc['CFI'].values[0])   if 'CFI'   in fit_vals.index else None
    rmsea_val= float(fit_vals.loc['RMSEA'].values[0]) if 'RMSEA' in fit_vals.index else None
    tli_val  = float(fit_vals.loc['TLI'].values[0])   if 'TLI'   in fit_vals.index else None
    fit_txt  = f'CFI={cfi_val:.3f}  TLI={tli_val:.3f}  RMSEA={rmsea_val:.3f}'
except Exception:
    fit_txt = '적합도 산출 중'

ax.text(6.0, 0.5, fit_txt, ha='center', va='center', fontsize=9.5,
        color=DARK, fontweight='bold',
        bbox=dict(facecolor='#F0F9FF', edgecolor='#0EA5E9', linewidth=1.2, pad=5))

ax.text(6.0, 6.85,
        '교권침해 구조방정식 모형 (SEM v2) — 제도공백 잠재변수 × 인과경로',
        ha='center', va='top', fontsize=12, fontweight='bold', color=DARK)

ax.text(6.0, -0.1,
        '출처: 윤정향, 교원 인권상황 실태조사 2024 (KOSSDA A1-2024-0073) | n=10,888\n'
        '제도공백 잠재변수: A1_12·A1_13(제도 운영 실태) + C6_8(공식 절차) + B8_1(교권보호위 신뢰) 역코딩',
        ha='center', va='top', fontsize=7.5, color=MUTED, style='italic')

plt.tight_layout()
out = BASE + 'output/sem_v2_제도공백_경로도.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f'\n저장: {out}')
plt.close(fig)
print('✅ SEM v2 완료')
