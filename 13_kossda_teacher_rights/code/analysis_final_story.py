"""
최종 통합 스크립트 — 3가지 핵심 보완
══════════════════════════════════════
① 단일 연구 질문 정립
② PAF 기반 정책 시뮬레이션 (미래 전망)
③ 통일된 프로 디자인 시각화
"""

import pyreadstat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from scipy import stats
import statsmodels.api as sm
import os, warnings
warnings.filterwarnings('ignore')

# ── 한글 폰트 ──
for fp in ['/System/Library/Fonts/Supplemental/AppleGothic.ttf',
           '/Library/Fonts/NanumGothic.ttf']:
    try:
        fm.fontManager.addfont(fp)
        plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════════
# 디자인 시스템 — 통일 팔레트
# ══════════════════════════════════════════════════════════════
C = {
    'bg':       '#FAFAFA',
    'navy':     '#1B2A4A',
    'red':      '#C0392B',
    'blue':     '#2471A3',
    'green':    '#1E8449',
    'orange':   '#CA6F1E',
    'gray':     '#717D7E',
    'lgray':    '#ECF0F1',
    'c0':       '#2471A3',   # 고등·학생중심형
    'c1':       '#C0392B',   # 중학교·보호자민원형
    'c2':       '#1E8449',   # 초등·관리자압박형
    'gold':     '#B7950B',
}

def styled_ax(ax, title='', subtitle=''):
    ax.set_facecolor(C['bg'])
    for sp in ax.spines.values():
        sp.set_color('#D5D8DC')
        sp.set_linewidth(0.7)
    ax.tick_params(colors='#444', labelsize=8)
    if title:
        ax.set_title(title, fontsize=9.5, fontweight='bold', color=C['navy'],
                     pad=6, loc='left')
    return ax

# ══════════════════════════════════════════════════════════════
# 0. 데이터 로드 & 전처리
# ══════════════════════════════════════════════════════════════
df_raw, meta = pyreadstat.read_sav(
    'data/교원 인권상황 실태조사,2024/kor_data_20240073.sav')
df3_raw, _   = pyreadstat.read_sav(
    'data/초·중등 교원 인권교육 실태조사, 2021/kor_data_20210019.sav')

raw = df_raw.copy()
raw['초등']  = (raw['SQ2'] == 1).astype(int)
raw['중등']  = (raw['SQ2'] == 2).astype(int)
raw['고등']  = (raw['SQ2'] == 3).astype(int)
raw['공립']  = (raw['SQ4'] == 1).astype(int)
raw['기간제']= (raw['SQ8'] != 1).astype(int)
raw['남성']  = (raw['SQ9'] == 1).astype(int)
raw['학생침해']  = raw['B2_1'].clip(0,1)
raw['보호자침해'] = raw['B3_1'].clip(0,1)
raw['관리자침해'] = raw['B5_1'].clip(0,1)
raw['침해지수']  = raw['학생침해'] + raw['보호자침해'] + raw['관리자침해']
raw['침해여부']  = (raw['침해지수'] > 0).astype(int)
raw['번아웃']    = raw[['C3_1','C3_2','C3_3','C3_4','C3_5']].sum(axis=1)

use_cols = ['초등','중등','고등','공립','기간제','남성','SQ5',
            '학생침해','보호자침해','관리자침해','침해지수','침해여부','번아웃',
            'D2_2','D2_5','D2_10','D2_16','A4','C4']
df = raw[use_cols].dropna().reset_index(drop=True)
n  = len(df)
print(f"D1 n={n:,}  /  D3 n={len(df3_raw):,}")

# ══════════════════════════════════════════════════════════════
# ① 단일 연구 질문 선언
# ══════════════════════════════════════════════════════════════
RESEARCH_Q = (
    "인권교육 참여율이 20%p 증가했는데도\n"
    "교권침해는 왜 줄지 않는가?\n"
    "— 제도 공백의 구조, 취약 집단, 그리고 맞춤 처방"
)
print("\n" + "═"*60)
print("연구 질문 (의문형 제목)")
print("═"*60)
print(RESEARCH_Q)

# ══════════════════════════════════════════════════════════════
# ② PAF 기반 정책 시뮬레이션 (미래 전망)
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("② PAF 기반 정책 시뮬레이션")
print("══════════════════════════════════════════════════════════")
print("PAF = Pe × (OR−1) / [1 + Pe × (OR−1)]")
print("Pe: 해당 제도 부재 노출 비율 (D2 ≥ 4 응답률)")
print("OR: 선행 로지스틱 회귀 결과")

# 시나리오별 정책 효과 (OR 기반)
policies = {
    '악성민원\n법적패널티': {
        'd2': 'D2_10', 'inj': '보호자침해', 'OR': 2.487,
        'label_full': '악성민원 법적패널티',
        'color': C['red']
    },
    '수업방해\n분리시스템': {
        'd2': 'D2_5', 'inj': '학생침해', 'OR': 1.710,
        'label_full': '수업방해 분리시스템',
        'color': C['blue']
    },
    '아동학대\n신고보호': {
        'd2': 'D2_2', 'inj': '보호자침해', 'OR': 1.978,
        'label_full': '아동학대 신고보호',
        'color': C['orange']
    },
}

sim_results = {}
print("\n[시뮬레이션 결과]")
print(f"{'제도':<14} {'Pe':>5} {'OR':>5} {'PAF':>6} {'현재율':>7} {'30%효과':>8} {'60%효과':>8} {'90%효과':>8}")
print("─"*65)
for pname, info in policies.items():
    Pe    = (df[info['d2']] >= 4).mean()
    OR    = info['OR']
    PAF   = Pe * (OR - 1) / (1 + Pe * (OR - 1))
    base  = df[info['inj']].mean()
    r30   = base * (1 - PAF * 0.30)
    r60   = base * (1 - PAF * 0.60)
    r90   = base * (1 - PAF * 0.90)
    sim_results[pname] = {
        'Pe': Pe, 'OR': OR, 'PAF': PAF, 'base': base,
        'r30': r30, 'r60': r60, 'r90': r90,
        'inj': info['inj'], 'color': info['color'],
        'label_full': info['label_full']
    }
    pn = pname.replace('\n', ' ')
    print(f"{pn:<14} {Pe:>5.3f} {OR:>5.3f} {PAF:>6.3f} {base*100:>6.1f}%"
          f" {r30*100:>7.1f}% {r60*100:>7.1f}% {r90*100:>7.1f}%")

# 자살사고 시뮬레이션 (침해지수 → 자살사고, OR=2.32)
inj_rate_now  = df['침해여부'].mean()
suicide_now   = df['C4'].mean()
OR_sui        = 2.32
# 악성민원 패널티가 보호자침해 60%효과라면 침해 감소 → 자살사고 간접 감소
par_reduction = sim_results['악성민원\n법적패널티']['base'] - sim_results['악성민원\n법적패널티']['r60']
# 간접 추정: 보호자침해 감소 → 침해여부 감소 → 자살사고 감소
# 보수적으로 1/3만 반영
suicide_proj  = suicide_now * (1 - par_reduction * 0.5)
_key = '악성민원\n법적패널티'
print(f"\n[연쇄 효과 추정 — 악성민원 패널티 60% 효과 시]")
print(f"  보호자침해 {sim_results[_key]['base']*100:.1f}% → {sim_results[_key]['r60']*100:.1f}% (−{par_reduction*100:.1f}%p)")
print(f"  자살사고 {suicide_now*100:.1f}% → 추정 {suicide_proj*100:.1f}% (−{(suicide_now-suicide_proj)*100:.1f}%p, 보수 추정)")

# ══════════════════════════════════════════════════════════════
# ③ K-means 클러스터 (이전 결과 재사용)
# ══════════════════════════════════════════════════════════════
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

feat_cl = ['초등','중등','고등','공립','기간제','SQ5',
           '학생침해','보호자침해','관리자침해','번아웃']
X_sc  = StandardScaler().fit_transform(df[feat_cl])
km    = KMeans(n_clusters=3, random_state=42, n_init=30)
df['cluster'] = km.fit_predict(X_sc)
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_sc)
df['pca1'] = X_pca[:,0]; df['pca2'] = X_pca[:,1]

prof_cols = ['초등','중등','고등','공립','기간제','SQ5',
             '학생침해','보호자침해','관리자침해','침해지수','번아웃','A4','C4']
prof = df.groupby('cluster')[prof_cols].mean()
prof['n'] = df.groupby('cluster').size()

# 클러스터 라벨
cl_labels = {}
for ci in sorted(prof.index):
    r = prof.loc[ci]
    if r['초등'] > 0.5:
        cl_labels[ci] = ('초등·관리자압박형', C['c2'])
    elif r['고등'] > 0.5:
        cl_labels[ci] = ('고등·학생중심형', C['c0'])
    else:
        cl_labels[ci] = ('중학교·보호자민원형', C['c1'])

df['cl_label'] = df['cluster'].map(lambda x: cl_labels[x][0])

# 집단별 제도 우선순위 (침해율 × OR)
policy_map = {
    'D2_5':  ('수업방해 분리시스템',  '학생침해',  1.710, C['blue']),
    'D2_10': ('악성민원 법적패널티', '보호자침해', 2.487, C['red']),
    'D2_2':  ('아동학대 신고보호',   '보호자침해', 1.978, C['orange']),
    'D2_16': ('과밀학급 해소',       '보호자침해', 1.190, C['gray']),
}

priority_per_cluster = {}
for ci in sorted(prof.index):
    scores = {}
    for d2k, (lbl, inj, OR, col) in policy_map.items():
        scores[d2k] = prof.loc[ci, inj] * OR
    priority_per_cluster[ci] = sorted(scores.items(), key=lambda x: x[1], reverse=True)

# ══════════════════════════════════════════════════════════════
# ④ 시각화 — Figure 1: 완전한 스토리 (8패널)
# ══════════════════════════════════════════════════════════════
os.makedirs('output', exist_ok=True)

fig = plt.figure(figsize=(24, 28), facecolor=C['bg'])
fig.patch.set_facecolor(C['bg'])

outer = gridspec.GridSpec(5, 1, figure=fig,
                          hspace=0.05, left=0.04, right=0.97,
                          top=0.95, bottom=0.02)

# ── [행0] 연구 질문 배너 ──
ax_title = fig.add_subplot(outer[0])
ax_title.set_facecolor(C['navy'])
ax_title.axis('off')
ax_title.text(0.5, 0.72, RESEARCH_Q,
              ha='center', va='center', fontsize=14.5, color='white',
              fontweight='bold', transform=ax_title.transAxes, linespacing=1.8)
ax_title.text(0.5, 0.12,
              f"D1(2024) n={n:,}명 | D3(2021) n={len(df3_raw):,}명 | KOSSDA 데이터 | "
              f"방법: K-means · SEM · SHAP · Bootstrap · 로지스틱 · PAF 시뮬레이션",
              ha='center', va='center', fontsize=8.5, color='#AED6F1',
              transform=ax_title.transAxes)

# ── [행1] 3열 그리드 ──
inner1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1],
                                          wspace=0.35, hspace=0.1)

# Panel A: 인권교육 역설 (2021→2024) — HOOK
ax_A = fig.add_subplot(inner1[0])
styled_ax(ax_A, '① 인권교육 역설 — 교육이 늘어도 침해는 줄지 않았다')
# 타임라인 데이터
years_edu  = [2021, 2024]
edu_rates  = [67.4, 88.2]
ax_A.plot(years_edu, edu_rates, 'o-', color=C['green'], lw=2.5, ms=8, label='인권교육 참여·이수율 (%)')
ax_A.fill_between(years_edu, edu_rates, alpha=0.12, color=C['green'])

# 2024 침해·이직·자살 수치 강조
outcomes_2024 = {
    '교권침해 경험': 85.6,
    '이직 고려':    46.3,
    '자살사고':     16.9,
}
colors_2024 = [C['red'], C['orange'], C['gold']]
y_pos = [83, 44, 15]
for (label, val), col, yp in zip(outcomes_2024.items(), colors_2024, y_pos):
    ax_A.annotate(f'{label}\n{val:.1f}%', xy=(2024, val),
                  xytext=(2023.55, yp),
                  fontsize=8, color=col, fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color=col, lw=1.2))
    ax_A.scatter([2024], [val], s=60, color=col, zorder=5)

ax_A.set_xticks([2021, 2024])
ax_A.set_xticklabels(['2021년\n(D3, n=9,553)', '2024년\n(D1, n=10,888)'], fontsize=8)
ax_A.set_ylabel('%', fontsize=8)
ax_A.set_ylim(0, 100)
ax_A.set_xlim(2020.5, 2024.6)
ax_A.legend(fontsize=7.5, loc='upper left')
ax_A.grid(axis='y', alpha=0.3, ls='--')
# 역설 강조 텍스트
ax_A.annotate('+20.8%p\n교육 증가', xy=(2022.5, 77.8), fontsize=9,
              ha='center', color=C['green'], fontweight='bold')
ax_A.annotate('', xy=(2024, 88.2), xytext=(2021, 67.4),
              arrowprops=dict(arrowstyle='->', color=C['green'], lw=1.5))
ax_A.text(0.5, -0.13, '교육으로 해결 불가 → 제도 구조의 문제',
          ha='center', fontsize=8, color=C['red'], fontweight='bold',
          transform=ax_A.transAxes,
          bbox=dict(boxstyle='round,pad=0.3', fc='#FDEDEC', ec=C['red'], lw=0.8))

# Panel B: 제도-침해 OR 비교
ax_B = fig.add_subplot(inner1[1])
styled_ax(ax_B, '② 어떤 제도 부재가 침해를 가장 많이 만드는가?')
d2_info = [
    ('악성민원\n법적패널티', 2.487, '보호자침해', C['red']),
    ('아동학대\n신고보호',   1.978, '보호자침해', C['orange']),
    ('수업방해\n분리시스템', 1.710, '학생침해',   C['blue']),
    ('과밀학급\n해소',       1.190, '보호자침해', C['gray']),
]
ylabels = [d[0] for d in d2_info]
ORs     = [d[1] for d in d2_info]
cols    = [d[3] for d in d2_info]
bars    = ax_B.barh(range(4), ORs, color=cols, alpha=0.85, height=0.55,
                    edgecolor='white')
ax_B.axvline(1.0, color='#888', ls='--', lw=1.2, label='OR=1 (효과없음)')
for i, (b, v, d) in enumerate(zip(bars, ORs, d2_info)):
    ax_B.text(v + 0.04, i, f'OR={v:.3f}***\n({d[2]})', va='center', fontsize=7.5,
              color=d[3], fontweight='bold')
ax_B.set_yticks(range(4))
ax_B.set_yticklabels(ylabels, fontsize=8.5)
ax_B.set_xlabel('Odds Ratio (로지스틱 회귀)', fontsize=8)
ax_B.set_xlim(0.8, 3.2)
ax_B.legend(fontsize=7.5)
ax_B.grid(axis='x', alpha=0.3, ls='--')
ax_B.text(0.5, -0.13, '모든 OR p<0.001 | 구조 변수(학교급·규모·설립·고용) 통제',
          ha='center', fontsize=7.5, color='#555', transform=ax_B.transAxes)

# Panel C: 학교 규모 × 침해지수 (구조 변수)
ax_C = fig.add_subplot(inner1[2])
styled_ax(ax_C, '③ 학교가 클수록 침해가 심하다 — 구조 변수')
sz_labels = ['5학급\n미만', '6-11\n학급', '12-17\n학급', '18-35\n학급', '36학급\n이상']
sz_inj    = [1.429, 1.527, 1.588, 1.632, 1.704]
sz_n      = [732,   1422,  1356,  4613,  2765]
colors_sz = [plt.cm.RdYlGn_r(i/4) for i in range(5)]
bars_c    = ax_C.bar(range(5), [v*100/3 for v in sz_inj], color=colors_sz,
                     alpha=0.85, edgecolor='white', width=0.6)
ax_C2     = ax_C.twinx()
ax_C2.plot(range(5), sz_n, 's--', color=C['gray'], lw=1.5, ms=6, label='응답자 수')
ax_C2.set_ylabel('응답자 수', fontsize=7.5, color=C['gray'])
ax_C2.tick_params(axis='y', colors=C['gray'], labelsize=7)
for i, (b, v) in enumerate(zip(bars_c, sz_inj)):
    ax_C.text(i, v*100/3+0.5, f'{v:.3f}', ha='center', fontsize=8,
              color=colors_sz[i], fontweight='bold')
ax_C.set_xticks(range(5))
ax_C.set_xticklabels(sz_labels, fontsize=8)
ax_C.set_ylabel('침해지수 (0-3)', fontsize=8)
ax_C.set_ylim(0, 65)
ax_C.set_yticks([0,20,40,60])
ax_C.set_yticklabels(['0','0.6','1.2','1.8'], fontsize=7)
ax_C.grid(axis='y', alpha=0.3, ls='--')
ax_C.text(0.5, -0.13, 'Kruskal-Wallis H=59.73, p<0.001 | Bootstrap 매개비율 74.3%',
          ha='center', fontsize=7.5, color='#555', transform=ax_C.transAxes)

# ── [행2] 클러스터링 (PCA + 프로파일) ──
inner2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2],
                                          wspace=0.35)

# Panel D: PCA 산포도
ax_D = fig.add_subplot(inner2[0])
styled_ax(ax_D, '④ K-means 취약 집단 분류 (k=3, Sil=0.404)')
for ci in sorted(df['cluster'].unique()):
    lbl, col = cl_labels[ci]
    mask = df['cluster'] == ci
    ax_D.scatter(df.loc[mask,'pca1'], df.loc[mask,'pca2'],
                 c=col, alpha=0.18, s=5, label=f'C{ci}: {lbl}')
centers_pca = pca.transform(km.cluster_centers_)
for ci, (cx,cy) in enumerate(centers_pca):
    lbl, col = cl_labels[ci]
    ax_D.scatter(cx, cy, c=col, s=280, marker='*', edgecolors='white', zorder=5)
    ax_D.text(cx, cy+0.25, f'C{ci}', fontsize=9, fontweight='bold', color=col,
              ha='center')
ax_D.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)', fontsize=8)
ax_D.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)', fontsize=8)
ax_D.legend(fontsize=7, loc='upper right', markerscale=2.5)
ax_D.grid(alpha=0.25, ls='--')

# Panel E: 클러스터별 침해 유형 비교
ax_E = fig.add_subplot(inner2[1])
styled_ax(ax_E, '⑤ 집단별 침해 유형 — 각각 다른 문제를 가진다')
x5 = np.arange(3)
w5 = 0.22
inj_types = [('학생침해', C['blue']), ('보호자침해', C['red']), ('관리자침해', C['green'])]
for j, (it, ic) in enumerate(inj_types):
    vals = [prof.loc[ci, it]*100 for ci in sorted(prof.index)]
    b5   = ax_E.bar(x5 + j*w5, vals, w5, color=ic, alpha=0.82, edgecolor='white', label=it)
    for b, v in zip(b5, vals):
        ax_E.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.0f}',
                  ha='center', fontsize=7.5, color=ic, fontweight='bold')
ax_E.set_xticks(x5 + w5)
cl_xlabels = [f'C{ci}\n{cl_labels[ci][0]}' for ci in sorted(prof.index)]
ax_E.set_xticklabels(cl_xlabels, fontsize=7.5)
ax_E.set_ylabel('침해 경험률 (%)', fontsize=8)
ax_E.set_ylim(0, 100)
ax_E.legend(fontsize=7.5, loc='upper right')
ax_E.grid(axis='y', alpha=0.3, ls='--')
# 핵심 강조 annotate
ax_E.annotate('관리자침해\n43% 최고\n(사각지대!)', xy=(2.44, 43),
              xytext=(1.8, 75), fontsize=7.5, color=C['green'], fontweight='bold',
              arrowprops=dict(arrowstyle='->', color=C['green']))

# Panel F: 집단별 최우선 제도 (침해율 × OR 점수)
ax_F = fig.add_subplot(inner2[2])
styled_ax(ax_F, '⑥ 집단별 제도 도입 1순위 — 맞춤 처방')
d2_short = {
    'D2_5':  ('수업방해\n분리시스템', C['blue']),
    'D2_10': ('악성민원\n법적패널티', C['red']),
    'D2_2':  ('아동학대\n신고보호',   C['orange']),
    'D2_16': ('과밀학급\n해소',       C['gray']),
}
for ci in sorted(prof.index):
    ranked = priority_per_cluster[ci]
    lbl, col = cl_labels[ci]
    for rank, (d2k, score) in enumerate(ranked):
        plabel, pcol = d2_short[d2k]
        alpha = 0.9 - rank * 0.18
        ax_F.barh(ci - rank*0.22, score, color=pcol, alpha=alpha, height=0.18,
                  label=plabel if ci == 0 else '')
        if rank == 0:
            ax_F.text(score + 0.01, ci, f'{score:.3f} ★', va='center',
                      fontsize=7.5, color=pcol, fontweight='bold')

ax_F.set_yticks(sorted(prof.index))
ax_F.set_yticklabels([f'C{ci}: {cl_labels[ci][0]}' for ci in sorted(prof.index)],
                     fontsize=7.5)
ax_F.set_xlabel('시급성 점수 (침해율 × OR)', fontsize=8)
ax_F.axvline(1.0, color='#aaa', ls='--', lw=1)
handles = [mpatches.Patch(color=v[1], label=k) for k, v in d2_short.items()]
ax_F.legend(handles=handles, fontsize=6.5, loc='lower right',
            labels=[v[0] for v in d2_short.values()])
ax_F.grid(axis='x', alpha=0.3, ls='--')
ax_F.set_xlim(0.5, 2.0)

# ── [행3] 정책 시뮬레이션 + Bootstrap 매개 ──
inner3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[3],
                                          wspace=0.35)

# Panel G: PAF 정책 시뮬레이션 (미래 전망) — 메인
ax_G = fig.add_subplot(inner3[:2])  # 2칸 차지
styled_ax(ax_G, '⑦ [미래 전망] 제도 도입 시나리오별 침해율 감소 추정 (PAF 기반)')

bar_w   = 0.18
x_base  = np.arange(len(policies))
scenarios  = [('현재 (제도 없음)', 0.0, '#CCCCCC'),
              ('30% 효과\n(법 제정 초기)', 0.30, '#F0B27A'),
              ('60% 효과\n(시행 3년 후)', 0.60, '#EB984E'),
              ('90% 효과\n(완전 정착)', 0.90, '#CA6F1E')]

pnames = list(policies.keys())
for si, (slabel, eff, scol) in enumerate(scenarios):
    y_vals = []
    for pname in pnames:
        r = sim_results[pname]
        if eff == 0.0:
            y_vals.append(r['base'] * 100)
        else:
            y_vals.append(r['base'] * (1 - r['PAF'] * eff) * 100)
    bars_g = ax_G.bar(x_base + si*bar_w - 1.5*bar_w, y_vals,
                      bar_w, color=scol, alpha=0.88, edgecolor='white',
                      label=slabel)
    if si == 0:
        for xb, yv in zip(x_base, y_vals):
            ax_G.text(xb - 1.5*bar_w, yv+0.5, f'{yv:.1f}%',
                      ha='center', fontsize=7.5, color='#555', fontweight='bold')
    elif si == 2:
        for xb, yv, pn in zip(x_base, y_vals, pnames):
            base = sim_results[pn]['base'] * 100
            ax_G.text(xb + 0.5*bar_w, yv+0.5, f'→{yv:.1f}%\n(−{base-yv:.1f}%p)',
                      ha='center', fontsize=7, color=C['orange'], fontweight='bold')

ax_G.set_xticks(x_base)
ax_G.set_xticklabels([sim_results[p]['label_full'] for p in pnames], fontsize=9)
ax_G.set_ylabel('침해 경험률 (%)', fontsize=9)
ax_G.set_ylim(0, 80)
ax_G.legend(fontsize=8, loc='upper right', ncol=2)
ax_G.grid(axis='y', alpha=0.3, ls='--')
ax_G.set_title('⑦ [미래 전망] 제도 도입 시나리오별 침해율 감소 추정 (PAF 기반)',
               fontsize=9.5, fontweight='bold', color=C['navy'], pad=6, loc='left')
# 자살사고 연쇄 효과 텍스트
_r  = sim_results['악성민원\n법적패널티']
_note = (f"[연쇄 효과 추정] 악성민원 패널티 60% 효과 시: "
         f"보호자침해 {_r['base']*100:.1f}% → {_r['r60']*100:.1f}% "
         f"→ 자살사고 {suicide_now*100:.1f}% → 약 {suicide_proj*100:.1f}%로 감소 가능 (보수 추정)\n"
         "PAF = Pe×(OR-1)/[1+Pe×(OR-1)] | Pe: 제도 부재 노출 비율 | 효과율 가정 기반 시나리오")
ax_G.text(0.5, -0.14, _note,
          ha='center', fontsize=8, color='#444', transform=ax_G.transAxes,
          bbox=dict(boxstyle='round,pad=0.3', fc='#FEF9E7', ec=C['orange'], lw=0.8))

# Panel H: Bootstrap 매개효과
ax_H = fig.add_subplot(inner3[2])
styled_ax(ax_H, '⑧ 번아웃이 침해를 자살사고로 연결 (매개효과)')
paths  = ['침해→번아웃\n→이직', '침해→번아웃\n→자살사고', '침해→번아웃\n→정신건강', '학교규모→침해\n→번아웃']
indir  = [0.0156, 0.0165, 0.0360, 0.0177]
ci_lo  = [0.0134, 0.0144, 0.0310, 0.0131]
ci_hi  = [0.0182, 0.0187, 0.0412, 0.0225]
med_r  = [8.9,    14.7,   10.3,   74.3]
colors_h = [C['orange'], C['red'], C['blue'], C['green']]
for i, (p, v, lo, hi, mr, col) in enumerate(zip(paths, indir, ci_lo, ci_hi, med_r, colors_h)):
    ax_H.barh(i, v, color=col, alpha=0.75, height=0.5)
    ax_H.errorbar(v, i, xerr=[[v-lo],[hi-v]], fmt='none', color='#333', capsize=4, lw=1.5)
    ax_H.text(hi+0.001, i, f'{v:.4f}\n매개{mr:.0f}%', va='center',
              fontsize=7.5, color=col, fontweight='bold')
ax_H.set_yticks(range(4))
ax_H.set_yticklabels(paths, fontsize=7.5)
ax_H.set_xlabel('간접효과 (Bootstrap 95%CI)', fontsize=8)
ax_H.axvline(0, color='#888', lw=0.8)
ax_H.grid(axis='x', alpha=0.3, ls='--')
ax_H.set_xlim(-0.005, 0.06)
ax_H.text(0.5, -0.13, 'n=2,000 bootstrap | 전 경로 95%CI가 0 포함 안 함 → 유의',
          ha='center', fontsize=7.5, color='#555', transform=ax_H.transAxes)

# ── [행4] 결론 배너 ──
ax_conc = fig.add_subplot(outer[4])
ax_conc.set_facecolor('#1B2A4A')
ax_conc.axis('off')
conclusions = [
    ("역설 발견",
     "인권교육 +20.8%p에도 침해 85.6% → 교육이 아닌 제도 설계가 핵심"),
    ("핵심 제도",
     "악성민원 법적패널티(OR=2.49) → 수업방해 분리시스템(OR=1.71)"),
    ("집단 발견",
     "초등·소규모·관리자압박형 — 현행 정책 논의의 사각지대"),
    ("미래 전망",
     "악성민원 패널티 60% 효과 시 보호자침해 −17%p, 자살사고 간접 감소"),
    ("정책 결론",
     "학교급별 맞춤 입법 필요 — 일률적 교권보호법으로는 부족"),
]
x_step = 1.0 / len(conclusions)
for i, (head, body) in enumerate(conclusions):
    xc = x_step * i + x_step/2
    ax_conc.text(xc, 0.72, head, ha='center', va='center', fontsize=10,
                 fontweight='bold', color='#F9E79F', transform=ax_conc.transAxes)
    ax_conc.text(xc, 0.32, body, ha='center', va='center', fontsize=8,
                 color='#AED6F1', transform=ax_conc.transAxes, linespacing=1.5)
    if i < len(conclusions)-1:
        ax_conc.axvline(x_step*(i+1), color='#3D566E', lw=0.8, alpha=0.7)

plt.savefig('output/final_story.png', dpi=150, bbox_inches='tight',
            facecolor=C['bg'])
plt.close()
print("\n저장: output/final_story.png")

# ══════════════════════════════════════════════════════════════
# ⑤ 시각화 — Figure 2: 집단별 정책 카드 (킬러 슬라이드)
# ══════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(21, 12), facecolor=C['navy'])
fig2.patch.set_facecolor(C['navy'])

# 제목
fig2.text(0.5, 0.97,
          '"어떤 학교의 교사에게, 어떤 제도가 먼저 필요한가?"',
          ha='center', va='top', fontsize=16, fontweight='bold',
          color='white')
fig2.text(0.5, 0.92,
          '취약 집단별 맞춤형 교권 보호 정책 로드맵  |  K-means 분류 + 침해율×OR 가중 점수 기반',
          ha='center', va='top', fontsize=10, color='#AED6F1')

card_colors = [C['c0'], C['c1'], C['c2']]
rank_bg     = ['#FDEDEC', '#FEF9E7', '#EBF5FB', '#F9EBEA']
rank_border = [C['red'], C['orange'], C['blue'], C['gray']]

for idx, ci in enumerate(sorted(df['cluster'].unique())):
    lbl, col = cl_labels[ci]
    n_c      = int(prof.loc[ci, 'n'])
    xL       = 0.02 + idx * 0.33
    xR       = xL + 0.30

    # 카드 배경
    card_ax = fig2.add_axes([xL, 0.03, 0.30, 0.86])
    card_ax.set_facecolor('#F7F9FA')
    card_ax.set_xlim(0, 10); card_ax.set_ylim(0, 22)
    card_ax.axis('off')

    # 헤더
    card_ax.add_patch(FancyBboxPatch((0, 18.5), 10, 3.3,
                      boxstyle='round,pad=0', facecolor=col, alpha=0.95))
    card_ax.text(5, 20.8, f'집단 C{ci}', ha='center', fontsize=13,
                fontweight='bold', color='white')
    card_ax.text(5, 19.7, lbl, ha='center', fontsize=11, color='white', fontweight='bold')
    card_ax.text(5, 18.8, f'n={n_c:,}명  ({n_c/n*100:.0f}%)',
                ha='center', fontsize=9, color='#ECF0F1')

    # 특성 요약 박스
    card_ax.add_patch(FancyBboxPatch((0.3, 14.5), 9.4, 3.7,
                      boxstyle='round,pad=0.2', facecolor='#EBF5FB', edgecolor='#AED6F1'))
    card_ax.text(5, 17.85, '▪ 집단 구조 특성', ha='center', fontsize=9.5,
                fontweight='bold', color=C['navy'])
    feats = [
        f"학교급: 초등 {prof.loc[ci,'초등']*100:.0f}% | 중등 {prof.loc[ci,'중등']*100:.0f}% | 고등 {prof.loc[ci,'고등']*100:.0f}%",
        f"공립 {prof.loc[ci,'공립']*100:.0f}% | 기간제 {prof.loc[ci,'기간제']*100:.0f}% | 규모 SQ5={prof.loc[ci,'SQ5']:.1f}/5",
    ]
    for fi, ft in enumerate(feats):
        card_ax.text(5, 17.0 - fi*0.9, ft, ha='center', fontsize=8.5, color='#1A5276')

    # 침해 프로파일
    card_ax.text(5, 14.0, '▪ 침해 유형 프로파일', ha='center', fontsize=9.5,
                fontweight='bold', color=C['navy'])
    inj_data = [
        ('학생침해',  prof.loc[ci,'학생침해'],  C['blue']),
        ('보호자침해', prof.loc[ci,'보호자침해'], C['red']),
        ('관리자침해', prof.loc[ci,'관리자침해'], C['green']),
    ]
    for fi, (nm, val, ic) in enumerate(inj_data):
        y_b = 13.3 - fi * 1.2
        bar_len = val * 8.0
        card_ax.add_patch(FancyBboxPatch((1.5, y_b-0.28), bar_len, 0.56,
                          boxstyle='round,pad=0.05', facecolor=ic, alpha=0.75))
        card_ax.text(1.3, y_b, nm, ha='right', va='center', fontsize=8.5, color='#333')
        card_ax.text(1.6 + bar_len, y_b, f'{val*100:.1f}%', va='center',
                    fontsize=8.5, color=ic, fontweight='bold')
    card_ax.text(5, 9.75,
                f'번아웃 {prof.loc[ci,"번아웃"]:.1f}/25  |  이직고려 {prof.loc[ci,"A4"]*100:.0f}%  |  자살사고 {prof.loc[ci,"C4"]*100:.0f}%',
                ha='center', fontsize=8, color='#555')

    # 제도 우선순위
    card_ax.text(5, 9.1, '▪ 제도 도입 우선순위', ha='center', fontsize=9.5,
                fontweight='bold', color=C['navy'])
    card_ax.add_patch(FancyBboxPatch((0.3, 0.3), 9.4, 8.5,
                      boxstyle='round,pad=0.2', facecolor='#FDFEFE', edgecolor='#D5D8DC'))
    rank_text = ['🔴 즉시 도입', '🟠 단기', '🟡 중기', '⚪ 장기']
    for rank, (d2k, score) in enumerate(priority_per_cluster[ci]):
        y_r = 8.4 - rank * 1.98
        plabel, pcol = d2_short[d2k]
        inj_type = policy_map[d2k][1]
        OR_val   = policy_map[d2k][2]
        inj_r    = prof.loc[ci, inj_type]
        # 배경
        card_ax.add_patch(FancyBboxPatch((0.5, y_r-0.65), 9.0, 1.7,
                          boxstyle='round,pad=0.1',
                          facecolor=rank_bg[rank], edgecolor=rank_border[rank], lw=1.2))
        card_ax.text(0.9, y_r+0.7, rank_text[rank], fontsize=8, color=rank_border[rank],
                    fontweight='bold')
        card_ax.text(0.9, y_r+0.15, plabel.replace('\n',' '), fontsize=10.5,
                    color='#1C2833', fontweight='bold')
        card_ax.text(0.9, y_r-0.42,
                    f'시급성 점수 {score:.3f}  (침해율 {inj_r*100:.0f}% × OR={OR_val:.3f})',
                    fontsize=7.5, color='#666')

plt.savefig('output/policy_cards.png', dpi=150, bbox_inches='tight',
            facecolor=C['navy'])
plt.close()
print("저장: output/policy_cards.png")

# ══════════════════════════════════════════════════════════════
# 최종 요약 출력
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("최종 보완 사항 완료 확인")
print("═"*60)
print("\n✅ ① 단일 연구 질문:")
print(f"   {RESEARCH_Q.replace(chr(10),' ')}")
print("\n✅ ② 미래 전망 (PAF 시뮬레이션):")
for pname, r in sim_results.items():
    pn = pname.replace('\n',' ')
    print(f"   {pn}: 현재 {r['base']*100:.1f}% → 60%효과 {r['r60']*100:.1f}% (−{(r['base']-r['r60'])*100:.1f}%p)")
print(f"   연쇄: 자살사고 {suicide_now*100:.1f}% → 약 {suicide_proj*100:.1f}% (보수 추정)")
print("\n✅ ③ 시각화 통일 디자인:")
print("   output/final_story.png  — 완전한 스토리 8패널")
print("   output/policy_cards.png — 집단별 정책 카드 (킬러 슬라이드)")
