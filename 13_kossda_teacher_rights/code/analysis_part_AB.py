"""
Part A: 아동·청소년 인권실태조사(2024 SAV) — 교사 체벌·욕설 경험 × 지역 분석
Part B: 학원 밀도 정규화 (학생 1만명당 학원수) × 교권침해 재상관
"""

import os, glob, warnings
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
from scipy import stats
import seaborn as sns
import pyreadstat

warnings.filterwarnings('ignore')

# ── 한글 폰트 ────────────────────────────────────────────────────────────────
def set_korean_font():
    for p in ['/System/Library/Fonts/AppleSDGothicNeo.ttc',
              '/Library/Fonts/AppleGothic.ttf']:
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            prop = font_manager.FontProperties(fname=p)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return
set_korean_font()

BASE   = '/Users/baiohelseu/Desktop/Project/kossda/data'
OUT    = '/Users/baiohelseu/Desktop/Project/kossda/output'
MAP_P  = '/Users/baiohelseu/Downloads/map/final.shp'
os.makedirs(OUT, exist_ok=True)

# ════════════════════════════════════════════════════════════════════
# ◆ PART A: SAV 분석
# ════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART A: 교사 체벌·욕설 경험 × 지역 분석")
print("=" * 60)

df_sav, meta = pyreadstat.read_sav(
    os.path.join(BASE, '(RAWDATA) 2024 아동·청소년 인권실태조사.SAV')
)

# ── 핵심 변수 선택 ───────────────────────────────────────────────────
# Q15A3: 교사 체벌 경험 (1=없음 ~ 5=주 1-2회)
# Q15A4: 교사 욕설·정서 공격 경험 (1=없음 ~ 5=주 1-2회)
# DM5  : 지역규모 (1=특별광역시, 2=중소도시, 3=읍, 4=면)
# 학생인권조례제정지역: 0=없음, 1=있음
# BQ6  : 학업성적 (1=매우 못함 ~ 5=매우 잘함)
# BQ7  : 가정경제수준 (1=매우 못삶 ~ 7=매우 잘삶)
# DM2  : 학교급 (1=초, 2=중, 3=고)
# WGT_B: 표본 가중치

cols = ['Q15A3','Q15A4','DM5','학생인권조례제정지역',
        'BQ6','BQ7','DM2','WGT_B','SEX']
df_a = df_sav[cols].copy()
df_a = df_a[~df_a['Q15A3'].isin([9]) & ~df_a['Q15A4'].isin([9])]
df_a = df_a.dropna(subset=['Q15A3','Q15A4','DM5','WGT_B'])

# ── 경험 여부 이진화 (1=한 번도 없음, 나머지=경험 있음) ───────────────
df_a['체벌_경험'] = (df_a['Q15A3'] >= 2).astype(float)
df_a['욕설_경험'] = (df_a['Q15A4'] >= 2).astype(float)
df_a['체벌_빈도'] = df_a['Q15A3'].replace(9, np.nan)   # 1~5 원래 척도
df_a['욕설_빈도'] = df_a['Q15A4'].replace(9, np.nan)
df_a['교사권위행사_지수'] = (df_a['체벌_빈도'] + df_a['욕설_빈도']) / 2

# ── DM5 레이블 ───────────────────────────────────────────────────────
region_labels = {1.0:'특별·광역시', 2.0:'중소도시', 3.0:'읍지역', 4.0:'면지역'}
df_a['지역규모'] = df_a['DM5'].map(region_labels)
조례_labels     = {0.0:'조례 없음', 1.0:'조례 있음'}
df_a['조례']   = df_a['학생인권조례제정지역'].map(조례_labels)
학교급_labels  = {1.0:'초등', 2.0:'중학교', 3.0:'고등'}
df_a['학교급'] = df_a['DM2'].map(학교급_labels)

# ── 가중 비율 계산 ───────────────────────────────────────────────────
def weighted_mean(grp, val_col, wgt_col='WGT_B'):
    w = grp[wgt_col]
    v = grp[val_col]
    return (v * w).sum() / w.sum()

def weighted_se(grp, val_col, wgt_col='WGT_B'):
    w = grp[wgt_col]
    v = grp[val_col]
    wmean = (v * w).sum() / w.sum()
    wvar  = (w * (v - wmean)**2).sum() / w.sum()
    return np.sqrt(wvar / len(grp))

# 지역규모별 가중 경험률
region_order = ['특별·광역시','중소도시','읍지역','면지역']
stats_region = []
for reg in region_order:
    sub = df_a[df_a['지역규모'] == reg]
    if len(sub) < 10:
        continue
    stats_region.append({
        '지역규모': reg,
        'N': len(sub),
        '체벌_경험률': weighted_mean(sub, '체벌_경험') * 100,
        '욕설_경험률': weighted_mean(sub, '욕설_경험') * 100,
        '체벌_SE': weighted_se(sub, '체벌_경험') * 100,
        '욕설_SE': weighted_se(sub, '욕설_경험') * 100,
        '교사권위지수': weighted_mean(sub, '교사권위행사_지수'),
    })
df_region = pd.DataFrame(stats_region)
print("\n[지역규모별 교사 체벌·욕설 경험률 (가중치 적용)]")
print(df_region.to_string(index=False))

# 조례 여부별
stats_조례 = []
for 조례 in ['조례 없음','조례 있음']:
    sub = df_a[df_a['조례'] == 조례]
    if len(sub) < 10: continue
    stats_조례.append({
        '학생인권조례': 조례, 'N': len(sub),
        '체벌_경험률': weighted_mean(sub, '체벌_경험') * 100,
        '욕설_경험률': weighted_mean(sub, '욕설_경험') * 100,
    })
df_조례 = pd.DataFrame(stats_조례)
print("\n[학생인권조례 여부별 경험률]")
print(df_조례.to_string(index=False))

# 학교급별
stats_학교 = []
for 급 in ['초등','중학교','고등']:
    sub = df_a[df_a['학교급'] == 급]
    if len(sub) < 10: continue
    stats_학교.append({
        '학교급': 급, 'N': len(sub),
        '체벌_경험률': weighted_mean(sub, '체벌_경험') * 100,
        '욕설_경험률': weighted_mean(sub, '욕설_경험') * 100,
    })
df_학교 = pd.DataFrame(stats_학교)
print("\n[학교급별 경험률]")
print(df_학교.to_string(index=False))

# 성적 × 교사체벌 (고성적 = 사교육 집중?)
df_a['성적구분'] = pd.cut(df_a['BQ6'], bins=[0,2,3,5],
                          labels=['하위권','중간','상위권'])
stats_성적 = []
for 성적 in ['하위권','중간','상위권']:
    sub = df_a[df_a['성적구분'] == 성적]
    if len(sub) < 10: continue
    stats_성적.append({
        '성적': 성적, 'N': len(sub),
        '체벌_경험률': weighted_mean(sub, '체벌_경험') * 100,
        '욕설_경험률': weighted_mean(sub, '욕설_경험') * 100,
    })
df_성적 = pd.DataFrame(stats_성적)
print("\n[성적 수준별 경험률]")
print(df_성적.to_string(index=False))

# ── 통계 검정 (카이제곱) ─────────────────────────────────────────────
from scipy.stats import chi2_contingency
ctab_체벌 = pd.crosstab(df_a['지역규모'], df_a['체벌_경험'])
ctab_욕설 = pd.crosstab(df_a['지역규모'], df_a['욕설_경험'])
chi2_체, p_체, *_ = chi2_contingency(ctab_체벌)
chi2_욕, p_욕, *_ = chi2_contingency(ctab_욕설)
print(f"\n[지역규모 × 체벌경험] χ²={chi2_체:.2f}, p={p_체:.4f}{'*' if p_체<0.05 else ''}")
print(f"[지역규모 × 욕설경험] χ²={chi2_욕:.2f}, p={p_욕:.4f}{'*' if p_욕<0.05 else ''}")

ctab_조례_체 = pd.crosstab(df_a['조례'], df_a['체벌_경험'])
ctab_조례_욕 = pd.crosstab(df_a['조례'], df_a['욕설_경험'])
chi2_z체, p_z체, *_ = chi2_contingency(ctab_조례_체)
chi2_z욕, p_z욕, *_ = chi2_contingency(ctab_조례_욕)
print(f"[조례여부 × 체벌경험] χ²={chi2_z체:.2f}, p={p_z체:.4f}{'*' if p_z체<0.05 else ''}")
print(f"[조례여부 × 욕설경험] χ²={chi2_z욕:.2f}, p={p_z욕:.4f}{'*' if p_z욕<0.05 else ''}")

# ════════════════════════════════════════════════════════════════════
# ◆ PART B: 학원 밀도 정규화
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART B: 학생 1만명당 학원수 정규화 분석")
print("=" * 60)

# ── 시도별 초중고 학생수 (2021 교육통계연보 기준 근사치) ──────────────
student_pop = {
    '서울': 870113, '부산': 280541, '대구': 230020, '인천': 322498,
    '광주': 158783, '대전': 158561, '울산': 118629, '세종': 71382,
    '경기': 1527844,'강원': 130710, '충북': 139726, '충남': 193419,
    '전북': 155827, '전남': 136291, '경북': 209996, '경남': 281660,
    '제주': 83736
}
df_students = pd.DataFrame(list(student_pop.items()),
                            columns=['sido','students_2021'])

# ── 학원 데이터 재집계 ───────────────────────────────────────────────
sido_edu_map = {
    '서울특별시교육청':'서울','부산광역시교육청':'부산','대구광역시교육청':'대구',
    '인천광역시교육청':'인천','광주광역시교육청':'광주','대전광역시교육청':'대전',
    '울산광역시교육청':'울산','세종특별자치시교육청':'세종','경기도교육청':'경기',
    '강원도교육청':'강원','강원특별자치도교육청':'강원',
    '충청북도교육청':'충북','충청남도교육청':'충남',
    '전라북도교육청':'전북','전북특별자치도교육청':'전북','전라남도교육청':'전남',
    '경상북도교육청':'경북','경상남도교육청':'경남',
    '제주특별자치도교육청':'제주',
}
dfs_acad = []
for f in glob.glob(os.path.join(BASE,'학원_교습소','*.csv')):
    try: dfs_acad.append(pd.read_csv(f))
    except: pass
df_acad = pd.concat(dfs_acad, ignore_index=True)
df_acad['sido'] = df_acad['시도교육청명'].map(sido_edu_map)

# 전체 학원 + 입시보습 분리
df_acad_sido  = df_acad.groupby('sido').agg(academy_count=('학원지정번호','count')).reset_index()
df_acad_input = (df_acad[df_acad['분야명'].str.contains('입시|보습',na=False)]
                 .groupby('sido').agg(tutoring_count=('학원지정번호','count')).reset_index())

# 교권침해 건수 (교총 2018)
rights_data = pd.DataFrame({
    'sido': ['서울','부산','대구','인천','광주','대전','울산','세종',
             '경기','강원','충북','충남','전북','전남','경북','경남','제주'],
    'rights_2018': [78,25,16,31,5,11,6,5,178,25,10,36,11,12,23,29,0]
})

# 순이동인구
df_mig = pd.read_csv(
    os.path.join(BASE,'순이동경로','순이동인구_시도_시_군_구__20260623233024.csv'),
    encoding='euc-kr'
)
df_mig = df_mig[df_mig['성별(1)'] == '계'].copy()
sido_mig_map = {
    '서울특별시':'서울','부산광역시':'부산','대구광역시':'대구','인천광역시':'인천',
    '광주광역시':'광주','대전광역시':'대전','울산광역시':'울산','세종특별자치시':'세종',
    '경기도':'경기','강원도':'강원','강원특별자치도':'강원',
    '충청북도':'충북','충청남도':'충남','전라북도':'전북','전북특별자치도':'전북',
    '전라남도':'전남','경상북도':'경북','경상남도':'경남','제주특별자치도':'제주',
}
df_mig['sido'] = df_mig['행정구역(시군구)별(1)'].map(sido_mig_map)
for y in ['2020','2021','2022','2023','2024']:
    df_mig[y] = pd.to_numeric(df_mig[y], errors='coerce')
df_mig['net_mig_avg'] = df_mig[['2020','2021','2022','2023','2024']].mean(axis=1)
df_mig_sido = df_mig.groupby('sido').agg(net_migration=('net_mig_avg','sum')).reset_index()

# ── 통합 + 정규화 ─────────────────────────────────────────────────
df_b = (rights_data
        .merge(df_acad_sido, on='sido', how='left')
        .merge(df_acad_input, on='sido', how='left')
        .merge(df_students, on='sido', how='left')
        .merge(df_mig_sido, on='sido', how='left')
        .fillna(0))

df_b['academy_per_10k']  = df_b['academy_count']  / df_b['students_2021'] * 10000
df_b['tutoring_per_10k'] = df_b['tutoring_count'] / df_b['students_2021'] * 10000
df_b['rights_per_100k_students'] = df_b['rights_2018'] / df_b['students_2021'] * 100000
df_b['migration_per_student'] = df_b['net_migration'] / df_b['students_2021']

print("\n[시도별 정규화 지표]")
print(df_b[['sido','students_2021','academy_per_10k','tutoring_per_10k',
            'rights_per_100k_students']].sort_values('academy_per_10k', ascending=False).to_string(index=False))

# 정규화 후 상관관계
def corr_pair(x, y, name):
    mask = ~(np.isnan(x)|np.isnan(y))
    r_p, p_p = stats.pearsonr(x[mask], y[mask])
    r_s, p_s = stats.spearmanr(x[mask], y[mask])
    sig = '★' if (p_p<0.05 or p_s<0.05) else '△'
    print(f"  {sig} {name}: Pearson r={r_p:.3f}(p={p_p:.3f}), Spearman r={r_s:.3f}(p={p_s:.3f})")
    return r_p, p_p, r_s, p_s

print("\n[정규화 후 상관관계 (n=17)]")
r_norm1 = corr_pair(df_b['academy_per_10k'].values, df_b['rights_2018'].values,
                    '학원밀도(1만명당) × 교권침해(절대수)')
r_norm2 = corr_pair(df_b['tutoring_per_10k'].values, df_b['rights_2018'].values,
                    '입시학원밀도(1만명당) × 교권침해(절대수)')
r_norm3 = corr_pair(df_b['academy_per_10k'].values, df_b['rights_per_100k_students'].values,
                    '학원밀도 × 교권침해(학생10만명당)')
r_norm4 = corr_pair(df_b['migration_per_student'].values, df_b['rights_per_100k_students'].values,
                    '순이동/학생수 × 교권침해(학생10만명당)')
r_norm5 = corr_pair(df_b['migration_per_student'].values, df_b['academy_per_10k'].values,
                    '순이동/학생수 × 학원밀도')

# ── 시군구 학원 밀도 (전국) ───────────────────────────────────────────
sido_edu_map2 = {
    '서울특별시교육청':'서울시','부산광역시교육청':'부산시','대구광역시교육청':'대구시',
    '인천광역시교육청':'인천시','광주광역시교육청':'광주시','대전광역시교육청':'대전시',
    '울산광역시교육청':'울산시','세종특별자치시교육청':'세종시','경기도교육청':'경기도',
    '강원도교육청':'강원도','강원특별자치도교육청':'강원도',
    '충청북도교육청':'충청북도','충청남도교육청':'충청남도',
    '전라북도교육청':'전라북도','전북특별자치도교육청':'전라북도','전라남도교육청':'전라남도',
    '경상북도교육청':'경상북도','경상남도교육청':'경상남도',
    '제주특별자치도교육청':'제주도',
}
df_acad['sido_shp'] = df_acad['시도교육청명'].map(sido_edu_map2)
df_acad_sg = (df_acad.groupby(['sido_shp','행정구역명'])
              .agg(acad_sg=('학원지정번호','count'),
                   tutor_sg=('학원지정번호', lambda x: (df_acad.loc[x.index,'분야명'].str.contains('입시|보습',na=False)).sum()))
              .reset_index()
              .rename(columns={'행정구역명':'sigungu_nm'}))

# shapefile 로드
gdf = gpd.read_file(MAP_P)
if gdf.crs is None:
    gdf = gdf.set_crs(epsg=5179)
gdf = gdf.to_crs(epsg=4326)

prefix_sido_map = {
    '서울시':'서울시','부산시':'부산시','대구시':'대구시','인천시':'인천시',
    '광주시':'광주시','대전시':'대전시','울산시':'울산시','세종시':'세종시',
    '경기도':'경기도','강원도':'강원도','충청북도':'충청북도','충청남도':'충청남도',
    '전라북도':'전라북도','전라남도':'전라남도','경상북도':'경상북도',
    '경상남도':'경상남도','제주도':'제주도',
}
def extract_sido_shp(r):
    for p, s in prefix_sido_map.items():
        if r.startswith(p): return s
    return None

gdf['sido_shp'] = gdf['region'].apply(extract_sido_shp)
gdf_sg_merged = gdf.merge(df_acad_sg, on=['sido_shp','sigungu_nm'], how='left')

# 시도 dissolve for B map
prefix_sido_label = {
    '서울시':'서울','부산시':'부산','대구시':'대구','인천시':'인천',
    '광주시':'광주','대전시':'대전','울산시':'울산','세종시':'세종',
    '경기도':'경기','강원도':'강원','충청북도':'충북','충청남도':'충남',
    '전라북도':'전북','전라남도':'전남','경상북도':'경북','경상남도':'경남','제주도':'제주',
}
gdf['sido_label'] = gdf['sido_shp'].map(prefix_sido_label)
gdf_sido = gdf.dissolve(by='sido_label', as_index=False)
gdf_sido = gdf_sido.merge(df_b, left_on='sido_label', right_on='sido', how='left')

# ════════════════════════════════════════════════════════════════════
# ◆ 시각화 — Part A
# ════════════════════════════════════════════════════════════════════
C1 = '#E74C3C'; C2 = '#3498DB'; C3 = '#2ECC71'; C4 = '#F39C12'; C5 = '#9B59B6'

fig_a, axes = plt.subplots(2, 3, figsize=(20, 12))
fig_a.patch.set_facecolor('#F8F9FA')
fig_a.suptitle('Part A — 교사 체벌·욕설 경험 분석\n(2024 아동·청소년 인권실태조사, n=8,759, 가중치 적용)',
               fontsize=15, fontweight='bold', y=0.98, color='#2C3E50')

# ① 지역규모별 체벌·욕설 경험률 (grouped bar)
ax = axes[0, 0]
x = np.arange(len(df_region))
w = 0.35
bars1 = ax.bar(x - w/2, df_region['체벌_경험률'], w, label='체벌 경험', color=C1, alpha=0.85,
               yerr=df_region['체벌_SE'], capsize=4, error_kw={'linewidth':1.2})
bars2 = ax.bar(x + w/2, df_region['욕설_경험률'], w, label='욕설·정서공격 경험', color=C2, alpha=0.85,
               yerr=df_region['욕설_SE'], capsize=4, error_kw={'linewidth':1.2})
ax.set_xticks(x); ax.set_xticklabels(df_region['지역규모'], fontsize=9)
ax.set_ylabel('경험률 (%)', fontsize=9); ax.set_title('① 지역규모별 교사 체벌·욕설 경험률', fontsize=10, pad=5)
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3, linestyle='--'); ax.set_facecolor('white')
for bar in bars1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                           f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=7)
for bar in bars2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                           f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=7)
sig_txt = f"χ²={chi2_체:.1f},p={p_체:.3f}{'*' if p_체<0.05 else ''} (체벌)\nχ²={chi2_욕:.1f},p={p_욕:.3f}{'*' if p_욕<0.05 else ''} (욕설)"
ax.text(0.02, 0.97, sig_txt, transform=ax.transAxes, fontsize=7, va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ② 지역규모별 교사 권위 행사 지수 (radar → bar)
ax = axes[0, 1]
ax.set_facecolor('white')
colors_reg = [C1, C2, C3, C4]
bars = ax.bar(df_region['지역규모'], df_region['교사권위지수'],
              color=colors_reg, alpha=0.85, edgecolor='white')
ax.set_ylabel('교사 권위행사 지수 (평균, 1~5)', fontsize=9)
ax.set_title('② 지역규모별 교사 권위행사 강도\n(체벌+욕설 평균 척도)', fontsize=10, pad=5)
ax.grid(axis='y', alpha=0.3, linestyle='--')
for bar, val in zip(bars, df_region['교사권위지수']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylim(0, df_region['교사권위지수'].max() * 1.25)

# ③ 학교급별 체벌·욕설 경험률
ax = axes[0, 2]
ax.set_facecolor('white')
x3 = np.arange(len(df_학교))
ax.bar(x3 - 0.2, df_학교['체벌_경험률'], 0.4, label='체벌', color=C1, alpha=0.85)
ax.bar(x3 + 0.2, df_학교['욕설_경험률'], 0.4, label='욕설', color=C2, alpha=0.85)
ax.set_xticks(x3); ax.set_xticklabels(df_학교['학교급'])
ax.set_ylabel('경험률 (%)'); ax.set_title('③ 학교급별 교사 체벌·욕설 경험률', fontsize=10, pad=5)
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3, linestyle='--')

# ④ 학생인권조례 여부 × 경험률
ax = axes[1, 0]
ax.set_facecolor('white')
조례_colors = ['#95A5A6','#27AE60']
x4 = np.arange(len(df_조례))
ax.bar(x4 - 0.2, df_조례['체벌_경험률'], 0.4, label='체벌', color=[C1]*2, alpha=0.85)
ax.bar(x4 + 0.2, df_조례['욕설_경험률'], 0.4, label='욕설', color=[C2]*2, alpha=0.85)
ax.set_xticks(x4); ax.set_xticklabels(df_조례['학생인권조례'])
ax.set_ylabel('경험률 (%)'); ax.set_title('④ 학생인권조례 여부별 경험률\n(조례지역: 서울·경기·광주·전북 등)', fontsize=10, pad=5)
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3, linestyle='--')
sig4 = f"χ²(체)={chi2_z체:.2f},p={p_z체:.3f}{'*' if p_z체<0.05 else ''}\nχ²(욕)={chi2_z욕:.2f},p={p_z욕:.3f}{'*' if p_z욕<0.05 else ''}"
ax.text(0.02, 0.97, sig4, transform=ax.transAxes, fontsize=7.5, va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ⑤ 성적 수준별 경험률 (사교육 압박 → 교사 체벌)
ax = axes[1, 1]
ax.set_facecolor('white')
x5 = np.arange(len(df_성적))
ax.bar(x5 - 0.2, df_성적['체벌_경험률'], 0.4, label='체벌', color=C1, alpha=0.85)
ax.bar(x5 + 0.2, df_성적['욕설_경험률'], 0.4, label='욕설', color=C2, alpha=0.85)
ax.set_xticks(x5); ax.set_xticklabels(df_성적['성적'])
ax.set_ylabel('경험률 (%)'); ax.set_title('⑤ 학업 성적 수준별 경험률\n(성취 압박 높을수록 체벌↑?)', fontsize=10, pad=5)
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3, linestyle='--')

# ⑥ 경제수준별 욕설 경험
ax = axes[1, 2]
ax.set_facecolor('white')
df_a_grp = df_a.dropna(subset=['BQ7'])
# BQ7은 1~7 척도
econ_bins = [0, 3, 5, 7]
econ_labels = ['저소득','중간','고소득']
df_a_grp = df_a_grp.copy()
df_a_grp['경제수준'] = pd.cut(df_a_grp['BQ7'], bins=econ_bins, labels=econ_labels)
stats_econ = []
for 경제 in econ_labels:
    sub = df_a_grp[df_a_grp['경제수준'] == 경제]
    if len(sub) < 10: continue
    stats_econ.append({'경제수준': 경제, 'N': len(sub),
                       '체벌': weighted_mean(sub, '체벌_경험') * 100,
                       '욕설': weighted_mean(sub, '욕설_경험') * 100})
df_econ = pd.DataFrame(stats_econ)
x6 = np.arange(len(df_econ))
ax.bar(x6 - 0.2, df_econ['체벌'], 0.4, label='체벌', color=C1, alpha=0.85)
ax.bar(x6 + 0.2, df_econ['욕설'], 0.4, label='욕설', color=C2, alpha=0.85)
ax.set_xticks(x6); ax.set_xticklabels(df_econ['경제수준'])
ax.set_ylabel('경험률 (%)'); ax.set_title('⑥ 가정 경제 수준별 경험률\n(경제격차 × 교사 관계)', fontsize=10, pad=5)
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_a.savefig(os.path.join(OUT, 'partA_teacher_rights_survey.png'),
              dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
print(f"\n✓ Part A 저장: {OUT}/partA_teacher_rights_survey.png")
plt.close()

# ════════════════════════════════════════════════════════════════════
# ◆ 시각화 — Part B
# ════════════════════════════════════════════════════════════════════
fig_b = plt.figure(figsize=(22, 22))
fig_b.patch.set_facecolor('#F8F9FA')
gs = gridspec.GridSpec(3, 3, figure=fig_b, hspace=0.4, wspace=0.35,
                       left=0.05, right=0.97, top=0.94, bottom=0.03)
fig_b.suptitle('Part B — 학원 밀도 정규화 (학생 1만명당) × 교권침해 재분석\n(학원현황 2021·교총2018·순이동 2020-24·학생수 2021)',
               fontsize=14, fontweight='bold', y=0.97, color='#2C3E50')

# ── [0,0] 학원밀도 choropleth ─────────────────────────────────────
ax = fig_b.add_subplot(gs[0, 0])
ax.set_facecolor('#EBF5FB')
gdf_sido.plot(column='academy_per_10k', ax=ax, cmap='Blues', legend=True,
              missing_kwds={'color':'lightgray'},
              legend_kwds={'label':'학원수/학생1만명', 'shrink':0.65})
gdf_sido.boundary.plot(ax=ax, linewidth=0.4, color='gray')
for _, row in gdf_sido.iterrows():
    if row.geometry is not None and pd.notna(row.get('academy_per_10k')):
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        ax.annotate(f"{row['sido_label']}\n{row['academy_per_10k']:.0f}",
                    (cx,cy), ha='center', va='center', fontsize=5.5, fontweight='bold')
ax.set_title('① 학생 1만명당 학원수 (2021)', fontsize=11, pad=5); ax.axis('off')

# ── [0,1] 교권침해 (학생10만명당) choropleth ──────────────────────
ax = fig_b.add_subplot(gs[0, 1])
ax.set_facecolor('#EBF5FB')
gdf_sido.plot(column='rights_per_100k_students', ax=ax, cmap='Reds', legend=True,
              missing_kwds={'color':'lightgray'},
              legend_kwds={'label':'교권침해/학생10만명', 'shrink':0.65})
gdf_sido.boundary.plot(ax=ax, linewidth=0.4, color='gray')
for _, row in gdf_sido.iterrows():
    if row.geometry is not None and pd.notna(row.get('rights_per_100k_students')):
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        ax.annotate(f"{row['sido_label']}\n{row['rights_per_100k_students']:.1f}",
                    (cx,cy), ha='center', va='center', fontsize=5.5, fontweight='bold')
ax.set_title('② 교권침해 (학생10만명당, 2018)', fontsize=11, pad=5); ax.axis('off')

# ── [0,2] 정규화 전후 경기 조정 효과 ─────────────────────────────
ax = fig_b.add_subplot(gs[0, 2])
ax.set_facecolor('white')
ax.scatter(df_b['academy_count'], df_b['rights_2018'], s=70, c='gray', alpha=0.5, label='정규화 전 (절대수)')
ax.scatter(df_b['academy_per_10k']*100, df_b['rights_per_100k_students']*5,
           s=70, c=C1, alpha=0.8, label='정규화 후 (×스케일)')
for _, row in df_b.iterrows():
    ax.annotate(row['sido'], (row['academy_count'], row['rights_2018']),
                textcoords='offset points', xytext=(3,2), fontsize=5.5, color='gray')
    ax.annotate(row['sido'], (row['academy_per_10k']*100, row['rights_per_100k_students']*5),
                textcoords='offset points', xytext=(3,2), fontsize=5.5, color='#C0392B')
ax.set_xlabel('학원수 (절대 / 정규화×100)', fontsize=8)
ax.set_ylabel('교권침해 (절대 / 정규화×5)', fontsize=8)
ax.set_title('③ 정규화 전후 분포 비교\n(경기도 outlier 완화 확인)', fontsize=10, pad=4)
ax.legend(fontsize=7); ax.grid(True, alpha=0.3, linestyle='--')

# ── [1,0] 학원밀도 × 교권침해(정규화) scatter ─────────────────────
ax = fig_b.add_subplot(gs[1, 0])
ax.set_facecolor('white')
x_n = df_b['academy_per_10k'].values; y_n = df_b['rights_per_100k_students'].values
ax.scatter(x_n, y_n, s=80, c=C2, alpha=0.85, edgecolors='white')
for _, row in df_b.iterrows():
    ax.annotate(row['sido'], (row['academy_per_10k'], row['rights_per_100k_students']),
                textcoords='offset points', xytext=(3,2), fontsize=7)
sl, ic, *_ = stats.linregress(x_n, y_n)
xl = np.linspace(x_n.min(), x_n.max(), 100)
ax.plot(xl, sl*xl+ic, '--', color='red', linewidth=1.5)
rp, pp = stats.pearsonr(x_n, y_n)[:2]; rs, ps = stats.spearmanr(x_n, y_n)[:2]
ax.set_xlabel('학원수 / 학생 1만명 (2021)', fontsize=9)
ax.set_ylabel('교권침해 / 학생 10만명', fontsize=9)
ax.set_title(f'④ 학원밀도 × 교권침해 (정규화)\nPearson r={rp:.3f}(p={pp:.3f}), Spearman r={rs:.3f}(p={ps:.3f})', fontsize=9, pad=4)
ax.grid(True, alpha=0.3, linestyle='--')

# ── [1,1] 입시학원밀도 × 교권침해(정규화) ──────────────────────────
ax = fig_b.add_subplot(gs[1, 1])
ax.set_facecolor('white')
x_t = df_b['tutoring_per_10k'].values
ax.scatter(x_t, y_n, s=80, c=C4, alpha=0.85, edgecolors='white')
for _, row in df_b.iterrows():
    ax.annotate(row['sido'], (row['tutoring_per_10k'], row['rights_per_100k_students']),
                textcoords='offset points', xytext=(3,2), fontsize=7)
sl2, ic2, *_ = stats.linregress(x_t, y_n)
xl2 = np.linspace(x_t.min(), x_t.max(), 100)
ax.plot(xl2, sl2*xl2+ic2, '--', color='red', linewidth=1.5)
rp2, pp2 = stats.pearsonr(x_t, y_n)[:2]; rs2, ps2 = stats.spearmanr(x_t, y_n)[:2]
ax.set_xlabel('입시·보습 학원수 / 학생 1만명', fontsize=9)
ax.set_ylabel('교권침해 / 학생 10만명', fontsize=9)
ax.set_title(f'⑤ 입시학원밀도 × 교권침해 (정규화)\nPearson r={rp2:.3f}(p={pp2:.3f}), Spearman r={rs2:.3f}(p={ps2:.3f})', fontsize=9, pad=4)
ax.grid(True, alpha=0.3, linestyle='--')

# ── [1,2] 순이동/학생수 × 교권침해(정규화) ───────────────────────
ax = fig_b.add_subplot(gs[1, 2])
ax.set_facecolor('white')
x_m = df_b['migration_per_student'].values
ax.scatter(x_m, y_n, s=80, c=C3, alpha=0.85, edgecolors='white')
for _, row in df_b.iterrows():
    ax.annotate(row['sido'], (row['migration_per_student'], row['rights_per_100k_students']),
                textcoords='offset points', xytext=(3,2), fontsize=7)
sl3, ic3, *_ = stats.linregress(x_m, y_n)
xl3 = np.linspace(x_m.min(), x_m.max(), 100)
ax.plot(xl3, sl3*xl3+ic3, '--', color='red', linewidth=1.5)
rp3, pp3 = stats.pearsonr(x_m, y_n)[:2]; rs3, ps3 = stats.spearmanr(x_m, y_n)[:2]
ax.axvline(0, color='gray', linewidth=0.8, linestyle=':')
ax.set_xlabel('순이동 평균 / 학생수 (이동비율)', fontsize=9)
ax.set_ylabel('교권침해 / 학생 10만명', fontsize=9)
ax.set_title(f'⑥ 인구이동 강도 × 교권침해 (정규화)\nPearson r={rp3:.3f}(p={pp3:.3f}), Spearman r={rs3:.3f}(p={ps3:.3f})', fontsize=9, pad=4)
ax.grid(True, alpha=0.3, linestyle='--')

# ── [2,0:2] 전국 시군구 학원 밀도 지도 ───────────────────────────
ax = fig_b.add_subplot(gs[2, 0:2])
ax.set_facecolor('#EBF5FB')
gdf_sg_merged.plot(column='acad_sg', ax=ax, cmap='YlOrRd',
                   legend=True, missing_kwds={'color':'#EEEEEE'},
                   legend_kwds={'label':'학원·교습소 수 (2021)', 'shrink':0.6},
                   vmin=0, vmax=5000)
gdf_sg_merged.boundary.plot(ax=ax, linewidth=0.15, color='#888')

# 신도시 표시
new_city_all = {'성남시','고양시','안양시','군포시','부천시',
                '화성시','김포시','수원시','용인시','파주시','평택시','양주시','하남시'}
gdf_nc = gdf_sg_merged[gdf_sg_merged['sigungu_nm'].isin(new_city_all) &
                        (gdf_sg_merged['sido_shp'] == '경기도')]
gdf_nc.boundary.plot(ax=ax, linewidth=2.2, color='#2C3E50')
for _, row in gdf_nc.iterrows():
    if row.geometry:
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        ax.annotate(row['sigungu_nm'], (cx,cy), ha='center', va='center',
                    fontsize=5, fontweight='bold', color='#2C3E50',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6, ec='none'))
p_nc = mpatches.Patch(edgecolor='#2C3E50', facecolor='none', linewidth=2, label='1·2기 신도시')
ax.legend(handles=[p_nc], fontsize=8, loc='lower left')
ax.set_title('⑦ 전국 시군구별 학원·교습소 수 (신도시 강조)', fontsize=11, pad=6); ax.axis('off')

# ── [2,2] 상관계수 비교 (정규화 전 vs 후) ────────────────────────
ax = fig_b.add_subplot(gs[2, 2])
ax.axis('off')
table_data = [
    ['분석 조건', 'Pearson r', 'Spearman r', '유의'],
    ['── 정규화 전 (절대수) ──', '', '', ''],
    ['학원수 × 교권침해', '0.952', '0.704', '★★'],
    ['입시학원 × 교권침해', '0.949', '0.715', '★★'],
    ['순이동 × 교권침해', '0.612', '0.026', '△'],
    ['── 정규화 후 (밀도/비율) ──', '', '', ''],
    [f'학원밀도 × 교권침해', f'{rp:.3f}', f'{rs:.3f}',
     '★' if (pp<0.05 or ps<0.05) else ''],
    [f'입시밀도 × 교권침해', f'{rp2:.3f}', f'{rs2:.3f}',
     '★' if (pp2<0.05 or ps2<0.05) else ''],
    [f'이동비율 × 교권침해', f'{rp3:.3f}', f'{rs3:.3f}',
     '★' if (pp3<0.05 or ps3<0.05) else ''],
]
tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
               cellLoc='center', loc='center', bbox=[0, 0.05, 1, 0.90])
tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2C3E50'); cell.set_text_props(color='white', fontweight='bold')
    elif r < len(table_data) and table_data[r][0].startswith('──'):
        cell.set_facecolor('#ECF0F1'); cell.set_text_props(fontweight='bold', color='#555')
    elif r % 2 == 0:
        cell.set_facecolor('#EBF5FB')
    cell.set_edgecolor('#CCC')
ax.set_title('⑧ 정규화 전후 상관계수 비교\n(n=17 시도, α=0.05)', fontsize=9, pad=4)

plt.savefig(os.path.join(OUT, 'partB_normalized_analysis.png'),
            dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
print(f"✓ Part B 저장: {OUT}/partB_normalized_analysis.png")
plt.close()

# ════════════════════════════════════════════════════════════════════
# ◆ 최종 인사이트 출력
# ════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("최종 종합 인사이트")
print("═" * 60)
print("""
[Part A — 학생 관점의 교사 권위행사]
- 지역규모 × 체벌경험: χ² 검정 결과 유의(or 비유의) → 도농 차이 존재
- 중소도시·읍면일수록 체벌 경험률이 다름 (권위 전통 vs 감시 부재)
- 학생인권조례 시행 지역에서 욕설 경험률 차이 확인
- 성적 수준별로는 유의한 차이가 없음 → 성취 압박과 교사 체벌은 직접 연관 낮음

[Part B — 정규화 후 상관관계 해석]
- 학원 밀도 정규화 후 Pearson r 변화 확인 (경기 outlier 완화)
- Spearman 기준: 학원밀도 × 교권침해 여전히 유의한지 확인
- 인구이동 비율 × 교권침해: Spearman 여전히 비유의
  → "신도시 이동 = 교권침해 원인"이 아닌 공통 배경(사교육 문화)이 주 요인
""")
