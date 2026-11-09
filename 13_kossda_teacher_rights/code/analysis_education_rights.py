"""
교권침해 × 신도시/인구이동 × 사교육 분석
Data sources:
  - 교총 교권상담실적.xlsx (교권침해 건수, 2018, 시도별)
  - 학원_교습소/*.csv (학원·교습소 현황, 2021)
  - 순이동경로/순이동인구_*.csv (순이동인구, 2016-2025, 시군구별)
  - /Downloads/map/final.shp (시군구 경계 shapefile)
"""

import os, glob, warnings
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import matplotlib.gridspec as gridspec
from scipy import stats
import seaborn as sns

warnings.filterwarnings('ignore')

# ── 한글 폰트 설정 ────────────────────────────────────────────────────────────
def set_korean_font():
    candidates = [
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',
        '/Library/Fonts/AppleGothic.ttf',
        '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
    ]
    for p in candidates:
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            prop = font_manager.FontProperties(fname=p)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return prop.get_name()
    plt.rcParams['font.family'] = 'DejaVu Sans'
    return 'DejaVu Sans'

FONT_NAME = set_korean_font()
print(f"Font: {FONT_NAME}")

BASE = '/Users/baiohelseu/Desktop/Project/kossda/data'
MAP_PATH = '/Users/baiohelseu/Downloads/map/final.shp'
OUT_DIR = '/Users/baiohelseu/Desktop/Project/kossda/output'
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. 교권침해 데이터 (교총 2018, 시도별) ─ 직접 입력 (merged-cell 누락 보완)
# ══════════════════════════════════════════════════════════════════════════════
sido_order = ['서울','부산','대구','인천','광주','대전','울산','세종',
              '경기','강원','충북','충남','전북','전남','경북','경남','제주']

# 교총 2018 보고서 <표 3>: 시도별 교권침해 상담사례 건수(계)
rights_cases_2018 = [78,25,16,31,5,11,6,5,178,25,10,36,11,12,23,29,0]

df_rights = pd.DataFrame({
    'sido': sido_order,
    'rights_2018': rights_cases_2018
})

# ══════════════════════════════════════════════════════════════════════════════
# 2. 학원·교습소 데이터 (시도별 집계)
# ══════════════════════════════════════════════════════════════════════════════
sido_edu_map = {
    '서울특별시교육청': '서울', '부산광역시교육청': '부산', '대구광역시교육청': '대구',
    '인천광역시교육청': '인천', '광주광역시교육청': '광주', '대전광역시교육청': '대전',
    '울산광역시교육청': '울산', '세종특별자치시교육청': '세종', '경기도교육청': '경기',
    '강원도교육청': '강원', '강원특별자치도교육청': '강원',
    '충청북도교육청': '충북', '충청남도교육청': '충남',
    '전라북도교육청': '전북', '전북특별자치도교육청': '전북', '전라남도교육청': '전남',
    '경상북도교육청': '경북', '경상남도교육청': '경남',
    '제주특별자치도교육청': '제주',
}

academy_files = glob.glob(os.path.join(BASE, '학원_교습소', '*.csv'))
dfs_acad = []
for f in academy_files:
    try:
        d = pd.read_csv(f)
        dfs_acad.append(d)
    except Exception:
        pass
df_acad = pd.concat(dfs_acad, ignore_index=True)
df_acad['sido'] = df_acad['시도교육청명'].map(sido_edu_map)
df_acad_sido = (df_acad.groupby('sido')
                .agg(academy_count=('학원지정번호','count'),
                     total_capacity=('정원합계','sum'))
                .reset_index())

# 입시·보습 계열만 따로 (주요 사교육 지표)
df_acad_input = df_acad[df_acad['분야명'].str.contains('입시|보습', na=False)]
df_acad_input_sido = (df_acad_input.groupby('sido')
                      .agg(tutoring_count=('학원지정번호','count'))
                      .reset_index())

print("학원 시도별 집계:\n", df_acad_sido.sort_values('academy_count', ascending=False).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 3. 순이동인구 데이터 (시도별 집계, 2020-2024 평균)
# ══════════════════════════════════════════════════════════════════════════════
df_mig = pd.read_csv(
    os.path.join(BASE, '순이동경로', '순이동인구_시도_시_군_구__20260623233024.csv'),
    encoding='euc-kr'
)
# 성별 필터 (계만 사용)
df_mig = df_mig[df_mig['성별(1)'] == '계'].copy()
# 시도명 정규화
sido_mig_map = {
    '서울특별시':'서울','부산광역시':'부산','대구광역시':'대구','인천광역시':'인천',
    '광주광역시':'광주','대전광역시':'대전','울산광역시':'울산','세종특별자치시':'세종',
    '경기도':'경기','강원도':'강원','강원특별자치도':'강원',
    '충청북도':'충북','충청남도':'충남',
    '전라북도':'전북','전북특별자치도':'전북','전라남도':'전남',
    '경상북도':'경북','경상남도':'경남','제주특별자치도':'제주',
}
df_mig['sido'] = df_mig['행정구역(시군구)별(1)'].map(sido_mig_map)

years = ['2020','2021','2022','2023','2024']
for y in years:
    df_mig[y] = pd.to_numeric(df_mig[y], errors='coerce')
df_mig['net_mig_avg'] = df_mig[years].mean(axis=1)

# 시도별 순이동 합산 (시군구 합산)
df_mig_sido = (df_mig.groupby('sido')
               .agg(net_migration=('net_mig_avg','sum'))
               .reset_index())

# 시군구별 순이동 (신도시 분석용)
df_mig['sigungu'] = df_mig['행정구역(시군구)별(2)']
df_mig_sigungu = df_mig[['sido','sigungu','net_mig_avg']].copy()

print("\n순이동 시도별 합산:\n", df_mig_sido.sort_values('net_migration', ascending=False).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 4. 통합 데이터프레임 (시도 단위)
# ══════════════════════════════════════════════════════════════════════════════
df_merged = (df_rights
             .merge(df_acad_sido, on='sido', how='left')
             .merge(df_acad_input_sido, on='sido', how='left')
             .merge(df_mig_sido, on='sido', how='left'))
df_merged = df_merged.fillna(0)
print("\n통합 데이터:\n", df_merged.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 5. 상관 분석
# ══════════════════════════════════════════════════════════════════════════════
def corr_report(x, y, xlabel, ylabel):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    r_p, p_p = stats.pearsonr(x, y)
    r_s, p_s = stats.spearmanr(x, y)
    print(f"\n{xlabel} × {ylabel}")
    print(f"  Pearson  r={r_p:.3f}, p={p_p:.4f}{'*' if p_p<0.05 else ''}")
    print(f"  Spearman r={r_s:.3f}, p={p_s:.4f}{'*' if p_s<0.05 else ''}")
    return r_p, p_p, r_s, p_s

print("\n" + "="*60)
print("상관 분석 결과")
print("="*60)

r1 = corr_report(df_merged['academy_count'].values,
                 df_merged['rights_2018'].values,
                 '학원수', '교권침해건수')
r2 = corr_report(df_merged['tutoring_count'].values,
                 df_merged['rights_2018'].values,
                 '입시·보습 학원수', '교권침해건수')
r3 = corr_report(df_merged['net_migration'].values,
                 df_merged['rights_2018'].values,
                 '순이동인구', '교권침해건수')
r4 = corr_report(df_merged['net_migration'].values,
                 df_merged['academy_count'].values,
                 '순이동인구', '학원수')

# ══════════════════════════════════════════════════════════════════════════════
# 6. 신도시 지역 정의
# ══════════════════════════════════════════════════════════════════════════════
new_city_1st = {
    '성남시':'분당 (1기)', '고양시':'일산 (1기)',
    '안양시':'평촌 (1기)', '군포시':'산본 (1기)', '부천시':'중동 (1기)',
}
new_city_2nd = {
    '화성시':'동탄 (2기)', '김포시':'김포한강 (2기)',
    '수원시':'광교 (2기)', '용인시':'광교 (2기)',
    '파주시':'파주운정 (2기)', '평택시':'고덕 (2기)',
    '양주시':'양주 (2기)', '하남시':'위례 (2기)',
}
new_city_incheon = {'서구': '검단 (2기)'}

# 시군구별 Migration 데이터에 신도시 태그
df_gyeonggi_mig = df_mig_sigungu[df_mig_sigungu['sido'] == '경기'].copy()
df_gyeonggi_mig['신도시'] = df_gyeonggi_mig['sigungu'].map({**new_city_1st, **new_city_2nd})
df_gyeonggi_mig['세대'] = df_gyeonggi_mig['sigungu'].map(
    {k:'1기' for k in new_city_1st} | {k:'2기' for k in new_city_2nd}
)

# ══════════════════════════════════════════════════════════════════════════════
# 7. Shapefile 로드 및 데이터 병합
# ══════════════════════════════════════════════════════════════════════════════
gdf = gpd.read_file(MAP_PATH)
if gdf.crs is None:
    gdf = gdf.set_crs(epsg=5179)  # Korean TM projection
gdf = gdf.to_crs(epsg=4326)

# 시도 추출 (region 컬럼: '서울시종로구', '경기도성남시' 등)
prefix_sido_map = {
    '서울시':'서울','부산시':'부산','대구시':'대구','인천시':'인천',
    '광주시':'광주','대전시':'대전','울산시':'울산','세종시':'세종',
    '경기도':'경기','강원도':'강원','충청북도':'충북','충청남도':'충남',
    '전라북도':'전북','전라남도':'전남','경상북도':'경북','경상남도':'경남',
    '제주도':'제주',
}
def extract_sido(region_str):
    for prefix, sido in prefix_sido_map.items():
        if region_str.startswith(prefix):
            return sido
    return None

gdf['sido'] = gdf['region'].apply(extract_sido)

# 시도 단위 dissolve
gdf_sido = gdf.dissolve(by='sido', as_index=False)
gdf_sido = gdf_sido.merge(df_merged, on='sido', how='left')

# 시군구 단위 학원 집계 (경기도)
df_acad_gyeonggi = df_acad[df_acad['sido'] == '경기'].copy()
df_acad_gyeonggi_sg = (df_acad_gyeonggi.groupby('행정구역명')
                       .agg(academy_count_sg=('학원지정번호','count'))
                       .reset_index()
                       .rename(columns={'행정구역명':'sigungu_nm'}))

gdf_gyeonggi = gdf[gdf['sido'] == '경기'].copy()
gdf_gyeonggi = gdf_gyeonggi.merge(df_acad_gyeonggi_sg, on='sigungu_nm', how='left')
gdf_gyeonggi = gdf_gyeonggi.merge(
    df_gyeonggi_mig.rename(columns={'sigungu':'sigungu_nm'})[['sigungu_nm','net_mig_avg','신도시','세대']],
    on='sigungu_nm', how='left'
)

# ══════════════════════════════════════════════════════════════════════════════
# 8. 시각화
# ══════════════════════════════════════════════════════════════════════════════

COLOR_RIGHTS = '#E74C3C'
COLOR_ACAD   = '#3498DB'
COLOR_MIG    = '#2ECC71'
COLOR_1ST    = '#F39C12'
COLOR_2ND    = '#9B59B6'

fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor('#F8F9FA')

gs = gridspec.GridSpec(4, 3, figure=fig,
                       hspace=0.45, wspace=0.35,
                       left=0.05, right=0.97, top=0.94, bottom=0.03)

# ── 제목 ──────────────────────────────────────────────────────────────────────
fig.suptitle('교권침해 × 신도시/인구이동 × 사교육 분석\n(교총 2018·학원현황 2021·순이동 2020-2024)',
             fontsize=17, fontweight='bold', y=0.97, color='#2C3E50')

# ── [0,0] 전국 교권침해 choropleth ────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#EBF5FB')
gdf_sido.plot(column='rights_2018', ax=ax1, cmap='Reds', legend=True,
              missing_kwds={'color':'lightgray'},
              legend_kwds={'label':'교권침해 건수 (2018)', 'shrink':0.6})
gdf_sido.boundary.plot(ax=ax1, linewidth=0.4, color='gray')
ax1.set_title('① 교권침해 건수 (2018, 시도별)', fontsize=11, pad=6)
ax1.axis('off')
for _, row in gdf_sido.iterrows():
    if row.geometry is not None and row['rights_2018'] > 0:
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        ax1.annotate(f"{row['sido']}\n{int(row['rights_2018'])}",
                     (cx, cy), ha='center', va='center', fontsize=5.5,
                     color='black', fontweight='bold')

# ── [0,1] 학원수 choropleth ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('#EBF5FB')
gdf_sido.plot(column='academy_count', ax=ax2, cmap='Blues', legend=True,
              missing_kwds={'color':'lightgray'},
              legend_kwds={'label':'학원·교습소 수 (2021)', 'shrink':0.6})
gdf_sido.boundary.plot(ax=ax2, linewidth=0.4, color='gray')
ax2.set_title('② 학원·교습소 수 (2021, 시도별)', fontsize=11, pad=6)
ax2.axis('off')
for _, row in gdf_sido.iterrows():
    if row.geometry is not None and row['academy_count'] > 0:
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        ax2.annotate(f"{row['sido']}\n{int(row['academy_count'])}",
                     (cx, cy), ha='center', va='center', fontsize=5.5, color='black')

# ── [0,2] 순이동인구 choropleth ───────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor('#EBF5FB')
vmax = gdf_sido['net_migration'].abs().max()
gdf_sido.plot(column='net_migration', ax=ax3, cmap='RdYlGn',
              vmin=-vmax, vmax=vmax, legend=True,
              missing_kwds={'color':'lightgray'},
              legend_kwds={'label':'순이동 평균 (명/년, 2020-24)', 'shrink':0.6})
gdf_sido.boundary.plot(ax=ax3, linewidth=0.4, color='gray')
ax3.set_title('③ 순이동인구 평균 (2020-2024, 시도별)', fontsize=11, pad=6)
ax3.axis('off')
for _, row in gdf_sido.iterrows():
    if row.geometry is not None:
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        val = row['net_migration']
        ax3.annotate(f"{row['sido']}\n{int(val):+,}",
                     (cx, cy), ha='center', va='center', fontsize=5, color='#1a1a1a')

# ── [1,0] 교권침해 × 학원수 scatter ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor('white')
x = df_merged['academy_count'].values
y = df_merged['rights_2018'].values
ax4.scatter(x, y, s=80, c=COLOR_ACAD, alpha=0.8, edgecolors='white', linewidth=0.5)
for _, row in df_merged.iterrows():
    ax4.annotate(row['sido'], (row['academy_count'], row['rights_2018']),
                 textcoords='offset points', xytext=(3,3), fontsize=7, color='#444')
slope, intercept, *_ = stats.linregress(x, y)
xline = np.linspace(x.min(), x.max(), 100)
ax4.plot(xline, slope*xline+intercept, '--', color='red', linewidth=1.5)
r_p, p_p = stats.pearsonr(x, y)[:2]
r_s, p_s = stats.spearmanr(x, y)[:2]
ax4.set_xlabel('학원·교습소 수 (2021)', fontsize=9)
ax4.set_ylabel('교권침해 건수 (2018)', fontsize=9)
ax4.set_title(f'④ 교권침해 × 학원수\nPearson r={r_p:.3f} (p={p_p:.3f}), Spearman r={r_s:.3f}',
              fontsize=9, pad=4)
ax4.grid(True, alpha=0.3, linestyle='--')

# ── [1,1] 교권침해 × 입시보습 학원수 scatter ─────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor('white')
x5 = df_merged['tutoring_count'].values
y5 = df_merged['rights_2018'].values
ax5.scatter(x5, y5, s=80, c='#E67E22', alpha=0.8, edgecolors='white', linewidth=0.5)
for _, row in df_merged.iterrows():
    ax5.annotate(row['sido'], (row['tutoring_count'], row['rights_2018']),
                 textcoords='offset points', xytext=(3,3), fontsize=7, color='#444')
slope5, int5, *_ = stats.linregress(x5, y5)
xline5 = np.linspace(x5.min(), x5.max(), 100)
ax5.plot(xline5, slope5*xline5+int5, '--', color='red', linewidth=1.5)
r_p5, p_p5 = stats.pearsonr(x5, y5)[:2]
r_s5, p_s5 = stats.spearmanr(x5, y5)[:2]
ax5.set_xlabel('입시·보습 학원수 (2021)', fontsize=9)
ax5.set_ylabel('교권침해 건수 (2018)', fontsize=9)
ax5.set_title(f'⑤ 교권침해 × 입시·보습 학원수\nPearson r={r_p5:.3f} (p={p_p5:.3f})', fontsize=9, pad=4)
ax5.grid(True, alpha=0.3, linestyle='--')

# ── [1,2] 교권침해 × 순이동인구 scatter ──────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor('white')
x6 = df_merged['net_migration'].values
y6 = df_merged['rights_2018'].values
ax6.scatter(x6, y6, s=80, c=COLOR_MIG, alpha=0.8, edgecolors='white', linewidth=0.5)
for _, row in df_merged.iterrows():
    ax6.annotate(row['sido'], (row['net_migration'], row['rights_2018']),
                 textcoords='offset points', xytext=(3,3), fontsize=7, color='#444')
slope6, int6, *_ = stats.linregress(x6, y6)
xline6 = np.linspace(x6.min(), x6.max(), 100)
ax6.plot(xline6, slope6*xline6+int6, '--', color='red', linewidth=1.5)
r_p6, p_p6 = stats.pearsonr(x6, y6)[:2]
r_s6, p_s6 = stats.spearmanr(x6, y6)[:2]
ax6.set_xlabel('순이동 평균 (명/년, 2020-2024)', fontsize=9)
ax6.set_ylabel('교권침해 건수 (2018)', fontsize=9)
ax6.set_title(f'⑥ 교권침해 × 순이동인구\nPearson r={r_p6:.3f} (p={p_p6:.3f})', fontsize=9, pad=4)
ax6.grid(True, alpha=0.3, linestyle='--')
ax6.axvline(0, color='gray', linewidth=0.8, linestyle=':')

# ── [2,0:2] 경기도 신도시 순이동 지도 ────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0:2])
ax7.set_facecolor('#EBF5FB')
# Base plot
gdf_gyeonggi.plot(column='net_mig_avg', ax=ax7, cmap='RdYlGn',
                  vmin=-20000, vmax=20000, legend=True,
                  missing_kwds={'color':'lightgray'},
                  legend_kwds={'label':'순이동 평균 (명/년)', 'shrink':0.7})
gdf_gyeonggi.boundary.plot(ax=ax7, linewidth=0.5, color='#555')

# 신도시 경계 강조
gdf_1st = gdf_gyeonggi[gdf_gyeonggi['sigungu_nm'].isin(new_city_1st.keys())]
gdf_2nd = gdf_gyeonggi[gdf_gyeonggi['sigungu_nm'].isin(new_city_2nd.keys())]
gdf_1st.boundary.plot(ax=ax7, linewidth=2.5, color=COLOR_1ST)
gdf_2nd.boundary.plot(ax=ax7, linewidth=2.5, color=COLOR_2ND)

# 라벨
for _, row in gdf_gyeonggi.iterrows():
    if row.geometry is not None:
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        nm = row.get('신도시')
        if pd.notna(nm):
            ax7.annotate(nm, (cx, cy), ha='center', va='center',
                         fontsize=6.5, fontweight='bold', color='black',
                         bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7, ec='none'))
        else:
            ax7.annotate(row['sigungu_nm'], (cx, cy), ha='center', va='center',
                         fontsize=5, color='#333')

p1 = mpatches.Patch(edgecolor=COLOR_1ST, facecolor='none', linewidth=2, label='1기 신도시')
p2 = mpatches.Patch(edgecolor=COLOR_2ND, facecolor='none', linewidth=2, label='2기 신도시')
ax7.legend(handles=[p1, p2], fontsize=8, loc='lower left')
ax7.set_title('⑦ 경기도 시군구별 순이동인구 & 신도시 위치', fontsize=11, pad=6)
ax7.axis('off')

# ── [2,2] 경기도 신도시 학원 수 bar ──────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ax8.set_facecolor('white')
city_mig = df_gyeonggi_mig.dropna(subset=['신도시']).copy()
city_mig = city_mig.sort_values('net_mig_avg', ascending=False)
colors_bar = [COLOR_1ST if s == '1기' else COLOR_2ND for s in city_mig['세대']]
bars = ax8.barh(city_mig['신도시'], city_mig['net_mig_avg'], color=colors_bar, alpha=0.85)
ax8.axvline(0, color='gray', linewidth=0.8)
ax8.set_xlabel('순이동 평균 (명/년, 2020-2024)', fontsize=8)
ax8.set_title('⑧ 신도시별 순이동인구 비교', fontsize=10, pad=4)
ax8.grid(axis='x', alpha=0.3, linestyle='--')
p1b = mpatches.Patch(color=COLOR_1ST, label='1기 신도시')
p2b = mpatches.Patch(color=COLOR_2ND, label='2기 신도시')
ax8.legend(handles=[p1b, p2b], fontsize=7)
ax8.tick_params(labelsize=7)

# ── [3,0:2] 신도시 학원 현황 (경기도 시군구별) ───────────────────────────────
ax9 = fig.add_subplot(gs[3, 0:2])
ax9.set_facecolor('#EBF5FB')
gdf_gyeonggi_acad = gdf_gyeonggi.copy()
gdf_gyeonggi_acad.plot(column='academy_count_sg', ax=ax9, cmap='Blues',
                       legend=True, missing_kwds={'color':'lightgray'},
                       legend_kwds={'label':'학원·교습소 수 (2021)', 'shrink':0.7})
gdf_gyeonggi.boundary.plot(ax=ax9, linewidth=0.5, color='#555')
gdf_1st.boundary.plot(ax=ax9, linewidth=2.5, color=COLOR_1ST)
gdf_2nd.boundary.plot(ax=ax9, linewidth=2.5, color=COLOR_2ND)
for _, row in gdf_gyeonggi_acad.iterrows():
    if row.geometry is not None:
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        ax9.annotate(row['sigungu_nm'], (cx, cy), ha='center', va='center',
                     fontsize=5.5, color='#222')
ax9.legend(handles=[p1, p2], fontsize=8, loc='lower left')
ax9.set_title('⑨ 경기도 시군구별 학원·교습소 수 & 신도시', fontsize=11, pad=6)
ax9.axis('off')

# ── [3,2] 상관계수 요약 테이블 ───────────────────────────────────────────────
ax10 = fig.add_subplot(gs[3, 2])
ax10.axis('off')
table_data = [
    ['비교 쌍', 'Pearson r', 'p값', 'Spearman r', '유의'],
    ['교권침해 × 학원수', f'{r_p:.3f}', f'{p_p:.3f}', f'{r_s:.3f}', '✓' if p_p<0.05 or p_s<0.05 else ''],
    ['교권침해 × 입시학원', f'{r_p5:.3f}', f'{p_p5:.3f}', f'{r_s5:.3f}', '✓' if p_p5<0.05 or p_s5<0.05 else ''],
    ['교권침해 × 순이동', f'{r_p6:.3f}', f'{p_p6:.3f}', f'{r_s6:.3f}', '✓' if p_p6<0.05 or p_s6<0.05 else ''],
    ['학원수 × 순이동', f'{r4[0]:.3f}', f'{r4[1]:.3f}', f'{r4[2]:.3f}', '✓' if r4[1]<0.05 or r4[3]<0.05 else ''],
]
tbl = ax10.table(cellText=table_data[1:], colLabels=table_data[0],
                 cellLoc='center', loc='center',
                 bbox=[0, 0.1, 1, 0.85])
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#EBF5FB')
    cell.set_edgecolor('#CCC')
ax10.set_title('⑩ 상관계수 요약\n(α=0.05, n=17 시도)', fontsize=9, pad=4)

# ── 저장 ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'education_rights_analysis.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\n✓ 저장 완료: {out_path}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 9. 회귀 요약 출력
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("최종 인사이트 요약")
print("="*60)
results = [
    ("교권침해 × 학원수",        r_p, p_p, r_s, p_s),
    ("교권침해 × 입시·보습학원", r_p5, p_p5, r_s5, p_s5),
    ("교권침해 × 순이동인구",    r_p6, p_p6, r_s6, p_s6),
    ("학원수 × 순이동인구",      r4[0], r4[1], r4[2], r4[3]),
]
for name, rp, pp, rs, ps in results:
    sig = "★ 유의(p<0.05)" if (pp < 0.05 or ps < 0.05) else "비유의"
    print(f"  {name}: Pearson r={rp:.3f} / Spearman r={rs:.3f} → {sig}")
