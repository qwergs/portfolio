"""
신도시 변수 기반 교권침해 고도화 분석
- 신도시 여부 / 세대 / 개수 변수 생성
- 순이동인구 × 신도시 × 학원밀집도 × 교권침해 통합 분석
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
font_paths = [
    '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
    '/Library/Fonts/NanumGothic.ttf',
]
for fp in font_paths:
    try:
        fm.fontManager.addfont(fp)
        plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────────────────
# 1. 신도시 데이터 정의
# ─────────────────────────────────────────────────────────
newtown_raw = [
    # (세대, 신도시명, 시도, 시군구_원본, 매핑_시도, 매핑_시군구)
    ('1기', '분당신도시',     '경기', '성남시 분당구',   '경기도', '성남시'),
    ('1기', '일산신도시',     '경기', '고양시 일산동구', '경기도', '고양시'),
    ('1기', '일산신도시',     '경기', '고양시 일산서구', '경기도', '고양시'),
    ('1기', '평촌신도시',     '경기', '안양시 동안구',   '경기도', '안양시'),
    ('1기', '산본신도시',     '경기', '군포시',           '경기도', '군포시'),
    ('1기', '중동신도시',     '경기', '부천시',           '경기도', '부천시'),

    ('2기', '판교신도시',     '경기', '성남시 분당구',   '경기도', '성남시'),
    ('2기', '동탄1신도시',    '경기', '화성시',           '경기도', '화성시'),
    ('2기', '동탄2신도시',    '경기', '화성시',           '경기도', '화성시'),
    ('2기', '광교신도시',     '경기', '수원시 영통구',   '경기도', '수원시'),
    ('2기', '광교신도시',     '경기', '용인시 수지구',   '경기도', '용인시'),
    ('2기', '위례신도시',     '서울', '송파구',           '서울특별시', '송파구'),
    ('2기', '위례신도시',     '경기', '성남시 수정구',   '경기도', '성남시'),
    ('2기', '위례신도시',     '경기', '하남시',           '경기도', '하남시'),
    ('2기', '운정신도시',     '경기', '파주시',           '경기도', '파주시'),
    ('2기', '검단신도시',     '인천', '서구',             '인천광역시', '서구'),
    ('2기', '한강신도시',     '경기', '김포시',           '경기도', '김포시'),
    ('2기', '양주신도시',     '경기', '양주시',           '경기도', '양주시'),
    ('2기', '세교신도시',     '경기', '오산시',           '경기도', '오산시'),
    ('2기', '고덕국제신도시', '경기', '평택시',           '경기도', '평택시'),
    ('2기', '탕정신도시',     '충남', '아산시',           '충청남도', '아산시'),
    ('2기', '배방신도시',     '충남', '아산시',           '충청남도', '아산시'),
    ('2기', '세종신도시',     '세종', '세종시',           '세종특별자치시', '세종특별자치시'),

    ('3기', '남양주왕숙',     '경기', '남양주시',         '경기도', '남양주시'),
    ('3기', '왕숙2지구',      '경기', '남양주시',         '경기도', '남양주시'),
    ('3기', '하남교산',       '경기', '하남시',           '경기도', '하남시'),
    ('3기', '고양창릉',       '경기', '고양시 덕양구',   '경기도', '고양시'),
    ('3기', '부천대장',       '경기', '부천시',           '경기도', '부천시'),
    ('3기', '인천계양',       '인천', '계양구',           '인천광역시', '계양구'),
    ('3기', '과천과천',       '경기', '과천시',           '경기도', '과천시'),
    ('3기', '안산장상',       '경기', '안산시',           '경기도', '안산시'),
    ('3기', '용인플랫폼시티', '경기', '용인시',           '경기도', '용인시'),
]

nt_df = pd.DataFrame(newtown_raw, columns=['세대','신도시명','시도','원본시군구','매핑시도','매핑시군구'])

# 시군구 단위 신도시 집계
nt_sigungu = (nt_df.groupby(['매핑시도','매핑시군구'])
              .agg(
                  신도시_개수=('신도시명','nunique'),
                  신도시명목록=('신도시명', lambda x: '/'.join(sorted(set(x)))),
                  세대목록=('세대', lambda x: '/'.join(sorted(set(x)))),
                  최고세대=('세대', lambda x: sorted(set(x))[-1]),  # 3기 > 2기 > 1기
              ).reset_index())
nt_sigungu['신도시_여부'] = 1
nt_sigungu['1기_있음'] = nt_sigungu['세대목록'].str.contains('1기').astype(int)
nt_sigungu['2기_있음'] = nt_sigungu['세대목록'].str.contains('2기').astype(int)
nt_sigungu['3기_있음'] = nt_sigungu['세대목록'].str.contains('3기').astype(int)
print("신도시 시군구 목록:")
print(nt_sigungu[['매핑시도','매핑시군구','신도시_개수','세대목록','신도시명목록']].to_string(index=False))

# ─────────────────────────────────────────────────────────
# 2. 순이동인구 데이터 로드
# ─────────────────────────────────────────────────────────
mig = pd.read_csv(
    '/Users/baiohelseu/Desktop/Project/kossda/data/순이동경로/순이동인구_시도_시_군_구__20260623233024.csv',
    encoding='euc-kr', header=1
)
mig.columns = ['시도','시군구','성별'] + [str(y) for y in range(2016, 2026)]
mig = mig[mig['성별']=='계'].reset_index(drop=True)
year_cols = [str(y) for y in range(2016, 2026)]
for c in year_cols:
    mig[c] = pd.to_numeric(mig[c].astype(str).str.replace(',',''), errors='coerce')

# 집계: 평균 순이동 (2016-2025), 최근 순이동 (2020-2025)
mig['순이동_평균'] = mig[year_cols].mean(axis=1)
mig['순이동_최근'] = mig[[str(y) for y in range(2020,2026)]].mean(axis=1)
mig['순이동_2020'] = mig['2020']
mig['순이동_2024'] = mig['2024']
mig['시군구_정규'] = mig['시군구'].str.strip()

# ─────────────────────────────────────────────────────────
# 3. 학원 데이터 로드 (시군구별 학원 수)
# ─────────────────────────────────────────────────────────
import glob
files = glob.glob('/Users/baiohelseu/Desktop/Project/kossda/data/학원_교습소/*.csv')
dfs = []
for f in files:
    try:
        dfs.append(pd.read_csv(f, encoding='utf-8', low_memory=False))
    except Exception:
        try:
            dfs.append(pd.read_csv(f, encoding='cp949', low_memory=False))
        except Exception:
            pass
academy = pd.concat(dfs, ignore_index=True)

# 교육청명 → 시도 매핑
edu_to_sido = {
    '서울특별시교육청':'서울특별시', '부산광역시교육청':'부산광역시',
    '대구광역시교육청':'대구광역시', '인천광역시교육청':'인천광역시',
    '광주광역시교육청':'광주광역시', '대전광역시교육청':'대전광역시',
    '울산광역시교육청':'울산광역시', '세종특별자치시교육청':'세종특별자치시',
    '경기도교육청':'경기도', '강원특별자치도교육청':'강원특별자치도',
    '강원도교육청':'강원도', '충청북도교육청':'충청북도', '충청남도교육청':'충청남도',
    '전북특별자치도교육청':'전북특별자치도', '전라북도교육청':'전라북도',
    '전라남도교육청':'전라남도', '경상북도교육청':'경상북도',
    '경상남도교육청':'경상남도', '제주특별자치도교육청':'제주특별자치도',
}
academy['시도_정규'] = academy['시도교육청명'].map(edu_to_sido).fillna(academy['시도교육청명'])
acad_count = (academy.groupby(['시도_정규','행정구역명'])
              .size().reset_index(name='학원수'))
acad_count.rename(columns={'행정구역명':'시군구_정규'}, inplace=True)

# ─────────────────────────────────────────────────────────
# 4. 통합 데이터 구축
# ─────────────────────────────────────────────────────────
base = mig[['시도','시군구_정규','순이동_평균','순이동_최근','순이동_2020','순이동_2024'] + year_cols].copy()
base.rename(columns={'시도':'시도_mig'}, inplace=True)

# 학원 수 병합
merged = base.merge(
    acad_count, left_on=['시도_mig','시군구_정규'], right_on=['시도_정규','시군구_정규'], how='left'
).drop(columns=['시도_정규'], errors='ignore')

# 신도시 변수 병합
merged = merged.merge(
    nt_sigungu[['매핑시도','매핑시군구','신도시_여부','신도시_개수','세대목록',
                '신도시명목록','최고세대','1기_있음','2기_있음','3기_있음']],
    left_on=['시도_mig','시군구_정규'],
    right_on=['매핑시도','매핑시군구'],
    how='left'
).drop(columns=['매핑시도','매핑시군구'], errors='ignore')

merged['신도시_여부'] = merged['신도시_여부'].fillna(0).astype(int)
merged['신도시_개수'] = merged['신도시_개수'].fillna(0).astype(int)
merged['최고세대'] = merged['최고세대'].fillna('없음')
merged['1기_있음'] = merged['1기_있음'].fillna(0).astype(int)
merged['2기_있음'] = merged['2기_있음'].fillna(0).astype(int)
merged['3기_있음'] = merged['3기_있음'].fillna(0).astype(int)
merged['학원수'] = merged['학원수'].fillna(0)

print(f"\n통합 데이터 shape: {merged.shape}")
print(f"신도시 있는 시군구: {merged['신도시_여부'].sum()}개")
print(f"신도시 없는 시군구: {(merged['신도시_여부']==0).sum()}개")

# 신도시 시군구 요약
nt_rows = merged[merged['신도시_여부']==1][
    ['시도_mig','시군구_정규','세대목록','신도시_개수','신도시명목록','순이동_평균','순이동_최근','학원수']
].sort_values('순이동_최근', ascending=False)
print("\n신도시 시군구 순이동인구 랭킹:")
print(nt_rows.to_string(index=False))

# ─────────────────────────────────────────────────────────
# 5. 교총 교권침해 데이터 (시도 단위)
# ─────────────────────────────────────────────────────────
kwon_data = {
    '서울특별시':78, '부산광역시':25, '대구광역시':17, '인천광역시':31,
    '광주광역시':11, '대전광역시':10, '울산광역시':9, '세종특별자치시':1,
    '경기도':178, '강원도':25, '충청북도':22, '충청남도':36,
    '전라북도':16, '전라남도':8, '경상북도':23, '경상남도':29, '제주특별자치도':0,
}
# 학생 수 (만 명, 2021 기준 근사치)
student_pop = {
    '서울특별시':94, '부산광역시':32, '대구광역시':24, '인천광역시':31,
    '광주광역시':16, '대전광역시':15, '울산광역시':11, '세종특별자치시':4,
    '경기도':145, '강원도':14, '충청북도':12, '충청남도':18,
    '전라북도':15, '전라남도':11, '경상북도':19, '경상남도':28, '제주특별자치도':7,
}
# 시도별 신도시 있음 여부 (1기+2기+3기)
sido_newtown = merged[merged['신도시_여부']==1].groupby('시도_mig').agg(
    신도시시군구수=('신도시_여부','sum'),
    신도시순이동평균=('순이동_평균','mean'),
    신도시순이동최근=('순이동_최근','mean'),
    시도내학원수=('학원수','sum'),
).reset_index()

sido_all = merged.groupby('시도_mig').agg(
    전체시군구수=('시군구_정규','count'),
    전체순이동평균=('순이동_평균','mean'),
    전체순이동최근=('순이동_최근','mean'),
    학원수합계=('학원수','sum'),
).reset_index()
sido_all['교권침해건수'] = sido_all['시도_mig'].map(kwon_data)
sido_all['학생수만명'] = sido_all['시도_mig'].map(student_pop)
sido_all['교권침해_정규화'] = sido_all['교권침해건수'] / sido_all['학생수만명']
sido_all = sido_all.merge(sido_newtown[['시도_mig','신도시시군구수','신도시순이동최근']],
                          on='시도_mig', how='left')
sido_all['신도시시군구수'] = sido_all['신도시시군구수'].fillna(0)
sido_all['신도시비율'] = sido_all['신도시시군구수'] / sido_all['전체시군구수']

# ─────────────────────────────────────────────────────────
# 6. 시각화
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 18))
fig.suptitle('신도시·인구이동 × 교권침해 고도화 분석', fontsize=16, fontweight='bold', y=0.98)

colors_gen = {'없음':'#CCCCCC', '1기':'#2196F3', '2기':'#FF9800', '3기':'#F44336'}

# ─── Panel 1: 신도시 vs 비신도시 순이동인구 비교 (박스플롯) ───
ax = axes[0, 0]
nt_yes = merged[merged['신도시_여부']==1]['순이동_최근'].dropna()
nt_no  = merged[merged['신도시_여부']==0]['순이동_최근'].dropna()
bp = ax.boxplot([nt_no, nt_yes], patch_artist=True,
                labels=['비신도시\n시군구', '신도시\n시군구'],
                medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set_facecolor('#90CAF9')
bp['boxes'][1].set_facecolor('#FF8A65')
stat, pval = stats.mannwhitneyu(nt_yes, nt_no, alternative='two-sided')
ax.set_title(f'신도시 여부별 순이동인구 (2020-2025 평균)\nMann-Whitney p={pval:.4f}', fontsize=10)
ax.set_ylabel('순이동인구 (명)')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.text(1.5, nt_yes.max()*0.85, f'신도시 중위: {nt_yes.median():,.0f}명\n비신도시 중위: {nt_no.median():,.0f}명',
        ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ─── Panel 2: 세대별 순이동인구 비교 ───
ax = axes[0, 1]
gen_groups = {}
for gen in ['없음','1기','2기','3기']:
    if gen == '없음':
        vals = merged[merged['신도시_여부']==0]['순이동_최근'].dropna().values
    else:
        vals = merged[merged['최고세대']==gen]['순이동_최근'].dropna().values
    gen_groups[gen] = vals

bp2 = ax.boxplot(list(gen_groups.values()), patch_artist=True, labels=list(gen_groups.keys()),
                 medianprops=dict(color='black', linewidth=2))
for patch, (gen, _) in zip(bp2['boxes'], gen_groups.items()):
    patch.set_facecolor(colors_gen[gen])
ax.set_title('신도시 세대별 순이동인구 (2020-2025 평균)', fontsize=10)
ax.set_ylabel('순이동인구 (명)')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
for i, (gen, vals) in enumerate(gen_groups.items()):
    ax.text(i+1, ax.get_ylim()[1]*0.9, f'n={len(vals)}\n중위\n{np.median(vals):,.0f}',
            ha='center', fontsize=7)

# ─── Panel 3: 신도시 시군구 순이동 시계열 ───
ax = axes[0, 2]
nt_ts = merged[merged['신도시_여부']==1].groupby('시군구_정규')[year_cols].mean()
non_ts = merged[merged['신도시_여부']==0][year_cols].mean()
years = [int(y) for y in year_cols]
ax.plot(years, non_ts.values, color='#90CAF9', linewidth=2, label='비신도시 평균', linestyle='--')
# 상위 신도시만 표시
top_nt = nt_rows.head(8)['시군구_정규'].tolist()
for sg, color in zip(top_nt, plt.cm.tab10.colors):
    sub = merged[merged['시군구_정규']==sg]
    if len(sub):
        vals = sub[year_cols].values[0].astype(float)
        ax.plot(years, vals, linewidth=1.5, label=sg, alpha=0.8)
ax.set_title('주요 신도시 시군구 순이동 시계열', fontsize=10)
ax.set_ylabel('순이동인구 (명)')
ax.set_xlabel('연도')
ax.legend(fontsize=6, ncol=2)
ax.axhline(0, color='gray', linestyle=':')

# ─── Panel 4: 신도시 vs 비신도시 학원 수 비교 ───
ax = axes[1, 0]
ac_yes = merged[merged['신도시_여부']==1]['학원수'].dropna()
ac_no  = merged[merged['신도시_여부']==0]['학원수'].dropna()
stat2, pval2 = stats.mannwhitneyu(ac_yes, ac_no, alternative='two-sided')
bp3 = ax.boxplot([ac_no, ac_yes], patch_artist=True,
                 labels=['비신도시', '신도시'],
                 medianprops=dict(color='black', linewidth=2))
bp3['boxes'][0].set_facecolor('#90CAF9')
bp3['boxes'][1].set_facecolor('#FF8A65')
ax.set_title(f'신도시 여부별 학원 수\nMann-Whitney p={pval2:.4f}', fontsize=10)
ax.set_ylabel('학원 수 (개)')
ax.text(1.5, ac_yes.max()*0.8, f'신도시 중위: {ac_yes.median():.0f}개\n비신도시 중위: {ac_no.median():.0f}개',
        ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ─── Panel 5: 신도시 개수 vs 순이동 산점도 ───
ax = axes[1, 1]
for _, row in merged.iterrows():
    c = '#FF8A65' if row['신도시_여부'] else '#90CAF9'
    ax.scatter(row['신도시_개수'] + np.random.uniform(-0.1,0.1), row['순이동_최근'],
               alpha=0.4, color=c, s=20)
# 상위 시군구 라벨
for _, row in nt_rows.head(10).iterrows():
    ax.annotate(row['시군구_정규'], (row['신도시_개수'], row['순이동_최근']),
                fontsize=6, ha='left', xytext=(5,0), textcoords='offset points')
ax.set_xlabel('시군구 내 신도시 개수')
ax.set_ylabel('순이동인구 최근 평균 (명)')
ax.set_title('신도시 개수 × 순이동인구', fontsize=10)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
r_val, p_nt = stats.spearmanr(merged['신도시_개수'], merged['순이동_최근'].fillna(0))
ax.text(0.05, 0.95, f'Spearman r={r_val:.3f}\np={p_nt:.4f}', transform=ax.transAxes,
        fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='lightyellow'))

# ─── Panel 6: 시도별 신도시 비율 vs 교권침해 정규화 ───
ax = axes[1, 2]
valid = sido_all.dropna(subset=['교권침해_정규화','신도시비율'])
for _, row in valid.iterrows():
    ax.scatter(row['신도시비율']*100, row['교권침해_정규화'],
               s=row['학원수합계']/50, alpha=0.7, color='#5C6BC0')
    ax.annotate(row['시도_mig'].replace('특별시','').replace('광역시','').replace('도','').replace('특별자치시',''),
                (row['신도시비율']*100, row['교권침해_정규화']),
                fontsize=7, xytext=(5,0), textcoords='offset points')
r2, p2 = stats.spearmanr(valid['신도시비율'], valid['교권침해_정규화'])
ax.set_xlabel('시도 내 신도시 시군구 비율 (%)')
ax.set_ylabel('교권침해 정규화 (건/학생 만 명)')
ax.set_title(f'신도시 비율 × 교권침해 (시도 단위, n=17)\nSpearman r={r2:.3f}, p={p2:.3f}', fontsize=10)

# ─── Panel 7: 시도별 순이동인구 vs 교권침해 ───
ax = axes[2, 0]
for _, row in sido_all.dropna(subset=['교권침해_정규화']).iterrows():
    ax.scatter(row['전체순이동최근'], row['교권침해_정규화'],
               s=80, alpha=0.8, color='#EF5350')
    label = row['시도_mig'].replace('특별시','').replace('광역시','').replace('특별자치시','').replace('도','')[:3]
    ax.annotate(label, (row['전체순이동최근'], row['교권침해_정규화']),
                fontsize=7, xytext=(5,2), textcoords='offset points')
r3, p3 = stats.spearmanr(
    sido_all.dropna(subset=['교권침해_정규화'])['전체순이동최근'],
    sido_all.dropna(subset=['교권침해_정규화'])['교권침해_정규화']
)
ax.set_xlabel('시도 평균 순이동인구 (최근)')
ax.set_ylabel('교권침해 정규화')
ax.set_title(f'시도 순이동인구 × 교권침해\nSpearman r={r3:.3f}, p={p3:.3f}', fontsize=10)
ax.axhline(0, color='gray', linestyle='--', alpha=0.3)

# ─── Panel 8: 신도시 TOP 시군구 순이동 바차트 ───
ax = axes[2, 1]
top20 = merged.nlargest(20, '순이동_최근')[['시군구_정규','순이동_최근','신도시_여부','세대목록']].copy()
colors_bar = ['#FF8A65' if x else '#90CAF9' for x in top20['신도시_여부']]
bars = ax.barh(top20['시군구_정규'], top20['순이동_최근'], color=colors_bar)
ax.set_xlabel('순이동인구 (최근 평균, 명)')
ax.set_title('순이동 TOP 20 시군구\n(주황=신도시, 파랑=비신도시)', fontsize=10)
ax.axvline(0, color='black', linewidth=0.8)
for bar, (_, row) in zip(bars, top20.iterrows()):
    if row['신도시_여부']:
        ax.text(bar.get_width()+50, bar.get_y()+bar.get_height()/2,
                row.get('세대목록',''), va='center', fontsize=6, color='gray')

# ─── Panel 9: 경기도 시군구 학원수 × 순이동 × 신도시 버블차트 ───
ax = axes[2, 2]
gyeonggi = merged[merged['시도_mig']=='경기도'].copy()
gyeonggi['학원수_safe'] = gyeonggi['학원수'].fillna(0)
for gen in ['없음','1기','2기','3기']:
    sub = gyeonggi[gyeonggi['최고세대']==gen]
    ax.scatter(sub['학원수_safe'], sub['순이동_최근'],
               s=80, alpha=0.7, color=colors_gen[gen], label=gen,
               edgecolors='white', linewidth=0.5)
for _, row in gyeonggi[gyeonggi['신도시_여부']==1].iterrows():
    ax.annotate(row['시군구_정규'], (row['학원수_safe'], row['순이동_최근']),
                fontsize=6, xytext=(3,3), textcoords='offset points')
r4, p4 = stats.spearmanr(gyeonggi['학원수_safe'], gyeonggi['순이동_최근'].fillna(0))
ax.set_xlabel('학원 수 (개)')
ax.set_ylabel('순이동인구 최근 평균 (명)')
ax.set_title(f'경기도: 학원수 × 순이동 × 신도시\nSpearman r={r4:.3f}, p={p4:.4f}', fontsize=10)
handles = [mpatches.Patch(color=v, label=k) for k,v in colors_gen.items()]
ax.legend(handles=handles, fontsize=8, title='신도시 세대')

plt.tight_layout()
out_path = '/Users/baiohelseu/Desktop/Project/kossda/output/newtown_analysis.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n저장 완료: {out_path}")

# ─────────────────────────────────────────────────────────
# 7. 핵심 수치 요약 출력
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("핵심 통계 요약")
print("="*60)
print(f"\n[신도시 vs 비신도시 순이동인구 비교]")
print(f"  신도시 시군구 (n={len(nt_yes)}): 중위 {nt_yes.median():,.0f}명, 평균 {nt_yes.mean():,.0f}명")
print(f"  비신도시 시군구 (n={len(nt_no)}): 중위 {nt_no.median():,.0f}명, 평균 {nt_no.mean():,.0f}명")
print(f"  Mann-Whitney U p={pval:.4f}")

print(f"\n[신도시 vs 비신도시 학원수 비교]")
print(f"  신도시: 중위 {ac_yes.median():.0f}개, 평균 {ac_yes.mean():.0f}개")
print(f"  비신도시: 중위 {ac_no.median():.0f}개, 평균 {ac_no.mean():.0f}개")
print(f"  Mann-Whitney U p={pval2:.4f}")

print(f"\n[상관분석]")
print(f"  신도시 개수 × 순이동: Spearman r={r_val:.3f}, p={p_nt:.4f}")
print(f"  신도시 비율 × 교권침해: Spearman r={r2:.3f}, p={p2:.3f} (시도, n=17)")
print(f"  시도 순이동 × 교권침해: Spearman r={r3:.3f}, p={p3:.3f} (시도, n=17)")
print(f"  경기 학원수 × 순이동: Spearman r={r4:.3f}, p={p4:.4f}")

print(f"\n[세대별 순이동 중위값]")
for gen, vals in gen_groups.items():
    print(f"  {gen}: {np.median(vals):,.0f}명 (n={len(vals)})")

print(f"\n[순이동 TOP 5 신도시 시군구]")
print(nt_rows[['시도_mig','시군구_정규','세대목록','신도시_개수','순이동_최근']].head(5).to_string(index=False))
