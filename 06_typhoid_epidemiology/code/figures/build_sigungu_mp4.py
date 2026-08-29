"""Build sigungu-level MP4 animation: yearly typhoid crude rate (per 100k) by sigungu, 2001-2024.
Matches user-uploaded hepatitis_a_crude_rate.mp4 reference style (960x960, 2 fps, discrete bins, white bg)."""
import pandas as pd, numpy as np, os, re
import geopandas as gpd
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.patches import Patch

DATA = '/sessions/epic-determined-feynman/mnt/논문_이건상/장티푸스/data'
SHP  = '/sessions/epic-determined-feynman/mnt/논문_이건상/data_sort/쉐이프파일/map/final.shp'
FIG  = '/sessions/epic-determined-feynman/mnt/논문_이건상/장티푸스/FBPD_R2/figures'
FINAL_FIG = '/sessions/epic-determined-feynman/mnt/논문_이건상/장티푸스/FBPD_R2/FINAL_SUBMISSION/figures'

# --- Load shapefile ---
gdf = gpd.read_file(SHP, encoding='utf-8').set_crs('EPSG:5179', allow_override=True)
print(f'Shapefile: {len(gdf)} sigungu')

# --- Load case data with proper sigungu key ---
SIDO_LONG_TO_SHORT = {
    '강원':'강원도','경기':'경기도','경남':'경상남도','경북':'경상북도',
    '광주':'광주시','대구':'대구시','대전':'대전시','부산':'부산시',
    '서울':'서울시','세종':'세종시','울산':'울산시','인천':'인천시',
    '전남':'전라남도','전북':'전라북도','제주':'제주도',
    '충남':'충청남도','충북':'충청북도',
    '서울특별시':'서울시','부산광역시':'부산시','대구광역시':'대구시','인천광역시':'인천시',
    '광주광역시':'광주시','대전광역시':'대전시','울산광역시':'울산시','세종특별자치시':'세종시',
    '경기도':'경기도','강원도':'강원도','강원특별자치도':'강원도',
    '충청북도':'충청북도','충청남도':'충청남도',
    '전라북도':'전라북도','전북특별자치도':'전라북도','전라남도':'전라남도',
    '경상북도':'경상북도','경상남도':'경상남도',
    '제주특별자치도':'제주도','제주도':'제주도',
}
PARENT_CITIES = ['수원시','성남시','안양시','안산시','고양시','용인시','창원시','청주시','천안시','전주시','포항시']
def collapse_parent(s):
    s = str(s).strip()
    for pc in PARENT_CITIES:
        if s.startswith(pc): return pc
    return s

df1 = pd.read_csv(f'{DATA}/df1_장티푸스.csv')
df1.columns = [c.strip().replace('\ufeff','') for c in df1.columns]
df1['year'] = df1['year'].astype(int); df1 = df1[(df1.year>=2001)&(df1.year<=2024)]
df1['sido_short'] = df1['sido'].map(SIDO_LONG_TO_SHORT)
df1['sigungu_clean'] = df1['sigungu'].apply(collapse_parent)
df1['region_key'] = df1['sido_short'] + df1['sigungu_clean']
yearly_cases = df1.groupby(['year','region_key'], as_index=False)['cases'].sum()

# --- Population from KOSIS hierarchical CSVs ---
import glob
SIDO_NAMES = {'서울특별시':'서울시','부산광역시':'부산시','대구광역시':'대구시','인천광역시':'인천시',
              '광주광역시':'광주시','대전광역시':'대전시','울산광역시':'울산시','세종특별자치시':'세종시',
              '경기도':'경기도','강원도':'강원도','강원특별자치도':'강원도',
              '충청북도':'충청북도','충청남도':'충청남도',
              '전라북도':'전라북도','전북특별자치도':'전라북도','전라남도':'전라남도',
              '경상북도':'경상북도','경상남도':'경상남도',
              '제주특별자치도':'제주도','제주도':'제주도'}

pop_rows = []
for path in sorted(glob.glob('/sessions/epic-determined-feynman/mnt/논문_이건상/data_sort/인구/*.csv')):
    yr = int(os.path.basename(path).split('.')[0])
    if not (2001 <= yr <= 2024): continue
    df = pd.read_csv(path, encoding='cp949')
    df = df[df['연령별'] == '계']
    cur_sido = None
    for _, r in df.iterrows():
        name = str(r['행정구역별']).strip()
        if name in SIDO_NAMES:
            cur_sido = SIDO_NAMES[name]
            # Also record the sido total for sido itself (Sejong has no sub-sigungu)
            try:
                pop_rows.append((yr, cur_sido + cur_sido if cur_sido in ['세종시'] else None, int(r['전체']), 'sido_self'))
            except: pass
            continue
        if name == '전국' or cur_sido is None: continue
        clean = re.sub(r'^[가-힣]+-', '', name)
        try:
            pop = int(r['전체'])
        except: continue
        rk = cur_sido + collapse_parent(clean)
        pop_rows.append((yr, rk, pop, 'sigungu'))

pop_df = pd.DataFrame(pop_rows, columns=['year','region_key','pop','kind']).dropna(subset=['region_key'])
pop_df = pop_df.groupby(['year','region_key'], as_index=False)['pop'].sum()
print(f'Pop rows: {len(pop_df)}; unique regions: {pop_df["region_key"].nunique()}')

# Merge with cases
yearly = yearly_cases.merge(pop_df, on=['year','region_key'], how='left')
yearly['rate'] = yearly['cases'] / yearly['pop'] * 1e5

# Fill in zero-case sigungu (some sigungu have no cases in some years - they should still show 0)
all_pairs = pd.MultiIndex.from_product([range(2001,2025), yearly['region_key'].unique()],
                                        names=['year','region_key']).to_frame(index=False)
yearly = all_pairs.merge(yearly, on=['year','region_key'], how='left')
yearly['cases'] = yearly['cases'].fillna(0)
yearly = yearly.merge(pop_df, on=['year','region_key'], how='left', suffixes=('','_p'))
yearly['pop'] = yearly['pop'].fillna(yearly['pop_p'])
yearly = yearly.drop(columns=['pop_p'])
yearly['rate'] = yearly['cases'] / yearly['pop'] * 1e5

# Match against shapefile keys to find unmatched
shp_keys = set(gdf['region'].unique())
data_keys = set(yearly['region_key'].unique())
print(f'Data regions: {len(data_keys)}; matched with shp: {len(data_keys & shp_keys)}; missing: {len(shp_keys - data_keys)}')
print('Missing in data:', sorted(shp_keys - data_keys))

# Save merged
yearly.to_csv('/sessions/epic-determined-feynman/mnt/outputs/sigungu_yearly_rate.csv', index=False)
print(f'\nrate range: 0 – {yearly["rate"].max():.2f}/100k (NaN={yearly["rate"].isna().sum()})')
print(f'rate quantiles 0.5/0.7/0.9/0.95/0.99: ',
      yearly['rate'].quantile([0.5, 0.7, 0.9, 0.95, 0.99]).round(2).tolist())

# === Animate ===
years = sorted(yearly['year'].unique())
print(f'Years to animate: {years[0]}–{years[-1]} ({len(years)} frames)')

# Match user reference: discrete bins like [0, 1.96, 3.28, 5.70, 10.01, 269.82]
# For typhoid the rates are lower; let's pick bins based on quantiles
nz = yearly[(yearly['rate']>0) & yearly['rate'].notna()]['rate']
bins_q = nz.quantile([0.4, 0.6, 0.8, 0.95]).round(2).tolist()
bins = [0, 1e-9] + bins_q + [yearly['rate'].max()]
print(f'Bins: {bins}')

cmap = mcolors.LinearSegmentedColormap.from_list('reds', ['#FFFFFF','#FFE5CC','#FFB37A','#FF7B3F','#D0381E','#7A0B0B'])
norm = mcolors.BoundaryNorm(bins, cmap.N)

# Pre-merge geometry per frame for speed
gdf['region'] = gdf['region'].astype(str)
fig, ax = plt.subplots(figsize=(9.6, 9.6), dpi=100, constrained_layout=False)
fig.patch.set_facecolor('white')

def draw(yr):
    ax.clear()
    sub = yearly[yearly.year == yr]
    pdata = gdf.merge(sub[['region_key','rate']], left_on='region', right_on='region_key', how='left')
    pdata.plot(column='rate', ax=ax, cmap=cmap, norm=norm,
               edgecolor='gray', linewidth=0.25, missing_kwds={'color':'#EEEEEE'})
    ax.set_title(f'[Typhoid fever] Crude rate — {yr}', fontsize=18, fontweight='bold', pad=12)
    ax.set_axis_off()
    ax.set_aspect('equal')

# Draw frame 1 to set extent
draw(years[0])
xlim = ax.get_xlim(); ylim = ax.get_ylim()

# Manual colorbar with discrete bins on right side
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, ticks=bins, fraction=0.04, pad=0.02)
cbar.ax.set_yticklabels([f'{b:g}' if b != 1e-9 else '1.0e-9' for b in bins], fontsize=10)
plt.subplots_adjust(left=0.03, right=0.92, top=0.95, bottom=0.03)

def update(yr):
    ax.clear()
    sub = yearly[yearly.year == yr]
    pdata = gdf.merge(sub[['region_key','rate']], left_on='region', right_on='region_key', how='left')
    pdata.plot(column='rate', ax=ax, cmap=cmap, norm=norm,
               edgecolor='gray', linewidth=0.25, missing_kwds={'color':'#EEEEEE'})
    ax.set_title(f'[Typhoid fever] Crude rate — {yr}', fontsize=18, fontweight='bold', pad=12)
    ax.set_axis_off()
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    return [ax]

ani = animation.FuncAnimation(fig, update, frames=years, interval=500, repeat=True, blit=False)
mp4_path = f'{FIG}/Figure_S4.mp4'
print('Encoding MP4...')
ani.save(mp4_path, writer='ffmpeg', fps=2, dpi=100, savefig_kwargs={'facecolor':'white'},
         extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18'])
plt.close()
print(f'Saved {mp4_path} ({os.path.getsize(mp4_path):,} bytes)')

# Copy to FINAL_SUBMISSION
import shutil
shutil.copy2(mp4_path, f'{FINAL_FIG}/Figure_S4_animated.mp4')
print(f'Copied to FINAL_SUBMISSION/figures/Figure_S4_animated.mp4')
