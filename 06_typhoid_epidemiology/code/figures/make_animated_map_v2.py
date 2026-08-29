"""Better sido mapping; correct population denominators."""
import geopandas as gpd, pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
import matplotlib.animation as animation
import os

DATA = '/sessions/epic-determined-feynman/mnt/논문_이건상/장티푸스/data'
SHP  = '/sessions/epic-determined-feynman/mnt/논문_이건상/data_sort/쉐이프파일/map/final.shp'
FIG  = '/sessions/epic-determined-feynman/mnt/논문_이건상/장티푸스/FBPD_R2/figures'

# Comprehensive sido mapping covering both short and full names
SIDO_PATTERNS = [
    ('서울특별시','Seoul'),('서울시','Seoul'),('서울','Seoul'),
    ('부산광역시','Busan'),('부산시','Busan'),('부산','Busan'),
    ('대구광역시','Daegu'),('대구시','Daegu'),('대구','Daegu'),
    ('인천광역시','Incheon'),('인천시','Incheon'),('인천','Incheon'),
    ('광주광역시','Gwangju'),('광주시','Gwangju'),('광주','Gwangju'),
    ('대전광역시','Daejeon'),('대전시','Daejeon'),('대전','Daejeon'),
    ('울산광역시','Ulsan'),('울산시','Ulsan'),('울산','Ulsan'),
    ('세종특별자치시','Sejong'),('세종시','Sejong'),('세종','Sejong'),
    ('경기도','Gyeonggi'),('경기','Gyeonggi'),
    ('강원특별자치도','Gangwon'),('강원도','Gangwon'),('강원','Gangwon'),
    ('충청북도','Chungbuk'),('충북','Chungbuk'),
    ('충청남도','Chungnam'),('충남','Chungnam'),
    ('전북특별자치도','Jeonbuk'),('전라북도','Jeonbuk'),('전북','Jeonbuk'),
    ('전라남도','Jeonnam'),('전남','Jeonnam'),
    ('경상북도','Gyeongbuk'),('경북','Gyeongbuk'),
    ('경상남도','Gyeongnam'),('경남','Gyeongnam'),
    ('제주특별자치도','Jeju'),('제주도','Jeju'),('제주','Jeju'),
]
def sido_of(s):
    s = str(s)
    for k, v in SIDO_PATTERNS:
        if s.startswith(k): return v
    return None

# --- Shapefile ---
gdf = gpd.read_file(SHP, encoding='utf-8').set_crs('EPSG:5179', allow_override=True)
gdf['sido_en'] = gdf['region'].apply(sido_of)
print('Shapefile sido counts:', gdf['sido_en'].value_counts(dropna=False).to_dict())
sido_geo = gdf.dropna(subset=['sido_en']).dissolve(by='sido_en', as_index=False)[['sido_en','geometry']]
print(f'Dissolved to {len(sido_geo)} sido polygons')

# --- Yearly cases by sido ---
df1 = pd.read_csv(f'{DATA}/df1_장티푸스.csv')
df1.columns = [c.strip().replace('\ufeff','') for c in df1.columns]
df1['year'] = df1['year'].astype(int)
df1 = df1[(df1.year>=2001)&(df1.year<=2024)]
df1['sido_en'] = df1['sido'].apply(sido_of)
print('df1 unmapped sido:', df1[df1['sido_en'].isna()]['sido'].unique())
df1 = df1.dropna(subset=['sido_en'])
yr_sido = df1.groupby(['year','sido_en'], as_index=False)['cases'].sum()

# --- Sido populations (from R1 tbl_sido_rate.csv mean_pop) ---
prov = pd.read_csv('/sessions/epic-determined-feynman/mnt/논문_이건상/장티푸스/reanalysis_2026_FoodbornePathogensDisease/tables/tbl_sido_rate.csv')
sido_pop = prov[['sido_en','mean_pop']]
yr_sido = yr_sido.merge(sido_pop, on='sido_en', how='left')
yr_sido['rate_per_100k'] = yr_sido['cases'] / yr_sido['mean_pop'] * 1e5
print(yr_sido.head())
print(f'Years: {yr_sido.year.min()}–{yr_sido.year.max()}, sidos: {yr_sido.sido_en.nunique()}')

# Color scale - cap at 95th pct to avoid extreme outlier washing
vmax = float(yr_sido['rate_per_100k'].quantile(0.97))
print(f'Color scale vmax = {vmax:.2f}/100k')
norm = mcolors.Normalize(vmin=0, vmax=vmax)
cmap = plt.cm.YlOrRd

# === 6-panel snapshot ===
years_to_show = [2001, 2007, 2013, 2018, 2021, 2024]
fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
fig.suptitle('Figure S4.  Annual sido-level typhoid incidence rate per 100,000 population, selected years (2001–2024)',
             fontsize=13, fontweight='bold')
for ax, yr in zip(axes.flat, years_to_show):
    sub = yr_sido[yr_sido.year == yr]
    plot_data = sido_geo.merge(sub, on='sido_en', how='left')
    plot_data.plot(column='rate_per_100k', ax=ax, cmap=cmap, norm=norm,
                   edgecolor='gray', linewidth=0.4, missing_kwds={'color':'lightgrey'})
    ax.set_title(f'{yr}', fontsize=12, fontweight='bold'); ax.set_axis_off()
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02, shrink=0.8)
cbar.set_label('Typhoid incidence rate (per 100,000)', fontsize=10)
plt.savefig(f'{FIG}/Figure_S4.png', dpi=180, bbox_inches='tight')
plt.close()
print('Saved Figure_S4.png')

# === Animated GIF, 24 frames ===
fig2, ax2 = plt.subplots(figsize=(8.5, 9), constrained_layout=True)
def update(yr):
    ax2.clear()
    sub = yr_sido[yr_sido.year == yr]
    pdata = sido_geo.merge(sub, on='sido_en', how='left')
    pdata.plot(column='rate_per_100k', ax=ax2, cmap=cmap, norm=norm,
               edgecolor='gray', linewidth=0.5, missing_kwds={'color':'lightgrey'})
    # Overlay sido labels
    for _, r in pdata.iterrows():
        cent = r.geometry.centroid
        rate = r['rate_per_100k']
        if pd.notna(rate):
            color = 'white' if rate > vmax*0.55 else 'black'
            ax2.annotate(f"{r['sido_en']}\n{rate:.2f}", (cent.x, cent.y),
                         ha='center', va='center', fontsize=7, color=color)
    ax2.set_title(f'Typhoid fever incidence per 100,000 — {yr}', fontsize=14, fontweight='bold')
    ax2.set_axis_off()
ani = animation.FuncAnimation(fig2, update, frames=range(2001, 2025), interval=500, repeat=True)
gif_path = f'{FIG}/Figure_S4.gif'
ani.save(gif_path, writer='pillow', fps=2, dpi=110)
plt.close()
print(f'Saved Figure_S4.gif ({os.path.getsize(gif_path):,} bytes)')

# === Plotly interactive HTML ===
import plotly.express as px
import json
sido_wgs = sido_geo.to_crs('EPSG:4326')
geo = json.loads(sido_wgs.to_json())
# Plotly choropleth_mapbox or choropleth
yr_sido_complete = yr_sido.copy()
yr_sido_complete['rate_per_100k'] = yr_sido_complete['rate_per_100k'].fillna(0)
yr_sido_complete['year'] = yr_sido_complete['year'].astype(int)

fig_p = px.choropleth(
    yr_sido_complete,
    geojson=geo,
    locations='sido_en',
    featureidkey='properties.sido_en',
    color='rate_per_100k',
    color_continuous_scale='YlOrRd',
    range_color=(0, vmax),
    animation_frame='year',
    labels={'rate_per_100k':'Rate / 100,000', 'sido_en':'Sido'},
    title='Figure S4. Annual typhoid fever incidence rate by sido, 2001–2024 (interactive)'
)
fig_p.update_geos(fitbounds='locations', visible=False, projection_type='mercator')
fig_p.update_layout(margin={'r':0,'t':50,'l':0,'b':0}, height=700)
html_path = f'{FIG}/Figure_S4.html'
fig_p.write_html(html_path, include_plotlyjs='cdn', auto_play=False)
print(f'Saved Figure_S4.html ({os.path.getsize(html_path):,} bytes)')

# clean up unused intermediate
geojson_path = f'{FIG}/sido_kr.geojson'
sido_wgs.to_file(geojson_path, driver='GeoJSON')
