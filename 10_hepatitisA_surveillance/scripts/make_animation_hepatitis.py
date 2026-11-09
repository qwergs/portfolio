"""Hepatitis A — sigungu × year animated choropleth (COLOUR: Reds).
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

DATA_CSV = '/sessions/elegant-quirky-hamilton/mnt/논문_이건상/A형간염/data/hepa_yearly_sigungu.csv'
GEOJSON  = '/sessions/elegant-quirky-hamilton/mnt/논문_이건상/장출/무제 폴더/기타/ehec-streamlit/data/korea_sigungu.geojson'
OUT_MP4  = '/sessions/elegant-quirky-hamilton/mnt/outputs/hepa_work/hepatitis_a_crude_rate.mp4'
OUT_GIF  = '/sessions/elegant-quirky-hamilton/mnt/outputs/hepa_work/hepatitis_a_crude_rate.gif'

DOUBLED_CITIES = ['수원시','성남시','안양시','안산시','고양시','용인시',
                  '창원시','청주시','천안시','전주시','포항시']

def make_white_to_red_cmap(n=6):
    reds = mpl.colormaps['Reds'].resampled(n)
    return mpl.colors.ListedColormap([(1,1,1,1)] + [reds(i) for i in range(1, n)])

def quintile_bins(values, eps=1e-9):
    v = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy()
    pos = v[np.isfinite(v) & (v > eps)]
    if len(pos) == 0:
        return np.array([0, eps, eps*2, eps*3, eps*4, eps*5, eps*6])
    qs = np.quantile(pos, [0.2, 0.4, 0.6, 0.8, 1.0])
    return np.concatenate(([0.0, eps], qs))

def fmt_tick(x):
    if x == 0: return '0'
    if x < 0.01: return f'{x:.2e}'
    return f'{x:.2f}'

df = pd.read_csv(DATA_CSV, encoding='utf-8-sig')
df.columns = df.columns.str.strip()
gdf = gpd.read_file(GEOJSON)

def needs_double_fix(region):
    for city in DOUBLED_CITIES:
        if region.endswith(city): return True
    return False
df['_needs_fix'] = df.region.apply(needs_double_fix)
df.loc[df._needs_fix, 'pop'] = df.loc[df._needs_fix, 'pop'] / 2
df.loc[df._needs_fix, 'crude_rate'] = df.loc[df._needs_fix, 'crude_rate'] * 2

print(f'CSV rows: {len(df):,}; sigungu: {df.region.nunique()}; '
      f'years: {df.year.min()}–{df.year.max()}')
print(f'GeoJSON polygons: {len(gdf)}')

years = sorted(df.year.unique())
edges = quintile_bins(df.crude_rate.values)
cmap = make_white_to_red_cmap(6)
norm = mpl.colors.BoundaryNorm(edges, ncolors=cmap.N, clip=True)

fig, ax = plt.subplots(figsize=(8, 8))
cax = fig.add_axes([0.88, 0.20, 0.03, 0.60])
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cb = fig.colorbar(sm, cax=cax, boundaries=edges)
cb.set_ticks(edges)
cb.set_ticklabels([fmt_tick(t) for t in edges], fontsize=10)
cb.set_label('Crude incidence rate (per 100,000)', fontsize=11)

DISEASE_LABEL = 'Hepatitis A'

def update(frame):
    ax.clear(); ax.axis('off')
    yr = years[frame]
    d = df[df.year == yr][['region','crude_rate']].copy()
    d['crude_rate'] = pd.to_numeric(d['crude_rate'], errors='coerce')
    m = gdf.merge(d, on='region', how='left')
    m.plot(ax=ax, column='crude_rate', cmap=cmap, norm=norm,
           linewidth=0.3, edgecolor='black',
           missing_kwds=dict(color='lightgrey'))
    ax.set_title(f'[{DISEASE_LABEL}] Crude rate — {yr}',
                 fontsize=15, fontweight='bold')
    return ()

print(f'Rendering {len(years)} frames…')
ani = animation.FuncAnimation(fig, update, frames=len(years),
                              interval=500, repeat=True)
ani.save(OUT_MP4, writer='ffmpeg', fps=2, dpi=120,
         extra_args=['-vcodec','libx264','-pix_fmt','yuv420p'])
print(f'Wrote {OUT_MP4}')
ani.save(OUT_GIF, writer='pillow', fps=2, dpi=80)
print(f'Wrote {OUT_GIF}')
plt.close(fig)
