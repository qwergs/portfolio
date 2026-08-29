"""Smaller HTML by aggressive geometry simplification."""
import geopandas as gpd, pandas as pd, json, os
import plotly.graph_objects as go

DATA = '/sessions/epic-determined-feynman/mnt/논문_이건상/장티푸스/data'
SHP  = '/sessions/epic-determined-feynman/mnt/논문_이건상/data_sort/쉐이프파일/map/final.shp'
FIG  = '/sessions/epic-determined-feynman/mnt/논문_이건상/장티푸스/FBPD_R2/figures'

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

gdf = gpd.read_file(SHP, encoding='utf-8').set_crs('EPSG:5179', allow_override=True)
gdf['sido_en'] = gdf['region'].apply(sido_of)
sido_geo = gdf.dropna(subset=['sido_en']).dissolve(by='sido_en', as_index=False)[['sido_en','geometry']]
# Aggressive simplification — single-precision coords are fine for choropleth
sido_geo['geometry'] = sido_geo['geometry'].simplify(tolerance=2000, preserve_topology=True)
sido_geo = sido_geo.to_crs('EPSG:4326')
print(f'sido count: {len(sido_geo)}')

# Manually round coordinates to reduce JSON size
def round_geom(geom, prec=3):
    from shapely.geometry import mapping, shape
    d = mapping(geom)
    def round_coords(c):
        if isinstance(c, (list, tuple)):
            if c and isinstance(c[0], (int, float)):
                return [round(c[0], prec), round(c[1], prec)]
            return [round_coords(x) for x in c]
        return c
    d['coordinates'] = round_coords(d['coordinates'])
    return shape(d)

sido_geo['geometry'] = sido_geo['geometry'].apply(lambda g: round_geom(g, 3))
geo = json.loads(sido_geo.to_json())
for f in geo['features']:
    f['id'] = f['properties']['sido_en']
gjsize = len(json.dumps(geo).encode())
print(f'GeoJSON size: {gjsize:,} bytes')

# Cases
df1 = pd.read_csv(f'{DATA}/df1_장티푸스.csv')
df1.columns = [c.strip().replace('\ufeff','') for c in df1.columns]
df1['year'] = df1['year'].astype(int)
df1 = df1[(df1.year>=2001)&(df1.year<=2024)]
df1['sido_en'] = df1['sido'].apply(sido_of)
df1 = df1.dropna(subset=['sido_en'])
yr_sido = df1.groupby(['year','sido_en'], as_index=False)['cases'].sum()
prov = pd.read_csv('/sessions/epic-determined-feynman/mnt/논문_이건상/장티푸스/reanalysis_2026_FoodbornePathogensDisease/tables/tbl_sido_rate.csv')
yr_sido = yr_sido.merge(prov[['sido_en','mean_pop']], on='sido_en', how='left')
yr_sido['rate_per_100k'] = (yr_sido['cases'] / yr_sido['mean_pop'] * 1e5).fillna(0).round(3)
years = sorted(yr_sido['year'].unique())
vmax = float(yr_sido['rate_per_100k'].quantile(0.97))

# IMPORTANT: include geojson only ONCE in the first trace, not per-frame
# Plotly auto-reuses geojson when geographic features are referenced by id only
init = yr_sido[yr_sido.year == years[0]]
fig_p = go.Figure(
    data=[go.Choropleth(
        geojson=geo,
        locations=init['sido_en'].tolist(),
        z=init['rate_per_100k'].tolist(),
        featureidkey='properties.sido_en',
        colorscale='YlOrRd', zmin=0, zmax=vmax,
        colorbar=dict(title='Rate /100k'),
        text=[f"{s}: {r}/100k" for s,r in zip(init['sido_en'], init['rate_per_100k'])],
        hoverinfo='text'
    )],
)

# Frames: only update z and text, not geojson (saves a lot of size)
frames = []
for yr in years:
    sub = yr_sido[yr_sido.year == yr]
    frames.append(go.Frame(
        data=[go.Choropleth(
            locations=sub['sido_en'].tolist(),
            z=sub['rate_per_100k'].tolist(),
            featureidkey='properties.sido_en',
            colorscale='YlOrRd', zmin=0, zmax=vmax,
            text=[f"{s}: {r}/100k" for s,r in zip(sub['sido_en'], sub['rate_per_100k'])],
            hoverinfo='text'
        )],
        name=str(yr)
    ))
fig_p.frames = frames

fig_p.update_geos(fitbounds='locations', visible=False, projection_type='mercator')
fig_p.update_layout(
    title=dict(text='Figure S4. Annual sido-level typhoid fever incidence rate, 2001–2024 (interactive)',
               x=0.5, font=dict(size=14)),
    height=720,
    updatemenus=[dict(type='buttons', showactive=False,
        buttons=[
            dict(label='▶ Play', method='animate',
                 args=[None, dict(frame=dict(duration=600, redraw=True), fromcurrent=True)]),
            dict(label='⏸ Pause', method='animate',
                 args=[[None], dict(mode='immediate')])
        ], x=0.05, y=0.0)],
    sliders=[dict(
        steps=[dict(method='animate',
                    args=[[str(yr)], dict(mode='immediate', frame=dict(duration=400, redraw=True))],
                    label=str(yr)) for yr in years],
        currentvalue=dict(prefix='Year: ', font=dict(size=14)),
        x=0.1, y=0.05, len=0.85
    )]
)
out = f'{FIG}/Figure_S4.html'
fig_p.write_html(out, include_plotlyjs='cdn', auto_play=False)
print(f'Saved Figure_S4.html ({os.path.getsize(out):,} bytes)')
