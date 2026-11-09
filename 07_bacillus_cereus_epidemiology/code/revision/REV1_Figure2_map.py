# -*- coding: utf-8 -*-
"""Figure 2 (REV1) — crude incidence of B. cereus food poisoning by district,
Republic of Korea, 2011–2024.

Reviewer R1-3 / N8: scale bar, north arrow, Seoul Capital Area inset,
rounded legend breaks, zero-case districts as their own class, 300 dpi,
projected CRS stated (EPSG:5179).
"""
import glob, unicodedata
import pandas as pd, geopandas as gpd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

OUT = "/home/claude/fig2"
NFC = lambda s: unicodedata.normalize("NFC", str(s))
mpl.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "ps.fonttype": 42})

# ------------------------------------------------------------------ data
g = gpd.read_file(glob.glob("/mnt/user-data/uploads/**/final.gpkg", recursive=True)[0])
g = g.set_crs(5179, allow_override=True)
g["region"] = g["region"].map(NFC)
d = pd.read_csv(glob.glob("/mnt/user-data/uploads/**/BC_district_incidence.csv",
                          recursive=True)[0])
for c in ("region", "sido", "sigungu"):
    d[c] = d[c].map(NFC)
m = g.merge(d[["region", "sido", "sigungu", "cases_total", "pop_mean", "rate_per100k"]],
            on="region", how="left", validate="one_to_one")
assert len(m) == 229 and m["rate_per100k"].notna().all()

BREAKS = [0.10, 0.25, 0.50, 1.00]
LABELS = ["0 (no notified cases)", ">0 – 0.10", "0.10 – 0.25",
          "0.25 – 0.50", "0.50 – 1.00", "> 1.00"]
RAMP   = ["#fee6a0", "#fdc353", "#f79521", "#dd5b13", "#9c3204"]
ZERO   = "#e9edef"
COLORS = [ZERO] + RAMP

def klass(v):
    if v <= 0:
        return 0
    for i, b in enumerate(BREAKS):
        if v <= b:
            return i + 1
    return len(BREAKS) + 1

m["cls"]  = m["rate_per100k"].map(klass)
m["fill"] = m["cls"].map(lambda i: COLORS[i])
counts = m["cls"].value_counts().reindex(range(6), fill_value=0)
sido = m.dissolve(by="sido")

# Seoul Capital Area window: Seoul + Gyeonggi, widened west for mainland Incheon
CAP = m[m["sido"].isin(["서울시", "인천시", "경기도"])]
sgx0, sgy0, sgx1, sgy1 = m[m["sido"].isin(["서울시", "경기도"])].total_bounds
CX0, CX1 = sgx0 - 30000, sgx1 + 6000
CY0, CY1 = sgy0 - 6000, sgy1 + 6000

# --------------------------------------------------------------- helpers
def paint(ax, gdf, lw=0.22):
    gdf.plot(ax=ax, color=gdf["fill"], edgecolor="#ffffff", linewidth=lw, zorder=2)

def scalebar(ax, xf, yf, km, hf=0.010, fs=6.6):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    L = km * 1000.0; half = L / 2.0
    px = x0 + xf * (x1 - x0); py = y0 + yf * (y1 - y0); hh = hf * (y1 - y0)
    ax.add_patch(Rectangle((px, py), half, hh, fc="#2b2b2b", ec="#2b2b2b", lw=0.4, zorder=6))
    ax.add_patch(Rectangle((px + half, py), half, hh, fc="white", ec="#2b2b2b", lw=0.4, zorder=6))
    for f, lab in ((0.0, "0"), (0.5, f"{km // 2}"), (1.0, f"{km} km")):
        ax.text(px + f * L, py + hh * 1.45, lab, ha="center", va="bottom",
                fontsize=fs, color="#2b2b2b", zorder=6)

def north_arrow(ax, xf, yf, sf=0.050, fs=7.5):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    px = x0 + xf * (x1 - x0); py = y0 + yf * (y1 - y0); L = sf * (y1 - y0)
    ax.annotate("", xy=(px, py + L), xytext=(px, py),
                arrowprops=dict(arrowstyle="-|>", color="#2b2b2b", lw=1.0,
                                mutation_scale=9), zorder=6)
    ax.text(px, py + L * 1.05, "N", ha="center", va="bottom", fontsize=fs,
            fontweight="bold", color="#2b2b2b", zorder=6)

# ---------------------------------------------------------------- figure
FW, FH = 7.2, 4.6
fig = plt.figure(figsize=(FW, FH), dpi=300)
fig.patch.set_facecolor("white")

ax = fig.add_axes([0.005, 0.02, 0.60, 0.96])
paint(ax, m)
sido.boundary.plot(ax=ax, color="#6b6b6b", linewidth=0.5, zorder=3)
ax.set_aspect("equal"); ax.set_axis_off()
xmin, ymin, xmax, ymax = m.total_bounds
px, py = (xmax - xmin) * 0.035, (ymax - ymin) * 0.035
ax.set_xlim(xmin - px, xmax + px)
ax.set_ylim(ymin - py, ymax + py)

north_arrow(ax, 0.845, 0.925)
scalebar(ax, 0.035, 0.055, 100)

top2 = m.nlargest(2, "rate_per100k").reset_index(drop=True)
for _, r in top2.iterrows():
    if r["rate_per100k"] > 12:                 # 13.7 -> Gokseong-gun (Jeollanam-do)
        eng, dx, dy = "Gokseong-gun", -152000, -138000
    else:                                       # 10.6 -> Hoengseong-gun (Gangwon-do)
        eng, dx, dy = "Hoengseong-gun", 238000, 38000
    c = r.geometry.representative_point()
    ax.annotate(f"{eng}\n{r['rate_per100k']:.1f}",
                xy=(c.x, c.y), xytext=(c.x + dx, c.y + dy),
                fontsize=6.2, ha="center", va="center", color="#2b2b2b",
                linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color="#5c5c5c", lw=0.55,
                                shrinkA=0, shrinkB=1.5), zorder=7)

# ------------------------------------------------------------------ inset
axi = fig.add_axes([0.655, 0.525, 0.335, 0.435])
paint(axi, CAP, lw=0.28)
CAP.dissolve(by="sido").boundary.plot(ax=axi, color="#6b6b6b", linewidth=0.5, zorder=3)
axi.set_aspect("equal"); axi.set_xlim(CX0, CX1); axi.set_ylim(CY0, CY1)
axi.set_xticks([]); axi.set_yticks([])
for sp in axi.spines.values():
    sp.set_edgecolor("#2b2b2b"); sp.set_linewidth(0.85)
axi.set_title("Seoul Capital Area", fontsize=7.2, pad=3.0, color="#2b2b2b")
scalebar(axi, 0.045, 0.05, 40, hf=0.016, fs=5.7)

ax.add_patch(Rectangle((CX0, CY0), CX1 - CX0, CY1 - CY0, fill=False,
                       ec="#2b2b2b", lw=0.7, ls=(0, (3, 2)), zorder=5))

# ------------------------------------------------------------------ legend
axl = fig.add_axes([0.645, 0.105, 0.35, 0.40]); axl.set_axis_off()
handles = [Patch(fc=COLORS[i], ec="#8d9499", lw=0.45,
                 label=f"{LABELS[i]}   (n = {counts[i]})") for i in range(6)]
leg = axl.legend(handles=handles, loc="upper left", frameon=False, fontsize=6.9,
                 handlelength=1.45, handleheight=1.0, labelspacing=0.55,
                 borderaxespad=0, alignment="left",
                 title="Mean annual crude incidence\nper 100,000 population",
                 title_fontsize=7.3)
leg._legend_box.align = "left"

fig.text(0.652, 0.190,
         "229 districts; 1,223 notified cases, 2011\u20132024.\n"
         "Classes use rounded cut points; districts with no\n"
         "notified case form a separate class.\n"
         "Projection: Korea 2000 / Unified CS (EPSG:5179).",
         fontsize=5.9, color="#4a4a4a", va="top", ha="left", linespacing=1.6)

for ext, kw in (("png", dict(dpi=300)),
                ("pdf", dict()),
                ("tiff", dict(dpi=300, pil_kwargs={"compression": "tiff_lzw"}))):
    fig.savefig(f"{OUT}/Figure2_incidence_map.{ext}", facecolor="white", **kw)
plt.close(fig)
print("classes", dict(counts))
