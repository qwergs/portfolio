#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_all_figures.py — IJEHR B. cereus (REV1): regenerate every figure.

Output  : <B_cereus>/figures/*.png | *.pdf | *.tiff   (300 dpi, TIFF is LZW)
Produces: Figure 1-4  and  Supplementary Figure S1-S5.

Run
    python3 make_all_figures.py                 # everything it can build
    python3 make_all_figures.py --only 2 3 4    # just those
    python3 make_all_figures.py --list          # show inputs each figure needs

Inputs are located by name anywhere under <B_cereus>, so the exact folder
layout does not matter. Korean filenames are matched NFC/NFD-insensitively
(macOS stores NFD, most tools emit NFC).

Figures 1, S2, S3d-g and S4 need the district-year analysis panel, which only
R can build. Run  figures/00_export_figure_data.R  once; it writes
figures/_data/*.csv. Any figure whose inputs are missing is SKIPPED with a
message, never a crash.
"""
from __future__ import annotations

import argparse
import sys
import warnings
import unicodedata
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

# ----------------------------------------------------------------- settings
ROOT = Path("/Users/guns/Desktop/과거자료/학부생_전처리_논문/도시_농촌_분석/B_cereus")
OUT = ROOT / "figures"
DATA = OUT / "_data"

DPI = 300
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Apple SD Gothic Neo", "AppleGothic",
                        "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.7, "axes.edgecolor": "#333333",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})

C_OVERALL, C_URBAN, C_RURAL = "#1f6fb4", "#e8623c", "#2e9e6b"
GREY = "#6b6b6b"

warnings.filterwarnings("ignore", message="Glyph .* missing from font")

NFC = lambda s: unicodedata.normalize("NFC", str(s))

# Korean register names -> the English labels used in the manuscript
KO2EN = {
    "성비": "Sex ratio",
    "고령인구비율": "Elderly-population rate",
    "도시지역인구비율": "Urban-area population rate",
    "상수도보급률": "Piped-water coverage",
    "하수도보급률": "Sewerage coverage",
    "고도처리인구보급률": "Advanced-treatment coverage",
    "고도처리인구보급률(%)": "Advanced-treatment coverage",
    "정화조인구": "Septic-tank population",
    "하수처리구역외_정화조인구": "Septic-tank population",
    "가정용지하수개소수": "Household groundwater wells",
    "가정용_개소수": "Household groundwater wells",
    "민방위용지하수이용량": "Civil-defence groundwater use",
    "민방위용_이용량": "Civil-defence groundwater use",
    "유지": "Reservoir area",
    "하천": "Stream and river area",
    "목장용지": "Pasture area",
    "답": "Paddy-field area",
    "임야": "Forest area",
    "의사수": "Physician count",
    "관내진료비전체": "In-area medical expenditure",
    "관내진료비_전체": "In-area medical expenditure",
    "연간인플루엔자예방접종률_표준화율": "Influenza-vaccination uptake",
    "인플루엔자예방접종률": "Influenza-vaccination uptake",
    "건강생활실천율_조율": "Healthy-lifestyle practice rate",
    "건강생활실천율": "Healthy-lifestyle practice rate",
    "걷기실천율_표준화율": "Walking-practice rate",
    "걷기실천율": "Walking-practice rate",
    "재정자립도": "Fiscal independence",
    "재정자주도": "Fiscal autonomy",
}
en = lambda v: KO2EN.get(NFC(v), NFC(v))


# ------------------------------------------------------------------- helpers
_INDEX: dict[str, Path] | None = None


def _build_index() -> dict[str, Path]:
    """map NFC-normalised file name -> path, for every file under ROOT."""
    idx: dict[str, Path] = {}
    for p in ROOT.rglob("*"):
        if p.is_file():
            idx.setdefault(NFC(p.name), p)
    return idx


def find(name: str, required: bool = True) -> Path | None:
    """locate a file by name anywhere under ROOT (NFC/NFD tolerant)."""
    global _INDEX
    if _INDEX is None:
        _INDEX = _build_index()
    p = _INDEX.get(NFC(name))
    if p is None and required:
        raise FileNotFoundError(name)
    return p


def sheet(xlsx_name: str, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(find(xlsx_name), sheet_name=sheet_name)


def sheet_any(xlsx_name: str, sheet_name: str) -> pd.DataFrame:
    """Same as sheet(), but tries every copy of the file under ROOT.

    Some copies live in iCloud and are not downloaded; reading one of those
    raises OSError 35 (resource deadlock avoided), so fall through to the next.
    """
    err: Exception | None = None
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or NFC(p.name) != NFC(xlsx_name):
            continue
        try:
            return pd.read_excel(p, sheet_name=sheet_name)
        except Exception as e:                                # noqa: BLE001
            err = e
    raise err or FileNotFoundError(xlsx_name)


def panel() -> pd.DataFrame:
    """district-year analysis panel exported by 00_export_figure_data.R"""
    f = DATA / "panel_district_year.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f}  (run figures/00_export_figure_data.R first)")
    d = pd.read_csv(f)
    d["region"] = d["region"].map(NFC)
    return d


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=DPI, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    fig.savefig(OUT / f"{stem}.tiff", dpi=DPI, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"      -> {stem}.png / .pdf / .tiff")


def parse_est(s) -> tuple[float, float, float, bool]:
    """'1.75 (1.07 to 2.94) *'  ->  (1.75, 1.07, 2.94, True)"""
    if not isinstance(s, str) or not s.strip() or s.strip() in {"-", "—"}:
        return (np.nan,) * 3 + (False,)
    star = "*" in s
    t = s.replace("*", "").strip()
    est = float(t.split("(")[0])
    lo, hi = (float(v) for v in t.split("(")[1].rstrip(")").replace("–", "to")
              .replace("-", "to").split("to")[:2])
    return est, lo, hi, star


def forest_axes(ax, labels, groups, xlim, xlabel):
    """groups: list of (name, colour, list[(est, lo, hi, sig)]) aligned to labels"""
    n = len(labels)
    offs = np.linspace(0.26, -0.26, len(groups)) if len(groups) > 1 else [0.0]
    for gi, (gname, col, vals) in enumerate(groups):
        ys = [n - 1 - i + offs[gi] for i in range(n)]
        for y, (e, lo, hi, sig) in zip(ys, vals):
            if not np.isfinite(e):
                continue
            ax.plot([lo, hi], [y, y], color=col, lw=1.5 if sig else 1.0,
                    solid_capstyle="round", zorder=3)
            ax.plot([e], [y], "o", ms=5.6 if sig else 4.4, color=col,
                    mfc=col if sig else "white", mew=1.2, zorder=4)
    ax.axvline(1, color="#999999", lw=0.9, ls=(0, (4, 3)), zorder=1)
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(n - 1 - i - 0.5, n - 1 - i + 0.5, color="#f6f7f8", zorder=0)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ticks = [0.25, 0.5, 1, 2, 4]
    ax.set_xticks([t for t in ticks if xlim[0] <= t <= xlim[1]])
    ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: f"{v:g}"))
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.tick_params(axis="x", labelsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


# ================================================================== Figure 1
def figure1():
    d = panel()
    d["stratum"] = np.where(d["area_2"].map(NFC) == "농촌", "Rural", "Urban")
    g = (d.groupby(["year", "stratum"], as_index=False)
           .agg(cases=("cases", "sum"), pop=("population", "sum")))
    g["rate"] = g["cases"] / g["pop"] * 1e5
    o = (d.groupby("year", as_index=False)
           .agg(cases=("cases", "sum"), pop=("population", "sum")))
    o["rate"] = o["cases"] / o["pop"] * 1e5

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=DPI)
    for name, col in (("Rural", C_RURAL), ("Urban", C_URBAN)):
        s = g[g["stratum"] == name].sort_values("year")
        ax.plot(s["year"], s["rate"], "-o", color=col, lw=1.8, ms=4.6,
                label=name, zorder=3)
    ax.plot(o["year"], o["rate"], "--", color=GREY, lw=1.3, label="Overall", zorder=2)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("Crude incidence per 100,000 population", fontsize=9)
    ax.set_xticks(sorted(o["year"].unique()))
    ax.tick_params(labelsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e6e8ea", lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
    fig.tight_layout()
    save(fig, "Figure1_trend")


# ================================================================== Figure 2
def figure2():
    import geopandas as gpd

    g = gpd.read_file(find("final.shp")).set_crs(5179, allow_override=True)
    g["region"] = g["region"].map(NFC)
    d = pd.read_csv(find("BC_district_incidence.csv"))
    for c in ("region", "sido", "sigungu"):
        d[c] = d[c].map(NFC)
    m = g.merge(d[["region", "sido", "sigungu", "cases_total", "pop_mean",
                   "rate_per100k"]], on="region", how="left", validate="one_to_one")
    assert len(m) == 229 and m["rate_per100k"].notna().all()

    BREAKS = [0.10, 0.25, 0.50, 1.00]
    LABELS = ["0 (no notified cases)", ">0 – 0.10", "0.10 – 0.25",
              "0.25 – 0.50", "0.50 – 1.00", "> 1.00"]
    COLORS = ["#e9edef", "#fee6a0", "#fdc353", "#f79521", "#dd5b13", "#9c3204"]

    def klass(v):
        if v <= 0:
            return 0
        for i, b in enumerate(BREAKS):
            if v <= b:
                return i + 1
        return len(BREAKS) + 1

    m["cls"] = m["rate_per100k"].map(klass)
    m["fill"] = m["cls"].map(lambda i: COLORS[i])
    counts = m["cls"].value_counts().reindex(range(6), fill_value=0)
    sido = m.dissolve(by="sido")

    CAP = m[m["sido"].isin(["서울시", "인천시", "경기도"])]
    sgx0, sgy0, sgx1, sgy1 = m[m["sido"].isin(["서울시", "경기도"])].total_bounds
    CX0, CX1 = sgx0 - 30000, sgx1 + 6000
    CY0, CY1 = sgy0 - 6000, sgy1 + 6000

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

    fig = plt.figure(figsize=(7.2, 4.6), dpi=DPI)
    fig.patch.set_facecolor("white")

    ax = fig.add_axes([0.005, 0.02, 0.60, 0.96])
    paint(ax, m)
    sido.boundary.plot(ax=ax, color="#6b6b6b", linewidth=0.5, zorder=3)
    ax.set_aspect("equal"); ax.set_axis_off()
    xmin, ymin, xmax, ymax = m.total_bounds
    px, py = (xmax - xmin) * 0.035, (ymax - ymin) * 0.035
    ax.set_xlim(xmin - px, xmax + px); ax.set_ylim(ymin - py, ymax + py)
    north_arrow(ax, 0.845, 0.925)
    scalebar(ax, 0.035, 0.055, 100)

    for _, r in m.nlargest(2, "rate_per100k").reset_index(drop=True).iterrows():
        if r["rate_per100k"] > 12:
            eng, dx, dy = "Gokseong-gun", -152000, -138000
        else:
            eng, dx, dy = "Hoengseong-gun", 238000, 38000
        c = r.geometry.representative_point()
        ax.annotate(f"{eng}\n{r['rate_per100k']:.1f}", xy=(c.x, c.y),
                    xytext=(c.x + dx, c.y + dy), fontsize=6.2, ha="center",
                    va="center", color="#2b2b2b", linespacing=1.3,
                    arrowprops=dict(arrowstyle="-", color="#5c5c5c", lw=0.55,
                                    shrinkA=0, shrinkB=1.5), zorder=7)

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
             "229 districts; 1,223 notified cases, 2011–2024.\n"
             "Classes use rounded cut points; districts with no\n"
             "notified case form a separate class.\n"
             "Projection: Korea 2000 / Unified CS (EPSG:5179).",
             fontsize=5.9, color="#4a4a4a", va="top", ha="left", linespacing=1.6)
    save(fig, "Figure2_incidence_map")
    print("      classes:", dict(counts))


# ================================================================== Figure 3
def figure3():
    t = sheet("Table2_M2_revised.xlsx", "Table2_revised")
    t = t[t["Covariate"].notna()].reset_index(drop=True)
    labels = t["Covariate"].tolist()
    groups = []
    for col, name, colour in (("Overall IRR (95% CrI)", "Overall", C_OVERALL),
                              ("Urban IRR (95% CrI)", "Urban", C_URBAN),
                              ("Rural IRR (95% CrI)", "Rural", C_RURAL)):
        groups.append((name, colour, [parse_est(v) for v in t[col]]))

    n = len(labels)
    fig, ax = plt.subplots(figsize=(7.2, 0.40 * n + 1.9), dpi=DPI)
    forest_axes(ax, labels, groups, (0.18, 6.2),
                "Adjusted incidence-rate ratio per 1 SD (95% credible interval, log scale)")

    # domain brackets on the right
    dom = t["Domain"].ffill()
    for name in dom.dropna().unique():
        idx = np.where(dom.values == name)[0]
        y0, y1 = n - 1 - idx.max(), n - 1 - idx.min()
        ax.text(1.012, (y0 + y1) / 2, name, transform=ax.get_yaxis_transform(),
                fontsize=7.6, color=GREY, va="center", ha="left")
        if idx.min() > 0:
            ax.axhline(y1 + 0.5, color="#d4d7da", lw=0.7, zorder=1)

    hs = [plt.Line2D([], [], color=c, marker="o", ms=5.6, lw=1.5, label=l)
          for l, c in (("Overall", C_OVERALL), ("Urban", C_URBAN), ("Rural", C_RURAL))]
    hs.append(plt.Line2D([], [], color=GREY, marker="o", ms=4.4, mfc="white",
                         lw=1.0, label="Credible interval includes 1"))
    bot = 1.35 / fig.get_figheight()
    fig.subplots_adjust(left=0.30, right=0.80, top=0.985, bottom=bot)
    fig.legend(handles=hs, frameon=False, fontsize=8, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, 0.008))
    save(fig, "Figure3_forest_M2")


# ================================================================== Figure 4
def figure4():
    t = sheet("TableS8_interaction_M2.xlsx", "TableS8_interaction")
    t = t[t["Covariate"].notna()].reset_index(drop=True)
    col = [c for c in t.columns if c.startswith("Interaction IRR")][0]
    vals = [parse_est(v) for v in t[col]]
    labels = t["Covariate"].tolist()
    n = len(labels)

    fig, ax = plt.subplots(figsize=(7.6, 0.40 * n + 1.7), dpi=DPI)
    forest_axes(ax, labels, [("", C_OVERALL, vals)], (0.12, 7.5),
                "Interaction incidence-rate ratio, urban relative to rural "
                "(95% CrI, log scale)")
    for i, (e, lo, hi, sig) in enumerate(vals):
        if sig and np.isfinite(e):
            ax.text(1.015, n - 1 - i, f"{e:.2f} ({lo:.2f} to {hi:.2f})",
                    transform=ax.get_yaxis_transform(), fontsize=7.6,
                    color=C_OVERALL, va="center", ha="left", fontweight="bold")
    ax.text(0.02, 1.006, "← stronger in rural", transform=ax.transAxes,
            fontsize=7.8, style="italic", color=GREY, ha="left")
    ax.text(0.72, 1.006, "stronger in urban →", transform=ax.transAxes,
            fontsize=7.8, style="italic", color=GREY, ha="left")
    hs = [plt.Line2D([], [], color=C_OVERALL, marker="o", ms=5.6, lw=1.5,
                     label="Urban and rural effects differ (CrI excludes 1)"),
          plt.Line2D([], [], color=C_OVERALL, marker="o", ms=4.4, mfc="white",
                     lw=1.0, label="No evidence of a difference")]
    bot = 1.25 / fig.get_figheight()
    fig.subplots_adjust(left=0.27, right=0.74, top=0.955, bottom=bot)
    fig.legend(handles=hs, frameon=False, fontsize=8, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 0.008))
    save(fig, "Figure4_interaction_M2")


# ========================================================= Supplementary S1
def figure_s1():
    """Study-flow diagram. The exclusion box sits in its own column to the
    right of the main trunk, so the vertical arrow never crosses it."""
    MAIN_X, MAIN_W = 4.0, 7.2          # main column: 0.40 .. 7.60
    EXC_X, EXC_W = 11.05, 6.2          # exclusion column: 7.95 .. 14.15
    H, H_EXC = 1.35, 1.65

    main = [
        (8.75, "All districts of the Republic of Korea, 2011\u20132024\n"
               "229 districts \u00d7 14 years = 3,206 district-years\n"
               "1,223 notified cases", "#eef3f8"),
        (6.20, "Descriptive analysis\n229 districts, 3,206 district-years\n"
               "(Table 1, Figures 1 and 2)", "#f7f7f7"),
        (1.70, "Analytic sample for all models\n223 districts, 3,122 district-years\n"
               "Urban 146 (2,044)   Rural 77 (1,078)", "#eaf4ee"),
    ]
    exc_y = 3.95
    exc = ("Excluded before modelling (Queen contiguity)\n"
           "6 districts sharing no land boundary with another district\n"
           "Ongjin, Wando, Jindo, Geoje, Namhae, Ulleung\n"
           "33 cases (2.7%) in 84 district-years")

    fig, ax = plt.subplots(figsize=(9.6, 5.9), dpi=DPI)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.985, bottom=0.015)
    ax.set_xlim(0.05, 14.55); ax.set_ylim(0.70, 9.75); ax.axis("off")

    def box(x, y, w, h, txt, fc, ec="#5c6b78"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.10,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=0.9, zorder=3))
        ax.text(x, y, txt, ha="center", va="center", fontsize=8.4,
                linespacing=1.55, zorder=4)

    for y, txt, fc in main:
        box(MAIN_X, y, MAIN_W, H, txt, fc)
    box(EXC_X, exc_y, EXC_W, H_EXC, exc, "#fdf1ea", ec="#a05a34")

    def arrow(x0, y0, x1, y1, col="#5c6b78", lw=1.1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.45",
                                    color=col, lw=lw, shrinkA=0, shrinkB=0,
                                    mutation_scale=14))

    arrow(MAIN_X, main[0][0] - H / 2, MAIN_X, main[1][0] + H / 2)
    arrow(MAIN_X, main[1][0] - H / 2, MAIN_X, main[2][0] + H / 2)
    arrow(MAIN_X, exc_y, EXC_X - EXC_W / 2, exc_y, col="#a05a34", lw=1.0)

    save(fig, "FigureS1_study_flow")


# ========================================================= Supplementary S2
def figure_s2():
    d = panel()
    d["stratum"] = np.where(d["area_2"].map(NFC) == "\ub18d\ucd0c", "Rural", "Urban")
    mth = DATA / "monthly_cases.csv"
    mor = DATA / "moran.csv"

    # the 21 pre-specified covariates, English labels, manuscript order
    ORDER = ["Sex ratio", "Elderly-population rate", "Urban-area population rate",
             "Piped-water coverage", "Sewerage coverage", "Advanced-treatment coverage",
             "Septic-tank population", "Household groundwater wells",
             "Civil-defence groundwater use", "Reservoir area", "Stream and river area",
             "Pasture area", "Paddy-field area", "Forest area", "Physician count",
             "In-area medical expenditure", "Influenza-vaccination uptake",
             "Healthy-lifestyle practice rate", "Walking-practice rate",
             "Fiscal independence", "Fiscal autonomy"]
    col_of = {}
    for c in d.columns:
        lab = en(c)
        if lab in ORDER and lab not in col_of and pd.api.types.is_numeric_dtype(d[c]):
            col_of[lab] = c
    covs = [lab for lab in ORDER if lab in col_of]

    # landscape layout, so the figure fills a landscape supplementary page
    fig = plt.figure(figsize=(13.6, 7.4), dpi=DPI)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.32, 1.0],
                          hspace=1.05, wspace=0.28,
                          left=0.055, right=0.985, top=0.955, bottom=0.07)

    # (a) monthly seasonality, (b) year x month heatmap
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    if mth.exists():
        mm = pd.read_csv(mth)
        sM = mm.groupby("month", as_index=False)["cases"].sum()
        warm = sM["month"].isin([7, 8, 9, 10])
        axa.bar(sM.loc[warm, "month"], sM.loc[warm, "cases"], color="#dd5b13",
                edgecolor="#8f3c08", lw=0.5, label="July to October")
        axa.bar(sM.loc[~warm, "month"], sM.loc[~warm, "cases"], color="#fdc353",
                edgecolor="#b8650f", lw=0.5, label="other months")
        tot = sM["cases"].sum()
        axa.set_xticks(range(1, 13))
        axa.set_xlabel("Month", fontsize=8)
        axa.set_ylabel("Notified cases", fontsize=8)
        axa.legend(frameon=False, fontsize=7.4, loc="upper left")
        axa.text(0.985, 0.97,
                 f"July to October {sM.loc[warm,'cases'].sum():,.0f} "
                 f"({sM.loc[warm,'cases'].sum()/tot*100:.1f}%)\n"
                 f"December {sM.loc[sM['month']==12,'cases'].iloc[0]:,.0f} "
                 f"({sM.loc[sM['month']==12,'cases'].iloc[0]/tot*100:.1f}%)",
                 transform=axa.transAxes, fontsize=7, va="top", ha="right",
                 linespacing=1.5)
        piv = mm.pivot_table(index="year", columns="month", values="cases",
                             aggfunc="sum").reindex(columns=range(1, 13)).fillna(0)
        im = axb.imshow(piv.values, aspect="auto", cmap="YlOrBr",
                        extent=[0.5, 12.5, piv.index.max() + 0.5, piv.index.min() - 0.5])
        axb.set_xticks(range(1, 13)); axb.set_yticks(piv.index)
        axb.tick_params(labelsize=7.5)
        axb.set_xlabel("Month", fontsize=8); axb.set_ylabel("Year", fontsize=8)
        fig.colorbar(im, ax=axb, shrink=0.85, label="Cases")
    else:
        for ax_, t in ((axa, "(a)"), (axb, "(b)")):
            ax_.text(.5, .5, f"{t} monthly_cases.csv not found", ha="center",
                     va="center", fontsize=8, color="#a33"); ax_.axis("off")
    axa.set_title("(a) Monthly seasonality", fontsize=9, loc="left")
    axb.set_title("(b) Year by month, notified cases", fontsize=9, loc="left")

    # (c) standardised covariate distributions by stratum
    axc = fig.add_subplot(gs[1, :])
    for i, lab in enumerate(covs):
        c = col_of[lab]
        v_all = pd.to_numeric(d[c], errors="coerce")
        z = (v_all - v_all.mean()) / v_all.std(ddof=0)
        for j, (name, colr) in enumerate((("Urban", C_URBAN), ("Rural", C_RURAL))):
            v = z[d["stratum"] == name].dropna()
            if len(v):
                bp = axc.boxplot([v], positions=[i + (j - 0.5) * 0.34], widths=0.28,
                                 patch_artist=True, showfliers=False,
                                 medianprops=dict(color="#333", lw=0.9))
                bp["boxes"][0].set(facecolor=colr, alpha=0.55, lw=0.6, edgecolor=colr)
    axc.set_xticks(range(len(covs)))
    axc.set_xticklabels(covs, rotation=35, ha="right", fontsize=6.9)
    axc.set_xlim(-0.7, len(covs) - 0.3)
    axc.set_ylabel("Standardised value (z)", fontsize=8)
    axc.axhline(0, color="#bbb", lw=0.7)
    axc.set_title("(c) Standardised covariate distributions by stratum "
                  "(21 pre-specified covariates)", fontsize=9, loc="left")
    axc.legend(handles=[Patch(fc=C_URBAN, alpha=.55, label="Urban"),
                        Patch(fc=C_RURAL, alpha=.55, label="Rural")],
               frameon=False, fontsize=8, ncol=2, loc="upper right")

    # (d) raw vs log1p, (e) cut points — household groundwater wells (log1p in the model)
    pick = "Household groundwater wells" if "Household groundwater wells" in covs else covs[0]
    v = pd.to_numeric(d[col_of[pick]], errors="coerce").dropna()

    axd = fig.add_subplot(gs[2, 0])
    axd.hist(v, bins=40, color="#9ecae1", edgecolor="#4a7fa5", lw=0.4)
    axd.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    axd.tick_params(labelsize=7.6)
    axd.set_xlabel(f"{pick}, raw", fontsize=8)
    axd.set_ylabel("District-years", fontsize=8)
    axd.set_title("(d) Raw versus log1p transformation", fontsize=9, loc="left")
    axd_in = axd.inset_axes([0.46, 0.46, 0.5, 0.48])
    axd_in.hist(np.log1p(v), bins=40, color="#f79521", edgecolor="#b8650f", lw=0.4)
    axd_in.set_title("log1p", fontsize=7.2, color="#b8650f", pad=2)
    axd_in.tick_params(labelsize=6.5)

    axe = fig.add_subplot(gs[2, 1])
    axe.hist(v, bins=40, color="#d9d9d9", edgecolor="#9b9b9b", lw=0.4)
    # cut points are close together in a skewed covariate, so the labels are
    # stacked at descending heights instead of all sitting at the top
    cuts = [(0.25, "--", "Q1"), (1 / 3, ":", "T1"), (0.5, "-", "median"),
            (2 / 3, ":", "T2"), (0.75, "--", "Q3")]
    top = axe.get_ylim()[1]
    for k, (q, ls, lab) in enumerate(cuts):
        xq = v.quantile(q)
        axe.axvline(xq, color="#c0392b", lw=1.0, ls=ls)
        axe.annotate(f"{lab} {xq:,.0f}", xy=(xq, top * (0.93 - 0.135 * k)),
                     xytext=(6, 0), textcoords="offset points",
                     fontsize=6.6, color="#c0392b", ha="left", va="center")
    axe.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    axe.tick_params(labelsize=7.6)
    axe.set_xlabel(pick, fontsize=8)
    axe.set_ylabel("District-years", fontsize=8)
    axe.set_title("(e) Categorical transformation cut points", fontsize=9, loc="left")

    # (f) district incidence by stratum
    axf = fig.add_subplot(gs[2, 2])
    r = (d.groupby(["region", "stratum"], as_index=False)
           .agg(cases=("cases", "sum"), pop=("population", "mean")))
    r["rate"] = r["cases"] / (r["pop"] * d["year"].nunique()) * 1e5
    bins = np.linspace(0, float(np.ceil(r["rate"].max())), 41)
    for name, colr in (("Urban", C_URBAN), ("Rural", C_RURAL)):
        axf.hist(r.loc[r["stratum"] == name, "rate"], bins=bins, alpha=0.65,
                 color=colr, label=name, edgecolor="white", lw=0.4)
    axf.set_yscale("symlog", linthresh=1)
    axf.set_xlabel("Mean annual crude incidence per 100,000", fontsize=8)
    axf.set_ylabel("Districts (log scale)", fontsize=8)
    axf.legend(frameon=False, fontsize=8)
    axf.set_title("(f) District incidence by stratum", fontsize=9, loc="left")

    # (g) Moran scatterplot
    axg = fig.add_subplot(gs[0, 2])
    if mor.exists():
        mo = pd.read_csv(mor)
        x = (mo["rate"] - mo["rate"].mean()) / mo["rate"].std(ddof=0)
        y = (mo["lag"] - mo["lag"].mean()) / mo["lag"].std(ddof=0)
        axg.scatter(x, y, s=12, color=C_OVERALL, alpha=0.65, lw=0)
        b = np.polyfit(x, y, 1)[0]
        xx = np.linspace(x.min(), x.max(), 10)
        axg.plot(xx, b * xx, color="#c0392b", lw=1.2)
        axg.axhline(0, color="#bbb", lw=0.7); axg.axvline(0, color="#bbb", lw=0.7)
        pcol = "p_analytic" if "p_analytic" in mo.columns else (
            "p" if "p" in mo.columns else None)
        txt = (f"Moran's I = {mo['I'].iloc[0]:.3f}" +
               (f", p = {mo[pcol].iloc[0]:.2f}" if pcol else ""))
        axg.text(0.04, 0.95, txt, transform=axg.transAxes, fontsize=8, va="top")
        axg.set_xlabel("Standardised incidence", fontsize=8)
        axg.set_ylabel("Spatial lag", fontsize=8)
    else:
        axg.text(.5, .5, "moran.csv not found", ha="center", va="center",
                 fontsize=8, color="#a33"); axg.axis("off")
    axg.set_title("(g) Moran scatterplot (223 contiguous districts)",
                  fontsize=9, loc="left")

    save(fig, "FigureS2_descriptive_panels")


# ========================================================= Supplementary S3
def figure_s3():
    fig = plt.figure(figsize=(13.4, 9.4), dpi=DPI)
    gs = fig.add_gridspec(3, 3, hspace=0.46, wspace=0.55,
                          left=0.055, right=0.985, top=0.945, bottom=0.065,
                          width_ratios=[1.0, 0.90, 1.22])

    # (a) DIC by likelihood and spatial structure
    axa = fig.add_subplot(gs[0, 0:2])
    sp = sheet("REV1_M2_공간비교_260915_2144.xlsx", "60_spatial_compare_M2")
    order = ["NonSpatial", "ICAR", "BYM"]
    w = 0.36
    for k, (fam, col) in enumerate((("ZINB", C_OVERALL), ("NB", "#b0b7bd"))):
        s = sp[sp["likelihood"] == fam].set_index("structure").reindex(order)
        axa.bar(np.arange(3) + (k - 0.5) * w, s["DIC"], width=w, color=col,
                edgecolor="white", lw=0.6, label=fam)
        for i, v in enumerate(s["DIC"]):
            axa.text(i + (k - 0.5) * w, v + 3, f"{v:,.1f}", ha="center",
                     fontsize=7.4, color="#333")
    axa.set_xticks(range(3)); axa.set_xticklabels(order, fontsize=8.5)
    axa.set_ylabel("Deviance information criterion", fontsize=8.5)
    axa.set_ylim(1400, sp["DIC"].max() * 1.02)
    axa.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")
    axa.set_title("(a) Model comparison by DIC (19-covariate specification)",
                  fontsize=9, loc="left")

    # (b) strata map, (c) standardised incidence-ratio map
    try:
        import geopandas as gpd
        g = gpd.read_file(find("final.shp")).set_crs(5179, allow_override=True)
        g["region"] = g["region"].map(NFC)
        d = panel()
        cls = (d.groupby("region")["area_2"].first().map(NFC)
                 .map({"농촌": "Rural"}).fillna("Urban"))
        agg = d.groupby("region").agg(cases=("cases", "sum"),
                                      pop=("population", "sum"))
        exp = agg["pop"] * (agg["cases"].sum() / agg["pop"].sum())
        agg["sir"] = agg["cases"] / exp
        g = g.merge(cls.rename("stratum"), left_on="region", right_index=True, how="left")
        g = g.merge(agg["sir"], left_on="region", right_index=True, how="left")

        axb = fig.add_subplot(gs[1, 0])
        g.assign(c=g["stratum"].map({"Urban": C_URBAN, "Rural": C_RURAL})).plot(
            ax=axb, color=g["stratum"].map({"Urban": C_URBAN, "Rural": C_RURAL}),
            edgecolor="white", lw=0.2)
        axb.set_axis_off(); axb.set_aspect("equal")
        axb.legend(handles=[Patch(fc=C_URBAN, label="Urban"),
                            Patch(fc=C_RURAL, label="Rural")],
                   frameon=False, fontsize=8, loc="upper right")
        axb.set_title("(b) Urban and rural strata", fontsize=9, loc="left")

        axc = fig.add_subplot(gs[1, 1])
        g.plot(ax=axc, column="sir", cmap="RdYlBu_r", edgecolor="white", lw=0.2,
               legend=True, vmin=0, vmax=min(4, np.nanpercentile(g["sir"], 98)),
               legend_kwds={"shrink": 0.55, "label": "SIR",
                            "pad": 0.01, "aspect": 18})
        axc.set_axis_off(); axc.set_aspect("equal")
        axc.set_title("(c) Standardised incidence ratio", fontsize=9, loc="left")
    except Exception as e:                                   # noqa: BLE001
        for k, t in ((gs[1, 0], "(b)"), (gs[1, 1], "(c)")):
            a = fig.add_subplot(k)
            a.text(.5, .5, f"{t} skipped: {e}", ha="center", va="center",
                   fontsize=7.5, color="#a33", wrap=True); a.axis("off")

    # (d) period sweep, (e) calibration summary
    axd = fig.add_subplot(gs[2, 0])
    sw = DATA / "period_sweep.csv"
    if sw.exists():
        s = pd.read_csv(sw)
        axd.plot(s["start"], s["DIC"], "-o", color=C_OVERALL, lw=1.6, ms=4.6)
        axd.set_xlabel("Start year of the analytic window", fontsize=8)
        axd.set_ylabel("DIC", fontsize=8)
    else:
        axd.text(.5, .5, "period_sweep.csv not found\n(Supplementary Table S9)",
                 ha="center", va="center", fontsize=8, color="#a33"); axd.axis("off")
    axd.set_title("(d) Period sweep", fontsize=9, loc="left")

    axe = fig.add_subplot(gs[2, 1])
    cal = sheet("REV1_M2_적합도_260915_2138.xlsx", "50_fit_calibration_M2")
    st = ["Overall", "Urban", "Rural"]
    for k, (fam, col) in enumerate((("ZINB", C_OVERALL), ("NB", "#b0b7bd"))):
        s = cal[cal["likelihood"] == fam].set_index("stratum").reindex(st)
        axe.bar(np.arange(3) + (k - 0.5) * 0.36, s["mean_PIT_mid"], width=0.36,
                color=col, edgecolor="white", lw=0.6, label=fam)
    axe.axhline(0.5, color="#c0392b", lw=1.0, ls="--")
    axe.set_xticks(range(3)); axe.set_xticklabels(st, fontsize=8.5)
    axe.set_ylim(0.45, 0.52); axe.set_ylabel("Mean mid-PIT", fontsize=8.5)
    axe.legend(frameon=False, fontsize=8.5, ncol=2)
    axe.set_title("(e) Posterior calibration (mid-PIT, 0.5 = calibrated)",
                  fontsize=9, loc="left")

    # (f) sensitivity excluding 2020  [original 21-covariate specification]
    axf = fig.add_subplot(gs[2, 2])
    try:
        full = sheet("REV1_추가분석_260915_1551.xlsx", "01_main_IRR_재현")
        full = full[full["stratum"] == "Overall"][["var_kr", "IRR"]].copy()
        full["var_kr"] = full["var_kr"].map(NFC)
        cov = sheet_any("B_cereus_SUPP_2011to2024_260620_2016.xlsx", "S5_COVIDexcluded")
        cov = cov.rename(columns={"Variable": "var_kr", "IRR": "IRR_ex"})
        cov["var_kr"] = cov["var_kr"].map(NFC)
        m = full.merge(cov[["var_kr", "IRR_ex", "Sig"]], on="var_kr", how="inner")
        sg = m["Sig"].notna()
        lim = (0.35, 3.2)
        axf.plot(lim, lim, color="#999999", lw=0.9, ls=(0, (4, 3)), zorder=1)
        axf.axvline(1, color="#dddddd", lw=0.8, zorder=0)
        axf.axhline(1, color="#dddddd", lw=0.8, zorder=0)
        axf.scatter(m.loc[~sg, "IRR"], m.loc[~sg, "IRR_ex"], s=30, zorder=3,
                    facecolor="white", edgecolor=GREY, lw=1.1,
                    label="95% CrI includes 1")
        axf.scatter(m.loc[sg, "IRR"], m.loc[sg, "IRR_ex"], s=34, zorder=4,
                    color=C_OVERALL, label="95% CrI excludes 1")
        axf.set_xscale("log"); axf.set_yscale("log")
        axf.set_xlim(*lim); axf.set_ylim(*lim)
        tk = [0.5, 1, 2, 3]
        axf.set_xticks(tk); axf.set_yticks(tk)
        for ax_ in (axf.get_xaxis(), axf.get_yaxis()):
            ax_.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axf.minorticks_off()
        axf.tick_params(labelsize=8)
        axf.set_xlabel("IRR, full period 2011 to 2024", fontsize=8.5)
        axf.set_ylabel("IRR, 2020 excluded", fontsize=8.5)
        axf.legend(frameon=False, fontsize=7.4, loc="upper left")
        axf.set_aspect("equal")
        for s in ("top", "right"):
            axf.spines[s].set_visible(False)
    except Exception as e:                                   # noqa: BLE001
        axf.text(.5, .5, f"(f) skipped: {e}", ha="center", va="center",
                 fontsize=7.5, color="#a33", wrap=True); axf.axis("off")
    axf.set_title("(f) Sensitivity excluding 2020\n"
                  "     (original 21-covariate specification)",
                  fontsize=9, loc="left")

    # (g) food-pathway model, 2020 to 2024
    axg = fig.add_subplot(gs[0:2, 2])
    try:
        raw = sheet_any("B_cereus_SUPP_2011to2024_260620_2016.xlsx",
                    "S7_FoodSensitivity_2020_2024")
        hdr = [NFC(v) for v in raw.iloc[1].tolist()]
        fp = raw.iloc[2:].copy(); fp.columns = hdr
        fp = fp[fp["var_kr"].notna()]
        vals, labels = [], []
        for _, r in fp.iterrows():
            e, lo, hi = (float(r["IRR"]), float(r["lo"]), float(r["hi"]))
            vals.append((e, lo, hi, isinstance(r["sig"], str) and "★" in r["sig"]))
            labels.append(en(r["var_kr"]))
        forest_axes(axg, labels, [("Food pathway", C_OVERALL, vals)],
                    (0.02, 70),
                    "Incidence-rate ratio (95% CrI, log scale)")
        tk = [0.05, 0.25, 1, 4, 20]
        axg.set_xticks(tk)
        axg.minorticks_off()
        axg.tick_params(axis="y", labelsize=7.6)
        axg.get_xaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axg.legend(handles=[
            plt.Line2D([], [], color=C_OVERALL, marker="o", ms=5.6, lw=1.5,
                       label="95% CrI excludes 1"),
            plt.Line2D([], [], color=C_OVERALL, marker="o", ms=4.4, lw=1.0,
                       mfc="white", mew=1.2, label="95% CrI includes 1")],
            frameon=False, fontsize=7.4, loc="upper right")
    except Exception as e:                                   # noqa: BLE001
        axg.text(.5, .5, f"(g) skipped: {e}", ha="center", va="center",
                 fontsize=7.5, color="#a33", wrap=True); axg.axis("off")
    axg.set_title("(g) Food-pathway model, 2020 to 2024\n"
                  "     (livestock and shellfish covariates added)",
                  fontsize=9, loc="left")

    save(fig, "FigureS3_diagnostic_panels")


# ========================================================= Supplementary S4
def figure_s4():
    a = sheet("REV1_추가분석_260915_1551.xlsx", "07_missing_audit").fillna(0)
    a = a[a["variable"].notna()].copy()
    a["label"] = a["variable"].map(en)
    a = a[a["variable"].map(lambda v: NFC(v) in KO2EN)]          # the 21 pre-specified
    if a.empty:
        raise FileNotFoundError("07_missing_audit has none of the 21 pre-specified covariates")
    stat_cols = [c for c in ("observed", "interpolated", "carried_forward_backward",
                             "recoded_zero_year_to_NA", "missing") if c in a.columns]
    tot = a.groupby("label")[stat_cols].sum()
    frac = tot.div(tot.sum(axis=1), axis=0) * 100
    frac = frac.sort_values("observed")

    fig, ax = plt.subplots(figsize=(8.4, 0.34 * len(frac) + 1.9), dpi=DPI)
    cols = {"observed": "#9ecae1", "interpolated": "#f79521",
            "carried_forward_backward": "#fdc353",
            "recoded_zero_year_to_NA": "#dd5b13", "missing": "#c0392b"}
    names = {"observed": "observed", "interpolated": "interpolated",
             "carried_forward_backward": "carried forward or backward",
             "recoded_zero_year_to_NA": "zero recoded to missing",
             "missing": "missing"}
    left = np.zeros(len(frac))
    for c in stat_cols:
        ax.barh(range(len(frac)), frac[c], left=left, height=0.72,
                color=cols[c], edgecolor="white", lw=0.4, label=names[c])
        left += frac[c].values
    ax.set_yticks(range(len(frac)))
    ax.set_yticklabels(frac.index, fontsize=7.6)
    ax.set_xlabel("Percentage of district-years, 2011\u20132024 (229 districts)", fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(frac) - 0.3)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    bot = 1.05 / fig.get_figheight()
    fig.subplots_adjust(left=0.30, right=0.98, top=0.985, bottom=bot)
    fig.legend(frameon=False, fontsize=7.8, ncol=3, loc="lower center",
               bbox_to_anchor=(0.62, 0.005))
    save(fig, "FigureS4_missing_interpolation")


# ========================================================= Supplementary S5
def figure_s5():
    h = sheet("REV1_M2_적합도_260915_2138.xlsx", "51_PIT_hist_M2")
    cal = sheet("REV1_M2_적합도_260915_2138.xlsx", "50_fit_calibration_M2")
    st = ["Overall", "Urban", "Rural"]
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.6), dpi=DPI, sharey="row")
    for j, s in enumerate(st):
        for i, (fam, col) in enumerate((("ZINB", C_OVERALL), ("NB", "#b0b7bd"))):
            ax = axes[i, j]
            b = h[(h["stratum"] == s) & (h["likelihood"] == fam)]
            ax.bar(b["lower"], b["height"], width=(b["upper"] - b["lower"]),
                   align="edge", color=col, edgecolor="white", lw=0.4)
            ax.axhline(1, color="#c0392b", lw=1.0, ls="--")
            ax.set_xlim(0, 1); ax.set_ylim(0, max(1.35, b["height"].max() * 1.12))
            ax.tick_params(labelsize=7.5)
            r = cal[(cal["stratum"] == s) & (cal["likelihood"] == fam)]
            if len(r):
                ax.text(0.03, 0.94,
                        f"mid-PIT {r['mean_PIT_mid'].iloc[0]:.3f}\n"
                        f"max dev {r['hist_max_dev_from_1'].iloc[0]:.3f}\n"
                        f"KS p {r['ks_p_randomised'].iloc[0]:.2f}",
                        transform=ax.transAxes, fontsize=7, va="top", linespacing=1.4)
            if i == 0:
                ax.set_title(s, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{fam}\nrelative frequency", fontsize=8.5)
            if i == 1:
                ax.set_xlabel("Non-randomised PIT", fontsize=8.5)
    fig.suptitle("Posterior predictive calibration: non-randomised PIT histograms "
                 "(uniform = well calibrated)", fontsize=9.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, "FigureS5_PIT_calibration")


# ===================================================================== main
FIGS = {
    "1": ("Figure 1  annual trend", figure1,
          ["figures/_data/panel_district_year.csv"]),
    "2": ("Figure 2  incidence map", figure2,
          ["final.shp", "BC_district_incidence.csv"]),
    "3": ("Figure 3  forest (M2)", figure3, ["Table2_M2_revised.xlsx"]),
    "4": ("Figure 4  interaction", figure4, ["TableS8_interaction_M2.xlsx"]),
    "S1": ("Figure S1 study flow", figure_s1, ["(constants only)"]),
    "S2": ("Figure S2 descriptive panels", figure_s2,
           ["figures/_data/panel_district_year.csv",
            "figures/_data/monthly_cases.csv", "figures/_data/moran.csv"]),
    "S3": ("Figure S3 diagnostic panels", figure_s3,
           ["REV1_M2_공간비교_*.xlsx", "REV1_M2_적합도_*.xlsx",
            "REV1_추가분석_*.xlsx", "B_cereus_SUPP_2011to2024_260620_2016.xlsx",
            "final.shp", "figures/_data/panel_district_year.csv"]),
    "S4": ("Figure S4 missing and interpolation", figure_s4,
           ["REV1_추가분석_*.xlsx"]),
    "S5": ("Figure S5 PIT calibration", figure_s5, ["REV1_M2_적합도_*.xlsx"]),
}


def main() -> int:
    global ROOT, OUT, DATA, _INDEX

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset, e.g. --only 2 3 4 S5")
    ap.add_argument("--root", default=None,
                    help="path to the B_cereus folder (default: the path set at "
                         "the top of this file)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for k, (name, _, needs) in FIGS.items():
            print(f"{k:>3}  {name}\n       needs: {', '.join(needs)}")
        return 0

    if a.root:
        ROOT = Path(a.root)
    OUT = ROOT / "figures"
    DATA = OUT / "_data"
    _INDEX = None

    if not ROOT.is_dir():
        print(f"ROOT not found: {ROOT}", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"ROOT {ROOT}\nOUT  {OUT}\n")

    keys = a.only or list(FIGS)
    done, skipped = [], []
    for k in keys:
        if k not in FIGS:
            print(f"  ?  unknown figure key {k!r}")
            continue
        name, fn, needs = FIGS[k]
        print(f"[{k}] {name}")
        try:
            fn()
            done.append(k)
        except FileNotFoundError as e:
            print(f"      SKIPPED, missing input: {e}")
            print(f"      needs: {', '.join(needs)}")
            skipped.append(k)
        except Exception as e:                               # noqa: BLE001
            print(f"      FAILED: {type(e).__name__}: {e}")
            skipped.append(k)
    print(f"\ndone {len(done)}: {' '.join(done)}")
    if skipped:
        print(f"skipped/failed {len(skipped)}: {' '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
