# Code — Typhoid Fever in the Republic of Korea, 2001–2024 (FBPD)

This folder collects the code that produced the final results, tables, and figures
for the manuscript `../01_manuscript_FINAL_complete.docx`.

Environment: Python 3.12 (pandas, numpy, statsmodels, scipy, PyMannKendall,
matplotlib, adjustText). All Monte-Carlo procedures use `seed = 42`.
Reproducibility package (data + code): https://doi.org/10.5281/zenodo.19810837

## analysis/  — statistical pipeline (tables & numbers)
- `typhoid_r2_analyses.py` — main analysis: annual trend (Hamed–Rao MK, Sen slope),
  negative-binomial GLM, dispersion, breakpoint/BIC search  → Table 1, Table 2
- `typhoid_r2_part2.py` — periodicity (Morlet wavelet, AR(1) null) and
  demographic / age-standardization analyses
- `typhoid_r2_part3.py` — spatial analysis: global Moran's I, LISA, empirical-Bayes
  shrinkage, period-stratified analysis  → Table 3, Table 4
- `recompute_weights.py` — robustness across 12 spatial-weight schemes (Supplementary Table S3)
- `rerun_power_n1000.py` — breakpoint-detection power simulation (Supplementary Table S8)
- `typhoid_make_R2.js` — data-assembly / helper script

## figures/  — figure generation
- `build_fig_wide.py`  ← **FINAL figure code** that produced the submitted
  `Figure_1.png` / `Figure_2.png` (wide layout, all text ≥6 pt at print size,
  legends outside plots, adjustText label de-cluttering).
- `build_fig_6pt.py` — earlier 6 pt-compliant version (portrait layout).
- `rebuild_combined_figs.py` — original composite Figure 1 & 2 builder (pre-revision).
- `rebuild_figS3_step_sim.py` — Supplementary Figure S3 (breakpoint step simulation).
- `make_animated_map_v2.py`, `build_sigungu_mp4.py`, `make_html_small.py`
  — Supplementary Figure S4 (year-by-year animated / interactive maps).

## Notes
- Scripts reference data under the project `data/` and table outputs under
  `tables_new/`; adjust the DATA / TBL / FIG path constants at the top of each
  script to your local layout before running.
- The final figures were built with `figures/build_fig_wide.py`.
