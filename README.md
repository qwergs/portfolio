# Data Science Portfolio — Geonsang Lee (이건상)

시계열·이상탐지·의료/헬스케어 데이터를 중심으로 한 데이터 사이언스 포트폴리오입니다.
6개 프로젝트와 제1저자 국제 저널 게재 논문(재현 패키지 포함)으로 구성됩니다.

> **Note on data & outputs** — 대회/기관 제공 원본 데이터(대용량 CSV·TDMS 등)와 라이선스 자료는 포함하지 않습니다.
> 노트북 출력(그래프·로그)은 용량을 위해 제거했으며, 코드는 실행 환경에 맞게 경로를 수정해 재현할 수 있습니다.

## ★ Featured Publication (First Author)

**Typhoid Fever in the Republic of Korea, 2001–2024** — *Foodborne Pathogens and Disease* (2026), Published online.
Geonsang Lee (제1저자), Seongdae Kim · Hongik University.
- Paper (DOI): https://doi.org/10.1177/15353141261465221
- Reproducibility package (data + code, CC-BY-4.0): https://doi.org/10.5281/zenodo.19810837
- KDCA 전수감시 3,535례 재분석 — Mann–Kendall · Negative Binomial GLM · Morlet wavelet · Moran's I/LISA · Empirical Bayes
- 데이터 수집·전처리부터 통계·공간 분석 설계·구현, 원고 전 섹션 집필, 재현 패키지 공개까지 전 과정 단독 수행.

## Projects

| # | Project | Domain | Key Methods |
|---|---------|--------|-------------|
| 01 | [IVF Pregnancy Prediction](./01_ivf_prediction) — LG Aimers 6th | Healthcare / Tabular ML | ExtraTrees · XGBoost · LightGBM · Optuna · probability calibration |
| 02 | [Seoul Education Data Analysis](./02_seoul_education) — 7th MOE Competition | Public Data | EDA · PCA · Ridge/Lasso/RF · Folium |
| 03 | [HIRA Healthcare Big Data](./03_hira_healthcare) | Healthcare / Claims | ICD-10 mapping · large-scale ETL (cp949) · readmission/mortality |
| 04 | [Behavioral-Finance Clustering](./04_field_finance) — FIELD | Quant Finance / Survey | DBSCAN + PCA · Markowitz / Black-Litterman / Risk Parity |
| 05 | [Bearing Anomaly Detection + Health Index](./05_bearing_anomaly) — KIMM Challenge | Industrial AI / Signal | **1D CNN AutoEncoder (PyTorch, self-built)** · TDMS · RMS/Kurtosis · HI threshold |
| 06 | [Typhoid Surveillance Epidemiology](./06_typhoid_epidemiology) — Published | Spatio-temporal Statistics | Mann–Kendall · NB GLM · Morlet wavelet · Moran's I/LISA · Empirical Bayes |

## Tech Stack

**Languages** Python · R · SQL
**ML / DL** scikit-learn · XGBoost · LightGBM · PyTorch · Optuna
**Stats / Epi** statsmodels (GLM·NB) · PyWavelets · PyMannKendall · libpysal (Moran's I) · Empirical Bayes
**Data / Signal** pandas · numpy · scipy · nptdms (TDMS) · large-scale ETL
**Viz** matplotlib · seaborn · folium · plotly

## Contact
Geonsang Lee (이건상) · Hongik University, Dept. of Industrial & Data Engineering
