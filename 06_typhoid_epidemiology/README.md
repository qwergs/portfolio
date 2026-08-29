# ★ Typhoid Fever Surveillance in the Republic of Korea, 2001–2024

> **제1저자 국제 저널 게재 논문 (First-author, Published)**
> *Foodborne Pathogens and Disease* (Mary Ann Liebert) · Manuscript FBPD-2026-0051

---

## 논문 정보 / Publication

**Title.** Typhoid fever in the Republic of Korea, 2001–2024: long-term decline, residual annual periodicity and southeastern coastal spatial clustering in 24 years of nationwide surveillance

- **Authors.** **Geonsang Lee (제1저자)**, Seongdae Kim (교신저자)
- **Affiliation.** Department of Biohealth Convergence Open Sharing System, Hongik University
- **Journal.** *Foodborne Pathogens and Disease*, 2026 — **Published online (2026-06-30)**
- **DOI.** [10.1177/15353141261465221](https://doi.org/10.1177/15353141261465221)
- **Reproducibility package (data + code).** Zenodo — https://doi.org/10.5281/zenodo.19810837 (CC-BY-4.0)
- **Reporting guideline.** STROBE · **IRB.** Hongik University 7002340-202511-HR-012

---

## 연구 개요 / Overview

한국 질병관리청(KDCA)에 2001–2024년 신고된 **장티푸스 전수 3,535례**(국내 3,051 · 해외유입 484 [13.7%])를 재분석하여, 장기 추세·연간 주기성·인구 패턴·공간 군집을 통합적으로 정량화한 24년 전수감시 역학 연구.

## 핵심 결과 / Key Findings

| 항목 | 결과 |
|---|---|
| 장기 추세 | Hamed–Rao Mann–Kendall τ = **−0.667** (p = 0.0011), Sen slope **−7.3%/yr** (95% CI −9.9 ~ −4.4) |
| 구조 변화점 | 후보 2018년 (부트스트랩 95% CI 2009–2022); 검정력 시뮬레이션 ≥25%/yr 효과 **≥98%** 검출 |
| 연간 주기성 | Morlet 웨이블릿 GWS 연간대(48–56주)가 AR(1) 적색잡음 귀무 대비 **2.60배** (ρ̂ = 0.182) |
| 공간 자기상관 | Global Moran's I = **0.403** (p = 0.0083, 9,999 permutations); LISA HH 부산·울산 |
| 강건성 | 남동해안 군집(경남·부산·울산)이 12개 공간가중·Empirical Bayes·기간층화에서 모두 보존 |
| 해외유입·내성 | 해외유입 13.7% 잔존; H58 시프로플록사신 내성주(인도 유입) 유입 이슈 연결 |

## 방법론 / Methods

Hamed–Rao modified Mann–Kendall · Sen slope · **Negative Binomial GLM (NB2, log-offset)** · Morlet continuous wavelet + 1,000-simulation AR(1) red-noise null · exhaustive BIC breakpoint search + parametric bootstrap CI · direct age-standardisation (2005 표준인구) · **Global/Local Moran's I (LISA)** + Benjamini–Hochberg FDR · Empirical Bayes 공간 축소(sigungu ≈140) · 기간층화(2001–2012 vs 2013–2024).

## 나의 역할 / My Contribution (제1저자)

데이터 수집·harmonization·전처리(11개 도시 인구중복 보정, KOSIS ±0.8%) → 전체 통계·공간 분석 설계 및 **Python 3.12 구현** → Abstract·Methods·Results·Discussion **전 섹션 단독 저술(원저)** → Figure 1–2 및 Supplementary S1–S4 제작 → **재현 패키지 공개**(Zenodo, seed=42).

## 폴더 구성 / Contents

```
06_typhoid_epidemiology/
└── code/
    ├── analysis/   추세·주기성·공간 통계 파이프라인 (typhoid_r2_*.py 등)
    └── figures/    출판용 그림 생성 코드 (build_fig_wide.py 등)
```

> 이 레포에는 **분석 코드만** 포함합니다. 원고·Supplementary·그림·원자료는 논문(DOI)과 Zenodo 재현 패키지에 공개되어 있습니다.

**Environment.** Python 3.12 (pandas, numpy, statsmodels, scipy, PyMannKendall, matplotlib). 모든 Monte-Carlo 절차 `seed = 42`.

---

*Reproducibility package: https://doi.org/10.5281/zenodo.19810837*
