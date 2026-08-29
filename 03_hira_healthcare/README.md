# 03. HIRA 빅데이터 경진대회 — 호흡기 질환 분석

> **HIRA (Health Insurance Review & Assessment Service) Big Data Competition**
> 건강보험심사평가원 / 2025년

---

## 🇰🇷 한국어 개요

### 프로젝트 배경
건강보험심사평가원(HIRA)이 매년 개최하는 빅데이터 경진대회 출품작입니다. **HIRA NBD(National Big Data)** 의료 청구 데이터(EDU200/300/400/530 등)를 활용하여 호흡기 질환의 진료·재입원·사망률 패턴을 분석했습니다.

### 데이터셋
- **EDU200** — 일반 진료 데이터 (16.9MB)
- **EDU300** — 외래 진료 (82.6MB)
- **EDU400** — 명세서·처방 (12.1MB)
- **EDU530** — 추가 진료 정보 (9.7MB)
- **2023년 생활 속 질병·진료행위 통계** PDF
- **고령인구비율 / 시·군·구별 현재흡연율** 외부 통계 (보조)

### 분석 대상 질환 (ICD-10 기반)
호흡기질환을 세부 카테고리로 분류:
- **감기** (J00–J06)
- **인플루엔자** (J09–J11)
- **결핵** (A15, A16, A19)
- **만성폐쇄성폐질환(COPD)** (J43, J40 등)
- **폐렴, 천식, 기흉** 등 추가 분류

### 분석 파이프라인
1. **데이터 로딩** — `cp949` 인코딩 다중 CSV 통합
2. **ICD-10 코드 매핑** — 주상병/부상병 코드를 질환 카테고리로 매핑하는 함수형 추출 (`extract_respiratory_and_categories`)
3. **재입원/입원 여부 컬럼 생성** — 수진자 고유번호 기준 입원 이력 추적
4. **사망률 계산** — 진료결과 == 4 (사망) 비율
5. **카테고리별 EDA** — 폐렴 환자 사망률, 환자 그룹별 통계

### 산출물
- 코드: `code/` — 전처리·EDA·병합 노트북, ICD-10 코드 매핑 데이터셋
- 데이터: `merged_result.xlsx`, `merged_expanded_result.xlsx`, `merged200_400.xlsx`
- 외부 자료: `2023년 생활 속 질병·진료행위 통계.pdf`, ICD-10-CSV-master 코드북

### 주요 인사이트
- 호흡기질환 코드 매핑은 단일 prefix(`J*`)로 잡되, 결핵·기흉 등 비-J코드 일부 포함 여부를 옵션화
- 폐렴 환자의 사망률을 별도 컬럼으로 산출 (`(진료결과 == 4).sum() / total`)
- 재입원 여부를 환자별 시계열 청구 데이터에서 추출

---

## 🇺🇸 English Summary

### Problem
Submission to the **HIRA Big Data Competition** (2025) using the HIRA NBD national medical-claims dataset (EDU200/300/400/530). Focused on respiratory-disease patterns: visit counts, readmission flags, and mortality rates.

### Data
- EDU200/300/400/530 medical claims (CP949 encoded CSVs, 9–82 MB each)
- 2023 disease & procedure statistics PDF
- Auxiliary stats: elderly population ratio, smoking-rate by district

### Pipeline
1. Multi-encoding (CP949) CSV ingestion via pandas
2. **ICD-10 mapping** — function `extract_respiratory_and_categories()` maps primary/secondary diagnosis codes to disease buckets:
   - Common cold (J00–J06)
   - Influenza (J09–J11)
   - Tuberculosis (A15/A16/A19)
   - COPD (J43, J40 family)
   - Pneumonia, asthma, pneumothorax
3. Generate **readmission** and **inpatient** flags per patient
4. Compute mortality rates by disease (treatment-result code == 4)
5. EDA: cohort statistics, mortality breakdowns

### Key insights
- Combined J-prefix matching with selective non-J codes (e.g., A15 for TB, S270 for traumatic pneumothorax)
- Mortality computed cleanly as `(treatment_result == 4).sum() / total`
- Patient timelines reconstructed via `수진자고유번호` (patient ID) sorting

---

## 📂 Files

| File | Description |
|---|---|
| `preprocessing_eda.ipynb` | 전처리 및 탐색적 분석 / Preprocessing & EDA |
| `merge_claims.ipynb` | 청구데이터 병합 / Claims data merge |

> 원본 데이터(대회·기관 제공)는 라이선스·용량 문제로 포함하지 않으며, 노트북 출력은 제거했습니다. 코드 내 경로는 실행 환경에 맞게 수정하세요.
> Raw datasets are not included (license/size); notebook outputs were cleared. Adjust file paths to your environment.

## 🛠️ 사용 기술 / Tech Stack

`Python` · `pandas (cp949 encoding)` · `numpy` · `R (Rstudio)` · `ICD-10 code mapping` · `Healthcare claims data engineering`

---

## 🏷️ 키워드 / Keywords
`Healthcare Claims` · `ICD-10` · `Respiratory Disease` · `EDA` · `Mortality Analysis` · `Readmission Modeling` · `HIRA NBD`
