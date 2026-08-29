# 02. 제7회 서울교육 데이터 활용 아이디어 공모전

> **7th Seoul Education Data Idea Competition (MOE)**
> 서울특별시교육청 / 교육부 주관

---

## 🇰🇷 한국어 개요

### 프로젝트 배경
서울특별시교육청이 주최한 데이터 활용 공모전 출품작입니다. 서울 25개 자치구의 **학업성취도 데이터**(국어/수학/영어 척도점수)와 **사회·복지·행정 통계** 데이터를 결합하여, 학업 성취에 영향을 주는 외생 변수들을 통계적으로 규명하는 것이 목표였습니다.

### 분석 주제
1. **학업 성취도 vs 복지·사회 인프라** — 미혼모자 공동생활가정, 성폭력 피해자 보호시설, 자립 지원 시설, 가정폭력 피해자 보호시설 등이 국·영·수 점수와 어떻게 상관되는가
2. **유흥업소 밀집도 vs 청소년 범죄율** — 자치구별 유흥업소 영업 인허가 정보와 청소년 범죄 통계 결합
3. **학부모 민원** — 청소년상담복지센터 현황 등

### 데이터셋
- 자치구별 국어/수학/영어 척도점수 평균 및 보통학력 이상 비율
- 사회복지 시설 통계 (보호시설, 공동생활가정 등)
- 인구·경제 통계 (지역내총생산, 인구 이동)
- 유흥업소 영업 인허가 정보, 청소년 범죄 통계
- 청소년상담복지센터 현황

### 분석 방법
1. **데이터 통합** — 자치구·연도 멀티인덱스로 다양한 출처 데이터 병합
2. **상관관계 분석** — Pearson 상관계수, 양/음 상관 변수 식별
3. **PCA 학업성취 잠재 지수** — 국어/수학/영어 비율 변수를 입력으로 첫 번째 주성분(PC1)을 "학업성취 지수"로 정의
4. **회귀 분석**
   - OLS 회귀 (다중공선성 진단 포함)
   - 상호작용항을 활용한 회귀
   - **Ridge / Lasso / Random Forest** 비교 (5-fold CV)
   - **데이터 누수 검증:** 종속변수 자체가 설명변수에 포함된 경우 R²≈1이 되는 함정을 발견·교정
5. **공간 시각화** — Folium 기반 서울시 자치구 단위 점수 지도

### 결과 및 인사이트
- **양의 상관:** 미혼모자 공동생활가정, 성폭력 피해자 보호시설, 여성 부랑인 시설 → 사회 안전망이 잘 갖춰진 지역일수록 학업 성취 지표가 높음
- **음의 상관:** 자립 지원 시설, 가정폭력 피해자 보호시설 → 사회적 취약 환경의 부정적 영향
- **순수 외생 변수만으로 5-fold CV:** 평균 R² ≈ 0.52 (분산의 약 50% 설명)
- **모델 비교:** Ridge는 과소적합, Lasso는 변동성 큼, RF는 일반화 능력 추가 검증 필요

### 산출물
- `code/` — 분석/시각화 노트북, Folium 지도 HTML
- `docs/` — 기획보고서, 발표 자료, 데이터 명세 (원본 폴더 참조)

---

## 🇺🇸 English Summary

### Problem
Submission to the **7th Seoul Education Data Idea Competition** organized by the Seoul Metropolitan Office of Education. Goal: identify exogenous (non-academic) factors influencing student academic achievement across Seoul's 25 districts (gu) by joining academic test data with social, welfare, and administrative statistics.

### Sub-topics
1. Academic achievement vs welfare/social infrastructure (shelters, group homes, etc.)
2. Density of entertainment-venue businesses vs juvenile crime rates
3. Parental complaints and youth counseling center coverage

### Pipeline
1. **Multi-source merge** — concatenate datasets on (year, district) multi-index
2. **Correlation screening** — identify positive vs negative correlates of subject scores
3. **PCA latent achievement index** — extract PC1 from above-baseline-rate variables across subjects
4. **Regression suite** — OLS (with VIF diagnostics), interaction-term OLS, Ridge / Lasso / Random Forest with 5-fold CV
5. **Leakage check** — caught a leakage trap where the dependent variable was inadvertently included as a feature (R²≈1)
6. **Spatial visualisation** — Folium choropleths of district-level scores

### Findings
- Strong **positive** correlation: areas with denser welfare infrastructure tend to have higher academic scores (likely indicating overall administrative capacity)
- Strong **negative** correlation: indicators of socio-economic vulnerability (self-support shelters, domestic-violence shelters)
- Pure exogenous-variable model: 5-fold CV R² ≈ 0.52 — explains ~50% of district-level score variance
- Among regularized models, Lasso slightly outperformed Ridge; RF needed more validation for generalisation

---

## 📂 Files

| File | Description |
|---|---|
| `seoul_education_regression.ipynb` | 최종 통합 회귀 분석 / Final integrated regression |
| `youth_crime_analysis.ipynb` | 유흥업소 밀집도–청소년 범죄율 분석 / Venue density vs juvenile crime |

> 원본 데이터(대회·기관 제공)는 라이선스·용량 문제로 포함하지 않으며, 노트북 출력은 제거했습니다. 코드 내 경로는 실행 환경에 맞게 수정하세요.
> Raw datasets are not included (license/size); notebook outputs were cleared. Adjust file paths to your environment.

## 🛠️ 사용 기술 / Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn (Ridge, Lasso, RandomForest, PCA)` · `statsmodels (OLS, VIF)` · `seaborn / matplotlib` · `Folium` · `AppleGothic 한글 폰트 처리`

---

## 🏷️ 키워드 / Keywords
`EDA` · `Correlation Analysis` · `PCA` · `Multivariate Regression` · `Regularization (Ridge/Lasso)` · `Spatial Visualisation (Folium)` · `Education Analytics` · `Public Data`
