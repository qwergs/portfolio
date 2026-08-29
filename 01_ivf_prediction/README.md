# 01. LG Aimers 6기 — 난임 환자 임신 성공 여부 예측

> **LG Aimers 6th — IVF Pregnancy Success Prediction Hackathon**
> LG AI 연구원 · LG화학 주관 / 2025년 2월 — 3월

---

## 🇰🇷 한국어 개요

### 프로젝트 배경
LG Aimers 6기 과정 수료 후 진행된 LG화학 주관 해커톤 본선 과제입니다. 난임 진료 환자의 시술 데이터를 활용하여 **임신 성공 여부를 확률로 예측**하는 이진 분류 문제를 풀었습니다.

### 문제 정의
- **태스크:** 이진 분류 (Binary Classification)
- **타깃 변수:** `임신 성공 여부` (0/1)
- **평가 지표:** ROC-AUC (대회 리더보드 기준)
- **데이터 규모:** Train 약 25만 행, Test 약 9만 행, 60+ 변수
- **변수 특성:** 카테고리형 변수가 절대 다수 (시술 시기·유형, 불임 원인, 배아/난자/정자 관련 정보 등)

### 데이터 전처리
1. **ID 컬럼 제거** — 식별자성 컬럼은 학습에 불필요
2. **카테고리형 변수 정의** — 약 50개의 카테고리형 컬럼을 명시적으로 리스트화
   - 시술 관련: 시술 시기/유형, 특정 시술 유형, 배란 자극·유도 여부, 단일 배아 이식 여부, PGD/PGS 여부
   - 불임 원인: 남성/여성/부부 주·부 불임 원인, 난관 질환, 자궁내막증, 정자 형태/운동성 등
   - 시술 횟수: 총/IVF/DI 시술·임신·출산 횟수
3. **문자열 변환** (`astype(str)`) — 숫자형으로 인식되는 카테고리 값을 방지
4. **OrdinalEncoder** — `handle_unknown='use_encoded_value', unknown_value=-1` 옵션으로 unseen 카테고리 안전 처리

### 모델링 전략
- **베이스라인:** ExtraTreesClassifier (트리 기반 앙상블)
- **확장 실험:** XGBoost, LightGBM
- **하이퍼파라미터 튜닝:** Optuna (Bayesian Optimization) — 다수의 시드/CV fold 조합으로 안정성 확보
- **교차검증:** Stratified K-Fold (k=5, 25, 35, 50 까지 다양하게 실험)
- **최종 제출:** Optuna로 튜닝된 트리 앙상블 + 확률 캘리브레이션

### 결과 및 인사이트
- 카테고리형 변수가 압도적이라 트리 기반 앙상블이 강력한 성능
- Optuna로 fold 수와 트리 깊이를 함께 튜닝하니 LB 점수 안정화
- 데이터 누수 방지를 위해 OrdinalEncoder는 train으로 fit, test에 transform만 적용

### 산출물
- 코드: `code/` 폴더 (Jupyter, Python, HTML 변환본)
- 리포트/PT: `docs/` 폴더 — `LGaimers 결과.pdf`, `LGaimers 결과 ppt.pdf` (원본 폴더 참조)

---

## 🇺🇸 English Summary

### Problem
A binary classification task to predict the **probability of pregnancy success** for IVF/DI patients using clinical and procedure metadata. Hosted by LG Chem during the LG Aimers 6th cohort hackathon (Feb–Mar 2025).

### Data
- 250k+ train rows / 90k+ test rows, 60+ features
- Heavily categorical: procedure type, infertility causes (male/female/couple), embryo/sperm/egg attributes, cycle counts
- Target: binary `pregnancy_success`

### Pipeline
1. Drop ID columns
2. Explicitly cast 50 categorical columns to string
3. Encode with `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` to safely handle unseen categories at inference time
4. Train ExtraTrees / XGBoost / LightGBM ensembles
5. Hyperparameter tuning with **Optuna** across multiple seeds and CV configurations (5-, 25-, 35-, 50-fold)
6. Probability calibration before submission

### Outcome
Tree-based ensembles dominated, given the categorical-heavy schema. Optuna tuning over fold count + tree depth produced the most stable leaderboard score. Strict train-only encoder fitting prevented leakage.

---

## 📂 Files

| File | Description |
|---|---|
| `ivf_prediction.py` | 최종 제출 파이프라인 (스크립트) / Final submission pipeline (script) |
| `ivf_prediction.ipynb` | 최종 파이프라인 노트북 (출력 제거) / Final pipeline notebook |

> 원본 데이터(대회·기관 제공)는 라이선스·용량 문제로 포함하지 않으며, 노트북 출력은 제거했습니다. 코드 내 경로는 실행 환경에 맞게 수정하세요.
> Raw datasets are not included (license/size); notebook outputs were cleared. Adjust file paths to your environment.

## 🛠️ 사용 기술 / Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn (ExtraTreesClassifier, OrdinalEncoder)` · `XGBoost` · `LightGBM` · `Optuna`

---

## 🏷️ 키워드 / Keywords
`Binary Classification` · `Tabular ML` · `Categorical Encoding` · `Tree Ensembles` · `Optuna HPO` · `Healthcare ML` · `IVF`
