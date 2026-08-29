# 04. FIELD — 금융 행동경제학 페르소나 모델링 + 포트폴리오 최적화

> **FIELD Project — Behavioral-Finance Persona Clustering & Portfolio Optimization**
> 2단계 (Phase 1 / Phase 2) 구성 / 2025년

---

## 🇰🇷 한국어 개요

이 프로젝트는 **두 단계**로 구성된 금융 데이터 분석/최적화 과제입니다.

### Phase 1 — 가구원 데이터 군집화 + 행동경제학 페르소나
**Task 1.1.** 가구원 데이터를 전처리·군집화하고 결과를 해석
**Task 1.2.** 군집화 결과 기반으로 행동경제학적 특징을 가진 페르소나 구축

#### 데이터
- 가구원 데이터셋 (CSV) + 가구원 변수 명세서 (XLSX) + 코드북 (CSV)
- 변수: 소득(`p_16_inc_all`), 경제활동(`p16aa009`), 카드 사용(`p16bd002/008`), 행복도(`p16gk001/004`), 세금 인식(`p16gb001`), 건강 상태(`p16ge001`), 은퇴 상태(`p16ge030`) 등

#### 전처리
- 파일 인코딩 자동 탐지 (`chardet`)
- 코드북 + 데이터 컬럼을 **멀티인덱스**로 결합 (코드북 대조 불편 해소)
- **금액 변수 결측치** → 선형 보간법 (`interpolate(method='linear')`)
- 표준화 (`StandardScaler`) + **PCA** (차원 축소)

#### 군집화
- **DBSCAN** 으로 군집 탐색 (밀도 기반)
- 다양한 클러스터 수 실험 (3개, 5개 등)
- 각 클러스터별 변수 평균/분포로 특성 요약 → 페르소나 매핑
- 예시 페르소나: 고소득·활동적 카드사용자, 저소득·은퇴 안정형, 중간 소득·세금 회피형 등

### Phase 2 — 자산 포트폴리오 최적화
**Task:** 다자산 (암호화폐 / 채권 / 국내·해외 ETF / 주식) 포트폴리오의 비중 최적화 모델 비교

#### 자산 풀
- **암호화폐:** 비트코인, 이더리움
- **채권:** 한국 국고채
- **ETF:** 국내 ETF
- **주식:** KOSPI Top 100

#### 모델 비교 (5종)
1. **Markowitz Mean-Variance Portfolio (MVP)** — 평균-분산 최적화
2. **Black-Litterman** — 시장 균형 수익률(CAPM) + 투자자 주관적 뷰 결합
3. **Risk Parity** — 자산별 위험기여도(Risk Contribution)를 동일하게 조정
4. **Equal Weight** — 단순 등가중
5. **Minimum Variance** — 변동성 최소화

각 모델별 장단점·목적함수·수학적 정의를 노트북 마크다운에 명시.

#### 평가 지표
- 연환산 수익률 (Annualized Return)
- 변동성 (Volatility)
- Sharpe Ratio
- Maximum Drawdown
- 자산 비중 변화 시각화

#### 인사이트
- Risk Parity는 수익률 예측 없이도 작동, 분산 효과 우수 (보수적 운용)
- Black-Litterman은 시장 데이터 + 주관적 뷰 결합으로 평균-분산의 한계 보완
- Mean-Variance는 단기 수익률 추정에 민감 → 자주 리밸런싱 필요

---

## 🇺🇸 English Summary

A **two-phase** project combining behavioral-finance segmentation with quantitative portfolio optimization.

### Phase 1 — Household Survey Clustering & Persona Modeling
- Clustered the household-member micro-survey dataset (income, card usage, happiness, tax perception, health, retirement)
- Pipeline: encoding auto-detection (`chardet`) → multi-index column join with codebook → linear interpolation for monetary missing values → `StandardScaler` + `PCA` → **DBSCAN** density-based clustering
- Mapped clusters to behavioral-finance personas (e.g. high-income active card user; low-income retired stable; mid-income tax-averse)

### Phase 2 — Multi-Asset Portfolio Optimization
- Universe: BTC, ETH, KR sovereign bonds, domestic ETF, KOSPI Top-100
- Compared **5 portfolio models**: Markowitz MVP, Black-Litterman, Risk Parity, Equal Weight, Minimum Variance
- Evaluated by annualized return, volatility, Sharpe ratio, max drawdown, weight stability

### Findings
- **Risk Parity** delivered the most stable diversification without needing return forecasts
- **Black-Litterman** improved over Markowitz by blending CAPM equilibrium with subjective views
- **Mean-Variance** was sensitive to short-window expected returns and required frequent rebalancing

---

## 📂 Files

| File | Description |
|---|---|
| `phase1_clustering.ipynb` | 행동경제 페르소나 군집화 (DBSCAN+PCA) / Behavioral persona clustering |
| `phase2_modeling.ipynb` | 모델링 / Modeling |
| `phase2_portfolio_optimization.ipynb` | 포트폴리오 최적화 (Markowitz/Black-Litterman/Risk Parity) / Portfolio optimization |

> 원본 데이터(대회·기관 제공)는 라이선스·용량 문제로 포함하지 않으며, 노트북 출력은 제거했습니다. 코드 내 경로는 실행 환경에 맞게 수정하세요.
> Raw datasets are not included (license/size); notebook outputs were cleared. Adjust file paths to your environment.

## 🛠️ 사용 기술 / Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn (DBSCAN, PCA, StandardScaler)` · `chardet` · `cvxpy / scipy.optimize` (포트폴리오 최적화) · `matplotlib (AppleGothic)` · 행동경제학 도메인 지식

---

## 🏷️ 키워드 / Keywords
`Unsupervised Learning` · `DBSCAN` · `PCA` · `Behavioral Finance` · `Persona Modeling` · `Portfolio Optimization` · `Markowitz` · `Black-Litterman` · `Risk Parity` · `Quant Finance`
