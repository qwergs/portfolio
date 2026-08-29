# 05. 기계 베어링 진동 데이터 — 이상치 탐지 및 Health Index 모델링

> **Bearing Vibration Anomaly Detection & Remaining Useful Life (RUL) Modeling**
> NBD (KIMM Bearing Test) Dataset / 2025년 상반기

---

## 🇰🇷 한국어 개요

### 프로젝트 배경
한국기계연구원(KIMM)이 제공한 베어링 시뮬레이터 시험 데이터(TDMS 포맷)를 활용하여 베어링의 **이상 상태 감지**와 **Health Index(HI)** 추적을 수행한 산업 AI 프로젝트입니다.

### 데이터 형식
- **TDMS (NI Test Data Storage)** 바이너리 포맷
- 또는 raw binary `.egg` 파일 (numpy `frombuffer` 로 직접 파싱)
- **7개 채널:**
  - **CH1, CH2, CH3, CH4** — 진동(Vibration) 가속도 신호
  - **Torque [Nm]** — 회전 토크
  - **TC SP Front [℃], TC SP Rear [℃]** — 베어링 전·후방 온도
- **샘플링 레이트:** 25.6 kHz
- 10초 단위로 분할 → 256,000 샘플 / 세그먼트

### 특징(Feature) 추출
**진동 채널 (시간 영역 통계량):**
- 평균 (mean), 표준편차 (std), **RMS** (Root Mean Square)
- 최대값/최소값, **Peak-to-Peak**
- **Skewness, Kurtosis** (분포 비대칭도/첨도)

**운영 채널:**
- 10초 구간 평균 (Torque, 온도)

### Health Index (HI) 모델링
1. 정상(Healthy) 구간의 통계 분포로 기준선(baseline) 설정
2. 각 사이클의 통계량을 정상 분포 기준으로 정규화 → **HI** 산출
3. **HI 임계값(HI_THRESH)** 시각화/검증 (Train1 ~ Train8 별 다중 검증)
4. HI = 0.4 등 임계 라인을 기준으로 이상 시점 판정

### 딥러닝 확장
- **1D Convolutional AutoEncoder (CovAE)**
- 진동 신호의 **denoising + feature extraction** 동시 수행
- PyTorch 구현 (`cnn_hs_nsp.pth` — 학습 완료 가중치 동봉)
- Reconstruction error 기반 이상치 탐지

### 시각화
- 채널별 시계열 시각화
- 세그먼트별 RMS / Kurtosis 추세
- 이상 세그먼트 자동 추출 → `anomaly_images/` (NSP/RGB/Spec 변환 이미지)
- HI 사이클 추적 그래프 (axhline으로 임계값 표시)

### 모델 성능
- 최종 발표 노트북에서 **오차율 22.56% → 이후 21.56% 개선**
- 일부 실험에서 **오차율 10% (전설의 오차율)** 달성 — 데이터 누수 의심으로 추가 검증

### 산출물
- 코드: `code/` — TDMS 로딩, 특징 추출, HI 검증, 딥러닝 학습
- 모델: `cnn_hs_nsp.pth` (PyTorch 가중치)
- 발표 자료: `발표/` 구조도 (CovAE 1D conv 자동인코더), 진동/운영 분석 결과 이미지

---

## 🇺🇸 English Summary

### Problem
Industrial-AI project on bearing-vibration anomaly detection and **Health Index (HI)** estimation using the **KIMM Bearing Simulator** dataset (TDMS format).

### Data
- TDMS / raw binary `.egg` files
- 7 channels: 4 vibration (CH1–CH4) + Torque + Front/Rear bearing temperatures
- Sampling rate **25.6 kHz**, segmented into 10-second windows (256k samples each)

### Feature Engineering
- **Time-domain stats** per vibration channel: mean, std, RMS, min/max, peak-to-peak, skewness, kurtosis
- **Operational features**: 10s-window averages of torque & temperature

### Health Index Pipeline
1. Establish baseline statistics from healthy operation
2. Normalize each cycle's stats to baseline → produce HI value
3. Validate HI threshold across multiple train sets (Train1–Train8)
4. Threshold lines (e.g. HI = 0.4) flag anomalous cycles

### Deep Learning
- **1D Convolutional AutoEncoder** in PyTorch (`cnn_hs_nsp.pth`)
- Joint **denoising + feature extraction**
- Anomaly score from reconstruction error

### Results
- Final presented model: error rate ~21.56% (improved from 22.56%)
- A "legendary 10% error" run was achieved — flagged for further leakage review

---

## 📂 Files

| File | Description |
|---|---|
| `autoencoder_model.ipynb` | 직접 설계한 1D CNN AutoEncoder (PyTorch) / Self-built 1D CNN AutoEncoder |
| `bearing_pipeline.ipynb` | 신호처리·특징추출·이상탐지 파이프라인 / Signal processing & anomaly pipeline |
| `health_index_threshold.ipynb` | Health Index 임계값 설계·검증 / Health Index threshold design & validation |

> 원본 데이터(대회·기관 제공)는 라이선스·용량 문제로 포함하지 않으며, 노트북 출력은 제거했습니다. 코드 내 경로는 실행 환경에 맞게 수정하세요.
> Raw datasets are not included (license/size); notebook outputs were cleared. Adjust file paths to your environment.

## 🛠️ 사용 기술 / Tech Stack

`Python` · `nptdms` (TDMS reader) · `numpy` · `pandas` · `scipy.stats (skew, kurtosis)` · `PyTorch (Conv1D AutoEncoder)` · `matplotlib` · `Google Colab + Drive` · 신호처리 도메인 지식

---

## 🏷️ 키워드 / Keywords
`Industrial AI` · `Vibration Analysis` · `Bearing Fault Diagnosis` · `Anomaly Detection` · `Health Index` · `RUL (Remaining Useful Life)` · `1D CNN AutoEncoder` · `Time-series Feature Engineering` · `TDMS / NI`
