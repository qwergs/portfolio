# LG Aimers 6기: 난임 시술 임신 성공 예측

LG AI연구원 교육 프로그램 해커톤 과제 (2025.2–3). 예선 169위 / 794팀 (상위 21%). 4명 팀에서 전체 코드 설계를 맡았습니다.

- 데이터: 난임 시술 기록 약 25만 건, 범주형 변수 60여 개
- 정답 정보가 섞인 변수를 빼고 범주형 변수를 변환
- ExtraTrees, XGBoost, LightGBM을 Optuna로 튜닝하고 조합
- 예측 확률을 실제 비율에 맞게 보정, 평가 지표는 ROC-AUC

파일: `ivf_prediction.ipynb`, `ivf_prediction.py`
