# 기계 베어링 이상 탐지와 상태 지표

KSPHM·KIMM 기계데이터 챌린지 (2025). 본선 진출, 3등 (수상 4팀). 4명 팀에서 코드 설계와 오토인코더 설계를 맡았습니다.

- 데이터: 한국기계연구원 베어링 시험 데이터, 7채널 센서, 초당 25,600회 측정
- 10초 단위로 신호를 자르고 RMS, 첨도 등 통계값 추출
- 정상 신호만 학습한 1D CNN 오토인코더가 복원하지 못하는 정도로 이상을 판단
- 정상 구간을 기준으로 마모 정도를 수치화한 상태 지표(Health Index) 설계

파일: `bearing_pipeline.ipynb`, `autoencoder_model.ipynb`, `health_index_threshold.ipynb`
