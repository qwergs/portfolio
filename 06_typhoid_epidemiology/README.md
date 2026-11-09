# 장티푸스 24년 신고자료 분석 (2001–2024)

Lee G, Kim S. *Foodborne Pathogens and Disease*, 2026. 제1저자.
논문: https://doi.org/10.1177/15353141261465221 · 데이터와 코드: https://doi.org/10.5281/zenodo.19810837

질병관리청에 신고된 장티푸스 3,535건을 다시 분석했습니다.

- 증가·감소 추세 검정(Mann–Kendall)과 음이항 회귀
- 웨이블릿으로 계절 패턴 확인, 추세가 바뀌는 시점 탐색
- 지역 군집 분석(Moran's I), 환자가 적은 지역은 Empirical Bayes로 보정
- 시뮬레이션으로 분석이 변화를 잡아낼 수 있는지 확인

결과: 연 7.3% 감소, 해마다 반복되는 계절성이 남아 있음, 남동해안 지역 군집 (Moran's I = 0.403)

파일: `code/analysis` (분석), `code/figures` (그림)
