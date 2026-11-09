# 제7회 서울교육 데이터 활용 공모전

서울 25개 자치구의 학업성취도와 복지·인구·경제 통계의 관계를 분석했습니다 (2025, 팀원으로 참여).

- 국어·수학·영어 점수를 PCA로 묶어 학업성취 지수를 만듦
- OLS, Ridge, Lasso, Random Forest를 5-fold 교차검증으로 비교
- 설명력이 거의 100%로 나온 결과를 확인하다가 정답이 섞인 변수를 찾아 제거 (수정 후 설명력 약 52%)
- Folium으로 자치구 지도 시각화

파일: `seoul_education_regression.ipynb`, `youth_crime_analysis.ipynb`
