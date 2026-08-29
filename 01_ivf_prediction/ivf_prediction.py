# %% [markdown]
# ### Import

# %%
import pandas as pd
import numpy as np

from sklearn.preprocessing import  OrdinalEncoder
from sklearn.ensemble import ExtraTreesClassifier

# %% [markdown]
# ### Data Load

# %%
# 데이터 로드 및 전처리 (불필요한 ID열 제거)
train = pd.read_csv('/Users/qwergs/Desktop/LGAimer /헤커톤/open/train.csv').drop(columns=['ID'])
test = pd.read_csv('/Users/qwergs/Desktop/LGAimer /헤커톤/open/test.csv').drop(columns=['ID'])

# %%
# X : 모델의 feature로 사용할 데이터 
# y : 예측할 목표 값(라벨). 즉, 임신 성공 여부 컬럼을 예측 
X = train.drop('임신 성공 여부', axis=1)
y = train['임신 성공 여부']

# %% [markdown]
# ### Data Pre-processing

# %%
# 카테고리형 변수 목록 정의 및 문자열 변환 
# categorical_columns 리스트를 정의하여 카테고리형(문자형)변수를 지정

categorical_columns = [
    "시술 시기 코드",
    "시술 당시 나이",
    "시술 유형",
    "특정 시술 유형",
    "배란 자극 여부",
    "배란 유도 유형",
    "단일 배아 이식 여부",
    "착상 전 유전 검사 사용 여부",
    "착상 전 유전 진단 사용 여부",
    "남성 주 불임 원인",
    "남성 부 불임 원인",
    "여성 주 불임 원인",
    "여성 부 불임 원인",
    "부부 주 불임 원인",
    "부부 부 불임 원인",
    "불명확 불임 원인",
    "불임 원인 - 난관 질환",
    "불임 원인 - 남성 요인",
    "불임 원인 - 배란 장애",
    "불임 원인 - 여성 요인",
    "불임 원인 - 자궁경부 문제",
    "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도",
    "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성",
    "불임 원인 - 정자 형태",
    "배아 생성 주요 이유",
    "총 시술 횟수",
    "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수",
    "DI 시술 횟수",
    "총 임신 횟수",
    "IVF 임신 횟수",
    "DI 임신 횟수",
    "총 출산 횟수",
    "IVF 출산 횟수",
    "DI 출산 횟수",
    "난자 출처",
    "정자 출처",
    "난자 기증자 나이",
    "정자 기증자 나이",
    "동결 배아 사용 여부",
    "신선 배아 사용 여부",
    "기증 배아 사용 여부",
    "대리모 여부",
    "PGD 시술 여부",
    "PGS 시술 여부"
]

# %%
# 카테고리형 컬럼들을 문자열로 변환
# astype(str)을 사용하여 모든 카테고리형 변수를 문자열 타입으로 변환 
    # 문자열 변환은 OrdinalEncoder 적용 전에 필요 
    # 숫자로 표현된 카테고리 값이 실수형 데이터로 인식되는 것을 방지 
for col in categorical_columns:
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)

# %% [markdown]
# ##### **카테고리형 변수 인코딩**
# - OrdinalEncoder를 사용하여 카테고리형 변수를 정수로 변환 
#     - handle_unknown = 'use_encoder_value', unknown_value = - 1옵션을 통해, 훈련 데이터에 없는 새로운 카테고리 값이 등장하면 -1로 처리하도록 설정 
#     - fit_transform(X[categorical_columns]) : 훈련 데이터를 학습(fit) 하여 변환(transform)합니다.
#     - transform(test[categorical_columns]) : 동일한 변환 규칙을 테스트 데이터에 적용합니다. 

# %%
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

X_train_encoded = X.copy()
X_train_encoded[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])

X_test_encoded = test.copy()
X_test_encoded[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# %%
numeric_columns = [
    "임신 시도 또는 마지막 임신 경과 연수",
    "총 생성 배아 수",
    "미세주입된 난자 수",
    "미세주입에서 생성된 배아 수",
    "이식된 배아 수",
    "미세주입 배아 이식 수",
    "저장된 배아 수",
    "미세주입 후 저장된 배아 수",
    "해동된 배아 수",
    "해동 난자 수",
    "수집된 신선 난자 수",
    "저장된 신선 난자 수",
    "혼합된 난자 수",
    "파트너 정자와 혼합된 난자 수",
    "기증자 정자와 혼합된 난자 수",
    "난자 채취 경과일",
    "난자 해동 경과일",
    "난자 혼합 경과일",
    "배아 이식 경과일",
    "배아 해동 경과일"
]

# %%
# 결측치 처리 : 결측값을 0으로 채움 
    # 임신과 관련된 실험 데이터에서 결측치는 특정 측정이 이루어지지 않았음을 의미할 가능성이 큼(따라서 0으로 채우는 것은 논리적으로 타당할 수 있음)
X_train_encoded[numeric_columns] = X_train_encoded[numeric_columns].fillna(0)
X_test_encoded[numeric_columns] = X_test_encoded[numeric_columns].fillna(0)

# %%
# 모델 학습 
model = ExtraTreesClassifier(random_state=42)

model.fit(X_train_encoded, y)

# %%
import optuna
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# Optuna objective 함수 정의
def objective(trial):
    # 하이퍼파라미터 탐색 공간 정의
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 10), # 최대 10이 적당 이 이상은 과적합 
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
    }
    
    # XGBoost 모델 생성
    model = XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42,
        **param
    )
    
    # 3-Fold 교차 검증을 통해 ROC-AUC 점수를 평가
    scores = cross_val_score(model, X_train_encoded, y, cv=10, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

# 최적화 스터디 생성 (목표: ROC-AUC 최대화)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# 최적 하이퍼파라미터와 최고 ROC-AUC 점수 출력
print("Best hyperparameters:", study.best_params)
print("Best ROC-AUC Score:", study.best_value)

# %%
# 최적 하이퍼파라미터로 모델 재생성 및 학습
best_params = study.best_params  # Optuna에서 찾은 최적 파라미터
best_model = XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42,
    **best_params
)
best_model.fit(X_train_encoded, y)

# 테스트 데이터에 대한 예측 수행 (양성 클래스 확률)
pred_proba_test = best_model.predict_proba(X_test_encoded)[:, 1]

# 제출 파일 불러오기 및 결과 할당
sample_submission = pd.read_csv('/Users/qwergs/Desktop/LGAimer /헤커톤/open/sample_submission.csv')
sample_submission['probability'] = pred_proba_test

# 파일 저장 (제출 파일 생성)
sample_submission.to_csv('/Users/qwergs/Desktop/final_submission.csv', index=False)


