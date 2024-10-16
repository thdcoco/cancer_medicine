import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import xgboost as xgb
import lightgbm as lgb

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

# SUBCLASS 가 범주형이기 때문에 LabelEncoder 사용
le_subclass = LabelEncoder()
train['SUBCLASS'] = le_subclass.fit_transform(train['SUBCLASS'])

# 변환된 레이블 확인
for i, label in enumerate(le_subclass.classes_):
    print(f"원래 레이블: {label}, 변환된 숫자: {i}")



from sklearn.model_selection import train_test_split
X_train, y_train = train.drop(columns="SUBCLASS"), train["SUBCLASS"]

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)


train_all = pd.concat((X_train, y_train), axis=1)
val_all = pd.concat((X_val, y_val), axis=1)

## x 의 경우도 범주형으로 구성되어 있어, 알맞은 인코딩 필요
X = train_all.drop(columns=['SUBCLASS', 'ID'])
y_subclass = train_all['SUBCLASS']

categorical_columns = X.select_dtypes(include=['object', 'category']).columns
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_encoded = X.copy()
X_encoded[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])


test_X = test.drop(columns=['ID'])
test_X_encoded = test_X.copy()
test_X_encoded[categorical_columns] = ordinal_encoder.transform(test_X[categorical_columns])

X_val = val_all.drop(columns=['SUBCLASS', 'ID'])
val_y_encoded = val_all['SUBCLASS']


val_x_encoded = X_val.copy()
val_x_encoded[categorical_columns] = ordinal_encoder.transform(X_val[categorical_columns])

file_path = ''
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import numpy as np
import joblib
# 기본 설정
X_train = X_encoded
y_train = y_subclass
random_states = [0, 22, 42, 1215]  # 사용할 random_state 값들

# 결과 저장용 리스트
all_macro_f1_scores = []
all_class_f1_scores = []
all_feature_importances = []

# 테스트 데이터 예측 결과 저장용 배열 초기화
final_lgb_test_preds_proba = np.zeros((test_X_encoded.shape[0], len(np.unique(y_train))))

# 다양한 random_state로 반복
for r_state in random_states:
    print(f"## Random State: {r_state} ##\n")
    
    # Stratified K-Fold 설정
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=r_state)

    # LightGBM 모델 파라미터 설정
    params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train)),
        'learning_rate': 0.01786236294491012,
        'random_state': 42,
        'metric': 'multi_logloss',
        'early_stopping_rounds': 100,
        'n_jobs': -1,
        'verbose': -1,
        'num_leaves': 31,
        'min_data_in_leaf': 27,
        'lambda_l1': 0.00046641762342032746,
        'lambda_l2': 8.228074508440626e-06,
        'min_split_gain': 0.0008507877755254931,
        'min_child_weight': 0.00044655605235435
    }

    # F1 스코어 및 중요도 저장용 리스트 초기화
    macro_f1_score_list = []
    feature_importance_list = []
    class_f1_scores = []

    # 각 클래스의 샘플 수 계산
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (len(class_counts) * class_counts)

    # 가중치 배열 생성
    weights = np.array([class_weights[label] for label in y_train])

    for idx, (train_idx, _) in enumerate(folds.split(X_train, y_train)):
        print('#' * 40, f'폴드 {idx+1} / {folds.n_splits}', "#" * 40)
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        weights_tr = weights[train_idx]
        
        d_train = lgb.Dataset(X_tr, label=y_tr, weight=weights_tr)
        d_valid = lgb.Dataset(val_x_encoded, label=val_y_encoded, reference=d_train)
        
        lgb_model = lgb.train(params=params,
                              train_set=d_train,
                              num_boost_round=2000,
                              valid_sets=[d_train, d_valid])
        
        # 테스트 데이터 예측 (여러 random_state 결과를 평균)
        final_lgb_test_preds_proba += lgb_model.predict(test_X_encoded, num_iteration=lgb_model.best_iteration) / (len(random_states)*folds.n_splits)
        
        # 검증 데이터 예측
        val_preds_proba = lgb_model.predict(val_x_encoded, num_iteration=lgb_model.best_iteration)
        val_preds = np.argmax(val_preds_proba, axis=1)
        model_path = os.path.join(file_path + "./xgb_model_path", f"xgb_model_rs{r_state}_fold{idx+1}.joblib")
        joblib.dump(lgb_model, model_path)
        print(f"모델이 '{model_path}'에 저장되었습니다.\n")            
        
        # 다중 클래스 F1 스코어 계산 (클래스별 F1 스코어 포함)
        macro_f1 = f1_score(val_y_encoded, val_preds, average='macro')
        class_f1 = f1_score(val_y_encoded, val_preds, average=None)
        macro_f1_score_list.append(macro_f1)
        class_f1_scores.append(class_f1)
        
        print(f'폴드 {idx+1} Macro F1 score: {macro_f1}\n')
        print(f'폴드 {idx+1} 클래스별 F1 score:\n{classification_report(val_y_encoded, val_preds)}\n')
        
        feature_importance_list.append(lgb_model.feature_importance())

    # 각 random_state에 대한 결과 저장
    avg_macro_f1 = np.mean(macro_f1_score_list)
    all_macro_f1_scores.append(avg_macro_f1)
    all_class_f1_scores.append(class_f1_scores)
    all_feature_importances.append(feature_importance_list)

    print(f'Random State {r_state} 검증 평균 Macro F1 score: {avg_macro_f1}\n')

# 전체 결과 요약
for i, r_state in enumerate(random_states):
    print(f"## Random State {r_state} 결과 ##")
    print(f"검증 평균 Macro F1 score: {all_macro_f1_scores[i]}")
    print(f"클래스별 F1 score:\n{np.mean(all_class_f1_scores[i], axis=0)}\n")
    print("#" * 80)