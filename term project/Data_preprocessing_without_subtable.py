import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from category_encoders import OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import re

# ---------------------------------------------------------------------
# 1. 원본 데이터 로드 및 병합 (application_train + application_test)
# ---------------------------------------------------------------------
train = pd.read_csv('term_project_data/application_train.csv')
test  = pd.read_csv('term_project_data/application_test.csv')
app   = pd.concat([train, test], ignore_index=True)

# ---------------------------------------------------------------------
# 2. 결측치 처리 ─────────────────────────────────────────────────────
# ---------------------------------------------------------------------
protect = ['TARGET','SK_ID_CURR','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','AMT_ANNUITY']
drop_cols = app.columns[app.isnull().mean()>0.5].difference(protect)
app = app.drop(columns=drop_cols)

np.random.seed(0)
for col in app.select_dtypes('object'):
    if app[col].isnull().any():
        probs = app[col].value_counts(normalize=True)
        app.loc[app[col].isnull(), col] = np.random.choice(probs.index, p=probs.values,
                                                           size=app[col].isnull().sum())

base_cols = ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','AMT_ANNUITY']
num_temp  = app.select_dtypes(np.number).fillna(app.mean(numeric_only=True))
extra     = (num_temp.corr()['TARGET'].abs()
             .drop(base_cols+['TARGET'], errors='ignore')
             .nlargest(4).index.tolist())
important = base_cols + [x for x in extra if x not in base_cols]

other_num = app.select_dtypes(np.number).columns.difference(important)
app[other_num] = app[other_num].fillna(app[other_num].mean())

imp = IterativeImputer(random_state=0)
app[important] = imp.fit_transform(app[important])

# ---------------------------------------------------------------------
# 3. 인코딩 (Label, OneHot, Ordinal) ────────────────────────────────
# ---------------------------------------------------------------------
cat_cols = app.select_dtypes('object').columns

label_encoded = app.copy()
le = LabelEncoder()
for c in cat_cols:
    label_encoded[c] = le.fit_transform(label_encoded[c])

one_hot_encoded = pd.get_dummies(app, columns=cat_cols, dtype=int)

ordinal_encoded = app.copy()
oe = OrdinalEncoder(cols=cat_cols, handle_unknown='impute')
ordinal_encoded = oe.fit_transform(ordinal_encoded)

dfs = {"Label":label_encoded, "OneHot":one_hot_encoded, "Ordinal":ordinal_encoded}
print("✅ 인코딩 완료")

# ---------------------------------------------------------------------
# 4. 피처 셀렉션 (층화 샘플링 + Mutual Info + LGBM)
# ---------------------------------------------------------------------
def clean_names(cols):
    return [re.sub(r'[^0-9A-Za-z_]', '_', c) for c in cols]

selected, col_maps = {}, {}

for n, df in dfs.items():
    df_train = df[df['TARGET'].notnull()]
    original_cols = df_train.drop(['SK_ID_CURR','TARGET'], axis=1).columns
    clean = clean_names(original_cols)
    col_maps[n] = dict(zip(clean, original_cols))

    X = df_train.drop(['SK_ID_CURR','TARGET'], axis=1).copy()
    X.columns = clean
    y = df_train['TARGET'].astype(int)

    if len(df_train) > 60000:
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=60000, random_state=0, stratify=y)
    else:
        X_sample = X
        y_sample = y

    mi = pd.Series(mutual_info_classif(X_sample, y_sample, random_state=0), index=X.columns)
    mi_sel = mi.nlargest(int(len(mi) * 0.75)).index

    lgb_train = lgb.Dataset(X_sample[mi_sel], label=y_sample)
    model = lgb.train({
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'seed': 0,
        'num_threads': 1
    }, lgb_train, num_boost_round=50)

    lgb_imp = pd.Series(model.feature_importance('gain'), index=X[mi_sel].columns).nlargest(200).index
    selected[n] = [col_maps[n][c] for c in lgb_imp]

# ---------------------------------------------------------------------
# 5. 스케일링된 슬림 데이터 생성
# ---------------------------------------------------------------------
scalers = {"MinMax":preprocessing.MinMaxScaler(),
           "Robust":preprocessing.RobustScaler(),
           "Standard":preprocessing.StandardScaler()}

for name, df in dfs.items():
    slim = df[['SK_ID_CURR','TARGET'] + selected[name]].copy()
    feats = slim.drop(['SK_ID_CURR','TARGET'], axis=1)
    for s_name, scaler in scalers.items():
        scaled = pd.DataFrame(scaler.fit_transform(feats), columns=feats.columns)
        scaled['SK_ID_CURR'] = slim['SK_ID_CURR'].values
        scaled['TARGET']     = slim['TARGET'].values
        cols = ['SK_ID_CURR'] + list(feats.columns) + ['TARGET']
        scaled = scaled[cols]
        scaled.to_csv(f'{name}_Slim_{s_name}.csv', index=False)
        print(f'[{name}] {s_name} → shape {scaled.shape}  → saved.')

print("✅ 전체 파이프라인 완료")

# ---------------------------------------------------------------------
# 6. Train / Test 데이터로 다시 분리
# ---------------------------------------------------------------------
train = pd.read_csv('term_project_data/application_train.csv')
test  = pd.read_csv('term_project_data/application_test.csv')

n_train = train.shape[0]
n_test = test.shape[0]

for name, df in dfs.items():
    for s_name in scalers.keys():
        slim_df = pd.read_csv(f"{name}_Slim_{s_name}.csv")

        slim_train = slim_df.iloc[:n_train]
        slim_test = slim_df.iloc[n_train:]

        slim_test = slim_test.drop(columns=['TARGET'])

        slim_train.to_csv(f"{name}_Train_Slim_{s_name}.csv", index=False)
        slim_test.to_csv(f"{name}_Test_Slim_{s_name}.csv", index=False)

        print(f"[{name}] {s_name} → Train: {slim_train.shape}, Test: {slim_test.shape} → saved.")

print("✅ Train / Test 분리 완료")
