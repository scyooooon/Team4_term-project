import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
app = pd.read_csv('term_project_data/application_train.csv')

# 데이터 구조 확인
print("데이터 크기 (rows, columns):", app.shape)
print("\n컬럼 데이터 타입별 개수:")
print(app.dtypes.value_counts())

# 수치형 변수 요약 통계
print("\n수치형 변수 통계 요약:")
print(app.describe().transpose())

# 범주형 변수 요약
categorical = app.select_dtypes(include='object')
print("범주형 변수 고유값 수 및 최빈값:")
for col in categorical.columns:
    print(f"{col}: 고유값 수 = {categorical[col].nunique()}, 최빈값 = {categorical[col].mode().iloc[0]}")

# 5. 결측치 분석
missing = app.isnull().sum()
missing_percent = 100 * missing / len(app)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing Percent': missing_percent
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Percent', ascending=False)

print("\n결측치 비율 상위 10개:")
print(missing_df.head(10))

# target 변수의 분포 확인
plt.figure(figsize=(6, 4))
sns.countplot(x='TARGET', data=app)
plt.title('Distribution of TARGET Variable')
plt.xlabel('Loan Repayment Status (0: Repaid, 1: Default)')
plt.ylabel('Count')
plt.show()

# 대출 금액 분포 확인
plt.figure(figsize=(8, 6))
sns.histplot(app['AMT_CREDIT'], kde=True)
plt.title('Distribution of Loan Amount (AMT_CREDIT)')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.show()

# 대출 계약 유형별 분포 확인
plt.figure(figsize=(6, 4))
sns.countplot(x='NAME_CONTRACT_TYPE', data=app)
plt.title('Distribution of Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Count')
plt.show()

# target 변수에 따른 대출 금액 분포 비교 (박스 플롯)

plt.figure(figsize=(12, 6))
sns.boxplot(x='NAME_INCOME_TYPE', y='AMT_CREDIT', hue='TARGET', data=app)
plt.title('Loan Amount by Income Type and Loan Repayment Status')
plt.xlabel('Income Type')
plt.ylabel('Loan Amount')
plt.xticks(rotation=45)
plt.legend(title='TARGET', labels=['Repaid (0)', 'Default (1)'])
plt.tight_layout()
plt.show()

# TARGET과의 상관관계 분석
correlation = app.corr(numeric_only=True)
target_corr = correlation['TARGET'].drop('TARGET').sort_values(key=abs, ascending=False)

# 상관관계 히트맵 시각화
top_corr_features = target_corr.head(10).index.tolist()
plt.figure(figsize=(10, 8))
sns.heatmap(app[top_corr_features + ['TARGET']].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation with TARGET - Top 10 Features')
plt.tight_layout()
plt.show()

# 나이 및 연령대 구간 생성
app['AGE'] = (-app['DAYS_BIRTH']) // 365
app['AGE_GROUP'] = pd.cut(app['AGE'], bins=[20, 30, 40, 50, 60, 70], labels=['20s', '30s', '40s', '50s', '60s'])

# 연령대별 연체율 계산
age_target = app.groupby('AGE_GROUP', observed=False)['TARGET'].mean().reset_index()

# 연령대별 시각화
plt.figure(figsize=(8, 6))
sns.barplot(x='AGE_GROUP', y='TARGET', data=age_target)
plt.title('Loan Default Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Default Rate (TARGET Mean)')
plt.ylim(0, age_target['TARGET'].max() + 0.02)
plt.tight_layout()
plt.show()
