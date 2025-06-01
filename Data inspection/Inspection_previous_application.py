import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 데이터 불러오기
prev = pd.read_csv('term_project_data/previous_application.csv')

# 데이터 구조 확인
print("데이터 크기 (rows, columns):", prev.shape)
print("\n컬럼 데이터 타입별 개수:")
print(prev.dtypes.value_counts())

# 수치형 변수 요약 통계
print("\n수치형 변수 통계 요약:")
print(prev.describe().transpose())

# 범주형 변수 요약
categorical = prev.select_dtypes(include='object')
print("\n범주형 변수 고유값 수 및 최빈값:")
for col in categorical.columns:
    print(f"{col}: 고유값 수 = {categorical[col].nunique()}, 최빈값 = {categorical[col].mode().iloc[0]}")

# 결측치 분석
missing = prev.isnull().sum()
missing_percent = 100 * missing / len(prev)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing Percent': missing_percent
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Percent', ascending=False)

print("\n결측치 비율 상위 10개:")
print(missing_df.head(10))

# 이전 신청 상태 분포
plt.figure(figsize=(8, 5))
sns.countplot(x='NAME_CONTRACT_STATUS', data=prev)
plt.title('Distribution of Previous Application Status')
plt.xlabel('Application Status')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# 계약 유형 분포
plt.figure(figsize=(10, 5))
sns.countplot(x='NAME_CONTRACT_TYPE', data=prev)
plt.title('Distribution of Previous Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# 신청 금액 vs 승인 금액 분포
mean_amounts = prev.groupby('NAME_CONTRACT_TYPE')[['AMT_APPLICATION', 'AMT_CREDIT']].mean().reset_index()
melted = mean_amounts.melt(id_vars='NAME_CONTRACT_TYPE', var_name='Amount_Type', value_name='Mean_Amount')
plt.figure(figsize=(10, 6))
sns.barplot(data=melted, x='NAME_CONTRACT_TYPE', y='Mean_Amount', hue='Amount_Type')
plt.title('Mean Application vs Credit Amount by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Mean Amount')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Amount Type')
plt.tight_layout()
plt.show()

# 4. 계약 유형별 신청 상태
plt.figure(figsize=(12, 6))
sns.countplot(x='NAME_CONTRACT_TYPE', hue='NAME_CONTRACT_STATUS', data=prev)
plt.title('Application Status by Loan Type')
plt.xlabel('Loan Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Application Status')
plt.show()

# 이전 신청일로부터 경과 일수 분포
plt.figure(figsize=(8, 5))
sns.histplot(prev['DAYS_DECISION'], bins=50, kde=True)
plt.title('Distribution of Days Since Previous Application')
plt.xlabel('Days Since Decision')
plt.ylabel('Count')
plt.show()

# 전체 수치형 변수 기반 상관관계 히트맵
numerical_cols = ['AMT_APPLICATION',
    'AMT_CREDIT',
    'AMT_DOWN_PAYMENT',
    'AMT_GOODS_PRICE',
    'DAYS_DECISION',
    'RATE_DOWN_PAYMENT',
    'DAYS_FIRST_DRAWING',
    'DAYS_FIRST_DUE',
    'DAYS_LAST_DUE',
    'DAYS_TERMINATION']
corr = prev[numerical_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
