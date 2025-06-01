import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
bureau = pd.read_csv('term_project_data/bureau.csv')

# 데이터 구조 확인
print("데이터 크기 (rows, columns):", bureau.shape)
print("\n컬럼 데이터 타입별 개수:")
print(bureau.dtypes.value_counts())

# 수치형 변수 요약 통계
print("\n수치형 변수 통계 요약:")
print(bureau.describe().transpose())

# 범주형 변수 요약
categorical = bureau.select_dtypes(include='object')
print("\n범주형 변수 고유값 수 및 최빈값:")
for col in categorical.columns:
    print(f"{col}: 고유값 수 = {categorical[col].nunique()}, 최빈값 = {categorical[col].mode().iloc[0]}")

# 결측치 분석
missing = bureau.isnull().sum()
missing_percent = 100 * missing / len(bureau)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing Percent': missing_percent
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Percent', ascending=False)

print("\n결측치 비율 상위 10개:")
print(missing_df.head(10))

# 대출 활성화 상태별 빈도 (CREDIT_ACTIVE)
plt.figure(figsize=(8, 5))
sns.countplot(x='CREDIT_ACTIVE', data=bureau)
plt.title('Distribution of Credit Active Status')
plt.xlabel('Credit Active Status')
plt.ylabel('Count')
plt.show()

# 대출 유형별 빈도 (CREDIT_TYPE)
plt.figure(figsize=(10, 6))
sns.countplot(x='CREDIT_TYPE', data=bureau)
plt.title('Distribution of Credit Type')
plt.xlabel('Credit Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')  # x축 레이블 회전
plt.show()

# 날짜 관련 변수의 분포 확인: DAYS_CREDIT
plt.figure(figsize=(8, 4))
sns.histplot(bureau["DAYS_CREDIT"], bins=50, kde=True)
plt.title("Distribution of DAYS_CREDIT")
plt.xlabel("DAYS_CREDIT")
plt.tight_layout()
plt.show()

# 수치형 변수 간 상관관계 히트맵

numeric_cols = [
    "DAYS_CREDIT", "DAYS_CREDIT_ENDDATE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
    "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE", "DAYS_CREDIT_UPDATE", "AMT_ANNUITY"
]
corr = bureau[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Selected Numerical Features")
plt.tight_layout()
plt.show()
