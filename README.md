# Team4_term-project
Team4_Writeup  
Home Credit Default Risk | Kaggle
________________________________________
### Objective Setting  
금융기관은 대출을 승인하기 전에 고객이 상환 의무를 신뢰성 있게 이행할 수 있는지, 즉 채무불이행 위험이 있는지 평가합니다. 이 프로젝트의 목표는 외부 재무 데이터뿐만 아니라 다양한 고객 정보(소득, 대출 금액, 연령, 고용 기간 등)를 활용하여 대출 불이행 위험을 예측하는 견고한 분류 모델을 개발하는 것입니다. 또한 신용 위험이 높은 고객을 그룹화하는 클러스터링 모델을 통해 고객의 신용등급을 알 수 있게 합니다.
________________________________________
## Data Inspection  
### application_train.csv - 고객 정보 기반으로 대출 연체 여부(TARGET) 예측  
데이터 개요:  
- 행(Row) 수: 122
- 열(Column) 수: 307511  
Column 데이터 타입별 개수:  
float64:   65  
int64:     41  
object:    16  

#### Distribution of TARGET Variable  
![image](https://github.com/user-attachments/assets/b7ff1500-5eb2-480c-a981-c3cb92cea665)
TARGET=0이 대다수이며, TARGET=1은 약 8% 수준으로 연체자의 수가 매우 적다.  
이는 심각한 클래스 불균형 문제로, 향후 모델 학습 시 Stratify나 Undersampling을 고려해야함.  

#### Distribution of Loan Amount (AMT_CREDIT)  
![image](https://github.com/user-attachments/assets/b41758dc-898c-4683-8f37-065f7e493e56)
대부분의 대출 금액이 30만 ~ 1000만 단위에 몰려 있음.  
positive skewed 있음 → 로그 변환 등 정규화 필요해보임.  

#### Distribution of Contract Type  
![image](https://github.com/user-attachments/assets/0c5bcd7a-bcd4-465b-8f41-2b77727cdbff)
Cash loans가 대부분을 차지하며, Revolving loans는 비교적 적음.  
계약 유형별로 연체율이 다를 수 있어 중요한 범주형 변수로 작용 가능.  

#### Loan Amount by Income Type and Loan Repayment Status  
![image](https://github.com/user-attachments/assets/25acba2c-64f8-4b76-a150-db2216649e5a)

#### Correlation with TARGET - Top 10 Features  
![image](https://github.com/user-attachments/assets/4d656aa0-d65c-4101-98cb-6c6f9a3d9ec9)
EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3와 TARGET 사이에 가장 강한 음의 상관관계 확인됨.  
이 변수들은 외부 신용 점수이며, 높을수록 연체 가능성이 낮음.  

#### Loan Default Rate by Age Group  
![image](https://github.com/user-attachments/assets/8523e223-03ec-43ea-a4f0-3648b456db1e)
젊은 층(20~30대)의 연체율이 상대적으로 높고, 고령층일수록 연체율이 낮음.  
고령층일수록 안정된 금융 상태  

### bureau.csv- 외부 신용 정보(과거 대출 이력)  
데이터 개요:  
- 행(Row) 수: 17  
- 열(Column) 수: 1716428  
Column 데이터 타입별 개수:  
float64:   8  
int64:     6  
object:    3  

#### Distribution of Credit Active Status  
![image](https://github.com/user-attachments/assets/65cb4580-b9cc-4d3b-a6b4-1b25f223a20f)
Closed가 압도적으로 많고, 그 다음이 Active 상태  
이는 대부분의 대출 계좌가 이미 종료되었고, 일부만 현재 활동 중이라는 것을 보임  
Bad debt와 Sold는 상대적으로 매우 드물게 나타나며, 분석 시 다르게 취급하거나 그룹화가 필요  

#### Distribution of Credit Type  
![image](https://github.com/user-attachments/assets/28d66857-171d-4b2d-9062-fffa3d279ed2)
가장 많은 대출 유형은 Consumer credit, 그 다음은 Credit card입니다.  
Mortgage, Car loan 등은 일부 존재하지만 편중 현상이 크기 때문에 향후 분석 시 상위 몇 개만 유지하거나 범주 통합이 필요  
나머지 희귀 대출 유형들은 카테고리 수 감소 또는 기타로 통합하는 방식으로 처리 가능성 고려.  

#### Distribution of DAYS_CREDIT  
![image](https://github.com/user-attachments/assets/9a6cc506-23a5-4dd7-8103-e5500b1e5381)
값이 0에 가까울수록 최근 조회된 계좌, 음의 값일수록 오래된 계좌를 의미.  
대부분의 계좌는 최근 몇 년 이내에 조회되었으며, 멀리 과거로 갈수록 빈도가 줌  
이는 최신 신용 정보에 대한 가중치를 줄 때 유용하게 사용 가능.  

#### Correlation Heatmap of Selected Numerical Features  
![image](https://github.com/user-attachments/assets/9aeec750-d4d2-4dde-8510-023db5b25014)
AMT_CREDIT_SUM ↔ AMT_CREDIT_SUM_DEBT 간의 상관계수가 0.68로 가장 높음 → 이는 전체 대출 잔액이 부채 잔액과 밀접히 연관되어 있음을 의미.  
DAYS_CREDIT ↔ DAYS_CREDIT_UPDATE 간에도 높은 양의 상관관계 (0.69) → 조회 시점과 업데이트 시점 간 간격이 적은 경우가 많음.  
나머지 변수들은 대부분 상관관계가 낮음.  

### previous_application- - 기존 대출 신청 내역  
데이터 개요:  
- 행(Row) 수: 37  
- 열(Column) 수: 1670214  
Column 데이터 타입별 개수:  
float64:   16  
int64:     15  
object:    6  

#### Distribution of Previous Application Status
 
대부분의 과거 신청은 Approved 상태로 처리
Canceled와 Refused 꽤 있음.
고객이 이전에 승인된 대출 경험이 있는지, 반복적으로 거절되었는지에 따라 신용 리스크를 예측 가능

Distribution of Previous Contract Type
 
주로 Cash loans와 Consumer loans가 많고, Revolving loans도 일부 존재
고객이 어떤 유형의 대출을 선호했는지 파악할 수 있으며, 이후 대출 유형 추천에 반영 가능

Mean Application vs Credit Amount by Contract Type
 
모든 계약 유형에서 승인된 금액(AMT_CREDIT)이 신청 금액(AMT_APPLICATION)보다 대부분 크거나 비슷함.
특히 Revolving loans에서는 평균 승인 금액이 신청 금액보다 높음

Application Status by Loan Type
 
Consumer loans는 승인률이 높고 거절 및 취소도 일부 있음.
Cash loans는 승인 외에도 취소율이 꽤 높음.
Revolving loans는 승인, 거절, 취소가 균형 있게 분포됨.
대출 상품별로 승인률 차이가 존재하므로 신용 점수 모델에 loan type을 반영할 수 있음.

Distribution of Days Since Previous Application
최근 신청이 많고, 시간이 오래된 과거 신청은 점차 줌.

Correlation Heatmap of Numerical Features
 
AMT_APPLICATION, AMT_CREDIT, AMT_GOODS_PRICE는 서로 강한 양의 상관관계 
DAYS_LAST_DUE와 DAYS_TERMINATION도 0.93의 높은 상관관계를 보임.



________________________________________
Data Preprocessing
Merge할 table 선정 이유
1. bureau.csv - 외부 신용 정보(과거 대출 이력)
설명: 신청자의 과거 외부 금융기관에서 받은 대출 정보(예: 다른 은행/카드사)
의의:
•	연체 이력(CREDIT_DAY_OVERDUE), 현재 대출 상태(CREDIT_ACTIVE), 총 대출 금액(AMT_CREDIT_SUM) 등의 변수는 신용 위험도 판단에 직결됨.
•	금융기관이 실제로 가장 중요하게 보는 항목: 과거 채무 이행 여부
활용 근거:
과거에 얼마나 자주, 얼마만큼의 대출을 받고 상환했는지를 보면 해당 고객이 책임감 있게 채무를 상환하는 성향인지 판단할 수 있다. 이는 현재 대출의 연체 가능성을 예측하는 데 매우 유의미하다.

2. previous_application.csv - 기존 대출 신청 내역
설명: 고객이 Home Credit에 이전에 신청했던 대출 내역 (승인/거절 포함)
의의:
•	NAME_CONTRACT_STATUS(승인/거절), AMT_APPLICATION, DAYS_DECISION 등으로 과거 신청 행동을 통해 위험도 예측 가능.
•	예를 들어 과거 반복적으로 거절당한 고객은 신용도가 낮을 가능성이 높음.
•	이전 신청 금액 대비 승인 금액의 차이 → 금융기관의 신뢰도 판단 척도
활용 근거:
고객이 과거에 Home Credit에서 어떤 조건으로 대출을 신청했고, 그것이 승인되었는지 거절되었는지를 보면 고객의 경제적 패턴이나 대출 행동의 변화를 파악할 수 있다. 특히 다수의 거절 이력은 신용 위험 신호로 해석될 수 있다.
중요 :
Merge를 위해서는 application_test에도 똑같은 테이블을 merge하여 똑같은 전처리 과정을 거쳐야하지만, bureau와 previous_application은 따로 test 파일이 존재하지 않음. 따라서 모든 테이블을 Merge해서 마지막에 train과 test로 split하는 현재 과정으로서는 test 데이터의 값을 이미 알고 있는 상태에서 학습하게 되는 데이터 누출이 생길 수 있음.
따라서 마지막에 app_train과 app_test로만 학습한 모델과 성능 비교 예정임.

1.	데이터 로딩 및 병합
-	주요 테이블:
application_train.csv(학습 데이터): 307,511개의 행, 122개의 컬럼
application_test.csv(테스트 데이터): 48,744개의 행, 121개의 컬럼
 
-	서브 테이블:
bureau.csv: 외부 신용 정보(과거 대출 이력)
previous_application.csv: 기존 대출 신청 내역
 
-	병합 과정:
-	train과 test를 하나의 데이터프레임으로 합쳐 app으로 통합 (데이터 개요: 356,255개의 행, 122개의 컬럼)
-	bureau와 previous_application 데이터를 각 고객(SK_ID_CURR) 단위로 집계 후 병합
2.	범주형 결측치 처리
-	처리 대상: bureau와 previous_application의 범주형 컬럼 (object 타입)
-	처리 방법: 모든 범주형 결측치를 "Missing" 문자열로 대체
 
3.	수치형 결측치 처리
-	처리 대상: bureau와 previous_application의 수치형 컬럼
-	처리 방법: ID 컬럼 (SK_ID_BUREAU, SK_ID_CURR, SK_ID_PREV)을 제외한 모든 수치형 결측치를 각 컬럼의 평균값으로 대체
 
4.	서브 테이블 집계 및 병합
-	bureau 집계:
-	수치형: 고객 단위로 mean, max, min, sum, std 계산
-	범주형: one-hot 인코딩 후 고객 단위 평균 계산
 
-	previous_application 집계:
-	수치형: 고객 단위로 mean, max, min, sum, std 계산
-	범주형: one-hot 인코딩 후 고객 단위 평균 계산
 
-	병합: 모든 집계 결과를 SK_ID_CURR 기준으로 app 데이터프레임에 병합
 
5.	결측치 처리
-	app 테이블 결측률 50% 이상 컬럼 삭제
-	이때, standard한 application_train 중요 컬럼(TARGET, SK_ID_CURR, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, AMT_ANNUITY)은 보호
-	EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3: 신용 점수 또는 외부 신용정보(신용평가기관에서 제공하는 신용 등급 같은 지표
-	AMT_ANNUITY: 대출 상환 기간 동안 매달 내야 하는 상환 금액(연금 형태)
 
-	범주형 결측치: value 비율에 따라 랜덤으로 채움
 
-	수치형 결측치: 
1)	app 테이블에서 직접 중요 컬럼 선정을 위한 임시 채움 (평균으로만) 상관계수 계산
 
2)	최종 중요 컬럼:
 
2.1)	중요 컬럼: Iterative Imputer로 채움 (랜덤 고정)
 
2.2)	나머지 컬럼: 평균으로 채움
 
6.	인코딩(범주형 컬럼)
-	Label Encoding: 각 범주를 단일 숫자로 변환
 
-	One-Hot Encoding: 각 범주를 별도 이진(binary) 열로 확장
 
-	Ordinal Encoding: 각 범주를 순서가 있는 단일 숫자로 변환
 
ex) NAME_INCOME_TYPE column’
	Label encoding	Ordinal encoding	One-hot encoding
 	 	 	 
7.	피처 셀렉션 (층화 샘플링 + Mutual Info + LGBM)
-	최종 모델 성능에 가장 큰 영향을 미치는 200개의 중요 피처를 선택함
1)	데이터셋 준비(X, y): 
X - TARGET과 ID 컬럼을 제외한 나머지 피처들 
y - TARGET 값만 추출하여 정수형으로 변환
 
2)	층화 샘플링: 데이터가 매우 크므로, 60,000개의 데이터만 선택
 
3)	Mutual Info: 각 피처와 TARGET 간의 상호 정보량을 계산하여 중요도를 평가하고 상위 75%의 Mutual Info 점수를 가진 피처만 선택함
 
4)	LGBM: Mutual Info로 선택된 피처들을 사용하여 LGBM 모델 학습
 
5)	200개의 최종 피처 선택: 
LGBM 모델 학습 후 각 피처의 중요도를 ‘gain’ 기준으로 평가하고 가장 중요한 200개 피처만 선택
* 'gain': “정보 획득 (Information Gain)”을 기준으로 피처의 중요도를 계산
* 정보 획득 (Information Gain): 해당 피처가 모델의 예측 정확도를 얼마나 향상시키는지를 나타냄
 
8.	스케일링
-	MinMax: 0~1 범위로 정규화
-	Robust: 중앙값과 Interquartile range(IQR) 기반 스케일링
-	Standard: 평균 0, 표준편차 1로 정규화
 
	각 인코딩 방식 및 스케일링 조합에 대해 200개의 최종 피처와 SK_ID_CURR, TARGET을 포함한 슬림 데이터셋 생성
 
ex) AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY column
	MinMaxScaler	RobustScaler	StandardScaler
 	 	 	 
9.	Train / Test 데이터로 분리 및 파일 저장
-	슬림 데이터를 원래 train (307,511개의 행)과 test (48,744개의 행)로 분리
 
-	Test 데이터는 TARGET 제거
 
-	파일 저장
 
 
 
________________________________________
Modeling & Evaluation
여러 모델을 학습하고 평가하여 이진 분류 작업에 가장 적합한 데이터 전처리 전략(인코딩 및 스케일링)과 모델을 찾는 것을 목표로 합니다. 추가로, 가장 성능이 좋은 전처리 조합을 선택하여 테스트 데이터에 K-means 클러스터링을 적용하고, 그 결과로 고객들의 신용등급을 시각화합니다.
라이브러리 임포트
•	임포트: pandas(데이터 처리), numpy(수치 연산), re(정규 표현식-이름 정제용), sklearn(모델 선택, 평가지표, 클러스터링, PCA), xgboost, lightgbm(그래디언트 부스팅 모델), matplotlib 및 seaborn(시각화), os(디렉토리 작업), pickle(파이썬 객체 저장/로딩) 등 필요한 라이브러리를 가져옵니다
•	실행 플래그: 
o	RUN_ONLY_STAGE_2: True로 설정하면 1단계(모델 학습)를 건너뛰고, 스크립트는 이전 실행 결과를 로드하여 2단계(클러스터링 및 시각화)를 실행하려고 시도합니다. 
o	PERFORMANCE_FILE: 1단계 결과를 저장/로드할 파일의 이름입니다.


전역 매개변수: 
o	clean_col_names(): 특수 문자를 밑줄(_)로 바꿔 컬럼 이름을 정제하는 함수입니다.
o	encodings, scalers: 반복적으로 적용할 인코딩 및 스케일링 방법들의 목록입니다.
o	N_ESTIMATORS, RANDOM_STATE, N_JOBS, N_SPLITS_KFOLD: 모델 및 교차 검증을 위한 공통 하이퍼파라미터입니다. (N_ESTIMATORS: 추정기 개수, RANDOM_STATE: 난수 시드, N_JOBS: 사용할 CPU 코어 수, N_SPLITS_KFOLD: K-겹 교차 검증 분할 수)
o	PLOTS_DIR: 생성된 시각화 자료를 저장할 디렉토리입니다.
 
1단계: 모델 학습, (계층적 K-겹 교차 검증을 사용한) 평가 및 예측 생성
이 단계는 지정된 인코딩 및 스케일링 방법의 모든 조합을 반복합니다.
1.	데이터 로딩 및 전처리 (각 조합별):
o	전처리된 학습 및 테스트 세트의 파일 이름을 구성합니다 (예: Label_Train_Slim_MinMax.csv).
o	특정 조합에 대한 데이터가 없을 경우 FileNotFoundError를 처리합니다.
o	특성(X_train_full, X_test_final)과 타겟(y_train_full)을 분리합니다.
o	clean_col_names()를 사용하여 컬럼 이름을 정제합니다.
o	테스트 세트(X_test_final)가 학습 세트와 동일한 컬럼을 갖도록 보장하고, 누락된 컬럼은 0으로 채웁니다.
 
 
2.	계층적 K-fold 교차 검증 (각 조합별):
o	StratifiedKFold를 사용하여 교차 검증의 각 폴드(fold)가 전체 데이터셋과 유사한 타겟 클래스 비율을 갖도록 합니다. 이는 불균형 데이터셋에 매우 중요합니다.
 
o	각 폴드에 대해: 
•	전체 학습 데이터를 해당 폴드별 학습 세트(X_train_fold, y_train_fold)와 검증 세트(X_val_fold, y_val_fold)로 분할합니다.
•	scale_pos_weight를 계산합니다: 이 파라미터는 XGBoost 및 LightGBM 분류기에서 소수 클래스에 더 많은 가중치를 부여하여 클래스 불균형을 처리하는 데 사용됩니다. (클래스 0의 개수) / (클래스 1의 개수)로 계산됩니다.
 
•	5개 모델 학습 및 평가: 
-	XGBoost Classifier: ROC AUC 및 F1-점수로 평가됩니다.
-	LightGBM Classifier: ROC AUC 및 F1-점수로 평가됩니다.
-	Random Forest Classifier: 불균형 처리를 위해 class_weight='balanced'를 사용합니다. ROC AUC 및 F1-점수로 평가됩니다.
-	XGBoost Regressor: 출력은 연속적인 값입니다. 분류기로서 평가하기 위해 예측값을 이진화합니다 (임계값 0.5). ROC AUC, (이진화된 예측에 대한) F1-점수 및 RMSE로 평가됩니다.
-	LightGBM Regressor: XGBoost 회귀기와 유사하게 ROC AUC, (이진화된 예측에 대한) F1-점수 및 RMSE로 평가됩니다.
각 모델의 폴드 점수를 해당 목록(예: xgb_clf_fold_aucs)에 추가합니다.

3.	교차 검증 결과 집계 (각 조합별):
o	각 모델에 대해 모든 폴드에 걸쳐 AUC, F1 및 RMSE의 평균과 표준 편차를 계산합니다.
o	이러한 평균 CV 결과를 출력합니다.
o	현재 인코딩/스케일링 조합에 대해 XGBoost, LightGBM 또는 랜덤 포레스트 Classifier가 달성한 최대 평균 Classifier AUC(max_classifier_avg_cv_auc)를 식별합니다
 
4.	조합 성능 저장 (각 조합별):
o	all_combination_performance 목록에 딕셔너리를 추가합니다. 이 딕셔너리에는 다음이 포함됩니다: 
-	인코딩 및 스케일링 방법.
-	max_classifier_avg_cv_auc.
-	다음에 생성될 테스트 세트 예측 파일의 이름.
-	상세한 cv_results_summary (5개 모델 모두에 대한 평균 AUC, F1, RMSE).
 
5.	테스트 세트 예측 (전체 데이터 모델 사용, 각 조합별):
o	교차 검증 후, 5개 모델 각각은 이번에는 전체 X_train_full 및 y_train_full 데이터로 다시 학습됩니다.
o	XGBoost/LightGBM Classifier를 위해 전체 학습 데이터를 사용하여 full_scale_pos_weight가 계산됩니다.
o	이러한 전체 데이터 모델은 X_test_final 데이터에 대한 예측을 생성하는 데 사용됩니다.
o	Classifier 예측(확률) 및 Regressor 예측(연속 값)은 CSV 파일(예: Label_MinMax_XGB_Clf_Probs_predictions.csv)로 저장됩니다.
6.	1단계 결과 저장:
o	all_combination_performance가 비어있지 않으면 피클 파일(PERFORMANCE_FILE)로 저장됩니다.
o	can_proceed_to_stage_2 플래그가 True로 설정됩니다.
 
________________________________________
RUN_ONLY_STAGE_2 = True 처리
o	이 플래그가 설정되면 1단계가 건너뛰어집니다.
o	스크립트는 PERFORMANCE_FILE에서 all_combination_performance를 로드하려고 시도합니다.
o	FileNotFoundError 또는 로드된 파일이 비어있는 경우에 대한 오류 처리가 구현되어 있습니다.
o	성공적인 로딩 여부에 따라 can_proceed_to_stage_2가 설정됩니다.


2단계: 상위 5개 조합에 대한 K-평균 클러스터링 및 시각화
이 단계는 can_proceed_to_stage_2가 참이고 all_combination_performance가 채워져 있는 경우에만 실행됩니다.
1.	상위 조합 선택:
o	all_combination_performance는 max_classifier_val_auc (실제로는 1단계의 최대 평균 CV AUC임)를 기준으로 내림차순으로 정렬됩니다.
o	상위 5개 조합이 선택됩니다.
 
2.	각 상위 조합 처리:
상위 5개 조합 각각에 대해: 
o	시각화를 위해 PLOTS_DIR 아래에 전용 하위 디렉토리가 생성됩니다 (예: model_visualizations/TOP5_CV_Label_MinMax).
o	클러스터링을 위한 데이터 로딩: 
-	K-평균 클러스터링을 위한 특성(X_test_for_clustering)을 얻기 위해 원본 테스트 데이터 파일({enc_method}_Test_Slim_{scale_method}.csv)을 다시 로드합니다.
-	일관성을 보장하기 위해 해당 학습 데이터 파일을 로드하여 컬럼 이름 참조를 얻습니다.
-	이전에 저장된 테스트 세트에 대한 XGBoost Classifier 확률(xgb_clf_probs_file_path)과 XGBoost Regressor예측(xgb_reg_preds_file_path)을 로드합니다.
o	K-means 클러스터링 평가 및 실행: 
-	Elbow Method: 
-	다양한 k 값(2에서 10까지)에 대해 K-means를 실행합니다.
-	각 k에 대해 군집 내 제곱합(WCSS 또는 이너셔)을 계산합니다.
-	WCSS 대 k 그래프(엘보우 플롯)를 생성하고 저장합니다. 이는 적절한 군집 수를 시각적으로 식별하는 데 도움이 됩니다.
-	주요 K-평균 클러스터링: 
-	고정된 n_clusters_chosen = 3으로 K-평균을 수행합니다.
-	X_test_for_clustering의 각 샘플에 군집 레이블이 할당됩니다.
 
-	Silhouette Score: 
군집의 품질을 평가하기 위해 계산됩니다 (객체가 다른 군집에 비해 자체 군집과 얼마나 유사한지). 높을수록 좋습니다.
 
위험 그룹 할당: 
-	군집 레이블은 로드된 XGBoost 분류기 확률(Prob_XGB_Clf_for_ClusterRisk)과 병합됩니다.
-	각 군집에 대한 평균 Prob_XGB_Clf_for_ClusterRisk가 계산됩니다.
-	군집은 이 평균 확률에 따라 정렬됩니다.
-	이 정렬된 순서에 따라 군집은 "낮은 위험", "중간 위험", "높은 위험"으로 레이블이 지정됩니다.
-	SK_ID_CURR, Cluster_Label, Risk_Group 및 Prob_XGB_Clf_for_ClusterRisk가 CSV 파일로 저장됩니다.
 
-	시각화: 
-	XGBoost Regressor 예측 분포: 전체 XGBoost 회귀 모델의 테스트 세트 예측 분포를 보여주는 히스토그램입니다.
-	위험 그룹별 XGBoost Regressor 예측: K-평균으로 정의된 여러 위험 그룹에 걸쳐 XGBoost Regressor 예측을 비교하는 상자 그림(boxplot)입니다.
-	위험 그룹별 고객 수: 각 위험 그룹의 고객 수를 보여주는 막대 그래프(countplot)입니다.
-	위험 그룹별 XGBoost Classifier 확률: K-평균으로 정의된 여러 위험 그룹에 걸쳐 XGBoost Classifier 확률을 비교하는 상자 그림입니다.
-	클러스터의 PCA 플롯: 
-	주성분 분석(PCA)을 적용하여 X_test_for_clustering의 차원을 2개의 구성 요소로 줄입니다.
-	이 두 주성분에 대한 산점도(scatter plot)가 생성됩니다. 점들은 Risk_Group(사용 가능하고 구별되는 경우) 또는 Cluster_Label에 따라 색상이 지정됩니다. 마커 스타일도 클러스터를 구분할 수 있습니다. 이는 2D 공간에서 클러스터를 시각화합니다.
-	파일 로딩, PCA 및 실루엣 점수 계산에 대한 오류 처리가 포함되어 있습니다.
________________________________________
Output :
•	1단계: 
o	각 모델 및 데이터 조합에 대한 학습 과정 및 CV 결과를 자세히 보여주는 콘솔 출력.
o	각 모델 및 데이터 조합에 대한 테스트 세트 예측
o	하나의 인코딩-스케일링 조합에 대한 성능 요약 및 예측 파일 경로를 담은 딕셔너리 목록을 저장하는 피클 파일(all_combination_performance_cv.pkl).
•	2단계: 
o	상위 5개 조합에 대한 클러스터링 과정을 자세히 보여주는 콘솔 출력.
o	각 상위 조합에 대해 model_visualizations/ 아래의 하위 디렉토리에 다음이 포함됩니다: 
-	_Elbow_Method.png: WCSS 대 군집 수 그래프.
-	_KMeans_Clusters.csv: SK_ID_CURR, Cluster_Label, Risk_Group 및 XGBoost Classifier 확률을 포함하는 CSV.
-	_XGB_Reg_Test_Pred_Dist.png: XGBoost Regressor 예측의 히스토그램.
-	_XGB_Reg_Pred_per_RiskGroup.png: 위험 그룹별 XGBoost Regressor 예측의 상자 그림.
-	_Cluster_Sizes.png: 위험 그룹별 고객 수의 막대 그래프.
-	_Prob_Dist_per_RiskGroup.png: 위험 그룹별 XGBoost Classifier 확률의 상자 그림.
-	_PCA_Clusters_RiskGroup.png: 위험 그룹별로 색상이 지정된 클러스터의 2D PCA 산점도.
#각 폴드마다 classifier AUC F1 점수
 ![image](https://github.com/user-attachments/assets/f282d064-bf22-46a1-a6ff-71ccc7187de2)

#5번 폴드 후 classifier의 평균 AUC, F1 점수
![image](https://github.com/user-attachments/assets/86ee25f7-8581-489e-9d2e-0070acd777a4)

#클러스터링 결과
 ![image](https://github.com/user-attachments/assets/f8d80fea-29d7-4645-94d3-7f41022a3504)

사진 첨부
 ![image](https://github.com/user-attachments/assets/e9bad8f2-322f-4c04-bf26-b2f1015a60ae)
 ![image](https://github.com/user-attachments/assets/8aade836-5f67-4a51-bddd-295169a8ebba)
 ![image](https://github.com/user-attachments/assets/d035dc66-4873-414e-9b9d-06231c76f786)

테이블을 merge 했을 경우 :
 ![image](https://github.com/user-attachments/assets/c3390995-3c20-473c-8482-d4fbe142d57a)

테이블을 merge 하지 않았을 경우 :
 ![image](https://github.com/user-attachments/assets/de7865cc-1d06-47f8-8ef5-a004d38702b3)

Merge 했을 경우가 AUC가 살짝 높지만, 데이터 누출 위험이 있음.
테이블을 merge하지 않고 One-Hot encoding, Robust Scaling을 적용한 application_train, application_test만을 사용한 Xgboost / LightGBM 비교하여 best model 선정 

-LightGBM
분류기와 Hyperparameter Grid 정의 :
  ![image](https://github.com/user-attachments/assets/6961423e-08e3-40b5-8f49-953416db5044)

Staratify, k-fold 적용 :
 ![image](https://github.com/user-attachments/assets/ec5ef031-f3c1-4253-82b1-0faf966ef960)

Hyperparameter 튜닝 결과 :
 ![image](https://github.com/user-attachments/assets/c06398a7-9aa5-4290-9ac0-9e1f51c8e3ce)


채택된 Hyperparameter로 재학습 :
 ![image](https://github.com/user-attachments/assets/42808f4f-5af9-49d2-a691-a18c9eb7e35b)

실제 테스트 셋으로 예측 :
 ![image](https://github.com/user-attachments/assets/ff49b43c-d3e8-4cff-8c7f-13363cb561af)

-XGBoost
분류기와 Hyperparameter Grid 정의 :
 ![image](https://github.com/user-attachments/assets/1f6e42ab-a0b6-4cfa-8c9d-262f81c06c61)

Staratify, k-fold 적용 :
 ![image](https://github.com/user-attachments/assets/ef3f84ea-3c93-475d-a226-e50955cc9d95)

Hyperparameter 튜닝 결과 :
 ![image](https://github.com/user-attachments/assets/75dc3de7-a937-4e22-822b-84c4318e2618)

채택된 Hyperparameter로 재학습 :
 ![image](https://github.com/user-attachments/assets/29e7bace-0756-4b96-8f37-b72e1a0cd7e3)

실제 테스트 셋으로 예측 :
 ![image](https://github.com/user-attachments/assets/cecda7da-ac83-433e-adc7-f93273d3f458)



