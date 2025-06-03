import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, silhouette_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# --- SCRIPT EXECUTION FLAGS ---
RUN_ONLY_STAGE_2 = False
PERFORMANCE_FILE = "all_combination_performance_cv.pkl"
# --- END SCRIPT EXECUTION FLAGS ---

# Define a function to clean column names
def clean_col_names(cols):
    return [re.sub(r'[^0-9A-Za-z_]', '_', c) for c in cols]

# Define encoding and scaling methods
encodings = ["Label", "OneHot", "Ordinal"]
scalers = ["MinMax", "Robust", "Standard"]

# Common model parameters
N_ESTIMATORS = 100
RANDOM_STATE = 42
N_JOBS = -1
N_SPLITS_KFOLD = 5

PLOTS_DIR = "model_visualizations"
os.makedirs(PLOTS_DIR, exist_ok=True)

print("## Starting Model Training, Evaluation (with Stratified K-Fold), Selective Clustering (with Elbow & Silhouette), and Visualization Pipeline")
print("---")

all_combination_performance = []
can_proceed_to_stage_2 = False

if not RUN_ONLY_STAGE_2:
    print("\n--- STAGE 1: Training models and generating predictions for ALL combinations (using Stratified K-Fold for evaluation) ---")
    for enc_method in encodings:
        for scale_method in scalers:
            print(f"\n### Processing Combination: Encoding: {enc_method}, Scaling: {scale_method}")
            print("---")

            train_file = f"{enc_method}_Train_Slim_{scale_method}.csv"
            test_file = f"{enc_method}_Test_Slim_{scale_method}.csv"

            try:
                df_train_full = pd.read_csv(train_file)
                df_test_full = pd.read_csv(test_file)
            except FileNotFoundError:
                print(f"Files for {enc_method} - {scale_method} not found. Skipping.")
                if not os.path.exists(train_file): print(f" Missing train file: {train_file}")
                if not os.path.exists(test_file): print(f" Missing test file: {test_file}")
                print("---")
                continue

            print(f"Loaded {train_file} (Shape: {df_train_full.shape})")
            print(f"Loaded {test_file} (Shape: {df_test_full.shape})")

            y_train_full = df_train_full['TARGET']
            X_train_full = df_train_full.drop(columns=['SK_ID_CURR', 'TARGET'])
            X_train_full.columns = clean_col_names(X_train_full.columns)

            sk_id_curr_test = df_test_full['SK_ID_CURR']
            X_test_final = df_test_full.drop(columns=['SK_ID_CURR'])
            X_test_final.columns = clean_col_names(X_test_final.columns)
            X_test_final = X_test_final.reindex(columns=X_train_full.columns, fill_value=0)

            print(f"\nFull training set shape: {X_train_full.shape}")

            skf = StratifiedKFold(n_splits=N_SPLITS_KFOLD, shuffle=True, random_state=RANDOM_STATE)

            xgb_clf_fold_aucs, xgb_clf_fold_f1s = [], []
            lgbm_clf_fold_aucs, lgbm_clf_fold_f1s = [], []
            rf_clf_fold_aucs, rf_clf_fold_f1s = [], []
            xgb_reg_fold_aucs, xgb_reg_fold_f1s, xgb_reg_fold_rmses = [], [], []
            lgbm_reg_fold_aucs, lgbm_reg_fold_f1s, lgbm_reg_fold_rmses = [], [], []

            print(f"\nStarting Stratified {N_SPLITS_KFOLD}-Fold Cross-Validation...")
            for fold_num, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
                print(f"\n--- Fold {fold_num + 1}/{N_SPLITS_KFOLD} ---") # Print current fold
                X_train_fold, X_val_fold = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
                y_train_fold, y_val_fold = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

                scale_pos_weight_fold = np.sum(y_train_fold == 0) / (np.sum(y_train_fold == 1) + 1e-6)
                print(f"Class distribution in this training fold: 0s={np.sum(y_train_fold == 0)}, 1s={np.sum(y_train_fold == 1)}")
                print(f"Calculated scale_pos_weight for this fold: {scale_pos_weight_fold:.2f}")

                # XGBoost Classifier
                print("Training XGBoost Classifier (Fold)...")
                xgb_clf_fold = xgb.XGBClassifier(n_estimators=N_ESTIMATORS, scale_pos_weight=scale_pos_weight_fold, random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss', n_jobs=N_JOBS)
                xgb_clf_fold.fit(X_train_fold, y_train_fold)
                y_pred_proba_xgb_clf_fold = xgb_clf_fold.predict_proba(X_val_fold)[:, 1]
                y_pred_xgb_clf_fold = xgb_clf_fold.predict(X_val_fold)
                fold_auc_xgb = roc_auc_score(y_val_fold, y_pred_proba_xgb_clf_fold)
                fold_f1_xgb = f1_score(y_val_fold, y_pred_xgb_clf_fold)
                xgb_clf_fold_aucs.append(fold_auc_xgb)
                xgb_clf_fold_f1s.append(fold_f1_xgb)
                print(f"XGB Classifier - Fold {fold_num+1} Val AUC: {fold_auc_xgb:.4f}, F1: {fold_f1_xgb:.4f}")


                # LightGBM Classifier
                print("Training LightGBM Classifier (Fold)...")
                lgbm_clf_fold = lgb.LGBMClassifier(n_estimators=N_ESTIMATORS, scale_pos_weight=scale_pos_weight_fold, random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1)
                lgbm_clf_fold.fit(X_train_fold, y_train_fold)
                y_pred_proba_lgbm_clf_fold = lgbm_clf_fold.predict_proba(X_val_fold)[:, 1]
                y_pred_lgbm_clf_fold = lgbm_clf_fold.predict(X_val_fold)
                fold_auc_lgbm = roc_auc_score(y_val_fold, y_pred_proba_lgbm_clf_fold)
                fold_f1_lgbm = f1_score(y_val_fold, y_pred_lgbm_clf_fold)
                lgbm_clf_fold_aucs.append(fold_auc_lgbm)
                lgbm_clf_fold_f1s.append(fold_f1_lgbm)
                print(f"LGBM Classifier - Fold {fold_num+1} Val AUC: {fold_auc_lgbm:.4f}, F1: {fold_f1_lgbm:.4f}")

                # Random Forest Classifier
                print("Training Random Forest Classifier (Fold)...")
                rf_clf_fold = RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=N_JOBS)
                rf_clf_fold.fit(X_train_fold, y_train_fold)
                y_pred_proba_rf_clf_fold = rf_clf_fold.predict_proba(X_val_fold)[:, 1]
                y_pred_rf_clf_fold = rf_clf_fold.predict(X_val_fold)
                fold_auc_rf = roc_auc_score(y_val_fold, y_pred_proba_rf_clf_fold)
                fold_f1_rf = f1_score(y_val_fold, y_pred_rf_clf_fold)
                rf_clf_fold_aucs.append(fold_auc_rf)
                rf_clf_fold_f1s.append(fold_f1_rf)
                print(f"RF Classifier - Fold {fold_num+1} Val AUC: {fold_auc_rf:.4f}, F1: {fold_f1_rf:.4f}")

                # XGBoost Regressor
                print("Training XGBoost Regressor (Fold)...")
                xgb_reg_fold = xgb.XGBRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, objective='reg:squarederror', eval_metric='rmse', n_jobs=N_JOBS)
                xgb_reg_fold.fit(X_train_fold, y_train_fold)
                y_pred_xgb_reg_fold = xgb_reg_fold.predict(X_val_fold)
                y_pred_xgb_reg_fold_binary = (y_pred_xgb_reg_fold > 0.5).astype(int)
                fold_auc_xgb_reg = roc_auc_score(y_val_fold, y_pred_xgb_reg_fold)
                fold_f1_xgb_reg = f1_score(y_val_fold, y_pred_xgb_reg_fold_binary)
                fold_rmse_xgb_reg = np.sqrt(mean_squared_error(y_val_fold, y_pred_xgb_reg_fold))
                xgb_reg_fold_aucs.append(fold_auc_xgb_reg)
                xgb_reg_fold_f1s.append(fold_f1_xgb_reg)
                xgb_reg_fold_rmses.append(fold_rmse_xgb_reg)
                print(f"XGB Regressor - Fold {fold_num+1} Val AUC: {fold_auc_xgb_reg:.4f}, F1 (thresh@0.5): {fold_f1_xgb_reg:.4f}, RMSE: {fold_rmse_xgb_reg:.4f}")

                # LightGBM Regressor
                print("Training LightGBM Regressor (Fold)...")
                lgbm_reg_fold = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, objective='regression_l2', metric='rmse', n_jobs=N_JOBS, verbose=-1)
                lgbm_reg_fold.fit(X_train_fold, y_train_fold)
                y_pred_lgbm_reg_fold = lgbm_reg_fold.predict(X_val_fold)
                y_pred_lgbm_reg_fold_binary = (y_pred_lgbm_reg_fold > 0.5).astype(int)
                fold_auc_lgbm_reg = roc_auc_score(y_val_fold, y_pred_lgbm_reg_fold)
                fold_f1_lgbm_reg = f1_score(y_val_fold, y_pred_lgbm_reg_fold_binary)
                fold_rmse_lgbm_reg = np.sqrt(mean_squared_error(y_val_fold, y_pred_lgbm_reg_fold))
                lgbm_reg_fold_aucs.append(fold_auc_lgbm_reg)
                lgbm_reg_fold_f1s.append(fold_f1_lgbm_reg)
                lgbm_reg_fold_rmses.append(fold_rmse_lgbm_reg)
                print(f"LGBM Regressor - Fold {fold_num+1} Val AUC: {fold_auc_lgbm_reg:.4f}, F1 (thresh@0.5): {fold_f1_lgbm_reg:.4f}, RMSE: {fold_rmse_lgbm_reg:.4f}")

                if (fold_num + 1) == N_SPLITS_KFOLD :
                     print(f"\nCompleted All {N_SPLITS_KFOLD} Folds for {enc_method}-{scale_method}")


            results_cv = {}
            results_cv['XGB_Classifier'] = {'AUC': np.mean(xgb_clf_fold_aucs), 'F1': np.mean(xgb_clf_fold_f1s)}
            results_cv['LGBM_Classifier'] = {'AUC': np.mean(lgbm_clf_fold_aucs), 'F1': np.mean(lgbm_clf_fold_f1s)}
            results_cv['RF_Classifier'] = {'AUC': np.mean(rf_clf_fold_aucs), 'F1': np.mean(rf_clf_fold_f1s)}
            results_cv['XGB_Regressor'] = {'AUC': np.mean(xgb_reg_fold_aucs), 'F1': np.mean(xgb_reg_fold_f1s), 'RMSE': np.mean(xgb_reg_fold_rmses)}
            results_cv['LGBM_Regressor'] = {'AUC': np.mean(lgbm_reg_fold_aucs), 'F1': np.mean(lgbm_reg_fold_f1s), 'RMSE': np.mean(lgbm_reg_fold_rmses)}

            # --- 여기가 수정된 부분: 평균 교차 검증 결과 출력 ---
            print("\n--- Average Cross-Validation Results ---")
            print(f"XGB Classifier  - Avg CV AUC: {results_cv['XGB_Classifier']['AUC']:.4f} (Std: {np.std(xgb_clf_fold_aucs):.4f}), Avg F1: {results_cv['XGB_Classifier']['F1']:.4f} (Std: {np.std(xgb_clf_fold_f1s):.4f})")
            print(f"LGBM Classifier - Avg CV AUC: {results_cv['LGBM_Classifier']['AUC']:.4f} (Std: {np.std(lgbm_clf_fold_aucs):.4f}), Avg F1: {results_cv['LGBM_Classifier']['F1']:.4f} (Std: {np.std(lgbm_clf_fold_f1s):.4f})")
            print(f"RF Classifier   - Avg CV AUC: {results_cv['RF_Classifier']['AUC']:.4f} (Std: {np.std(rf_clf_fold_aucs):.4f}), Avg F1: {results_cv['RF_Classifier']['F1']:.4f} (Std: {np.std(rf_clf_fold_f1s):.4f})")
            print(f"XGB Regressor   - Avg CV AUC: {results_cv['XGB_Regressor']['AUC']:.4f} (Std: {np.std(xgb_reg_fold_aucs):.4f}), Avg F1: {results_cv['XGB_Regressor']['F1']:.4f} (Std: {np.std(xgb_reg_fold_f1s):.4f}), Avg RMSE: {results_cv['XGB_Regressor']['RMSE']:.4f} (Std: {np.std(xgb_reg_fold_rmses):.4f})")
            print(f"LGBM Regressor  - Avg CV AUC: {results_cv['LGBM_Regressor']['AUC']:.4f} (Std: {np.std(lgbm_reg_fold_aucs):.4f}), Avg F1: {results_cv['LGBM_Regressor']['F1']:.4f} (Std: {np.std(lgbm_reg_fold_f1s):.4f}), Avg RMSE: {results_cv['LGBM_Regressor']['RMSE']:.4f} (Std: {np.std(lgbm_reg_fold_rmses):.4f})")
            # --- 여기까지 수정된 부분 ---

            max_classifier_avg_cv_auc = 0.0
            if results_cv['XGB_Classifier']['AUC'] > max_classifier_avg_cv_auc: max_classifier_avg_cv_auc = results_cv['XGB_Classifier']['AUC']
            if results_cv['LGBM_Classifier']['AUC'] > max_classifier_avg_cv_auc: max_classifier_avg_cv_auc = results_cv['LGBM_Classifier']['AUC']
            if results_cv['RF_Classifier']['AUC'] > max_classifier_avg_cv_auc: max_classifier_avg_cv_auc = results_cv['RF_Classifier']['AUC']

            all_combination_performance.append({
                'enc_method': enc_method,
                'scale_method': scale_method,
                'max_classifier_val_auc': max_classifier_avg_cv_auc,
                'xgb_clf_prob_file': f"{enc_method}_{scale_method}_XGB_Clf_Probs_predictions.csv",
                'xgb_reg_pred_file': f"{enc_method}_{scale_method}_XGB_Reg_predictions.csv",
                'cv_results_summary': results_cv
            })

            # --- Test Set Predictions (Full Data Models) ---
            print("\nGenerating Test Set Predictions for current combination (Full Data Models)...") # Added newline for better separation
            full_scale_pos_weight = np.sum(y_train_full == 0) / (np.sum(y_train_full == 1) + 1e-6)
            print(f"Calculated scale_pos_weight for full data classifiers: {full_scale_pos_weight:.2f}")

            print("Training XGBoost Classifier (Full Data)...")
            xgb_clf_full = xgb.XGBClassifier(n_estimators=N_ESTIMATORS, scale_pos_weight=full_scale_pos_weight, random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss', n_jobs=N_JOBS)
            xgb_clf_full.fit(X_train_full, y_train_full)
            xgb_test_clf_probs = xgb_clf_full.predict_proba(X_test_final)[:, 1]
            pd.DataFrame({'SK_ID_CURR': sk_id_curr_test, 'TARGET_Prob': xgb_test_clf_probs}).to_csv(f"{enc_method}_{scale_method}_XGB_Clf_Probs_predictions.csv", index=False)
            # For brevity, I'm not repeating all full model trainings and savings, assume they are correct as in your script.
            # Make sure to add print statements before each full model training if you want to see "Training LGBM Classifier (Full Data)..." etc.

            print("Training LightGBM Classifier (Full Data)...")
            lgbm_clf_full = lgb.LGBMClassifier(n_estimators=N_ESTIMATORS, scale_pos_weight=full_scale_pos_weight, random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1)
            lgbm_clf_full.fit(X_train_full, y_train_full)
            lgbm_test_clf_probs = lgbm_clf_full.predict_proba(X_test_final)[:, 1]
            pd.DataFrame({'SK_ID_CURR': sk_id_curr_test, 'TARGET_Prob': lgbm_test_clf_probs}).to_csv(f"{enc_method}_{scale_method}_LGBM_Clf_Probs_predictions.csv", index=False)

            print("Training Random Forest Classifier (Full Data)...")
            rf_clf_full = RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=N_JOBS)
            rf_clf_full.fit(X_train_full, y_train_full)
            rf_test_clf_probs = rf_clf_full.predict_proba(X_test_final)[:, 1]
            pd.DataFrame({'SK_ID_CURR': sk_id_curr_test, 'TARGET_Prob': rf_test_clf_probs}).to_csv(f"{enc_method}_{scale_method}_RF_Clf_Probs_predictions.csv", index=False)

            print("Training XGBoost Regressor (Full Data)...")
            xgb_reg_full = xgb.XGBRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, objective='reg:squarederror', eval_metric='rmse', n_jobs=N_JOBS)
            xgb_reg_full.fit(X_train_full, y_train_full)
            xgb_test_reg_preds = xgb_reg_full.predict(X_test_final)
            pd.DataFrame({'SK_ID_CURR': sk_id_curr_test, 'TARGET_RegPred': xgb_test_reg_preds}).to_csv(f"{enc_method}_{scale_method}_XGB_Reg_predictions.csv", index=False)

            print("Training LightGBM Regressor (Full Data)...")
            lgbm_reg_full = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, objective='regression_l2', metric='rmse', n_jobs=N_JOBS, verbose=-1)
            lgbm_reg_full.fit(X_train_full, y_train_full)
            lgbm_test_reg_preds = lgbm_reg_full.predict(X_test_final)
            pd.DataFrame({'SK_ID_CURR': sk_id_curr_test, 'TARGET_RegPred': lgbm_test_reg_preds}).to_csv(f"{enc_method}_{scale_method}_LGBM_Reg_predictions.csv", index=False)

            print(f"All predictions for {enc_method}-{scale_method} saved.")
            print("---")

    if all_combination_performance:
        with open(PERFORMANCE_FILE, "wb") as f:
            pickle.dump(all_combination_performance, f)
        print(f"\nINFO: Stage 1 results (all_combination_performance with CV) saved to {PERFORMANCE_FILE}")
        can_proceed_to_stage_2 = True
    else:
        print("\nWARNING: Stage 1 did not produce any results to save (all_combination_performance is empty).")
        can_proceed_to_stage_2 = False
else: # RUN_ONLY_STAGE_2 is True
    print(f"\n--- SKIPPING STAGE 1: Attempting to load results from {PERFORMANCE_FILE} for STAGE 2 ---")
    try:
        with open(PERFORMANCE_FILE, "rb") as f:
            all_combination_performance = pickle.load(f)
        if not all_combination_performance:
            print(f"ERROR: Loaded performance file {PERFORMANCE_FILE} is empty or contains no data.")
            print("Cannot proceed with Stage 2. Please run Stage 1 first.")
            can_proceed_to_stage_2 = False
        else:
            print(f"INFO: Successfully loaded {len(all_combination_performance)} combination results from {PERFORMANCE_FILE}.")
            can_proceed_to_stage_2 = True
    except FileNotFoundError:
        print(f"ERROR: Performance file {PERFORMANCE_FILE} not found.")
        print("Cannot proceed with Stage 2. Please run Stage 1 first to generate this file.")
        can_proceed_to_stage_2 = False
    except Exception as e:
        print(f"ERROR: Could not load performance file {PERFORMANCE_FILE}. Error: {e}")
        print("Cannot proceed with Stage 2.")
        can_proceed_to_stage_2 = False

# --- STAGE 2: K-Means Clustering and Visualization for Top 5 Combinations ---
if can_proceed_to_stage_2 and all_combination_performance:
    print("\n--- STAGE 2: Performing K-Means Clustering (with Elbow & Silhouette) and Visualization for TOP 5 combinations ---")

    sorted_combinations = sorted(all_combination_performance, key=lambda x: x['max_classifier_val_auc'], reverse=True)
    top_5_combinations = sorted_combinations[:5]

    print(f"\nTop 5 combinations selected based on max average cross-validated AUC of classifiers:")
    if not top_5_combinations:
        print("No combinations were processed or loaded successfully to select top 5 for Stage 2.")
    else:
        for i, combo_info in enumerate(top_5_combinations):
            print(f"{i+1}. Encoding: {combo_info['enc_method']}, Scaling: {combo_info['scale_method']}, Max Avg CV AUC: {combo_info['max_classifier_val_auc']:.4f}")
            # Optionally print more details from cv_results_summary if needed:
            # print(f"   CV Details for {combo_info['enc_method']}-{combo_info['scale_method']}:")
            # for model_name, metrics in combo_info['cv_results_summary'].items():
            #     print(f"     {model_name}: {metrics}")


    for combo_info in top_5_combinations:
        enc_method = combo_info['enc_method']
        scale_method = combo_info['scale_method']
        xgb_clf_probs_file_path = combo_info['xgb_clf_prob_file']
        xgb_reg_preds_file_path = combo_info['xgb_reg_pred_file']

        combo_plot_dir = os.path.join(PLOTS_DIR, f"TOP5_CV_{enc_method}_{scale_method}")
        os.makedirs(combo_plot_dir, exist_ok=True)

        print(f"\n--- Processing Top Combination for Clustering & Viz: Encoding: {enc_method}, Scaling: {scale_method} ---")

        train_file_for_cols = f"{enc_method}_Train_Slim_{scale_method}.csv"
        test_file_for_clustering = f"{enc_method}_Test_Slim_{scale_method}.csv"

        try:
            df_train_for_cols = pd.read_csv(train_file_for_cols)
            X_train_cols_ref_orig = df_train_for_cols.drop(columns=['SK_ID_CURR', 'TARGET']).columns
            X_train_cols_ref_cleaned = clean_col_names(X_train_cols_ref_orig)

            df_test_for_clustering_full = pd.read_csv(test_file_for_clustering)
            current_sk_id_curr_test = df_test_for_clustering_full['SK_ID_CURR']
            X_test_for_clustering = df_test_for_clustering_full.drop(columns=['SK_ID_CURR'])
            X_test_for_clustering.columns = clean_col_names(X_test_for_clustering.columns)
            X_test_for_clustering = X_test_for_clustering.reindex(columns=X_train_cols_ref_cleaned, fill_value=0)

            df_xgb_clf_probs_test = pd.read_csv(xgb_clf_probs_file_path)
            df_xgb_reg_preds_test = pd.read_csv(xgb_reg_preds_file_path)

        except FileNotFoundError:
            print(f"Data or prediction files for {enc_method}-{scale_method} not found for clustering/visualization. Skipping this top combination.")
            if not os.path.exists(train_file_for_cols): print(f" Missing: {train_file_for_cols}")
            if not os.path.exists(test_file_for_clustering): print(f" Missing: {test_file_for_clustering}")
            if not os.path.exists(xgb_clf_probs_file_path): print(f" Missing: {xgb_clf_probs_file_path}")
            if not os.path.exists(xgb_reg_preds_file_path): print(f" Missing: {xgb_reg_preds_file_path}")
            continue
        except Exception as e:
            print(f"Error loading data for {enc_method}-{scale_method}: {e}. Skipping this top combination.")
            continue

        # --- K-Means Clustering Evaluation & Execution ---
        df_clusters = None
        cluster_labels = None

        if X_test_for_clustering.shape[0] > 1 and X_test_for_clustering.shape[1] > 0:
            # 1. Elbow Method
            print(f"Performing Elbow Method for K-Means on {enc_method}-{scale_method}...")
            wcss = []
            k_range = range(2, 11)
            for k_val in k_range:
                kmeans_elbow = KMeans(n_clusters=k_val, random_state=RANDOM_STATE, n_init='auto')
                kmeans_elbow.fit(X_test_for_clustering)
                wcss.append(kmeans_elbow.inertia_)

            plt.figure(figsize=(10, 6))
            plt.plot(k_range, wcss, marker='o', linestyle='--')
            plt.title(f'Elbow Method for Optimal K\n{enc_method} - {scale_method}')
            plt.xlabel('Number of Clusters (K)')
            plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
            plt.xticks(list(k_range))
            plt.grid(True)
            elbow_plot_path = os.path.join(combo_plot_dir, f"TOP5_CV_{enc_method}_{scale_method}_Elbow_Method.png")
            plt.savefig(elbow_plot_path)
            plt.close()
            print(f"Elbow method plot saved to: {elbow_plot_path}")

            # 2. Main K-Means Clustering
            n_clusters_chosen = 3
            print(f"Performing K-Means Clustering with n_clusters={n_clusters_chosen} for {enc_method}-{scale_method}...")
            kmeans = KMeans(n_clusters=n_clusters_chosen, random_state=RANDOM_STATE, n_init='auto')
            cluster_labels = kmeans.fit_predict(X_test_for_clustering)
            df_clusters = pd.DataFrame({'SK_ID_CURR': current_sk_id_curr_test, 'Cluster_Label': cluster_labels})

            # 3. Silhouette Score
            if cluster_labels is not None and len(np.unique(cluster_labels)) > 1 and len(np.unique(cluster_labels)) < X_test_for_clustering.shape[0]:
                try:
                    sil_score = silhouette_score(X_test_for_clustering, cluster_labels)
                    print(f"Silhouette Score (for n_clusters={n_clusters_chosen}): {sil_score:.4f}")
                except ValueError as ve:
                    print(f"Could not calculate Silhouette Score for n_clusters={n_clusters_chosen}: {ve}")
            else:
                print(f"Silhouette Score cannot be computed for n_clusters={n_clusters_chosen} (not enough unique clusters or samples).")

            if df_clusters is not None:
                df_clusters = pd.merge(df_clusters, df_xgb_clf_probs_test[['SK_ID_CURR', 'TARGET_Prob']], on='SK_ID_CURR', how='left')
                df_clusters.rename(columns={'TARGET_Prob': 'Prob_XGB_Clf_for_ClusterRisk'}, inplace=True)

                if 'Prob_XGB_Clf_for_ClusterRisk' in df_clusters.columns and not df_clusters['Prob_XGB_Clf_for_ClusterRisk'].isnull().all():
                    cluster_risk_profile = df_clusters.groupby('Cluster_Label')['Prob_XGB_Clf_for_ClusterRisk'].mean().sort_values()
                    risk_group_map = {}
                    if len(cluster_risk_profile) == n_clusters_chosen:
                        risk_levels = ['Low Risk', 'Medium Risk', 'High Risk']
                        if n_clusters_chosen == 2: risk_levels = ['Low Risk', 'High Risk']
                        for i, cluster_idx in enumerate(cluster_risk_profile.index):
                             if i < len(risk_levels):
                                risk_group_map[cluster_idx] = risk_levels[i]
                             else:
                                risk_group_map[cluster_idx] = f"Risk Group {i+1}"
                    df_clusters['Risk_Group'] = df_clusters['Cluster_Label'].map(risk_group_map)
                    print("Customer Segmentation by K-Means (based on XGB Classifier risk from full data model):")
                    print(df_clusters['Risk_Group'].value_counts())
                    df_clusters[['SK_ID_CURR', 'Cluster_Label', 'Risk_Group', 'Prob_XGB_Clf_for_ClusterRisk']].to_csv(os.path.join(combo_plot_dir, f"TOP5_CV_{enc_method}_{scale_method}_KMeans_Clusters.csv"), index=False)
                else:
                    print(f"Skipping K-Means risk labeling for {enc_method}-{scale_method}: Prob_XGB_Clf_for_ClusterRisk not found/all null.")
                    if df_clusters is not None:
                         df_clusters[['SK_ID_CURR', 'Cluster_Label']].to_csv(os.path.join(combo_plot_dir, f"TOP5_CV_{enc_method}_{scale_method}_KMeans_Clusters_Basic.csv"), index=False)
        else:
            print(f"Skipping K-Means clustering and evaluation for {enc_method}-{scale_method} due to insufficient samples/features in X_test_for_clustering (Samples: {X_test_for_clustering.shape[0]}, Features: {X_test_for_clustering.shape[1]}).")

        # --- Visualizations ---
        if df_clusters is not None and 'SK_ID_CURR' in df_clusters.columns and cluster_labels is not None:
            print(f"Generating visualizations for {enc_method}-{scale_method}...")

            df_reg_preds_with_risk = pd.merge(df_clusters, df_xgb_reg_preds_test[['SK_ID_CURR', 'TARGET_RegPred']], on='SK_ID_CURR', how='left')
            if 'TARGET_RegPred' in df_reg_preds_with_risk.columns and not df_reg_preds_with_risk['TARGET_RegPred'].isnull().all():
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df_reg_preds_with_risk, x='TARGET_RegPred', kde=True, bins=50)
                plt.title(f'Distribution of XGB Regressor Test Predictions (Full Model)\n{enc_method} - {scale_method}')
                plt.xlabel('Predicted Value (XGB Regressor)')
                plt.ylabel('Frequency')
                plt.tight_layout()
                plt.savefig(os.path.join(combo_plot_dir, f"TOP5_CV_{enc_method}_{scale_method}_XGB_Reg_Test_Pred_Dist.png"))
                plt.close()

                if 'Risk_Group' in df_reg_preds_with_risk.columns and not df_reg_preds_with_risk['Risk_Group'].isnull().all() and df_reg_preds_with_risk['Risk_Group'].nunique() > 1:
                    plt.figure(figsize=(12, 7))
                    sns.boxplot(data=df_reg_preds_with_risk, x='Risk_Group', y='TARGET_RegPred', order=sorted(df_reg_preds_with_risk['Risk_Group'].dropna().unique()))
                    plt.title(f'XGB Regressor Test Predictions by K-Means Risk Group (Full Model)\n{enc_method} - {scale_method}')
                    plt.xlabel('K-Means Defined Risk Group')
                    plt.ylabel('Predicted Value (XGB Regressor)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(combo_plot_dir, f"TOP5_CV_{enc_method}_{scale_method}_XGB_Reg_Pred_per_RiskGroup.png"))
                    plt.close()

            if 'Risk_Group' in df_clusters.columns and not df_clusters['Risk_Group'].isnull().all() and df_clusters['Risk_Group'].nunique() > 1:
                plt.figure(figsize=(8, 6))
                sns.countplot(data=df_clusters, x='Risk_Group', order=sorted(df_clusters['Risk_Group'].dropna().unique()))
                plt.title(f'Customer Count per Risk Group\n{enc_method} - {scale_method}')
                plt.xlabel('Risk Group')
                plt.ylabel('Number of Customers')
                plt.tight_layout()
                plt.savefig(os.path.join(combo_plot_dir, f"TOP5_CV_{enc_method}_{scale_method}_Cluster_Sizes.png"))
                plt.close()

            if 'Risk_Group' in df_clusters.columns and 'Prob_XGB_Clf_for_ClusterRisk' in df_clusters.columns and \
               not df_clusters['Risk_Group'].isnull().all() and not df_clusters['Prob_XGB_Clf_for_ClusterRisk'].isnull().all() and df_clusters['Risk_Group'].nunique() > 1 :
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df_clusters, x='Risk_Group', y='Prob_XGB_Clf_for_ClusterRisk', order=sorted(df_clusters['Risk_Group'].dropna().unique()))
                plt.title(f'XGB Classifier Probabilities by K-Means Risk Group (Full Model)\n{enc_method} - {scale_method}')
                plt.xlabel('K-Means Defined Risk Group')
                plt.ylabel('Predicted Probability (XGB Classifier)')
                plt.tight_layout()
                plt.savefig(os.path.join(combo_plot_dir, f"TOP5_CV_{enc_method}_{scale_method}_Prob_Dist_per_RiskGroup.png"))
                plt.close()

            if X_test_for_clustering.shape[1] >= 2:
                try:
                    pca = PCA(n_components=2, random_state=RANDOM_STATE)
                    X_test_pca = pca.fit_transform(X_test_for_clustering)
                    df_pca = pd.DataFrame(data=X_test_pca, columns=['PC1', 'PC2'])

                    df_pca_clustered = df_clusters.copy()
                    if len(df_pca) == len(df_pca_clustered):
                        df_pca['Cluster_Label'] = df_pca_clustered['Cluster_Label'].values
                        if 'Risk_Group' in df_pca_clustered.columns:
                            df_pca['Risk_Group'] = df_pca_clustered['Risk_Group'].values
                        else:
                            df_pca['Risk_Group'] = 'N/A'
                    else:
                        print(f"PCA length {len(df_pca)} mismatch with df_clusters {len(df_pca_clustered)}. PCA plot might not have accurate hue/style.")
                        # Attempt to assign labels from kmeans directly, ensure indices align or slice
                        if kmeans is not None and hasattr(kmeans, 'labels_'):
                             df_pca['Cluster_Label'] = kmeans.labels_[:len(df_pca)] if len(kmeans.labels_) >= len(df_pca) else pd.Series(kmeans.labels_).reindex(df_pca.index, fill_value=-1).values # Fallback
                        else:
                             df_pca['Cluster_Label'] = -1 # Fallback if kmeans object not available or no labels
                        df_pca['Risk_Group'] = 'N/A'


                    plt.figure(figsize=(12, 8))
                    hue_order_pca = None
                    palette_pca = None
                    style_pca = 'Cluster_Label' if 'Cluster_Label' in df_pca else None

                    if 'Risk_Group' in df_pca.columns and df_pca['Risk_Group'].nunique() > 1 and 'N/A' not in df_pca['Risk_Group'].unique():
                        hue_order_pca = sorted(df_pca['Risk_Group'].dropna().unique())
                        if len(hue_order_pca) == 3 and all(item in ['Low Risk', 'Medium Risk', 'High Risk'] for item in hue_order_pca):
                             palette_pca = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
                        else:
                             palette_pca = sns.color_palette(n_colors=len(hue_order_pca))
                        sns.scatterplot(x='PC1', y='PC2', hue='Risk_Group', style=style_pca, data=df_pca,
                                        palette=palette_pca, hue_order=hue_order_pca,
                                        alpha=0.7, s=50)
                        plt.legend(title='Risk Group & Cluster')
                    else:
                         sns.scatterplot(x='PC1', y='PC2', hue='Cluster_Label' if 'Cluster_Label' in df_pca else None, data=df_pca, palette='viridis', legend='full', alpha=0.7, s=50)
                         plt.legend(title='Cluster Label')

                    plt.title(f'Customer Clusters (PCA) - Risk Groups\n{enc_method} - {scale_method}')
                    plt.xlabel('Principal Component 1')
                    plt.ylabel('Principal Component 2')
                    plt.tight_layout()
                    plt.savefig(os.path.join(combo_plot_dir, f"TOP5_CV_{enc_method}_{scale_method}_PCA_Clusters_RiskGroup.png"))
                    plt.close()
                except Exception as e:
                    print(f"Error during PCA plot generation for {enc_method}-{scale_method}: {e}")
            elif X_test_for_clustering.shape[0] > 0:
                print(f"Skipping PCA plot for {enc_method}-{scale_method} as X_test_for_clustering has {X_test_for_clustering.shape[1]} features (needs >=2).")
        else:
            print(f"Clustering was not successful or df_clusters is invalid for {enc_method}-{scale_method}. Skipping visualizations.")

        print(f"Processing for top combination {enc_method}-{scale_method} complete.")
        print("---")
else:
    if not RUN_ONLY_STAGE_2:
        print("\nINFO: STAGE 1 (with CV) completed but no results were generated, so STAGE 2 cannot run.")

print("\nFull pipeline execution attempt complete.")
#----------------------------Select best preprocessing (not merge, one-hot encoding, minmax scaling)------------------------------
#----------------------------Using XGB------------------------
# 1. 데이터 로딩: 전처리된 One-Hot 인코딩 + Robust Scaling 적용된 학습 데이터 불러오기
df = pd.read_csv("OneHot_Train_Slim_Robust.csv")

# 2. 특성(X)과 타겟(y) 분리
X = df.drop(columns=["SK_ID_CURR", "TARGET"])
y = df["TARGET"].astype(int)

# 3. 클래스 불균형 대응용 scale_pos_weight 계산
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# 4. 기본 XGBoost 분류기 설정
base_xgb_clf = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1
)

# 5. 하이퍼파라미터 후보군 확장
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# 6. Stratified K-Fold 설정
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# 7. RandomizedSearchCV 설정 및 수행
random_search = RandomizedSearchCV(
    estimator=base_xgb_clf,
    param_distributions=param_grid,
    n_iter=100,
    scoring='roc_auc',
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# 8. 하이퍼파라미터 탐색 수행
random_search.fit(X, y)

# 9. 최적 파라미터 및 AUC 출력
print(f"✅ Best Params: {random_search.best_params_}")
print(f"✅ Best AUC: {random_search.best_score_:.4f}")

# 10. 최적 파라미터 기반 최종 모델 재학습
best_params = random_search.best_params_
final_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1,
    **best_params
)
final_model.fit(X, y)

# 11. 테스트 데이터 로딩 (OneHot + Robust Scaling이 이미 적용된 상태)
test_df = pd.read_csv("OneHot_Test_Slim_Robust.csv")
X_test = test_df.drop(columns=["SK_ID_CURR"])

# 12. 테스트셋 예측
y_pred = final_model.predict(X_test)

# 13. 예측 결과 저장 (SK_ID_CURR 포함)
submission = pd.DataFrame({
    "SK_ID_CURR": test_df["SK_ID_CURR"],
    "TARGET": y_pred.astype(int)
})
submission.to_csv("xgb_submission.csv", index=False)
print("✅ Submission saved to xgb_submission.csv")


#----------------------------Using lightGBM------------------------

# 1. 데이터 로딩
df = pd.read_csv("OneHot_Train_Slim_Robust.csv")

# 2. 특성과 타겟 분리
X = df.drop(columns=["SK_ID_CURR", "TARGET"])
y = df["TARGET"].astype(int)

# 🔧 특수 문자 제거 (LightGBM이 feature name에서 오류 발생 방지)
X.columns = X.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

# 3. 클래스 불균형 대응
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# 4. LightGBM 분류기 정의
base_lgb_clf = lgb.LGBMClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

# 5. 하이퍼파라미터 그리드
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# 6. 교차검증 설정
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# 7. RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=base_lgb_clf,
    param_distributions=param_grid,
    n_iter=100,
    scoring='roc_auc',
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=42,
    error_score='raise'  # 디버깅을 위한 옵션 (선택)
)

# 8. 모델 학습
random_search.fit(X, y)

# 9. 결과 출력
print(f"✅ Best Params: {random_search.best_params_}")
print(f"✅ Best AUC: {random_search.best_score_:.4f}")

# 10. 최종 모델 학습
best_params = random_search.best_params_
final_model = lgb.LGBMClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    **best_params
)
final_model.fit(X, y)

# 11. 테스트 데이터 로딩
test_df = pd.read_csv("OneHot_Test_Slim_Robust.csv")
X_test = test_df.drop(columns=["SK_ID_CURR"])

# 테스트셋 컬럼 이름도 동일하게 처리
X_test.columns = X_test.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

# 12. 예측
y_pred = final_model.predict(X_test)

# 13. 결과 저장
submission = pd.DataFrame({
    "SK_ID_CURR": test_df["SK_ID_CURR"],
    "TARGET": y_pred.astype(int)
})
submission.to_csv("lgbm_submission.csv", index=False)
print("✅ Submission saved to lgbm_submission.csv")