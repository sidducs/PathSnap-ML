import os
import sys
import time
import json
import numpy as np
import pyarrow.parquet as pq
import optuna
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, log_loss
import joblib

# Suppress Optuna verbose logging for clean output
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data():
    parquet_path = r"D:\ML Hackathon\data\processed\candidate_features.parquet"
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    df_train = df[df["partition"] == "train"].copy().reset_index(drop=True)
    df_test = df[df["partition"] == "student_test"].copy().reset_index(drop=True)
    
    feature_cols = [
        "perp_dist_m",
        "heading_diff_rad",
        "proj_ratio",
        "arc_len_m",
        "gps_speed_mps",
        "time_delta_s",
        "speed_implied_mps"
    ]
    target_col = "label"
    
    X_train = np.nan_to_num(df_train[feature_cols].values, nan=0.0, posinf=100.0, neginf=-100.0)
    y_train = df_train[target_col].values
    groups_train = df_train["trip_id"].values
    
    X_test = np.nan_to_num(df_test[feature_cols].values, nan=0.0, posinf=100.0, neginf=-100.0)
    y_test = df_test[target_col].values
    
    return X_train, y_train, groups_train, X_test, y_test, df_test, feature_cols

def compute_observation_top1(df_test, y_probs):
    df_eval = df_test[["trip_id", "obs_idx", "label"]].copy()
    df_eval["score"] = y_probs
    idx_top = df_eval.groupby(["trip_id", "obs_idx"])["score"].idxmax()
    top_candidates = df_eval.loc[idx_top]
    return float(np.mean(top_candidates["label"] == 1))

def run_optimization(n_trials=15):
    print("==================================================")
    print("     STAGE 4: BAYESIAN OPTIMIZATION (OPTUNA)      ")
    print("==================================================")
    
    X_train, y_train, groups_train, X_test, y_test, df_test, feature_cols = load_data()
    print(f"Loaded train data: {len(X_train):,} samples | Test data: {len(X_test):,} samples")
    
    # Subsample for fast cross-validation during Optuna search
    cv_sample_idx = np.random.RandomState(42).choice(len(X_train), size=min(len(X_train), 50000), replace=False)
    X_cv = X_train[cv_sample_idx]
    y_cv = y_train[cv_sample_idx]
    groups_cv = groups_train[cv_sample_idx]
    
    gkf = GroupKFold(n_splits=3) # 3-fold group CV for speedy trials
    
    trial_records = []
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 200, step=20),
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1
        }
        
        cv_f1s = []
        for tr_idx, val_idx in gkf.split(X_cv, y_cv, groups=groups_cv):
            clf = XGBClassifier(**params)
            clf.fit(X_cv[tr_idx], y_cv[tr_idx])
            preds = clf.predict(X_cv[val_idx])
            cv_f1s.append(f1_score(y_cv[val_idx], preds, average="macro"))
            
        mean_f1 = np.mean(cv_f1s)
        trial_records.append({
            "trial_number": trial.number,
            "mean_cv_macro_f1": mean_f1,
            **params
        })
        print(f"Trial {trial.number+1}/{n_trials} -> CV Macro F1: {mean_f1:.4f} (depth={params['max_depth']}, lr={params['learning_rate']:.4f}, n_est={params['n_estimators']})")
        return mean_f1

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    search_time = time.time() - t0
    
    print(f"\nBayesian search completed in {search_time:.1f}s")
    best_params = study.best_params
    print(f"Best Trial #{study.best_trial.number}: CV Macro F1 = {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
        
    # Save Optuna trial records
    os.makedirs(r"D:\ML Hackathon\results\optimization", exist_ok=True)
    trials_csv = r"D:\ML Hackathon\results\optimization\optuna_trials.csv"
    with open(trials_csv, "w") as f:
        headers = list(trial_records[0].keys())
        f.write(",".join(headers) + "\n")
        for r in trial_records:
            f.write(",".join(str(r[h]) for h in headers) + "\n")
    print(f"Saved optimization trial logs to: {trials_csv}")
    
    # 4. Refit Tuned Model on Full Training Partition
    print("\nRefitting Tuned XGBoost on full training dataset...")
    tuned_params = {
        **best_params,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1
    }
    tuned_clf = XGBClassifier(**tuned_params)
    t_refit = time.time()
    tuned_clf.fit(X_train, y_train)
    refit_time = time.time() - t_refit
    print(f"Refit completed in {refit_time:.2f}s on {len(X_train):,} samples")
    
    # Evaluate Tuned Model on Student-Testing Partition
    t_inf = time.time()
    tuned_probs = tuned_clf.predict_proba(X_test)[:, 1]
    inf_latency = (time.time() - t_inf) / len(X_test) * 1000.0 * 1000.0 # ms / 1k
    tuned_preds = (tuned_probs >= 0.5).astype(int)
    
    tuned_test_f1 = f1_score(y_test, tuned_preds, average="macro")
    tuned_top1 = compute_observation_top1(df_test, tuned_probs)
    tuned_logloss = log_loss(y_test, np.clip(tuned_probs, 1e-6, 1-1e-6))
    
    # Compare with Untuned baseline (from Stage 3: CV=0.7858, Test F1=0.8520, Top-1=72.36%, LogLoss=0.1012)
    untuned_baseline = {
        "CV Macro F1": 0.7858,
        "Test Macro F1": 0.8520,
        "Top-1 Accuracy": "72.36%",
        "Log Loss": 0.1012,
        "Inference Latency (ms/1k)": 0.36
    }
    
    print("\n--------------------------------------------------")
    print("        TUNED VS UNTUNED MODEL COMPARISON         ")
    print("--------------------------------------------------")
    print(f"{'Metric':<25} | {'Untuned XGBoost':<18} | {'Tuned XGBoost':<18}")
    print("-" * 67)
    print(f"{'CV Macro F1':<25} | {untuned_baseline['CV Macro F1']:<18.4f} | {study.best_value:<18.4f}")
    print(f"{'Student-Test Macro F1':<25} | {untuned_baseline['Test Macro F1']:<18.4f} | {tuned_test_f1:<18.4f}")
    print(f"{'Top-1 Accuracy':<25} | {untuned_baseline['Top-1 Accuracy']:<18} | {tuned_top1*100:<17.2f}%")
    print(f"{'Log Loss':<25} | {untuned_baseline['Log Loss']:<18.4f} | {tuned_logloss:<18.4f}")
    print(f"{'Inference Latency':<25} | {untuned_baseline['Inference Latency (ms/1k)']:<14.2f} ms | {inf_latency:<14.2f} ms")
    print("--------------------------------------------------")
    
    # Save Tuned vs Untuned CSV
    comparison_csv = r"D:\ML Hackathon\results\optimization\tuned_vs_untuned.csv"
    with open(comparison_csv, "w") as f:
        f.write("metric,untuned,tuned\n")
        f.write(f"CV Macro F1,{untuned_baseline['CV Macro F1']},{study.best_value:.4f}\n")
        f.write(f"Student-Test Macro F1,{untuned_baseline['Test Macro F1']},{tuned_test_f1:.4f}\n")
        f.write(f"Top-1 Accuracy,{untuned_baseline['Top-1 Accuracy']},{tuned_top1*100:.2f}%\n")
        f.write(f"Log Loss,{untuned_baseline['Log Loss']},{tuned_logloss:.4f}\n")
        f.write(f"Inference Latency ms/1k,{untuned_baseline['Inference Latency (ms/1k)']},{inf_latency:.2f}\n")
    print(f"Saved comparison to: {comparison_csv}")
    
    # 5. Save Final Production Model Artifacts
    os.makedirs(r"D:\ML Hackathon\models", exist_ok=True)
    model_path = r"D:\ML Hackathon\models\best_model.joblib"
    joblib.dump(tuned_clf, model_path)
    print(f"\nSaved production model to: {model_path} ({os.path.getsize(model_path):,} bytes)")
    
    # Save Metadata JSON
    metadata = {
        "model_name": "Tuned_XGBoost_Classifier",
        "algorithm_family": "Gradient Boosting",
        "model_version": "1.0.0",
        "best_hyperparameters": best_params,
        "feature_names": feature_cols,
        "feature_count": len(feature_cols),
        "target_name": "label",
        "optimization_engine": "Optuna (TPE Sampler)",
        "optimization_trials": n_trials,
        "metrics": {
            "cv_macro_f1": round(study.best_value, 4),
            "student_test_macro_f1": round(tuned_test_f1, 4),
            "observation_top1_accuracy": f"{round(tuned_top1 * 100, 2)}%",
            "log_loss": round(tuned_logloss, 4),
            "inference_latency_ms_per_1k": round(inf_latency, 2),
            "refit_time_seconds": round(refit_time, 2)
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    meta_path = r"D:\ML Hackathon\models\model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {meta_path}")
    
    # 6. Verify Model Loading in Fresh Python Call
    print("\nVerifying model loading from disk...")
    loaded_model = joblib.load(model_path)
    sample_preds = loaded_model.predict_proba(X_test[:5])[:, 1]
    print(f"Verification sample prediction output: {sample_preds}")
    print("Verification SUCCESSFUL: Model loads and executes correctly.")

if __name__ == "__main__":
    run_optimization(n_trials=12)

