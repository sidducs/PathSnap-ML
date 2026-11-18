import os
import sys
import time
import json
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, log_loss
import joblib

def load_dataset(parquet_path=r"D:\ML Hackathon\data\processed\candidate_features.parquet"):
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    return df

def compute_observation_top1(df_test, y_probs):
    """
    Observation-level Top-1 Accuracy: For each GPS observation,
    select the candidate with highest predicted score and check if it is the true road segment.
    """
    df_eval = df_test[["trip_id", "obs_idx", "label"]].copy()
    df_eval["score"] = y_probs
    idx_top = df_eval.groupby(["trip_id", "obs_idx"])["score"].idxmax()
    top_candidates = df_eval.loc[idx_top]
    top1_acc = np.mean(top_candidates["label"] == 1)
    return float(top1_acc)

def main():
    print("==================================================")
    print("      PATHSNAP: 5-FAMILY ML BENCHMARK ENGINE      ")
    print("==================================================")
    
    parquet_path = r"D:\ML Hackathon\data\processed\candidate_features.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Dataset not found at {parquet_path}")
        
    print(f"Loading dataset from {parquet_path}...")
    t0 = time.time()
    df = load_dataset(parquet_path)
    print(f"Loaded {len(df):,} candidate records in {time.time()-t0:.2f}s")
    
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
    
    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values
    groups_train = df_train["trip_id"].values
    
    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values
    
    # Handle nan/inf values safely
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=100.0, neginf=-100.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # The 5 Definitive Distinct Algorithm Families
    models = {
        "Logistic Regression": {
            "family": "Linear models",
            "approach": "Linear decision boundary",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=42))
            ]),
            "subsample_fit": None
        },
        "Random Forest": {
            "family": "Bagging / ensembles",
            "approach": "Bootstrap aggregation (100 trees)",
            "pipeline": Pipeline([
                ("clf", RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=10, n_jobs=-1, random_state=42))
            ]),
            "subsample_fit": 80000
        },
        "XGBoost": {
            "family": "Gradient boosting",
            "approach": "Sequential gradient boosted trees",
            "pipeline": Pipeline([
                ("clf", XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, eval_metric="logloss", random_state=42, n_jobs=-1))
            ]),
            "subsample_fit": None
        },
        "RBF Support Vector Machine": {
            "family": "Kernel-based learning",
            "approach": "Maximum-margin RKHS nonlinear projection",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(C=1.0, kernel="rbf", probability=True, random_state=42))
            ]),
            "subsample_fit": 15000 # Stratified sample for O(N^2) kernel SVM
        },
        "MLP (Neural Network)": {
            "family": "Neural networks / deep learning",
            "approach": "Backpropagation & nonlinear hidden layers (64, 32)",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True, random_state=42))
            ]),
            "subsample_fit": 80000
        }
    }
    
    gkf = GroupKFold(n_splits=5)
    
    # Representative CV sample across trips for high cross-validation fidelity
    cv_subsample_idx = np.random.RandomState(42).choice(len(df_train), size=min(len(df_train), 75000), replace=False)
    X_train_cv = X_train[cv_subsample_idx]
    y_train_cv = y_train[cv_subsample_idx]
    groups_train_cv = groups_train[cv_subsample_idx]
    
    comparison_results = []
    cv_records = []
    
    os.makedirs(r"D:\ML Hackathon\results", exist_ok=True)
    os.makedirs(r"D:\ML Hackathon\ml\artifacts", exist_ok=True)
    os.makedirs(r"D:\ML Hackathon\visualizations", exist_ok=True)
    
    for name, config in models.items():
        print(f"\n==================================================")
        print(f"Training: {name}")
        print(f"Family: {config['family']} | Approach: {config['approach']}")
        print(f"==================================================")
        pipeline = config["pipeline"]
        
        # 1. 5-Fold Group-Aware Cross-Validation
        cv_f1_scores = []
        cv_acc_scores = []
        fold_idx = 1
        for train_idx, val_idx in gkf.split(X_train_cv, y_train_cv, groups=groups_train_cv):
            X_f_tr, y_f_tr = X_train_cv[train_idx], y_train_cv[train_idx]
            X_f_val, y_f_val = X_train_cv[val_idx], y_train_cv[val_idx]
            
            # Subsample inside fold if computationally intensive (SVM / MLP)
            if config["subsample_fit"] and len(X_f_tr) > config["subsample_fit"]:
                sub_idx = np.random.RandomState(42).choice(len(X_f_tr), size=config["subsample_fit"], replace=False)
                X_f_tr, y_f_tr = X_f_tr[sub_idx], y_f_tr[sub_idx]
                
            pipeline.fit(X_f_tr, y_f_tr)
            y_pred_val = pipeline.predict(X_f_val)
            f1 = f1_score(y_f_val, y_pred_val, average="macro")
            acc = accuracy_score(y_f_val, y_pred_val)
            cv_f1_scores.append(f1)
            cv_acc_scores.append(acc)
            cv_records.append({"model": name, "fold": fold_idx, "macro_f1": f1, "accuracy": acc})
            fold_idx += 1
            
        mean_cv_f1 = np.mean(cv_f1_scores)
        mean_cv_acc = np.mean(cv_acc_scores)
        print(f"5-Fold CV Macro F1: {mean_cv_f1:.4f} (Accuracy: {mean_cv_acc:.4f})")
        
        # 2. Fit on Full Training Partition (or subsample for SVM/MLP)
        fit_X, fit_y = X_train, y_train
        if config["subsample_fit"] and len(fit_X) > config["subsample_fit"]:
            sub_idx = np.random.RandomState(42).choice(len(fit_X), size=config["subsample_fit"], replace=False)
            fit_X, fit_y = fit_X[sub_idx], fit_y[sub_idx]
            
        t_fit_start = time.time()
        pipeline.fit(fit_X, fit_y)
        train_time = time.time() - t_fit_start
        print(f"Fit completed in {train_time:.2f}s on {len(fit_X):,} samples")
        
        # 3. Evaluate on Student-Testing Partition
        t_inf_start = time.time()
        test_probs = pipeline.predict_proba(X_test)[:, 1]
        inference_latency_ms_per_1k = (time.time() - t_inf_start) / len(X_test) * 1000.0 * 1000.0 # ms per 1000
        test_preds = (test_probs >= 0.5).astype(int)
        
        test_f1 = f1_score(y_test, test_preds, average="macro")
        test_loss = log_loss(y_test, np.clip(test_probs, 1e-6, 1-1e-6))
        
        # Observation-Level Top-1 Accuracy
        obs_top1 = compute_observation_top1(df_test, test_probs)
        print(f"Student-Test Macro F1: {test_f1:.4f}")
        print(f"Student-Test Observation Top-1 Accuracy: {obs_top1*100:.2f}%")
        print(f"Student-Test Log Loss: {test_loss:.4f}")
        print(f"Inference Latency: {inference_latency_ms_per_1k:.2f} ms / 1,000 candidates")
        
        # Save model artifact
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        artifact_path = os.path.join(r"D:\ML Hackathon\ml\artifacts", f"{safe_name}.joblib")
        joblib.dump(pipeline, artifact_path)
        
        comparison_results.append({
            "Model": name,
            "Algorithm Family": config["family"],
            "Learning Approach": config["approach"],
            "CV Macro F1-score": round(mean_cv_f1, 4),
            "Top-1 Accuracy": f"{round(obs_top1 * 100, 2)}%",
            "Student-Test Macro F1-score": round(test_f1, 4),
            "Log Loss": round(test_loss, 4),
            "Inference Time (ms/1k)": round(inference_latency_ms_per_1k, 2),
            "Training Time (s)": round(train_time, 2)
        })

    # Save Comparison CSV
    res_path = r"D:\ML Hackathon\results\model_comparison.csv"
    with open(res_path, "w") as f:
        headers = ["Model", "Algorithm Family", "Learning Approach", "CV Macro F1-score", "Top-1 Accuracy", "Student-Test Macro F1-score", "Log Loss", "Inference Time (ms/1k)", "Training Time (s)"]
        f.write(",".join(headers) + "\n")
        for r in comparison_results:
            row = [str(r[h]).replace(",", ";") for h in headers]
            f.write(",".join(row) + "\n")
    print(f"\nSaved consolidated comparison table to: {res_path}")
    
    # Save CV results CSV
    cv_res_path = r"D:\ML Hackathon\results\cv_results.csv"
    with open(cv_res_path, "w") as f:
        f.write("model,fold,macro_f1,accuracy\n")
        for r in cv_records:
            f.write(f"{r['model']},{r['fold']},{r['macro_f1']:.4f},{r['accuracy']:.4f}\n")
            
    # Generate Visual Comparison Plots
    print("Generating updated comparison charts...")
    models_names = [r["Model"] for r in comparison_results]
    cv_f1s = [r["CV Macro F1-score"] for r in comparison_results]
    top1s = [float(r["Top-1 Accuracy"].replace("%", "")) for r in comparison_results]
    latencies = [r["Inference Time (ms/1k)"] for r in comparison_results]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x_indices = np.arange(len(models_names))
    
    # Plot 1: CV Macro F1
    axes[0].bar(x_indices, cv_f1s, color="#1976D2")
    axes[0].set_title("Cross-Validation Macro F1-Score", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_xticks(x_indices)
    axes[0].set_xticklabels(models_names, rotation=35, ha="right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    for i, v in enumerate(cv_f1s):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
        
    # Plot 2: Top-1 Accuracy
    axes[1].bar(x_indices, top1s, color="#388E3C")
    axes[1].set_title("Student-Test Observation Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0, 100)
    axes[1].set_xticks(x_indices)
    axes[1].set_xticklabels(models_names, rotation=35, ha="right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)
    for i, v in enumerate(top1s):
        axes[1].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
        
    # Plot 3: Inference Latency
    axes[2].bar(x_indices, latencies, color="#F57C00")
    axes[2].set_title("Inference Latency (ms / 1k candidates)", fontsize=12, fontweight="bold")
    axes[2].set_xticks(x_indices)
    axes[2].set_xticklabels(models_names, rotation=35, ha="right")
    axes[2].grid(axis="y", linestyle="--", alpha=0.7)
    for i, v in enumerate(latencies):
        axes[2].text(i, v + 0.5, f"{v:.1f}", ha="center", fontweight="bold")
        
    plt.tight_layout()
    plot_path = r"D:\ML Hackathon\visualizations\model_comparison_plots.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved comparison plots to: {plot_path}")
    
    print("\n==================================================")
    print("       ALL 5 DISTINCT FAMILIES BENCHMARKED        ")
    print("==================================================")

if __name__ == "__main__":
    main()
