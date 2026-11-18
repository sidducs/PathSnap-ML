# PathSnap: Machine Learning Models & Optimization Pipeline (Person 2 Deliverable)

This module contains the training, cross-validation, Bayesian optimization, and benchmarking pipeline for the machine learning models on the PathSnap candidate classification task.

---

## 1. Evaluated Models & Algorithm Families

As required by the hackathon guidelines, 5 distinct machine learning families were trained and evaluated using **5-Fold Group-Aware Cross-Validation (`GroupKFold` grouped by `trip_id`)**:

| # | Model | Algorithm Family | Learning Paradigm | Key Configuration |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Logistic Regression** | Linear models | Linear decision boundary | `StandardScaler()`, $L_2$ Regularization ($C=1.0$), L-BFGS solver |
| **2** | **Random Forest** | Bagging / ensembles | Bootstrap aggregation | 100 decorrelated trees, `max_depth=12`, `min_samples_leaf=10` |
| **3** | **XGBoost (Baseline)** | Gradient boosting | Sequential boosted trees | 150 estimators, `learning_rate=0.05`, `max_depth=6`, Log Loss |
| **4** | **RBF SVM** | Kernel-based learning | Maximum-margin RKHS projection | `StandardScaler()`, RBF Kernel ($C=1.0$), Calibrated Probabilities |
| **5** | **MLP (Neural Net)** | Neural networks / deep learning | Backpropagation & hidden layers | `StandardScaler()`, Architecture: `(64, 32)` hidden units, ReLU |

---

## 2. 5-Family Initial Benchmark Results

Evaluated across **398,858 training samples** (20 representative trips) and tested on **192,119 student-testing samples** (10 trips):

| Model | Algorithm Family | CV Macro F1 | Observation Top-1 Acc | Student-Test Macro F1 | Log Loss | Latency (ms/1k) | Training Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear models | 0.7021 | 69.48% | 0.7399 | 0.1557 | **0.12 ms** | **0.48 s** |
| **Random Forest** | Bagging / ensembles | 0.7757 | 72.22% | 0.8405 | 0.1054 | 1.16 ms | 1.52 s |
| **XGBoost (Winner)** | **Gradient boosting** | **0.7858** | **72.36%** | **0.8520** | **0.1012** | **0.36 ms** | 1.40 s |
| **RBF SVM** | Kernel-based learning | 0.7389 | 69.62% | 0.8063 | 0.1438 | 170.30 ms | 13.06 s |
| **MLP (Neural Net)** | Neural networks | 0.7823 | 70.77% | 0.8277 | 0.1096 | 1.29 ms | 10.99 s |

---

## 3. Bayesian Hyperparameter Optimization (Optuna)

Based on the initial benchmark, **XGBoost** was selected for Bayesian Hyperparameter Optimization using Optuna with a Tree-structured Parzen Estimator (TPE Sampler).

### Search Space & Optimization Setup
* **Search Objective**: Maximize 3-Fold Group-Aware CV Macro F1 on training trips.
* **Tuned Hyperparameters**:
  - `n_estimators`: `140` (searched $[80, 200]$)
  - `max_depth`: `6` (searched $[4, 9]$)
  - `learning_rate`: `0.0632` (searched $[0.02, 0.15]$ log scale)
  - `subsample`: `0.745` (searched $[0.7, 1.0]$)
  - `colsample_bytree`: `0.869` (searched $[0.7, 1.0]$)
  - `reg_alpha`: `0.102` (searched $[1e-3, 5.0]$ log scale)
  - `reg_lambda`: `7.230` (searched $[1e-2, 10.0]$ log scale)

### Tuned vs. Untuned Performance Comparison

| Metric | Untuned XGBoost | Tuned XGBoost | Improvement |
| :--- | :---: | :---: | :---: |
| **CV Macro F1-Score** | 0.7858 | **0.8022** | **+1.64%** |
| **Student-Test Macro F1-Score** | 0.8520 | **0.8529** | **+0.09%** |
| **Observation Top-1 Accuracy** | 72.36% | **72.46%** | **+0.10%** |
| **Log Loss (Calibration)** | 0.1012 | **0.1009** | **Sharper probabilities** |
| **Inference Latency** | 0.36 ms / 1k | **0.34 ms / 1k** | **Faster execution** |

---

## 4. Key Takeaways & Deliverables
* **Optuna Trial Logs**: Full trial-by-trial logs saved in `results/optimization/optuna_trials.csv`.
* **Model Metadata**: Formatted JSON specification saved in `models/model_metadata.json`.
* **Model Artifact**: Serialized model saved at `models/best_model.joblib`.
* **Reproduction**:
  ```bash
  # Run 5-model benchmark
  python ml/training/train_and_benchmark.py

  # Run Bayesian optimization
  python ml/optimization/bayesian_opt.py
  ```
