# PathSnap: Machine Learning Models (Person 2 Deliverable)

This module contains the training, cross-validation, and benchmarking pipeline for the 5 distinct machine learning model families evaluated on the PathSnap candidate classification task.

---

## 1. Evaluated Models & Algorithm Families

As required by the hackathon guidelines, 5 distinct machine learning families were trained using **5-Fold Group-Aware Cross-Validation (`GroupKFold` grouped by `trip_id`)**:

| # | Model | Algorithm Family | Learning Paradigm | Key Configuration |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Logistic Regression** | Linear models | Linear decision boundary | `StandardScaler()`, $L_2$ Regularization ($C=1.0$), L-BFGS solver |
| **2** | **Random Forest** | Bagging / ensembles | Bootstrap aggregation | 100 decorrelated trees, max depth = 12, min leaf = 10 |
| **3** | **XGBoost** | Gradient boosting | Sequential boosted trees | 150 estimators, learning rate = 0.05, max depth = 6, Log Loss |
| **4** | **RBF SVM** | Kernel-based learning | Maximum-margin RKHS projection | `StandardScaler()`, RBF Kernel ($C=1.0$), Calibrated Probabilities |
| **5** | **MLP (Neural Net)** | Neural networks / deep learning | Backpropagation & hidden layers | `StandardScaler()`, Architecture: `(64, 32)` hidden units, ReLU |

---

## 2. Experimental Benchmark Results

Evaluated across **398,858 training samples** (20 representative trips) and tested on **192,119 student-testing samples** (10 trips):

| Model | Algorithm Family | CV Macro F1 | Observation Top-1 Acc | Student-Test Macro F1 | Log Loss | Latency (ms/1k) | Training Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear models | 0.7021 | 69.48% | 0.7399 | 0.1557 | **0.12 ms** | **0.48 s** |
| **Random Forest** | Bagging / ensembles | 0.7757 | 72.22% | 0.8405 | 0.1054 | 1.16 ms | 1.52 s |
| **XGBoost** | **Gradient boosting** | **0.7858** | **72.36%** | **0.8520** | **0.1012** | **0.36 ms** | 1.40 s |
| **RBF SVM** | Kernel-based learning | 0.7389 | 69.62% | 0.8063 | 0.1438 | 170.30 ms | 13.06 s |
| **MLP (Neural Net)** | Neural networks | 0.7823 | 70.77% | 0.8277 | 0.1096 | 1.29 ms | 10.99 s |

---

## 3. Key Observations & Model Selection
* **Overall Winner**: **XGBoost** achieved the highest CV Macro F1 (`0.7858`), highest test Macro F1 (`0.8520`), highest Top-1 Accuracy (`72.36%`), and lowest Log Loss (`0.1012`) with ultra-fast latency (`0.36 ms`).
* **Inference Complexity**: RBF SVM proved computationally heavy during inference (170 ms per 1,000 candidates), making it impractical for live API requests.
* **Winning Model Role**: XGBoost was chosen as the primary classifier to provide calibrated emission probabilities for the sequential HMM stage.

---

## 4. Execution Command
To reproduce the 5-model benchmark:
```bash
python ml/training/train_and_benchmark.py
```
Output results will be written to:
* `results/model_comparison.csv`
* `results/cv_results.csv`
* `visualizations/model_comparison_plots.png`
