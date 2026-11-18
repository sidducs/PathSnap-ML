# PathSnap: Complete Screenshots & Visualisations Index
**Supporting Document for Deliverable 2: Consolidated Technical Report (PDF)**  
*All figures are high-resolution (200+ DPI) and stored in `report/screenshots/` and `report/images/`.*

---

## 1. Section 2.2: Dataset Understanding and Exploratory Data Analysis (EDA)
| Figure File | Report Caption | Description & Findings |
| :--- | :--- | :--- |
| `eda_overview_distributions.png` | **Figure 1: Dataset Partition & Observation Overview** | Distribution of observations across 80 training trips vs. 10 student-test trips. |
| `speed_distribution.png` | **Figure 2: Vehicle Velocity & GPS Speed Outlier Analysis** | Instantaneous vehicle speed distribution showing 3,407 flagged multipath outliers (>120 km/h). |
| `timestamp_delta_distribution.png` | **Figure 3: GPS Sampling Delta & Dropout Gap Distribution** | Shows 94.3% nominal 1 Hz intervals, with 1,260 time gaps (>5s) documented. |
| `road_network_and_route_scaling.png` | **Figure 4: Road Network Scaling & Graph Topology** | Road arc vs. node scaling across 2,024,174 nodes and 4,061,945 road segments. |

---

## 2. Section 3: Model Experimentation and Comparison (5 Models)
| Figure File | Report Caption | Description & Findings |
| :--- | :--- | :--- |
| `section3_model_benchmark_comparison.png` | **Figure 5: 5-Model Benchmark Performance Comparison** | Four-panel comparative benchmark across Logistic Regression, Random Forest, XGBoost, RBF SVM, and MLP. Details Cross-Validation Macro F1, Student-Test Macro F1, Top-1 Accuracy, and Inference Latency. |

---

## 3. Section 4: Bayesian Hyperparameter Optimisation
| Figure File | Report Caption | Description & Findings |
| :--- | :--- | :--- |
| `section4_bayesian_optimization_trials.png` | **Figure 6: Optuna Bayesian Optimisation Convergence & Feature Importance** | Left: Objective function convergence over 50 trials (group-aware CV). Right: fANOVA hyperparameter importance ranking (learning rate, depth, subsample). |

---

## 4. Section 5: HMM and Viterbi Route Reconstruction
| Figure File | Report Caption | Description & Findings |
| :--- | :--- | :--- |
| `section5_hmm_vs_classifier_route.png` | **Figure 7: Classifier-Only Jumps vs. HMM Continuous Trajectory** | Side-by-side trajectory showing how independent classification jumps between parallel roads, while HMM Log-Viterbi enforces continuous road network topology. |

---

## 5. Section 6: Geographical Visualisation Using Folium
| File | Report Caption | Description |
| :--- | :--- | :--- |
| `visualisations/road_network.html` | **Interactive Road Network Map** | Committed interactive Folium HTML map visualizing road arcs and topology. |
| `visualisations/gps_trajectory.html` | **Interactive GPS Trajectory Map** | Committed interactive Folium HTML map with raw GPS breadcrumbs. |
| `visualisations/reference_route.html` | **Interactive Ground-Truth Route** | Ground-truth vehicle route rendered over the graph. |

---

## 6. Section 7: API, UI and Two-Laptop Deployment
| Figure File | Report Caption | Description & Findings |
| :--- | :--- | :--- |
| `ui_dashboard_laptop2_trip09.png` | **Figure 8: Evaluator UI on Laptop 2 (Client Node)** | Full dashboard view running on Laptop 2, displaying live connected status to `http://10.30.202.24:8000`, 94.25% route continuity, and interactive Leaflet map. |
| `ui_dashboard_laptop2_trip00.png` | **Figure 9: Evaluator UI Highway Trajectory View** | Full view of 54.38 km highway evaluation trip with start/destination pins and layer toggles. |
| `section7_cmd_terminal_laptop1_logs.png` | **Figure 10: Laptop 1 (Model Server) Terminal & API Execution Logs** | Live Windows Command Prompt screenshot demonstrating FastAPI startup, health checks, and incoming `POST /predict` inference requests (345.69 ms). |
