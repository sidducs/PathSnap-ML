# PathSnap ML — Data Engineering, Preprocessing & Feature Engineering
**Owner:** HRUSHIKESH KARNAM (Data Engineering & Preprocessing)  
**Handover Recipient:** Siddhu (Classical ML, Optuna & HMM Modeling)

---

## 1. System Overview

PathSnap is a robust, production-grade GPS map-matching and route-reconstruction platform. This subsystem implements the complete data engineering pipeline:
1. **Raw File Ingestion & Quality Validation**: Ingests whitespace-separated `.track`, `.nodes`, `.arcs`, and `.route` files without silent row modification.
2. **Conformal Local Planar Projections**: Computes metric Cartesian coordinates using trip-specific local Mercator projections to eliminate distortion across worldwide regions.
3. **Directed Road Network Topology**: Constructs directed graphs $G=(V, E)$ preserving exact 0-based node and arc identifiers, segment lengths, bearings, and topological adjacency.
4. **Line-Geometry Spatial Candidate Indexing**: Utilizes broad-phase `cKDTree` spatial search over sampled segment interiors coupled with exact perpendicular point-to-segment distance calculations in physical meters.
5. **Topological-Temporal Reference Route Alignment**: Maps GPS observations monotonically to ground-truth route sequences, distinguishing `aligned`, `ambiguous`, `candidate_miss`, and `unmatched` observations.
6. **27 Engineered Candidate Features**: Extracts spatial, directional, temporal/kinematic, and network reachability features into clean, finite Parquet datasets.
7. **Interactive Folium Visualisations**: Generates standalone geographic visualisations.

---

## 2. Dataset Structure & Partition Separation

The official dataset contains:
- **80 Training Trips (`data/raw/Student folder/train/`)**: Fully permissive for EDA, candidate radius experimentation, feature development, route alignment, label generation, and model training.
- **10 Student Testing Trips (`data/raw/Student folder/student_test/`)**: Permitted exclusively for development-stage evaluation and model comparison. Ground truth labels are strictly isolated and withheld from classifier training.
- **10 Evaluator Testing Trips**: Held out and completely isolated by evaluators. No ground truth or features are leaked.

### File Specifications
- `.track`: Three whitespace-separated columns (`longitude`, `latitude`, `timestamp`). Row order corresponds to `observation_index` ($0, 1, \dots, N-1$).
- `.nodes`: Two whitespace-separated columns (`longitude`, `latitude`). Row index ($0, 1, \dots, |V|-1$) is the zero-based `node_id`.
- `.arcs`: Two integer columns (`from_node_id`, `to_node_id`). Row index ($0, 1, \dots, |E|-1$) is the zero-based `arc_id`.
- `.route`: Single integer column containing the chronological sequence of reference `arc_id`s representing the ground-truth traversal.

---

## 3. Handover Artifacts for Siddhu's Modeling Pipeline

All handover artifacts are located in `data/processed/`:

```
data/processed/
├── training/
│   ├── candidate_features.parquet      # Master training candidate feature matrix
│   ├── feature_schema.json             # Complete feature definitions, data types & units
│   ├── alignment_audit.csv             # Per-trip route alignment diagnostic metrics
│   └── trip_processing_summary.csv     # Per-trip observation and candidate counts
└── student_test/
    ├── candidate_features.parquet      # Features for student evaluation (labels isolated)
    └── trip_processing_summary.csv     # Test processing metadata
```

### Feature Schema Summary

#### Identifiers (Preserved for Indexing — Exclude from Classifier Training)
- `trip_id`: Trip folder name.
- `observation_index`: 0-based sequential GPS observation number.
- `timestamp`: Observation time in seconds.
- `arc_id`: Candidate road segment ID.
- `source_node`: Candidate arc origin node.
- `target_node`: Candidate arc destination node.

#### Engineered Feature Columns (27 Continuous / Finite Features)
1. **Spatial Features**:
   - `distance_m`: Perpendicular Euclidean distance in meters from GPS point to road segment.
   - `candidate_rank`: Distance-ordered integer rank ($0 = \text{nearest}$).
   - `projection_t`: Normalized fractional projection position along segment $[0.0, 1.0]$.
   - `segment_length_m`: Physical length of candidate road segment in meters.
   - `proj_offset_x`, `proj_offset_y`: Planar Cartesian displacement offsets $(P_{proj} - P_{obs})$.
   - `log_distance_m`: Natural logarithm $\ln(1 + \text{distance\_m})$.
   - `emission_prob_proxy`: Gaussian emission density proxy $\exp(-d^2 / (2 \cdot 15.0^2))$.
2. **Directional Features**:
   - `gps_bearing`: Instantaneous GPS trajectory heading $[0, 360)^\circ$.
   - `road_bearing`: Directed road segment azimuth $[0, 360)^\circ$.
   - `heading_delta`: Absolute heading difference $|\Delta \theta| \in [0, 180]^\circ$.
   - `heading_cos_sim`: Cosine similarity $\cos(\Delta \theta) \in [-1.0, 1.0]$.
   - `direction_agreement`: Binary indicator ($1.0$ if $|\Delta \theta| \le 45^\circ$, else $0.0$).
   - `is_antiparallel`: Binary indicator ($1.0$ if $|\Delta \theta| \ge 135^\circ$ [wrong way], else $0.0$).
3. **Temporal & Movement Features**:
   - `dt_prev`: Sampling time elapsed $\Delta t = t_i - t_{i-1}$ in seconds.
   - `gps_displacement_m`: Haversine distance between consecutive GPS points.
   - `gps_speed_m_s`: Trajectory speed $\text{displacement} / \Delta t$ in m/s.
   - `gps_accel_m_s2`: Instantaneous acceleration in $\text{m/s}^2$.
4. **Road Network & Transition Features**:
   - `source_node_out_degree`: Number of outgoing arcs from candidate start node.
   - `target_node_in_degree`: Number of incoming arcs to candidate target node.
   - `source_node_in_degree`: Number of incoming arcs to candidate start node.
   - `target_node_out_degree`: Number of outgoing arcs to candidate target node.
   - `is_source_deadend`: $1.0$ if source node has in-degree == 0, else $0.0$.
   - `is_target_deadend`: $1.0$ if target node has out-degree == 0, else $0.0$.
   - `is_connected_to_prev_top_cand`: $1.0$ if candidate directly connects to top candidate at $i-1$.
   - `graph_transition_dist_m`: Shortest directed network distance (Dijkstra) from previous top candidate.
   - `transition_excess_ratio`: $(\text{network\_distance} - \text{gps\_displacement}) / \max(1.0, \text{gps\_displacement})$.

#### Target Labels & Audit Flags
- `label`: Binary classification target ($1$ for true reference route segment, $0$ for competing candidates).
- `alignment_status`: Association confidence status:
  - `aligned`: Ground-truth arc unambiguously verified.
  - `ambiguous`: Multiple plausible route candidates near junction (excluded from training loss).
  - `candidate_miss`: True route arc missed by spatial search radius.
  - `unmatched`: GPS observation $> 120\text{ m}$ away from route.
- `route_index`: Sequence position in `.route` ($0, 1, \dots$).

---

## 4. Execution Commands

All scripts are reproducible and run inside standard Python:

```bash
# 1. Run complete unit test suite
pytest -v

# 2. Inspect raw dataset inventory (generates results/dataset_inventory.csv)
python scripts/inspect_dataset.py

# 3. Generate automated EDA reports and diagnostics
python scripts/generate_dataset_report.py

# 4. Generate candidate features Parquet handover datasets
python scripts/build_handover_dataset.py

# 5. Generate interactive Folium visualisations
python scripts/generate_maps.py

# 6. Run deliverables audit and verification
python scripts/generate_deliverables_audit.py
```
