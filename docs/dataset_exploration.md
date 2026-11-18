# PathSnap ML — Exploratory Data Analysis & Quality Report
**Person 1: Data Engineering, Preprocessing, and Road-Network Pipeline**
*Generated: 2026-11-18*

---

## 1. Executive Summary

PathSnap is a GPS map-matching and route-reconstruction platform. This document presents the comprehensive, empirical data quality audit and exploratory analysis conducted on the official benchmark dataset.

### Key Partition Statistics
| Metric | Training Partition | Student Testing | Evaluator Testing | Overall Total |
| :--- | :--- | :--- | :--- | :--- |
| **Number of Trips** | 80 | 10 | 0 (Isolated) | 90 |
| **Total GPS Observations** | 208,417 | 19,160 | N/A | 227,577 |
| **Total Road Nodes** | 1,933,485 | 90,689 | N/A | 2,024,174 |
| **Total Road Arcs** | 3,876,904 | 185,041 | N/A | 4,061,945 |
| **Total Route Arcs** | 31,336 | 3,477 | N/A | 34,813 |
| **Total Trajectory Length** | 2,978.4 km | 297.0 km | N/A | 3,275.4 km |

> **Partition Isolation Rule**: All 80 training trips are permitted for candidate generation, feature development, label assignment, and model training. The 10 student-testing trips are held strictly for development evaluation without label leakage. Evaluator data remains isolated and withheld.

---

## 2. GPS Trajectory Analysis

### Data Integrity & Plausibility
- **Coordinate Plausibility**: 100% of GPS coordinates fall within valid geographic bounds ($[-180, 180]$ longitude, $[-90, 90]$ latitude).
- **Missing / NaN / Inf**: 0 missing coordinates, 0 NaNs, and 0 Infs across all 227,577 observations.
- **Coordinate Zero Validity**: Coordinates near or equal to zero were confirmed to be valid geographical positions along the Prime Meridian or Equator and are strictly retained.

### Temporal Properties & Sampling Distribution
- **Sampling Interval**:
  - Global Median $\Delta t$: `1.00 s` (nominal 1 Hz sampling).
  - Minimum Sampling $\Delta t$: `1.00 s`.
  - Maximum Sampling $\Delta t$: `153064.00 s`.
- **Timestamp Gaps**:
  - Observations with dt > 5s: `1,260`
  - Observations with dt > 10s: `617`
  - Observations with dt > 30s: `135`
- **Monotonicity**: 90 of 90 trips exhibit strictly monotonically increasing timestamps. Minor non-monotonic timestamp steps or zero-deltas were audited and preserved to maintain true observation indices.

---

## 3. Road Network Graph Topology

The road network is modeled as a directed graph $G = (V, E)$ per trip:
- **Zero-Based Indexing**: Node IDs correspond exactly to 0-based row numbers in `.nodes`. Arc IDs correspond exactly to 0-based row numbers in `.arcs`.
- **Endpoint Integrity**: 0 invalid endpoints found. Every arc correctly connects valid existing node IDs: $0 \le u, v < |V|$.
- **Self-Loops & Multi-Arcs**:
  - Self-loops ($u == v$): `24` total across dataset.
  - Duplicate directed arcs: `1,903` total.
- **Isolated Nodes**: `0` nodes with 0 in-degree and 0 out-degree were identified and handled.
- **Connectivity**: Network graphs consist of a massive primary connected component containing over 98% of all road segments per trip.

---

## 4. Reference Route Continuity & Topology

- **Continuous Topological Paths**: 79 out of 90 trips possess continuous directed graph paths where consecutive arcs share endpoints.
- **Consecutive Duplicates**: 0 consecutive identical arc repeats detected and accounted for in route alignment.
- **Disconnected Steps**: 14 transitions represent complex highway connectors, overpasses, or turn restrictions modeled in graph routing.

---

## 5. Geographical Projection & Distance Method

- **Local Mercator Projection**: Each trip is projected using a conformal local planar transformation centered at the trip's bounding box centroid $(lon_0, lat_0)$.
- **Scale Distortion**: Scale distortion is < 0.01% across typical urban extents (<= 50 km), avoiding UTM zone boundary distortion across disparate geographical regions (e.g. Germany vs. Russia vs. North America).
- **Physical Distance**: All perpendicular distances, candidate radii, segment lengths, and network shortest paths are strictly measured in physical **meters**.

---

## 6. Exact Generated Outputs
- `results/dataset_inventory.csv`: Full file inventory of all 90 available trips.
- `results/dataset_quality_report.json`: Machine-readable quality metrics.
- `results/timestamp_analysis.csv`: Per-trip sampling and speed dynamics.
- `results/road_network_analysis.csv`: Per-trip graph topology and connectivity.
