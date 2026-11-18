"""Candidate-level feature engineering for map-matching classifiers.

Requirements satisfied:
1. Spatial features: perpendicular distance, candidate rank, segment length, projection fraction.
2. Directional features: GPS bearing, road bearing, heading delta, cosine similarity, agreement flag.
3. Temporal and movement features: delta t, displacement, speed, acceleration.
4. Road network features: degrees, connectivity to previous candidate, network transition distance.
5. Finite values guarantee: imputes boundary/missing values with deterministic defaults.
6. Clean separation: identifiers vs. features vs. labels to prevent accidental target leakage.
7. Exportable JSON schema with descriptions and units for Siddhu's ML pipeline.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.preprocessing.graph import DirectedRoadGraph
from ml.preprocessing.loader import TripData
from ml.preprocessing.projection import (
    calculate_bearing,
    calculate_heading_difference,
    haversine_distance,
)

# Identifiers that MUST be preserved but NOT used as direct predictive features
IDENTIFIER_COLUMNS = [
    "trip_id",
    "observation_index",
    "timestamp",
    "arc_id",
    "source_node",
    "target_node",
]

# Engineered feature columns
FEATURE_COLUMNS = [
    # Spatial features
    "distance_m",
    "candidate_rank",
    "projection_t",
    "segment_length_m",
    "proj_offset_x",
    "proj_offset_y",
    "log_distance_m",
    "emission_prob_proxy",
    # Directional features
    "gps_bearing",
    "road_bearing",
    "heading_delta",
    "heading_cos_sim",
    "direction_agreement",
    "is_antiparallel",
    # Temporal & movement features
    "dt_prev",
    "gps_displacement_m",
    "gps_speed_m_s",
    "gps_accel_m_s2",
    # Road network & transition features
    "source_node_out_degree",
    "target_node_in_degree",
    "source_node_in_degree",
    "target_node_out_degree",
    "is_source_deadend",
    "is_target_deadend",
    "is_connected_to_prev_top_cand",
    "graph_transition_dist_m",
    "transition_excess_ratio",
]

# Label & audit columns
TARGET_COLUMNS = [
    "label",
    "alignment_status",
    "route_index",
]


def get_feature_schema() -> Dict[str, Any]:
    """Return JSON schema describing all feature columns, types, units, and descriptions."""
    schema = {
        "identifiers": {
            "trip_id": {"type": "string", "description": "Unique identifier of the trip"},
            "observation_index": {"type": "integer", "description": "Zero-based index of the GPS observation in chronological order"},
            "timestamp": {"type": "float", "unit": "seconds", "description": "Timestamp in seconds from trip start"},
            "arc_id": {"type": "integer", "description": "Candidate directed road segment arc ID"},
            "source_node": {"type": "integer", "description": "Origin node ID of the candidate arc"},
            "target_node": {"type": "integer", "description": "Destination node ID of the candidate arc"},
        },
        "features": {
            "distance_m": {"type": "float", "unit": "meters", "description": "Perpendicular distance from GPS observation to candidate segment"},
            "candidate_rank": {"type": "integer", "description": "Rank of candidate ordered by distance_m (0 = nearest)"},
            "projection_t": {"type": "float", "unit": "ratio [0, 1]", "description": "Fractional projection position along the segment (0=source, 1=target)"},
            "segment_length_m": {"type": "float", "unit": "meters", "description": "Physical length of the candidate road segment"},
            "proj_offset_x": {"type": "float", "unit": "meters", "description": "Cartesian X offset from GPS point to projected point on segment"},
            "proj_offset_y": {"type": "float", "unit": "meters", "description": "Cartesian Y offset from GPS point to projected point on segment"},
            "log_distance_m": {"type": "float", "unit": "log(meters+1)", "description": "Natural logarithm of (1 + distance_m)"},
            "emission_prob_proxy": {"type": "float", "unit": "probability proxy", "description": "Gaussian emission probability exp(-dist^2 / (2 * 15^2))"},
            "gps_bearing": {"type": "float", "unit": "degrees [0, 360)", "description": "GPS trajectory instantaneous heading from previous observation"},
            "road_bearing": {"type": "float", "unit": "degrees [0, 360)", "description": "Orientation of the directed road segment from source to target"},
            "heading_delta": {"type": "float", "unit": "degrees [0, 180]", "description": "Smallest angular difference between GPS heading and road bearing"},
            "heading_cos_sim": {"type": "float", "unit": "[-1, 1]", "description": "Cosine of heading difference (1 = same direction, -1 = reverse)"},
            "direction_agreement": {"type": "float", "unit": "binary {0.0, 1.0}", "description": "1.0 if heading_delta <= 45 degrees, else 0.0"},
            "is_antiparallel": {"type": "float", "unit": "binary {0.0, 1.0}", "description": "1.0 if heading_delta >= 135 degrees (opposing flow), else 0.0"},
            "dt_prev": {"type": "float", "unit": "seconds", "description": "Time elapsed since previous GPS observation"},
            "gps_displacement_m": {"type": "float", "unit": "meters", "description": "Haversine distance between previous and current GPS observations"},
            "gps_speed_m_s": {"type": "float", "unit": "meters/second", "description": "GPS movement speed between previous and current observations"},
            "gps_accel_m_s2": {"type": "float", "unit": "meters/second^2", "description": "Estimated acceleration between consecutive speed measurements"},
            "source_node_out_degree": {"type": "integer", "description": "Number of outgoing road segments from candidate source node"},
            "target_node_in_degree": {"type": "integer", "description": "Number of incoming road segments to candidate target node"},
            "source_node_in_degree": {"type": "integer", "description": "Number of incoming road segments to candidate source node"},
            "target_node_out_degree": {"type": "integer", "description": "Number of outgoing road segments from candidate target node"},
            "is_source_deadend": {"type": "float", "unit": "binary {0.0, 1.0}", "description": "1.0 if source node has in-degree == 0"},
            "is_target_deadend": {"type": "float", "unit": "binary {0.0, 1.0}", "description": "1.0 if target node has out-degree == 0"},
            "is_connected_to_prev_top_cand": {"type": "float", "unit": "binary {0.0, 1.0}", "description": "1.0 if candidate is directly reachable from previous top candidate"},
            "graph_transition_dist_m": {"type": "float", "unit": "meters", "description": "Shortest directed network path distance from previous top candidate"},
            "transition_excess_ratio": {"type": "float", "unit": "ratio", "description": "(graph_transition_dist - gps_displacement) / max(1, gps_displacement)"},
        },
        "targets": {
            "label": {"type": "integer", "values": [0, 1], "description": "1 if candidate arc matches ground-truth route arc, 0 otherwise"},
            "alignment_status": {"type": "string", "values": ["aligned", "ambiguous", "candidate_miss", "unmatched"], "description": "Quality status of reference route association"},
            "route_index": {"type": "integer", "description": "Sequence position in reference route if matched, or -1"},
        },
    }
    return schema


def compute_candidate_features(
    trip: TripData,
    candidates_df: pd.DataFrame,
    graph: Optional[DirectedRoadGraph] = None,
) -> pd.DataFrame:
    """Compute candidate-level features for a trip.

    Args:
        trip: TripData instance.
        candidates_df: Candidate DataFrame (possibly with route labels).
        graph: DirectedRoadGraph instance.

    Returns:
        pd.DataFrame containing IDENTIFIER_COLUMNS + FEATURE_COLUMNS (+ TARGET_COLUMNS if present).
    """
    if candidates_df.empty:
        # Return empty DataFrame with full schema
        cols = IDENTIFIER_COLUMNS + FEATURE_COLUMNS
        if "label" in candidates_df.columns:
            cols += TARGET_COLUMNS
        return pd.DataFrame(columns=cols)

    if graph is None:
        graph = DirectedRoadGraph(trip.nodes, trip.arcs)

    track_df = trip.track.set_index("observation_index")
    num_obs = len(trip.track)

    # 1. Precompute GPS trajectory movement features
    lons = trip.track["lon"].to_numpy()
    lats = trip.track["lat"].to_numpy()
    times = trip.track["time"].to_numpy()

    dt_array = np.zeros(num_obs, dtype=float)
    disp_array = np.zeros(num_obs, dtype=float)
    speed_array = np.zeros(num_obs, dtype=float)
    gps_bearing_array = np.zeros(num_obs, dtype=float)

    if num_obs > 1:
        dt_array[1:] = np.maximum(0.001, times[1:] - times[:-1])
        disp_array[1:] = haversine_distance(lons[:-1], lats[:-1], lons[1:], lats[1:])
        speed_array[1:] = disp_array[1:] / dt_array[1:]
        gps_bearing_array[1:] = calculate_bearing(lons[:-1], lats[:-1], lons[1:], lats[1:])
        # For the very first point, copy forward from point 1
        gps_bearing_array[0] = gps_bearing_array[1]
        dt_array[0] = dt_array[1]
        disp_array[0] = disp_array[1]
        speed_array[0] = speed_array[1]

    # Accelerations
    accel_array = np.zeros(num_obs, dtype=float)
    if num_obs > 1:
        accel_array[1:] = (speed_array[1:] - speed_array[:-1]) / dt_array[1:]

    # Map observation_index -> previous top candidate arc_id for transition features
    # Rank 0 candidate is the spatial nearest candidate
    rank0_candidates = candidates_df[candidates_df["candidate_rank"] == 0]
    rank0_map: Dict[int, int] = dict(zip(rank0_candidates["observation_index"], rank0_candidates["arc_id"]))

    # 2. Extract and vectorise candidate columns
    obs_idx = candidates_df["observation_index"].to_numpy(dtype=int)
    arc_ids = candidates_df["arc_id"].to_numpy(dtype=int)
    src_nodes = candidates_df["source_node"].to_numpy(dtype=int)
    tgt_nodes = candidates_df["target_node"].to_numpy(dtype=int)
    dist_m = candidates_df["distance_m"].to_numpy(dtype=float)
    rank = candidates_df["candidate_rank"].to_numpy(dtype=int)
    proj_t = candidates_df["projection_t"].to_numpy(dtype=float)
    proj_x = candidates_df["proj_x"].to_numpy(dtype=float)
    proj_y = candidates_df["proj_y"].to_numpy(dtype=float)
    gps_x = candidates_df["gps_x"].to_numpy(dtype=float)
    gps_y = candidates_df["gps_y"].to_numpy(dtype=float)
    seg_len = candidates_df["segment_length_m"].to_numpy(dtype=float)
    road_bearing = candidates_df["road_bearing"].to_numpy(dtype=float)

    n_cands = len(candidates_df)

    # 3. Spatial features
    proj_offset_x = proj_x - gps_x
    proj_offset_y = proj_y - gps_y
    log_distance_m = np.log1p(np.maximum(0.0, dist_m))
    # Gaussian proxy with sigma = 15.0m
    emission_prob_proxy = np.exp(-0.5 * (dist_m / 15.0) ** 2)

    # 4. Directional features
    c_gps_bearing = gps_bearing_array[obs_idx]
    heading_delta = calculate_heading_difference(c_gps_bearing, road_bearing)
    heading_cos_sim = np.cos(np.radians(heading_delta))
    direction_agreement = np.where(heading_delta <= 45.0, 1.0, 0.0)
    is_antiparallel = np.where(heading_delta >= 135.0, 1.0, 0.0)

    # 5. Temporal & movement features
    c_dt_prev = dt_array[obs_idx]
    c_gps_disp = disp_array[obs_idx]
    c_gps_speed = speed_array[obs_idx]
    c_gps_accel = accel_array[obs_idx]

    # 6. Road network topological features
    src_out_deg = graph.out_degrees[src_nodes]
    tgt_in_deg = graph.in_degrees[tgt_nodes]
    src_in_deg = graph.in_degrees[src_nodes]
    tgt_out_deg = graph.out_degrees[tgt_nodes]
    is_src_deadend = np.where(src_in_deg == 0, 1.0, 0.0)
    is_tgt_deadend = np.where(tgt_out_deg == 0, 1.0, 0.0)

    # 7. Transition & connectivity to previous top candidate
    is_connected = np.zeros(n_cands, dtype=float)
    graph_trans_dist = np.full(n_cands, 300.0, dtype=float)  # Default cap

    # Cache single-source Dijkstra neighborhoods from prev exit nodes
    node_dijkstra_cache: Dict[int, Dict[int, float]] = {}

    for i in range(n_cands):
        curr_obs = obs_idx[i]
        curr_arc = arc_ids[i]
        curr_from_node = src_nodes[i]
        prev_obs = curr_obs - 1

        if prev_obs in rank0_map:
            prev_top_arc = rank0_map[prev_obs]
            if prev_top_arc == curr_arc:
                is_connected[i] = 1.0
                graph_trans_dist[i] = 0.0
            elif graph.arc_to[prev_top_arc] == curr_from_node:
                is_connected[i] = 1.0
                graph_trans_dist[i] = 0.0
            else:
                is_connected[i] = 0.0
                prev_exit_node = int(graph.arc_to[prev_top_arc])
                if prev_exit_node not in node_dijkstra_cache:
                    node_dijkstra_cache[prev_exit_node] = graph.dijkstra_neighborhood(prev_exit_node, max_distance=300.0)
                dists = node_dijkstra_cache[prev_exit_node]
                graph_trans_dist[i] = dists.get(curr_from_node, 300.0)
        else:
            # First observation has no predecessor
            is_connected[i] = 1.0
            graph_trans_dist[i] = 0.0

    # Excess ratio: (graph_distance - gps_displacement) / max(1.0, gps_displacement)
    trans_excess_ratio = (graph_trans_dist - c_gps_disp) / np.maximum(1.0, c_gps_disp)
    # Clip extreme values to keep finite
    trans_excess_ratio = np.clip(trans_excess_ratio, -5.0, 20.0)

    # Build feature DataFrame
    feat_df = pd.DataFrame({
        "trip_id": candidates_df["trip_id"].to_numpy(),
        "observation_index": obs_idx,
        "timestamp": candidates_df["timestamp"].to_numpy(dtype=float),
        "arc_id": arc_ids,
        "source_node": src_nodes,
        "target_node": tgt_nodes,
        # Spatial
        "distance_m": dist_m,
        "candidate_rank": rank,
        "projection_t": proj_t,
        "segment_length_m": seg_len,
        "proj_offset_x": proj_offset_x,
        "proj_offset_y": proj_offset_y,
        "log_distance_m": log_distance_m,
        "emission_prob_proxy": emission_prob_proxy,
        # Directional
        "gps_bearing": c_gps_bearing,
        "road_bearing": road_bearing,
        "heading_delta": heading_delta,
        "heading_cos_sim": heading_cos_sim,
        "direction_agreement": direction_agreement,
        "is_antiparallel": is_antiparallel,
        # Temporal & movement
        "dt_prev": c_dt_prev,
        "gps_displacement_m": c_gps_disp,
        "gps_speed_m_s": c_gps_speed,
        "gps_accel_m_s2": c_gps_accel,
        # Road network & transition
        "source_node_out_degree": src_out_deg,
        "target_node_in_degree": tgt_in_deg,
        "source_node_in_degree": src_in_deg,
        "target_node_out_degree": tgt_out_deg,
        "is_source_deadend": is_src_deadend,
        "is_target_deadend": is_tgt_deadend,
        "is_connected_to_prev_top_cand": is_connected,
        "graph_transition_dist_m": graph_trans_dist,
        "transition_excess_ratio": trans_excess_ratio,
    })

    # Append labels if available
    for col in TARGET_COLUMNS:
        if col in candidates_df.columns:
            feat_df[col] = candidates_df[col].to_numpy()

    # Ensure no NaN or infinite values in feature columns
    for col in FEATURE_COLUMNS:
        feat_df[col] = feat_df[col].fillna(0.0)
        feat_df[col] = np.nan_to_num(feat_df[col].to_numpy(), nan=0.0, posinf=999.0, neginf=-999.0)

    return feat_df

