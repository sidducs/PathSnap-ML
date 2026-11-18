"""Data quality inspection, validation, and diagnostics for PathSnap trips.

Requirements satisfied:
1. Plausibility checks for GPS coordinates (lat in [-90, 90], lon in [-180, 180], 0 is valid).
2. Detection of NaNs, Infs, and duplicate observations.
3. Timestamp ordering and gap analysis.
4. Road network topology diagnostics (self-loops, duplicate arcs, isolated nodes, degree statistics).
5. Reference route continuity analysis (consecutive arc connectivity, gap count).
6. Non-destructive diagnostics: does not silently mutate or discard records.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.preprocessing.loader import TripData


def haversine_distance_vector(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Vectorized Haversine distance in meters between coordinate arrays."""
    r_earth = 6371008.8  # Earth mean radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r_earth * c


def validate_track_data(track_df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect GPS track quality and compute diagnostic statistics."""
    lon = track_df["lon"].to_numpy()
    lat = track_df["lat"].to_numpy()
    t = track_df["time"].to_numpy()
    n_obs = len(track_df)

    # 1. Coordinate validity
    valid_lat = (-90.0 <= lat) & (lat <= 90.0)
    valid_lon = (-180.0 <= lon) & (lon <= 180.0)
    invalid_coords = (~valid_lat) | (~valid_lon)
    num_invalid_coords = int(np.sum(invalid_coords))

    # 2. Duplicate observations
    num_exact_duplicates = int(track_df.duplicated(subset=["lon", "lat", "time"]).sum())
    num_spatial_duplicates = int(track_df.duplicated(subset=["lon", "lat"]).sum())
    num_time_duplicates = int(track_df.duplicated(subset=["time"]).sum())

    # 3. Timestamp analysis
    dt = np.diff(t)
    is_monotonically_increasing = bool(np.all(dt > 0))
    is_non_decreasing = bool(np.all(dt >= 0))
    num_negative_time_jumps = int(np.sum(dt < 0))
    num_zero_time_deltas = int(np.sum(dt == 0))

    if len(dt) > 0:
        dt_mean = float(np.mean(dt))
        dt_median = float(np.median(dt))
        dt_min = float(np.min(dt))
        dt_max = float(np.max(dt))
        dt_std = float(np.std(dt))
        num_gaps_gt_5s = int(np.sum(dt > 5.0))
        num_gaps_gt_10s = int(np.sum(dt > 10.0))
        num_gaps_gt_30s = int(np.sum(dt > 30.0))
        duration_s = float(t[-1] - t[0])
    else:
        dt_mean = dt_median = dt_min = dt_max = dt_std = 0.0
        num_gaps_gt_5s = num_gaps_gt_10s = num_gaps_gt_30s = 0
        duration_s = 0.0

    # 4. Trajectory length and speed
    if n_obs > 1:
        step_distances = haversine_distance_vector(lon[:-1], lat[:-1], lon[1:], lat[1:])
        total_length_m = float(np.sum(step_distances))
        valid_dt_mask = dt > 0
        speeds = np.zeros_like(dt)
        speeds[valid_dt_mask] = step_distances[valid_dt_mask] / dt[valid_dt_mask]
        mean_speed_m_s = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
        max_speed_m_s = float(np.max(speeds)) if len(speeds) > 0 else 0.0
    else:
        total_length_m = 0.0
        mean_speed_m_s = 0.0
        max_speed_m_s = 0.0

    # 5. Spatial bounding box
    bbox = {
        "min_lon": float(np.min(lon)),
        "max_lon": float(np.max(lon)),
        "min_lat": float(np.min(lat)),
        "max_lat": float(np.max(lat)),
        "center_lon": float(np.mean(lon)),
        "center_lat": float(np.mean(lat)),
        "span_lon_deg": float(np.max(lon) - np.min(lon)),
        "span_lat_deg": float(np.max(lat) - np.min(lat)),
    }

    return {
        "num_observations": n_obs,
        "num_invalid_coords": num_invalid_coords,
        "num_exact_duplicates": num_exact_duplicates,
        "num_spatial_duplicates": num_spatial_duplicates,
        "num_time_duplicates": num_time_duplicates,
        "is_monotonically_increasing": is_monotonically_increasing,
        "is_non_decreasing": is_non_decreasing,
        "num_negative_time_jumps": num_negative_time_jumps,
        "num_zero_time_deltas": num_zero_time_deltas,
        "duration_s": duration_s,
        "sampling_dt_mean": dt_mean,
        "sampling_dt_median": dt_median,
        "sampling_dt_min": dt_min,
        "sampling_dt_max": dt_max,
        "sampling_dt_std": dt_std,
        "num_gaps_gt_5s": num_gaps_gt_5s,
        "num_gaps_gt_10s": num_gaps_gt_10s,
        "num_gaps_gt_30s": num_gaps_gt_30s,
        "total_trajectory_length_m": total_length_m,
        "mean_speed_m_s": mean_speed_m_s,
        "max_speed_m_s": max_speed_m_s,
        "bbox": bbox,
    }


def validate_road_network(nodes_df: pd.DataFrame, arcs_df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect road network topology and degree statistics."""
    num_nodes = len(nodes_df)
    num_arcs = len(arcs_df)

    from_nodes = arcs_df["from_node"].to_numpy()
    to_nodes = arcs_df["to_node"].to_numpy()

    # 1. Endpoint validity
    invalid_from = (from_nodes < 0) | (from_nodes >= num_nodes)
    invalid_to = (to_nodes < 0) | (to_nodes >= num_nodes)
    num_invalid_endpoints = int(np.sum(invalid_from | invalid_to))

    # 2. Self-loops (from_node == to_node)
    num_self_loops = int(np.sum(from_nodes == to_nodes))

    # 3. Duplicate directed arcs
    num_duplicate_arcs = int(arcs_df.duplicated(subset=["from_node", "to_node"]).sum())

    # 4. In-degree and Out-degree calculation
    out_degrees = np.bincount(from_nodes[~invalid_from], minlength=num_nodes)
    in_degrees = np.bincount(to_nodes[~invalid_to], minlength=num_nodes)
    total_degrees = in_degrees + out_degrees

    num_isolated_nodes = int(np.sum(total_degrees == 0))
    num_sources = int(np.sum((out_degrees > 0) & (in_degrees == 0)))
    num_sinks = int(np.sum((in_degrees > 0) & (out_degrees == 0)))

    # 5. Connected components (Weakly connected via BFS)
    # Build undirected adjacency for weak connectivity
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    for u, v in zip(from_nodes, to_nodes):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u].append(v)
            adj[v].append(u)

    visited = np.zeros(num_nodes, dtype=bool)
    num_weakly_connected_components = 0
    max_component_size = 0

    for i in range(num_nodes):
        if not visited[i]:
            num_weakly_connected_components += 1
            comp_size = 0
            stack = [i]
            visited[i] = True
            while stack:
                curr = stack.pop()
                comp_size += 1
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            if comp_size > max_component_size:
                max_component_size = comp_size

    # Node geographic bbox
    n_lon = nodes_df["lon"].to_numpy()
    n_lat = nodes_df["lat"].to_numpy()
    node_bbox = {
        "min_lon": float(np.min(n_lon)),
        "max_lon": float(np.max(n_lon)),
        "min_lat": float(np.min(n_lat)),
        "max_lat": float(np.max(n_lat)),
    }

    return {
        "num_nodes": num_nodes,
        "num_arcs": num_arcs,
        "num_invalid_endpoints": num_invalid_endpoints,
        "num_self_loops": num_self_loops,
        "num_duplicate_arcs": num_duplicate_arcs,
        "num_isolated_nodes": num_isolated_nodes,
        "num_source_nodes": num_sources,
        "num_sink_nodes": num_sinks,
        "mean_out_degree": float(np.mean(out_degrees)),
        "max_out_degree": int(np.max(out_degrees)) if len(out_degrees) > 0 else 0,
        "mean_in_degree": float(np.mean(in_degrees)),
        "max_in_degree": int(np.max(in_degrees)) if len(in_degrees) > 0 else 0,
        "num_weakly_connected_components": num_weakly_connected_components,
        "largest_component_nodes": max_component_size,
        "node_bbox": node_bbox,
    }


def validate_route(arcs_df: pd.DataFrame, route_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Inspect reference route sequence continuity and topology."""
    if route_df is None or len(route_df) == 0:
        return {
            "has_route": False,
            "num_route_arcs": 0,
            "is_continuous": False,
            "num_consecutive_duplicates": 0,
            "num_disconnected_steps": 0,
            "num_invalid_route_arcs": 0,
        }

    num_arcs = len(arcs_df)
    route_arcs = route_df["arc_id"].to_numpy()
    n_route = len(route_arcs)

    # 1. Validate route arc IDs
    invalid_mask = (route_arcs < 0) | (route_arcs >= num_arcs)
    num_invalid_route_arcs = int(np.sum(invalid_mask))

    # 2. Consecutive duplicates
    if n_route > 1:
        consecutive_dups = int(np.sum(route_arcs[:-1] == route_arcs[1:]))
    else:
        consecutive_dups = 0

    # 3. Connectivity along the reference path
    # For consecutive arcs A1 and A2, A1.to_node must equal A2.from_node for continuous path
    num_disconnected_steps = 0
    if n_route > 1:
        from_nodes = arcs_df["from_node"].to_numpy()
        to_nodes = arcs_df["to_node"].to_numpy()

        for i in range(n_route - 1):
            a1 = route_arcs[i]
            a2 = route_arcs[i + 1]
            if 0 <= a1 < num_arcs and 0 <= a2 < num_arcs:
                if to_nodes[a1] != from_nodes[a2]:
                    num_disconnected_steps += 1
            else:
                num_disconnected_steps += 1

    is_continuous = (n_route > 0) and (num_disconnected_steps == 0) and (num_invalid_route_arcs == 0)

    return {
        "has_route": True,
        "num_route_arcs": n_route,
        "is_continuous": is_continuous,
        "num_consecutive_duplicates": consecutive_dups,
        "num_disconnected_steps": num_disconnected_steps,
        "num_invalid_route_arcs": num_invalid_route_arcs,
    }


def clean_and_diagnose_trip(trip: TripData) -> Dict[str, Any]:
    """Run full diagnostic suite on a TripData instance."""
    track_report = validate_track_data(trip.track)
    road_report = validate_road_network(trip.nodes, trip.arcs)
    route_report = validate_route(trip.arcs, trip.route)

    return {
        "trip_id": trip.trip_id,
        "partition": trip.partition,
        "track_diagnostics": track_report,
        "road_diagnostics": road_report,
        "route_diagnostics": route_report,
    }

