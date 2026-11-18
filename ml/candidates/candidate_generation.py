"""GPS-to-Road candidate generation using two-phase spatial indexing.

Requirements satisfied:
1. Spatial indexing suitable for line geometries: samples intermediate points on segments
   so that long segments are never missed.
2. Broad-phase spatial search with scipy.spatial.cKDTree followed by exact point-to-segment
   perpendicular distance in meters.
3. Preserves original arc IDs, source_node, target_node.
4. Configurable search radius and maximum candidate count.
5. Deterministic sorting and ranking (candidate_rank = 0, 1, 2, ...).
6. Identifies and reports observations with zero candidates.
7. Deduplication guarantee: exactly one candidate per unique arc per observation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ml.preprocessing.graph import DirectedRoadGraph
from ml.preprocessing.loader import TripData


class RoadSegmentSpatialIndex:
    """Two-phase spatial index for directed road segments.

    Broad phase: cKDTree over endpoints and interior sampled points along segments.
    Narrow phase: Exact perpendicular 2D point-to-segment Euclidean distance.
    """

    def __init__(self, graph: DirectedRoadGraph, sample_interval_m: float = 25.0):
        self.graph = graph
        self.sample_interval = float(sample_interval_m)

        # Build sampled point index
        pt_x: List[float] = []
        pt_y: List[float] = []
        pt_arc_id: List[int] = []

        num_arcs = graph.num_arcs
        for a_id in range(num_arcs):
            ax, ay = graph.arc_start_x[a_id], graph.arc_start_y[a_id]
            bx, by = graph.arc_end_x[a_id], graph.arc_end_y[a_id]
            length = graph.arc_lengths[a_id]

            # Start point
            pt_x.append(ax)
            pt_y.append(ay)
            pt_arc_id.append(a_id)

            # End point
            pt_x.append(bx)
            pt_y.append(by)
            pt_arc_id.append(a_id)

            # Intermediate samples along the segment
            if length > self.sample_interval:
                num_steps = int(np.ceil(length / self.sample_interval))
                for step in range(1, num_steps):
                    frac = step / float(num_steps)
                    pt_x.append(ax + frac * (bx - ax))
                    pt_y.append(ay + frac * (by - ay))
                    pt_arc_id.append(a_id)

        self.sample_coords = np.column_stack([pt_x, pt_y])
        self.sample_arc_ids = np.array(pt_arc_id, dtype=int)
        self.kdtree = cKDTree(self.sample_coords)

    def query_point(
        self,
        px: float,
        py: float,
        radius_m: float = 60.0,
        max_candidates: int = 15,
    ) -> List[Dict[str, Any]]:
        """Query road segments within radius_m of (px, py).

        Returns list of candidate dictionaries sorted by distance_m ascending.
        """
        # Broad phase: query points within radius + sample_interval / 2
        broad_radius = radius_m + (self.sample_interval / 2.0)
        idx_neighbors = self.kdtree.query_ball_point([px, py], r=broad_radius)

        if not idx_neighbors:
            return []

        # Unique arc IDs from broad phase
        unique_arcs = set(self.sample_arc_ids[idx_neighbors])

        candidates = []
        for a_id in unique_arcs:
            ax = self.graph.arc_start_x[a_id]
            ay = self.graph.arc_start_y[a_id]
            bx = self.graph.arc_end_x[a_id]
            by = self.graph.arc_end_y[a_id]

            vx = bx - ax
            vy = by - ay
            wx = px - ax
            wy = py - ay

            c1 = wx * vx + wy * vy
            c2 = vx * vx + vy * vy

            if c2 <= 1e-12:
                t = 0.0
                proj_x = ax
                proj_y = ay
            else:
                t = max(0.0, min(1.0, c1 / c2))
                proj_x = ax + t * vx
                proj_y = ay + t * vy

            dist_m = float(np.hypot(px - proj_x, py - proj_y))

            if dist_m <= radius_m:
                candidates.append({
                    "arc_id": int(a_id),
                    "source_node": int(self.graph.arc_from[a_id]),
                    "target_node": int(self.graph.arc_to[a_id]),
                    "distance_m": dist_m,
                    "projection_t": float(t),
                    "proj_x": float(proj_x),
                    "proj_y": float(proj_y),
                    "segment_length_m": float(self.graph.arc_lengths[a_id]),
                    "road_bearing": float(self.graph.arc_bearings[a_id]),
                })

        # Sort strictly by distance_m, break ties deterministically by arc_id
        candidates.sort(key=lambda c: (c["distance_m"], c["arc_id"]))

        # Limit to max_candidates
        selected = candidates[:max_candidates]
        for rank, c in enumerate(selected):
            c["candidate_rank"] = rank

        return selected


def generate_candidates_for_trip(
    trip: TripData,
    graph: Optional[DirectedRoadGraph] = None,
    spatial_index: Optional[RoadSegmentSpatialIndex] = None,
    candidate_radius_m: float = 60.0,
    max_candidates: int = 15,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate candidate road segments for all GPS observations in a trip.

    Returns:
        Tuple of (candidates_df, coverage_summary).
    """
    if graph is None:
        graph = DirectedRoadGraph(trip.nodes, trip.arcs)

    if spatial_index is None:
        spatial_index = RoadSegmentSpatialIndex(graph)

    track_df = trip.track
    gps_lon = track_df["lon"].to_numpy()
    gps_lat = track_df["lat"].to_numpy()
    gps_time = track_df["time"].to_numpy()
    obs_indices = track_df["observation_index"].to_numpy()
    num_obs = len(track_df)

    # Project GPS coordinates
    gps_x, gps_y = graph.projection.forward(gps_lon, gps_lat)

    candidate_rows: List[Dict[str, Any]] = []
    zero_candidate_obs: List[int] = []
    candidate_counts: List[int] = []

    for i in range(num_obs):
        px, py = gps_x[i], gps_y[i]
        cands = spatial_index.query_point(
            px, py, radius_m=candidate_radius_m, max_candidates=max_candidates
        )

        n_cands = len(cands)
        candidate_counts.append(n_cands)

        if n_cands == 0:
            zero_candidate_obs.append(int(obs_indices[i]))
        else:
            for c in cands:
                candidate_rows.append({
                    "trip_id": trip.trip_id,
                    "observation_index": int(obs_indices[i]),
                    "timestamp": float(gps_time[i]),
                    "candidate_rank": int(c["candidate_rank"]),
                    "arc_id": int(c["arc_id"]),
                    "source_node": int(c["source_node"]),
                    "target_node": int(c["target_node"]),
                    "distance_m": float(c["distance_m"]),
                    "projection_t": float(c["projection_t"]),
                    "proj_x": float(c["proj_x"]),
                    "proj_y": float(c["proj_y"]),
                    "gps_x": float(px),
                    "gps_y": float(py),
                    "gps_lon": float(gps_lon[i]),
                    "gps_lat": float(gps_lat[i]),
                    "segment_length_m": float(c["segment_length_m"]),
                    "road_bearing": float(c["road_bearing"]),
                })

    if candidate_rows:
        candidates_df = pd.DataFrame(candidate_rows)
    else:
        # Empty DataFrame with expected columns
        candidates_df = pd.DataFrame(columns=[
            "trip_id", "observation_index", "timestamp", "candidate_rank",
            "arc_id", "source_node", "target_node", "distance_m",
            "projection_t", "proj_x", "proj_y", "gps_x", "gps_y",
            "gps_lon", "gps_lat", "segment_length_m", "road_bearing",
        ])

    coverage_summary = {
        "trip_id": trip.trip_id,
        "total_observations": num_obs,
        "observations_with_candidates": num_obs - len(zero_candidate_obs),
        "observations_without_candidates": len(zero_candidate_obs),
        "zero_candidate_indices": zero_candidate_obs,
        "coverage_ratio": float(1.0 - (len(zero_candidate_obs) / max(1, num_obs))),
        "mean_candidates_per_obs": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "median_candidates_per_obs": float(np.median(candidate_counts)) if candidate_counts else 0.0,
        "min_candidates": int(np.min(candidate_counts)) if candidate_counts else 0,
        "max_candidates": int(np.max(candidate_counts)) if candidate_counts else 0,
        "total_candidate_pairs": len(candidate_rows),
        "candidate_radius_m": float(candidate_radius_m),
        "max_candidates_limit": int(max_candidates),
    }

    return candidates_df, coverage_summary


def evaluate_candidate_coverage(
    trips: List[TripData],
    radii: List[float] = [30.0, 50.0, 60.0, 75.0, 100.0],
    max_candidates: int = 15,
) -> pd.DataFrame:
    """Evaluate candidate generation coverage across multiple radii on development trips."""
    results = []
    for r in radii:
        total_obs = 0
        covered_obs = 0
        total_cands = 0
        for trip in trips:
            graph = DirectedRoadGraph(trip.nodes, trip.arcs)
            idx = RoadSegmentSpatialIndex(graph)
            _, summary = generate_candidates_for_trip(
                trip, graph=graph, spatial_index=idx, candidate_radius_m=r, max_candidates=max_candidates
            )
            total_obs += summary["total_observations"]
            covered_obs += summary["observations_with_candidates"]
            total_cands += summary["total_candidate_pairs"]

        results.append({
            "radius_m": r,
            "total_observations": total_obs,
            "covered_observations": covered_obs,
            "coverage_percent": (covered_obs / max(1, total_obs)) * 100.0,
            "mean_candidates_per_obs": total_cands / max(1, total_obs),
        })

    return pd.DataFrame(results)

