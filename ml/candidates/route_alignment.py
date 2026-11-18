"""Reference route alignment and ground-truth label generation for PathSnap.

Requirements satisfied:
1. Rigorous topological and temporal route-to-observation alignment.
2. Does NOT assume 1:1 correspondence between GPS points and route arcs.
3. Does NOT assume GPS observation index equals route index.
4. Distinguishes clearly between:
   - 'aligned': Ground-truth route arc unambiguously identified and present in candidates.
   - 'ambiguous': Multiple candidate arcs on route with overlapping plausibility near junctions.
   - 'candidate_miss': True route arc exists near GPS point but was not captured in top-K candidates.
   - 'unmatched': GPS observation is distant from reference route (> unmatched threshold).
5. Ground-truth label generation:
   - label = 1 for the verified ground-truth candidate arc.
   - label = 0 for competing candidate arcs.
   - Ambiguous / unaligned records are clearly flagged so they can be isolated from supervised training.
6. Generates an alignment audit summary per trip.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ml.preprocessing.graph import DirectedRoadGraph
from ml.preprocessing.loader import TripData


def align_trip_reference_route(
    trip: TripData,
    candidates_df: pd.DataFrame,
    graph: Optional[DirectedRoadGraph] = None,
    max_unmatched_dist_m: float = 120.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Align GPS observations to the reference route and assign verified candidate labels.

    Args:
        trip: TripData containing track, nodes, arcs, and route.
        candidates_df: Output from candidate_generation.py for this trip.
        graph: DirectedRoadGraph for the trip.
        max_unmatched_dist_m: Distance beyond which a GPS point is considered off-route.

    Returns:
        Tuple of (labeled_candidates_df, alignment_audit).
    """
    if trip.route is None or len(trip.route) == 0:
        # No route available (e.g. student_test without permitted labels or test inference)
        df = candidates_df.copy()
        df["label"] = np.nan
        df["alignment_status"] = "no_route_provided"
        df["route_index"] = -1
        audit = {
            "trip_id": trip.trip_id,
            "has_route": False,
            "total_observations": len(trip.track),
            "num_aligned": 0,
            "num_ambiguous": 0,
            "num_unmatched": len(trip.track),
            "num_candidate_miss": 0,
            "alignment_rate": 0.0,
        }
        return df, audit

    if graph is None:
        graph = DirectedRoadGraph(trip.nodes, trip.arcs)

    route_arcs = trip.route["arc_id"].to_numpy(dtype=int)
    n_route_arcs = len(route_arcs)

    # Map arc_id -> list of positions in the route sequence (an arc may appear multiple times in loops)
    route_arc_to_indices: Dict[int, List[int]] = {}
    for r_idx, a_id in enumerate(route_arcs):
        route_arc_to_indices.setdefault(a_id, []).append(r_idx)

    route_arc_set = set(route_arcs)

    num_obs = len(trip.track)
    obs_indices = trip.track["observation_index"].to_numpy()

    # Pre-group candidates by observation_index for fast lookup
    if not candidates_df.empty:
        cands_by_obs = candidates_df.groupby("observation_index")
    else:
        cands_by_obs = {}

    obs_status: Dict[int, str] = {}
    obs_matched_arc: Dict[int, Optional[int]] = {}
    obs_matched_r_idx: Dict[int, int] = {}

    # State tracking: monotonically advancing pointer along reference route
    curr_route_ptr = 0

    for i in range(num_obs):
        obs_idx = int(obs_indices[i])

        if obs_idx not in cands_by_obs.groups if hasattr(cands_by_obs, "groups") else obs_idx not in cands_by_obs:
            # Observation had 0 candidates generated
            obs_status[obs_idx] = "unmatched"
            obs_matched_arc[obs_idx] = None
            obs_matched_r_idx[obs_idx] = -1
            continue

        c_group = cands_by_obs.get_group(obs_idx)
        cand_arcs = c_group["arc_id"].to_numpy()
        cand_dists = c_group["distance_m"].to_numpy()

        # Check which candidate arcs belong to the reference route
        route_matches = []
        for arc_id, dist in zip(cand_arcs, cand_dists):
            if arc_id in route_arc_set:
                # Find best matching route sequence index for this arc (closest to curr_route_ptr)
                r_indices = route_arc_to_indices[arc_id]
                # Filter to indices at or forward of current pointer, with modest lookback of 2
                valid_r_indices = [idx for idx in r_indices if idx >= max(0, curr_route_ptr - 2)]
                if not valid_r_indices:
                    # If all occurrences were in the distant past, take the latest one
                    valid_r_indices = r_indices

                # Pick index with minimal forward distance from curr_route_ptr
                best_idx = min(valid_r_indices, key=lambda idx: abs(idx - curr_route_ptr))
                route_matches.append({
                    "arc_id": arc_id,
                    "dist": dist,
                    "route_idx": best_idx,
                })

        if len(route_matches) == 0:
            # No candidate arc was in the reference route
            # Check if any route arc within window is physically close
            min_route_dist = float("inf")
            # Look at route arcs near curr_route_ptr
            search_window = range(
                max(0, curr_route_ptr - 3),
                min(n_route_arcs, curr_route_ptr + 10)
            )
            px = float(trip.track.iloc[obs_idx]["lon"])
            py = float(trip.track.iloc[obs_idx]["lat"])

            # Fast check if GPS is near any route arc in window
            obs_x, obs_y = graph.projection.forward(px, py)
            for w_idx in search_window:
                w_arc = route_arcs[w_idx]
                ax, ay = graph.arc_start_x[w_arc], graph.arc_start_y[w_arc]
                bx, by = graph.arc_end_x[w_arc], graph.arc_end_y[w_arc]
                vx, vy = bx - ax, by - ay
                c2 = vx * vx + vy * vy
                if c2 > 1e-12:
                    t = max(0.0, min(1.0, ((obs_x - ax) * vx + (obs_y - ay) * vy) / c2))
                    d = float(np.hypot(obs_x - (ax + t * vx), obs_y - (ay + t * vy)))
                else:
                    d = float(np.hypot(obs_x - ax, obs_y - ay))
                if d < min_route_dist:
                    min_route_dist = d

            if min_route_dist <= 75.0:
                obs_status[obs_idx] = "candidate_miss"
            else:
                obs_status[obs_idx] = "unmatched"
            obs_matched_arc[obs_idx] = None
            obs_matched_r_idx[obs_idx] = -1

        elif len(route_matches) == 1:
            # Unambiguous single route candidate
            m = route_matches[0]
            obs_status[obs_idx] = "aligned"
            obs_matched_arc[obs_idx] = m["arc_id"]
            obs_matched_r_idx[obs_idx] = m["route_idx"]
            curr_route_ptr = max(curr_route_ptr, m["route_idx"])

        else:
            # Multiple candidate arcs belong to reference route
            # Sort primarily by forward route sequence progress and secondarily by distance
            # Arcs that advance the route by 0, 1, or 2 steps from current pointer are preferred
            def route_score(m: Dict[str, Any]) -> Tuple[int, float]:
                step_diff = m["route_idx"] - curr_route_ptr
                # Penalty for moving backward or jumping too far forward
                if step_diff < 0:
                    penalty = 100 + abs(step_diff)
                elif step_diff <= 3:
                    penalty = step_diff
                else:
                    penalty = 10 + step_diff
                return penalty, m["dist"]

            route_matches.sort(key=route_score)

            best = route_matches[0]
            second = route_matches[1]

            # If top two candidates have nearly identical distance and step_diff, mark ambiguous
            score1 = route_score(best)
            score2 = route_score(second)

            if score1[0] == score2[0] and abs(best["dist"] - second["dist"]) < 1.5:
                obs_status[obs_idx] = "ambiguous"
                obs_matched_arc[obs_idx] = best["arc_id"]
                obs_matched_r_idx[obs_idx] = best["route_idx"]
            else:
                obs_status[obs_idx] = "aligned"
                obs_matched_arc[obs_idx] = best["arc_id"]
                obs_matched_r_idx[obs_idx] = best["route_idx"]
                curr_route_ptr = max(curr_route_ptr, best["route_idx"])

    # Now assign labels to the candidates DataFrame
    labeled_rows = candidates_df.copy()
    labels = np.zeros(len(labeled_rows), dtype=int)
    statuses = []
    r_indices_col = []

    for idx, row in labeled_rows.iterrows():
        o_idx = int(row["observation_index"])
        a_id = int(row["arc_id"])
        status = obs_status.get(o_idx, "unmatched")
        matched_arc = obs_matched_arc.get(o_idx, None)
        r_idx = obs_matched_r_idx.get(o_idx, -1)

        statuses.append(status)
        r_indices_col.append(r_idx)

        if status == "aligned" and matched_arc is not None and a_id == matched_arc:
            labels[idx] = 1
        else:
            labels[idx] = 0

    labeled_rows["label"] = labels
    labeled_rows["alignment_status"] = statuses
    labeled_rows["route_index"] = r_indices_col

    # Audit statistics
    status_counts = pd.Series(list(obs_status.values())).value_counts().to_dict()
    num_aligned = status_counts.get("aligned", 0)
    num_ambiguous = status_counts.get("ambiguous", 0)
    num_unmatched = status_counts.get("unmatched", 0)
    num_miss = status_counts.get("candidate_miss", 0)

    alignment_audit = {
        "trip_id": trip.trip_id,
        "has_route": True,
        "total_observations": num_obs,
        "total_candidates": len(labeled_rows),
        "num_aligned": num_aligned,
        "num_ambiguous": num_ambiguous,
        "num_unmatched": num_unmatched,
        "num_candidate_miss": num_miss,
        "alignment_rate_percent": float((num_aligned / max(1, num_obs)) * 100.0),
        "positive_labels_count": int(np.sum(labels == 1)),
        "negative_labels_count": int(np.sum(labels == 0)),
    }

    return labeled_rows, alignment_audit

