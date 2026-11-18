"""Unit tests for reference route alignment and ground truth labeling."""

import pytest
import numpy as np
import pandas as pd

from ml.preprocessing.loader import TripData
from ml.preprocessing.graph import DirectedRoadGraph
from ml.candidates.candidate_generation import generate_candidates_for_trip
from ml.candidates.route_alignment import align_trip_reference_route


def test_reference_route_alignment():
    nodes_df = pd.DataFrame({
        "node_id": [0, 1, 2],
        "lon": [8.500, 8.510, 8.520],
        "lat": [49.800, 49.800, 49.800],
    })
    # Two arcs: Arc 0 (0->1) and Arc 1 (1->2)
    arcs_df = pd.DataFrame({
        "arc_id": [0, 1],
        "from_node": [0, 1],
        "to_node": [1, 2],
    })
    # GPS trajectory moving from Arc 0 to Arc 1
    track_df = pd.DataFrame({
        "lon": [8.505, 8.515],
        "lat": [49.8001, 49.8001],
        "time": [1.0, 2.0],
        "observation_index": [0, 1],
    })
    # Ground truth route: [0, 1]
    route_df = pd.DataFrame({
        "route_order": [0, 1],
        "arc_id": [0, 1],
    })

    trip = TripData(
        trip_id="align_test_trip",
        partition="train",
        track=track_df,
        nodes=nodes_df,
        arcs=arcs_df,
        route=route_df,
    )

    graph = DirectedRoadGraph(nodes_df, arcs_df)
    cands_df, _ = generate_candidates_for_trip(trip, graph=graph, candidate_radius_m=60.0)

    labeled_df, audit = align_trip_reference_route(trip, cands_df, graph=graph)

    assert "label" in labeled_df.columns
    assert "alignment_status" in labeled_df.columns
    assert audit["num_aligned"] == 2
    assert audit["alignment_rate_percent"] == 100.0

    # Observation 0 must have positive label on Arc 0
    obs0_labels = labeled_df[labeled_df["observation_index"] == 0]
    matched_0 = obs0_labels[obs0_labels["arc_id"] == 0]
    assert len(matched_0) == 1
    assert matched_0.iloc[0]["label"] == 1

    # Observation 1 must have positive label on Arc 1
    obs1_labels = labeled_df[labeled_df["observation_index"] == 1]
    matched_1 = obs1_labels[obs1_labels["arc_id"] == 1]
    assert len(matched_1) == 1
    assert matched_1.iloc[0]["label"] == 1
