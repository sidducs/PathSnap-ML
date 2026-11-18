"""Unit tests for candidate generation and spatial indexing."""

import pytest
import numpy as np
import pandas as pd

from ml.preprocessing.loader import TripData
from ml.preprocessing.graph import DirectedRoadGraph
from ml.candidates.candidate_generation import (
    RoadSegmentSpatialIndex,
    generate_candidates_for_trip,
)


@pytest.fixture
def synthetic_trip_for_candidates():
    # Road: straight horizontal road from (0, 0) to (0.01, 0)
    # Node 0 at (0, 0), Node 1 at (0.01, 0)
    nodes_df = pd.DataFrame({
        "node_id": [0, 1, 2],
        "lon": [8.500, 8.510, 8.520],
        "lat": [49.800, 49.800, 49.800],
    })
    arcs_df = pd.DataFrame({
        "arc_id": [0, 1],
        "from_node": [0, 1],
        "to_node": [1, 2],
    })
    # GPS points slightly offset in latitude
    track_df = pd.DataFrame({
        "lon": [8.505, 8.515, 8.999],  # 3rd point is far away (out of radius)
        "lat": [49.8002, 49.8001, 49.999],
        "time": [1.0, 2.0, 3.0],
        "observation_index": [0, 1, 2],
    })
    return TripData(
        trip_id="cand_test_trip",
        partition="train",
        track=track_df,
        nodes=nodes_df,
        arcs=arcs_df,
    )


def test_spatial_index_and_candidate_generation(synthetic_trip_for_candidates):
    trip = synthetic_trip_for_candidates
    graph = DirectedRoadGraph(trip.nodes, trip.arcs)
    spatial_index = RoadSegmentSpatialIndex(graph, sample_interval_m=20.0)

    candidates_df, summary = generate_candidates_for_trip(
        trip, graph=graph, spatial_index=spatial_index, candidate_radius_m=60.0, max_candidates=10
    )

    # Observation 0 should find Arc 0
    obs0_cands = candidates_df[candidates_df["observation_index"] == 0]
    assert len(obs0_cands) >= 1
    assert 0 in obs0_cands["arc_id"].values
    assert obs0_cands.iloc[0]["candidate_rank"] == 0
    assert obs0_cands.iloc[0]["distance_m"] < 60.0

    # Observation 2 is far away and should have 0 candidates within 60m
    obs2_cands = candidates_df[candidates_df["observation_index"] == 2]
    assert len(obs2_cands) == 0

    # Summary checks
    assert summary["total_observations"] == 3
    assert summary["observations_with_candidates"] == 2
    assert summary["observations_without_candidates"] == 1
    assert 2 in summary["zero_candidate_indices"]


def test_no_duplicate_candidates_per_observation(synthetic_trip_for_candidates):
    trip = synthetic_trip_for_candidates
    candidates_df, _ = generate_candidates_for_trip(trip, candidate_radius_m=100.0)
    # Each observation should have unique arc_ids
    for obs_idx, group in candidates_df.groupby("observation_index"):
        assert len(group["arc_id"]) == len(group["arc_id"].unique())

