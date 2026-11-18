"""Unit tests for feature engineering."""

import pytest
import numpy as np
import pandas as pd

from ml.preprocessing.loader import TripData
from ml.preprocessing.graph import DirectedRoadGraph
from ml.candidates.candidate_generation import generate_candidates_for_trip
from ml.features.feature_engineering import (
    compute_candidate_features,
    get_feature_schema,
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
)


@pytest.fixture
def synthetic_trip_with_candidates():
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
    track_df = pd.DataFrame({
        "lon": [8.502, 8.508],
        "lat": [49.8001, 49.8001],
        "time": [10.0, 15.0],
        "observation_index": [0, 1],
    })
    trip = TripData(
        trip_id="feat_test_trip",
        partition="train",
        track=track_df,
        nodes=nodes_df,
        arcs=arcs_df,
    )
    graph = DirectedRoadGraph(nodes_df, arcs_df)
    cands_df, _ = generate_candidates_for_trip(trip, graph=graph, candidate_radius_m=60.0)
    return trip, graph, cands_df


def test_feature_columns_and_finite_values(synthetic_trip_with_candidates):
    trip, graph, cands_df = synthetic_trip_with_candidates
    feat_df = compute_candidate_features(trip, cands_df, graph=graph)

    # All identifier and feature columns must be present
    for col in IDENTIFIER_COLUMNS:
        assert col in feat_df.columns, f"Missing identifier column {col}"
    for col in FEATURE_COLUMNS:
        assert col in feat_df.columns, f"Missing feature column {col}"

    # Features must be strictly finite (no NaN, no Inf)
    for col in FEATURE_COLUMNS:
        assert not feat_df[col].isna().any(), f"Column {col} has NaNs"
        assert np.isfinite(feat_df[col].to_numpy()).all(), f"Column {col} has Infs"


def test_schema_documentation():
    schema = get_feature_schema()
    assert "identifiers" in schema
    assert "features" in schema
    assert "targets" in schema
    for col in FEATURE_COLUMNS:
        assert col in schema["features"], f"Feature {col} not documented in schema"
