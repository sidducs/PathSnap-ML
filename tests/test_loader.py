"""Unit tests for ml.preprocessing.loader and ml.preprocessing.cleaning."""

import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from ml.preprocessing.loader import (
    load_track,
    load_nodes,
    load_arcs,
    load_route,
    load_trip,
    validate_trip_endpoints,
    validate_route_arc_ids,
    MalformedFileError,
    ValidationError,
    TripData,
)
from ml.preprocessing.cleaning import (
    validate_track_data,
    validate_road_network,
    validate_route,
    clean_and_diagnose_trip,
)


@pytest.fixture
def temp_trip_dir(tmp_path):
    """Create a valid synthetic trip directory."""
    trip_id = "test_trip_01"
    d = tmp_path / trip_id
    d.mkdir()

    # Track: lon, lat, time (3 rows)
    track_content = "8.580 49.850 1.0\n8.581 49.851 2.0\n8.582 49.852 3.0\n"
    (d / f"{trip_id}.track").write_text(track_content)

    # Nodes: lon, lat (3 nodes: 0, 1, 2)
    nodes_content = "8.580 49.850\n8.581 49.851\n8.582 49.852\n"
    (d / f"{trip_id}.nodes").write_text(nodes_content)

    # Arcs: from_node, to_node (2 arcs: 0->1, 1->2)
    arcs_content = "0 1\n1 2\n"
    (d / f"{trip_id}.arcs").write_text(arcs_content)

    # Route: arc_ids (2 arcs: 0, 1)
    route_content = "0\n1\n"
    (d / f"{trip_id}.route").write_text(route_content)

    return d, trip_id


def test_load_track_valid(temp_trip_dir):
    d, trip_id = temp_trip_dir
    df = load_track(d / f"{trip_id}.track")
    assert len(df) == 3
    assert list(df.columns) == ["lon", "lat", "time", "observation_index"]
    assert df["observation_index"].tolist() == [0, 1, 2]
    assert df["time"].tolist() == [1.0, 2.0, 3.0]


def test_load_nodes_valid(temp_trip_dir):
    d, trip_id = temp_trip_dir
    df = load_nodes(d / f"{trip_id}.nodes")
    assert len(df) == 3
    assert list(df.columns) == ["node_id", "lon", "lat"]
    assert df["node_id"].tolist() == [0, 1, 2]


def test_load_arcs_valid(temp_trip_dir):
    d, trip_id = temp_trip_dir
    df = load_arcs(d / f"{trip_id}.arcs")
    assert len(df) == 2
    assert list(df.columns) == ["arc_id", "from_node", "to_node"]
    assert df["arc_id"].tolist() == [0, 1]
    assert df["from_node"].tolist() == [0, 1]
    assert df["to_node"].tolist() == [1, 2]


def test_load_route_valid(temp_trip_dir):
    d, trip_id = temp_trip_dir
    df = load_route(d / f"{trip_id}.route")
    assert len(df) == 2
    assert list(df.columns) == ["route_order", "arc_id"]
    assert df["arc_id"].tolist() == [0, 1]


def test_load_trip_complete(temp_trip_dir):
    d, trip_id = temp_trip_dir
    trip = load_trip(d, partition="train")
    assert isinstance(trip, TripData)
    assert trip.num_observations == 3
    assert trip.num_nodes == 3
    assert trip.num_arcs == 2
    assert trip.num_route_arcs == 2


def test_load_empty_file(tmp_path):
    empty_file = tmp_path / "empty.track"
    empty_file.write_text("")
    with pytest.raises(MalformedFileError):
        load_track(empty_file)


def test_load_malformed_columns(tmp_path):
    bad_file = tmp_path / "bad.track"
    bad_file.write_text("8.580 49.850\n")  # Only 2 columns instead of 3
    with pytest.raises(MalformedFileError):
        load_track(bad_file)


def test_invalid_endpoints_validation(tmp_path):
    nodes_df = pd.DataFrame({"node_id": [0, 1], "lon": [8.5, 8.6], "lat": [49.8, 49.9]})
    arcs_df = pd.DataFrame({"arc_id": [0], "from_node": [0], "to_node": [99]})  # Node 99 does not exist!
    with pytest.raises(ValidationError):
        validate_trip_endpoints(nodes_df, arcs_df)


def test_invalid_route_arc_validation(tmp_path):
    arcs_df = pd.DataFrame({"arc_id": [0, 1], "from_node": [0, 1], "to_node": [1, 2]})
    route_df = pd.DataFrame({"route_order": [0], "arc_id": [55]})  # Arc 55 does not exist!
    with pytest.raises(ValidationError):
        validate_route_arc_ids(arcs_df, route_df)


def test_clean_and_diagnose_trip(temp_trip_dir):
    d, trip_id = temp_trip_dir
    trip = load_trip(d)
    diagnostics = clean_and_diagnose_trip(trip)
    assert diagnostics["trip_id"] == trip_id
    assert diagnostics["track_diagnostics"]["num_observations"] == 3
    assert diagnostics["track_diagnostics"]["num_invalid_coords"] == 0
    assert diagnostics["road_diagnostics"]["num_nodes"] == 3
    assert diagnostics["route_diagnostics"]["is_continuous"] is True

