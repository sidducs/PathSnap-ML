"""Unit tests for projection and directed road network graph."""

import pytest
import numpy as np
import pandas as pd

from ml.preprocessing.projection import (
    LocalMercatorProjection,
    haversine_distance,
    calculate_bearing,
    calculate_heading_difference,
)
from ml.preprocessing.graph import DirectedRoadGraph


def test_haversine_known_distance():
    # 1 degree latitude is approximately 111,195 meters
    dist = haversine_distance(0.0, 0.0, 0.0, 1.0)
    assert 111000.0 < dist < 111500.0


def test_haversine_zero_coordinate_valid():
    # Lon=0, Lat=0 is perfectly valid Prime Meridian & Equator
    dist = haversine_distance(0.0, 0.0, 0.0, 0.0)
    assert dist == 0.0


def test_bearing_cardinal_directions():
    # North
    assert calculate_bearing(0.0, 0.0, 0.0, 1.0) == pytest.approx(0.0, abs=1e-2)
    # East
    assert calculate_bearing(0.0, 0.0, 1.0, 0.0) == pytest.approx(90.0, abs=1e-2)
    # South
    assert calculate_bearing(0.0, 1.0, 0.0, 0.0) == pytest.approx(180.0, abs=1e-2)
    # West
    assert calculate_bearing(1.0, 0.0, 0.0, 0.0) == pytest.approx(270.0, abs=1e-2)


def test_heading_difference():
    assert calculate_heading_difference(10.0, 20.0) == pytest.approx(10.0)
    assert calculate_heading_difference(350.0, 10.0) == pytest.approx(20.0)
    assert calculate_heading_difference(0.0, 180.0) == pytest.approx(180.0)


def test_projection_roundtrip():
    proj = LocalMercatorProjection(lon_0=8.5, lat_0=49.8)
    orig_lon, orig_lat = 8.55, 49.85
    x, y = proj.forward(orig_lon, orig_lat)
    rec_lon, rec_lat = proj.inverse(x, y)
    assert rec_lon == pytest.approx(orig_lon, abs=1e-6)
    assert rec_lat == pytest.approx(orig_lat, abs=1e-6)


def test_directed_graph_properties():
    # Triangle: 0 -> 1 -> 2 -> 0
    nodes_df = pd.DataFrame({
        "node_id": [0, 1, 2],
        "lon": [8.50, 8.51, 8.50],
        "lat": [49.80, 49.80, 49.81],
    })
    arcs_df = pd.DataFrame({
        "arc_id": [0, 1, 2],
        "from_node": [0, 1, 2],
        "to_node": [1, 2, 0],
    })

    graph = DirectedRoadGraph(nodes_df, arcs_df)
    assert graph.num_nodes == 3
    assert graph.num_arcs == 3

    # Check directedness: 0 -> 1 exists, but 1 -> 0 does not
    assert 0 in graph.out_arcs[0]
    assert 1 not in graph.out_arcs[0]

    # Lengths must be positive
    assert np.all(graph.arc_lengths > 0)

    # Shortest path in directed graph
    # Path 0 -> 1 -> 2
    d_0_to_2 = graph.shortest_path_distance(0, 2)
    assert d_0_to_2 < float("inf")
    # Path 2 -> 1 requires going 2 -> 0 -> 1
    d_2_to_1 = graph.shortest_path_distance(2, 1)
    assert d_2_to_1 < float("inf")

    # Arc transition distance
    # Arc 0 (0->1) to Arc 1 (1->2): to_node(0) == from_node(1) -> 0 distance
    assert graph.arc_transition_distance(0, 1) == 0.0
    # Arc 1 (1->2) to Arc 0 (0->1): to_node(1)=2 to from_node(0)=0 via Arc 2
    d_trans = graph.arc_transition_distance(1, 0)
    assert d_trans == pytest.approx(graph.arc_lengths[2], rel=1e-3)

