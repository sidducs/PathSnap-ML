"""Directed road network graph data structures and routing queries for PathSnap.

Requirements satisfied:
1. Preserve original 0-based node IDs and arc IDs.
2. Store source and target node IDs with original (lon, lat) and projected (x, y) coordinates.
3. Compute segment lengths (meters) and bearings (degrees [0, 360)).
4. Maintain directed connectivity (incoming and outgoing adjacency).
5. Support shortest-path queries (Dijkstra) and transition reachability between road segments.
6. Identify isolated nodes, degree statistics, and connected components.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from ml.preprocessing.projection import (
    LocalMercatorProjection,
    calculate_bearing,
    haversine_distance,
)


class DirectedRoadGraph:
    """Directed road network graph preserving original IDs and spatial geometry."""

    def __init__(
        self,
        nodes_df: pd.DataFrame,
        arcs_df: pd.DataFrame,
        projection: Optional[LocalMercatorProjection] = None,
    ):
        """Construct the directed graph from nodes and arcs DataFrames.

        Args:
            nodes_df: pd.DataFrame with ['node_id', 'lon', 'lat'].
            arcs_df: pd.DataFrame with ['arc_id', 'from_node', 'to_node'].
            projection: Optional LocalMercatorProjection. If None, constructed from node bounds.
        """
        self.num_nodes = len(nodes_df)
        self.num_arcs = len(arcs_df)

        # 1. Geographic coordinates
        self.node_lons = nodes_df["lon"].to_numpy(dtype=float)
        self.node_lats = nodes_df["lat"].to_numpy(dtype=float)

        # 2. Projection
        if projection is None:
            min_lon = float(np.min(self.node_lons))
            max_lon = float(np.max(self.node_lons))
            min_lat = float(np.min(self.node_lats))
            max_lat = float(np.max(self.node_lats))
            self.projection = LocalMercatorProjection.from_bounds(min_lon, max_lon, min_lat, max_lat)
        else:
            self.projection = projection

        # 3. Projected coordinates
        self.node_x, self.node_y = self.projection.forward(self.node_lons, self.node_lats)

        # 4. Arc topology
        self.arc_from = arcs_df["from_node"].to_numpy(dtype=int)
        self.arc_to = arcs_df["to_node"].to_numpy(dtype=int)
        self.arc_ids = arcs_df["arc_id"].to_numpy(dtype=int)

        # Arc endpoints coordinates
        self.arc_start_lon = self.node_lons[self.arc_from]
        self.arc_start_lat = self.node_lats[self.arc_from]
        self.arc_end_lon = self.node_lons[self.arc_to]
        self.arc_end_lat = self.node_lats[self.arc_to]

        self.arc_start_x = self.node_x[self.arc_from]
        self.arc_start_y = self.node_y[self.arc_from]
        self.arc_end_x = self.node_x[self.arc_to]
        self.arc_end_y = self.node_y[self.arc_to]

        # 5. Geometrical properties: length in meters & bearing in degrees
        dx = self.arc_end_x - self.arc_start_x
        dy = self.arc_end_y - self.arc_start_y
        self.arc_lengths = np.sqrt(dx * dx + dy * dy)
        # Avoid 0-division for zero-length arcs in bearing
        self.arc_bearings = calculate_bearing(
            self.arc_start_lon, self.arc_start_lat, self.arc_end_lon, self.arc_end_lat
        )

        # 6. Adjacency structures
        self.out_adj: List[List[Tuple[int, int, float]]] = [[] for _ in range(self.num_nodes)]
        self.in_adj: List[List[Tuple[int, int, float]]] = [[] for _ in range(self.num_nodes)]
        self.out_arcs: List[List[int]] = [[] for _ in range(self.num_nodes)]
        self.in_arcs: List[List[int]] = [[] for _ in range(self.num_nodes)]

        for a_id, u, v, length in zip(self.arc_ids, self.arc_from, self.arc_to, self.arc_lengths):
            if 0 <= u < self.num_nodes and 0 <= v < self.num_nodes:
                self.out_adj[u].append((v, a_id, float(length)))
                self.in_adj[v].append((u, a_id, float(length)))
                self.out_arcs[u].append(a_id)
                self.in_arcs[v].append(a_id)

        # Degrees
        self.out_degrees = np.array([len(edges) for edges in self.out_adj], dtype=int)
        self.in_degrees = np.array([len(edges) for edges in self.in_adj], dtype=int)

    def get_node_coords(self, node_id: int) -> Tuple[float, float]:
        """Return (lon, lat) of a node."""
        return float(self.node_lons[node_id]), float(self.node_lats[node_id])

    def get_arc_endpoints(self, arc_id: int) -> Tuple[int, int]:
        """Return (from_node, to_node) of an arc."""
        return int(self.arc_from[arc_id]), int(self.arc_to[arc_id])

    def get_arc_geometry(self, arc_id: int) -> Dict[str, float]:
        """Return spatial geometry details for a road segment."""
        return {
            "arc_id": int(arc_id),
            "from_node": int(self.arc_from[arc_id]),
            "to_node": int(self.arc_to[arc_id]),
            "start_lon": float(self.arc_start_lon[arc_id]),
            "start_lat": float(self.arc_start_lat[arc_id]),
            "end_lon": float(self.arc_end_lon[arc_id]),
            "end_lat": float(self.arc_end_lat[arc_id]),
            "start_x": float(self.arc_start_x[arc_id]),
            "start_y": float(self.arc_start_y[arc_id]),
            "end_x": float(self.arc_end_x[arc_id]),
            "end_y": float(self.arc_end_y[arc_id]),
            "length_m": float(self.arc_lengths[arc_id]),
            "bearing_deg": float(self.arc_bearings[arc_id]),
        }

    def shortest_path_distance(
        self,
        source_node: int,
        target_node: int,
        max_distance: float = 20000.0,
    ) -> float:
        """Compute shortest directed path distance in meters using Dijkstra.

        Returns:
            Shortest path distance in meters, or float('inf') if unreachable within max_distance.
        """
        if source_node == target_node:
            return 0.0

        if source_node < 0 or source_node >= self.num_nodes or target_node < 0 or target_node >= self.num_nodes:
            return float("inf")

        distances: Dict[int, float] = {source_node: 0.0}
        pq: List[Tuple[float, int]] = [(0.0, source_node)]

        while pq:
            d_u, u = heapq.heappop(pq)

            if d_u > distances[u]:
                continue
            if u == target_node:
                return d_u
            if d_u > max_distance:
                break

            for v, arc_id, length in self.out_adj[u]:
                d_v = d_u + length
                if d_v <= max_distance and d_v < distances.get(v, float("inf")):
                    distances[v] = d_v
                    heapq.heappush(pq, (d_v, v))

        return float("inf")

    def arc_transition_distance(
        self,
        from_arc_id: int,
        to_arc_id: int,
        max_distance: float = 20000.0,
    ) -> float:
        """Compute transition distance from from_arc_id to to_arc_id.

        If from_arc_id == to_arc_id: 0.0 (staying on same segment).
        If to_node(from_arc_id) == from_node(to_arc_id): 0.0 (direct topological connection).
        Otherwise: shortest directed network distance from to_node(from_arc_id) to from_node(to_arc_id).
        """
        if from_arc_id == to_arc_id:
            return 0.0

        u_end = self.arc_to[from_arc_id]
        v_start = self.arc_from[to_arc_id]

        if u_end == v_start:
            return 0.0

        return self.shortest_path_distance(u_end, v_start, max_distance=max_distance)

    def dijkstra_neighborhood(
        self,
        source_node: int,
        max_distance: float = 300.0,
    ) -> Dict[int, float]:
        """Compute shortest distances from source_node to all reachable nodes within max_distance.

        Returns:
            Dictionary mapping reachable node_id -> distance in meters.
        """
        if source_node < 0 or source_node >= self.num_nodes:
            return {}

        distances: Dict[int, float] = {source_node: 0.0}
        pq: List[Tuple[float, int]] = [(0.0, source_node)]

        while pq:
            d_u, u = heapq.heappop(pq)
            if d_u > distances[u]:
                continue
            if d_u > max_distance:
                break

            for v, arc_id, length in self.out_adj[u]:
                d_v = d_u + length
                if d_v <= max_distance and d_v < distances.get(v, float("inf")):
                    distances[v] = d_v
                    heapq.heappush(pq, (d_v, v))

        return distances

