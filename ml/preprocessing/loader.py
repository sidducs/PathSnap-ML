"""Robust loaders for PathSnap raw dataset files (.track, .nodes, .arcs, .route).

Requirements satisfied:
1. Correctly parse whitespace-separated files (handles tabs, multiple spaces).
2. Handle empty or malformed files with descriptive exceptions.
3. Validate column counts and numeric types.
4. Preserve original row indices (0-based node IDs and arc IDs).
5. Preserve GPS observation order and timestamps.
6. Validate arc endpoints against node indices.
7. Validate route arc IDs against available arcs.
8. Maintain a consistent internal representation via TripData dataclass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class LoaderError(Exception):
    """Base exception for loader failures."""
    pass


class MalformedFileError(LoaderError):
    """Raised when a file does not conform to expected format or column count."""
    pass


class ValidationError(LoaderError):
    """Raised when data values violate domain constraints (e.g. invalid endpoint IDs)."""
    pass


@dataclass
class TripData:
    """Standardized in-memory container for a complete trip."""
    trip_id: str
    partition: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: Optional[pd.DataFrame] = None
    metadata: Dict[str, Union[int, float, str, bool]] = field(default_factory=dict)

    @property
    def num_observations(self) -> int:
        return len(self.track)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_arcs(self) -> int:
        return len(self.arcs)

    @property
    def num_route_arcs(self) -> int:
        return len(self.route) if self.route is not None else 0


def load_track(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load a .track file.

    Format: three whitespace-separated columns without header:
        longitude (float), latitude (float), timestamp (float).

    Returns:
        pd.DataFrame with columns ['lon', 'lat', 'time', 'observation_index'].
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Track file not found: {path}")

    if path.stat().st_size == 0:
        raise MalformedFileError(f"Track file is empty: {path}")

    try:
        # Whitespace delimiter matches any consecutive whitespace (spaces, tabs)
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            engine="python",
            dtype={0: float, 1: float, 2: float},
        )
    except Exception as e:
        raise MalformedFileError(f"Failed to parse track file {path}: {e}") from e

    if df.shape[1] != 3:
        raise MalformedFileError(
            f"Track file {path} must have exactly 3 columns (lon, lat, time), found {df.shape[1]}"
        )

    # Check for NaNs or Infs
    if df.isna().any().any():
        raise MalformedFileError(f"Track file {path} contains NaN values")
    if np.isinf(df.values).any():
        raise MalformedFileError(f"Track file {path} contains Infinite values")

    df.columns = ["lon", "lat", "time"]
    df["observation_index"] = np.arange(len(df), dtype=int)
    return df


def load_nodes(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load a .nodes file.

    Format: two whitespace-separated columns without header:
        longitude (float), latitude (float).
    The row index (0-based) represents the node ID.

    Returns:
        pd.DataFrame with columns ['node_id', 'lon', 'lat'].
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Nodes file not found: {path}")

    if path.stat().st_size == 0:
        raise MalformedFileError(f"Nodes file is empty: {path}")

    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            engine="python",
            dtype={0: float, 1: float},
        )
    except Exception as e:
        raise MalformedFileError(f"Failed to parse nodes file {path}: {e}") from e

    if df.shape[1] != 2:
        raise MalformedFileError(
            f"Nodes file {path} must have exactly 2 columns (lon, lat), found {df.shape[1]}"
        )

    if df.isna().any().any():
        raise MalformedFileError(f"Nodes file {path} contains NaN values")
    if np.isinf(df.values).any():
        raise MalformedFileError(f"Nodes file {path} contains Infinite values")

    df.columns = ["lon", "lat"]
    df["node_id"] = np.arange(len(df), dtype=int)
    # Reorder columns
    df = df[["node_id", "lon", "lat"]]
    return df


def load_arcs(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load an .arcs file.

    Format: two integer columns without header:
        from_node_id (int), to_node_id (int).
    The row index (0-based) represents the arc ID.

    Returns:
        pd.DataFrame with columns ['arc_id', 'from_node', 'to_node'].
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Arcs file not found: {path}")

    if path.stat().st_size == 0:
        raise MalformedFileError(f"Arcs file is empty: {path}")

    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            engine="python",
            dtype={0: int, 1: int},
        )
    except Exception as e:
        raise MalformedFileError(f"Failed to parse arcs file {path}: {e}") from e

    if df.shape[1] != 2:
        raise MalformedFileError(
            f"Arcs file {path} must have exactly 2 columns (from_node, to_node), found {df.shape[1]}"
        )

    if df.isna().any().any():
        raise MalformedFileError(f"Arcs file {path} contains NaN values")

    df.columns = ["from_node", "to_node"]
    df["arc_id"] = np.arange(len(df), dtype=int)
    df = df[["arc_id", "from_node", "to_node"]]
    return df


def load_route(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load a .route file.

    Format: one integer column without header:
        chronological sequence of reference arc IDs.

    Returns:
        pd.DataFrame with columns ['route_order', 'arc_id'].
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Route file not found: {path}")

    if path.stat().st_size == 0:
        raise MalformedFileError(f"Route file is empty: {path}")

    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            engine="python",
            dtype={0: int},
        )
    except Exception as e:
        raise MalformedFileError(f"Failed to parse route file {path}: {e}") from e

    if df.shape[1] != 1:
        raise MalformedFileError(
            f"Route file {path} must have exactly 1 column (arc_id), found {df.shape[1]}"
        )

    if df.isna().any().any():
        raise MalformedFileError(f"Route file {path} contains NaN values")

    df.columns = ["arc_id"]
    df["route_order"] = np.arange(len(df), dtype=int)
    df = df[["route_order", "arc_id"]]
    return df


def validate_trip_endpoints(nodes_df: pd.DataFrame, arcs_df: pd.DataFrame) -> None:
    """Validate that all arc endpoints refer to valid node indices."""
    num_nodes = len(nodes_df)
    min_from, max_from = arcs_df["from_node"].min(), arcs_df["from_node"].max()
    min_to, max_to = arcs_df["to_node"].min(), arcs_df["to_node"].max()

    if min_from < 0 or max_from >= num_nodes:
        invalid_from = arcs_df[(arcs_df["from_node"] < 0) | (arcs_df["from_node"] >= num_nodes)]
        raise ValidationError(
            f"Arc endpoints exceed valid node range [0, {num_nodes-1}]. "
            f"Found {len(invalid_from)} invalid 'from_node' entries."
        )

    if min_to < 0 or max_to >= num_nodes:
        invalid_to = arcs_df[(arcs_df["to_node"] < 0) | (arcs_df["to_node"] >= num_nodes)]
        raise ValidationError(
            f"Arc endpoints exceed valid node range [0, {num_nodes-1}]. "
            f"Found {len(invalid_to)} invalid 'to_node' entries."
        )


def validate_route_arc_ids(arcs_df: pd.DataFrame, route_df: pd.DataFrame) -> None:
    """Validate that all route arc IDs exist in the arcs DataFrame."""
    num_arcs = len(arcs_df)
    min_arc, max_arc = route_df["arc_id"].min(), route_df["arc_id"].max()

    if min_arc < 0 or max_arc >= num_arcs:
        invalid_arcs = route_df[(route_df["arc_id"] < 0) | (route_df["arc_id"] >= num_arcs)]
        raise ValidationError(
            f"Route arc IDs exceed valid arc range [0, {num_arcs-1}]. "
            f"Found {len(invalid_arcs)} invalid arc IDs in route."
        )


def load_trip(
    trip_dir: Union[str, Path],
    trip_id: Optional[str] = None,
    partition: str = "train",
    require_route: bool = False,
    validate: bool = True,
) -> TripData:
    """Load all files for a trip directory and return a structured TripData object.

    Args:
        trip_dir: Path to directory containing trip files (e.g. data/raw/.../00000001).
        trip_id: Trip ID string. Inferred from directory name if None.
        partition: Partition name ('train', 'student_test', 'evaluator_test').
        require_route: If True, raises an error if .route file is missing.
        validate: If True, performs structural endpoint and route ID validation.
    """
    trip_path = Path(trip_dir)
    if not trip_path.is_dir():
        raise FileNotFoundError(f"Trip directory does not exist: {trip_path}")

    if trip_id is None:
        trip_id = trip_path.name

    track_path = trip_path / f"{trip_id}.track"
    nodes_path = trip_path / f"{trip_id}.nodes"
    arcs_path = trip_path / f"{trip_id}.arcs"
    route_path = trip_path / f"{trip_id}.route"

    track_df = load_track(track_path)
    nodes_df = load_nodes(nodes_path)
    arcs_df = load_arcs(arcs_path)

    route_df = None
    if route_path.is_file():
        route_df = load_route(route_path)
    elif require_route:
        raise FileNotFoundError(f"Route file is required but missing: {route_path}")

    if validate:
        validate_trip_endpoints(nodes_df, arcs_df)
        if route_df is not None:
            validate_route_arc_ids(arcs_df, route_df)

    metadata = {
        "trip_id": trip_id,
        "partition": partition,
        "track_file": str(track_path),
        "nodes_file": str(nodes_path),
        "arcs_file": str(arcs_path),
        "route_file": str(route_path) if route_df is not None else "",
        "track_bytes": track_path.stat().st_size,
        "nodes_bytes": nodes_path.stat().st_size,
        "arcs_bytes": arcs_path.stat().st_size,
        "route_bytes": route_path.stat().st_size if route_df is not None else 0,
        "num_track": len(track_df),
        "num_nodes": len(nodes_df),
        "num_arcs": len(arcs_df),
        "num_route": len(route_df) if route_df is not None else 0,
    }

    return TripData(
        trip_id=trip_id,
        partition=partition,
        track=track_df,
        nodes=nodes_df,
        arcs=arcs_df,
        route=route_df,
        metadata=metadata,
    )

