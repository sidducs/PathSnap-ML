"""Reading raw trip files.

SEPARATION OF CONCERNS (data-leakage rule):
  * `load_trip_input`  reads .track / .nodes / .arcs ONLY. It never opens .route, and its return type
    (`TripInput`) has no field that could carry a label. Everything in the prediction pipeline is typed
    against `TripInput`.
  * `load_reference_route` is the only function that opens .route. It is imported solely by
    `core/reference.py` (server-side evaluation) and offline scripts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRIP_ID_RE = re.compile(r"^[0-9]{1,12}$")


@dataclass(frozen=True)
class TripInput:
    trip_id: str
    lon: np.ndarray        # (n,)  longitude, degrees
    lat: np.ndarray        # (n,)  latitude, degrees
    t: np.ndarray          # (n,)  seconds
    nodes_lonlat: np.ndarray  # (N, 2) lon, lat ; row index = node id
    arcs: np.ndarray       # (M, 2) from_node, to_node ; row index = arc id

    @property
    def n_obs(self) -> int:
        return len(self.t)


def _read(path: Path, dtype=float) -> np.ndarray:
    # Files are whitespace separated, headerless, CRLF in places -> let pandas' C parser handle it.
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype)
    return df.to_numpy()


def trip_files(set_dir: Path, trip_id: str) -> dict[str, Path]:
    if not TRIP_ID_RE.match(trip_id):
        raise ValueError(f"invalid trip id: {trip_id!r}")
    d = set_dir / trip_id
    return {ext: d / f"{trip_id}.{ext}" for ext in ("track", "nodes", "arcs", "route")}


def list_trip_ids(set_dir: Path) -> list[str]:
    if not set_dir.is_dir():
        return []
    return sorted(p.name for p in set_dir.iterdir() if p.is_dir() and TRIP_ID_RE.match(p.name))


def load_trip_input(set_dir: Path, trip_id: str) -> TripInput:
    f = trip_files(set_dir, trip_id)
    track = _read(f["track"])
    nodes = _read(f["nodes"])
    arcs = _read(f["arcs"], dtype=np.int64).astype(np.int64)
    if track.shape[1] != 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError(f"trip {trip_id}: unexpected column counts {track.shape}, {nodes.shape}, {arcs.shape}")
    lon, lat, t = track[:, 0], track[:, 1], track[:, 2]
    if not (np.isfinite(track).all() and (np.abs(lon) <= 180).all() and (np.abs(lat) <= 90).all()):
        raise ValueError(f"trip {trip_id}: invalid GPS values (NaN or out-of-range lon/lat)")
    if (np.diff(t) < 0).any():  # defensive: restore chronological order (stable)
        order = np.argsort(t, kind="stable")
        lon, lat, t = lon[order], lat[order], t[order]
    if arcs.min() < 0 or arcs.max() >= len(nodes):
        raise ValueError(f"trip {trip_id}: arc references a non-existent node")
    return TripInput(trip_id, lon, lat, t, nodes, arcs)


def load_reference_route(set_dir: Path, trip_id: str) -> np.ndarray | None:
    """Reference arc-id sequence. SERVER-SIDE EVALUATION / OFFLINE USE ONLY."""
    p = trip_files(set_dir, trip_id)["route"]
    if not p.exists():
        return None
    return _read(p, dtype=np.int64).astype(np.int64).ravel()
