"""Server-side reference data (ground truth). Imported ONLY by evaluation / offline scripts.

The supplied .route file is a sequence of arc ids, NOT one label per GPS point. To score observation-level
metrics we align each GPS observation to an arc of the reference route with a monotone Viterbi pass
(the vehicle cannot move backwards along its own route). Alignment quality is reported so label noise is
visible rather than hidden (see Dataset Exploration, section 6: reference routes are not error-free).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely

from .candidates import Observations
from .geo import project_to_segments
from .hmm import viterbi_decode
from .network import RoadNetwork
from .trip_io import load_reference_route


@dataclass
class ReferenceLabels:
    route: np.ndarray         # reference arc-id sequence (as supplied)
    obs_arc: np.ndarray       # (n_obs,) reference arc per observation (aligned)
    align_dist_m: np.ndarray  # (n_obs,) distance from observation to its aligned arc
    route_connectivity: float  # share of consecutive reference arcs that are actually adjacent


def align_observations_to_route(obs: Observations, net: RoadNetwork, route: np.ndarray,
                                sigma_m: float = 12.0, radius_m: float = 150.0, k: int = 6,
                                step_penalty: float = 0.05, max_step_cost: int = 10,
                                backward_penalty: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    m = len(route)
    rx0, ry0, rx1, ry1 = net.x0[route], net.y0[route], net.x1[route], net.y1[route]
    tree = shapely.STRtree(shapely.linestrings(np.stack(
        [np.column_stack([rx0, ry0]), np.column_stack([rx1, ry1])], axis=1)))
    pts = shapely.points(obs.x, obs.y)
    pi, ji = tree.query(pts, predicate="dwithin", distance=radius_m)
    have = np.zeros(obs.n, dtype=bool)
    have[pi] = True
    miss = np.flatnonzero(~have)
    if len(miss):
        p2, j2 = tree.query_nearest(pts[miss], all_matches=False)
        pi = np.concatenate([pi, miss[p2]])
        ji = np.concatenate([ji, j2])
    d, _ = project_to_segments(obs.x[pi], obs.y[pi], rx0[ji], ry0[ji], rx1[ji], ry1[ji])
    order = np.lexsort((ji, d, pi))
    pi, ji, d = pi[order], ji[order], d[order]
    counts = np.bincount(pi, minlength=obs.n)
    starts = np.concatenate([[0], np.cumsum(counts)])
    keep = (np.arange(len(pi)) - starts[pi]) < k
    pi, ji, d = pi[keep], ji[keep], d[keep]
    offs = np.concatenate([[0], np.cumsum(np.bincount(pi, minlength=obs.n))])

    j_list = [ji[offs[i]:offs[i + 1]] for i in range(obs.n)]
    emission = [-0.5 * (d[offs[i]:offs[i + 1]] / sigma_m) ** 2 for i in range(obs.n)]

    def transition(t: int) -> np.ndarray:
        step = j_list[t][None, :] - j_list[t - 1][:, None]
        return np.where(step >= 0, -step_penalty * np.minimum(step, max_step_cost), -backward_penalty)

    local = viterbi_decode(emission, transition)
    j_best = np.array([j_list[i][local[i]] for i in range(obs.n)], dtype=np.int64)
    dist_best = np.array([d[offs[i] + local[i]] for i in range(obs.n)])
    return route[j_best], dist_best


class ReferenceStore:
    """Loads reference routes lazily and aligns them to observations. Lives server-side only."""

    def __init__(self, set_dir: Path):
        self.set_dir = Path(set_dir)
        self._cache: dict[str, ReferenceLabels | None] = {}

    def get(self, trip_id: str, obs: Observations, net: RoadNetwork) -> ReferenceLabels | None:
        if trip_id in self._cache:
            return self._cache[trip_id]
        route = load_reference_route(self.set_dir, trip_id)
        if route is None:
            self._cache[trip_id] = None
            return None
        obs_arc, dist = align_observations_to_route(obs, net, route)
        conn = float(np.mean(net.arc_to[route[:-1]] == net.arc_from[route[1:]])) if len(route) > 1 else 1.0
        lab = ReferenceLabels(route, obs_arc, dist, conn)
        self._cache[trip_id] = lab
        return lab
