"""Candidate generation + per-candidate feature engineering.

Formulation: every GPS observation gets a small set of candidate road segments (arcs). The classifier
scores each (observation, candidate) pair -> P(candidate is the arc the vehicle is on).

!!  TRAIN / INFERENCE PARITY  !!
Training code (ML members) MUST call `build_candidates()` from this file to create its feature matrix.
`FEATURE_NAMES` is the contract: the saved model bundle stores the names it was trained on and the
backend refuses to load a model whose features do not match.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo import project_to_segments, wrap_angle
from .network import RoadNetwork
from .trip_io import TripInput

FEATURE_NAMES: tuple[str, ...] = (
    "dist_m",          # distance observation -> arc (m)
    "dist_rank",       # 0 = nearest candidate for this observation
    "dist_gap_m",      # dist_m - nearest candidate's dist_m
    "n_candidates",    # size of this observation's candidate set
    "frac_along",      # position of foot point along arc, 0..1
    "arc_len_m",
    "heading_cos",     # cos(angle between vehicle heading and arc bearing); 0 if heading unknown
    "heading_abs",     # |angle| in rad, 0..pi ; 0 if heading unknown
    "heading_valid",   # 1 if the vehicle is moving fast enough for a reliable heading
    "speed_mps",
    "dt_prev_s",       # seconds since previous observation (capped at 60)
    "out_degree_to",   # branching at the arc's end node
    "in_degree_from",  # merging at the arc's start node
    "has_reverse_twin",
)


@dataclass
class Observations:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    heading: np.ndarray        # rad, math convention
    heading_valid: np.ndarray  # bool
    speed: np.ndarray          # m/s
    dt_prev: np.ndarray        # s (0 for first obs)

    @property
    def n(self) -> int:
        return len(self.t)


def make_observations(trip: TripInput, net: RoadNetwork, window: int = 2,
                      min_move_m: float = 4.0, max_window_s: float = 10.0) -> Observations:
    """Project GPS to metres and derive heading/speed from a +-`window` central difference."""
    x, y = net.proj.to_xy(trip.lon, trip.lat)
    t = trip.t
    n = len(t)
    idx = np.arange(n)
    lo, hi = np.clip(idx - window, 0, n - 1), np.clip(idx + window, 0, n - 1)
    dx, dy, dt = x[hi] - x[lo], y[hi] - y[lo], t[hi] - t[lo]
    disp = np.hypot(dx, dy)
    ok = (dt > 0) & (dt <= max_window_s) & (disp >= min_move_m)
    heading = np.where(ok, np.arctan2(dy, dx), 0.0)
    speed = np.where((dt > 0) & (dt <= max_window_s), disp / np.where(dt > 0, dt, 1.0), 0.0)
    dt_prev = np.concatenate([[0.0], np.diff(t)])
    return Observations(x, y, t, heading, ok, speed, dt_prev)


@dataclass
class CandidateSet:
    n_obs: int
    obs_idx: np.ndarray    # (P,) observation index of each pair, grouped by observation
    arc_id: np.ndarray     # (P,)
    dist: np.ndarray       # (P,) metres
    frac: np.ndarray       # (P,)
    offsets: np.ndarray    # (n_obs+1,) CSR offsets: pairs of obs i are offsets[i]:offsets[i+1]
    X: np.ndarray          # (P, len(FEATURE_NAMES))
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def span(self, i: int) -> slice:
        return slice(self.offsets[i], self.offsets[i + 1])

    def contains(self, obs_arc: np.ndarray) -> np.ndarray:
        """bool per observation: is the given arc in that observation's candidate set?"""
        out = np.zeros(self.n_obs, dtype=bool)
        hit = self.arc_id == obs_arc[self.obs_idx]
        out[self.obs_idx[hit]] = True
        return out


def build_candidates(obs: Observations, net: RoadNetwork, radius_m: float = 75.0, max_candidates: int = 8,
                     expand: tuple[float, ...] = (2.0, 4.0, 8.0)) -> CandidateSet:
    """K nearest arcs within `radius_m`; the radius is widened for observations that find nothing, and
    as a last resort the single nearest arc is used - so EVERY observation has >= 1 candidate."""
    n = obs.n
    pi, ai = net.query_radius(obs.x, obs.y, radius_m)
    have = np.zeros(n, dtype=bool)
    have[pi] = True
    for mult in expand:
        missing = np.flatnonzero(~have)
        if len(missing) == 0:
            break
        p2, a2 = net.query_radius(obs.x[missing], obs.y[missing], radius_m * mult)
        pi = np.concatenate([pi, missing[p2]])
        ai = np.concatenate([ai, a2])
        have[missing[p2]] = True
    missing = np.flatnonzero(~have)
    if len(missing):
        nearest = net.nearest_arc(obs.x[missing], obs.y[missing])
        pi = np.concatenate([pi, missing])
        ai = np.concatenate([ai, nearest])

    dist, frac = project_to_segments(obs.x[pi], obs.y[pi], net.x0[ai], net.y0[ai], net.x1[ai], net.y1[ai])

    # sort by (observation, distance, arc id) and keep the K nearest per observation
    order = np.lexsort((ai, dist, pi))
    pi, ai, dist, frac = pi[order], ai[order], dist[order], frac[order]
    counts = np.bincount(pi, minlength=n)
    starts = np.concatenate([[0], np.cumsum(counts)])
    rank = np.arange(len(pi)) - starts[pi]
    keep = rank < max_candidates
    pi, ai, dist, frac, rank = pi[keep], ai[keep], dist[keep], frac[keep], rank[keep]
    counts = np.bincount(pi, minlength=n)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    # ---- features ----
    first = offsets[:-1][pi]
    dmin = dist[first]
    h_diff = np.abs(wrap_angle(obs.heading[pi] - net.arc_bearing[ai]))
    hv = obs.heading_valid[pi]
    X = np.column_stack([
        dist,
        rank,
        dist - dmin,
        counts[pi],
        frac,
        net.arc_len[ai],
        np.where(hv, np.cos(h_diff), 0.0),
        np.where(hv, h_diff, 0.0),
        hv.astype(float),
        obs.speed[pi],
        np.minimum(obs.dt_prev[pi], 60.0),
        net.out_degree[net.arc_to[ai]],
        net.in_degree[net.arc_from[ai]],
        (net.reverse_of[ai] >= 0).astype(float),
    ]).astype(np.float64)
    assert X.shape[1] == len(FEATURE_NAMES)
    return CandidateSet(n, pi, ai, dist, frac, offsets, X)
