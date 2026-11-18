"""HMM + Viterbi route decoding.

Hidden states   : at time t the state is "which candidate arc of observation t the vehicle is on".
                  State sets differ per time step (<= K candidates), and each state keeps its arc id,
                  so the candidate -> road-segment correspondence is preserved.
Emission        : log of the classifier's candidate posterior  p_i / sum_j p_j  (scaled by `emission_weight`).
Transition      : topological. From arc a at t-1 to arc b at t:
                    same arc                     -> 0
                    b reachable in h arc-hops    -> -hop_penalty * h          (h <= max_hops_eff)
                    b is the reverse twin of a   -> -uturn_penalty
                    otherwise (disconnected)     -> -unreachable_penalty      (finite => decoding never breaks)
                  max_hops_eff grows with the time gap (a vehicle covers more road in a longer interval).
Gaps / restarts : if dt > gap_break_s the chain is restarted (no transition across a long outage).
Recurrence      : delta_t(j) = emission_t(j) + max_i [ delta_{t-1}(i) + trans(i -> j) ],  with back-pointers.

`viterbi_decode` is generic (any emission/transition) so it is also used for reference-label alignment.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable

import numpy as np

from .candidates import CandidateSet
from .network import RoadNetwork


@dataclass
class HMMConfig:
    hop_penalty: float = 0.7
    uturn_penalty: float = 4.0
    unreachable_penalty: float = 12.0
    max_hops: int = 3
    extra_hops_per_sec: float = 1.0
    max_hops_cap: int = 8
    gap_break_s: float = 30.0
    emission_weight: float = 1.0
    min_prob: float = 1e-6

    @classmethod
    def load(cls, path: Path) -> "HMMConfig":
        raw = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"hmm_config has unknown keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)


def viterbi_decode(emission: list[np.ndarray],
                   transition: Callable[[int], np.ndarray],
                   restart: np.ndarray | None = None) -> np.ndarray:
    """Generic Viterbi over variable-size state sets.

    emission[t]   : (K_t,) log-emission scores
    transition(t) : (K_{t-1}, K_t) log-transition matrix, called for t >= 1 (not for restarts)
    restart[t]    : True -> restart the chain at t (no dependency on t-1)
    Returns the best local state index per step.
    """
    T = len(emission)
    if T == 0:
        return np.zeros(0, dtype=np.int64)
    restart = np.zeros(T, dtype=bool) if restart is None else restart
    back: list[np.ndarray | None] = [None] * T
    deltas: list[np.ndarray] = [emission[0].astype(float)]
    for t in range(1, T):
        if restart[t]:
            deltas.append(emission[t].astype(float))
            continue
        cand = deltas[-1][:, None] + transition(t)
        arg = cand.argmax(axis=0)
        back[t] = arg
        deltas.append(cand[arg, np.arange(cand.shape[1])] + emission[t])
    path = np.empty(T, dtype=np.int64)
    path[T - 1] = deltas[T - 1].argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = deltas[t - 1].argmax() if back[t] is None else back[t][path[t]]
    return path


def emission_from_scores(cands: CandidateSet, scores: np.ndarray, cfg: HMMConfig) -> list[np.ndarray]:
    out = []
    for i in range(cands.n_obs):
        s = np.maximum(scores[cands.span(i)], cfg.min_prob)
        out.append(cfg.emission_weight * np.log(s / s.sum()))
    return out


def decode_route(cands: CandidateSet, scores: np.ndarray, t: np.ndarray, net: RoadNetwork,
                 cfg: HMMConfig) -> np.ndarray:
    """Return the decoded arc id for every observation."""
    emission = emission_from_scores(cands, scores, cfg)
    arcs = [cands.arc_id[cands.span(i)] for i in range(cands.n_obs)]
    dt = np.concatenate([[0.0], np.diff(t)])
    restart = dt > cfg.gap_break_s

    def transition(step: int) -> np.ndarray:
        prev, cur = arcs[step - 1], arcs[step]
        h_eff = int(min(cfg.max_hops + cfg.extra_hops_per_sec * max(dt[step] - 1.0, 0.0), cfg.max_hops_cap))
        M = np.full((len(prev), len(cur)), -cfg.unreachable_penalty)
        for i, a in enumerate(prev):
            a = int(a)
            reach = net.hops_from(a)
            twin = int(net.reverse_of[a])
            for j, b in enumerate(cur):
                b = int(b)
                if b == a:
                    M[i, j] = 0.0
                    continue
                h = reach.get(b)
                if h is not None and h <= h_eff:
                    M[i, j] = -cfg.hop_penalty * h
                elif b == twin:
                    M[i, j] = -cfg.uturn_penalty
        return M

    local = viterbi_decode(emission, transition, restart)
    return np.array([arcs[i][local[i]] for i in range(cands.n_obs)], dtype=np.int64)


def build_route(pred_arcs: np.ndarray, net: RoadNetwork, fill_max_hops: int = 12) -> tuple[list[int], list[int]]:
    """Collapse per-observation arcs into an ordered route.

    Returns (route_raw, route_filled). `route_raw` = consecutive duplicates removed (this is what
    route-continuity is measured on). `route_filled` additionally inserts the shortest connecting arcs
    wherever two consecutive arcs are not adjacent but are reachable within `fill_max_hops`.
    """
    raw: list[int] = []
    for a in pred_arcs:
        a = int(a)
        if not raw or raw[-1] != a:
            raw.append(a)
    filled: list[int] = []
    for a in raw:
        if filled:
            bridge = net.shortest_arc_path(filled[-1], a, fill_max_hops)
            if bridge:
                filled.extend(bridge)
        filled.append(a)
    return raw, filled
