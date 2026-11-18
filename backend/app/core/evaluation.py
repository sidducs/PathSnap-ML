"""Observation-level and route-level metrics (computed server-side against reference labels)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from .candidates import CandidateSet
from .network import RoadNetwork

EPS = 1e-15


def route_continuity(route_raw: list[int], net: RoadNetwork) -> tuple[float, int]:
    """Share of consecutive route arcs that are physically adjacent (arc_to[a] == arc_from[b]), and #breaks."""
    if len(route_raw) < 2:
        return 1.0, 0
    a = np.asarray(route_raw[:-1])
    b = np.asarray(route_raw[1:])
    ok = net.arc_to[a] == net.arc_from[b]
    return float(ok.mean()), int((~ok).sum())


def candidate_metrics(cands: CandidateSet, obs_arc: np.ndarray, scores: np.ndarray | None) -> dict:
    present = cands.contains(obs_arc)
    out = {"candidate_miss_rate": float(1.0 - present.mean()),
           "mean_candidates_per_obs": float(len(cands.arc_id) / cands.n_obs),
           "log_loss_pairwise": None, "log_loss_ranking": None}
    if scores is None:
        return out
    y = (cands.arc_id == obs_arc[cands.obs_idx]).astype(float)
    p = np.clip(scores, EPS, 1 - EPS)
    out["log_loss_pairwise"] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    # ranking log loss: -log of the normalised posterior of the true arc, over observations where it is a candidate
    losses = []
    for i in np.flatnonzero(present):
        sl = cands.span(i)
        s = np.clip(scores[sl], EPS, None)
        losses.append(-np.log(s[cands.arc_id[sl] == obs_arc[i]][0] / s.sum()))
    out["log_loss_ranking"] = float(np.mean(losses)) if losses else None
    return out


def sequence_metrics(pred_arcs: np.ndarray, obs_arc: np.ndarray, route_raw: list[int],
                     ref_route: np.ndarray, net: RoadNetwork) -> dict:
    correct = pred_arcs == obs_arc
    twin = net.reverse_of[obs_arc]
    labels = np.union1d(obs_arc, pred_arcs)
    kw = dict(labels=labels, average="macro", zero_division=0)
    cont, breaks = route_continuity(route_raw, net)
    pred_set, ref_set = set(route_raw), set(int(a) for a in ref_route)
    inter = len(pred_set & ref_set)
    rp = inter / len(pred_set) if pred_set else 0.0
    rr = inter / len(ref_set) if ref_set else 0.0
    return {
        "top1_accuracy": float(correct.mean()),
        "top1_accuracy_direction_agnostic": float((correct | (pred_arcs == twin)).mean()),
        "macro_f1": float(f1_score(obs_arc, pred_arcs, **kw)),
        "macro_precision": float(precision_score(obs_arc, pred_arcs, **kw)),
        "macro_recall": float(recall_score(obs_arc, pred_arcs, **kw)),
        "route_continuity": cont,
        "route_breaks": breaks,
        "route_arc_precision": rp,
        "route_arc_recall": rr,
        "route_arc_f1": (2 * rp * rr / (rp + rr)) if (rp + rr) else 0.0,
        "n_route_arcs_pred": len(pred_set),
        "n_route_arcs_ref": len(ref_set),
    }
