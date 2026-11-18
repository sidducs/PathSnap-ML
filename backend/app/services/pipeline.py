"""End-to-end prediction pipeline:  trip id -> candidates -> scores -> (HMM/Viterbi) -> route -> metrics.

Leakage design:
  * prediction stages receive only `TripInput`/`Observations`/`CandidateSet` (no labels exist in them);
  * reference labels are fetched from `ReferenceStore` AFTER the final prediction is fixed, and are used
    only to compute metrics. If a trip has no .route file the pipeline still predicts (metrics = None).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Settings
from ..core.candidates import CandidateSet, Observations, build_candidates, make_observations
from ..core.evaluation import candidate_metrics, route_continuity, sequence_metrics
from ..core.hmm import HMMConfig, build_route, decode_route
from ..core.network import RoadNetwork
from ..core.reference import ReferenceLabels, ReferenceStore
from ..core.trip_io import TripInput, list_trip_ids, load_trip_input
from .registry import ModelRegistry

log = logging.getLogger("pathsnap.pipeline")


class TripStore:
    """Caches parsed trips + road networks (building the spatial index of a 100k-arc network takes ~1 s)."""

    def __init__(self, set_dir: Path, max_cached: int = 12):
        self.set_dir = Path(set_dir)
        self.ids = list_trip_ids(self.set_dir)
        self._cache: OrderedDict[str, tuple[TripInput, RoadNetwork, Observations]] = OrderedDict()
        self._max = max_cached

    def __contains__(self, trip_id: str) -> bool:
        return trip_id in self.ids

    def get(self, trip_id: str, radius_hint: float | None = None):
        if trip_id not in self.ids:
            raise KeyError(trip_id)
        if trip_id in self._cache:
            self._cache.move_to_end(trip_id)
            return self._cache[trip_id]
        trip = load_trip_input(self.set_dir, trip_id)
        net = RoadNetwork.from_trip(trip)
        _ = net.tree  # build spatial index now
        obs = make_observations(trip, net)
        self._cache[trip_id] = (trip, net, obs)
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)
        return self._cache[trip_id]


@dataclass
class PipelineRun:
    trip: TripInput
    net: RoadNetwork
    obs: Observations
    cands: CandidateSet
    scores: np.ndarray
    matcher_name: str
    used_hmm: bool
    clf_arcs: np.ndarray
    hmm_arcs: np.ndarray | None
    final_arcs: np.ndarray
    route_raw: list[int]
    route_filled: list[int]
    reference: ReferenceLabels | None
    metrics: dict | None
    timing_ms: dict = field(default_factory=dict)


class PathSnapPipeline:
    def __init__(self, settings: Settings, store: TripStore, registry: ModelRegistry, reference: ReferenceStore):
        self.s, self.store, self.registry, self.reference = settings, store, registry, reference
        self._lock = threading.Lock()  # HMM/BFS caches are not thread-safe; evaluation is one request at a time

    def run(self, trip_id: str, use_hmm: bool = True, matcher: str | None = None,
            hmm_cfg: HMMConfig | None = None) -> PipelineRun:
        with self._lock:
            return self._run(trip_id, use_hmm, matcher, hmm_cfg)

    def _run(self, trip_id, use_hmm, matcher_name, hmm_cfg) -> PipelineRun:
        T = {}
        t0 = time.perf_counter()
        trip, net, obs = self.store.get(trip_id)
        T["load"] = time.perf_counter() - t0

        m = self.registry.get(matcher_name)
        cfg = hmm_cfg or self.registry.hmm_cfg

        t = time.perf_counter()
        cands = build_candidates(obs, net, self.s.candidate_radius_m, self.s.max_candidates)
        T["candidates"] = time.perf_counter() - t

        t = time.perf_counter()
        scores = m.score(cands)
        T["scoring"] = time.perf_counter() - t
        # classifier-only: highest score per observation
        clf_arcs = np.empty(cands.n_obs, dtype=np.int64)
        for i in range(cands.n_obs):
            sl = cands.span(i)
            clf_arcs[i] = cands.arc_id[sl][np.argmax(scores[sl])]

        hmm_arcs = None
        t = time.perf_counter()
        if use_hmm:
            hmm_arcs = decode_route(cands, scores, obs.t, net, cfg)
        T["hmm_viterbi"] = time.perf_counter() - t
        final = hmm_arcs if use_hmm else clf_arcs

        route_raw, route_filled = build_route(final, net)
        T["inference_total"] = time.perf_counter() - t0

        # ---- evaluation (labels enter ONLY here, after prediction is final) ----
        t = time.perf_counter()
        ref = self.reference.get(trip_id, obs, net)
        metrics = None
        if ref is not None:
            raw_clf, _ = build_route(clf_arcs, net)
            metrics = {
                "classifier_only": sequence_metrics(clf_arcs, ref.obs_arc, raw_clf, ref.route, net),
                "hmm": sequence_metrics(hmm_arcs, ref.obs_arc, build_route(hmm_arcs, net)[0], ref.route, net)
                if hmm_arcs is not None else None,
                # continuity of the gap-filled route that is actually returned/drawn
                "returned_route_continuity": route_continuity(route_filled, net)[0],
                "candidates": candidate_metrics(cands, ref.obs_arc, scores if m.has_probabilities else None),
                "reference_quality": {
                    "mean_align_dist_m": float(ref.align_dist_m.mean()),
                    "share_align_dist_gt_30m": float((ref.align_dist_m > 30).mean()),
                    "reference_route_connectivity": ref.route_connectivity,
                },
            }
        T["evaluation"] = time.perf_counter() - t
        timing_ms = {k: round(v * 1000, 1) for k, v in T.items()}
        return PipelineRun(trip, net, obs, cands, scores, m.name, use_hmm, clf_arcs, hmm_arcs, final,
                           route_raw, route_filled, ref, metrics, timing_ms)
