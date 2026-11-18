"""Candidate scorers: turn a CandidateSet into one score per (observation, candidate) pair.

NearestRoadMatcher  - baseline (Required model #1): score decreases with distance, argmax = nearest road.
ClassifierMatcher   - wraps a trained sklearn-style classifier loaded from ml/artifacts/.

Model bundle contract (joblib.dump of a dict) - what ML members must export:
    {
      "model":          fitted estimator with predict_proba(X) -> (P, 2)   (class 1 = "true candidate"),
                        OR a sklearn Pipeline (scaler + classifier) with the same interface,
      "feature_names":  list[str]  - exactly the columns the model was trained on (subset/ordered
                        selection of core.candidates.FEATURE_NAMES),
      "name":           str, e.g. "xgboost_tuned",
      "meta":           dict (optional: cv scores, params, training date, seed ...)
    }
A bare estimator (no dict) is accepted too and is assumed to use all FEATURE_NAMES in order.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .candidates import FEATURE_NAMES, CandidateSet


class Matcher(Protocol):
    name: str
    has_probabilities: bool

    def score(self, cands: CandidateSet) -> np.ndarray: ...


@dataclass
class NearestRoadMatcher:
    sigma_m: float = 15.0
    name: str = "nearest_road"
    has_probabilities: bool = False

    def score(self, cands: CandidateSet) -> np.ndarray:
        return np.exp(-0.5 * (cands.dist / self.sigma_m) ** 2) + 1e-12


class ClassifierMatcher:
    has_probabilities = True

    def __init__(self, model, feature_names: list[str], name: str, meta: dict | None = None):
        bad = [f for f in feature_names if f not in FEATURE_NAMES]
        if bad:
            raise ValueError(f"model expects features unknown to the backend: {bad}. "
                             f"Backend provides: {list(FEATURE_NAMES)}")
        self.model = model
        self.feature_names = list(feature_names)
        self._cols = [FEATURE_NAMES.index(f) for f in self.feature_names]
        self.name = name
        self.meta = meta or {}

    def score(self, cands: CandidateSet) -> np.ndarray:
        proba = self.model.predict_proba(cands.X[:, self._cols])
        return np.asarray(proba)[:, 1]

    @classmethod
    def load(cls, path: Path) -> "ClassifierMatcher":
        import joblib

        bundle = joblib.load(path)
        if isinstance(bundle, dict):
            model = bundle["model"]
            names = list(bundle.get("feature_names", FEATURE_NAMES))
            name = bundle.get("name", Path(path).stem)
            meta = bundle.get("meta", {})
        else:
            model, names, name, meta = bundle, list(FEATURE_NAMES), Path(path).stem, {}
        if not hasattr(model, "predict_proba"):
            raise ValueError("model must implement predict_proba (needed for HMM emissions and log loss)")
        n_in = getattr(model, "n_features_in_", None)
        if n_in is not None and n_in != len(names):
            raise ValueError(f"model was fitted on {n_in} features but bundle lists {len(names)}")
        return cls(model, names, name, meta)
