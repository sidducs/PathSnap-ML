"""Loads trained artifacts from ml/artifacts/ at startup.

Expected files (all produced by the ML members):
    model.joblib       classifier bundle (see core/matchers.py for the contract)
    hmm_config.json    tuned decoder parameters (keys of core.hmm.HMMConfig); optional, defaults otherwise

Without model.joblib the API still runs with the nearest-road baseline and says so in /health.
Set REQUIRE_MODEL=true (do this on evaluation day) to make the server refuse to start without a model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import Settings
from ..core.hmm import HMMConfig
from ..core.matchers import ClassifierMatcher, Matcher, NearestRoadMatcher

log = logging.getLogger("pathsnap.registry")


@dataclass
class ModelRegistry:
    baseline: Matcher
    model: ClassifierMatcher | None
    hmm_cfg: HMMConfig
    hmm_cfg_loaded: bool
    default_name: str = field(init=False)

    def __post_init__(self):
        self.default_name = self.model.name if self.model else self.baseline.name

    @property
    def default(self) -> Matcher:
        return self.model or self.baseline

    def get(self, which: str | None) -> Matcher:
        """which: None/'default' -> best available; 'nearest_road' -> baseline."""
        if which in (None, "", "default"):
            return self.default
        if which == self.baseline.name:
            return self.baseline
        if self.model and which == self.model.name:
            return self.model
        raise KeyError(which)

    @classmethod
    def load(cls, s: Settings) -> "ModelRegistry":
        baseline = NearestRoadMatcher()
        model = None
        model_path = s.artifacts_dir / "model.joblib"
        if s.matcher_mode != "nearest":
            if model_path.exists():
                model = ClassifierMatcher.load(model_path)
                log.info("loaded classifier '%s' from %s (features: %s)", model.name, model_path, model.feature_names)
            elif s.matcher_mode == "model" or s.require_model:
                raise RuntimeError(f"model required but {model_path} not found")
            else:
                log.warning("no model at %s -> serving the nearest-road BASELINE only", model_path)
        cfg_path = s.artifacts_dir / "hmm_config.json"
        if cfg_path.exists():
            cfg, loaded = HMMConfig.load(cfg_path), True
            log.info("loaded HMM config from %s", cfg_path)
        else:
            cfg, loaded = HMMConfig(), False
            log.warning("no hmm_config.json -> using default HMM parameters")
        return cls(baseline, model, cfg, loaded)
