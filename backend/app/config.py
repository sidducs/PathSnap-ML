"""Runtime configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def _path(env: str, default: str) -> Path:
    p = Path(os.getenv(env, default))
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _bool(env: str, default: bool) -> bool:
    return os.getenv(env, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- server ---
    host: str
    port: int
    cors_origins: tuple[str, ...]
    log_level: str
    # --- data ---
    data_root: Path        # folder that contains student_test/ and evaluator_test/
    trip_set: str          # which partition /trips serves
    # --- models ---
    artifacts_dir: Path
    matcher_mode: str      # auto | nearest | model
    require_model: bool    # fail at startup if no trained model is found
    # --- behaviour ---
    expose_reference_route: bool  # NEVER enable on evaluation day
    warmup: bool
    candidate_radius_m: float
    max_candidates: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    origins = tuple(o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip())
    return Settings(
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        cors_origins=origins,
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        data_root=_path("DATA_ROOT", "./data"),
        # Safe default = development partition. Set TRIP_SET=evaluator_test on evaluation day only.
        trip_set=os.getenv("TRIP_SET", "student_test"),
        artifacts_dir=_path("ARTIFACTS_DIR", "./ml/artifacts"),
        matcher_mode=os.getenv("MATCHER", "auto").lower(),
        require_model=_bool("REQUIRE_MODEL", False),
        expose_reference_route=_bool("EXPOSE_REFERENCE_ROUTE", False),
        warmup=_bool("WARMUP", True),
        candidate_radius_m=float(os.getenv("CANDIDATE_RADIUS_M", "75")),
        max_candidates=int(os.getenv("MAX_CANDIDATES", "8")),
    )
