"""Generate an interactive Folium map for one trip using the real pipeline.

    python -m backend.scripts.make_folium_map 00000052 --set student_test
    -> results/maps/trip_00000052.html

Use student_test / train only. (The evaluator partition must stay untouched during development.)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..app.config import REPO_ROOT, get_settings
from ..app.core.reference import ReferenceStore
from ..app.services.pipeline import PathSnapPipeline, TripStore
from ..app.services.registry import ModelRegistry
from ..app.viz.folium_map import build_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trip_id")
    ap.add_argument("--set", default="student_test", choices=["train", "student_test"])
    ap.add_argument("--no-hmm", action="store_true")
    ap.add_argument("--matcher", default=None, help="'nearest_road' for the baseline")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    logging.basicConfig(level="INFO")
    s = get_settings()
    set_dir = s.data_root / a.set
    pipe = PathSnapPipeline(s, TripStore(set_dir), ModelRegistry.load(s), ReferenceStore(set_dir))
    run = pipe.run(a.trip_id, use_hmm=not a.no_hmm, matcher=a.matcher)
    out = Path(a.out) if a.out else REPO_ROOT / "results" / "maps" / f"trip_{a.trip_id}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_map(run).save(str(out))
    print("saved", out)
    if run.metrics:
        print({k: round(v, 3) for k, v in (run.metrics["hmm"] or run.metrics["classifier_only"]).items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
