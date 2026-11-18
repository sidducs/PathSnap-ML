# PathSnap Backend (FastAPI)

Owner: Member 3. Serves the full prediction pipeline: **trip id → candidates → classifier scores → HMM/Viterbi → route → server-side evaluation**.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env            # then edit DATA_ROOT, CORS_ORIGINS
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000     # from the repo root
python -m pytest backend/tests -q                                      # 13 tests, no hackathon data needed
```

`DATA_ROOT` must contain `student_test/` and `evaluator_test/` (each `<trip>/<trip>.{track,nodes,arcs,route}`). `data/` is git-ignored.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, trip set, which matcher is live, `model_loaded` |
| GET | `/trips` | the trips of `TRIP_SET` (id, #points, bbox, centre) — **no labels** |
| GET | `/trips/{id}` | GPS points of one trip |
| GET | `/trips/{id}/network?buffer_m=150` | road arcs near the track (for the map) |
| POST | `/predict` | `{"trip_id": "00000009", "use_hmm": true, "matcher": null}` → route + metrics |

`/predict` returns `predicted_arc_per_observation`, `route_arc_ids`, `route_geometry` (`[[lat,lon],[lat,lon]]` per arc),
`summary` (headline numbers for the UI), `metrics` (classifier-only vs HMM, candidate miss rate, log loss, continuity …) and `timing_ms`.
`matcher: "nearest_road"` runs the baseline, `use_hmm: false` gives classifier-only — handy for the report's ablation tables.

## What ML members must export to `ml/artifacts/`

1. **`model.joblib`** — `joblib.dump({"model": estimator_or_pipeline, "feature_names": [...], "name": "xgboost_tuned", "meta": {...}}, path)`.
   `estimator.predict_proba(X)[:, 1]` = P(candidate is the true arc). A sklearn `Pipeline` (scaler + model) is fine.
2. **`hmm_config.json`** — tuned decoder parameters, same keys as `backend/hmm_config.example.json`.

**Train/inference parity is critical:** build the training matrix with `backend.app.core.candidates.build_candidates()` and label it with
`backend.app.core.reference.align_observations_to_route()` (the supplied `.route` is a list of arcs, not one label per GPS point).
`feature_names` is checked at startup; a model trained on different features is rejected.

## Data-leakage guarantees (assessment policy §7)

* `load_trip_input()` never opens `.route`; `TripInput` has no label field; prediction stages cannot see labels.
* Reference labels are loaded in `core/reference.py` **after** the final prediction is fixed, only to compute metrics.
* `/predict` works with no `.route` file at all and returns identical predictions (tested).
* Reference geometry is not sent to the UI unless `EXPOSE_REFERENCE_ROUTE=true` (keep it false).
* No per-trip logic anywhere; trip ids are validated against the listed set (`^[0-9]{1,12}$`).

## Evaluation-day checklist (Laptop 1)

1. `TRIP_SET=evaluator_test`, `REQUIRE_MODEL=true`, `CORS_ORIGINS=http://<laptop-2-ip>:5173`.
2. Run uvicorn with `--host 0.0.0.0`; allow port 8000 in the OS firewall; check `http://<laptop-1-ip>:8000/health` from Laptop 2.
3. Confirm `/health` shows `model_loaded: true` and the right `matcher`.
4. Keep the terminal visible: each request prints `PREDICT trip=… client=<laptop-2 ip> … top1=… macroF1=…` (the log evidence evaluators ask for).
5. Click through **all ten** trips beforehand (the data of the evaluator partition may only be used for this final run).

## Modules

`core/trip_io` raw files · `core/geo` projection (local metres) · `core/network` arcs, adjacency, spatial index ·
`core/candidates` candidates + 14 features · `core/matchers` nearest-road baseline + classifier wrapper ·
`core/hmm` Viterbi + HMM decoder · `core/reference` label alignment (server-side) · `core/evaluation` metrics ·
`services/pipeline` orchestration · `services/registry` artifact loading · `viz/folium_map` + `scripts/make_folium_map.py` maps.

## Caveats

* Observation-level metrics use labels derived by aligning each GPS point to the reference route (monotone Viterbi). The report must say so;
  `metrics.reference_quality` reports the alignment distance. A few reference routes have small connectivity breaks.
* The HMM here is a working reference implementation (topological transitions); Member 2 owns its tuning (`hmm_config.json`).
