"""PathSnap API.   Run from the repo root:   python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .core.reference import ReferenceStore
from .schemas import (HealthResponse, NetworkResponse, PredictRequest, PredictResponse, TripDetail, TripSummary)
from .services.pipeline import PathSnapPipeline, TripStore
from .services.registry import ModelRegistry

log = logging.getLogger("pathsnap.api")


def _summarise(store: TripStore, trip_id: str) -> TripSummary:
    trip, _, _ = store.get(trip_id)
    return TripSummary(
        trip_id=trip_id, n_observations=trip.n_obs, duration_s=float(trip.t[-1] - trip.t[0]),
        bbox=[float(trip.lat.min()), float(trip.lon.min()), float(trip.lat.max()), float(trip.lon.max())],
        center=[float(trip.lat.mean()), float(trip.lon.mean())])


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    set_dir = s.data_root / s.trip_set
    store = TripStore(set_dir)
    if not store.ids:
        raise RuntimeError(f"no trips found in {set_dir} - check DATA_ROOT / TRIP_SET")
    registry = ModelRegistry.load(s)
    app.state.settings, app.state.store, app.state.registry = s, store, registry
    app.state.pipeline = PathSnapPipeline(s, store, registry, ReferenceStore(set_dir))
    log.info("trip_set=%s (%d trips) | default matcher=%s | HMM cfg=%s", s.trip_set, len(store.ids),
             registry.default.name, registry.hmm_cfg.to_dict())
    if s.warmup:  # parse files + build spatial indexes once, so the first demo request is fast
        t0 = time.perf_counter()
        for tid in store.ids:
            store.get(tid)
        log.info("warm-up of %d trips took %.1fs", len(store.ids), time.perf_counter() - t0)
    yield


app = FastAPI(title="PathSnap API", version="1.0.0", lifespan=lifespan)

_s = get_settings()
app.add_middleware(CORSMiddleware, allow_origins=list(_s.cors_origins), allow_methods=["GET", "POST", "OPTIONS"],
                   allow_headers=["*"], allow_credentials=False)


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    client = request.client.host if request.client else "?"
    log.info("%s %s -> %d  client=%s  %.0f ms", request.method, request.url.path, response.status_code, client,
             (time.perf_counter() - t0) * 1000)
    return response


def _known(request: Request, trip_id: str) -> TripStore:
    store: TripStore = request.app.state.store
    if trip_id not in store:
        raise HTTPException(404, f"unknown trip_id '{trip_id}'")
    return store


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    st = request.app.state
    return HealthResponse(status="ok", trip_set=st.settings.trip_set, n_trips=len(st.store.ids),
                          matcher=st.registry.default.name, model_loaded=st.registry.model is not None,
                          hmm_config_loaded=st.registry.hmm_cfg_loaded,
                          expose_reference_route=st.settings.expose_reference_route)


@app.get("/trips", response_model=list[TripSummary])
def trips(request: Request):
    store: TripStore = request.app.state.store
    return [_summarise(store, tid) for tid in store.ids]


@app.get("/trips/{trip_id}", response_model=TripDetail)
def trip_detail(trip_id: str, request: Request):
    store = _known(request, trip_id)
    trip, _, _ = store.get(trip_id)
    return TripDetail(**_summarise(store, trip_id).model_dump(), lat=trip.lat.tolist(), lon=trip.lon.tolist(),
                      t=trip.t.tolist())


@app.get("/trips/{trip_id}/network", response_model=NetworkResponse)
def trip_network(trip_id: str, request: Request, buffer_m: float = 150.0, max_arcs: int = 30000):
    """Road arcs within `buffer_m` of the GPS track - for drawing the map without shipping a 100k-arc network."""
    store = _known(request, trip_id)
    _, net, obs = store.get(trip_id)
    _, ai = net.query_radius(obs.x, obs.y, min(max(buffer_m, 1.0), 1000.0))
    ids = np.unique(ai)
    truncated = len(ids) > max_arcs
    ids = ids[:max_arcs]
    return NetworkResponse(trip_id=trip_id, n_arcs=len(ids), truncated=truncated,
                           arcs=[net.arc_lonlat(int(a)) for a in ids], arc_ids=[int(a) for a in ids])


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    st = request.app.state
    _known(request, req.trip_id)
    client = request.client.host if request.client else "?"
    try:
        run = st.pipeline.run(req.trip_id, use_hmm=req.use_hmm, matcher=req.matcher)
    except KeyError:
        raise HTTPException(400, f"unknown matcher '{req.matcher}'")
    net = run.net

    summary = None
    if run.metrics:
        main = run.metrics["hmm"] if run.used_hmm else run.metrics["classifier_only"]
        clf = run.metrics["classifier_only"]
        summary = {
            "top1_accuracy": main["top1_accuracy"], "macro_f1": main["macro_f1"],
            "route_continuity": main["route_continuity"], "route_arc_f1": main["route_arc_f1"],
            "candidate_miss_rate": run.metrics["candidates"]["candidate_miss_rate"],
            "classifier_only_top1_accuracy": clf["top1_accuracy"],
            "hmm_gain_top1": (main["top1_accuracy"] - clf["top1_accuracy"]) if run.used_hmm else None,
        }
        log.info("PREDICT trip=%s client=%s matcher=%s hmm=%s | top1=%.3f macroF1=%.3f continuity=%.3f | %s",
                 req.trip_id, client, run.matcher_name, run.used_hmm, summary["top1_accuracy"],
                 summary["macro_f1"], summary["route_continuity"], run.timing_ms)
    else:
        log.info("PREDICT trip=%s client=%s matcher=%s hmm=%s | no reference server-side | %s",
                 req.trip_id, client, run.matcher_name, run.used_hmm, run.timing_ms)

    ref_geo = None
    if st.settings.expose_reference_route and run.reference is not None:
        ref_geo = [net.arc_lonlat(int(a)) for a in run.reference.route]

    return PredictResponse(
        trip_id=req.trip_id, matcher=run.matcher_name, used_hmm=run.used_hmm, n_observations=run.obs.n,
        predicted_arc_per_observation=run.final_arcs.tolist(), route_arc_ids=run.route_filled,
        route_geometry=[net.arc_lonlat(a) for a in run.route_filled],
        summary=summary, metrics=run.metrics, timing_ms=run.timing_ms, reference_route_geometry=ref_geo)
