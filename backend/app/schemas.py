from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    trip_set: str
    n_trips: int
    matcher: str = Field(description="classifier that /predict uses by default")
    model_loaded: bool
    hmm_config_loaded: bool
    expose_reference_route: bool


class TripSummary(BaseModel):
    trip_id: str
    n_observations: int
    duration_s: float
    bbox: list[float] = Field(description="[min_lat, min_lon, max_lat, max_lon]")
    center: list[float] = Field(description="[lat, lon]")


class TripDetail(TripSummary):
    lat: list[float]
    lon: list[float]
    t: list[float]


class NetworkResponse(BaseModel):
    trip_id: str
    n_arcs: int
    truncated: bool
    arcs: list[list[list[float]]] = Field(description="each arc = [[lat, lon], [lat, lon]]")
    arc_ids: list[int]


class PredictRequest(BaseModel):
    trip_id: str = Field(pattern=r"^[0-9]{1,12}$")
    use_hmm: bool = True
    matcher: str | None = Field(default=None, description="omit for the default; 'nearest_road' = baseline")


class PredictResponse(BaseModel):
    trip_id: str
    matcher: str
    used_hmm: bool
    n_observations: int
    predicted_arc_per_observation: list[int]
    route_arc_ids: list[int]
    route_geometry: list[list[list[float]]] = Field(description="gap-filled predicted route, one [[lat,lon],[lat,lon]] per arc")
    summary: dict[str, Any] | None = Field(description="headline numbers for the UI; null when no reference exists server-side")
    metrics: dict[str, Any] | None
    timing_ms: dict[str, float]
    reference_route_geometry: list[list[list[float]]] | None = None
