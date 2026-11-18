"""Folium map: GPS observations, road network, reference route, classifier-only route, final predicted route.

OFFLINE / DEVELOPMENT USE (student_test or train partitions). The reference layer reads ground truth, so
this module is deliberately NOT imported by the API.
"""
from __future__ import annotations

import folium

from ..services.pipeline import PipelineRun


def build_map(run: PipelineRun, buffer_m: float = 120.0, max_gps_markers: int = 1500) -> folium.Map:
    net, obs, trip = run.net, run.obs, run.trip
    m = folium.Map(location=[float(trip.lat.mean()), float(trip.lon.mean())], zoom_start=14, tiles="cartodbpositron")
    m.fit_bounds([[float(trip.lat.min()), float(trip.lon.min())], [float(trip.lat.max()), float(trip.lon.max())]])

    # road network near the track
    roads = folium.FeatureGroup(name="Road network (within %d m of GPS)" % buffer_m, show=True)
    _, ai = net.query_radius(obs.x, obs.y, buffer_m)
    import numpy as np
    for a in np.unique(ai)[:20000]:
        folium.PolyLine(net.arc_lonlat(int(a)), color="#9aa0a6", weight=1.5, opacity=0.8).add_to(roads)
    roads.add_to(m)

    # raw GPS
    gps = folium.FeatureGroup(name="GPS observations", show=True)
    step = max(1, len(trip.lat) // max_gps_markers)
    for i in range(0, len(trip.lat), step):
        folium.CircleMarker([float(trip.lat[i]), float(trip.lon[i])], radius=2, color="#1a73e8", fill=True,
                            fill_opacity=0.8, weight=0, tooltip=f"obs {i}  t={trip.t[i]:.0f}s").add_to(gps)
    gps.add_to(m)

    def route_layer(name, arcs, color, weight, show, dash=None):
        g = folium.FeatureGroup(name=name, show=show)
        for a in arcs:
            folium.PolyLine(net.arc_lonlat(int(a)), color=color, weight=weight, opacity=0.85, dash_array=dash,
                            tooltip=f"arc {int(a)}").add_to(g)
        g.add_to(m)

    if run.reference is not None:
        route_layer("Reference route (green)", run.reference.route, "#188038", 7, True)
    from ..core.hmm import build_route
    if run.hmm_arcs is not None:
        route_layer("Classifier only (orange, dashed)", build_route(run.clf_arcs, net)[1], "#f29900", 4, False, "6 6")
    route_layer(f"Predicted route: {run.matcher_name}{' + HMM/Viterbi' if run.used_hmm else ''} (red)",
                run.route_filled, "#d93025", 4, True)

    legend = """<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 12px;
      border:1px solid #888;border-radius:6px;font:12px sans-serif;line-height:1.6">
      <b>Trip %s</b><br><span style="color:#1a73e8">&#9679;</span> GPS observations<br>
      <span style="color:#9aa0a6">&#9473;</span> road network<br>
      <span style="color:#188038">&#9473;</span> reference route<br>
      <span style="color:#f29900">&#9473;</span> classifier only<br>
      <span style="color:#d93025">&#9473;</span> predicted route</div>""" % trip.trip_id
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    return m
