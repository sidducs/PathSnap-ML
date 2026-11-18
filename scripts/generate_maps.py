"""Generate interactive Folium maps for GPS trajectory, road network, reference route, and candidates.

Outputs:
    visualisations/gps_trajectory.html
    visualisations/road_network.html
    visualisations/reference_route.html
    visualisations/candidate_visualisation.html
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import folium
from folium import plugins
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.preprocessing.loader import load_trip
from ml.preprocessing.graph import DirectedRoadGraph
from ml.candidates.candidate_generation import RoadSegmentSpatialIndex, generate_candidates_for_trip
from ml.candidates.route_alignment import align_trip_reference_route


def create_gps_trajectory_map(trip, out_html: Path):
    """Generate interactive map showing raw GPS trajectory with timestamp markers."""
    track = trip.track
    lons = track["lon"].to_numpy()
    lats = track["lat"].to_numpy()
    times = track["time"].to_numpy()

    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")

    # Polyline connecting observations
    coords = list(zip(lats, lons))
    folium.PolyLine(
        locations=coords,
        color="#2563EB",
        weight=3,
        opacity=0.8,
        tooltip=f"GPS Trajectory - Trip {trip.trip_id} ({len(track)} points)",
    ).add_to(m)

    # Subsampled observation circle markers
    step = max(1, len(track) // 200)
    for i in range(0, len(track), step):
        folium.CircleMarker(
            location=[lats[i], lons[i]],
            radius=4,
            color="#1D4ED8",
            fill=True,
            fill_color="#3B82F6",
            fill_opacity=0.9,
            popup=f"Obs {i}<br>Time: {times[i]:.1f}s<br>Coord: ({lats[i]:.6f}, {lons[i]:.6f})",
        ).add_to(m)

    # Start and End markers
    folium.Marker(
        location=[lats[0], lons[0]],
        popup=f"START: Trip {trip.trip_id}<br>Time: {times[0]:.1f}s",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)

    folium.Marker(
        location=[lats[-1], lons[-1]],
        popup=f"FINISH: Trip {trip.trip_id}<br>Time: {times[-1]:.1f}s",
        icon=folium.Icon(color="red", icon="stop", prefix="fa"),
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(str(out_html))
    print(f"Saved: {out_html}")


def create_road_network_map(trip, graph, out_html: Path):
    """Generate interactive map showing the local road network topology."""
    track = trip.track
    min_lon, max_lon = track["lon"].min() - 0.01, track["lon"].max() + 0.01
    min_lat, max_lat = track["lat"].min() - 0.01, track["lat"].max() + 0.01

    center_lat = float(track["lat"].mean())
    center_lon = float(track["lon"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")

    # Filter arcs within bounding box
    arc_mask = (
        (graph.arc_start_lon >= min_lon) & (graph.arc_start_lon <= max_lon) &
        (graph.arc_start_lat >= min_lat) & (graph.arc_start_lat <= max_lat)
    )
    visible_arc_ids = np.where(arc_mask)[0]

    # Limit to at most 1500 segments to keep HTML responsive
    if len(visible_arc_ids) > 1500:
        visible_arc_ids = visible_arc_ids[:1500]

    road_layer = folium.FeatureGroup(name="Directed Road Segments")
    for a_id in visible_arc_ids:
        start_coord = [graph.arc_start_lat[a_id], graph.arc_start_lon[a_id]]
        end_coord = [graph.arc_end_lat[a_id], graph.arc_end_lon[a_id]]
        folium.PolyLine(
            locations=[start_coord, end_coord],
            color="#4B5563",
            weight=2,
            opacity=0.6,
            popup=f"Arc ID: {a_id}<br>From: {graph.arc_from[a_id]} -> To: {graph.arc_to[a_id]}<br>Length: {graph.arc_lengths[a_id]:.1f}m<br>Bearing: {graph.arc_bearings[a_id]:.1f}°",
        ).add_to(road_layer)

    road_layer.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(str(out_html))
    print(f"Saved: {out_html}")


def create_reference_route_map(trip, graph, out_html: Path):
    """Generate interactive map highlighting reference route sequence."""
    if trip.route is None:
        print(f"No route file for trip {trip.trip_id}, skipping reference route map.")
        return

    route_arcs = trip.route["arc_id"].to_numpy(dtype=int)

    # Calculate center from route arcs
    route_lats = graph.arc_start_lat[route_arcs]
    route_lons = graph.arc_start_lon[route_arcs]

    center_lat = float(np.mean(route_lats))
    center_lon = float(np.mean(route_lons))

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")

    # Add background GPS trajectory
    gps_coords = list(zip(trip.track["lat"], trip.track["lon"]))
    folium.PolyLine(
        locations=gps_coords,
        color="#9CA3AF",
        weight=2,
        dash_array="5, 10",
        opacity=0.6,
        tooltip="GPS Observations (Reference Overlay)",
    ).add_to(m)

    # Highlight reference route arcs with sequence progression
    route_layer = folium.FeatureGroup(name="Reference Route Arcs")
    for seq_idx, a_id in enumerate(route_arcs):
        start_coord = [graph.arc_start_lat[a_id], graph.arc_start_lon[a_id]]
        end_coord = [graph.arc_end_lat[a_id], graph.arc_end_lon[a_id]]
        folium.PolyLine(
            locations=[start_coord, end_coord],
            color="#10B981",
            weight=4,
            opacity=0.9,
            popup=f"Step: {seq_idx}<br>Arc ID: {a_id}<br>Nodes: {graph.arc_from[a_id]} -> {graph.arc_to[a_id]}<br>Length: {graph.arc_lengths[a_id]:.1f}m",
        ).add_to(route_layer)

    route_layer.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(str(out_html))
    print(f"Saved: {out_html}")


def create_candidate_visualisation_map(trip, graph, candidates_df, out_html: Path):
    """Generate interactive map showing candidate associations and projection lines."""
    center_lat = float(trip.track["lat"].mean())
    center_lon = float(trip.track["lon"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="CartoDB positron")

    # Select a focused segment of 20 observations for clear inspection
    sample_obs = sorted(candidates_df["observation_index"].unique())[:25]
    subset = candidates_df[candidates_df["observation_index"].isin(sample_obs)]

    gps_layer = folium.FeatureGroup(name="GPS Observations")
    cand_layer = folium.FeatureGroup(name="Candidate Road Segments")
    proj_layer = folium.FeatureGroup(name="Projection Links")

    for obs_idx in sample_obs:
        obs_cands = subset[subset["observation_index"] == obs_idx]
        if obs_cands.empty:
            continue

        first = obs_cands.iloc[0]
        obs_coord = [first["gps_lat"], first["gps_lon"]]

        folium.CircleMarker(
            location=obs_coord,
            radius=5,
            color="#2563EB",
            fill=True,
            fill_color="#3B82F6",
            fill_opacity=1.0,
            popup=f"Obs {obs_idx}<br>Candidates: {len(obs_cands)}",
        ).add_to(gps_layer)

        for _, cand in obs_cands.iterrows():
            a_id = int(cand["arc_id"])
            is_pos = (cand.get("label", 0) == 1)
            color = "#10B981" if is_pos else "#F59E0B"
            weight = 4 if is_pos else 2
            opacity = 0.9 if is_pos else 0.5

            start_coord = [graph.arc_start_lat[a_id], graph.arc_start_lon[a_id]]
            end_coord = [graph.arc_end_lat[a_id], graph.arc_end_lon[a_id]]

            folium.PolyLine(
                locations=[start_coord, end_coord],
                color=color,
                weight=weight,
                opacity=opacity,
                popup=f"Obs {obs_idx} Candidate<br>Arc {a_id}<br>Rank: {cand['candidate_rank']}<br>Dist: {cand['distance_m']:.1f}m<br>Label: {cand.get('label', 'N/A')}",
            ).add_to(cand_layer)

            # Draw dashed projection line from GPS to projected point on segment
            if cand["candidate_rank"] == 0 or is_pos:
                proj_lon, proj_lat = graph.projection.inverse(cand["proj_x"], cand["proj_y"])
                folium.PolyLine(
                    locations=[obs_coord, [proj_lat, proj_lon]],
                    color="#EF4444" if not is_pos else "#059669",
                    weight=1.5,
                    dash_array="3, 5",
                    opacity=0.7,
                ).add_to(proj_layer)

    gps_layer.add_to(m)
    cand_layer.add_to(m)
    proj_layer.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(str(out_html))
    print(f"Saved: {out_html}")


def main():
    vis_dir = REPO_ROOT / "visualisations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Pick representative trip from training partition
    trip_id = "00000005"
    trip_path = REPO_ROOT / "data" / "raw" / "Student folder" / "train" / trip_id

    print(f"Generating Folium visualisations for representative training trip: {trip_id}")
    trip = load_trip(trip_path, partition="train", validate=True)
    graph = DirectedRoadGraph(trip.nodes, trip.arcs)
    spatial_index = RoadSegmentSpatialIndex(graph)

    # 1. GPS Trajectory Map
    create_gps_trajectory_map(trip, vis_dir / "gps_trajectory.html")

    # 2. Road Network Map
    create_road_network_map(trip, graph, vis_dir / "road_network.html")

    # 3. Reference Route Map
    create_reference_route_map(trip, graph, vis_dir / "reference_route.html")

    # 4. Candidate Visualisation Map
    cands_df, _ = generate_candidates_for_trip(trip, graph=graph, spatial_index=spatial_index, candidate_radius_m=60.0)
    labeled_cands, _ = align_trip_reference_route(trip, cands_df, graph=graph)
    create_candidate_visualisation_map(trip, graph, labeled_cands, vis_dir / "candidate_visualisation.html")

    print("All Folium visualisations generated successfully!")


if __name__ == "__main__":
    main()

