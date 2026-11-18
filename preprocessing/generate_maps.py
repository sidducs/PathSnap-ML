"""
PathSnap: Interactive Geospatial Visualization Suite (Folium)
--------------------------------------------------------------
Author: Senior Geospatial & ML Data Engineer
Project: PES University ML Hackathon - PathSnap

Generates rich, interactive HTML Folium maps visualizing:
1. Raw GPS observations with interactive tooltips (timestamps, speeds, bearings).
2. Start and End trip markers.
3. Candidate road network topology (arcs in the vicinity of trajectory).
4. Ground-truth reference routes (highlighted contiguous paths).
5. Data quality annotations (anomalies, long gaps, stationary clusters).
"""

import os
import sys
import logging
from typing import Optional, List

import numpy as np
import pandas as pd
import folium
from folium import plugins

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('PathSnap-Maps')


def create_trip_map(
    trip_id: str,
    split: str = "train",
    processed_dir: str = "processed_data",
    max_network_arcs: int = 1500,
    output_html_path: Optional[str] = None
) -> folium.Map:
    """
    Generate an interactive Folium map for a single trip.
    """
    trip_dir = os.path.join(processed_dir, split, trip_id)
    track_p = os.path.join(trip_dir, f"{trip_id}_track.parquet")
    arcs_p = os.path.join(trip_dir, f"{trip_id}_arcs.parquet")
    route_p = os.path.join(trip_dir, f"{trip_id}_route.parquet")

    # Fallback to CSV if parquet not present
    if not os.path.exists(track_p):
        track_p = os.path.join(trip_dir, f"{trip_id}_track.csv")
        arcs_p = os.path.join(trip_dir, f"{trip_id}_arcs.csv")
        route_p = os.path.join(trip_dir, f"{trip_id}_route.csv")

    if not os.path.exists(track_p):
        raise FileNotFoundError(f"Track data not found for trip {trip_id} in {trip_dir}")

    track_df = pd.read_parquet(track_p) if track_p.endswith('.parquet') else pd.read_csv(track_p)
    arcs_df = pd.read_parquet(arcs_p) if arcs_p.endswith('.parquet') else pd.read_csv(arcs_p)
    route_df = pd.read_parquet(route_p) if route_p.endswith('.parquet') else pd.read_csv(route_p)

    # Calculate center of trajectory
    center_lat = float(track_df['lat'].mean())
    center_lon = float(track_df['lon'].mean())

    # Initialize Folium Map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles=None
    )

    # Base layers
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap', control=True).add_to(m)
    folium.TileLayer('CartoDB positron', name='CartoDB Light (Clean)', control=True).add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark', control=True).add_to(m)

    # Feature groups
    fg_road_network = folium.FeatureGroup(name="Road Network Candidates (Nearby Arcs)", show=True)
    fg_reference_route = folium.FeatureGroup(name="Reference Route (Ground Truth)", show=True)
    fg_gps_track = folium.FeatureGroup(name="GPS Trajectory Line", show=True)
    fg_gps_points = folium.FeatureGroup(name="GPS Observations (Individual Points)", show=False)
    fg_anomalies = folium.FeatureGroup(name="Flagged Anomalies (Gaps / Spikes)", show=True)

    # 1. Road Network Arcs (filter to bounding box + buffer to keep HTML lightweight)
    t_min_lat, t_max_lat = track_df['lat'].min() - 0.008, track_df['lat'].max() + 0.008
    t_min_lon, t_max_lon = track_df['lon'].min() - 0.008, track_df['lon'].max() + 0.008

    nearby_arcs = arcs_df[
        (arcs_df['from_lat'] >= t_min_lat) & (arcs_df['from_lat'] <= t_max_lat) &
        (arcs_df['from_lon'] >= t_min_lon) & (arcs_df['from_lon'] <= t_max_lon)
    ]
    if len(nearby_arcs) > max_network_arcs:
        nearby_arcs = nearby_arcs.iloc[:max_network_arcs]

    for _, arc in nearby_arcs.iterrows():
        folium.PolyLine(
            locations=[[arc['from_lat'], arc['from_lon']], [arc['to_lat'], arc['to_lon']]],
            color="#8899a6",
            weight=1.5,
            opacity=0.5,
            tooltip=f"Arc ID: {int(arc['arc_id'])} | Len: {arc['length_m']:.1f}m"
        ).add_to(fg_road_network)

    # 2. Reference Route (Traversed Arcs)
    route_arc_ids = set(route_df['arc_id'].values)
    route_arcs_df = arcs_df.loc[arcs_df['arc_id'].isin(route_arc_ids)]

    for _, r_arc in route_arcs_df.iterrows():
        folium.PolyLine(
            locations=[[r_arc['from_lat'], r_arc['from_lon']], [r_arc['to_lat'], r_arc['to_lon']]],
            color="#27ae60",  # Emerald green
            weight=4.5,
            opacity=0.9,
            tooltip=f"Ref Route Arc: {int(r_arc['arc_id'])} | Length: {r_arc['length_m']:.1f}m"
        ).add_to(fg_reference_route)

    # 3. GPS Trajectory Line
    gps_coords = track_df[['lat', 'lon']].values.tolist()
    folium.PolyLine(
        locations=gps_coords,
        color="#2980b9",  # Strong blue
        weight=2.5,
        opacity=0.85,
        tooltip=f"Trip {trip_id} GPS Track ({len(track_df)} observations)"
    ).add_to(fg_gps_track)

    # 4. GPS Observation Points (Tooltips)
    for _, pt in track_df.iterrows():
        popup_html = f"""
        <div style='font-family: Arial; font-size: 11px;'>
            <b>Trip:</b> {pt['trip_id']}<br/>
            <b>Obs Index:</b> {int(pt['obs_index'])}<br/>
            <b>Timestamp:</b> {pt['time']:.1f}s<br/>
            <b>Delta t:</b> {pt['dt_seconds']:.1f}s<br/>
            <b>Speed:</b> {pt['speed_kmh']:.1f} km/h<br/>
            <b>Bearing:</b> {pt['bearing_deg']:.1f}&deg;<br/>
            <b>Stationary:</b> {bool(pt['is_stationary'])}<br/>
            <b>Time Gap:</b> {bool(pt['is_time_gap'])}<br/>
            <b>Speed Spike:</b> {bool(pt['is_speed_outlier'])}
        </div>
        """
        folium.CircleMarker(
            location=[pt['lat'], pt['lon']],
            radius=2.5,
            color="#1b4f72",
            fill=True,
            fill_color="#3498db",
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(fg_gps_points)

    # 5. Flagged Anomalies
    anom_df = track_df[track_df['is_time_gap'] | track_df['is_speed_outlier']]
    for _, apt in anom_df.iterrows():
        reason = []
        if apt['is_time_gap']:
            reason.append(f"Gap: {apt['dt_seconds']:.1f}s")
        if apt['is_speed_outlier']:
            reason.append(f"Speed Spike: {apt['speed_kmh']:.1f} km/h")
        r_str = ", ".join(reason)

        folium.CircleMarker(
            location=[apt['lat'], apt['lon']],
            radius=6.0,
            color="#c0392b",
            fill=True,
            fill_color="#e74c3c",
            fill_opacity=0.9,
            tooltip=f"Anomaly at idx {int(apt['obs_index'])}: {r_str}"
        ).add_to(fg_anomalies)

    # 6. Start & End Markers
    start_pt = track_df.iloc[0]
    end_pt = track_df.iloc[-1]

    folium.Marker(
        location=[start_pt['lat'], start_pt['lon']],
        popup=f"<b>Trip Start</b><br/>Time: {start_pt['time']:.1f}s<br/>Lon: {start_pt['lon']:.6f}<br/>Lat: {start_pt['lat']:.6f}",
        tooltip="Trip Origin",
        icon=folium.Icon(color="green", icon="play", prefix="fa")
    ).add_to(m)

    folium.Marker(
        location=[end_pt['lat'], end_pt['lon']],
        popup=f"<b>Trip Destination</b><br/>Time: {end_pt['time']:.1f}s<br/>Lon: {end_pt['lon']:.6f}<br/>Lat: {end_pt['lat']:.6f}",
        tooltip="Trip Destination",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")
    ).add_to(m)

    # Add all feature groups to map
    fg_road_network.add_to(m)
    fg_reference_route.add_to(m)
    fg_gps_track.add_to(m)
    fg_gps_points.add_to(m)
    fg_anomalies.add_to(m)

    # Add layer control and full screen toggle
    folium.LayerControl(collapsed=False).add_to(m)
    plugins.Fullscreen(position='topright').add_to(m)

    if output_html_path:
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
        m.save(output_html_path)
        logger.info(f"Saved interactive Folium map to '{output_html_path}'")

    return m


def generate_all_sample_maps(
    processed_dir: str = "processed_data",
    output_dir: str = "results/eda"
):
    """
    Generate interactive Folium maps for representative trips.
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_trips = [
        ('train', '00000001', 'trip_00000001_standard_map.html'),
        ('train', '00000002', 'trip_00000002_anomaly_map.html'),
        ('student_test', '00000000', 'trip_00000000_student_test_map.html')
    ]

    for split, trip_id, filename in sample_trips:
        out_p = os.path.join(output_dir, filename)
        logger.info(f"Generating Folium map for {split}/{trip_id} -> '{out_p}'")
        create_trip_map(
            trip_id=trip_id,
            split=split,
            processed_dir=processed_dir,
            output_html_path=out_p
        )
        
        # Also mirror to results/preprocessing
        preproc_p = os.path.join("results/preprocessing", filename)
        create_trip_map(
            trip_id=trip_id,
            split=split,
            processed_dir=processed_dir,
            output_html_path=preproc_p
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PathSnap Folium Map Generator")
    parser.add_argument("--processed-dir", default="processed_data", help="Directory with preprocessed trips")
    parser.add_argument("--output-dir", default="results/eda", help="Output directory for HTML maps")
    parser.add_argument("--trip-id", default=None, help="Optional specific trip ID to render")
    parser.add_argument("--split", default="train", help="Split for the specific trip (train or student_test)")
    args = parser.parse_args()

    if args.trip_id:
        out_file = os.path.join(args.output_dir, f"trip_{args.trip_id}_map.html")
        create_trip_map(args.trip_id, split=args.split, processed_dir=args.processed_dir, output_html_path=out_file)
    else:
        generate_all_sample_maps(processed_dir=args.processed_dir, output_dir=args.output_dir)
