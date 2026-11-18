"""
PathSnap: Reusable Geospatial Data Preprocessing Pipeline
---------------------------------------------------------
Author: Senior Geospatial & ML Data Engineer
Project: PES University ML Hackathon - PathSnap

This module implements a reproducible, robust data preprocessing pipeline:
1. Validates schema and integrity for all input files.
2. Performs WGS84 geodesic distance and projected coordinate transformation
   (UTM Zone 32N / EPSG:32632 for Germany).
3. Computes physical trajectory kinematics:
   - Sampling delta t
   - Haversine geodesic displacement
   - Instantaneous speed (m/s and km/h)
   - Compass heading / bearing (0-360 deg)
4. Employs non-destructive data quality flagging:
   - Never blindly removes GPS points or interpolates missing locations.
   - Flags stationary/idling points, sensor dropout gaps (> 5s), and multipath speed spikes (> 150 km/h).
5. Enriches road network arcs with spatial geometries, lengths (m), and orientations.
6. Validates ground-truth reference routes without fabricating point-level labels.
7. Exports structured datasets (Parquet and CSV) to 'processed_data/' and writes
   audit reports to 'results/preprocessing/'.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from pyproj import Transformer

# Setup logging
os.makedirs("results/preprocessing", exist_ok=True)
log_file = "results/preprocessing/preprocessing_audit.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger('PathSnap-Preprocessor')


def haversine_vec(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Calculate great-circle distances in meters between coordinate arrays."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi * 0.5) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda * 0.5) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return R * c


def calculate_bearing_vec(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Calculate initial compass bearing (0-360 degrees) from point 1 to point 2."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)

    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    bearing_rad = np.arctan2(y, x)
    bearing_deg = (np.degrees(bearing_rad) + 360.0) % 360.0
    return bearing_deg


class PathSnapPreprocessor:
    """
    Standardized, reproducible preprocessing pipeline for PathSnap.
    """
    def __init__(self, epsg_code: int = 32632):
        """
        Initialize preprocessor with projected coordinate system.
        EPSG:32632 corresponds to WGS84 / UTM Zone 32N (Germany/Central Europe).
        """
        self.epsg_code = epsg_code
        self.transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
        logger.info(f"Initialized Coordinate Transformer: EPSG:4326 (WGS84) -> EPSG:{epsg_code} (UTM Zone 32N)")

    def process_trip(
        self,
        raw_trip_dir: str,
        trip_id: str,
        split: str,
        output_base_dir: str,
        save_csv: bool = True,
        save_parquet: bool = True
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Preprocess all four files of a single trip, compute kinematics, quality flags,
        and save to destination directory.
        """
        track_p = os.path.join(raw_trip_dir, f"{trip_id}.track")
        nodes_p = os.path.join(raw_trip_dir, f"{trip_id}.nodes")
        arcs_p = os.path.join(raw_trip_dir, f"{trip_id}.arcs")
        route_p = os.path.join(raw_trip_dir, f"{trip_id}.route")

        audit_entries: List[Dict[str, Any]] = []

        # 1. Read files
        track_df = pd.read_csv(track_p, sep=r'\s+', header=None, names=['lon', 'lat', 'time'], dtype={'lon': float, 'lat': float, 'time': float})
        nodes_df = pd.read_csv(nodes_p, sep=r'\s+', header=None, names=['lon', 'lat'], dtype={'lon': float, 'lat': float})
        arcs_df = pd.read_csv(arcs_p, sep=r'\s+', header=None, names=['from_node', 'to_node'], dtype={'from_node': int, 'to_node': int})
        route_df = pd.read_csv(route_p, sep=r'\s+', header=None, names=['arc_id'], dtype={'arc_id': int})

        n_obs = len(track_df)
        num_nodes = len(nodes_df)
        num_arcs = len(arcs_df)
        n_route = len(route_df)

        # 2. Process Nodes (assign node_id, project to UTM 32N)
        nodes_df['node_id'] = np.arange(num_nodes, dtype=np.int64)
        utm_node_x, utm_node_y = self.transformer.transform(nodes_df['lon'].values, nodes_df['lat'].values)
        nodes_df['utm_easting'] = utm_node_x
        nodes_df['utm_northing'] = utm_node_y
        nodes_df = nodes_df[['node_id', 'lon', 'lat', 'utm_easting', 'utm_northing']]

        # 3. Process Arcs (assign arc_id, endpoints coordinates, length, bearing)
        arcs_df['arc_id'] = np.arange(num_arcs, dtype=np.int64)
        
        # Endpoints coordinates
        from_nodes = arcs_df['from_node'].values
        to_nodes = arcs_df['to_node'].values

        arcs_df['from_lon'] = nodes_df.loc[from_nodes, 'lon'].values
        arcs_df['from_lat'] = nodes_df.loc[from_nodes, 'lat'].values
        arcs_df['to_lon'] = nodes_df.loc[to_nodes, 'lon'].values
        arcs_df['to_lat'] = nodes_df.loc[to_nodes, 'lat'].values

        arcs_df['from_utm_x'] = nodes_df.loc[from_nodes, 'utm_easting'].values
        arcs_df['from_utm_y'] = nodes_df.loc[from_nodes, 'utm_northing'].values
        arcs_df['to_utm_x'] = nodes_df.loc[to_nodes, 'utm_easting'].values
        arcs_df['to_utm_y'] = nodes_df.loc[to_nodes, 'utm_northing'].values

        # Geodesic length in meters
        arcs_df['length_m'] = haversine_vec(
            arcs_df['from_lon'].values, arcs_df['from_lat'].values,
            arcs_df['to_lon'].values, arcs_df['to_lat'].values
        )

        # Arc orientation / bearing
        arcs_df['bearing_deg'] = calculate_bearing_vec(
            arcs_df['from_lon'].values, arcs_df['from_lat'].values,
            arcs_df['to_lon'].values, arcs_df['to_lat'].values
        )
        arcs_df['is_self_loop'] = (arcs_df['from_node'] == arcs_df['to_node'])

        # 4. Process Track (preserve exact ordering, add kinematics, UTM coords, quality flags)
        track_df['trip_id'] = trip_id
        track_df['obs_index'] = np.arange(n_obs, dtype=np.int64)
        
        utm_track_x, utm_track_y = self.transformer.transform(track_df['lon'].values, track_df['lat'].values)
        track_df['utm_easting'] = utm_track_x
        track_df['utm_northing'] = utm_track_y

        # Delta t
        dt_vals = np.zeros(n_obs, dtype=np.float64)
        if n_obs > 1:
            dt_vals[1:] = np.diff(track_df['time'].values)
        track_df['dt_seconds'] = dt_vals

        # Distance to previous point (Haversine meters)
        dist_vals = np.zeros(n_obs, dtype=np.float64)
        if n_obs > 1:
            dist_vals[1:] = haversine_vec(
                track_df['lon'].values[:-1], track_df['lat'].values[:-1],
                track_df['lon'].values[1:], track_df['lat'].values[1:]
            )
        track_df['dist_to_prev_m'] = dist_vals

        # Speed (m/s and km/h)
        speed_mps = np.zeros(n_obs, dtype=np.float64)
        valid_dt_mask = dt_vals > 0.0
        speed_mps[valid_dt_mask] = dist_vals[valid_dt_mask] / dt_vals[valid_dt_mask]
        track_df['speed_mps'] = speed_mps
        track_df['speed_kmh'] = speed_mps * 3.6

        # Forward bearing to next point
        bearing_vals = np.zeros(n_obs, dtype=np.float64)
        if n_obs > 1:
            bearing_vals[:-1] = calculate_bearing_vec(
                track_df['lon'].values[:-1], track_df['lat'].values[:-1],
                track_df['lon'].values[1:], track_df['lat'].values[1:]
            )
            bearing_vals[-1] = bearing_vals[-2]  # replicate last known heading
        track_df['bearing_deg'] = bearing_vals

        # Quality & Event Flags
        # 1. Stationary / idling: displacement < 0.5m while moving or stopped
        track_df['is_stationary'] = (dist_vals < 0.5)
        # 2. Time gap: sampling interval > 5.0 seconds
        track_df['is_time_gap'] = (dt_vals > 5.0)
        # 3. Speed outlier: speed > 150 km/h (multipath/glitch)
        track_df['is_speed_outlier'] = (track_df['speed_kmh'] > 150.0)

        # Log audit events
        n_stationary = int(track_df['is_stationary'].sum())
        n_gaps = int(track_df['is_time_gap'].sum())
        n_speed_outliers = int(track_df['is_speed_outlier'].sum())

        if n_gaps > 0:
            for g_idx in np.where(track_df['is_time_gap'].values)[0]:
                audit_entries.append({
                    'split': split, 'trip_id': trip_id, 'obs_index': int(g_idx),
                    'flag': 'TIME_GAP', 'value': float(track_df.loc[g_idx, 'dt_seconds']),
                    'action': 'FLAGGED_PRESERVED',
                    'justification': f"Sampling interval {track_df.loc[g_idx, 'dt_seconds']:.1f}s > 5s threshold; point preserved with quality flag."
                })

        if n_speed_outliers > 0:
            for s_idx in np.where(track_df['is_speed_outlier'].values)[0]:
                audit_entries.append({
                    'split': split, 'trip_id': trip_id, 'obs_index': int(s_idx),
                    'flag': 'SPEED_OUTLIER', 'value': float(track_df.loc[s_idx, 'speed_kmh']),
                    'action': 'FLAGGED_PRESERVED',
                    'justification': f"Instantaneous speed {track_df.loc[s_idx, 'speed_kmh']:.1f} km/h exceeds 150 km/h; flagged as sensor noise, sequence preserved."
                })

        # Order columns cleanly
        track_df = track_df[[
            'trip_id', 'obs_index', 'lon', 'lat', 'time',
            'utm_easting', 'utm_northing',
            'dt_seconds', 'dist_to_prev_m', 'speed_mps', 'speed_kmh', 'bearing_deg',
            'is_stationary', 'is_time_gap', 'is_speed_outlier'
        ]]

        # 5. Process Reference Route
        route_df['sequence_order'] = np.arange(n_route, dtype=np.int64)
        r_arcs = route_df['arc_id'].values
        route_df['from_node'] = arcs_df.loc[r_arcs, 'from_node'].values
        route_df['to_node'] = arcs_df.loc[r_arcs, 'to_node'].values
        route_df['length_m'] = arcs_df.loc[r_arcs, 'length_m'].values

        # Continuity flag
        is_conn = np.ones(n_route, dtype=bool)
        if n_route > 1:
            curr_to = route_df['to_node'].values[:-1]
            next_from = route_df['from_node'].values[1:]
            is_conn[1:] = (curr_to == next_from)
        route_df['is_connected_to_prev'] = is_conn

        n_disc = int((~is_conn).sum())
        if n_disc > 0:
            for d_idx in np.where(~is_conn)[0]:
                audit_entries.append({
                    'split': split, 'trip_id': trip_id, 'obs_index': int(d_idx),
                    'flag': 'ROUTE_DISCONTINUITY', 'value': float(route_df.loc[d_idx, 'arc_id']),
                    'action': 'FLAGGED_DOCUMENTED',
                    'justification': f"Route transition broken at step {d_idx}: to_node {route_df.loc[d_idx-1, 'to_node']} != from_node {route_df.loc[d_idx, 'from_node']}."
                })

        # 6. Save Processed Files
        trip_out_dir = os.path.join(output_base_dir, split, trip_id)
        os.makedirs(trip_out_dir, exist_ok=True)

        if save_parquet:
            track_df.to_parquet(os.path.join(trip_out_dir, f"{trip_id}_track.parquet"), index=False)
            nodes_df.to_parquet(os.path.join(trip_out_dir, f"{trip_id}_nodes.parquet"), index=False)
            arcs_df.to_parquet(os.path.join(trip_out_dir, f"{trip_id}_arcs.parquet"), index=False)
            route_df.to_parquet(os.path.join(trip_out_dir, f"{trip_id}_route.parquet"), index=False)

        if save_csv:
            track_df.to_csv(os.path.join(trip_out_dir, f"{trip_id}_track.csv"), index=False)
            nodes_df.to_csv(os.path.join(trip_out_dir, f"{trip_id}_nodes.csv"), index=False)
            arcs_df.to_csv(os.path.join(trip_out_dir, f"{trip_id}_arcs.csv"), index=False)
            route_df.to_csv(os.path.join(trip_out_dir, f"{trip_id}_route.csv"), index=False)

        # Compile trip summary metadata
        trip_meta = {
            'split': split,
            'trip_id': trip_id,
            'obs_count': n_obs,
            'duration_s': float(track_df['time'].iloc[-1] - track_df['time'].iloc[0]) if n_obs > 1 else 0.0,
            'distance_m': float(np.sum(dist_vals)),
            'mean_speed_kmh': float(np.mean(track_df['speed_kmh'])),
            'max_speed_kmh': float(np.max(track_df['speed_kmh'])),
            'num_nodes': num_nodes,
            'num_arcs': num_arcs,
            'route_arcs': n_route,
            'route_length_m': float(np.sum(route_df['length_m'])),
            'stationary_pts': n_stationary,
            'gap_pts': n_gaps,
            'speed_outliers': n_speed_outliers,
            'route_breaks': n_disc,
            'min_lon': float(track_df['lon'].min()),
            'max_lon': float(track_df['lon'].max()),
            'min_lat': float(track_df['lat'].min()),
            'max_lat': float(track_df['lat'].max()),
            'min_utm_easting': float(track_df['utm_easting'].min()),
            'max_utm_easting': float(track_df['utm_easting'].max()),
            'min_utm_northing': float(track_df['utm_northing'].min()),
            'max_utm_northing': float(track_df['utm_northing'].max())
        }

        return trip_meta, audit_entries


def run_preprocessing_pipeline(
    raw_data_dir: str = "Student folder",
    output_base_dir: str = "processed_data",
    reports_dir: str = "results/preprocessing",
    save_csv: bool = True,
    save_parquet: bool = True
):
    """
    Execute full dataset preprocessing pipeline across train and student_test.
    """
    logger.info(f"Starting Preprocessing Pipeline: Source='{raw_data_dir}', Target='{output_base_dir}'")
    
    os.makedirs(os.path.join(output_base_dir, "metadata"), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, "reports"), exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    preprocessor = PathSnapPreprocessor()
    splits = ['train', 'student_test']

    all_metadata: List[Dict[str, Any]] = []
    all_audit_logs: List[Dict[str, Any]] = []

    for split in splits:
        split_p = os.path.join(raw_data_dir, split)
        if not os.path.exists(split_p):
            logger.warning(f"Split '{split}' does not exist at {split_p}")
            continue

        trip_ids = sorted([d for d in os.listdir(split_p) if os.path.isdir(os.path.join(split_p, d))])
        logger.info(f"Processing partition '{split}' with {len(trip_ids)} trips...")

        for trip_id in trip_ids:
            trip_meta, audit_entries = preprocessor.process_trip(
                raw_trip_dir=os.path.join(split_p, trip_id),
                trip_id=trip_id,
                split=split,
                output_base_dir=output_base_dir,
                save_csv=save_csv,
                save_parquet=save_parquet
            )
            all_metadata.append(trip_meta)
            all_audit_logs.extend(audit_entries)

    # 1. Save Trips Metadata
    meta_df = pd.DataFrame(all_metadata)
    meta_csv = os.path.join(output_base_dir, "metadata", "trips_metadata.csv")
    meta_parquet = os.path.join(output_base_dir, "metadata", "trips_metadata.parquet")
    meta_df.to_csv(meta_csv, index=False)
    meta_df.to_parquet(meta_parquet, index=False)
    logger.info(f"Saved trips metadata to '{meta_csv}' and '{meta_parquet}'")

    # 2. Save Audit Log
    audit_df = pd.DataFrame(all_audit_logs)
    if audit_df.empty:
        audit_df = pd.DataFrame([{'status': 'Clean', 'message': 'No anomalies or gaps flagged.'}])
    audit_csv = os.path.join(output_base_dir, "reports", "data_cleaning_audit.csv")
    audit_df.to_csv(audit_csv, index=False)
    logger.info(f"Saved data cleaning audit log to '{audit_csv}' ({len(audit_df)} records)")

    # 3. Save Cleaning Summary
    cleaning_summary = {
        'total_trips_processed': len(meta_df),
        'total_gps_observations': int(meta_df['obs_count'].sum()),
        'total_road_nodes': int(meta_df['num_nodes'].sum()),
        'total_road_arcs': int(meta_df['num_arcs'].sum()),
        'total_route_arcs': int(meta_df['route_arcs'].sum()),
        'total_stationary_points_flagged': int(meta_df['stationary_pts'].sum()),
        'total_time_gaps_flagged': int(meta_df['gap_pts'].sum()),
        'total_speed_outliers_flagged': int(meta_df['speed_outliers'].sum()),
        'total_route_breaks_documented': int(meta_df['route_breaks'].sum()),
        'points_removed_count': 0,
        'points_interpolated_count': 0,
        'points_preservation_rate_pct': 100.0,
        'labels_fabricated_count': 0
    }
    summary_df = pd.DataFrame([cleaning_summary])
    summary_csv = os.path.join(reports_dir, "cleaning_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"Saved cleaning summary to '{summary_csv}'")

    # 4. Save Formal Data Dictionary
    save_data_dictionary(output_base_dir)

    logger.info("Preprocessing Pipeline Completed Successfully.")
    return cleaning_summary


def save_data_dictionary(output_base_dir: str):
    """Generate formal schema and data dictionary documentation."""
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "PathSnap Preprocessed Dataset Schema",
        "description": "Standardized schema for distributed map-matching and route reconstruction",
        "entities": {
            "track": {
                "columns": {
                    "trip_id": {"type": "string", "description": "Unique 8-character zero-padded trip identifier"},
                    "obs_index": {"type": "integer", "description": "Original 0-based chronological observation index"},
                    "lon": {"type": "number", "unit": "degrees", "crs": "EPSG:4326 (WGS84)", "description": "Longitude"},
                    "lat": {"type": "number", "unit": "degrees", "crs": "EPSG:4326 (WGS84)", "description": "Latitude"},
                    "time": {"type": "number", "unit": "seconds", "description": "Elapsed timestamp from trip epoch"},
                    "utm_easting": {"type": "number", "unit": "meters", "crs": "EPSG:32632 (UTM Zone 32N)", "description": "Projected X coordinate"},
                    "utm_northing": {"type": "number", "unit": "meters", "crs": "EPSG:32632 (UTM Zone 32N)", "description": "Projected Y coordinate"},
                    "dt_seconds": {"type": "number", "unit": "seconds", "description": "Sampling interval to previous observation"},
                    "dist_to_prev_m": {"type": "number", "unit": "meters", "description": "Haversine geodesic distance to previous observation"},
                    "speed_mps": {"type": "number", "unit": "m/s", "description": "Instantaneous ground speed"},
                    "speed_kmh": {"type": "number", "unit": "km/h", "description": "Instantaneous speed in km/h"},
                    "bearing_deg": {"type": "number", "unit": "degrees", "range": [0, 360], "description": "Forward azimuth heading to next observation"},
                    "is_stationary": {"type": "boolean", "description": "True if distance < 0.5m (stopped/idling)"},
                    "is_time_gap": {"type": "boolean", "description": "True if dt > 5.0s (sensor dropout)"},
                    "is_speed_outlier": {"type": "boolean", "description": "True if speed > 150 km/h (multipath jump)"}
                }
            },
            "nodes": {
                "columns": {
                    "node_id": {"type": "integer", "description": "0-based vertex index in road graph"},
                    "lon": {"type": "number", "unit": "degrees", "description": "Node WGS84 Longitude"},
                    "lat": {"type": "number", "unit": "degrees", "description": "Node WGS84 Latitude"},
                    "utm_easting": {"type": "number", "unit": "meters", "description": "Node UTM 32N Easting"},
                    "utm_northing": {"type": "number", "unit": "meters", "description": "Node UTM 32N Northing"}
                }
            },
            "arcs": {
                "columns": {
                    "arc_id": {"type": "integer", "description": "0-based directed edge index in road graph"},
                    "from_node": {"type": "integer", "description": "Origin vertex node_id"},
                    "to_node": {"type": "integer", "description": "Destination vertex node_id"},
                    "from_lon": {"type": "number", "description": "Origin WGS84 Longitude"},
                    "from_lat": {"type": "number", "description": "Origin WGS84 Latitude"},
                    "to_lon": {"type": "number", "description": "Destination WGS84 Longitude"},
                    "to_lat": {"type": "number", "description": "Destination WGS84 Latitude"},
                    "from_utm_x": {"type": "number", "unit": "meters", "description": "Origin UTM Easting"},
                    "from_utm_y": {"type": "number", "unit": "meters", "description": "Origin UTM Northing"},
                    "to_utm_x": {"type": "number", "unit": "meters", "description": "Destination UTM Easting"},
                    "to_utm_y": {"type": "number", "unit": "meters", "description": "Destination UTM Northing"},
                    "length_m": {"type": "number", "unit": "meters", "description": "Arc physical length in meters"},
                    "bearing_deg": {"type": "number", "unit": "degrees", "description": "Directed arc compass heading"},
                    "is_self_loop": {"type": "boolean", "description": "True if from_node == to_node"}
                }
            },
            "route": {
                "columns": {
                    "sequence_order": {"type": "integer", "description": "0-based chronological step index of traversed path"},
                    "arc_id": {"type": "integer", "description": "Reference road segment arc_id"},
                    "from_node": {"type": "integer", "description": "Arc origin vertex"},
                    "to_node": {"type": "integer", "description": "Arc destination vertex"},
                    "length_m": {"type": "number", "unit": "meters", "description": "Arc length in meters"},
                    "is_connected_to_prev": {"type": "boolean", "description": "True if arc smoothly connects from prior step"}
                }
            }
        }
    }

    schema_path = os.path.join(output_base_dir, "metadata", "dataset_schema.json")
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)
    logger.info(f"Saved dataset schema JSON to '{schema_path}'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PathSnap Data Preprocessing Pipeline")
    parser.add_argument("--data-dir", default="Student folder", help="Raw dataset root directory")
    parser.add_argument("--output-dir", default="processed_data", help="Output directory for preprocessed dataset")
    parser.add_argument("--reports-dir", default="results/preprocessing", help="Output directory for preprocessing reports")
    args = parser.parse_args()

    run_preprocessing_pipeline(
        raw_data_dir=args.data_dir,
        output_base_dir=args.output_dir,
        reports_dir=args.reports_dir,
        save_csv=True,
        save_parquet=True
    )
