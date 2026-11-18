"""
PathSnap: Dataset Schema & Topological Validation Suite
-------------------------------------------------------
Author: Senior Geospatial & ML Data Engineer
Project: PES University ML Hackathon - PathSnap

Validates dataset partitions (train, student_test) for:
- File presence (.track, .nodes, .arcs, .route)
- Column counts, delimiters, missing values, and data types
- Geographical coordinate plausibility (WGS84 & regional bounding box)
- Timestamp monotonicity and temporal gaps
- Road network integrity (node reference bounds, self-loops, duplicate arcs)
- Reference route validity and graph path continuity
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('PathSnap-Validator')


def validate_trip(
    trip_dir: str,
    trip_id: str,
    strict_route_continuity: bool = False
) -> Dict[str, Any]:
    """
    Validate all 4 files of a single trip against schema and integrity rules.
    """
    issues: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    # 1. File existence
    required_files = {
        'track': os.path.join(trip_dir, f"{trip_id}.track"),
        'nodes': os.path.join(trip_dir, f"{trip_id}.nodes"),
        'arcs': os.path.join(trip_dir, f"{trip_id}.arcs"),
        'route': os.path.join(trip_dir, f"{trip_id}.route")
    }

    for ftype, fpath in required_files.items():
        if not os.path.exists(fpath):
            issues.append({
                'trip_id': trip_id, 'file_type': ftype, 'severity': 'FATAL',
                'rule': 'File Existence', 'message': f"Missing file: {fpath}"
            })

    if issues:
        return {'trip_id': trip_id, 'is_valid': False, 'issues': issues, 'warnings': warnings}

    # 2. Parse .track
    try:
        track_df = pd.read_csv(
            required_files['track'], sep=r'\s+', header=None,
            names=['lon', 'lat', 'time'], dtype={'lon': float, 'lat': float, 'time': float}
        )
        if track_df.isna().sum().sum() > 0:
            issues.append({
                'trip_id': trip_id, 'file_type': 'track', 'severity': 'FATAL',
                'rule': 'No Nulls', 'message': 'Found NaN/Null in GPS track'
            })
        
        # Coordinate bounds (Germany / WGS84)
        if (track_df['lon'] < -180.0).any() or (track_df['lon'] > 180.0).any():
            issues.append({'trip_id': trip_id, 'file_type': 'track', 'severity': 'FATAL', 'rule': 'Longitude Range', 'message': 'Longitude out of [-180, 180]'})
        if (track_df['lat'] < -90.0).any() or (track_df['lat'] > 90.0).any():
            issues.append({'trip_id': trip_id, 'file_type': 'track', 'severity': 'FATAL', 'rule': 'Latitude Range', 'message': 'Latitude out of [-90, 90]'})
        
        # Timestamp monotonicity
        time_diffs = track_df['time'].diff().dropna()
        if (time_diffs <= 0).any():
            issues.append({'trip_id': trip_id, 'file_type': 'track', 'severity': 'FATAL', 'rule': 'Timestamp Monotonicity', 'message': 'Timestamps non-monotonic or duplicated'})
        
        # Large gaps warning
        large_gaps = int((time_diffs > 60.0).sum())
        if large_gaps > 0:
            warnings.append({'trip_id': trip_id, 'file_type': 'track', 'severity': 'WARNING', 'rule': 'Time Gap', 'message': f"{large_gaps} time gap(s) > 60s detected"})

    except Exception as e:
        issues.append({'trip_id': trip_id, 'file_type': 'track', 'severity': 'FATAL', 'rule': 'Parse Error', 'message': str(e)})
        track_df = pd.DataFrame()

    # 3. Parse .nodes
    try:
        nodes_df = pd.read_csv(
            required_files['nodes'], sep=r'\s+', header=None,
            names=['lon', 'lat'], dtype={'lon': float, 'lat': float}
        )
        if nodes_df.isna().sum().sum() > 0:
            issues.append({'trip_id': trip_id, 'file_type': 'nodes', 'severity': 'FATAL', 'rule': 'No Nulls', 'message': 'Found NaN/Null in nodes'})
        
        num_nodes = len(nodes_df)
        if num_nodes == 0:
            issues.append({'trip_id': trip_id, 'file_type': 'nodes', 'severity': 'FATAL', 'rule': 'Non-Empty', 'message': 'Nodes file is empty'})
    except Exception as e:
        issues.append({'trip_id': trip_id, 'file_type': 'nodes', 'severity': 'FATAL', 'rule': 'Parse Error', 'message': str(e)})
        nodes_df = pd.DataFrame()
        num_nodes = 0

    # 4. Check track bounded by nodes
    if not track_df.empty and not nodes_df.empty:
        t_min_lon, t_max_lon = track_df['lon'].min(), track_df['lon'].max()
        t_min_lat, t_max_lat = track_df['lat'].min(), track_df['lat'].max()
        n_min_lon, n_max_lon = nodes_df['lon'].min(), nodes_df['lon'].max()
        n_min_lat, n_max_lat = nodes_df['lat'].min(), nodes_df['lat'].max()

        if t_min_lon < n_min_lon or t_max_lon > n_max_lon or t_min_lat < n_min_lat or t_max_lat > n_max_lat:
            warnings.append({'trip_id': trip_id, 'file_type': 'track', 'severity': 'WARNING', 'rule': 'Bounding Box', 'message': 'Track points extend outside road nodes bounding box'})

    # 5. Parse .arcs
    try:
        arcs_df = pd.read_csv(
            required_files['arcs'], sep=r'\s+', header=None,
            names=['from_node', 'to_node'], dtype={'from_node': int, 'to_node': int}
        )
        if arcs_df.isna().sum().sum() > 0:
            issues.append({'trip_id': trip_id, 'file_type': 'arcs', 'severity': 'FATAL', 'rule': 'No Nulls', 'message': 'Found NaN/Null in arcs'})
        
        num_arcs = len(arcs_df)
        if num_arcs == 0:
            issues.append({'trip_id': trip_id, 'file_type': 'arcs', 'severity': 'FATAL', 'rule': 'Non-Empty', 'message': 'Arcs file is empty'})
        
        # Verify node indices
        if num_nodes > 0:
            invalid_from = ((arcs_df['from_node'] < 0) | (arcs_df['from_node'] >= num_nodes)).sum()
            invalid_to = ((arcs_df['to_node'] < 0) | (arcs_df['to_node'] >= num_nodes)).sum()
            if invalid_from > 0 or invalid_to > 0:
                issues.append({
                    'trip_id': trip_id, 'file_type': 'arcs', 'severity': 'FATAL',
                    'rule': 'Node Reference Integrity',
                    'message': f"Invalid node indices in arcs: {invalid_from} invalid from_nodes, {invalid_to} invalid to_nodes"
                })
        
        # Self-loops & duplicates warnings
        self_loops = int((arcs_df['from_node'] == arcs_df['to_node']).sum())
        if self_loops > 0:
            warnings.append({'trip_id': trip_id, 'file_type': 'arcs', 'severity': 'WARNING', 'rule': 'Self-Loops', 'message': f"{self_loops} self-loop arc(s) detected"})
        
        dup_arcs = int(arcs_df.duplicated().sum())
        if dup_arcs > 0:
            warnings.append({'trip_id': trip_id, 'file_type': 'arcs', 'severity': 'WARNING', 'rule': 'Duplicate Arcs', 'message': f"{dup_arcs} duplicate arc(s) detected"})

    except Exception as e:
        issues.append({'trip_id': trip_id, 'file_type': 'arcs', 'severity': 'FATAL', 'rule': 'Parse Error', 'message': str(e)})
        arcs_df = pd.DataFrame()
        num_arcs = 0

    # 6. Parse .route
    try:
        route_df = pd.read_csv(
            required_files['route'], sep=r'\s+', header=None,
            names=['arc_id'], dtype={'arc_id': int}
        )
        if route_df.isna().sum().sum() > 0:
            issues.append({'trip_id': trip_id, 'file_type': 'route', 'severity': 'FATAL', 'rule': 'No Nulls', 'message': 'Found NaN/Null in route'})
        
        route_len = len(route_df)
        if route_len == 0:
            issues.append({'trip_id': trip_id, 'file_type': 'route', 'severity': 'FATAL', 'rule': 'Non-Empty', 'message': 'Route file is empty'})
        
        # Verify arc indices
        if num_arcs > 0:
            invalid_route_arcs = ((route_df['arc_id'] < 0) | (route_df['arc_id'] >= num_arcs)).sum()
            if invalid_route_arcs > 0:
                issues.append({
                    'trip_id': trip_id, 'file_type': 'route', 'severity': 'FATAL',
                    'rule': 'Arc Reference Integrity',
                    'message': f"Invalid arc indices in route: {invalid_route_arcs} invalid arc_ids"
                })

        # Check path continuity
        if not arcs_df.empty and route_len > 1:
            route_arcs = route_df['arc_id'].values
            disc_count = 0
            for i in range(len(route_arcs) - 1):
                if arcs_df.loc[route_arcs[i], 'to_node'] != arcs_df.loc[route_arcs[i + 1], 'from_node']:
                    disc_count += 1
            if disc_count > 0:
                msg = f"{disc_count} disconnected transition(s) in reference route"
                if strict_route_continuity:
                    issues.append({'trip_id': trip_id, 'file_type': 'route', 'severity': 'FATAL', 'rule': 'Route Continuity', 'message': msg})
                else:
                    warnings.append({'trip_id': trip_id, 'file_type': 'route', 'severity': 'WARNING', 'rule': 'Route Continuity', 'message': msg})

    except Exception as e:
        issues.append({'trip_id': trip_id, 'file_type': 'route', 'severity': 'FATAL', 'rule': 'Parse Error', 'message': str(e)})

    is_valid = len(issues) == 0
    return {
        'trip_id': trip_id,
        'is_valid': is_valid,
        'gps_observations': len(track_df) if not track_df.empty else 0,
        'num_nodes': num_nodes,
        'num_arcs': num_arcs,
        'route_length': len(route_df) if 'route_df' in locals() and not route_df.empty else 0,
        'issues': issues,
        'warnings': warnings
    }


def validate_all(
    data_dir: str = "Student folder",
    output_dir: str = "results/preprocessing"
) -> bool:
    """
    Validate all trips across train and student_test partitions.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Validating dataset in '{data_dir}'...")

    splits = ['train', 'student_test']
    all_results = []
    all_issues = []
    all_warnings = []

    for split in splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            continue
        
        trip_ids = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
        for trip_id in trip_ids:
            tdir = os.path.join(split_path, trip_id)
            res = validate_trip(tdir, trip_id)
            res['split'] = split
            all_results.append({
                'split': split,
                'trip_id': trip_id,
                'is_valid': res['is_valid'],
                'gps_observations': res['gps_observations'],
                'road_nodes': res['num_nodes'],
                'road_arcs': res['num_arcs'],
                'route_length': res['route_length'],
                'issue_count': len(res['issues']),
                'warning_count': len(res['warnings'])
            })
            for iss in res['issues']:
                iss['split'] = split
                all_issues.append(iss)
            for w in res['warnings']:
                w['split'] = split
                all_warnings.append(w)

    val_df = pd.DataFrame(all_results)
    val_csv = os.path.join(output_dir, "validation_summary.csv")
    val_df.to_csv(val_csv, index=False)
    logger.info(f"Saved validation summary to '{val_csv}'")

    report_json = {
        'total_trips_validated': len(val_df),
        'valid_trips_count': int(val_df['is_valid'].sum()),
        'invalid_trips_count': int((~val_df['is_valid']).sum()),
        'total_issues': len(all_issues),
        'total_warnings': len(all_warnings),
        'issues': all_issues,
        'warnings': all_warnings
    }
    json_path = os.path.join(output_dir, "validation_report.json")
    with open(json_path, 'w') as f:
        json.dump(report_json, f, indent=2)
    logger.info(f"Saved detailed validation report to '{json_path}'")

    all_passed = (len(all_issues) == 0)
    if all_passed:
        logger.info(f"ALL {len(val_df)} TRIPS PASSED VALIDATION (100% Schema & Constraint Compliance).")
    else:
        logger.error(f"Validation FAILED: {len(all_issues)} critical issues found.")

    return all_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PathSnap Dataset Validation Suite")
    parser.add_argument("--data-dir", default="Student folder", help="Path to raw dataset")
    parser.add_argument("--output-dir", default="results/preprocessing", help="Output directory for validation reports")
    args = parser.parse_args()

    success = validate_all(args.data_dir, args.output_dir)
    sys.exit(0 if success else 1)
