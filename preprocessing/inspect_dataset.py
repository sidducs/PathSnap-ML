"""
PathSnap: Dataset Inspection & Exploratory Data Analysis (EDA)
--------------------------------------------------------------
Author: Senior Geospatial & ML Data Engineer
Project: PES University ML Hackathon - PathSnap

This module performs comprehensive, reproducible exploratory data analysis
across all dataset partitions (train and student_test), validating schema,
file integrity, GPS observations, road network graph topology, timestamp
distributions, and ground-truth reference routes.
"""

import os
import sys
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('PathSnap-EDA')


def haversine_np(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """
    Calculate great-circle distances between two points on Earth using Haversine formula.
    Inputs in degrees, returns distances in meters.
    """
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi * 0.5) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda * 0.5) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def bbox_area_km2(min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> float:
    """Approximate bounding box area in square kilometers."""
    width_m = haversine_np(np.array([min_lon]), np.array([min_lat]), np.array([max_lon]), np.array([min_lat]))[0]
    height_m = haversine_np(np.array([min_lon]), np.array([min_lat]), np.array([min_lon]), np.array([max_lat]))[0]
    return (width_m * height_m) / 1e6


def run_eda(
    data_dir: str = "Student folder",
    output_dir: str = "results/eda"
) -> Dict[str, Any]:
    """
    Execute end-to-end dataset inspection across all partitions.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting Dataset Exploration from '{data_dir}'...")
    logger.info(f"EDA Reports will be saved to '{output_dir}'")

    splits = ['train', 'student_test']
    all_trips_data = []
    missing_value_records = []
    timestamp_gap_records = []
    anomalies_records = []
    network_metrics_records = []
    all_time_deltas = []
    all_speeds_kmh = []

    for split in splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            logger.warning(f"Split directory {split_path} not found. Skipping.")
            continue
        
        trip_ids = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
        logger.info(f"Found {len(trip_ids)} trips in partition '{split}'")

        for trip_id in trip_ids:
            tdir = os.path.join(split_path, trip_id)
            track_file = os.path.join(tdir, f"{trip_id}.track")
            nodes_file = os.path.join(tdir, f"{trip_id}.nodes")
            arcs_file = os.path.join(tdir, f"{trip_id}.arcs")
            route_file = os.path.join(tdir, f"{trip_id}.route")

            # 1. File existence & size
            files_dict = {
                'track': track_file,
                'nodes': nodes_file,
                'arcs': arcs_file,
                'route': route_file
            }
            file_sizes = {}
            for k, fpath in files_dict.items():
                if os.path.exists(fpath):
                    file_sizes[k] = os.path.getsize(fpath) / 1024.0  # in KB
                else:
                    file_sizes[k] = 0.0
                    missing_value_records.append({
                        'split': split, 'trip_id': trip_id, 'file': k,
                        'issue': 'File Missing', 'details': f"{fpath} does not exist"
                    })

            # Read track
            track_df = pd.read_csv(track_file, sep=r'\s+', header=None, names=['lon', 'lat', 'time'], dtype={'lon': float, 'lat': float, 'time': float})
            # Read nodes
            nodes_df = pd.read_csv(nodes_file, sep=r'\s+', header=None, names=['lon', 'lat'], dtype={'lon': float, 'lat': float})
            # Read arcs
            arcs_df = pd.read_csv(arcs_file, sep=r'\s+', header=None, names=['from_node', 'to_node'], dtype={'from_node': int, 'to_node': int})
            # Read route
            route_df = pd.read_csv(route_file, sep=r'\s+', header=None, names=['arc_id'], dtype={'arc_id': int})

            # 2. Missing/Null checks
            for name, df in [('track', track_df), ('nodes', nodes_df), ('arcs', arcs_df), ('route', route_df)]:
                null_counts = df.isna().sum()
                for col, count in null_counts.items():
                    if count > 0:
                        missing_value_records.append({
                            'split': split, 'trip_id': trip_id, 'file': name,
                            'column': col, 'issue': 'Null values', 'count': int(count)
                        })

            # 3. GPS track metrics
            num_obs = len(track_df)
            track_min_lon, track_max_lon = track_df['lon'].min(), track_df['lon'].max()
            track_min_lat, track_max_lat = track_df['lat'].min(), track_df['lat'].max()
            track_bbox_area = bbox_area_km2(track_min_lon, track_max_lon, track_min_lat, track_max_lat)

            # Timestamps
            dt_series = track_df['time'].diff().dropna()
            all_time_deltas.extend(dt_series.values)
            non_monotonic = int((dt_series <= 0).sum())
            if non_monotonic > 0:
                anomalies_records.append({
                    'split': split, 'trip_id': trip_id, 'category': 'Non-Monotonic Timestamps',
                    'count': non_monotonic, 'details': 'One or more timestamps are non-increasing'
                })

            pct_1s = float((dt_series == 1.0).sum() / len(dt_series) * 100.0) if len(dt_series) > 0 else 0.0
            max_dt = float(dt_series.max()) if len(dt_series) > 0 else 0.0
            gaps_gt_5s = int((dt_series > 5.0).sum())

            # Detailed gaps audit
            for idx, dt_val in dt_series.items():
                if dt_val > 5.0:
                    t_prev = track_df.loc[idx - 1, 'time']
                    t_curr = track_df.loc[idx, 'time']
                    dist_m = haversine_np(
                        np.array([track_df.loc[idx - 1, 'lon']]), np.array([track_df.loc[idx - 1, 'lat']]),
                        np.array([track_df.loc[idx, 'lon']]), np.array([track_df.loc[idx, 'lat']])
                    )[0]
                    timestamp_gap_records.append({
                        'split': split, 'trip_id': trip_id,
                        'point_index': int(idx),
                        'time_prev_s': float(t_prev), 'time_curr_s': float(t_curr),
                        'gap_seconds': float(dt_val), 'distance_m': float(dist_m)
                    })
                    if dt_val > 60.0:
                        anomalies_records.append({
                            'split': split, 'trip_id': trip_id, 'category': 'Large Time Gap (>60s)',
                            'count': 1, 'details': f"Gap of {dt_val:.1f}s at index {idx}, distance moved: {dist_m:.1f}m"
                        })

            # Physical distance and speed
            dists = haversine_np(
                track_df['lon'].shift().values[1:], track_df['lat'].shift().values[1:],
                track_df['lon'].values[1:], track_df['lat'].values[1:]
            )
            total_distance_km = float(np.sum(dists) / 1000.0)
            speeds_mps = np.divide(dists, dt_series.values, out=np.zeros_like(dists), where=dt_series.values > 0)
            speeds_kmh = speeds_mps * 3.6
            all_speeds_kmh.extend(speeds_kmh)
            mean_speed_kmh = float(np.mean(speeds_kmh)) if len(speeds_kmh) > 0 else 0.0
            max_speed_kmh = float(np.max(speeds_kmh)) if len(speeds_kmh) > 0 else 0.0

            # Extreme speed outliers (> 150 km/h)
            high_speed_mask = speeds_kmh > 150.0
            if np.any(high_speed_mask):
                high_speed_indices = np.where(high_speed_mask)[0] + 1
                for h_idx in high_speed_indices:
                    anomalies_records.append({
                        'split': split, 'trip_id': trip_id, 'category': 'Extreme Speed Outlier (>150 km/h)',
                        'count': 1,
                        'details': f"Index {h_idx}: speed {speeds_kmh[h_idx-1]:.1f} km/h, dist {dists[h_idx-1]:.1f}m in {dt_series.iloc[h_idx-1]:.1f}s"
                    })

            # Stationary observations (consecutive dist < 0.5m)
            stationary_pts = int(np.sum(dists < 0.5))

            # 4. Road Network Validation & Graph Topology
            num_nodes = len(nodes_df)
            num_arcs = len(arcs_df)

            nodes_min_lon, nodes_max_lon = nodes_df['lon'].min(), nodes_df['lon'].max()
            nodes_min_lat, nodes_max_lat = nodes_df['lat'].min(), nodes_df['lat'].max()
            nodes_bbox_area = bbox_area_km2(nodes_min_lon, nodes_max_lon, nodes_min_lat, nodes_max_lat)

            # Node bounds check
            track_in_nodes = bool(
                track_min_lon >= nodes_min_lon and track_max_lon <= nodes_max_lon and
                track_min_lat >= nodes_min_lat and track_max_lat <= nodes_max_lat
            )

            # Arc reference check
            invalid_from = int(((arcs_df['from_node'] < 0) | (arcs_df['from_node'] >= num_nodes)).sum())
            invalid_to = int(((arcs_df['to_node'] < 0) | (arcs_df['to_node'] >= num_nodes)).sum())
            self_loops = int((arcs_df['from_node'] == arcs_df['to_node']).sum())
            dup_arcs = int(arcs_df.duplicated().sum())

            # NetworkX Graph construction for topology metrics
            G = nx.DiGraph()
            G.add_nodes_from(range(num_nodes))
            G.add_edges_from(zip(arcs_df['from_node'], arcs_df['to_node']))
            
            isolated_nodes = int(len(list(nx.isolates(G))))
            num_wcc = int(nx.number_weakly_connected_components(G))
            num_scc = int(nx.number_strongly_connected_components(G))

            network_metrics_records.append({
                'split': split, 'trip_id': trip_id,
                'num_nodes': num_nodes, 'num_arcs': num_arcs,
                'unique_edges': G.number_of_edges(),
                'isolated_nodes': isolated_nodes,
                'self_loops': self_loops,
                'duplicate_arcs': dup_arcs,
                'weakly_connected_components': num_wcc,
                'strongly_connected_components': num_scc,
                'network_bbox_area_km2': nodes_bbox_area
            })

            # 5. Reference Route Validation
            route_length = len(route_df)
            invalid_route_arcs = int(((route_df['arc_id'] < 0) | (route_df['arc_id'] >= num_arcs)).sum())

            route_arcs = route_df['arc_id'].values
            route_disc_transitions = 0
            for r_i in range(len(route_arcs) - 1):
                curr_to = arcs_df.loc[route_arcs[r_i], 'to_node']
                next_from = arcs_df.loc[route_arcs[r_i + 1], 'from_node']
                if curr_to != next_from:
                    route_disc_transitions += 1
                    anomalies_records.append({
                        'split': split, 'trip_id': trip_id, 'category': 'Disconnected Route Transition',
                        'count': 1,
                        'details': f"Step {r_i}->{r_i+1}: arc {route_arcs[r_i]} (to_node {curr_to}) -> arc {route_arcs[r_i+1]} (from_node {next_from})"
                    })

            # Calculate reference route physical length in km
            route_from_nodes = arcs_df.loc[route_arcs, 'from_node'].values
            route_to_nodes = arcs_df.loc[route_arcs, 'to_node'].values
            route_arc_dists = haversine_np(
                nodes_df.loc[route_from_nodes, 'lon'].values,
                nodes_df.loc[route_from_nodes, 'lat'].values,
                nodes_df.loc[route_to_nodes, 'lon'].values,
                nodes_df.loc[route_to_nodes, 'lat'].values
            )
            route_distance_km = float(np.sum(route_arc_dists) / 1000.0)

            # Compile trip record
            duration_s = float(track_df['time'].iloc[-1] - track_df['time'].iloc[0]) if num_obs > 1 else 0.0
            all_trips_data.append({
                'split': split,
                'trip_id': trip_id,
                'gps_observations': num_obs,
                'duration_seconds': duration_s,
                'duration_hours': duration_s / 3600.0,
                'total_distance_km': total_distance_km,
                'mean_speed_kmh': mean_speed_kmh,
                'max_speed_kmh': max_speed_kmh,
                'stationary_observations': stationary_pts,
                'pct_1s_intervals': pct_1s,
                'max_gap_seconds': max_dt,
                'gaps_gt_5s': gaps_gt_5s,
                'track_min_lon': track_min_lon, 'track_max_lon': track_max_lon,
                'track_min_lat': track_min_lat, 'track_max_lat': track_max_lat,
                'track_bbox_area_km2': track_bbox_area,
                'road_nodes': num_nodes,
                'road_arcs': num_arcs,
                'isolated_nodes': isolated_nodes,
                'wcc_components': num_wcc,
                'scc_components': num_scc,
                'self_loop_arcs': self_loops,
                'duplicate_arcs': dup_arcs,
                'track_within_network_bbox': track_in_nodes,
                'route_arc_count': route_length,
                'route_distance_km': route_distance_km,
                'route_disconnected_transitions': route_disc_transitions,
                'invalid_node_refs': invalid_from + invalid_to,
                'invalid_route_refs': invalid_route_arcs,
                'track_size_kb': file_sizes['track'],
                'nodes_size_kb': file_sizes['nodes'],
                'arcs_size_kb': file_sizes['arcs'],
                'route_size_kb': file_sizes['route'],
                'total_files_size_kb': sum(file_sizes.values())
            })

    trips_df = pd.DataFrame(all_trips_data)
    trips_csv_path = os.path.join(output_dir, "trip_level_statistics.csv")
    trips_df.to_csv(trips_csv_path, index=False)
    logger.info(f"Saved trip-level statistics to '{trips_csv_path}'")

    # 6. Save Missing Value Report
    missing_df = pd.DataFrame(missing_value_records)
    if missing_df.empty:
        missing_df = pd.DataFrame([{'status': 'Clean', 'message': 'Zero missing, null, or corrupted files/values found across all 90 trips'}])
    missing_csv_path = os.path.join(output_dir, "missing_value_report.csv")
    missing_df.to_csv(missing_csv_path, index=False)
    logger.info(f"Saved missing-value report to '{missing_csv_path}'")

    # 7. Save Timestamp Gap Analysis
    gap_df = pd.DataFrame(timestamp_gap_records)
    gap_csv_path = os.path.join(output_dir, "timestamp_gap_analysis.csv")
    gap_df.to_csv(gap_csv_path, index=False)
    logger.info(f"Saved timestamp-gap analysis to '{gap_csv_path}' ({len(gap_df)} gaps > 5s recorded)")

    # 8. Save Road Network Connectivity
    net_df = pd.DataFrame(network_metrics_records)
    net_csv_path = os.path.join(output_dir, "road_network_connectivity.csv")
    net_df.to_csv(net_csv_path, index=False)
    logger.info(f"Saved road network connectivity metrics to '{net_csv_path}'")

    # 9. Save Trajectory Anomalies
    anomalies_df = pd.DataFrame(anomalies_records)
    anomalies_csv_path = os.path.join(output_dir, "trajectory_anomalies.csv")
    anomalies_df.to_csv(anomalies_csv_path, index=False)
    logger.info(f"Saved trajectory anomalies to '{anomalies_csv_path}' ({len(anomalies_df)} anomalies found)")

    # 10. Generate Consolidated Dataset Summary
    summary_rows = []
    for split_label in ['train', 'student_test', 'all']:
        sub_df = trips_df if split_label == 'all' else trips_df[trips_df['split'] == split_label]
        summary_rows.append({
            'partition': split_label,
            'trip_count': len(sub_df),
            'total_gps_observations': int(sub_df['gps_observations'].sum()),
            'min_gps_obs': int(sub_df['gps_observations'].min()),
            'median_gps_obs': float(sub_df['gps_observations'].median()),
            'mean_gps_obs': float(sub_df['gps_observations'].mean()),
            'max_gps_obs': int(sub_df['gps_observations'].max()),
            'total_road_nodes': int(sub_df['road_nodes'].sum()),
            'mean_road_nodes': float(sub_df['road_nodes'].mean()),
            'total_road_arcs': int(sub_df['road_arcs'].sum()),
            'mean_road_arcs': float(sub_df['road_arcs'].mean()),
            'total_route_arcs': int(sub_df['route_arc_count'].sum()),
            'mean_route_arcs': float(sub_df['route_arc_count'].mean()),
            'total_distance_km': float(sub_df['total_distance_km'].sum()),
            'mean_trip_distance_km': float(sub_df['total_distance_km'].mean()),
            'total_duration_hours': float(sub_df['duration_hours'].sum()),
            'mean_trip_duration_min': float(sub_df['duration_seconds'].mean() / 60.0),
            'pct_1s_intervals_mean': float(sub_df['pct_1s_intervals'].mean()),
            'trips_with_gaps_gt_5s': int((sub_df['gaps_gt_5s'] > 0).sum()),
            'total_gaps_gt_5s': int(sub_df['gaps_gt_5s'].sum()),
            'max_gap_seconds': float(sub_df['max_gap_seconds'].max()),
            'total_self_loop_arcs': int(sub_df['self_loop_arcs'].sum()),
            'total_duplicate_arcs': int(sub_df['duplicate_arcs'].sum()),
            'trips_with_route_disc': int((sub_df['route_disconnected_transitions'] > 0).sum()),
            'total_route_disc_transitions': int(sub_df['route_disconnected_transitions'].sum()),
            'total_dataset_size_mb': float(sub_df['total_files_size_kb'].sum() / 1024.0)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, "dataset_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved dataset summary to '{summary_csv_path}'")

    # 11. Generate Publication-Quality EDA Visualizations
    generate_eda_plots(trips_df, all_time_deltas, all_speeds_kmh, output_dir)

    logger.info("Exploratory Data Analysis Completed Successfully.")
    return {
        'summary': summary_df.to_dict(orient='records'),
        'total_trips': len(trips_df),
        'anomalies_count': len(anomalies_df)
    }


def generate_eda_plots(
    trips_df: pd.DataFrame,
    time_deltas: List[float],
    speeds_kmh: List[float],
    output_dir: str
):
    """Generate comprehensive visual plots for EDA reports."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # Plot 1: Key Distributions Overview (4 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1.1 GPS observations distribution
    sns.histplot(data=trips_df, x='gps_observations', hue='split', kde=True, ax=axes[0, 0], bins=25)
    axes[0, 0].set_title("GPS Observations per Trip", fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel("Number of GPS Observations")
    axes[0, 0].set_ylabel("Trip Count")

    # 1.2 Road nodes distribution
    sns.histplot(data=trips_df, x='road_nodes', hue='split', kde=True, ax=axes[0, 1], bins=25)
    axes[0, 1].set_title("Road Network Nodes per Trip", fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel("Number of Nodes")
    axes[0, 1].set_ylabel("Trip Count")

    # 1.3 Route arc count distribution
    sns.histplot(data=trips_df, x='route_arc_count', hue='split', kde=True, ax=axes[1, 0], bins=25)
    axes[1, 0].set_title("Reference Route Traversed Arcs per Trip", fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel("Number of Route Arcs")
    axes[1, 0].set_ylabel("Trip Count")

    # 1.4 Trip Distance distribution
    sns.histplot(data=trips_df, x='total_distance_km', hue='split', kde=True, ax=axes[1, 1], bins=25)
    axes[1, 1].set_title("Total Trajectory Distance (km)", fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel("Distance (km)")
    axes[1, 1].set_ylabel("Trip Count")

    plt.tight_layout()
    p1_path = os.path.join(output_dir, "eda_overview_distributions.png")
    plt.savefig(p1_path, dpi=300)
    plt.close()
    logger.info(f"Saved EDA Overview Distributions plot to '{p1_path}'")

    # Plot 2: Timestamp Sampling Interval (Delta t) Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    dt_arr = np.array(time_deltas)
    
    # Left: Near 1s zoom (0.5 to 5.0 seconds)
    zoom_dt = dt_arr[(dt_arr >= 0.5) & (dt_arr <= 5.0)]
    sns.histplot(zoom_dt, bins=45, ax=ax1, color="#2b5c8f")
    ax1.set_title("GPS Sampling Delta t (Zoomed 0.5s - 5.0s)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Sampling Interval Delta t (seconds)")
    ax1.set_ylabel("Observation Count")
    ax1.axvline(1.0, color='red', linestyle='--', label=f'Nominal 1.0s ({np.mean(dt_arr == 1.0)*100:.1f}%)')
    ax1.legend()

    # Right: Boxplot of Gaps > 5s (log scale)
    gaps_gt5 = dt_arr[dt_arr > 5.0]
    if len(gaps_gt5) > 0:
        sns.boxplot(x=gaps_gt5, ax=ax2, color="#e67e22")
        ax2.set_xscale('log')
        ax2.set_title(f"Time Gaps > 5s Distribution (N={len(gaps_gt5)}, Log Scale)", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Time Gap (seconds, log scale)")
    
    plt.tight_layout()
    p2_path = os.path.join(output_dir, "timestamp_delta_distribution.png")
    plt.savefig(p2_path, dpi=300)
    plt.close()
    logger.info(f"Saved Timestamp Delta Distribution plot to '{p2_path}'")

    # Plot 3: Speed Distribution & Anomalies
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    spd_arr = np.array(speeds_kmh)
    
    # Left: Realistic speeds (0 to 140 km/h)
    valid_spd = spd_arr[(spd_arr >= 0.0) & (spd_arr <= 140.0)]
    sns.histplot(valid_spd, bins=50, ax=ax1, kde=True, color="#27ae60")
    ax1.set_title("Vehicle Speed Distribution (0 - 140 km/h)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Speed (km/h)")
    ax1.set_ylabel("Observation Count")
    ax1.axvline(np.median(valid_spd), color='black', linestyle=':', label=f'Median: {np.median(valid_spd):.1f} km/h')
    ax1.axvline(np.mean(valid_spd), color='blue', linestyle='--', label=f'Mean: {np.mean(valid_spd):.1f} km/h')
    ax1.legend()

    # Right: Speed Outliers (> 150 km/h)
    outlier_spd = spd_arr[spd_arr > 150.0]
    sns.stripplot(x=outlier_spd, ax=ax2, color="#c0392b", jitter=0.2, size=7)
    ax2.set_title(f"GPS Multipath / Jump Outliers (> 150 km/h, N={len(outlier_spd)})", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Outlier Speed (km/h)")

    plt.tight_layout()
    p3_path = os.path.join(output_dir, "speed_distribution.png")
    plt.savefig(p3_path, dpi=300)
    plt.close()
    logger.info(f"Saved Speed Distribution plot to '{p3_path}'")

    # Plot 4: Road Network Size & Graph Characteristics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.scatterplot(data=trips_df, x='road_nodes', y='road_arcs', hue='split', ax=ax1, alpha=0.8, s=60)
    ax1.set_title("Road Network Nodes vs. Directed Arcs", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Number of Nodes")
    ax1.set_ylabel("Number of Arcs")

    sns.scatterplot(data=trips_df, x='gps_observations', y='route_arc_count', hue='split', ax=ax2, alpha=0.8, s=60)
    ax2.set_title("GPS Observations vs. Reference Route Arcs", fontsize=12, fontweight='bold')
    ax2.set_xlabel("GPS Track Observations")
    ax2.set_ylabel("Reference Route Arcs")

    plt.tight_layout()
    p4_path = os.path.join(output_dir, "road_network_and_route_scaling.png")
    plt.savefig(p4_path, dpi=300)
    plt.close()
    logger.info(f"Saved Network Scaling plot to '{p4_path}'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PathSnap Dataset Inspection & EDA")
    parser.add_argument("--data-dir", default="Student folder", help="Path to raw dataset root folder")
    parser.add_argument("--output-dir", default="results/eda", help="Path to output directory for EDA reports")
    args = parser.parse_args()

    run_eda(data_dir=args.data_dir, output_dir=args.output_dir)
