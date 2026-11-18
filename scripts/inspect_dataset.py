"""Inspect the complete raw PathSnap dataset and generate dataset_inventory.csv.

Outputs:
    results/dataset_inventory.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.preprocessing.loader import load_trip


def inspect_all_trips(raw_base_dir: Path) -> pd.DataFrame:
    """Scan all partitions and trips to compile an exhaustive dataset inventory."""
    inventory_records: List[Dict[str, Any]] = []

    partitions = ["train", "student_test", "evaluator_test"]

    for partition in partitions:
        part_dir = raw_base_dir / partition
        if not part_dir.is_dir():
            print(f"[Inventory] Partition directory not found (isolated/held-out): {part_dir}")
            continue

        trip_dirs = sorted([p for p in part_dir.iterdir() if p.is_dir()])
        print(f"[Inventory] Scanning partition '{partition}' ({len(trip_dirs)} trips found)...")

        for t_dir in trip_dirs:
            trip_id = t_dir.name
            track_file = t_dir / f"{trip_id}.track"
            nodes_file = t_dir / f"{trip_id}.nodes"
            arcs_file = t_dir / f"{trip_id}.arcs"
            route_file = t_dir / f"{trip_id}.route"

            has_track = track_file.is_file()
            has_nodes = nodes_file.is_file()
            has_arcs = arcs_file.is_file()
            has_route = route_file.is_file()

            track_bytes = track_file.stat().st_size if has_track else 0
            nodes_bytes = nodes_file.stat().st_size if has_nodes else 0
            arcs_bytes = arcs_file.stat().st_size if has_arcs else 0
            route_bytes = route_file.stat().st_size if has_route else 0

            status = "VALID"
            err_msg = ""
            n_obs = n_nodes = n_arcs = n_route = 0

            try:
                trip = load_trip(t_dir, trip_id=trip_id, partition=partition, validate=True)
                n_obs = trip.num_observations
                n_nodes = trip.num_nodes
                n_arcs = trip.num_arcs
                n_route = trip.num_route_arcs
            except Exception as e:
                status = "LOAD_ERROR"
                err_msg = str(e)

            inventory_records.append({
                "partition": partition,
                "trip_id": trip_id,
                "has_track": has_track,
                "has_nodes": has_nodes,
                "has_arcs": has_arcs,
                "has_route": has_route,
                "track_size_bytes": track_bytes,
                "nodes_size_bytes": nodes_bytes,
                "arcs_size_bytes": arcs_bytes,
                "route_size_bytes": route_bytes,
                "total_size_bytes": track_bytes + nodes_bytes + arcs_bytes + route_bytes,
                "num_gps_observations": n_obs,
                "num_road_nodes": n_nodes,
                "num_road_arcs": n_arcs,
                "num_route_arcs": n_route,
                "processing_status": status,
                "error_notes": err_msg,
            })

    df = pd.DataFrame(inventory_records)
    return df


def main():
    raw_dir = REPO_ROOT / "data" / "raw" / "Student folder"
    if not raw_dir.is_dir():
        # Fallback to data/raw if Student folder is not nested
        raw_dir = REPO_ROOT / "data" / "raw"

    print(f"Inspecting raw dataset at: {raw_dir}")
    df_inv = inspect_all_trips(raw_dir)

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "dataset_inventory.csv"
    df_inv.to_csv(out_csv, index=False)
    print(f"Dataset inventory saved to: {out_csv} ({len(df_inv)} trips recorded)")

    # Print summary counts
    print("\n--- INVENTORY SUMMARY ---")
    for part, group in df_inv.groupby("partition"):
        print(f"Partition: {part:<15} | Trips: {len(group):<4} | Total GPS Obs: {group['num_gps_observations'].sum():<8} | Arcs: {group['num_road_arcs'].sum():<8}")


if __name__ == "__main__":
    main()
