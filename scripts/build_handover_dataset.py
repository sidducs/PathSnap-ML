"""Build Siddhu's ML Handover Dataset and Student Testing Dataset.

Generates:
    data/processed/training/candidate_features.parquet
    data/processed/training/feature_schema.json
    data/processed/training/alignment_audit.csv
    data/processed/training/trip_processing_summary.csv

    data/processed/student_testing/candidate_features.parquet
    data/processed/student_testing/trip_processing_summary.csv
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.candidates.candidate_generation import (
    RoadSegmentSpatialIndex,
    generate_candidates_for_trip,
)
from ml.candidates.route_alignment import align_trip_reference_route
from ml.features.feature_engineering import (
    compute_candidate_features,
    get_feature_schema,
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMNS,
)
from ml.preprocessing.graph import DirectedRoadGraph
from ml.preprocessing.loader import load_trip


def process_partition(
    partition: str,
    raw_dir: Path,
    output_dir: Path,
    candidate_radius_m: float = 60.0,
    max_candidates: int = 15,
    is_training: bool = True,
) -> None:
    """Process all trips in a partition and export candidate features Parquet."""
    part_dir = raw_dir / partition
    if not part_dir.is_dir():
        print(f"Partition directory not found: {part_dir}")
        return

    trip_dirs = sorted([p for p in part_dir.iterdir() if p.is_dir()])
    print(f"\n==========================================")
    print(f"Processing Partition: '{partition}' ({len(trip_dirs)} trips)")
    print(f"Output Directory: {output_dir}")
    print(f"Settings: radius={candidate_radius_m}m, max_candidates={max_candidates}")
    print(f"==========================================")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_feature_dfs: List[pd.DataFrame] = []
    alignment_audits: List[Dict[str, Any]] = []
    processing_summaries: List[Dict[str, Any]] = []

    start_time = time.time()

    for idx, t_dir in enumerate(trip_dirs, 1):
        t0 = time.time()
        trip_id = t_dir.name

        # 1. Load raw trip
        trip = load_trip(t_dir, trip_id=trip_id, partition=partition, validate=True)

        # 2. Construct directed road graph & spatial index
        graph = DirectedRoadGraph(trip.nodes, trip.arcs)
        spatial_index = RoadSegmentSpatialIndex(graph, sample_interval_m=25.0)

        # 3. Generate candidate road segments
        cands_df, cov_summary = generate_candidates_for_trip(
            trip,
            graph=graph,
            spatial_index=spatial_index,
            candidate_radius_m=candidate_radius_m,
            max_candidates=max_candidates,
        )

        # 4. Route alignment (for training partition, or student test without leaking to training)
        if is_training:
            labeled_cands_df, audit = align_trip_reference_route(trip, cands_df, graph=graph)
            alignment_audits.append(audit)
        else:
            # For student testing, we isolate labels from feature matrix
            labeled_cands_df = cands_df.copy()
            labeled_cands_df["label"] = np.nan
            labeled_cands_df["alignment_status"] = "test_evaluation_holdout"
            labeled_cands_df["route_index"] = -1

        # 5. Compute candidate features
        feat_df = compute_candidate_features(trip, labeled_cands_df, graph=graph)
        all_feature_dfs.append(feat_df)

        elapsed = time.time() - t0
        processing_summaries.append({
            "trip_id": trip_id,
            "partition": partition,
            "num_gps_observations": trip.num_observations,
            "num_road_nodes": trip.num_nodes,
            "num_road_arcs": trip.num_arcs,
            "num_candidate_rows": len(feat_df),
            "observations_covered": cov_summary["observations_with_candidates"],
            "observations_missed": cov_summary["observations_without_candidates"],
            "coverage_ratio": cov_summary["coverage_ratio"],
            "mean_candidates_per_obs": cov_summary["mean_candidates_per_obs"],
            "processing_time_seconds": elapsed,
        })

        if idx % 5 == 0 or idx == len(trip_dirs):
            print(f"[{partition.upper()}] Processed {idx}/{len(trip_dirs)} trips ({len(feat_df)} cands in trip {trip_id}, {elapsed:.2f}s)", flush=True)

    # Combine all feature DataFrames
    print(f"Concatenating {len(all_feature_dfs)} trip feature matrices...")
    master_df = pd.concat(all_feature_dfs, ignore_index=True)

    # Save primary Parquet
    parquet_path = output_dir / "candidate_features.parquet"
    print(f"Writing Parquet to: {parquet_path} ({len(master_df):,} rows)...")
    master_df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")
    print(f"Parquet saved successfully! Size: {parquet_path.stat().st_size / (1024*1024):.2f} MB")

    # Save summary CSV
    summary_csv = output_dir / "trip_processing_summary.csv"
    pd.DataFrame(processing_summaries).to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # For training partition, export feature schema and alignment audit
    if is_training:
        schema_path = output_dir / "feature_schema.json"
        with open(schema_path, "w") as f:
            json.dump(get_feature_schema(), f, indent=2)
        print(f"Saved: {schema_path}")

        audit_csv = output_dir / "alignment_audit.csv"
        pd.DataFrame(alignment_audits).to_csv(audit_csv, index=False)
        print(f"Saved: {audit_csv}")

    total_time = time.time() - start_time
    print(f"Partition '{partition}' completed in {total_time:.1f}s.")


def main():
    raw_dir = REPO_ROOT / "data" / "raw" / "Student folder"
    processed_dir = REPO_ROOT / "data" / "processed"

    # 1. Process 80 training trips
    train_out = processed_dir / "training"
    process_partition(
        partition="train",
        raw_dir=raw_dir,
        output_dir=train_out,
        candidate_radius_m=60.0,
        max_candidates=15,
        is_training=True,
    )

    # 2. Process 10 student-testing trips (separate holdout, no labels leaked)
    test_out = processed_dir / "student_testing"
    process_partition(
        partition="student_test",
        raw_dir=raw_dir,
        output_dir=test_out,
        candidate_radius_m=60.0,
        max_candidates=15,
        is_training=False,
    )


if __name__ == "__main__":
    main()
