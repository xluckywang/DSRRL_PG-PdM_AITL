"""Compare validation-calibrated distance fusion with both frozen Exp8 scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import exp8_leave_one_mechanism as base
from exp8_neighbor_distance_comparison import NEIGHBOR, PROTOTYPE


FUSION = "max_validation_percentile"
KEY = ["scenario_uid", "date", "role", "held_out", "nearest_mode"]


def percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(reference), values, side="right") / len(reference)


def run(source: Path, output: Path) -> None:
    manifest_path = source / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, digest in manifest["frozen_artifacts_sha256"].items():
        base.require(base.sha256(source / filename) == digest, f"Frozen input changed: {filename}")
    base.require(
        manifest["k"] == 10 and manifest["validation_target_false_reject_rate"] == base.ALPHA
        and manifest["test_labels_joined"] is False,
        "Unexpected distance-run protocol",
    )
    old_manifest_path = Path(manifest["original_run"]) / "freeze_manifest.json"
    base.require(
        base.sha256(old_manifest_path) == manifest["original_manifest_sha256"],
        "Original run manifest changed",
    )
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    model_names = {
        row["held_out"]: {item["cluster_id"]: item["assigned_engineering_name"] for item in row["naming"]}
        for row in old_manifest["models"] if row["method"] == "TFRC"
    }
    scores = pd.read_csv(source / "frozen_scores.csv")
    base.require(
        set(scores["held_out"]) == set(base.MECHANISMS)
        and set(scores["score"]) == {PROTOTYPE, NEIGHBOR},
        "Unexpected score groups",
    )
    fusion_rows, threshold_rows = [], []
    for held_out, fold in scores.groupby("held_out", sort=False):
        wide = fold.pivot(index=KEY, columns="score", values="nearest_distance").reset_index()
        validation = wide["role"].eq("calibrate")
        query = wide["role"].eq("query")
        base.require(
            len(wide) == 2100 and validation.sum() == 900 and query.sum() == 1200
            and wide["scenario_uid"].is_unique
            and not set(wide.loc[validation, "date"]) & set(wide.loc[query, "date"])
            and np.isfinite(wide[[PROTOTYPE, NEIGHBOR]].to_numpy()).all(),
            f"Invalid pairing or split: {held_out}",
        )
        for component in (PROTOTYPE, NEIGHBOR):
            wide[f"{component}_percentile"] = percentile(
                wide.loc[validation, component].to_numpy(), wide[component].to_numpy()
            )
        fused = wide[[f"{PROTOTYPE}_percentile", f"{NEIGHBOR}_percentile"]].max(axis=1).to_numpy()
        threshold = base.validation_threshold(fused[validation])
        rejected = fused > threshold
        validation_rejected = int(rejected[validation].sum())
        base.require(validation_rejected <= 44, f"Joint validation budget exceeded: {held_out}")
        threshold_rows.append({
            "held_out": held_out, "score": FUSION, "threshold": threshold,
            "validation_count": int(validation.sum()), "validation_rejected": validation_rejected,
        })
        known_name = wide["nearest_mode"].map(model_names[held_out])
        base.require(known_name.notna().all(), f"Missing mode name: {held_out}")
        wide["score"] = FUSION
        wide["nearest_distance"] = fused  # Common metric column; this score is a percentile, not a distance.
        wide["rejected"] = rejected
        wide["prediction"] = np.where(rejected, "unknown", known_name)
        fusion_rows.append(wide)
        print(f"Froze {held_out}: joint threshold {threshold:.6f}, validation rejected {validation_rejected}/900")

    output.mkdir(parents=True, exist_ok=False)
    fused_scores = pd.concat(fusion_rows, ignore_index=True)
    fused_scores.to_csv(output / "frozen_fusion_scores.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output / "validation_thresholds.csv", index=False)
    base.write_json(output / "freeze_manifest.json", {
        "distance_run": str(source),
        "distance_run_manifest_sha256": base.sha256(manifest_path),
        "script_sha256": base.sha256(Path(__file__)),
        "score": "max of the two within-fold empirical percentiles from known validation",
        "percentile": "number of known validation distances <= input distance / 900",
        "calibration": "joint max cutoff with the same 5% nominal validation rejection target; reject strictly above it",
        "known_assignment": "original nearest prototype and sparse naming",
        "test_labels_joined": False,
        "frozen_artifacts_sha256": {
            name: base.sha256(output / name) for name in (
                "frozen_fusion_scores.csv", "validation_thresholds.csv",
            )
        },
    })

    # Open the same official test labels only after the fused scores and cutoffs exist.
    query = pd.concat([scores, fused_scores[scores.columns]], ignore_index=True).query("role == 'query'")
    uid = query["scenario_uid"].drop_duplicates().tolist()
    labels_name = "SEALED_aace_exp8_posthoc_labels.parquet"
    labels_path = Path(old_manifest["source"]) / labels_name
    base.require(base.sha256(labels_path) == old_manifest["source_sha256"][labels_name], "Labels changed")
    truth = base.read_labels(labels_path, uid, ["scenario_uid", "posthoc_class", "severity_level"], "fastparquet")
    query = query.merge(truth, on="scenario_uid", validate="many_to_one", how="left")
    query["is_unknown"] = query["held_out"].eq(query["posthoc_class"])
    base.require(query["posthoc_class"].notna().all() and len(query) == 14400, "Invalid test join")
    query.to_csv(output / "test_predictions.csv", index=False)
    totals, dates = [], []
    for (held_out, score), group in query.groupby(["held_out", "score"], sort=False):
        base.require(len(group) == 1200 and group["is_unknown"].sum() == 300, "Invalid test counts")
        key = {"held_out": held_out, "score": score}
        totals.append({**key, **base.summarize(group)})
        for date, rows in group.groupby("date"):
            dates.append({**key, "date": date, **base.summarize(rows)})
    table = pd.DataFrame(totals)
    table.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(dates).to_csv(output / "per_date.csv", index=False)
    print(table[[
        "held_out", "score", "unknown_rejected", "known_rejected",
        "known_correct", "unknown_auroc",
    ]].round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.distance_run.resolve(), args.output_dir.resolve())
