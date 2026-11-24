"""Compare fixed 10-NN scores before and after the archived TFRC transforms.

This is a development-only diagnostic. Each original validation date is held
out from calibration in turn; the original test dates are never scored here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

import exp8_leave_one_mechanism as base
from exp8_neighbor_distance_comparison import K


def score_stages(model, values):
    block = model.blocks["compact"]
    standardized = block.scaler.transform(values)
    whitened = block.pca.transform(standardized)
    projected = whitened * np.sqrt(block.pca.explained_variance_)
    np.testing.assert_allclose(
        projected,
        (standardized - block.pca.mean_) @ block.pca.components_.T,
        atol=5e-5, rtol=1e-4,
    )
    return {
        "standardized_182d": standardized,
        "projected_24d": projected,
        "whitened_24d": whitened,
        "normalized_24d": model.transform({"compact": values}),
    }


def run(archive: Path, output: Path) -> None:
    manifest_path = archive / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_path = archive / "split_manifest.csv"
    base.require(
        base.sha256(split_path) == manifest["frozen_artifacts_sha256"][split_path.name],
        "Unexpected original split file",
    )
    for name, digest in manifest["source_sha256"].items():
        base.require(base.sha256(base.SOURCE / name) == digest, f"Input differs: {name}")
    for name, digest in manifest["code_sha256"].items():
        base.require(base.sha256(Path(__file__).with_name(name)) == digest, f"Base code differs: {name}")
    base.require(
        manifest["fixed_configuration"] == {
            "input_dim": 182, "pca_dim": 24, "pca_solver": "randomized",
            "kmeans_modes": 3, "seed": base.SEED,
        } and manifest["validation_target_false_reject_rate"] == base.ALPHA,
        "Unexpected frozen configuration",
    )
    splits = pd.read_csv(split_path, keep_default_na=False)
    meta = pd.read_parquet(
        base.SOURCE / "aace_exp8_anonymous_metadata.parquet", engine="fastparquet",
        columns=["scenario_uid", "date", "split"],
    ).set_index("scenario_uid", verify_integrity=True)
    row_id = pd.Series(np.arange(len(meta)), index=meta.index)
    compact = (
        np.arange(base.EXP8.RESIDUAL_AGGREGATION_BLOCKS)[:, None] * base.EXP8.RESIDUAL_LOCAL_WIDTH
        + np.arange(base.EXP8.COMPACT_LOCAL_WIDTH)
    ).ravel()
    matrix = np.load(base.SOURCE / base.METHOD_FILES["TFRC"], mmap_mode="r", allow_pickle=False)
    predictions, thresholds = [], []
    for held_out, fold in splits.groupby("held_out", sort=False):
        fit = fold.loc[fold["role"].eq("fit")]
        dev = fold.loc[fold["split"].eq("label_free_validation")].copy()
        fit_dates, dev_dates, test_dates = (
            set(fit["date"]), set(dev["date"]), set(fold.loc[fold["role"].eq("query"), "date"])
        )
        base.require(
            (len(fit), len(dev), len(fit_dates), len(dev_dates)) == (2550, 1200, 17, 6)
            and fold["scenario_uid"].is_unique
            and not (fit_dates & dev_dates or test_dates & (fit_dates | dev_dates))
            and not fit["development_class"].eq(held_out).any()
            and dev["development_class"].eq(held_out).equals(dev["role"].eq("excluded_development"))
            and meta.loc[fit["scenario_uid"], "split"].eq("core_discovery").all()
            and meta.loc[dev["scenario_uid"], "split"].eq("label_free_validation").all(),
            f"Invalid development split: {held_out}",
        )
        # Slice just the two development partitions before fitting or scoring.
        fit_rows = row_id.loc[fit["scenario_uid"]].to_numpy()
        dev_rows = row_id.loc[dev["scenario_uid"]].to_numpy()
        fit_matrix = np.ascontiguousarray(matrix[fit_rows][:, compact])
        dev_matrix = np.ascontiguousarray(matrix[dev_rows][:, compact])
        model, archived_fit, *_ = base.train(fit_matrix, np.arange(len(fit)), base.SEED, "randomized")
        fit_spaces = score_stages(model, fit_matrix)
        dev_spaces = score_stages(model, dev_matrix)
        np.testing.assert_array_equal(fit_spaces["normalized_24d"], archived_fit)
        dev["is_unknown"] = dev["development_class"].eq(held_out)
        for stage, features in fit_spaces.items():
            knn = NearestNeighbors(n_neighbors=K, metric="euclidean", algorithm="brute").fit(features)
            distance = knn.kneighbors(dev_spaces[stage], return_distance=True)[0].mean(axis=1)
            for date in sorted(dev_dates):
                query = dev["date"].eq(date)
                calibration = distance[~query & ~dev["is_unknown"]]
                base.require(
                    (query.sum(), dev.loc[query, "is_unknown"].sum(), len(calibration)) == (200, 50, 750),
                    f"Invalid query/calibration count: {held_out}, {date}",
                )
                threshold = base.validation_threshold(calibration)
                thresholds.append({
                    "held_out": held_out, "stage": stage, "query_date": date,
                    "calibration_dates": "|".join(sorted(dev_dates - {date})),
                    "calibration_count": len(calibration), "threshold": threshold,
                    "calibration_rejected": int((calibration > threshold).sum()),
                })
                rows = dev.loc[query, ["scenario_uid", "date", "is_unknown", "development_class"]].copy()
                rows["held_out"], rows["stage"] = held_out, stage
                rows["distance"], rows["threshold"] = distance[query], threshold
                rows["rejected"] = rows["distance"] > threshold
                predictions.append(rows)

    frozen = pd.concat(predictions, ignore_index=True)
    keys = ["held_out", "stage", "scenario_uid"]
    base.require(len(frozen) == 19200 and not frozen.duplicated(keys).any(), "Incomplete frozen scores")
    output.mkdir(parents=True, exist_ok=False)
    frozen.to_csv(output / "frozen_development_scores.csv", index=False)
    pd.DataFrame(thresholds).to_csv(output / "development_thresholds.csv", index=False)
    # Severity annotates the scores after all models and thresholds have been frozen.
    severity = base.read_labels(
        base.SOURCE / "SEALED_aace_exp8_posthoc_labels.parquet", frozen["scenario_uid"].unique().tolist(),
        ["scenario_uid", "severity_level"], "fastparquet",
    )
    scored = frozen.merge(severity, on="scenario_uid", validate="many_to_one")
    rows, by_date = [], []
    for (held_out, stage), group in scored.groupby(["held_out", "stage"], sort=False):
        for subset, levels in (
            ("all_1_to_10", range(1, 11)), ("low_1_to_3", range(1, 4)),
            ("middle_4_to_7", range(4, 8)), ("high_8_to_10", range(8, 11)),
        ):
            for date, daily in group.groupby("date"):
                known = daily.loc[~daily["is_unknown"]]
                unknown = daily.loc[daily["is_unknown"] & daily["severity_level"].isin(levels)]
                compared = pd.concat([known, unknown])
                by_date.append({
                    "held_out": held_out, "stage": stage, "severity_group": subset, "date": date,
                    "unknown_count": len(unknown), "unknown_rejected": int(unknown["rejected"].sum()),
                    "known_count": len(known), "known_rejected": int(known["rejected"].sum()),
                    "unknown_auroc": roc_auc_score(compared["is_unknown"], compared["distance"]),
                })
            unknown = group.loc[group["is_unknown"] & group["severity_level"].isin(levels)]
            known = group.loc[~group["is_unknown"]]
            base.require(len(unknown) == 30 * len(levels) and len(known) == 900, "Unexpected severity counts")
            rows.append({
                "held_out": held_out, "stage": stage, "severity_group": subset,
                "unknown_count": len(unknown), "unknown_rejected": int(unknown["rejected"].sum()),
                "unknown_recall": float(unknown["rejected"].mean()),
                "known_count": len(known), "known_rejected": int(known["rejected"].sum()),
                "known_false_reject_rate": float(known["rejected"].mean()),
                "mean_daily_unknown_auroc": float(np.mean([item["unknown_auroc"] for item in by_date[-6:]])),
            })
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(by_date).to_csv(output / "per_date.csv", index=False)
    scored.to_csv(output / "development_predictions.csv", index=False)
    base.write_json(output / "manifest.json", {
        "purpose": "development-only diagnostic; not independent test evidence",
        "original_run": str(archive), "original_manifest_sha256": base.sha256(manifest_path),
        "script_sha256": base.sha256(Path(__file__)), "neighbors": K,
        "calibration": "other five original validation dates, known mechanisms only",
        "target_false_rejection": base.ALPHA, "fit": "17 discovery dates, known mechanisms only",
        "test_scored": False, "severity_use": "reporting only, added after score and threshold freeze",
        "outputs_sha256": {p.name: base.sha256(p) for p in sorted(output.glob("*.csv"))},
    })
    print(f"Verified {len(frozen)} development scores across four held-out mechanisms and four stages.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.original_run.resolve(), args.output_dir.resolve())
