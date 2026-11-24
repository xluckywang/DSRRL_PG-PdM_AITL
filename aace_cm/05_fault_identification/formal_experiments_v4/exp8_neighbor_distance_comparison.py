"""Compare prototype and local-neighborhood rejection on the frozen Exp8 splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

import exp8_leave_one_mechanism as base


K = 10
PROTOTYPE = "nearest_prototype"
NEIGHBOR = "mean_10nn"


def checked_input(archive: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((archive / "freeze_manifest.json").read_text(encoding="utf-8"))
    for group, directory in (
        ("frozen_artifacts_sha256", archive),
        ("source_sha256", base.SOURCE),
        ("code_sha256", Path(__file__).resolve().parent),
    ):
        for name, expected in manifest[group].items():
            base.require(base.sha256(directory / name) == expected, f"Hash mismatch: {name}")
    base.require(
        manifest["fixed_configuration"] == {
            "input_dim": 182, "pca_dim": 24, "pca_solver": "randomized",
            "kmeans_modes": 3, "seed": base.SEED,
        },
        "Expected the original randomized-PCA run",
    )
    base.require(
        manifest["validation_target_false_reject_rate"] == base.ALPHA,
        "The original validation rejection target differs",
    )
    splits = pd.read_csv(archive / "split_manifest.csv", keep_default_na=False)
    scores = pd.read_csv(archive / "frozen_scores.csv").query("method == 'TFRC'")
    return manifest, splits, scores


def make_scores(archive: Path, output: Path) -> None:
    original, splits, archived_scores = checked_input(archive)
    metadata = pd.read_parquet(
        base.SOURCE / "aace_exp8_anonymous_metadata.parquet",
        engine="fastparquet", columns=["scenario_uid", "date"],
    ).set_index("scenario_uid", verify_integrity=True)
    row_index = pd.Series(np.arange(len(metadata)), index=metadata.index)
    compact = (
        np.arange(base.EXP8.RESIDUAL_AGGREGATION_BLOCKS)[:, None] * base.EXP8.RESIDUAL_LOCAL_WIDTH
        + np.arange(base.EXP8.COMPACT_LOCAL_WIDTH)
    ).ravel()
    matrix = np.load(
        base.SOURCE / base.METHOD_FILES["TFRC"], mmap_mode="r", allow_pickle=False
    )[:, compact]
    output.mkdir(parents=True, exist_ok=False)
    frozen_scores, thresholds, geometry = [], [], []
    for held_out, split in splits.groupby("held_out", sort=False):
        fit = split.loc[split["role"].eq("fit")]
        cal = split.loc[split["role"].eq("calibrate")]
        query = split.loc[split["role"].eq("query")]
        dates = [set(frame["date"]) for frame in (fit, cal, query)]
        base.require(split["scenario_uid"].is_unique, f"Duplicate scenario: {held_out}")
        base.require(
            list(map(len, (fit, cal, query))) == [2550, 900, 1200]
            and list(map(len, dates)) == [17, 6, 6]
            and not (dates[0] & dates[1] or dates[0] & dates[2] or dates[1] & dates[2])
            and not any(frame["development_class"].eq(held_out).any() for frame in (fit, cal))
            and query["development_class"].eq("").all(),
            f"Invalid leave-one split: {held_out}",
        )
        model, fit_values, fit_labels, ids, centers = base.train(
            matrix, row_index.loc[fit["scenario_uid"]].to_numpy(), base.SEED, "randomized"
        )
        names, _ = base.EXP8.semantic_mapping(
            fit_values, fit_labels, ids, centers, fit["development_class"].to_numpy(), 1
        )
        neighbors = NearestNeighbors(n_neighbors=K, metric="euclidean", algorithm="brute").fit(fit_values)
        # Same center-to-center Euclidean calculation as Fig. 20(c), on known modes only.
        center_distances = cdist(centers, centers, metric="euclidean")
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                geometry.append({
                    "held_out": held_out, "mode_i": int(ids[i]), "mode_j": int(ids[j]),
                    "name_i": names[int(ids[i])], "name_j": names[int(ids[j])],
                    "distance": center_distances[i, j],
                })
        old = archived_scores.loc[archived_scores["held_out"].eq(held_out)]
        old_threshold = next(
            row["threshold"] for row in original["models"]
            if row["held_out"] == held_out and row["method"] == "TFRC"
        )
        fold_thresholds = {}
        for role, rows in (("calibrate", cal), ("query", query)):
            uids = rows["scenario_uid"]
            embedded = model.transform({"compact": matrix[row_index.loc[uids].to_numpy()]})
            mode, distance, _, _ = base.EXP8.nearest_prototype(embedded, ids, centers)
            local = neighbors.kneighbors(embedded, return_distance=True)[0].mean(axis=1)
            engineering_name = np.array([names[int(m)] for m in mode])
            old_rows = old.loc[old["role"].eq(role)].set_index("scenario_uid").loc[uids]
            np.testing.assert_array_equal(mode, old_rows["nearest_mode"])
            np.testing.assert_allclose(distance, old_rows["nearest_distance"], atol=2e-7, rtol=0)
            for method, values in ((PROTOTYPE, distance), (NEIGHBOR, local)):
                if role == "calibrate":
                    threshold = base.validation_threshold(values)
                    fold_thresholds[method] = threshold
                    thresholds.append({
                        "held_out": held_out, "score": method, "threshold": threshold,
                        "validation_count": len(values),
                        "validation_rejected": int((values > threshold).sum()),
                    })
                    if method == PROTOTYPE:
                        np.testing.assert_allclose(threshold, old_threshold, atol=1e-7, rtol=0)
                rejected = values > fold_thresholds[method]
                prediction = np.where(rejected, "unknown", engineering_name)
                if method == PROTOTYPE:
                    np.testing.assert_array_equal(rejected, old_rows["rejected"])
                    np.testing.assert_array_equal(prediction, old_rows["prediction"])
                frame = rows[["scenario_uid", "date", "role"]].copy()
                frame["held_out"] = held_out
                frame["score"] = method
                frame["nearest_mode"] = mode
                frame["nearest_distance"] = values
                frame["rejected"] = rejected
                frame["prediction"] = prediction
                frozen_scores.append(frame)
        print(f"Verified prototype replay and froze {held_out}", flush=True)

    scores = pd.concat(frozen_scores, ignore_index=True)
    scores.to_csv(output / "frozen_scores.csv", index=False)
    pd.DataFrame(thresholds).to_csv(output / "validation_thresholds.csv", index=False)
    pd.DataFrame(geometry).to_csv(output / "known_prototype_distances.csv", index=False)
    base.write_json(output / "freeze_manifest.json", {
        "original_run": str(archive),
        "original_manifest_sha256": base.sha256(archive / "freeze_manifest.json"),
        "code_sha256": base.sha256(Path(__file__)),
        "k": K,
        "space": "original 24-D normalized whitened PCA space",
        "neighbor_reference": "2550 known fitting scenarios only, Euclidean, mean of 10 nearest",
        "validation_target_false_reject_rate": base.ALPHA,
        "known_assignment": "same nearest prototype and sparse naming for both scores",
        "test_labels_joined": False,
        "frozen_artifacts_sha256": {
            name: base.sha256(output / name) for name in (
                "frozen_scores.csv", "validation_thresholds.csv", "known_prototype_distances.csv",
            )
        },
    })

    # The original labeled predictions are consulted only after the new scores and thresholds are frozen.
    old_truth = pd.read_csv(
        archive / "test_predictions.csv",
        usecols=["scenario_uid", "held_out", "method", "posthoc_class", "severity_level"],
    ).query("method == 'TFRC'").drop(columns="method")
    query = scores.loc[scores["role"].eq("query")].merge(
        old_truth, on=["scenario_uid", "held_out"], validate="many_to_one", how="left"
    )
    base.require(query["posthoc_class"].notna().all() and len(query) == 9600, "Invalid query join")
    query["is_unknown"] = query["posthoc_class"].eq(query["held_out"])
    query.to_csv(output / "test_predictions.csv", index=False)
    summary, dates = [], []
    for (held_out, method), group in query.groupby(["held_out", "score"], sort=False):
        base.require(int(group["is_unknown"].sum()) == 300, "Unexpected held-out class count")
        header = {"held_out": held_out, "score": method}
        summary.append({**header, **base.summarize(group)})
        for date, subset in group.groupby("date"):
            dates.append({**header, "date": date, **base.summarize(subset)})
    summary = pd.DataFrame(summary)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(dates).to_csv(output / "per_date.csv", index=False)
    print(summary[[
        "held_out", "score", "unknown_rejected", "known_rejected",
        "known_correct", "unknown_auroc",
    ]].round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        make_scores(args.original_run.resolve(), args.output_dir.resolve())
