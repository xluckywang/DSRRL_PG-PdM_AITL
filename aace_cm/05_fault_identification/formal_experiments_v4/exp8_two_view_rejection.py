"""R2-9 retrospective comparison: retain magnitude as well as direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from threadpoolctl import threadpool_limits

import exp8_leave_one_mechanism as base
from exp8_fused_distance_comparison import percentile
from exp8_neighbor_distance_comparison import checked_input, K, NEIGHBOR, PROTOTYPE
from exp8_representation_diagnostic import score_stages


MAGNITUDE = "whitened_24d_10nn"
JOINT = "two_view_max_percentile"
LOF = "normalized_24d_lof"


def run(archive: Path, previous: Path, output: Path) -> None:
    frozen, splits, prototype_archive = checked_input(archive)
    previous_manifest = json.loads((previous / "freeze_manifest.json").read_text(encoding="utf-8"))
    for name, digest in previous_manifest["frozen_artifacts_sha256"].items():
        base.require(base.sha256(previous / name) == digest, f"Frozen neighborhood output differs: {name}")
    original_manifest = archive / "freeze_manifest.json"
    base.require(
        base.sha256(original_manifest) == previous_manifest["original_manifest_sha256"]
        and previous_manifest["k"] == K and not previous_manifest["test_labels_joined"],
        "Unexpected neighborhood provenance",
    )
    neighbor_archive = pd.read_csv(previous / "frozen_scores.csv")
    meta = pd.read_parquet(
        base.SOURCE / "aace_exp8_anonymous_metadata.parquet", engine="fastparquet",
        columns=["scenario_uid", "date", "split"],
    ).set_index("scenario_uid", verify_integrity=True)
    index = pd.Series(np.arange(len(meta)), index=meta.index)
    compact = (
        np.arange(base.EXP8.RESIDUAL_AGGREGATION_BLOCKS)[:, None] * base.EXP8.RESIDUAL_LOCAL_WIDTH
        + np.arange(base.EXP8.COMPACT_LOCAL_WIDTH)
    ).ravel()
    matrix = np.load(base.SOURCE / base.METHOD_FILES["TFRC"], allow_pickle=False)[:, compact]
    all_scores, thresholds = [], []
    for held_out, fold in splits.groupby("held_out", sort=False):
        fit, cal, query = (fold.loc[fold["role"].eq(role)] for role in ("fit", "calibrate", "query"))
        dates = [set(rows["date"]) for rows in (fit, cal, query)]
        base.require(
            list(map(len, (fit, cal, query))) == [2550, 900, 1200]
            and list(map(len, dates)) == [17, 6, 6]
            and not (dates[0] & dates[1] or dates[0] & dates[2] or dates[1] & dates[2])
            and fold["scenario_uid"].is_unique
            and not pd.concat([fit, cal])["development_class"].eq(held_out).any()
            and query["development_class"].eq("").all(),
            f"Invalid fold: {held_out}",
        )
        fit_indices = index.loc[fit["scenario_uid"]].to_numpy()
        model, fit_values, fit_labels, ids, centers = base.train(matrix, fit_indices, base.SEED, "randomized")
        names, _ = base.EXP8.semantic_mapping(
            fit_values, fit_labels, ids, centers, fit["development_class"].to_numpy(), 1,
        )
        fit_spaces = score_stages(model, matrix[fit_indices])
        np.testing.assert_array_equal(fit_spaces["normalized_24d"], fit_values)
        near_direction = NearestNeighbors(n_neighbors=K, metric="euclidean", algorithm="brute").fit(fit_values)
        near_raw = NearestNeighbors(n_neighbors=K, metric="euclidean", algorithm="brute").fit(
            fit_spaces["whitened_24d"]
        )
        lof = LocalOutlierFactor(n_neighbors=K, metric="euclidean", algorithm="brute", novelty=True).fit(fit_values)
        evaluation = pd.concat([cal, query], ignore_index=True)[["scenario_uid", "date", "role"]]
        evaluation["held_out"] = held_out
        mask = evaluation["role"].eq("calibrate").to_numpy()
        uids = evaluation["scenario_uid"]
        spaces = score_stages(model, matrix[index.loc[uids].to_numpy()])
        unit_vectors = spaces["normalized_24d"]
        mode, prototype_distance, *_ = base.EXP8.nearest_prototype(unit_vectors, ids, centers)
        direction = near_direction.kneighbors(unit_vectors)[0].mean(axis=1)
        raw = near_raw.kneighbors(spaces["whitened_24d"])[0].mean(axis=1)
        joint = np.maximum(percentile(direction[mask], direction), percentile(raw[mask], raw))
        lof_distance = -lof.score_samples(unit_vectors)
        old_prototype = prototype_archive.loc[prototype_archive["held_out"].eq(held_out)].set_index("scenario_uid").loc[uids]
        old_nearest = neighbor_archive.loc[
            neighbor_archive["held_out"].eq(held_out) & neighbor_archive["score"].eq(NEIGHBOR)
        ].set_index("scenario_uid").loc[uids]
        for old, values in ((old_prototype, prototype_distance), (old_nearest, direction)):
            np.testing.assert_array_equal(evaluation["role"], old["role"])
            np.testing.assert_array_equal(mode, old["nearest_mode"])
            np.testing.assert_allclose(values, old["nearest_distance"], atol=2e-7, rtol=0)

        for method, values in (
            (PROTOTYPE, prototype_distance), (NEIGHBOR, direction), (MAGNITUDE, raw), (JOINT, joint), (LOF, lof_distance),
        ):
            base.require(np.isfinite(values).all(), f"Non-finite score: {held_out} / {method}")
            threshold = base.validation_threshold(values[mask])
            reject = values > threshold
            base.require(reject[mask].sum() <= 44, "Exceeded the shared validation rejection budget")
            records = evaluation.assign(score=method, nearest_mode=mode, nearest_distance=values, rejected=reject)
            records["prediction"] = np.where(reject, "unknown", records["nearest_mode"].map(names))
            if method in (PROTOTYPE, NEIGHBOR):
                old = old_prototype if method == PROTOTYPE else old_nearest
                np.testing.assert_array_equal(reject, old["rejected"])
                np.testing.assert_array_equal(records["prediction"], old["prediction"])
            all_scores.append(records)
            thresholds.append({
                "held_out": held_out, "score": method, "threshold": threshold,
                "calibration_count": int(mask.sum()), "calibration_rejected": int(reject[mask].sum()),
            })

    scores = pd.concat(all_scores, ignore_index=True)
    output.mkdir(parents=True, exist_ok=False)
    scores.to_csv(output / "frozen_scores.csv", index=False)
    pd.DataFrame(thresholds).to_csv(output / "validation_thresholds.csv", index=False)
    base.write_json(output / "freeze_manifest.json", {
        "original_archive": str(archive), "original_manifest_sha256": base.sha256(original_manifest),
        "neighborhood_archive": str(previous),
        "neighborhood_manifest_sha256": base.sha256(previous / "freeze_manifest.json"),
        "script_sha256": base.sha256(Path(__file__)),
        "scope": "retrospective comparison on existing Exp8 test dates; test labels have been examined in earlier analyses",
        "retained_magnitude_view": "same 24 whitened PCA components, before row-wise L2 normalization",
        "direction_view": "same 24 whitened PCA components, after row-wise L2 normalization",
        "joint": "max of the two empirical percentiles, both mapped using known validation only",
        "calibration": "shared nominal 5% budget on known validation only, with strictly greater-than rejection",
        "lof": "novelty mode, normalized 24-D input, Euclidean, 10 neighbors, higher = more novel",
        "known_assignment": "unchanged nearest prototype, one representative per mode",
        "test_labels_joined_before_freeze": False,
        "outputs_sha256": {p.name: base.sha256(p) for p in sorted(output.glob("*.csv"))},
    })

    test = scores.loc[scores["role"].eq("query")].copy()
    truth = base.read_labels(
        base.SOURCE / "SEALED_aace_exp8_posthoc_labels.parquet", test["scenario_uid"].unique().tolist(),
        ["scenario_uid", "posthoc_class", "severity_level"], "fastparquet",
    )
    test = test.merge(truth, on="scenario_uid", validate="many_to_one", how="left")
    test["is_unknown"] = test["held_out"].eq(test["posthoc_class"])
    base.require(len(test) == 24000 and test["posthoc_class"].notna().all(), "Invalid test join")
    test.to_csv(output / "test_predictions.csv", index=False)
    totals, per_date, high = [], [], []
    for (held_out, score), group in test.groupby(["held_out", "score"], sort=False):
        base.require(len(group) == 1200 and group["is_unknown"].sum() == 300, "Invalid test counts")
        key = {"held_out": held_out, "score": score}
        totals.append({**key, **base.summarize(group)})
        for date, rows in group.groupby("date"):
            per_date.append({**key, "date": date, **base.summarize(rows)})
        severe = group.loc[group["is_unknown"] & group["severity_level"].between(8, 10)]
        base.require(len(severe) == 90, "Unexpected high-severity sample count")
        high.append({**key, "severity_min": 8, "severity_max": 10,
                     "unknown_count": len(severe), "unknown_rejected": int(severe["rejected"].sum())})
    summary = pd.DataFrame(totals)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(per_date).to_csv(output / "per_date.csv", index=False)
    pd.DataFrame(high).to_csv(output / "high_severity.csv", index=False)
    print(summary[["held_out", "score", "unknown_rejected", "known_rejected", "known_correct", "unknown_auroc"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--neighbor-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.original_run.resolve(), args.neighbor_run.resolve(), args.output_dir.resolve())
