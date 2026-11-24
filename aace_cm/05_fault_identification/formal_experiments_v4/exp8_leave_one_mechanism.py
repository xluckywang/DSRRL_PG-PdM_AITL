"""R2-9: controlled leave-one-mechanism rejection on the frozen Exp8 date split."""

from __future__ import annotations

import argparse
from hashlib import file_digest
from importlib import import_module, metadata as package_metadata
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


EXP8 = import_module("44_aace_cm_exp8_core_prototype_assignment")
SOURCE = (
    Path(__file__).resolve().parent / "outputs" / "aace_cm_exp8_unknown_modes"
    / "formal_20260725_183900" / "representation"
)
SPLITS = ("core_discovery", "label_free_validation", "frozen_query")
MECHANISMS = tuple(EXP8.CLASS_ORDER[1:])
SEED = 42
PCA_DIM = 24
ALPHA = 0.05
STABILITY_ROUNDS = 20
METHOD_FILES = {
    "TFRC": "aace_exp8_residual_features.npy",
    "TemporalAE residual": "aace_exp8_ae_residual_features.npy",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train(matrix: np.ndarray, fit: np.ndarray, seed: int, solver: str):
    # Pin the solver used by the archived model: newer sklearn changed the auto dispatch.
    values = matrix[fit]
    scaler = StandardScaler().fit(values)
    pca = PCA(n_components=PCA_DIM, whiten=True, svd_solver=solver, random_state=seed)
    pca.fit(scaler.transform(values))
    block = EXP8.BlockModel("compact", scaler, pca)
    model = EXP8.RepresentationModel("compact", ("compact",), {"compact": block})
    fit_values = model.transform({"compact": matrix[fit]})
    labels = EXP8.fit_cluster(fit_values, "kmeans", len(MECHANISMS) - 1, 0, seed)
    ids, centers = EXP8.prototypes(fit_values, labels)
    require(len(ids) == len(MECHANISMS) - 1, "Unexpected number of prototypes")
    return model, fit_values, labels, ids, centers


def assign(model, matrix: np.ndarray, indices: np.ndarray, ids, centers):
    embedded = model.transform({"compact": matrix[indices]})
    return EXP8.nearest_prototype(embedded, ids, centers)[:2]


def validation_threshold(distances: np.ndarray) -> float:
    # Order statistic for a nominal 5% validation rejection target, not a test guarantee.
    rank = min(len(distances), math.ceil((len(distances) + 1) * (1 - ALPHA)))
    return float(np.partition(distances, rank - 1)[rank - 1])


def read_labels(path: Path, uids: list[str], columns: list[str], engine: str) -> pd.DataFrame:
    options = {"row_filter": True} if engine == "fastparquet" else {}
    return pd.read_parquet(
        path, columns=columns, engine=engine, filters=[("scenario_uid", "in", uids)], **options
    ).set_index("scenario_uid", verify_integrity=True).loc[uids].reset_index()


def summarize(group: pd.DataFrame) -> dict:
    unknown = group["is_unknown"].to_numpy(bool)
    reject = group["rejected"].to_numpy(bool)
    correct = group["prediction"].eq(group["posthoc_class"]).to_numpy(bool)
    known = ~unknown
    distance = group["nearest_distance"]
    return {
        "unknown_count": int(unknown.sum()),
        "known_count": int(known.sum()),
        "unknown_rejected": int((unknown & reject).sum()),
        "known_rejected": int((known & reject).sum()),
        "known_correct": int((known & correct).sum()),
        "unknown_recall": float(reject[unknown].mean()),
        "known_false_reject_rate": float(reject[known].mean()),
        "known_correct_assignment_rate": float(correct[known].mean()),
        "unknown_precision": float(unknown[reject].mean()) if reject.any() else None,
        "acceptance_rate": float((~reject).mean()),
        "unknown_auroc": float(roc_auc_score(unknown, distance)),
        "unknown_average_precision": float(average_precision_score(unknown, distance)),
    }


def run(source: Path, output: Path, engine: str, solver: str) -> None:
    manifest_path = source / "aace_exp8_representation_manifest.json"
    audit_path = source / "formal_audit.json"
    for path in (manifest_path, audit_path):
        require(json.loads(path.read_text(encoding="utf-8"))["passed"], f"Failed audit: {path}")
    metadata_path = source / "aace_exp8_anonymous_metadata.parquet"
    labels_path = source / "SEALED_aace_exp8_posthoc_labels.parquet"
    keys = ["scenario_uid", "date", "split", "routed_to_exp8"]
    meta = pd.read_parquet(metadata_path, columns=keys, engine=engine)
    csv_meta = pd.read_csv(source / "aace_exp8_anonymous_metadata.csv", usecols=keys)
    pd.testing.assert_frame_equal(meta, csv_meta[keys], check_dtype=False)
    require(meta["scenario_uid"].is_unique, "Duplicate scenario IDs")
    dates = meta["date"].to_numpy()
    indices = {
        split: np.flatnonzero(meta["routed_to_exp8"] & meta["split"].eq(split))
        for split in SPLITS
    }
    date_sets = {split: set(dates[rows]) for split, rows in indices.items()}
    require([len(indices[s]) for s in SPLITS] == [3400, 1200, 1200], "Unexpected split sizes")
    require([len(date_sets[s]) for s in SPLITS] == [17, 6, 6], "Unexpected date counts")
    require(all(not (a & b) for a, b in combinations(date_sets.values(), 2)), "Dates overlap")
    require(meta["routed_to_exp8"].sum() == 5800, "Unexpected routing count")
    discovery, validation, query = (indices[s] for s in SPLITS)
    dev = np.concatenate([discovery, validation])

    # The organizer uses only development labels to remove the held-out mechanism.
    # No test labels are passed to the fit, calibration, prediction or stability stages.
    label_columns = ["scenario_uid", "posthoc_class"]
    dev_labels = read_labels(labels_path, meta.iloc[dev]["scenario_uid"].tolist(), label_columns, engine)
    dev_truth = np.full(len(meta), "", dtype=object)
    dev_truth[dev] = dev_labels["posthoc_class"].to_numpy()
    for rows, count in ((discovery, 850), (validation, 300)):
        require(
            pd.Series(dev_truth[rows]).value_counts().to_dict() == dict.fromkeys(MECHANISMS, count),
            "Unexpected development mechanism counts",
        )
    compact_columns = (
        np.arange(EXP8.RESIDUAL_AGGREGATION_BLOCKS)[:, None] * EXP8.RESIDUAL_LOCAL_WIDTH
        + np.arange(EXP8.COMPACT_LOCAL_WIDTH)
    ).ravel()
    matrices = {}
    for method, filename in METHOD_FILES.items():
        matrix = np.load(source / filename, allow_pickle=False)
        require(matrix.shape == (len(meta), 406), f"Invalid matrix shape for {method}")
        matrices[method] = matrix[:, compact_columns]
        require(np.isfinite(matrices[method]).all(), f"Non-finite features for {method}")

    output.mkdir(parents=True, exist_ok=False)
    model_records, scores, splits, stability = [], [], [], []
    stability_predictions = {}
    for held_out in MECHANISMS:
        fit = discovery[dev_truth[discovery] != held_out]
        calibrate = validation[dev_truth[validation] != held_out]
        require((len(fit), len(calibrate)) == (2550, 900), "Invalid leave-out split")
        require(not np.isin(held_out, dev_truth[np.r_[fit, calibrate]]), "Held-out class in development")
        for role, rows in (
            ("fit", fit), ("calibrate", calibrate), ("query", query),
            ("excluded_development", dev[dev_truth[dev] == held_out]),
        ):
            frame = meta.iloc[rows][["scenario_uid", "date", "split"]].copy()
            frame["held_out"] = held_out
            frame["role"] = role
            frame["development_class"] = dev_truth[rows]
            splits.append(frame)
        fit_dates = np.unique(dates[fit])
        rng = np.random.default_rng(SEED)
        subsamples = [
            rng.choice(fit_dates, size=math.ceil(0.8 * len(fit_dates)), replace=False)
            for _ in range(STABILITY_ROUNDS)
        ]
        for method, matrix in matrices.items():
            model, fit_values, labels, ids, centers = train(matrix, fit, SEED, solver)
            names, evidence = EXP8.semantic_mapping(fit_values, labels, ids, centers, dev_truth[fit], 1)
            _, calibration_scores = assign(model, matrix, calibrate, ids, centers)
            threshold = validation_threshold(calibration_scores)
            model_records.append({
                "held_out": held_out,
                "method": method,
                "fit_count": len(fit),
                "calibration_count": len(calibrate),
                "threshold": threshold,
                "validation_rejected": int((calibration_scores > threshold).sum()),
                "naming": [
                    {**row, "representative_uid": meta.iloc[fit[row["representative_indices"][0]]]["scenario_uid"]}
                    for row in evidence
                ],
            })
            for role, rows in (("calibrate", calibrate), ("query", query)):
                assigned, distance = assign(model, matrix, rows, ids, centers)
                predicted = np.array([names[int(cluster)] for cluster in assigned])
                rejected = distance > threshold
                frame = meta.iloc[rows][["scenario_uid", "date"]].copy()
                frame["role"] = role
                frame["held_out"] = held_out
                frame["method"] = method
                frame["nearest_mode"] = assigned
                frame["nearest_distance"] = distance
                frame["rejected"] = rejected
                frame["prediction"] = np.where(rejected, "unknown", predicted)
                scores.append(frame)
            assignments = []
            for repeat, selected in enumerate(subsamples):
                subset = fit[np.isin(dates[fit], selected)]
                fitted, _, _, cluster_ids, prototypes = train(matrix, subset, SEED, solver)
                modes, _ = assign(fitted, matrix, query, cluster_ids, prototypes)
                assignments.append(modes)
                stability.append({
                    "held_out": held_out, "method": method, "repeat": repeat,
                    "fit_dates": ";".join(sorted(selected)), "fit_records": len(subset),
                })
            stability_predictions[(held_out, method)] = assignments
        print(f"Frozen predictions for {held_out}", flush=True)

    splits = pd.concat(splits, ignore_index=True)
    scores = pd.concat(scores, ignore_index=True)
    scores.to_csv(output / "frozen_scores.csv", index=False)
    splits.to_csv(output / "split_manifest.csv", index=False)
    pd.DataFrame(stability).to_csv(output / "stability_subsamples.csv", index=False)
    pd.DataFrame(
        {f"{held_out}|{method}|{repeat}": modes
         for (held_out, method), assignments in stability_predictions.items()
         for repeat, modes in enumerate(assignments)},
        index=pd.Index(meta.iloc[query]["scenario_uid"], name="scenario_uid"),
    ).to_csv(output / "stability_assignments.csv")
    write_json(output / "freeze_manifest.json", {
        "test_labels_joined": False,
        "development_labels_used_for": ["mechanism exclusion", "one representative per fitted mode"],
        "source": str(source),
        "source_sha256": {
            path.name: sha256(path) for path in (
                manifest_path, audit_path, metadata_path, labels_path,
                *(source / filename for filename in METHOD_FILES.values()),
            )
        },
        "code_sha256": {Path(p).name: sha256(Path(p)) for p in (__file__, EXP8.__file__)},
        "packages": {name: package_metadata.version(name) for name in (
            "numpy", "pandas", "scipy", "scikit-learn", engine,
        )},
        "date_splits": {name: sorted(values) for name, values in date_sets.items()},
        "fixed_configuration": {
            "input_dim": 182, "pca_dim": PCA_DIM, "pca_solver": solver,
            "kmeans_modes": 3, "seed": SEED,
        },
        "stability": {"rounds": STABILITY_ROUNDS, "fraction_of_fit_dates": 0.8, "seed": SEED},
        "validation_target_false_reject_rate": ALPHA,
        "rejection_rule": "distance > validation order statistic ceil((n+1)*(1-alpha))",
        "frozen_artifacts_sha256": {
            name: sha256(output / name) for name in (
                "frozen_scores.csv", "split_manifest.csv",
                "stability_subsamples.csv", "stability_assignments.csv",
            )
        },
        "models": model_records,
    })

    # Test labels are joined only for scoring; no fit or threshold changes follow.
    test_uids = meta.iloc[query]["scenario_uid"].tolist()
    test_labels = read_labels(
        labels_path, test_uids, label_columns + ["severity_level", "inject_fraction"], engine
    )
    require(test_labels["posthoc_class"].value_counts().to_dict() == dict.fromkeys(MECHANISMS, 300),
            "Unexpected query mechanism counts")
    predictions = scores.loc[scores["role"].eq("query")].merge(
        test_labels, on="scenario_uid", how="left", validate="many_to_one"
    )
    require(predictions["posthoc_class"].notna().all(), "Query labels did not align")
    predictions["is_unknown"] = predictions["posthoc_class"].eq(predictions["held_out"])
    predictions.to_csv(output / "test_predictions.csv", index=False)
    summaries, per_date, severity_rows = [], [], []
    test_truth = test_labels["posthoc_class"].to_numpy()
    for (held_out, method), group in predictions.groupby(["held_out", "method"], sort=False):
        key = {"held_out": held_out, "method": method}
        known_mask = test_truth != held_out
        values = stability_predictions[(held_out, method)]
        aris = np.array([
            adjusted_rand_score(a[known_mask], b[known_mask]) for a, b in combinations(values, 2)
        ])
        summaries.append({
            **key, **summarize(group), "stability_mean_ari": aris.mean(),
            "stability_median_ari": np.median(aris), "stability_min_ari": aris.min(),
        })
        for date, subset in group.groupby("date"):
            per_date.append({**key, "date": date, **summarize(subset)})
        for severity, subset in group.groupby("severity_level"):
            severity_rows.append({**key, "severity_level": severity, **summarize(subset)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(per_date).to_csv(output / "per_date.csv", index=False)
    pd.DataFrame(severity_rows).to_csv(output / "per_severity.csv", index=False)
    print(summary[[
        "held_out", "method", "unknown_recall", "known_false_reject_rate",
        "known_correct_assignment_rate", "unknown_auroc", "stability_mean_ari",
    ]].round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parquet-engine", choices=["fastparquet", "pyarrow"], default="fastparquet")
    parser.add_argument("--pca-solver", choices=["randomized", "auto"], default="randomized")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.source_dir.resolve(), args.output_dir.resolve(), args.parquet_engine, args.pca_solver)
