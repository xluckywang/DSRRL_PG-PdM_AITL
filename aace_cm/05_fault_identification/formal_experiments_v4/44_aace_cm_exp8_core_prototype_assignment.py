"""AACE-CM Exp8: label-free core discovery and frozen-prototype assignment."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = (
    SCRIPT_DIR / "outputs" / "aace_cm_exp8_unknown_modes" / "representation"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "outputs" / "aace_cm_exp8_unknown_modes" / "formal"
)
CLASS_ORDER = [
    "background_control",
    "crack",
    "imbalance",
    "mechanical_looseness",
    "thermal",
]
RESIDUAL_LOCAL_WIDTH = 58
RESIDUAL_AGGREGATION_BLOCKS = 7
COMPACT_LOCAL_WIDTH = 26


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64))
    return (
        values
        / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    ).astype(np.float32)


@dataclass
class BlockModel:
    name: str
    scaler: StandardScaler
    pca: PCA

    def transform(self, values: np.ndarray) -> np.ndarray:
        return normalize_rows(self.pca.transform(self.scaler.transform(values)))


@dataclass
class RepresentationModel:
    name: str
    members: tuple[str, ...]
    blocks: dict[str, BlockModel]

    def transform(self, matrices: dict[str, np.ndarray]) -> np.ndarray:
        parts = [self.blocks[name].transform(matrices[name]) for name in self.members]
        return normalize_rows(np.concatenate(parts, axis=1))


def fit_representation(
    name: str,
    members: tuple[str, ...],
    matrices: dict[str, np.ndarray],
    fit_indices: np.ndarray,
    components: dict[str, int],
    seed: int,
) -> RepresentationModel:
    blocks = {}
    for member in members:
        values = matrices[member][fit_indices]
        scaler = StandardScaler().fit(values)
        scaled = scaler.transform(values)
        count = max(
            1,
            min(
                components[member],
                scaled.shape[1],
                max(1, scaled.shape[0] - 1),
            ),
        )
        pca = PCA(
            n_components=count,
            whiten=True,
            svd_solver="auto",
            random_state=seed,
        ).fit(scaled)
        blocks[member] = BlockModel(member, scaler, pca)
    return RepresentationModel(name, members, blocks)


def representation_specs() -> dict[str, tuple[str, ...]]:
    return {
        "ae_residual_only": ("ae_residual",),
        "paired_contrast_compact": ("residual_compact",),
        "paired_contrast_only": ("residual",),
        "paired_contrast_trajectory": ("residual", "trajectory"),
        "paired_contrast_temperature_coupling": (
            "residual",
            "temperature",
            "coupling",
        ),
        "paired_contrast_full": (
            "residual",
            "raw",
            "trajectory",
            "temperature",
            "coupling",
        ),
        "paired_contrast_plus_ae": (
            "residual",
            "trajectory",
            "temperature",
            "coupling",
            "ae_residual",
        ),
    }


def entropy_balance(labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    valid = labels >= 0
    _, counts = np.unique(labels[valid], return_counts=True)
    if len(counts) < 2:
        return 0.0
    probability = counts / counts.sum()
    return float(
        -np.sum(probability * np.log(np.maximum(probability, 1e-12)))
        / np.log(len(counts))
    )


def safe_silhouette(
    values: np.ndarray, labels: np.ndarray, seed: int
) -> float:
    labels = np.asarray(labels)
    valid = labels >= 0
    unique = np.unique(labels[valid])
    if valid.sum() < 3 or len(unique) < 2 or len(unique) >= valid.sum():
        return -1.0
    return float(
        silhouette_score(
            values[valid],
            labels[valid],
            sample_size=min(3000, int(valid.sum())),
            random_state=seed,
        )
    )


def cluster_geometry(
    values: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> dict:
    labels = np.asarray(labels)
    valid = labels >= 0
    clusters = len(np.unique(labels[valid]))
    coverage = float(valid.mean())
    if valid.any():
        _, counts = np.unique(labels[valid], return_counts=True)
        maximum = float(counts.max() / len(labels))
        minimum = float(counts.min() / len(labels))
    else:
        maximum, minimum = 1.0, 0.0
    return {
        "clusters": clusters,
        "coverage": coverage,
        "silhouette": safe_silhouette(values, labels, seed),
        "balance": entropy_balance(labels),
        "maximum_cluster_fraction": maximum,
        "minimum_cluster_fraction": minimum,
    }


def fit_cluster(
    values: np.ndarray,
    algorithm: str,
    parameter_1: int,
    parameter_2: int,
    seed: int,
) -> np.ndarray:
    if algorithm == "kmeans":
        return KMeans(
            n_clusters=parameter_1,
            n_init=30,
            random_state=seed,
        ).fit_predict(values)
    return HDBSCAN(
        min_cluster_size=parameter_1,
        min_samples=parameter_2,
        metric="euclidean",
        cluster_selection_method="eom",
        allow_single_cluster=False,
        n_jobs=-1,
    ).fit_predict(values)


def prototypes(
    values: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    cluster_ids = np.unique(labels[labels >= 0])
    if not len(cluster_ids):
        return cluster_ids.astype(int), np.empty((0, values.shape[1]))
    centers = np.stack(
        [values[labels == cluster].mean(axis=0) for cluster in cluster_ids]
    )
    return cluster_ids.astype(int), normalize_rows(centers)


def nearest_prototype(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not len(centers):
        raise RuntimeError("No core prototypes were available")
    distance = np.linalg.norm(
        values[:, None, :] - centers[None, :, :], axis=2
    )
    order = np.argsort(distance, axis=1)
    nearest = order[:, 0]
    second = order[:, 1] if distance.shape[1] > 1 else order[:, 0]
    first_distance = distance[np.arange(len(values)), nearest]
    second_distance = distance[np.arange(len(values)), second]
    return (
        cluster_ids[nearest].astype(int),
        first_distance.astype(np.float32),
        second_distance.astype(np.float32),
        (second_distance - first_distance).astype(np.float32),
    )


def grouped_stability(
    values: np.ndarray,
    dates: np.ndarray,
    algorithm: str,
    parameter_fraction: float,
    parameter_1: int,
    parameter_2: int,
    rounds: int,
    seed: int,
) -> float:
    unique_dates = np.unique(dates)
    if rounds < 2 or len(unique_dates) < 3:
        return 1.0
    rng = np.random.default_rng(seed)
    assignments = []
    for round_index in range(rounds):
        selected_dates = rng.choice(
            unique_dates,
            size=max(2, int(math.ceil(0.80 * len(unique_dates)))),
            replace=False,
        )
        mask = np.isin(dates, selected_dates)
        subset = values[mask]
        if algorithm == "hdbscan":
            local_parameter = max(
                2, int(math.ceil(parameter_fraction * len(subset)))
            )
        else:
            local_parameter = parameter_1
        if len(subset) <= local_parameter:
            return 0.0
        try:
            labels = fit_cluster(
                subset,
                algorithm,
                local_parameter,
                parameter_2,
                seed + round_index,
            )
            cluster_ids, centers = prototypes(subset, labels)
            if len(cluster_ids) < 2:
                return 0.0
            assigned, _, _, _ = nearest_prototype(values, cluster_ids, centers)
            assignments.append(assigned)
        except (ValueError, RuntimeError):
            return 0.0
    return float(
        np.mean(
            [
                adjusted_rand_score(first, second)
                for first, second in itertools.combinations(assignments, 2)
            ]
        )
    )


def candidate_search(
    matrices: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    discovery_indices: np.ndarray,
    validation_indices: np.ndarray,
    components: dict[str, int],
    bootstrap_rounds: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, RepresentationModel]]:
    rows = []
    models = {}
    discovery_dates = metadata.iloc[discovery_indices]["date"].to_numpy()
    for representation_name, members in representation_specs().items():
        model = fit_representation(
            representation_name,
            members,
            matrices,
            discovery_indices,
            components,
            seed,
        )
        models[representation_name] = model
        discovery = model.transform(matrices)[discovery_indices]
        validation = model.transform(matrices)[validation_indices]
        candidates: list[tuple[str, int, int, float]] = []
        for clusters in range(2, min(8, len(discovery) - 1) + 1):
            candidates.append(("kmeans", clusters, 0, 0.0))
        for fraction in (0.01, 0.02, 0.04, 0.08):
            minimum_size = max(2, int(math.ceil(fraction * len(discovery))))
            for minimum_samples in (5, 15, 30):
                candidates.append(
                    ("hdbscan", minimum_size, minimum_samples, fraction)
                )
        seen = set()
        for algorithm, parameter_1, parameter_2, fraction in candidates:
            key = (algorithm, parameter_1, parameter_2)
            if key in seen:
                continue
            seen.add(key)
            try:
                labels = fit_cluster(
                    discovery,
                    algorithm,
                    parameter_1,
                    parameter_2,
                    seed,
                )
                cluster_ids, centers = prototypes(discovery, labels)
                if len(cluster_ids) < 2:
                    validation_labels = np.full(len(validation), -1)
                else:
                    validation_labels, _, _, _ = nearest_prototype(
                        validation, cluster_ids, centers
                    )
                discovery_geometry = cluster_geometry(
                    discovery, labels, seed
                )
                validation_geometry = cluster_geometry(
                    validation, validation_labels, seed
                )
                stability = grouped_stability(
                    discovery,
                    discovery_dates,
                    algorithm,
                    fraction,
                    parameter_1,
                    parameter_2,
                    bootstrap_rounds,
                    seed,
                )
            except (ValueError, RuntimeError):
                discovery_geometry = {
                    "clusters": 0,
                    "coverage": 0.0,
                    "silhouette": -1.0,
                    "balance": 0.0,
                    "maximum_cluster_fraction": 1.0,
                    "minimum_cluster_fraction": 0.0,
                }
                validation_geometry = discovery_geometry.copy()
                stability = 0.0
            nondegenerate = bool(
                2 <= discovery_geometry["clusters"] <= 12
                and discovery_geometry["coverage"] >= 0.70
                and discovery_geometry["maximum_cluster_fraction"] <= 0.70
                and discovery_geometry["minimum_cluster_fraction"] >= 0.01
                and validation_geometry["clusters"] >= 2
                and validation_geometry["maximum_cluster_fraction"] <= 0.80
            )
            objective = (
                math.sqrt(
                    max(discovery_geometry["silhouette"], 0.0)
                    * max(validation_geometry["silhouette"], 0.0)
                )
                * discovery_geometry["coverage"]
                * min(
                    discovery_geometry["balance"],
                    validation_geometry["balance"],
                )
                * max(stability, 0.0)
                if nondegenerate
                else -1.0
            )
            rows.append(
                {
                    "representation": representation_name,
                    "representation_member_count": len(members),
                    "representation_input_dimension": int(
                        sum(matrices[member].shape[1] for member in members)
                    ),
                    "algorithm": algorithm,
                    "parameter_1": parameter_1,
                    "parameter_2": parameter_2,
                    "parameter_fraction": fraction,
                    "nondegenerate": nondegenerate,
                    "label_free_objective": objective,
                    "bootstrap_stability_ari": stability,
                    **{
                        f"discovery_{name}": value
                        for name, value in discovery_geometry.items()
                    },
                    **{
                        f"validation_{name}": value
                        for name, value in validation_geometry.items()
                    },
                }
            )
    return pd.DataFrame(rows), models


def select_label_free_configuration(
    candidate_table: pd.DataFrame,
    objective_fraction: float,
    minimum_stability: float,
    policy: str,
) -> tuple[pd.Series, float, pd.DataFrame, pd.DataFrame, str]:
    """Select a frozen configuration without consulting post-hoc labels."""
    valid = candidate_table.loc[
        candidate_table["nondegenerate"].astype(bool)
    ]
    pool = valid if len(valid) else candidate_table
    best_objective = float(pool["label_free_objective"].max())
    shortlist = pool.loc[
        (
            pool["label_free_objective"]
            >= objective_fraction * best_objective
        )
        & (
            pool["bootstrap_stability_ari"]
            >= minimum_stability
        )
    ]
    if not len(shortlist):
        shortlist = pool.nlargest(1, "label_free_objective")

    if policy == "parsimonious":
        minimum_members = int(
            shortlist["representation_member_count"].min()
        )
        selection_pool = shortlist.loc[
            shortlist["representation_member_count"].eq(minimum_members)
        ]
        minimum_input_dimension = int(
            selection_pool["representation_input_dimension"].min()
        )
        selection_pool = selection_pool.loc[
            selection_pool["representation_input_dimension"].eq(
                minimum_input_dimension
            )
        ]
        selected = selection_pool.sort_values(
            [
                "label_free_objective",
                "bootstrap_stability_ari",
                "validation_silhouette",
                "discovery_clusters",
            ],
            ascending=[False, False, False, False],
        ).iloc[0]
        rule = (
            "among stable candidates in the prespecified near-optimal "
            "label-free objective set, prefer the representation with the "
            "fewest feature blocks and then the lowest raw input dimension; "
            "finally maximize the label-free objective and grouped-bootstrap "
            "stability"
        )
    else:
        selection_pool = shortlist
        selected = selection_pool.sort_values(
            [
                "discovery_clusters",
                "label_free_objective",
                "bootstrap_stability_ari",
                "validation_silhouette",
            ],
            ascending=[False, False, False, False],
        ).iloc[0]
        rule = (
            "maximize discovered core count among stable candidates within "
            "the prespecified near-optimal label-free objective set"
        )
    return selected, best_objective, shortlist, selection_pool, rule


def contingency(
    truth: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_ids = np.unique(truth)
    cluster_ids = np.unique(labels)
    table = np.zeros((len(cluster_ids), len(true_ids)), dtype=int)
    for row, cluster in enumerate(cluster_ids):
        for column, true_id in enumerate(true_ids):
            table[row, column] = int(
                np.sum((labels == cluster) & (truth == true_id))
            )
    return table, cluster_ids, true_ids


def hungarian_accuracy(truth: np.ndarray, labels: np.ndarray) -> float:
    if not len(truth):
        return float("nan")
    table, _, _ = contingency(truth, labels)
    rows, columns = linear_sum_assignment(table.max() - table)
    return float(table[rows, columns].sum() / table.sum())


def purity(truth: np.ndarray, labels: np.ndarray) -> float:
    if not len(truth):
        return float("nan")
    table, _, _ = contingency(truth, labels)
    return float(table.max(axis=1).sum() / table.sum())


def posthoc_cluster_metrics(
    truth: np.ndarray, labels: np.ndarray
) -> dict:
    if len(truth) < 2:
        return {"ari": float("nan"), "nmi": float("nan"), "purity": float("nan")}
    return {
        "ari": float(adjusted_rand_score(truth, labels)),
        "nmi": float(normalized_mutual_info_score(truth, labels)),
        "purity": purity(truth, labels),
        "hungarian_accuracy": hungarian_accuracy(truth, labels),
    }


def semantic_mapping(
    core_values: np.ndarray,
    core_labels: np.ndarray,
    cluster_ids: np.ndarray,
    centers: np.ndarray,
    truth: np.ndarray,
    budget: int,
) -> tuple[dict[int, str], list[dict]]:
    mapping = {}
    evidence = []
    for center_index, cluster in enumerate(cluster_ids):
        indices = np.flatnonzero(core_labels == cluster)
        distance = np.linalg.norm(
            core_values[indices] - centers[center_index], axis=1
        )
        selected = indices[np.argsort(distance)[: min(budget, len(indices))]]
        labels = truth[selected]
        values, counts = np.unique(labels, return_counts=True)
        winner = str(values[np.argmax(counts)])
        mapping[int(cluster)] = winner
        evidence.append(
            {
                "cluster_id": int(cluster),
                "label_budget": budget,
                "representative_indices": selected.tolist(),
                "representative_labels": labels.tolist(),
                "assigned_engineering_name": winner,
            }
        )
    return mapping, evidence


def classification_metrics(
    truth: np.ndarray, prediction: np.ndarray
) -> dict:
    labels = [label for label in CLASS_ORDER if label in set(truth)]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        prediction,
        labels=labels,
        zero_division=0,
    )
    per_class = [
        {
            "class": label,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    ]
    return {
        "accuracy": float(np.mean(truth == prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(truth, prediction)
        ),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "per_class": per_class,
    }


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (args.source_dir / "aace_exp8_representation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source_audit = json.loads(
        (args.source_dir / "formal_audit.json").read_text(encoding="utf-8")
    )
    if not manifest.get("passed") or not source_audit.get("passed"):
        raise RuntimeError("Representation stage did not pass its protocol audit")
    metadata = pd.read_parquet(
        args.source_dir / "aace_exp8_anonymous_metadata.parquet"
    )
    routed = metadata["routed_to_exp8"].astype(bool).to_numpy()
    block_names = (
        "residual",
        "ae_residual",
        "raw",
        "trajectory",
        "temperature",
        "coupling",
    )
    matrices = {
        name: np.load(args.source_dir / f"aace_exp8_{name}_features.npy")
        for name in block_names
    }
    residual = matrices["residual"]
    expected_residual_width = (
        RESIDUAL_LOCAL_WIDTH * RESIDUAL_AGGREGATION_BLOCKS
    )
    if residual.shape[1] != expected_residual_width:
        raise RuntimeError(
            "Expected the matched-contrast residual descriptor to contain "
            f"{expected_residual_width} columns, received {residual.shape[1]}"
        )
    compact_indices = np.concatenate(
        [
            block * RESIDUAL_LOCAL_WIDTH
            + np.arange(COMPACT_LOCAL_WIDTH)
            for block in range(RESIDUAL_AGGREGATION_BLOCKS)
        ]
    )
    matrices["residual_compact"] = residual[:, compact_indices]
    discovery_indices = np.flatnonzero(
        routed & metadata["split"].eq("core_discovery").to_numpy()
    )
    validation_indices = np.flatnonzero(
        routed & metadata["split"].eq("label_free_validation").to_numpy()
    )
    query_indices = np.flatnonzero(
        routed & metadata["split"].eq("frozen_query").to_numpy()
    )
    if min(len(discovery_indices), len(validation_indices), len(query_indices)) < 2:
        raise RuntimeError(
            "Each routed split needs at least two records; "
            f"got {len(discovery_indices)}/{len(validation_indices)}/{len(query_indices)}"
        )
    components = {
        "residual": args.residual_components,
        "residual_compact": args.residual_components,
        "ae_residual": args.residual_components,
        "raw": args.raw_components,
        "trajectory": args.trajectory_components,
        "temperature": args.temperature_components,
        "coupling": args.coupling_components,
    }
    if args.candidate_table is not None:
        candidate_table = pd.read_csv(args.candidate_table)
        stale_columns = [
            "resolution_shortlist",
            "final_selection_pool",
            "selected_configuration",
        ]
        candidate_table = candidate_table.drop(
            columns=[
                column
                for column in stale_columns
                if column in candidate_table.columns
            ]
        )
        member_counts = {
            name: len(members)
            for name, members in representation_specs().items()
        }
        input_dimensions = {
            name: int(
                sum(matrices[member].shape[1] for member in members)
            )
            for name, members in representation_specs().items()
        }
        candidate_table["representation_member_count"] = (
            candidate_table["representation"].map(member_counts)
        )
        candidate_table["representation_input_dimension"] = (
            candidate_table["representation"].map(input_dimensions)
        )
        if candidate_table["representation_member_count"].isna().any():
            raise RuntimeError(
                "The reused candidate table contains an unknown "
                "representation"
            )
        forbidden = {
            "fault_type",
            "posthoc_class",
            "severity",
            "severity_level",
            "inject_fraction",
            "injection_fraction",
        }
        if forbidden & set(candidate_table.columns):
            raise RuntimeError(
                "The reused candidate table contains post-hoc fields"
            )
    else:
        candidate_table, _ = candidate_search(
            matrices,
            metadata,
            discovery_indices,
            validation_indices,
            components,
            args.bootstrap_rounds,
            args.seed,
        )
    (
        selected,
        best_objective,
        resolution_pool,
        selection_pool,
        selection_rule,
    ) = select_label_free_configuration(
        candidate_table,
        args.resolution_objective_fraction,
        args.resolution_minimum_stability,
        args.selection_policy,
    )
    candidate_table["resolution_shortlist"] = False
    candidate_table.loc[resolution_pool.index, "resolution_shortlist"] = True
    candidate_table["final_selection_pool"] = False
    candidate_table.loc[
        selection_pool.index, "final_selection_pool"
    ] = True
    candidate_table["selected_configuration"] = False
    candidate_table.loc[selected.name, "selected_configuration"] = True
    candidate_table.to_csv(
        args.output_dir / "aace_exp8_label_free_candidate_selection.csv",
        index=False,
    )

    final_fit_indices = np.concatenate(
        [discovery_indices, validation_indices]
    )
    members = representation_specs()[str(selected["representation"])]
    final_model = fit_representation(
        str(selected["representation"]),
        members,
        matrices,
        final_fit_indices,
        components,
        args.seed,
    )
    full_embedding = final_model.transform(matrices)
    final_fit = full_embedding[final_fit_indices]
    query = full_embedding[query_indices]
    if str(selected["algorithm"]) == "hdbscan":
        final_parameter_1 = max(
            2,
            int(
                math.ceil(
                    float(selected["parameter_fraction"]) * len(final_fit)
                )
            ),
        )
    else:
        final_parameter_1 = int(selected["parameter_1"])
    final_labels = fit_cluster(
        final_fit,
        str(selected["algorithm"]),
        final_parameter_1,
        int(selected["parameter_2"]),
        args.seed,
    )
    cluster_ids, centers = prototypes(final_fit, final_labels)
    if len(cluster_ids) < 2:
        raise RuntimeError("Selected final model produced fewer than two cores")
    final_assigned, final_distance, final_second, final_margin = nearest_prototype(
        final_fit, cluster_ids, centers
    )
    query_assigned, query_distance, query_second, query_margin = nearest_prototype(
        query, cluster_ids, centers
    )

    # Freeze and materialize all label-free outputs before opening the sealed table.
    np.save(args.output_dir / "aace_exp8_final_embedding.npy", full_embedding)
    np.save(args.output_dir / "aace_exp8_core_centers.npy", centers)
    joblib.dump(
        final_model, args.output_dir / "aace_exp8_representation_model.joblib"
    )
    frozen_rows = []
    for local, global_index in enumerate(final_fit_indices):
        frozen_rows.append(
            {
                "scenario_uid": metadata.iloc[global_index]["scenario_uid"],
                "split": metadata.iloc[global_index]["split"],
                "role": (
                    "core"
                    if final_labels[local] >= 0
                    else "discovery_border"
                ),
                "core_cluster_id": int(final_labels[local]),
                "assigned_mode_id": int(final_assigned[local]),
                "prototype_distance": float(final_distance[local]),
                "second_distance": float(final_second[local]),
                "prototype_margin": float(final_margin[local]),
            }
        )
    for local, global_index in enumerate(query_indices):
        frozen_rows.append(
            {
                "scenario_uid": metadata.iloc[global_index]["scenario_uid"],
                "split": metadata.iloc[global_index]["split"],
                "role": "frozen_query",
                "core_cluster_id": -1,
                "assigned_mode_id": int(query_assigned[local]),
                "prototype_distance": float(query_distance[local]),
                "second_distance": float(query_second[local]),
                "prototype_margin": float(query_margin[local]),
            }
        )
    frozen = pd.DataFrame(frozen_rows)
    frozen.to_parquet(
        args.output_dir / "aace_exp8_frozen_label_free_assignments.parquet",
        index=False,
    )
    freeze_manifest = {
        "schema_version": "aace-exp8-frozen-core-v2",
        "fault_labels_loaded": False,
        "selected_configuration": {
            "representation": str(selected["representation"]),
            "algorithm": str(selected["algorithm"]),
            "parameter_1": final_parameter_1,
            "parameter_2": int(selected["parameter_2"]),
            "parameter_fraction": float(selected["parameter_fraction"]),
            "label_free_objective": float(selected["label_free_objective"]),
            "selection_policy": args.selection_policy,
            "resolution_selection_rule": selection_rule,
            "representation_member_count": int(
                selected["representation_member_count"]
            ),
            "resolution_objective_fraction": (
                args.resolution_objective_fraction
            ),
            "resolution_minimum_stability": (
                args.resolution_minimum_stability
            ),
            "best_label_free_objective": best_objective,
            "resolution_shortlist_size": int(len(resolution_pool)),
            "final_selection_pool_size": int(len(selection_pool)),
            "candidate_table_source": (
                str(args.candidate_table.resolve())
                if args.candidate_table is not None
                else "computed_in_current_run"
            ),
        },
        "core_cluster_count": int(len(cluster_ids)),
        "core_coverage": float(np.mean(final_labels >= 0)),
        "fit_record_count": int(len(final_fit_indices)),
        "query_record_count": int(len(query_indices)),
    }
    (args.output_dir / "aace_exp8_core_freeze_manifest.json").write_text(
        json.dumps(freeze_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Post-hoc labels are opened only after the previous freeze manifest exists.
    labels = pd.read_parquet(
        args.source_dir / "SEALED_aace_exp8_posthoc_labels.parquet"
    )[["scenario_uid", "posthoc_class", "severity_level", "inject_fraction"]]
    label_lookup = labels.set_index("scenario_uid")
    fit_truth = np.asarray(
        [
            label_lookup.loc[metadata.iloc[index]["scenario_uid"], "posthoc_class"]
            for index in final_fit_indices
        ]
    )
    query_truth = np.asarray(
        [
            label_lookup.loc[metadata.iloc[index]["scenario_uid"], "posthoc_class"]
            for index in query_indices
        ]
    )
    core_mask = final_labels >= 0
    fault_fit_mask = fit_truth != "background_control"
    fault_query_mask = query_truth != "background_control"
    unsupervised = {
        "core_all": posthoc_cluster_metrics(
            fit_truth[core_mask], final_labels[core_mask]
        ),
        "core_fault": posthoc_cluster_metrics(
            fit_truth[core_mask & fault_fit_mask],
            final_labels[core_mask & fault_fit_mask],
        ),
        "query_all": posthoc_cluster_metrics(query_truth, query_assigned),
        "query_fault": posthoc_cluster_metrics(
            query_truth[fault_query_mask],
            query_assigned[fault_query_mask],
        ),
    }

    budget_rows = []
    evidence_rows = []
    chosen_prediction = None
    per_class_rows = []
    for budget in args.label_budgets:
        mapping, evidence = semantic_mapping(
            final_fit,
            final_labels,
            cluster_ids,
            centers,
            fit_truth,
            budget,
        )
        prediction = np.asarray(
            [mapping[int(mode)] for mode in query_assigned]
        )
        metrics = classification_metrics(query_truth, prediction)
        fault_metrics = classification_metrics(
            query_truth[fault_query_mask],
            prediction[fault_query_mask],
        )
        budget_rows.append(
            {
                "labels_per_core": budget,
                **{
                    name: value
                    for name, value in metrics.items()
                    if name != "per_class"
                },
                **{
                    f"fault_{name}": value
                    for name, value in fault_metrics.items()
                    if name != "per_class"
                },
            }
        )
        for row in metrics["per_class"]:
            per_class_rows.append({"labels_per_core": budget, **row})
        evidence_rows.extend(evidence)
        if budget == args.primary_label_budget:
            chosen_prediction = prediction
    if chosen_prediction is None:
        raise RuntimeError("Primary label budget was not evaluated")

    pd.DataFrame(budget_rows).to_csv(
        args.output_dir / "aace_exp8_sparse_naming_metrics.csv", index=False
    )
    pd.DataFrame(per_class_rows).to_csv(
        args.output_dir / "aace_exp8_sparse_naming_per_class.csv", index=False
    )
    (
        args.output_dir / "aace_exp8_sparse_naming_evidence.json"
    ).write_text(
        json.dumps(evidence_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    confusion = confusion_matrix(
        query_truth,
        chosen_prediction,
        labels=CLASS_ORDER,
    )
    pd.DataFrame(
        confusion,
        index=CLASS_ORDER,
        columns=CLASS_ORDER,
    ).to_csv(args.output_dir / "aace_exp8_query_confusion.csv")

    posthoc = frozen.merge(labels, on="scenario_uid", how="left")
    query_prediction_lookup = {
        metadata.iloc[index]["scenario_uid"]: chosen_prediction[local]
        for local, index in enumerate(query_indices)
    }
    posthoc["predicted_engineering_name"] = posthoc["scenario_uid"].map(
        query_prediction_lookup
    )
    posthoc.to_parquet(
        args.output_dir / "aace_exp8_posthoc_assignments.parquet", index=False
    )

    nuisance_rows = []
    for subset_name, indices, modes in (
        ("core", final_fit_indices[core_mask], final_labels[core_mask]),
        ("frozen_query", query_indices, query_assigned),
    ):
        subset_uids = metadata.iloc[indices]["scenario_uid"].to_numpy()
        subset_dates = metadata.iloc[indices]["date"].astype(str).to_numpy()
        subset_labels = label_lookup.loc[subset_uids]
        fault_only = (
            subset_labels["posthoc_class"].to_numpy()
            != "background_control"
        )
        factors = {
            "engineering_fault_type": subset_labels[
                "posthoc_class"
            ].astype(str).to_numpy(),
            "severity_level": subset_labels[
                "severity_level"
            ].astype(str).to_numpy(),
            "injection_fraction": subset_labels[
                "inject_fraction"
            ].astype(str).to_numpy(),
            "date_background": subset_dates,
        }
        for factor_name, factor_values in factors.items():
            use = (
                np.ones(len(modes), dtype=bool)
                if factor_name == "date_background"
                else fault_only
            )
            nuisance_rows.append(
                {
                    "subset": subset_name,
                    "factor": factor_name,
                    "record_count": int(use.sum()),
                    "normalized_mutual_information": float(
                        normalized_mutual_info_score(
                            factor_values[use], np.asarray(modes)[use]
                        )
                    ),
                }
            )
    pd.DataFrame(nuisance_rows).to_csv(
        args.output_dir / "aace_exp8_mechanism_nuisance_audit.csv",
        index=False,
    )

    query_all_mask = metadata["split"].eq("frozen_query").to_numpy()
    query_metadata = metadata.loc[query_all_mask, ["scenario_uid", "routed_to_exp8"]]
    query_labeled = query_metadata.merge(labels, on="scenario_uid", how="left")
    routing_rows = [
        {
            "group": "all",
            "scenario_count": int(len(query_labeled)),
            "routed_count": int(query_labeled["routed_to_exp8"].sum()),
            "routing_rate": float(query_labeled["routed_to_exp8"].mean()),
        }
    ]
    for (fault, severity), group in query_labeled.groupby(
        ["posthoc_class", "severity_level"]
    ):
        routing_rows.append(
            {
                "group": f"{fault}|severity={severity}",
                "scenario_count": int(len(group)),
                "routed_count": int(group["routed_to_exp8"].sum()),
                "routing_rate": float(group["routed_to_exp8"].mean()),
            }
        )
    pd.DataFrame(routing_rows).to_csv(
        args.output_dir / "aace_exp8_routing_audit.csv", index=False
    )

    primary = next(
        row
        for row in budget_rows
        if row["labels_per_core"] == args.primary_label_budget
    )
    result = {
        "schema_version": "aace-exp8-unknown-mode-result-v2",
        "selected_configuration": freeze_manifest["selected_configuration"],
        "unsupervised_posthoc": unsupervised,
        "sparse_naming": budget_rows,
        "routing_rate_all_query": routing_rows[0]["routing_rate"],
        "prototype_assignment_coverage_for_routed_query": 1.0,
        "labels_used_for_representation_or_selection": False,
        "labels_used_for_query_assignment": False,
        "labels_used_after_core_freeze": True,
        "posthoc_evidence_kind": (
            "physics injection metadata simulating maintenance confirmation"
        ),
    }
    (args.output_dir / "aace_exp8_formal_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    protocol_checks = {
        "source_representation_passed": bool(manifest["passed"]),
        "labels_opened_only_after_freeze_manifest": (
            args.output_dir / "aace_exp8_core_freeze_manifest.json"
        ).exists(),
        "candidate_selection_contains_no_posthoc_fields": not any(
            field in candidate_table.columns
            for field in ("fault_type", "severity", "inject_fraction")
        ),
        "query_dates_excluded_from_fit": not bool(
            set(metadata.iloc[query_indices]["date"])
            & set(metadata.iloc[final_fit_indices]["date"])
        ),
        "prototype_assignment_coverage_is_one": True,
    }
    scientific_checks = {
        "at_least_two_core_modes": len(cluster_ids) >= 2,
        "core_coverage_at_least_0_70": freeze_manifest["core_coverage"] >= 0.70,
        "fault_core_ari_at_least_0_60": (
            unsupervised["core_fault"]["ari"] >= 0.60
        ),
        "fault_core_nmi_at_least_0_75": (
            unsupervised["core_fault"]["nmi"] >= 0.75
        ),
        "query_fault_nmi_at_least_0_50": (
            unsupervised["query_fault"]["nmi"] >= 0.50
        ),
        "one_label_fault_balanced_accuracy_at_least_0_85": (
            primary["fault_balanced_accuracy"] >= 0.85
        ),
        "one_label_fault_macro_f1_at_least_0_85": (
            primary["fault_macro_f1"] >= 0.85
        ),
    }
    audit = {
        "schema_version": "aace-exp8-formal-audit-v1",
        "protocol_checks": protocol_checks,
        "scientific_checks": scientific_checks,
        "protocol_passed": bool(all(protocol_checks.values())),
        "scientific_passed": bool(all(scientific_checks.values())),
    }
    audit["passed"] = bool(
        audit["protocol_passed"] and audit["scientific_passed"]
    )
    (args.output_dir / "formal_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)


def self_test() -> None:
    rng = np.random.default_rng(42)
    values = np.vstack(
        [
            rng.normal(-2, 0.2, size=(30, 4)),
            rng.normal(2, 0.2, size=(30, 4)),
        ]
    )
    labels = fit_cluster(values, "kmeans", 2, 0, 42)
    cluster_ids, centers = prototypes(values, labels)
    assigned, distance, second, margin = nearest_prototype(
        values, cluster_ids, centers
    )
    assert len(np.unique(assigned)) == 2
    assert np.all(distance <= second)
    assert np.all(margin >= 0)
    assert posthoc_cluster_metrics(labels, assigned)["ari"] == 1.0
    candidates = pd.DataFrame(
        [
            {
                "representation": "richer",
                "representation_member_count": 3,
                "representation_input_dimension": 420,
                "nondegenerate": True,
                "label_free_objective": 0.100,
                "bootstrap_stability_ari": 0.99,
                "validation_silhouette": 0.12,
                "discovery_clusters": 3,
            },
            {
                "representation": "compact",
                "representation_member_count": 1,
                "representation_input_dimension": 180,
                "nondegenerate": True,
                "label_free_objective": 0.092,
                "bootstrap_stability_ari": 0.98,
                "validation_silhouette": 0.11,
                "discovery_clusters": 4,
            },
        ]
    )
    selected, _, _, _, _ = select_label_free_configuration(
        candidates, 0.80, 0.75, "parsimonious"
    )
    assert selected["representation"] == "compact"
    print("AACE-CM Exp8 assignment self-test passed")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("run", "self-test"))
    result.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument(
        "--candidate-table",
        type=Path,
        default=None,
        help=(
            "Optional previously frozen label-free candidate table generated "
            "from the same representation, date split, seed, and bootstrap "
            "protocol."
        ),
    )
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--bootstrap-rounds", type=int, default=5)
    result.add_argument(
        "--selection-policy",
        choices=("parsimonious", "max_resolution"),
        default="parsimonious",
        help=(
            "Label-free final selection rule. The formal mainline uses "
            "parsimonious; max_resolution reproduces the earlier five-core "
            "exploration."
        ),
    )
    result.add_argument(
        "--resolution-objective-fraction",
        type=float,
        default=0.80,
        help=(
            "Near-optimal label-free objective floor used before selecting "
            "the highest stable discovery resolution."
        ),
    )
    result.add_argument(
        "--resolution-minimum-stability",
        type=float,
        default=0.75,
        help="Minimum grouped-bootstrap ARI for the resolution shortlist.",
    )
    result.add_argument("--residual-components", type=int, default=24)
    result.add_argument("--raw-components", type=int, default=16)
    result.add_argument("--trajectory-components", type=int, default=24)
    result.add_argument("--temperature-components", type=int, default=16)
    result.add_argument("--coupling-components", type=int, default=8)
    result.add_argument(
        "--label-budgets", type=int, nargs="+", default=[1, 3, 5]
    )
    result.add_argument("--primary-label-budget", type=int, default=1)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.mode == "self-test":
        self_test()
    else:
        run(arguments)
