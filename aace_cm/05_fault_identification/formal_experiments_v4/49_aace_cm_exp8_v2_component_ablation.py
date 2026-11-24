"""Controlled component ablation for the frozen compact AACE-CM Exp8 pipeline.

All variants use the formal cross-date split and one representative record per
discovered mode. Engineering labels are opened only after every representation,
cluster assignment, and prototype has been frozen.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_RUN = (
    SCRIPT_DIR
    / "outputs"
    / "aace_cm_exp8_unknown_modes"
    / "formal_20260725_183900"
)
DEFAULT_FORMAL_RESULT = (
    SCRIPT_DIR
    / "outputs"
    / "aace_cm_exp8_unknown_modes"
    / "formal_compact_selected_20260727"
)
DEFAULT_FIGURE_DIR = (
    SCRIPT_DIR.parent
    / "paper_figure_table_notebook"
    / "exp8_section_package"
    / "figures"
    / "aace_unknown_modes_compact_20260727"
)

LOCAL_FEATURE_WIDTH = 58
AGGREGATION_BLOCKS = 7
WAVEFORM_SLICE = slice(0, 16)
RELATION_SLICE = slice(16, 26)


def load_assignment_module() -> ModuleType:
    path = SCRIPT_DIR / "44_aace_cm_exp8_core_prototype_assignment.py"
    spec = importlib.util.spec_from_file_location(
        "aace_exp8_v2_ablation_assignment", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load assignment module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def repeated_indices(local_indices: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            block * LOCAL_FEATURE_WIDTH + local_indices
            for block in range(AGGREGATION_BLOCKS)
        ]
    )


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        path.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument(
        "--formal-result", type=Path, default=DEFAULT_FORMAL_RESULT
    )
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assignment = load_assignment_module()
    source_dir = args.source_run / "representation"
    output_dir = args.formal_result / "v2_component_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    source_audit = json.loads(
        (source_dir / "formal_audit.json").read_text(encoding="utf-8")
    )
    formal_audit = json.loads(
        (args.formal_result / "formal_audit.json").read_text(encoding="utf-8")
    )
    if not source_audit.get("passed") or not formal_audit.get("passed"):
        raise RuntimeError("The source representation or formal audit failed")

    metadata = pd.read_parquet(
        source_dir / "aace_exp8_anonymous_metadata.parquet"
    )
    routed = metadata["routed_to_exp8"].astype(bool).to_numpy()
    discovery = np.flatnonzero(
        routed & metadata["split"].eq("core_discovery").to_numpy()
    )
    validation = np.flatnonzero(
        routed & metadata["split"].eq("label_free_validation").to_numpy()
    )
    query = np.flatnonzero(
        routed & metadata["split"].eq("frozen_query").to_numpy()
    )
    fit_indices = np.concatenate([discovery, validation])

    residual = np.load(source_dir / "aace_exp8_residual_features.npy")
    ae_residual = np.load(source_dir / "aace_exp8_ae_residual_features.npy")
    expected_width = LOCAL_FEATURE_WIDTH * AGGREGATION_BLOCKS
    if residual.shape[1] != expected_width or ae_residual.shape[1] != expected_width:
        raise RuntimeError(
            f"Expected two {expected_width}-D descriptors, received "
            f"{residual.shape} and {ae_residual.shape}"
        )

    waveform = np.arange(WAVEFORM_SLICE.start, WAVEFORM_SLICE.stop)
    relation = np.arange(RELATION_SLICE.start, RELATION_SLICE.stop)
    keep_waveform = repeated_indices(waveform)
    keep_relation = repeated_indices(relation)
    keep_compact = repeated_indices(np.concatenate([waveform, relation]))

    variants = [
        {
            "variant": "no_matched_reference",
            "label": "AE residual instead of matched healthy reference",
            "matrix": ae_residual[:, keep_compact],
            "algorithm": "kmeans",
            "parameter_1": 4,
            "parameter_2": 0,
            "matched_reference": False,
            "waveform_shape": True,
            "channel_relations": True,
            "kmeans_prototypes": True,
        },
        {
            "variant": "no_waveform_shape",
            "label": "Without waveform-shape statistics",
            "matrix": residual[:, keep_relation],
            "algorithm": "kmeans",
            "parameter_1": 4,
            "parameter_2": 0,
            "matched_reference": True,
            "waveform_shape": False,
            "channel_relations": True,
            "kmeans_prototypes": True,
        },
        {
            "variant": "no_channel_relations",
            "label": "Without cross-channel relations",
            "matrix": residual[:, keep_waveform],
            "algorithm": "kmeans",
            "parameter_1": 4,
            "parameter_2": 0,
            "matched_reference": True,
            "waveform_shape": True,
            "channel_relations": False,
            "kmeans_prototypes": True,
        },
        {
            "variant": "hdbscan_instead_of_kmeans",
            "label": "HDBSCAN instead of KMeans",
            "matrix": residual[:, keep_compact],
            "algorithm": "hdbscan",
            "parameter_1": int(math.ceil(0.02 * len(fit_indices))),
            "parameter_2": 5,
            "matched_reference": True,
            "waveform_shape": True,
            "channel_relations": True,
            "kmeans_prototypes": False,
        },
        {
            "variant": "full_compact",
            "label": "Proposed-TFRC (compact)",
            "matrix": residual[:, keep_compact],
            "algorithm": "kmeans",
            "parameter_1": 4,
            "parameter_2": 0,
            "matched_reference": True,
            "waveform_shape": True,
            "channel_relations": True,
            "kmeans_prototypes": True,
        },
    ]

    frozen: list[dict] = []
    for variant in variants:
        name = str(variant["variant"])
        matrix = np.asarray(variant["matrix"], dtype=np.float32)
        matrices = {name: matrix}
        model = assignment.fit_representation(
            name,
            (name,),
            matrices,
            fit_indices,
            {name: 24},
            args.seed,
        )
        embedding = model.transform(matrices)
        fit_values = embedding[fit_indices]
        query_values = embedding[query]
        core_labels = assignment.fit_cluster(
            fit_values,
            str(variant["algorithm"]),
            int(variant["parameter_1"]),
            int(variant["parameter_2"]),
            args.seed,
        )
        cluster_ids, centers = assignment.prototypes(fit_values, core_labels)
        if len(cluster_ids) < 2:
            raise RuntimeError(f"{name} produced fewer than two modes")
        query_modes, _, _, margins = assignment.nearest_prototype(
            query_values, cluster_ids, centers
        )
        frozen.append(
            {
                **{key: value for key, value in variant.items() if key != "matrix"},
                "input_dimension": int(matrix.shape[1]),
                "embedding_dimension": int(embedding.shape[1]),
                "mode_count": int(len(cluster_ids)),
                "fit_values": fit_values,
                "core_labels": core_labels,
                "cluster_ids": cluster_ids,
                "centers": centers,
                "query_modes": query_modes,
                "mean_prototype_margin": float(np.mean(margins)),
            }
        )

    freeze_manifest = {
        "schema_version": "aace-exp8-v2-component-ablation-freeze-v1",
        "fault_labels_loaded": False,
        "split": {
            "discovery_records": int(len(discovery)),
            "validation_records": int(len(validation)),
            "fit_records": int(len(fit_indices)),
            "query_records": int(len(query)),
        },
        "hdbscan_selection": (
            "Prespecified full-V2 label-free grid choice maximizing the relaxed "
            "geometry/stability score among HDBSCAN candidates: fraction=0.02, "
            "min_samples=5; min_cluster_size is rescaled to the final fit set."
        ),
        "variants": [
            {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "fit_values",
                    "core_labels",
                    "cluster_ids",
                    "centers",
                    "query_modes",
                }
            }
            for item in frozen
        ],
    }
    freeze_path = output_dir / "aace_exp8_v2_ablation_freeze_manifest.json"
    freeze_path.write_text(
        json.dumps(freeze_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Open engineering evidence only after every variant has been frozen.
    sealed = pd.read_parquet(
        source_dir / "SEALED_aace_exp8_posthoc_labels.parquet"
    )[["scenario_uid", "posthoc_class"]]
    lookup = sealed.set_index("scenario_uid")["posthoc_class"]
    fit_uids = metadata.iloc[fit_indices]["scenario_uid"].astype(str).to_numpy()
    query_uids = metadata.iloc[query]["scenario_uid"].astype(str).to_numpy()
    fit_truth = lookup.loc[fit_uids].astype(str).to_numpy()
    query_truth = lookup.loc[query_uids].astype(str).to_numpy()

    rows = []
    predictions = []
    for item in frozen:
        valid = item["core_labels"] >= 0
        mapping, evidence = assignment.semantic_mapping(
            item["fit_values"],
            item["core_labels"],
            item["cluster_ids"],
            item["centers"],
            fit_truth,
            budget=1,
        )
        predicted = np.asarray(
            [mapping[int(mode)] for mode in item["query_modes"]]
        )
        classification = assignment.classification_metrics(
            query_truth, predicted
        )
        clustering = assignment.posthoc_cluster_metrics(
            query_truth, item["query_modes"]
        )
        rows.append(
            {
                "variant": item["variant"],
                "label": item["label"],
                "matched_reference": item["matched_reference"],
                "waveform_shape": item["waveform_shape"],
                "channel_relations": item["channel_relations"],
                "kmeans_prototypes": item["kmeans_prototypes"],
                "algorithm": item["algorithm"],
                "input_dimension": item["input_dimension"],
                "mode_count": item["mode_count"],
                "confirmation_records": item["mode_count"],
                "query_ari": clustering["ari"],
                "query_nmi": clustering["nmi"],
                "query_purity": clustering["purity"],
                "accuracy": classification["accuracy"],
                "balanced_accuracy": classification["balanced_accuracy"],
                "macro_f1": classification["macro_f1"],
                "mcc": float(matthews_corrcoef(query_truth, predicted)),
                "mean_prototype_margin": item["mean_prototype_margin"],
            }
        )
        for uid, truth, prediction, mode in zip(
            query_uids, query_truth, predicted, item["query_modes"]
        ):
            predictions.append(
                {
                    "variant": item["variant"],
                    "scenario_uid": uid,
                    "true_class": truth,
                    "predicted_class": prediction,
                    "assigned_mode_id": int(mode),
                }
            )
        (output_dir / f"{item['variant']}_naming_evidence.json").write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(
        output_dir / "aace_exp8_v2_component_ablation_summary.csv", index=False
    )
    pd.DataFrame(predictions).to_csv(
        output_dir / "aace_exp8_v2_component_ablation_predictions.csv",
        index=False,
    )

    audit = {
        "schema_version": "aace-exp8-v2-component-ablation-audit-v1",
        "source_representation_audit_passed": bool(source_audit["passed"]),
        "formal_result_audit_passed": bool(formal_audit["passed"]),
        "variants_frozen_before_labels_opened": freeze_path.exists(),
        "same_query_uids_for_all_variants": True,
        "full_compact_reproduces_formal_accuracy": bool(
            np.isclose(
                summary.loc[
                    summary["variant"].eq("full_compact"), "accuracy"
                ].iloc[0],
                0.9691666666666666,
            )
        ),
    }
    audit["passed"] = bool(all(audit.values()))
    (output_dir / "aace_exp8_v2_component_ablation_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not audit["passed"]:
        raise RuntimeError(f"Ablation audit failed: {audit}")

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.linewidth": 1.0,
        }
    )
    figure, axis = plt.subplots(figsize=(7.4, 3.9))
    positions = np.arange(len(summary))
    colors = ["#A7B6C2", "#76A5AF", "#7AA974", "#B785A7", "#147D92"]
    bars = axis.bar(
        positions,
        100 * summary["macro_f1"],
        color=colors,
        edgecolor="black",
        linewidth=0.7,
    )
    axis.set_xticks(
        positions,
        [
            "AE residual",
            "w/o waveform",
            "w/o channel",
            "HDBSCAN",
            "Proposed-TFRC",
        ],
        rotation=20,
        ha="right",
    )
    axis.set_ylabel("Macro-F1 (%)")
    axis.set_ylim(0, 105)
    for bar, value in zip(bars, 100 * summary["macro_f1"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    for spine in axis.spines.values():
        spine.set_visible(True)
    figure.tight_layout()
    save_figure(
        figure, args.figure_dir / "aace_exp8_v2_component_ablation"
    )

    print(summary.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
