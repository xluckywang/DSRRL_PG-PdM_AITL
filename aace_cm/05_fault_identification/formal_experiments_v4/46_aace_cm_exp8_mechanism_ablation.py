"""Run the frozen-representation ablation for AACE-CM Exp8.

Every representation variant selects its clustering configuration using only
the previously materialized label-free candidate table. Engineering labels are
opened only after all variant configurations, cores, and prototypes are frozen.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, normalized_mutual_info_score


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = (
    SCRIPT_DIR
    / "outputs"
    / "aace_cm_exp8_unknown_modes"
    / "formal_20260725_183900"
)
DEFAULT_FIGURE_DIR = (
    SCRIPT_DIR.parent
    / "paper_figure_table_notebook"
    / "exp8_section_package"
    / "figures"
    / "aace_unknown_modes_formal_20260725"
)
REPRESENTATION_LABELS = {
    "ae_residual_only": "Healthy-AE",
    "paired_contrast_only": "Paired contrast",
    "paired_contrast_trajectory": "+ trajectory",
    "paired_contrast_temperature_coupling": "+ temp./coupling",
    "paired_contrast_full": "+ full descriptors",
    "paired_contrast_plus_ae": "+ AE residual",
}


def load_assignment_module() -> ModuleType:
    path = SCRIPT_DIR / "44_aace_cm_exp8_core_prototype_assignment.py"
    spec = importlib.util.spec_from_file_location("aace_exp8_assignment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load assignment module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def select_label_free_candidate(
    table: pd.DataFrame,
    objective_fraction: float,
    minimum_stability: float,
) -> pd.Series:
    valid = table.loc[table["nondegenerate"].astype(bool)]
    pool = valid if len(valid) else table
    best = float(pool["label_free_objective"].max())
    shortlist = pool.loc[
        (pool["label_free_objective"] >= objective_fraction * best)
        & (pool["bootstrap_stability_ari"] >= minimum_stability)
    ]
    if not len(shortlist):
        shortlist = pool.nlargest(1, "label_free_objective")
    selected = shortlist.sort_values(
        [
            "discovery_clusters",
            "label_free_objective",
            "bootstrap_stability_ari",
            "validation_silhouette",
        ],
        ascending=[False, False, False, False],
    ).iloc[0].copy()
    selected["representation_best_objective"] = best
    selected["representation_shortlist_size"] = len(shortlist)
    return selected


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


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    def format_value(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value)

    columns = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        rows.append(
            "| " + " | ".join(format_value(value) for value in values) + " |"
        )
    return "\n".join(rows)


def run(args: argparse.Namespace) -> None:
    assignment = load_assignment_module()
    source_dir = args.run_dir / "representation"
    formal_dir = args.run_dir / "formal_result"
    output_dir = formal_dir / "mechanism_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads(
        (source_dir / "aace_exp8_representation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    formal_audit = json.loads(
        (formal_dir / "formal_audit.json").read_text(encoding="utf-8")
    )
    if not source_manifest.get("passed") or not formal_audit.get("passed"):
        raise RuntimeError("The formal representation/result audit did not pass")

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
    final_fit_indices = np.concatenate([discovery, validation])

    block_names = (
        "residual",
        "ae_residual",
        "raw",
        "trajectory",
        "temperature",
        "coupling",
    )
    matrices = {
        name: np.load(source_dir / f"aace_exp8_{name}_features.npy")
        for name in block_names
    }
    components = {
        "residual": args.residual_components,
        "ae_residual": args.residual_components,
        "raw": args.raw_components,
        "trajectory": args.trajectory_components,
        "temperature": args.temperature_components,
        "coupling": args.coupling_components,
    }
    candidates = pd.read_csv(
        formal_dir / "aace_exp8_label_free_candidate_selection.csv"
    )

    frozen_variants: dict[str, dict] = {}
    freeze_rows = []
    for representation, members in assignment.representation_specs().items():
        subset = candidates.loc[candidates["representation"].eq(representation)]
        selected = select_label_free_candidate(
            subset,
            args.resolution_objective_fraction,
            args.resolution_minimum_stability,
        )
        model = assignment.fit_representation(
            representation,
            members,
            matrices,
            final_fit_indices,
            components,
            args.seed,
        )
        embedding = model.transform(matrices)
        fit_values = embedding[final_fit_indices]
        query_values = embedding[query]
        if str(selected["algorithm"]) == "hdbscan":
            parameter_1 = max(
                2,
                int(
                    np.ceil(
                        float(selected["parameter_fraction"]) * len(fit_values)
                    )
                ),
            )
        else:
            parameter_1 = int(selected["parameter_1"])
        core_labels = assignment.fit_cluster(
            fit_values,
            str(selected["algorithm"]),
            parameter_1,
            int(selected["parameter_2"]),
            args.seed,
        )
        cluster_ids, centers = assignment.prototypes(fit_values, core_labels)
        if len(cluster_ids) < 2:
            raise RuntimeError(
                f"{representation} produced fewer than two frozen cores"
            )
        query_modes, query_distance, query_second, query_margin = (
            assignment.nearest_prototype(query_values, cluster_ids, centers)
        )
        frozen_variants[representation] = {
            "fit_values": fit_values,
            "query_values": query_values,
            "core_labels": core_labels,
            "cluster_ids": cluster_ids,
            "centers": centers,
            "query_modes": query_modes,
            "query_distance": query_distance,
            "query_second": query_second,
            "query_margin": query_margin,
        }
        freeze_rows.append(
            {
                "representation": representation,
                "members": list(members),
                "algorithm": str(selected["algorithm"]),
                "parameter_1": parameter_1,
                "parameter_2": int(selected["parameter_2"]),
                "label_free_objective": float(
                    selected["label_free_objective"]
                ),
                "representation_best_objective": float(
                    selected["representation_best_objective"]
                ),
                "bootstrap_stability_ari": float(
                    selected["bootstrap_stability_ari"]
                ),
                "core_count": int(len(cluster_ids)),
                "core_coverage": float(np.mean(core_labels >= 0)),
            }
        )

    freeze_manifest = {
        "schema_version": "aace-exp8-mechanism-ablation-freeze-v1",
        "fault_labels_loaded": False,
        "selection_basis": (
            "independent label-free selection within each representation"
        ),
        "resolution_objective_fraction": args.resolution_objective_fraction,
        "resolution_minimum_stability": args.resolution_minimum_stability,
        "variants": freeze_rows,
    }
    freeze_path = output_dir / "aace_exp8_ablation_freeze_manifest.json"
    freeze_path.write_text(
        json.dumps(freeze_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Only now open the sealed engineering evidence.
    labels = pd.read_parquet(
        source_dir / "SEALED_aace_exp8_posthoc_labels.parquet"
    )[["scenario_uid", "posthoc_class", "severity_level", "inject_fraction"]]
    lookup = labels.set_index("scenario_uid")
    fit_uids = metadata.iloc[final_fit_indices]["scenario_uid"].to_numpy()
    query_uids = metadata.iloc[query]["scenario_uid"].to_numpy()
    fit_truth = lookup.loc[fit_uids, "posthoc_class"].astype(str).to_numpy()
    query_truth = lookup.loc[query_uids, "posthoc_class"].astype(str).to_numpy()
    query_dates = metadata.iloc[query]["date"].astype(str).to_numpy()
    query_severity = (
        lookup.loc[query_uids, "severity_level"].astype(str).to_numpy()
    )
    query_fraction = (
        lookup.loc[query_uids, "inject_fraction"].astype(str).to_numpy()
    )

    summary_rows = []
    per_class_rows = []
    prediction_rows = []
    for frozen in freeze_rows:
        representation = frozen["representation"]
        values = frozen_variants[representation]
        core_mask = values["core_labels"] >= 0
        mapping, evidence = assignment.semantic_mapping(
            values["fit_values"],
            values["core_labels"],
            values["cluster_ids"],
            values["centers"],
            fit_truth,
            budget=1,
        )
        prediction = np.asarray(
            [mapping[int(mode)] for mode in values["query_modes"]]
        )
        metrics = assignment.classification_metrics(query_truth, prediction)
        core_cluster = assignment.posthoc_cluster_metrics(
            fit_truth[core_mask], values["core_labels"][core_mask]
        )
        query_cluster = assignment.posthoc_cluster_metrics(
            query_truth, values["query_modes"]
        )
        summary_rows.append(
            {
                **frozen,
                "core_ari": core_cluster["ari"],
                "core_nmi": core_cluster["nmi"],
                "query_ari": query_cluster["ari"],
                "query_nmi": query_cluster["nmi"],
                "query_severity_nmi": normalized_mutual_info_score(
                    query_severity, values["query_modes"]
                ),
                "query_fraction_nmi": normalized_mutual_info_score(
                    query_fraction, values["query_modes"]
                ),
                "query_date_nmi": normalized_mutual_info_score(
                    query_dates, values["query_modes"]
                ),
                "one_record_accuracy": metrics["accuracy"],
                "one_record_balanced_accuracy": metrics["balanced_accuracy"],
                "one_record_macro_f1": metrics["macro_f1"],
                "one_record_mcc": float(
                    matthews_corrcoef(query_truth, prediction)
                ),
                "mean_prototype_margin": float(
                    np.mean(values["query_margin"])
                ),
            }
        )
        for row in metrics["per_class"]:
            per_class_rows.append(
                {"representation": representation, **row}
            )
        for uid, truth, predicted, mode in zip(
            query_uids,
            query_truth,
            prediction,
            values["query_modes"],
        ):
            prediction_rows.append(
                {
                    "representation": representation,
                    "scenario_uid": uid,
                    "true_class": truth,
                    "predicted_class": predicted,
                    "assigned_mode_id": int(mode),
                }
            )
        (
            output_dir / f"{representation}_one_record_evidence.json"
        ).write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_dir / "aace_exp8_mechanism_ablation_summary.csv", index=False
    )
    pd.DataFrame(per_class_rows).to_csv(
        output_dir / "aace_exp8_mechanism_ablation_per_class.csv", index=False
    )
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "aace_exp8_mechanism_ablation_predictions.csv",
        index=False,
    )
    audit = {
        "schema_version": "aace-exp8-mechanism-ablation-audit-v1",
        "source_formal_audit_passed": True,
        "labels_opened_after_all_variant_freezes": freeze_path.exists(),
        "selection_table_contains_engineering_labels": any(
            column
            in {
                "posthoc_class",
                "severity_level",
                "inject_fraction",
                "date",
            }
            for column in candidates.columns
        ),
        "variant_count": len(summary),
        "passed": True,
    }
    audit["passed"] = bool(
        audit["source_formal_audit_passed"]
        and audit["labels_opened_after_all_variant_freezes"]
        and not audit["selection_table_contains_engineering_labels"]
    )
    (output_dir / "aace_exp8_mechanism_ablation_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not audit["passed"]:
        raise RuntimeError("Ablation protocol audit failed")

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.linewidth": 1.0,
        }
    )
    ordered = list(REPRESENTATION_LABELS)
    plot = summary.set_index("representation").loc[ordered]
    labels_for_plot = [REPRESENTATION_LABELS[name] for name in ordered]
    positions = np.arange(len(plot))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(
        positions,
        100 * plot["one_record_macro_f1"],
        color=["#a9b7c6", "#7aa6c2", "#4f8fb8", "#1f77b4", "#e69f00", "#8f79b8"],
        edgecolor="black",
        linewidth=0.7,
    )
    axes[0].set_ylabel("Macro-F1 (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels_for_plot, rotation=28, ha="right")
    axes[1].plot(
        positions,
        plot["query_nmi"],
        marker="o",
        linewidth=1.8,
        label="Fault mechanism",
        color="#0072b2",
    )
    axes[1].plot(
        positions,
        plot["query_severity_nmi"],
        marker="s",
        linewidth=1.6,
        label="Severity",
        color="#d55e00",
    )
    axes[1].plot(
        positions,
        plot["query_fraction_nmi"],
        marker="^",
        linewidth=1.6,
        label="Injection fraction",
        color="#009e73",
    )
    axes[1].set_ylabel("Normalized mutual information")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels_for_plot, rotation=28, ha="right")
    axes[1].legend(frameon=False, loc="upper left")
    for axis in axes:
        axis.spines["top"].set_visible(True)
        axis.spines["right"].set_visible(True)
        axis.tick_params(direction="out")
    figure.tight_layout()
    save_figure(
        figure, args.figure_dir / "aace_exp8_mechanism_ablation"
    )

    display_columns = [
        "representation",
        "algorithm",
        "core_count",
        "query_nmi",
        "query_severity_nmi",
        "query_fraction_nmi",
        "one_record_balanced_accuracy",
        "one_record_macro_f1",
        "one_record_mcc",
    ]
    markdown = [
        "# AACE-CM Exp8 mechanism-oriented ablation",
        "",
        dataframe_to_markdown(summary[display_columns]),
        "",
        "All configurations were selected without engineering labels. "
        "Post-hoc evidence was opened only after every variant was frozen.",
    ]
    (output_dir / "AACE_EXP8_MECHANISM_ABLATION.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(summary[display_columns].to_string(index=False))
    print(f"Saved ablation outputs to {output_dir}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    result.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument(
        "--resolution-objective-fraction", type=float, default=0.80
    )
    result.add_argument(
        "--resolution-minimum-stability", type=float, default=0.75
    )
    result.add_argument("--residual-components", type=int, default=24)
    result.add_argument("--raw-components", type=int, default=16)
    result.add_argument("--trajectory-components", type=int, default=24)
    result.add_argument("--temperature-components", type=int, default=16)
    result.add_argument("--coupling-components", type=int, default=8)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
