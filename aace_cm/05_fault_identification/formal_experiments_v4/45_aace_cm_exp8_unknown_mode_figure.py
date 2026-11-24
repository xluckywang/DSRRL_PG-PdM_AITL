"""Create publication assets for AACE-CM Exp8 unknown-mode discovery."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE, trustworthiness


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPRESENTATION = (
    SCRIPT_DIR / "outputs" / "aace_cm_exp8_unknown_modes" / "representation"
)
DEFAULT_RESULT = (
    SCRIPT_DIR / "outputs" / "aace_cm_exp8_unknown_modes" / "formal"
)
DEFAULT_FIGURE_DIR = (
    SCRIPT_DIR.parent
    / "paper_figure_table_notebook"
    / "exp8_section_package"
    / "figures"
    / "aace_unknown_modes"
)
CLASS_ORDER = [
    "background_control",
    "crack",
    "imbalance",
    "mechanical_looseness",
    "thermal",
]
CLASS_LABELS = {
    "background_control": "Background",
    "crack": "Crack",
    "imbalance": "Imbalance",
    "mechanical_looseness": "Mechanical looseness",
    "thermal": "Thermal",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        path.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
        bbox_inches="tight",
    )


def mode_tsne(
    representation_dir: Path,
    result_dir: Path,
    figure_dir: Path,
    seed: int,
    maximum_points: int,
) -> dict:
    metadata = pd.read_parquet(
        representation_dir / "aace_exp8_anonymous_metadata.parquet"
    )
    embedding = np.load(result_dir / "aace_exp8_final_embedding.npy")
    assignments = pd.read_parquet(
        result_dir / "aace_exp8_posthoc_assignments.parquet"
    )
    uid_to_index = {
        uid: index for index, uid in enumerate(metadata["scenario_uid"])
    }
    assignments = assignments.loc[
        assignments["scenario_uid"].isin(uid_to_index)
    ].copy()
    if len(assignments) > maximum_points:
        positions = (
            np.linspace(0, len(assignments) - 1, maximum_points)
            .round()
            .astype(int)
        )
        assignments = assignments.iloc[np.unique(positions)].copy()
    indices = np.asarray(
        [uid_to_index[uid] for uid in assignments["scenario_uid"]], dtype=int
    )
    values = embedding[indices]
    perplexity = min(30, max(5, (len(values) - 1) // 5))
    tsne_arguments = {
        "n_components": 2,
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": seed,
    }
    iteration_name = (
        "max_iter"
        if "max_iter" in inspect.signature(TSNE).parameters
        else "n_iter"
    )
    tsne_arguments[iteration_name] = 1500
    coordinates = TSNE(**tsne_arguments).fit_transform(values)
    trust = float(
        trustworthiness(
            values,
            coordinates,
            n_neighbors=min(15, max(2, len(values) // 10)),
        )
    )
    coordinate_table = assignments[
        [
            "scenario_uid",
            "role",
            "assigned_mode_id",
            "posthoc_class",
            "severity_level",
            "inject_fraction",
        ]
    ].copy()
    coordinate_table["tsne_1"] = coordinates[:, 0]
    coordinate_table["tsne_2"] = coordinates[:, 1]
    coordinate_table.to_csv(
        result_dir / "aace_exp8_tsne_coordinates.csv", index=False
    )

    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharex=True, sharey=True)
    mode_ids = sorted(assignments["assigned_mode_id"].unique())
    mode_cmap = plt.get_cmap("tab10")
    for color_index, mode in enumerate(mode_ids):
        mask = assignments["assigned_mode_id"].eq(mode).to_numpy()
        axes[0].scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=13,
            alpha=0.78,
            color=mode_cmap(color_index % 10),
            edgecolors="none",
            label=f"Mode {int(mode) + 1}",
        )
    class_colors = {
        "background_control": "#8c8c8c",
        "crack": "#d73027",
        "imbalance": "#4575b4",
        "mechanical_looseness": "#1a9850",
        "thermal": "#984ea3",
    }
    for name in CLASS_ORDER:
        mask = assignments["posthoc_class"].eq(name).to_numpy()
        if not mask.any():
            continue
        axes[1].scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=13,
            alpha=0.78,
            color=class_colors[name],
            edgecolors="none",
            label=CLASS_LABELS[name],
        )
    axes[0].set_title("Label-free discovered modes", fontweight="bold")
    axes[1].set_title("Post-hoc engineering evidence", fontweight="bold")
    for axis in axes:
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")
        for spine in axis.spines.values():
            spine.set_visible(True)
        axis.legend(
            loc="best",
            frameon=False,
            ncol=2 if len(mode_ids) > 4 else 1,
            handletextpad=0.3,
            columnspacing=0.7,
        )
    figure.tight_layout(w_pad=1.5)
    save_figure(figure, figure_dir / "aace_exp8_unknown_mode_tsne")
    plt.close(figure)
    return {
        "sampled_points": int(len(values)),
        "perplexity": int(perplexity),
        "trustworthiness_15nn": trust,
        "labels_used_to_fit_tsne": False,
        "labels_used_to_color_right_panel_only": True,
    }


def confusion_and_budget(result_dir: Path, figure_dir: Path) -> None:
    matrix = pd.read_csv(
        result_dir / "aace_exp8_query_confusion.csv", index_col=0
    ).reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0)
    active = [
        name
        for name in CLASS_ORDER
        if matrix.loc[name].sum() > 0 or matrix[name].sum() > 0
    ]
    matrix = matrix.loc[active, active]
    budget = pd.read_csv(
        result_dir / "aace_exp8_sparse_naming_metrics.csv"
    )
    values = matrix.to_numpy(dtype=float)
    row_sum = np.maximum(values.sum(axis=1, keepdims=True), 1.0)
    normalized = values / row_sum
    configure_style()
    figure, axes = plt.subplots(
        1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.18, 0.82]}
    )
    image = axes[0].imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    labels = [CLASS_LABELS[name] for name in active]
    axes[0].set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_xlabel("Named fault mode")
    axes[0].set_ylabel("Post-hoc reference")
    for row in range(len(labels)):
        for column in range(len(labels)):
            color = "white" if normalized[row, column] > 0.55 else "black"
            axes[0].text(
                column,
                row,
                f"{int(values[row, column])}\n{normalized[row, column] * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=7.5,
                color=color,
            )
    colorbar = figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("Row-normalized rate")
    axes[1].plot(
        budget["labels_per_core"],
        budget["fault_balanced_accuracy"] * 100,
        marker="o",
        linewidth=1.8,
        label="Balanced accuracy",
    )
    axes[1].plot(
        budget["labels_per_core"],
        budget["fault_macro_f1"] * 100,
        marker="s",
        linewidth=1.8,
        label="Macro-F1",
    )
    axes[1].set_xlabel("Confirmed records per core")
    axes[1].set_ylabel("Fault-mode performance (%)")
    axes[1].set_xticks(budget["labels_per_core"])
    lower = max(
        0.0,
        100.0
        * min(
            budget["fault_balanced_accuracy"].min(),
            budget["fault_macro_f1"].min(),
        )
        - 2.0,
    )
    axes[1].set_ylim(lower, 100.0)
    axes[1].legend(frameon=False)
    for axis in axes:
        for spine in axis.spines.values():
            spine.set_visible(True)
    figure.tight_layout(w_pad=1.5)
    save_figure(
        figure, figure_dir / "aace_exp8_query_confusion_and_label_budget"
    )
    plt.close(figure)


def routing_heatmap(
    representation_dir: Path,
    result_dir: Path,
    figure_dir: Path,
) -> None:
    metadata = pd.read_parquet(
        representation_dir / "aace_exp8_anonymous_metadata.parquet"
    )
    labels = pd.read_parquet(
        representation_dir / "SEALED_aace_exp8_posthoc_labels.parquet"
    )
    frame = metadata.merge(labels, on="scenario_uid", how="left")
    frame = frame.loc[
        frame["split"].eq("frozen_query")
        & frame["posthoc_class"].ne("background_control")
    ]
    pivot = frame.pivot_table(
        index="posthoc_class",
        columns="severity_level",
        values="routed_to_exp8",
        aggfunc="mean",
    ).reindex(index=CLASS_ORDER[1:])
    configure_style()
    figure, axis = plt.subplots(figsize=(6.8, 2.45))
    image = axis.imshow(
        pivot.to_numpy(dtype=float),
        cmap=ListedColormap(
            ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45"]
        ),
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    axis.set_xticks(
        range(len(pivot.columns)),
        [str(int(value)) for value in pivot.columns],
    )
    axis.set_yticks(
        range(len(pivot.index)),
        [CLASS_LABELS[value] for value in pivot.index],
    )
    axis.set_xlabel("Injected severity level (post-hoc audit only)")
    axis.set_ylabel("Fault mechanism")
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            value = pivot.iloc[row, column]
            axis.text(
                column,
                row,
                "N/A" if pd.isna(value) else f"{value * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if value >= 0.75 else "black",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.03)
    colorbar.set_label("Routing rate")
    for spine in axis.spines.values():
        spine.set_visible(True)
    figure.tight_layout()
    save_figure(figure, figure_dir / "aace_exp8_routing_by_severity")
    plt.close(figure)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    frame = frame.copy()
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("nan" if np.isnan(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_tables(result_dir: Path) -> None:
    result = json.loads(
        (result_dir / "aace_exp8_formal_result.json").read_text(
            encoding="utf-8"
        )
    )
    rows = []
    for stage, metrics in result["unsupervised_posthoc"].items():
        rows.append({"stage": stage, **metrics})
    pd.DataFrame(rows).to_csv(
        result_dir / "aace_exp8_table_unsupervised_metrics.csv", index=False
    )
    budget = pd.read_csv(result_dir / "aace_exp8_sparse_naming_metrics.csv")
    per_class = pd.read_csv(
        result_dir / "aace_exp8_sparse_naming_per_class.csv"
    )
    lines = [
        "# AACE-CM Exp8 formal tables",
        "",
        "## Label-free core and query audit",
        "",
        markdown_table(pd.DataFrame(rows)),
        "",
        "## Sparse semantic naming",
        "",
        markdown_table(budget),
        "",
        "## Per-class results",
        "",
        markdown_table(per_class),
        "",
    ]
    (result_dir / "AACE_EXP8_FORMAL_TABLES.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run(args: argparse.Namespace) -> None:
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    if not (args.result_dir / "aace_exp8_formal_result.json").exists():
        raise FileNotFoundError("Formal Exp8 result is missing")
    tsne_manifest = mode_tsne(
        args.representation_dir,
        args.result_dir,
        args.figure_dir,
        args.seed,
        args.maximum_tsne_points,
    )
    confusion_and_budget(args.result_dir, args.figure_dir)
    routing_heatmap(
        args.representation_dir, args.result_dir, args.figure_dir
    )
    write_tables(args.result_dir)
    manifest = {
        "schema_version": "aace-exp8-paper-assets-v1",
        "figures": [
            "aace_exp8_unknown_mode_tsne",
            "aace_exp8_query_confusion_and_label_budget",
            "aace_exp8_routing_by_severity",
        ],
        "tiff_dpi": 600,
        "tsne": tsne_manifest,
    }
    (args.result_dir / "aace_exp8_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


def self_test() -> None:
    matrix = np.asarray([[3, 1], [0, 4]])
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    assert np.allclose(normalized.sum(axis=1), 1.0)
    print("AACE-CM Exp8 figure self-test passed")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("run", "self-test"))
    result.add_argument(
        "--representation-dir", type=Path, default=DEFAULT_REPRESENTATION
    )
    result.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    result.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--maximum-tsne-points", type=int, default=4000)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.mode == "self-test":
        self_test()
    else:
        run(arguments)
