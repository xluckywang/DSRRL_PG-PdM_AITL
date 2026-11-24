#!/usr/bin/env python
"""Publication figures for the frozen AACE-CM Exp8 V2 cluster geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors


MODE_COLORS = {
    0: "#0072B2",
    1: "#E69F00",
    2: "#009E73",
    3: "#D55E00",
}
CLASS_COLORS = {
    "crack": "#D55E00",
    "imbalance": "#0072B2",
    "mechanical_looseness": "#009E73",
    "thermal": "#CC79A7",
}
CLASS_LABELS = {
    "crack": "Crack",
    "imbalance": "Imbalance",
    "mechanical_looseness": "Mechanical looseness",
    "thermal": "Thermal",
}
K_VALUES = np.asarray([1, 3, 5, 10, 20, 30, 50], dtype=int)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 1.0,
            "savefig.dpi": 600,
        }
    )


def close_axes(axis: plt.Axes) -> None:
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    axis.tick_params(width=1.0, length=3.5)


def save_figure(figure: plt.Figure, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path_without_suffix.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    figure.savefig(
        path_without_suffix.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.04,
    )
    figure.savefig(
        path_without_suffix.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def load_frozen_data(
    representation_dir: Path, result_dir: Path
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    metadata = pd.read_parquet(
        representation_dir / "aace_exp8_anonymous_metadata.parquet"
    )
    assignments = pd.read_parquet(
        result_dir / "aace_exp8_posthoc_assignments.parquet"
    )
    embedding = np.load(result_dir / "aace_exp8_final_embedding.npy")
    centers = np.load(result_dir / "aace_exp8_core_centers.npy")

    uid_to_index = {
        uid: index for index, uid in enumerate(metadata["scenario_uid"].astype(str))
    }
    if not assignments["scenario_uid"].astype(str).isin(uid_to_index).all():
        raise ValueError("Some frozen assignments are absent from anonymous metadata.")
    indices = np.asarray(
        [uid_to_index[str(uid)] for uid in assignments["scenario_uid"]], dtype=int
    )
    values = np.asarray(embedding[indices], dtype=np.float64)
    if values.ndim != 2 or centers.shape != (
        assignments["assigned_mode_id"].nunique(),
        values.shape[1],
    ):
        raise ValueError(
            f"Incompatible embedding/center shapes: {values.shape}, {centers.shape}."
        )
    return assignments.reset_index(drop=True), values, np.asarray(centers, dtype=float)


def fit_or_load_pacmap(
    assignments: pd.DataFrame,
    values: np.ndarray,
    coordinate_file: Path,
    seed: int,
) -> np.ndarray:
    if coordinate_file.exists():
        coordinates = pd.read_csv(coordinate_file)
        expected = assignments["scenario_uid"].astype(str).tolist()
        actual = coordinates["scenario_uid"].astype(str).tolist()
        if expected != actual:
            raise ValueError("PaCMAP coordinate order does not match frozen assignments.")
        return coordinates[["pacmap_1", "pacmap_2"]].to_numpy(dtype=float)

    try:
        import pacmap
    except ImportError as exc:
        raise RuntimeError(
            "PaCMAP is unavailable. Generate coordinates in an environment with "
            "pacmap>=0.8, then pass the resulting CSV with --coordinate-file."
        ) from exc

    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=15,
        MN_ratio=0.5,
        FP_ratio=2.0,
        random_state=seed,
    )
    coordinates = reducer.fit_transform(values, init="pca")
    coordinate_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "scenario_uid": assignments["scenario_uid"].astype(str),
            "pacmap_1": coordinates[:, 0],
            "pacmap_2": coordinates[:, 1],
        }
    ).to_csv(coordinate_file, index=False)
    return np.asarray(coordinates, dtype=float)


def plot_pacmap(
    assignments: pd.DataFrame, coordinates: np.ndarray, output_dir: Path
) -> None:
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharex=True, sharey=True)

    for mode in sorted(assignments["assigned_mode_id"].unique()):
        mask = assignments["assigned_mode_id"].eq(mode).to_numpy()
        axes[0].scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=8,
            alpha=0.62,
            color=MODE_COLORS[int(mode)],
            edgecolors="none",
            rasterized=True,
            label=f"Mode {int(mode) + 1}",
        )
    for name in CLASS_LABELS:
        mask = assignments["posthoc_class"].eq(name).to_numpy()
        axes[1].scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=8,
            alpha=0.62,
            color=CLASS_COLORS[name],
            edgecolors="none",
            rasterized=True,
            label=CLASS_LABELS[name],
        )

    axes[0].set_title("Label-free discovered modes", fontweight="bold")
    axes[1].set_title("Post-hoc engineering evidence", fontweight="bold")
    for axis in axes:
        axis.set_xlabel("PaCMAP 1")
        axis.set_ylabel("PaCMAP 2")
        close_axes(axis)
        axis.legend(
            frameon=False,
            loc="best",
            handletextpad=0.25,
            borderaxespad=0.2,
            markerscale=1.4,
        )
    figure.tight_layout(w_pad=1.4)
    save_figure(figure, output_dir / "aace_exp8_pacmap_projection")
    plt.close(figure)


def plot_pacmap_3d_static(
    assignments: pd.DataFrame,
    result_dir: Path,
    output_dir: Path,
) -> None:
    """Render the frozen 3-D PaCMAP coordinates at the explorer's ISO-like view."""
    coordinate_path = result_dir / "aace_exp8_pacmap_3d_coordinates.csv"
    coordinate_frame = pd.read_csv(coordinate_path)
    expected = assignments["scenario_uid"].astype(str).tolist()
    actual = coordinate_frame["scenario_uid"].astype(str).tolist()
    if expected != actual:
        raise ValueError("3-D PaCMAP coordinate order does not match assignments.")
    coordinates = coordinate_frame[
        ["pacmap_1", "pacmap_2", "pacmap_3"]
    ].to_numpy(dtype=float)

    configure_style()
    figure = plt.figure(figsize=(7.4, 5.3))
    axis = figure.add_subplot(111, projection="3d")
    for mode in sorted(assignments["assigned_mode_id"].unique()):
        mask = assignments["assigned_mode_id"].eq(mode).to_numpy()
        axis.scatter(
            coordinates[mask, 1],
            coordinates[mask, 0],
            coordinates[mask, 2],
            s=4.2,
            alpha=0.82,
            color=MODE_COLORS[int(mode)],
            edgecolors="none",
            depthshade=False,
            rasterized=True,
            label=f"Mode {int(mode) + 1}",
        )

    # Matplotlib renders the x-axis on the left and the y-axis on the right
    # for this fixed camera. Map the frozen coordinates accordingly so the
    # screen-space axis positions match the interactive reference view.
    axis.set_xlabel("PaCMAP 2", labelpad=1)
    axis.set_ylabel("PaCMAP 1", labelpad=1)
    axis.set_zlabel("")
    figure.text(
        0.875,
        0.52,
        "PaCMAP 3",
        rotation=90,
        ha="center",
        va="center",
        fontsize=11,
    )
    axis.view_init(elev=21, azim=132)
    axis.invert_zaxis()
    coordinate_ranges = np.ptp(coordinates, axis=0)
    axis.set_box_aspect(
        [coordinate_ranges[1], coordinate_ranges[0], coordinate_ranges[2]]
    )
    axis.grid(True)
    for pane in [axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane]:
        pane.set_facecolor((0.985, 0.988, 0.990, 1.0))
        pane.set_edgecolor((0.84, 0.86, 0.87, 1.0))
    for axis_component in [axis.xaxis, axis.yaxis, axis.zaxis]:
        axis_component._axinfo["grid"]["color"] = (0.84, 0.86, 0.87, 0.70)
        axis_component._axinfo["grid"]["linewidth"] = 0.55
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.17, 0.80),
        frameon=True,
        facecolor="white",
        edgecolor="#c8d0d4",
        framealpha=0.92,
        markerscale=1.7,
        fontsize=8.5,
    )
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.99)
    save_figure(
        figure,
        output_dir / "aace_exp8_pacmap_3d_iso_static",
    )
    plt.close(figure)


def plot_pacmap_topdown_2d(
    assignments: pd.DataFrame,
    result_dir: Path,
    output_dir: Path,
) -> None:
    """Render the PaCMAP-1/2 plane used by the user's preferred top view."""
    coordinate_frame = pd.read_csv(
        result_dir / "aace_exp8_pacmap_3d_coordinates.csv"
    )
    if (
        assignments["scenario_uid"].astype(str).tolist()
        != coordinate_frame["scenario_uid"].astype(str).tolist()
    ):
        raise ValueError("3-D PaCMAP coordinate order does not match assignments.")

    configure_style()
    # Match the companion prototype-separation panel so both figures retain
    # the same apparent typography when assembled at equal panel width.
    figure, axis = plt.subplots(figsize=(4.3, 3.7))
    for mode in sorted(assignments["assigned_mode_id"].unique()):
        mask = assignments["assigned_mode_id"].eq(mode).to_numpy()
        axis.scatter(
            coordinate_frame.loc[mask, "pacmap_2"],
            coordinate_frame.loc[mask, "pacmap_1"],
            s=7,
            alpha=0.78,
            color=MODE_COLORS[int(mode)],
            edgecolors="none",
            rasterized=True,
            label=f"Mode {int(mode) + 1}",
        )
    axis.set_xlabel("PaCMAP 2")
    axis.set_ylabel("PaCMAP 1")
    axis.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#c8d0d4",
        framealpha=0.92,
        markerscale=1.5,
    )
    close_axes(axis)
    figure.tight_layout()
    save_figure(figure, output_dir / "aace_exp8_pacmap2_vs_pacmap1")
    plt.close(figure)


def plot_prototype_separation_individual(
    center_distances: pd.DataFrame,
    output_dir: Path,
) -> None:
    configure_style()
    matrix = center_distances.to_numpy(dtype=float)
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    figure, axis = plt.subplots(figsize=(4.3, 3.7))
    image = axis.imshow(
        matrix,
        cmap="YlGnBu",
        vmin=0,
        vmax=float(upper.max()),
        interpolation="nearest",
        aspect="equal",
    )
    labels = center_distances.index.tolist()
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    threshold = float(upper.max()) * 0.55
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if matrix[row, column] > threshold else "black",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Euclidean distance", fontsize=10)
    colorbar.ax.tick_params(labelsize=8.5)
    close_axes(axis)
    figure.tight_layout()
    save_figure(figure, output_dir / "aace_exp8_prototype_separation")
    plt.close(figure)


def plot_local_consistency_individual(
    purity: pd.DataFrame,
    output_dir: Path,
) -> None:
    configure_style()
    figure, axis = plt.subplots(figsize=(4.8, 3.7))
    axis.plot(
        purity["k"],
        100 * purity["mode_purity"],
        marker="o",
        color="#0072B2",
        linewidth=1.8,
        markersize=4.8,
        label="Discovered mode",
    )
    axis.plot(
        purity["k"],
        100 * purity["engineering_purity"],
        marker="s",
        color="#D55E00",
        linewidth=1.8,
        markersize=4.5,
        label="Engineering class",
    )
    axis.set_xscale("log")
    axis.set_xticks(K_VALUES, [str(value) for value in K_VALUES])
    axis.set_ylim(75, 100)
    axis.set_xlabel("Number of nearest neighbors, $k$")
    axis.set_ylabel("Neighborhood purity (%)")
    axis.legend(frameon=False, loc="lower left")
    close_axes(axis)
    figure.tight_layout()
    save_figure(figure, output_dir / "aace_exp8_local_consistency")
    plt.close(figure)


def plot_main_geometry_composite(
    assignments: pd.DataFrame,
    result_dir: Path,
    center_distances: pd.DataFrame,
    purity: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Combine the intuitive 3-D view with two frozen-space geometry audits."""
    coordinate_frame = pd.read_csv(
        result_dir / "aace_exp8_pacmap_3d_coordinates.csv"
    )
    if (
        assignments["scenario_uid"].astype(str).tolist()
        != coordinate_frame["scenario_uid"].astype(str).tolist()
    ):
        raise ValueError("3-D PaCMAP coordinate order does not match assignments.")
    coordinates = coordinate_frame[
        ["pacmap_1", "pacmap_2", "pacmap_3"]
    ].to_numpy(dtype=float)

    configure_style()
    figure = plt.figure(figsize=(11.2, 5.15))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[2.15, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.24,
        hspace=0.34,
    )
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    for mode in sorted(assignments["assigned_mode_id"].unique()):
        mask = assignments["assigned_mode_id"].eq(mode).to_numpy()
        axis_3d.scatter(
            coordinates[mask, 1],
            coordinates[mask, 0],
            coordinates[mask, 2],
            s=3.8,
            alpha=0.82,
            color=MODE_COLORS[int(mode)],
            edgecolors="none",
            depthshade=False,
            rasterized=True,
            label=f"Mode {int(mode) + 1}",
        )
    axis_3d.set_xlabel("PaCMAP 2", labelpad=1)
    axis_3d.set_ylabel("PaCMAP 1", labelpad=1)
    axis_3d.set_zlabel("PaCMAP 3", labelpad=1)
    axis_3d.view_init(elev=21, azim=132)
    axis_3d.invert_zaxis()
    coordinate_ranges = np.ptp(coordinates, axis=0)
    axis_3d.set_box_aspect(
        [coordinate_ranges[1], coordinate_ranges[0], coordinate_ranges[2]]
    )
    for pane in [
        axis_3d.xaxis.pane,
        axis_3d.yaxis.pane,
        axis_3d.zaxis.pane,
    ]:
        pane.set_facecolor((0.985, 0.988, 0.990, 1.0))
        pane.set_edgecolor((0.84, 0.86, 0.87, 1.0))
    for axis_component in [
        axis_3d.xaxis,
        axis_3d.yaxis,
        axis_3d.zaxis,
    ]:
        axis_component._axinfo["grid"]["color"] = (0.84, 0.86, 0.87, 0.70)
        axis_component._axinfo["grid"]["linewidth"] = 0.55
    axis_3d.legend(
        loc="upper left",
        bbox_to_anchor=(0.14, 0.82),
        frameon=True,
        facecolor="white",
        edgecolor="#c8d0d4",
        framealpha=0.92,
        markerscale=1.7,
        fontsize=8.0,
    )

    axis_distance = figure.add_subplot(grid[0, 1])
    distance_values = center_distances.to_numpy(dtype=float)
    image = axis_distance.imshow(
        distance_values, cmap="YlGnBu", vmin=0.0, vmax=1.70
    )
    labels = list(center_distances.index)
    axis_distance.set_xticks(
        np.arange(len(labels)), labels, rotation=30, ha="right"
    )
    axis_distance.set_yticks(np.arange(len(labels)), labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = distance_values[row, column]
            axis_distance.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 1.0 else "black",
            )
    axis_distance.set_title("Prototype separation", fontsize=10)
    colorbar = figure.colorbar(
        image, ax=axis_distance, fraction=0.046, pad=0.04
    )
    colorbar.set_label("Euclidean distance", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=7.5)
    close_axes(axis_distance)

    axis_purity = figure.add_subplot(grid[1, 1])
    axis_purity.plot(
        purity["k"],
        100 * purity["mode_purity"],
        marker="o",
        linewidth=1.8,
        label="Discovered mode",
        color="#147D92",
    )
    axis_purity.plot(
        purity["k"],
        100 * purity["engineering_purity"],
        marker="s",
        linewidth=1.8,
        label="Engineering class",
        color="#D45A3A",
    )
    axis_purity.set_xlabel("Number of nearest neighbors, $k$")
    axis_purity.set_ylabel("Neighborhood purity (%)")
    axis_purity.set_ylim(75, 100)
    axis_purity.set_title("Local consistency in frozen 24-D space", fontsize=10)
    axis_purity.legend(frameon=False, fontsize=7.5, loc="lower left")
    close_axes(axis_purity)

    figure.subplots_adjust(left=0.03, right=0.98, bottom=0.10, top=0.97)
    save_figure(
        figure,
        output_dir / "aace_exp8_v2_mode_geometry_composite",
    )
    plt.close(figure)


def plot_pacmap_vs_prototype_margin_2d(
    assignments: pd.DataFrame,
    values: np.ndarray,
    centers: np.ndarray,
    coordinates: np.ndarray,
    mode_mapping: dict[int, str],
    output_dir: Path,
) -> None:
    """Compare all-mode PaCMAP with a 2-D prototype-margin projection."""
    configure_style()
    inverse_mapping = {name: mode for mode, name in mode_mapping.items()}
    crack_mode = inverse_mapping["crack"]
    looseness_mode = inverse_mapping["mechanical_looseness"]
    difficult_modes = [crack_mode, looseness_mode]

    distances = cdist(values, centers, metric="euclidean")
    # Two axes from the exact 3-D margin space used by the interactive explorer:
    # [d(Mode 4)-d(Mode 1), d(Mode 4)-d(Mode 2), d(Mode 4)-d(Mode 3)].
    mode4_distance = distances[:, 3]
    margin_mode4_mode2 = mode4_distance - distances[:, 1]
    margin_mode4_mode3 = mode4_distance - distances[:, 2]

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.35))
    for mode in sorted(assignments["assigned_mode_id"].unique()):
        mask = assignments["assigned_mode_id"].eq(mode).to_numpy()
        mode_label = f"Mode {int(mode) + 1}"
        axes[0].scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=8,
            alpha=0.55,
            color=MODE_COLORS[int(mode)],
            edgecolors="none",
            rasterized=True,
            label=mode_label,
        )
    for mode in difficult_modes:
        mask = assignments["assigned_mode_id"].eq(mode).to_numpy()
        mode_label = f"Mode {int(mode) + 1}"
        axes[1].scatter(
            margin_mode4_mode2[mask],
            margin_mode4_mode3[mask],
            s=8,
            alpha=0.60,
            color=MODE_COLORS[int(mode)],
            edgecolors="none",
            rasterized=True,
            label=mode_label,
        )

    axes[0].set_xlabel("PaCMAP 1")
    axes[0].set_ylabel("PaCMAP 2")
    axes[0].set_title("PaCMAP projection", fontweight="bold")

    axes[1].set_xlabel(
        r"$d(\mathrm{Mode\ 4})-d(\mathrm{Mode\ 2})$"
    )
    axes[1].set_ylabel(
        r"$d(\mathrm{Mode\ 4})-d(\mathrm{Mode\ 3})$"
    )
    axes[1].set_title("Prototype-margin view", fontweight="bold")

    for axis in axes:
        close_axes(axis)
        axis.legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.90,
            loc="best",
            handletextpad=0.25,
            borderaxespad=0.25,
            markerscale=1.4,
        )

    figure.tight_layout(w_pad=1.8)
    save_figure(
        figure,
        output_dir / "aace_exp8_pacmap_vs_prototype_margin_2d",
    )
    plt.close(figure)


def prototype_distance_table(centers: np.ndarray) -> pd.DataFrame:
    distances = cdist(centers, centers, metric="euclidean")
    labels = [f"Mode {index + 1}" for index in range(len(centers))]
    return pd.DataFrame(distances, index=labels, columns=labels)


def neighborhood_purity(
    values: np.ndarray, assignments: pd.DataFrame
) -> pd.DataFrame:
    model = NearestNeighbors(n_neighbors=int(K_VALUES.max()) + 1, metric="euclidean")
    neighbors = model.fit(values).kneighbors(return_distance=False)[:, 1:]
    mode = assignments["assigned_mode_id"].to_numpy()
    engineering = assignments["posthoc_class"].astype(str).to_numpy()
    rows: list[dict] = []
    for k in K_VALUES:
        local = neighbors[:, : int(k)]
        rows.append(
            {
                "k": int(k),
                "mode_purity": float(np.mean(mode[local] == mode[:, None])),
                "engineering_purity": float(
                    np.mean(engineering[local] == engineering[:, None])
                ),
            }
        )
    return pd.DataFrame(rows)


def prototype_distance_records(
    values: np.ndarray, centers: np.ndarray, assignments: pd.DataFrame
) -> pd.DataFrame:
    modes = assignments["assigned_mode_id"].to_numpy(dtype=int)
    raw_distance = np.linalg.norm(values - centers[modes], axis=1)
    pooled_median = float(np.median(raw_distance))
    if pooled_median <= 0:
        raise ValueError("The pooled prototype-distance median must be positive.")
    return pd.DataFrame(
        {
            "scenario_uid": assignments["scenario_uid"].astype(str),
            "assigned_mode_id": modes,
            "posthoc_class": assignments["posthoc_class"].astype(str),
            "prototype_distance": raw_distance,
            "normalized_prototype_distance": raw_distance / pooled_median,
        }
    )


def sparse_mode_mapping(result_dir: Path) -> dict[int, str]:
    evidence = json.loads(
        (result_dir / "aace_exp8_sparse_naming_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    mapping = {
        int(record["cluster_id"]): str(record["assigned_engineering_name"])
        for record in evidence
        if int(record["label_budget"]) == 1
    }
    if set(mapping) != {0, 1, 2, 3}:
        raise ValueError(f"Unexpected one-record mode mapping: {mapping}")
    return mapping


def difficult_pair_records(
    assignments: pd.DataFrame,
    values: np.ndarray,
    centers: np.ndarray,
    mode_mapping: dict[int, str],
) -> pd.DataFrame:
    inverse_mapping = {name: mode for mode, name in mode_mapping.items()}
    crack_center = centers[inverse_mapping["crack"]]
    looseness_center = centers[inverse_mapping["mechanical_looseness"]]
    crack_distance = np.linalg.norm(values - crack_center, axis=1)
    looseness_distance = np.linalg.norm(values - looseness_center, axis=1)
    records = assignments.copy()
    records["distance_to_crack_prototype"] = crack_distance
    records["distance_to_looseness_prototype"] = looseness_distance
    records["signed_prototype_margin"] = looseness_distance - crack_distance
    return records.loc[
        records["role"].eq("frozen_query")
        & records["posthoc_class"].isin(["crack", "mechanical_looseness"])
    ].reset_index(drop=True)


def plot_mode_correspondence_and_boundary(
    assignments: pd.DataFrame,
    pair_records: pd.DataFrame,
    output_dir: Path,
) -> None:
    configure_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(8.6, 3.35),
        gridspec_kw={"width_ratios": [1.0, 1.3]},
    )

    class_order = ["crack", "imbalance", "mechanical_looseness", "thermal"]
    query = assignments.loc[assignments["role"].eq("frozen_query")].copy()
    counts = pd.crosstab(query["assigned_mode_id"], query["posthoc_class"]).reindex(
        index=[0, 1, 2, 3], columns=class_order, fill_value=0
    )
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    image = axes[0].imshow(
        percentages.to_numpy(),
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        interpolation="nearest",
        aspect="auto",
    )
    axes[0].set_xticks(
        range(4), ["Crack", "Imbalance", "Mechanical\nlooseness", "Thermal"]
    )
    axes[0].set_yticks(range(4), [f"Mode {index}" for index in range(1, 5)])
    axes[0].set_xlabel("Post-hoc engineering class")
    axes[0].set_ylabel("Discovered mode")
    axes[0].set_title("Mode–class correspondence", fontweight="bold")
    for row in range(4):
        for column in range(4):
            percentage = percentages.iloc[row, column]
            count = int(counts.iloc[row, column])
            axes[0].text(
                column,
                row,
                f"{percentage:.1f}%\n($n$={count})",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if percentage >= 55 else "black",
            )
    colorbar = figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("Within-mode proportion (%)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    close_axes(axes[0])

    offsets = {"crack": -0.14, "mechanical_looseness": 0.14}
    markers = {"crack": "o", "mechanical_looseness": "s"}
    for class_name in ["crack", "mechanical_looseness"]:
        subset = pair_records.loc[pair_records["posthoc_class"].eq(class_name)]
        margin = subset["signed_prototype_margin"].to_numpy()
        correct = (margin > 0) if class_name == "crack" else (margin < 0)
        for severity in range(1, 11):
            severity_values = subset.loc[
                subset["severity_level"].eq(severity), "signed_prototype_margin"
            ].to_numpy()
            q1, median, q3 = np.quantile(severity_values, [0.25, 0.5, 0.75])
            axes[1].errorbar(
                median,
                severity + offsets[class_name],
                xerr=[[median - q1], [q3 - median]],
                fmt=markers[class_name],
                color=CLASS_COLORS[class_name],
                markersize=4.8,
                elinewidth=1.6,
                capsize=2.2,
                capthick=1.2,
                label=CLASS_LABELS[class_name] if severity == 1 else None,
            )
        wrong = subset.loc[~correct]
        axes[1].scatter(
            wrong["signed_prototype_margin"],
            wrong["severity_level"] + offsets[class_name],
            marker="x",
            s=17,
            linewidths=0.8,
            color=CLASS_COLORS[class_name],
            alpha=0.55,
            rasterized=True,
        )
    axes[1].axvline(0, color="#222222", linestyle="--", linewidth=1.2)
    axes[1].text(
        0.02, 0.97, "\u2190 Mechanical looseness (97.3%)",
        transform=axes[1].transAxes,
        ha="left", va="top", fontsize=8,
        color=CLASS_COLORS["mechanical_looseness"],
    )
    axes[1].text(
        0.98, 0.97, "Crack (90.0%) \u2192",
        transform=axes[1].transAxes,
        ha="right", va="top", fontsize=8,
        color=CLASS_COLORS["crack"],
    )
    axes[1].set_yticks(range(1, 11))
    axes[1].set_ylim(0.5, 10.5)
    axes[1].set_xlabel(
        r"$d(\mathrm{looseness\ prototype})-d(\mathrm{crack\ prototype})$"
    )
    axes[1].set_ylabel("Fault severity")
    axes[1].set_title("Crack–looseness boundary in 24-D", fontweight="bold")
    close_axes(axes[1])

    figure.tight_layout(w_pad=1.5)
    save_figure(
        figure, output_dir / "aace_exp8_mode_correspondence_and_boundary"
    )
    plt.close(figure)


def plot_geometry_diagnostics(
    center_distances: pd.DataFrame,
    purity: pd.DataFrame,
    output_dir: Path,
) -> None:
    configure_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.05),
        gridspec_kw={"width_ratios": [1.0, 1.08]},
    )

    matrix = center_distances.to_numpy()
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    image = axes[0].imshow(
        matrix,
        cmap="YlGnBu",
        vmin=0,
        vmax=float(upper.max()),
        interpolation="nearest",
        aspect="equal",
    )
    labels = center_distances.index.tolist()
    axes[0].set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axes[0].set_yticks(range(len(labels)), labels)
    threshold = float(upper.max()) * 0.55
    for row in range(len(labels)):
        for column in range(len(labels)):
            axes[0].text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if matrix[row, column] > threshold else "black",
            )
    colorbar = figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("Euclidean distance", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    axes[0].set_title("Prototype separation", fontweight="bold")
    close_axes(axes[0])

    axes[1].plot(
        purity["k"],
        100 * purity["mode_purity"],
        marker="o",
        color="#0072B2",
        linewidth=1.8,
        markersize=4.5,
        label="Discovered mode",
    )
    axes[1].plot(
        purity["k"],
        100 * purity["engineering_purity"],
        marker="s",
        color="#D55E00",
        linewidth=1.8,
        markersize=4.2,
        label="Engineering class",
    )
    axes[1].axhline(25, color="#999999", linestyle="--", linewidth=1.0)
    axes[1].set_xscale("log")
    axes[1].set_xticks(K_VALUES, [str(value) for value in K_VALUES])
    axes[1].set_ylim(20, 101)
    axes[1].set_xlabel("Number of nearest neighbors, $k$")
    axes[1].set_ylabel("Neighborhood purity (%)")
    axes[1].set_title("Local consistency in 24-D", fontweight="bold")
    axes[1].legend(frameon=False, loc="lower left")
    close_axes(axes[1])

    figure.tight_layout(w_pad=1.5)
    save_figure(figure, output_dir / "aace_exp8_cluster_geometry_diagnostics")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coordinate-file", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coordinates-only", action="store_true")
    parser.add_argument(
        "--skip-3d",
        action="store_true",
        help="Skip panels that require the separately generated 3-D coordinates.",
    )
    args = parser.parse_args()

    assignments, values, centers = load_frozen_data(
        args.representation_dir, args.result_dir
    )
    coordinate_file = (
        args.coordinate_file
        if args.coordinate_file is not None
        else args.result_dir / "aace_exp8_pacmap_coordinates.csv"
    )
    coordinates = fit_or_load_pacmap(
        assignments, values, coordinate_file, args.seed
    )
    if args.coordinates_only:
        print(f"Saved PaCMAP coordinates: {coordinate_file}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_pacmap(assignments, coordinates, args.output_dir)
    if not args.skip_3d:
        plot_pacmap_3d_static(assignments, args.result_dir, args.output_dir)
        plot_pacmap_topdown_2d(assignments, args.result_dir, args.output_dir)

    center_distances = prototype_distance_table(centers)
    purity = neighborhood_purity(values, assignments)
    plot_prototype_separation_individual(center_distances, args.output_dir)
    plot_local_consistency_individual(purity, args.output_dir)
    if not args.skip_3d:
        plot_main_geometry_composite(
            assignments,
            args.result_dir,
            center_distances,
            purity,
            args.output_dir,
        )
    distance_records = prototype_distance_records(values, centers, assignments)
    mode_mapping = sparse_mode_mapping(args.result_dir)
    plot_pacmap_vs_prototype_margin_2d(
        assignments,
        values,
        centers,
        coordinates,
        mode_mapping,
        args.output_dir,
    )
    pair_records = difficult_pair_records(
        assignments, values, centers, mode_mapping
    )
    center_distances.to_csv(args.output_dir / "aace_exp8_prototype_distances.csv")
    purity.to_csv(args.output_dir / "aace_exp8_neighborhood_purity.csv", index=False)
    distance_records.to_csv(
        args.output_dir / "aace_exp8_prototype_distance_distribution.csv",
        index=False,
    )
    pair_records.to_csv(
        args.output_dir / "aace_exp8_crack_looseness_boundary_audit.csv",
        index=False,
    )
    plot_mode_correspondence_and_boundary(
        assignments, pair_records, args.output_dir
    )
    plot_geometry_diagnostics(
        center_distances, purity, args.output_dir
    )

    manifest = {
        "schema_version": "aace-exp8-v2-cluster-geometry-figure-v1",
        "frozen_result_dir": str(args.result_dir),
        "sample_count": int(len(assignments)),
        "embedding_dimension": int(values.shape[1]),
        "pacmap": {
            "n_neighbors": 15,
            "MN_ratio": 0.5,
            "FP_ratio": 2.0,
            "seed": int(args.seed),
            "labels_used_to_fit": False,
            "trustworthiness_15nn": float(
                trustworthiness(values, coordinates, n_neighbors=15)
            ),
        },
        "neighborhood_purity_space": "frozen 24-D V2 embedding",
        "prototype_distance_space": "frozen 24-D V2 embedding",
        "main_figure": (
            None if args.skip_3d else
            "Single all-mode 3-D PaCMAP view using the frozen coordinates and "
            "a fixed ISO-like camera"
        ),
        "prototype_margin_coordinates": [
            "d(Mode 4)-d(Mode 1)",
            "d(Mode 4)-d(Mode 2)",
            "d(Mode 4)-d(Mode 3)",
        ],
        "prototype_margin_axes_shown": [
            "d(Mode 4)-d(Mode 2)",
            "d(Mode 4)-d(Mode 3)",
        ],
        "probability_density_figure_generated": False,
    }
    (args.output_dir / "aace_exp8_cluster_geometry_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
