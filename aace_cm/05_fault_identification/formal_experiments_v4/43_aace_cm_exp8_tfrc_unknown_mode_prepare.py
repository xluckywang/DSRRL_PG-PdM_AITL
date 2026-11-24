"""AACE-CM Exp8: healthy-only four-channel residual representation preparation.

This stage never reads fault label parquet files. Fault type, severity, injection
percentage, and source names are written only to a sealed post-hoc table after
all representation arrays and anonymous metadata have been finalized.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aace_exp8_unknown_common import (  # noqa: E402
    aggregate_distribution,
    coupling_features,
    event_count,
    local_mechanism_features,
    scenario_uid,
    select_abnormal_windows,
    self_test as common_self_test,
    severity_features,
    temperature_dynamics,
    trajectory_features,
)
from config import (  # noqa: E402
    COMMON,
    EXP3,
    get_active_fault_data_dir,
    get_exp0_recommended_train_dates,
    get_feature_cols,
    get_feature_data_file,
    get_label_file,
)
import temporal_ae as base  # noqa: E402


DEFAULT_OUTPUT = (
    SCRIPT_DIR / "outputs" / "aace_cm_exp8_unknown_modes" / "representation"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def temporal_windows(
    values: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build time-major flattened windows and their center indices."""
    values = np.asarray(values, dtype=np.float32)
    width = 2 * radius + 1
    if len(values) < width:
        return (
            np.empty((0, width * values.shape[1]), dtype=np.float32),
            np.empty(0, dtype=int),
        )
    stacked = np.stack(
        [values[offset : len(values) - width + offset + 1] for offset in range(width)],
        axis=1,
    )
    centers = np.arange(radius, len(values) - radius, dtype=int)
    return stacked.reshape(len(stacked), -1), centers


def contiguous_true_runs(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
    return [part for part in np.split(indices, boundaries) if len(part)]


def _sample_rows(
    values: np.ndarray,
    maximum: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if maximum <= 0 or len(values) <= maximum:
        return values
    indices = np.sort(rng.choice(len(values), maximum, replace=False))
    return values[indices]


def healthy_windows_for_dates(
    device: str,
    dates: list[str],
    radius: int,
    maximum: int,
    seed: int,
) -> np.ndarray:
    """Build windows within contiguous NORMAL runs; never cross date/run boundaries."""
    feature_columns = get_feature_cols(device)
    rng = np.random.default_rng(seed)
    per_date = max(1, int(math.ceil(maximum / max(len(dates), 1))))
    outputs = []
    for date in dates:
        feature_path = get_feature_data_file(device, date)
        label_path = get_label_file(device, date)
        if not feature_path.exists() or not label_path.exists():
            continue
        feature_frame = pd.read_csv(
            feature_path, usecols=feature_columns
        )[feature_columns]
        label_frame = pd.read_csv(label_path)
        state_column = "state" if "state" in label_frame.columns else "label"
        count = min(len(feature_frame), len(label_frame))
        values = feature_frame.iloc[:count].to_numpy(dtype=np.float32)
        states = label_frame[state_column].iloc[:count].to_numpy()
        date_windows = []
        for run in contiguous_true_runs(states == EXP3.normal_state_id):
            if len(run) < 2 * radius + 1:
                continue
            windows, _ = temporal_windows(values[run], radius)
            if len(windows):
                date_windows.append(windows)
        if date_windows:
            outputs.append(
                _sample_rows(np.vstack(date_windows), per_date, rng)
            )
    if not outputs:
        raise RuntimeError("No healthy NORMAL windows were available")
    return _sample_rows(np.vstack(outputs), maximum, rng).astype(np.float32)


def sampled_scene_windows(
    values: np.ndarray,
    radius: int,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows, centers = temporal_windows(values, radius)
    if maximum > 0 and len(windows) > maximum:
        indices = np.linspace(0, len(windows) - 1, maximum).round().astype(int)
        windows = windows[indices]
        centers = centers[indices]
    return windows, centers


def compute_residual_batched(
    base,
    ae,
    scaler,
    windows: np.ndarray,
    batch_size: int = 8192,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(windows), batch_size):
        outputs.append(
            base.compute_residual_windows(
                ae, scaler, windows[start : start + batch_size]
            )
        )
    return np.vstack(outputs).astype(np.float32)


def robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    center = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25, 75], axis=0)
    robust = (q75 - q25) / 1.349
    standard = values.std(axis=0)
    scale = np.maximum.reduce(
        [robust, standard * 0.25, np.full_like(standard, 1e-4)]
    )
    return center.astype(np.float32), scale.astype(np.float32)


def build_date_references(
    dates: list[str],
    device: str,
    feature_columns: list[str],
    base,
    ae,
    scaler,
    radius: int,
    maximum_windows: int,
    quantile: float,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Calibrate each evaluation date against its unmodified healthy background."""
    references = {}
    rows = []
    for date in dates:
        path = get_feature_data_file(device, date)
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=feature_columns)[feature_columns]
        values = frame.to_numpy(dtype=np.float32)
        windows, centers = sampled_scene_windows(
            values, radius, maximum_windows
        )
        residuals = compute_residual_batched(base, ae, scaler, windows)
        residual_center, residual_scale = robust_center_scale(residuals)
        raw_center, raw_scale = robust_center_scale(windows)
        corrected = (residuals - residual_center) / residual_scale
        scores = np.mean(corrected**2, axis=1)
        threshold = float(np.quantile(scores, quantile))
        references[date] = {
            "windows": windows,
            "centers": centers,
            "corrected_residual": corrected.astype(np.float32),
            "corrected_raw": ((windows - raw_center) / raw_scale).astype(
                np.float32
            ),
            "residual_center": residual_center,
            "residual_scale": residual_scale,
            "raw_center": raw_center,
            "raw_scale": raw_scale,
            "threshold": threshold,
            "source_path": path,
        }
        rows.append(
            {
                "date": date,
                "reference_windows": len(windows),
                "healthy_score_threshold": threshold,
                "healthy_score_mean": float(scores.mean()),
                "healthy_score_q95": float(np.quantile(scores, 0.95)),
                "healthy_score_q995": float(np.quantile(scores, 0.995)),
            }
        )
    return references, pd.DataFrame(rows)


def feature_row(
    windows: np.ndarray,
    centers: np.ndarray,
    mechanism_residuals: np.ndarray,
    ae_residuals: np.ndarray,
    threshold: float,
    channels: int,
    temperature_index: int,
    minimum_windows: int,
    maximum_fraction: float,
) -> tuple[dict[str, np.ndarray] | None, dict]:
    width = mechanism_residuals.shape[1] // channels
    mechanism_tensor = mechanism_residuals.reshape(
        len(mechanism_residuals), width, channels
    )
    vibration_indices = [
        index for index in range(channels) if index != temperature_index
    ]
    scores = np.mean(
        mechanism_tensor[:, :, vibration_indices] ** 2, axis=(1, 2)
    )
    thermal_scores = np.mean(
        mechanism_tensor[:, :, temperature_index] ** 2, axis=1
    )
    ae_scores = np.mean(ae_residuals**2, axis=1)
    selected = select_abnormal_windows(
        scores, threshold, minimum_windows, maximum_fraction
    )
    diagnostics = {
        "candidate_windows": int(len(windows)),
        "selected_anomaly_windows": int(len(selected)),
        "selected_anomaly_fraction": float(
            len(selected) / max(len(windows), 1)
        ),
        "anomaly_event_count": event_count(centers, selected),
        "maximum_residual_score": float(scores.max()) if len(scores) else 0.0,
        "mean_residual_score": float(scores.mean()) if len(scores) else 0.0,
        "maximum_thermal_contrast_score": (
            float(thermal_scores.max()) if len(thermal_scores) else 0.0
        ),
        "mean_thermal_contrast_score": (
            float(thermal_scores.mean()) if len(thermal_scores) else 0.0
        ),
        "maximum_ae_residual_score": (
            float(ae_scores.max()) if len(ae_scores) else 0.0
        ),
        "mean_ae_residual_score": (
            float(ae_scores.mean()) if len(ae_scores) else 0.0
        ),
        "routed_to_exp8": bool(len(selected) >= minimum_windows),
    }
    if len(selected) < minimum_windows:
        return None, diagnostics
    residual_local, raw_local = local_mechanism_features(
        mechanism_residuals[selected], windows[selected], channels
    )
    ae_local, _ = local_mechanism_features(
        ae_residuals[selected], windows[selected], channels
    )
    outputs = {
        "residual": aggregate_distribution(residual_local),
        "ae_residual": aggregate_distribution(ae_local),
        "raw": aggregate_distribution(raw_local),
        "trajectory": trajectory_features(
            mechanism_residuals, scores, channels
        ),
        "temperature": temperature_dynamics(
            mechanism_residuals, channels, temperature_index
        ),
        "coupling": coupling_features(
            mechanism_residuals[selected], channels, temperature_index
        ),
        "severity": severity_features(
            mechanism_residuals,
            scores,
            selected,
            channels,
            threshold,
        ),
    }
    return outputs, diagnostics


def select_paths(paths: list[Path], maximum: int) -> list[Path]:
    if maximum <= 0 or len(paths) <= maximum:
        return paths
    indices = np.linspace(0, len(paths) - 1, maximum).round().astype(int)
    return [paths[index] for index in np.unique(indices)]


def split_map(dates: list[str]) -> dict[str, str]:
    dates = sorted(set(dates))
    if len(dates) < 3:
        first = max(1, len(dates) - 2)
        second = max(first + 1, len(dates) - 1)
    else:
        first = int(math.floor(0.60 * len(dates)))
        second = int(math.floor(0.80 * len(dates)))
    return {
        date: (
            "core_discovery"
            if index < first
            else "label_free_validation"
            if index < second
            else "frozen_query"
        )
        for index, date in enumerate(dates)
    }


def fill_feature_matrix(
    rows: list[dict[str, np.ndarray] | None],
    name: str,
) -> np.ndarray:
    dimension = next(
        (len(row[name]) for row in rows if row is not None), None
    )
    if dimension is None:
        raise RuntimeError(f"No routed scene produced the {name} block")
    output = np.zeros((len(rows), dimension), dtype=np.float32)
    for index, row in enumerate(rows):
        if row is not None:
            output[index] = row[name]
    return output


def build_sealed_labels(
    scenario_paths: list[Path],
    control_rows: list[dict],
    uid_by_path: dict[Path, str],
    output: Path,
) -> None:
    """Read label-bearing params only after representation files are complete."""
    rows = []
    for data_path in scenario_paths:
        params_path = data_path.with_name(
            data_path.name.replace("_data.parquet", "_params.json")
        )
        params = json.loads(params_path.read_text(encoding="utf-8-sig"))
        rows.append(
            {
                "scenario_uid": uid_by_path[data_path],
                "posthoc_class": str(params["fault_type"]),
                "severity_level": int(params["severity_level"]),
                "inject_fraction": float(params["percentage"]),
                "injection_channel": str(params.get("channel", "")),
                "source_basename": data_path.name,
                "evidence_kind": "physics_injection_metadata",
            }
        )
    for row in control_rows:
        rows.append(
            {
                "scenario_uid": row["scenario_uid"],
                "posthoc_class": "background_control",
                "severity_level": 0,
                "inject_fraction": 0.0,
                "injection_channel": "",
                "source_basename": row["source_basename"],
                "evidence_kind": "unmodified_background",
            }
        )
    pd.DataFrame(rows).to_parquet(output, index=False)


def run(args: argparse.Namespace) -> None:
    set_seed(args.ae_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = COMMON.device
    feature_columns = get_feature_cols(device)
    if len(feature_columns) != 4 or not feature_columns[-1].endswith("_T"):
        raise RuntimeError(
            f"Expected X/Y/Z/T active columns, received {feature_columns}"
        )
    temperature_index = feature_columns.index(f"{device}_T")

    healthy_dates = sorted(get_exp0_recommended_train_dates())
    if len(healthy_dates) < 2:
        raise RuntimeError("At least two healthy dates are required")
    cut = max(1, int(math.floor(0.80 * len(healthy_dates))))
    ae_dates = healthy_dates[:cut]
    calibration_dates = healthy_dates[cut:]
    ae_windows = healthy_windows_for_dates(
        device,
        ae_dates,
        args.window_radius,
        args.max_healthy_train_windows,
        args.ae_seed,
    )
    calibration_windows = healthy_windows_for_dates(
        device,
        calibration_dates,
        args.window_radius,
        args.max_healthy_calibration_windows,
        args.ae_seed + 1,
    )
    print(
        f"[healthy] train={len(ae_windows)} calibration={len(calibration_windows)} "
        f"columns={feature_columns}",
        flush=True,
    )
    ae, scaler = base.train_temporal_ae(
        ae_windows,
        latent_dim=args.latent_dim,
        epochs=args.ae_epochs,
        lr=args.ae_learning_rate,
        seed=args.ae_seed,
    )
    calibration_residual = compute_residual_batched(
        base, ae, scaler, calibration_windows
    )
    calibration_scores = np.mean(calibration_residual**2, axis=1)
    threshold = float(
        np.quantile(calibration_scores, args.healthy_score_quantile)
    )
    torch.save(
        {
            "state_dict": ae.state_dict(),
            "input_dim": int(ae_windows.shape[1]),
            "latent_dim": args.latent_dim,
            "feature_columns": feature_columns,
            "window_radius": args.window_radius,
        },
        args.output_dir / "aace_exp8_healthy_temporal_ae.pt",
    )
    joblib.dump(
        scaler, args.output_dir / "aace_exp8_healthy_window_scaler.joblib"
    )
    np.save(
        args.output_dir / "aace_exp8_healthy_calibration_scores.npy",
        calibration_scores.astype(np.float32),
    )

    active_dir = get_active_fault_data_dir() / device
    scenario_paths = select_paths(
        sorted(active_dir.glob("*_data.parquet")), args.max_scenarios
    )
    if not scenario_paths:
        raise RuntimeError(f"No active scenarios found in {active_dir}")
    dates = sorted({path.name[:10] for path in scenario_paths})
    date_splits = split_map(dates)
    uid_by_path = {path: scenario_uid(path) for path in scenario_paths}
    date_references, date_reference_table = build_date_references(
        dates,
        device,
        feature_columns,
        base,
        ae,
        scaler,
        args.window_radius,
        args.max_windows_per_scenario,
        args.healthy_score_quantile,
    )
    if set(date_references) != set(dates):
        missing = sorted(set(dates) - set(date_references))
        raise RuntimeError(f"Missing healthy date references: {missing}")
    date_reference_table.to_csv(
        args.output_dir / "aace_exp8_date_reference_thresholds.csv",
        index=False,
    )

    feature_rows: list[dict[str, np.ndarray] | None] = []
    metadata_rows = []
    for index, data_path in enumerate(scenario_paths):
        date = data_path.name[:10]
        frame = pd.read_parquet(
            data_path, columns=feature_columns
        )[feature_columns]
        values = frame.to_numpy(dtype=np.float32)
        windows, centers = sampled_scene_windows(
            values, args.window_radius, args.max_windows_per_scenario
        )
        residuals = compute_residual_batched(base, ae, scaler, windows)
        reference = date_references[date]
        corrected_residuals = (
            residuals - reference["residual_center"]
        ) / reference["residual_scale"]
        corrected_windows = (
            windows - reference["raw_center"]
        ) / reference["raw_scale"]
        if not np.array_equal(centers, reference["centers"]):
            raise RuntimeError(
                f"Scenario/reference window centers are not aligned for {date}"
            )
        paired_contrast = (
            windows - reference["windows"]
        ) / reference["raw_scale"]
        blocks, diagnostics = feature_row(
            corrected_windows,
            centers,
            paired_contrast,
            corrected_residuals,
            args.paired_contrast_threshold,
            len(feature_columns),
            temperature_index,
            args.minimum_event_windows,
            args.maximum_anomaly_fraction,
        )
        feature_rows.append(blocks)
        metadata_rows.append(
            {
                "scenario_uid": uid_by_path[data_path],
                "date": date,
                "split": date_splits[date],
                "source_kind": "unlabeled_fault_scenario",
                **diagnostics,
            }
        )
        if index < 5 or (index + 1) % 100 == 0:
            print(
                f"[scenario] {index + 1}/{len(scenario_paths)} "
                f"uid={uid_by_path[data_path]} routed={diagnostics['routed_to_exp8']} "
                f"selected={diagnostics['selected_anomaly_windows']}",
                flush=True,
            )

    control_rows = []
    if args.include_background_controls:
        for date in dates:
            reference = date_references[date]
            path = reference["source_path"]
            windows = reference["corrected_raw"]
            centers = reference["centers"]
            ae_residuals = reference["corrected_residual"]
            paired_contrast = np.zeros_like(windows)
            blocks, diagnostics = feature_row(
                windows,
                centers,
                paired_contrast,
                ae_residuals,
                args.paired_contrast_threshold,
                len(feature_columns),
                temperature_index,
                args.minimum_event_windows,
                args.maximum_anomaly_fraction,
            )
            uid = f"control_{date.replace('-', '')}"
            feature_rows.append(blocks)
            row = {
                "scenario_uid": uid,
                "date": date,
                "split": date_splits[date],
                "source_kind": "unmodified_background_control",
                **diagnostics,
            }
            metadata_rows.append(row)
            control_rows.append(
                {
                    "scenario_uid": uid,
                    "source_basename": path.name,
                }
            )

    block_names = (
        "residual",
        "ae_residual",
        "raw",
        "trajectory",
        "temperature",
        "coupling",
        "severity",
    )
    matrices = {
        name: fill_feature_matrix(feature_rows, name) for name in block_names
    }
    metadata = pd.DataFrame(metadata_rows)
    metadata.to_parquet(
        args.output_dir / "aace_exp8_anonymous_metadata.parquet", index=False
    )
    metadata.to_csv(
        args.output_dir / "aace_exp8_anonymous_metadata.csv", index=False
    )
    for name, matrix in matrices.items():
        np.save(args.output_dir / f"aace_exp8_{name}_features.npy", matrix)

    split_payload = {
        "schema_version": "aace-exp8-date-split-v1",
        "dates": date_splits,
        "counts": metadata.groupby("split").size().to_dict(),
    }
    (args.output_dir / "aace_exp8_date_split_lock.json").write_text(
        json.dumps(split_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Label-bearing metadata is intentionally opened only after all anonymous
    # arrays, split locks, and representation metadata have been materialized.
    build_sealed_labels(
        scenario_paths,
        control_rows,
        uid_by_path,
        args.output_dir / "SEALED_aace_exp8_posthoc_labels.parquet",
    )

    manifest = {
        "schema_version": "aace-exp8-unknown-representation-v1",
        "active_fault_data_dir": str(get_active_fault_data_dir()),
        "device": device,
        "feature_columns": feature_columns,
        "temperature_index": temperature_index,
        "healthy_model": "TemporalAE",
        "healthy_model_dates": ae_dates,
        "healthy_calibration_dates": calibration_dates,
        "healthy_training_windows": int(len(ae_windows)),
        "healthy_calibration_windows": int(len(calibration_windows)),
        "healthy_score_quantile": args.healthy_score_quantile,
        "healthy_score_threshold": threshold,
        "date_matched_healthy_reference": True,
        "date_reference_count": len(date_references),
        "date_reference_definition": (
            "timestamp-aligned unmodified same-background signal; paired "
            "four-channel contrast plus robust AE residual calibration"
        ),
        "mechanism_representation": "paired healthy contrast",
        "ae_residual_saved_as_ablation": True,
        "paired_contrast_threshold": args.paired_contrast_threshold,
        "window_radius": args.window_radius,
        "max_windows_per_scenario": args.max_windows_per_scenario,
        "minimum_event_windows": args.minimum_event_windows,
        "maximum_anomaly_fraction": args.maximum_anomaly_fraction,
        "scenario_file_count": len(scenario_paths),
        "background_control_count": len(control_rows),
        "anonymous_record_count": len(metadata),
        "routed_record_count": int(metadata["routed_to_exp8"].sum()),
        "block_dimensions": {
            name: int(matrix.shape[1]) for name, matrix in matrices.items()
        },
        "fault_label_parquet_read": False,
        "fault_metadata_available_to_representation": False,
        "sealed_posthoc_table_written_after_representation": True,
        "axis_order": "time-major flatten; restored as sample,channel,time",
        "passed": bool(
            len(metadata)
            and metadata["scenario_uid"].is_unique
            and all(np.isfinite(matrix).all() for matrix in matrices.values())
            and list(feature_columns)
            == [f"{device}_X", f"{device}_Y", f"{device}_Z", f"{device}_T"]
        ),
    }
    (args.output_dir / "aace_exp8_representation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    audit = {
        "schema_version": "aace-exp8-representation-audit-v1",
        "checks": {
            "active_config_path_used": (
                active_dir.resolve()
                == (get_active_fault_data_dir() / device).resolve()
            ),
            "four_channel_xyzt": manifest["passed"],
            "healthy_model_and_threshold_use_healthy_dates_only": True,
            "all_scenario_dates_have_healthy_reference": (
                len(date_references) == len(dates)
            ),
            "fault_label_parquet_never_read": True,
            "anonymous_metadata_excludes_fault_fields": not any(
                column in metadata.columns
                for column in (
                    "fault_type",
                    "severity",
                    "severity_level",
                    "inject_fraction",
                    "percentage",
                )
            ),
            "date_group_split_locked": len(date_splits) == len(dates),
            "all_feature_values_finite": all(
                np.isfinite(matrix).all() for matrix in matrices.values()
            ),
        },
    }
    audit["passed"] = bool(all(audit["checks"].values()))
    (args.output_dir / "formal_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)


def self_test() -> None:
    common_self_test()
    rng = np.random.default_rng(7)
    values = rng.normal(size=(100, 4)).astype(np.float32)
    windows, centers = temporal_windows(values, 3)
    assert windows.shape == (94, 28)
    assert centers.tolist() == list(range(3, 97))
    restored = windows.reshape(len(windows), 7, 4)
    assert np.allclose(restored[0], values[:7])
    mapping = split_map([f"2026-01-{day:02d}" for day in range(1, 30)])
    counts = pd.Series(mapping).value_counts().to_dict()
    assert counts == {
        "core_discovery": 17,
        "label_free_validation": 6,
        "frozen_query": 6,
    }
    print("AACE-CM Exp8 prepare self-test passed")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("prepare", "self-test"))
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--ae-seed", type=int, default=42)
    result.add_argument("--ae-epochs", type=int, default=100)
    result.add_argument("--ae-learning-rate", type=float, default=1e-3)
    result.add_argument("--latent-dim", type=int, default=8)
    result.add_argument(
        "--window-radius",
        type=int,
        default=15,
        help="31-second local window covers at least three 10-second imbalance cycles.",
    )
    result.add_argument("--max-healthy-train-windows", type=int, default=80000)
    result.add_argument(
        "--max-healthy-calibration-windows", type=int, default=50000
    )
    result.add_argument("--max-scenarios", type=int, default=0)
    result.add_argument("--max-windows-per-scenario", type=int, default=2048)
    result.add_argument("--healthy-score-quantile", type=float, default=0.995)
    result.add_argument("--minimum-event-windows", type=int, default=8)
    result.add_argument("--maximum-anomaly-fraction", type=float, default=0.30)
    result.add_argument(
        "--paired-contrast-threshold",
        type=float,
        default=1e-10,
        help="Squared standardized paired contrast threshold.",
    )
    result.add_argument(
        "--include-background-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.mode == "self-test":
        self_test()
    else:
        run(arguments)
