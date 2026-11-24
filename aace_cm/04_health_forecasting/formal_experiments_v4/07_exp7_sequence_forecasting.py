"""
AACE-CM-Exp7 v3: cross-date degradation-state trajectory forecasting.

Task
----
Given a past health/degradation-state trajectory window extracted by the
fixed Exp3/Exp4 VAE+DBSCAN dual-space model, predict the future trajectory.

Design
------
The train/test split is grouped by real healthy-background date. Each date
contains the same fault_type x pct x severity combinations, so the main test is
"same fault condition, different real background dates".
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import math
import pickle
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from config import COMMON, PATHS, get_active_fault_data_dir, get_feature_cols
import shared_ae_cache as sac


def _load_legacy_exp7():
    path = THIS_DIR / "07_exp7_dshi_forecast_ratio.py"
    spec = importlib.util.spec_from_file_location("exp7_legacy_ratio", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load legacy Exp7 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LEGACY = _load_legacy_exp7()


METHOD_ORDER = [
    "Persistence",
    "Linear trend",
    "MLP",
    "LSTM",
    "GRU",
    "TCN",
    "Transformer Encoder",
    "xLSTM-Transformer",
    "xLSTM-Tuned",
]

METHOD_COLORS = {
    "Persistence": "#8C8C8C",
    "Linear trend": "#54A24B",
    "MLP": "#B279A2",
    "LSTM": "#7F3C8D",
    "GRU": "#11A579",
    "TCN": "#F2B701",
    "Transformer Encoder": "#6B6ECF",
    "xLSTM-Transformer": "#17BECF",
    "xLSTM-Tuned": "#0097A7",
}

METHOD_LABELS = {
    "Persistence": "Persistence",
    "Linear trend": "Linear trend",
    "MLP": "MLP",
    "LSTM": "LSTM",
    "GRU": "GRU",
    "TCN": "TCN",
    "Transformer Encoder": "Transformer",
    "xLSTM-Transformer": "xLSTM-Transformer",
    "xLSTM-Tuned": "GLM-xLSTM-T",
}

# Paper-approved qualitative cases. Keep these fixed across re-renders so that
# figure styling can change without silently changing the scientific examples.
def _paper_case_specs():
    """Load optional plot-case selectors without shipping production identifiers."""
    manifest = os.environ.get("AACE_FORECAST_PLOT_CASES")
    if not manifest:
        return []
    with Path(manifest).open(encoding="utf-8-sig") as handle:
        return [(str(item["scenario"]), int(item["start"])) for item in json.load(handle)]

DIAG_COLUMNS = [
    "Fold",
    "Lookback",
    "Horizon",
    "Method",
    "best_epoch",
    "best_val_loss",
    "epochs_ran",
    "train_size",
    "val_size",
    "x_mean",
    "x_std",
    "y_mean",
    "y_std",
    "aux_dim",
    "val_mae_raw",
    "val_rmse_raw",
    "lr",
    "weight_decay",
    "dropout",
    "warmup_epochs",
]


@dataclass(frozen=True)
class ScenarioInfo:
    name: str
    date: str
    fault_type: str
    pct: float
    severity: int
    data_path: Path
    label_path: Path


@dataclass
class WindowData:
    hist: np.ndarray
    target: np.ndarray
    features: np.ndarray
    scenario: np.ndarray
    date: np.ndarray
    fault_type: np.ndarray
    pct: np.ndarray
    severity: np.ndarray
    start: np.ndarray


def read_fault_bounds(scenarios: list[ScenarioInfo]) -> dict[str, tuple[int, int, int]]:
    """Return sequence length, first fault index, and one-past-last fault index."""
    bounds: dict[str, tuple[int, int, int]] = {}
    for sc in scenarios:
        labels = pd.read_parquet(sc.label_path)
        if "fault_label" not in labels.columns:
            bounds[sc.name] = (len(labels), -1, -1)
            continue
        y = labels["fault_label"].to_numpy().astype(bool)
        idx = np.flatnonzero(y)
        if len(idx) == 0:
            bounds[sc.name] = (len(y), -1, -1)
        else:
            bounds[sc.name] = (len(y), int(idx[0]), int(idx[-1]) + 1)
    return bounds


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def list_fault_scenarios(device: str, max_dates: int | None = None) -> list[ScenarioInfo]:
    device_dir = get_active_fault_data_dir() / device
    if not device_dir.exists():
        raise RuntimeError(f"Fault scenario directory not found: {device_dir}")

    rows: list[ScenarioInfo] = []
    for pf in sorted(device_dir.glob("*_params.json")):
        base = pf.name.replace("_params.json", "")
        m = re.match(r"^(\d{4}-\d{2}-\d{2})_(.+?)_(pct\d+)_(sev\d+)$", base)
        if not m:
            continue
        date, fault_type, pct_str, sev_str = m.groups()
        data_path = device_dir / f"{base}_data.parquet"
        label_path = device_dir / f"{base}_labels.parquet"
        if not data_path.exists() or not label_path.exists():
            continue
        rows.append(
            ScenarioInfo(
                name=base,
                date=date,
                fault_type=fault_type,
                pct=int(pct_str.replace("pct", "")) / 100.0,
                severity=int(sev_str.replace("sev", "")),
                data_path=data_path,
                label_path=label_path,
            )
        )

    if max_dates is not None:
        dates = sorted({r.date for r in rows})[:max_dates]
        rows = [r for r in rows if r.date in set(dates)]
    return rows


def verify_complete_date_grid(scenarios: list[ScenarioInfo]) -> dict[str, object]:
    dates = sorted({s.date for s in scenarios})
    fault_types = sorted({s.fault_type for s in scenarios})
    pcts = sorted({s.pct for s in scenarios})
    severities = sorted({s.severity for s in scenarios})
    expected = {(f, p, sev) for f in fault_types for p in pcts for sev in severities}
    missing: list[tuple[str, str, float, int]] = []
    by_date = {}
    for d in dates:
        combos = {(s.fault_type, s.pct, s.severity) for s in scenarios if s.date == d}
        by_date[d] = len(combos)
        for combo in expected - combos:
            missing.append((d, combo[0], combo[1], combo[2]))
    return {
        "num_scenarios": len(scenarios),
        "num_dates": len(dates),
        "dates": dates,
        "fault_types": fault_types,
        "pcts": pcts,
        "severities": severities,
        "expected_combinations_per_date": len(expected),
        "combinations_per_date": by_date,
        "missing_combinations": missing,
    }


def make_date_folds(dates: list[str], n_folds: int) -> list[set[str]]:
    n = max(2, min(int(n_folds), len(dates)))
    return [{d for i, d in enumerate(dates) if i % n == fold} for fold in range(n)]


def read_fault_signal(sc: ScenarioInfo, device: str) -> np.ndarray:
    df, _ = LEGACY._read_fault_data(sc.data_path, sc.label_path)
    cols = get_feature_cols(device)
    assert all(c in df.columns for c in cols), (
        f"Missing fault feature columns in {sc.name}: "
        f"{[c for c in cols if c not in df.columns]}"
    )
    return df[cols].values.astype(np.float32)


def compute_or_load_dshi(
    scenarios: list[ScenarioInfo],
    device: str,
    tag: str,
    refresh: bool = False,
) -> dict[str, np.ndarray]:
    cache_npz = PATHS.exp7_figures_dir / f"{tag}_dshi_cache_{device}.npz"
    cache_meta = PATHS.exp7_figures_dir / f"{tag}_dshi_cache_{device}_meta.csv"
    part_dir = PATHS.exp7_figures_dir / f"{tag}_dshi_cache_{device}_parts"
    final_dshi = sac.get_final_dshi_config()
    if cache_npz.exists() and cache_meta.exists() and not refresh:
        meta = pd.read_csv(cache_meta)
        required_cols = {
            "effective_w_latent",
            "effective_w_reconstruction",
            "effective_w_residual",
            "score_family",
        }
        if required_cols.issubset(meta.columns):
            matches = (
                np.allclose(meta["effective_w_latent"].astype(float), final_dshi["effective_w_latent"])
                and np.allclose(meta["effective_w_reconstruction"].astype(float), final_dshi["effective_w_reconstruction"])
                and np.allclose(meta["effective_w_residual"].astype(float), final_dshi["effective_w_residual"])
                and set(meta["score_family"].astype(str)) == {str(final_dshi["score_family"])}
            )
            if matches:
                loaded = np.load(cache_npz)
                return {row["scenario"]: loaded[row["array_key"]].astype(np.float32) for _, row in meta.iterrows()}
        print("  Existing DSHI cache uses old/missing weights; rebuilding with final Exp6 weights.", flush=True)
        if part_dir.exists():
            shutil.rmtree(part_dir)

    ae, scaler, pool_latent, pool_residual, info = sac.load_best_vae()
    ref_stats = sac.get_health_reference_stats(
        ae,
        scaler,
        pool_latent,
        pool_residual,
        device=device,
        window_w=int(info.get("w", 0)),
    )
    part_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    meta_rows = []
    for i, sc in enumerate(scenarios):
        key = f"arr_{i:05d}"
        part_path = part_dir / f"{key}.npy"
        if part_path.exists():
            arr = np.load(part_path).astype(np.float32)
        else:
            x = read_fault_signal(sc, device)
            dshi = sac.compute_dual_space_health_index(
                x,
                ae,
                scaler,
                pool_latent,
                pool_residual,
                x_is_normalized=False,
                ref_stats=ref_stats,
                device=device,
                window_w=int(info.get("w", 0)),
            )
            arr = np.nan_to_num(dshi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            tmp_path = part_path.with_suffix(".tmp.npy")
            np.save(tmp_path, arr)
            tmp_path.replace(part_path)
        arrays[key] = arr
        meta_rows.append(
            {
                "scenario": sc.name,
                "array_key": key,
                "date": sc.date,
                "fault_type": sc.fault_type,
                "pct": sc.pct,
                "severity": sc.severity,
                "length": len(arrays[key]),
                "score_family": final_dshi["score_family"],
                "effective_w_latent": final_dshi["effective_w_latent"],
                "effective_w_reconstruction": final_dshi["effective_w_reconstruction"],
                "effective_w_residual": final_dshi["effective_w_residual"],
                "weight_source": final_dshi["weight_source"],
            }
        )
        if (i + 1) % 100 == 0:
            print(f"  DSHI computed: {i + 1}/{len(scenarios)}", flush=True)

    np.savez_compressed(cache_npz, **arrays)
    pd.DataFrame(meta_rows).to_csv(cache_meta, index=False, encoding="utf-8-sig")
    return {r["scenario"]: arrays[r["array_key"]] for r in meta_rows}


def causal_smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(x, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    y = np.empty_like(x)
    c = np.cumsum(np.insert(x, 0, 0.0))
    for i in range(len(x)):
        start = max(0, i - window + 1)
        y[i] = (c[i + 1] - c[start]) / (i - start + 1)
    return y


def select_starts(
    n: int,
    lookback: int,
    horizon: int,
    slide: int,
    max_windows: int,
    rng: np.random.Generator,
    *,
    sample_mode: str = "uniform",
    fault_start: int | None = None,
    fault_end: int | None = None,
) -> np.ndarray:
    total = n - lookback - horizon + 1
    if total <= 0:
        return np.empty(0, dtype=int)
    starts = np.arange(0, total, max(1, slide), dtype=int)

    if sample_mode == "event" and fault_start is not None and fault_end is not None and fault_start >= 0:
        forecast_start = starts + lookback
        forecast_end = forecast_start + horizon
        target_overlaps_fault = (forecast_end > fault_start) & (forecast_start < fault_end)
        crosses_onset = (starts <= fault_start) & (forecast_end >= fault_start)
        near_fault_end = (starts <= fault_end) & (forecast_start >= max(fault_start, fault_end - horizon))
        candidates = starts[target_overlaps_fault | crosses_onset | near_fault_end]
        if len(candidates) == 0:
            center = int(np.clip(fault_start - lookback // 2, 0, total - 1))
            radius = max(lookback + horizon, 200)
            lo = max(0, center - radius)
            hi = min(total, center + radius + 1)
            candidates = np.arange(lo, hi, max(1, slide), dtype=int)
        starts = candidates

    if max_windows > 0 and len(starts) > max_windows:
        # Uniform coverage over the whole trajectory is more stable than purely random windows.
        pos = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = starts[np.unique(pos)]
        if len(starts) < max_windows:
            pool = np.setdiff1d(np.arange(0, total, max(1, slide)), starts)
            if len(pool) > 0:
                extra = rng.choice(pool, size=min(max_windows - len(starts), len(pool)), replace=False)
                starts = np.sort(np.concatenate([starts, extra]))
    return starts


def build_window_data(
    scenarios: list[ScenarioInfo],
    dshi_by_scenario: dict[str, np.ndarray],
    fault_bounds: dict[str, tuple[int, int, int]],
    dates: set[str],
    lookback: int,
    horizon: int,
    slide: int,
    smooth_window: int,
    max_windows_per_scenario: int,
    sample_mode: str,
    seed: int,
) -> WindowData:
    rng = np.random.default_rng(seed)
    hist, target, features = [], [], []
    scenario, date, fault_type, pct, severity, start_rows = [], [], [], [], [], []
    for sc in scenarios:
        if sc.date not in dates:
            continue
        dshi = causal_smooth(dshi_by_scenario[sc.name], smooth_window)
        _, fault_start, fault_end = fault_bounds.get(sc.name, (len(dshi), -1, -1))
        starts = select_starts(
            len(dshi),
            lookback,
            horizon,
            slide,
            max_windows_per_scenario,
            rng,
            sample_mode=sample_mode,
            fault_start=fault_start,
            fault_end=fault_end,
        )
        for st in starts:
            h = dshi[st:st + lookback].astype(np.float32)
            y = dshi[st + lookback:st + lookback + horizon].astype(np.float32)
            hist.append(h)
            target.append(y)
            features.append(LEGACY._feature_from_history(h))
            scenario.append(sc.name)
            date.append(sc.date)
            fault_type.append(sc.fault_type)
            pct.append(sc.pct)
            severity.append(sc.severity)
            start_rows.append(int(st))
    return WindowData(
        hist=np.asarray(hist, dtype=np.float32),
        target=np.asarray(target, dtype=np.float32),
        features=np.asarray(features, dtype=np.float32),
        scenario=np.asarray(scenario, dtype=object),
        date=np.asarray(date, dtype=object),
        fault_type=np.asarray(fault_type, dtype=object),
        pct=np.asarray(pct, dtype=np.float32),
        severity=np.asarray(severity, dtype=int),
        start=np.asarray(start_rows, dtype=int),
    )


def save_model_checkpoint(
    *,
    method: str,
    checkpoint_path: Path,
    model: object | None,
    train: WindowData,
    seed: int,
    fast: bool,
    max_deep_train: int,
    diagnostics: dict[str, object],
) -> None:
    """Persist the fitted forecaster and all preprocessing needed for replay."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "format_version": 1,
        "method": method,
        "lookback": int(train.hist.shape[1]),
        "horizon": int(train.target.shape[1]),
        "feature_dim": int(train.features.shape[1]),
        "seed": int(seed),
        "fast": bool(fast),
        "max_deep_train": int(max_deep_train),
        "residual_target": True,
        "diagnostics": diagnostics,
    }
    if method in {"Persistence", "Linear trend"}:
        checkpoint_path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if method == "MLP":
        with checkpoint_path.with_suffix(".pkl").open("wb") as f:
            pickle.dump({"metadata": meta, "pipeline": model}, f, protocol=pickle.HIGHEST_PROTOCOL)
        return
    if not isinstance(model, LEGACY.TorchSequenceRegressor) or model.model is None:
        raise RuntimeError(f"Cannot checkpoint unfitted deep model: {method}")
    payload = {
        "metadata": meta,
        "model_state_dict": model.model.state_dict(),
        "normalization": {
            "x_mean": model.x_mean,
            "x_std": model.x_std,
            "aux_mean": model.aux_mean,
            "aux_std": model.aux_std,
            "y_mean": model.y_mean,
            "y_std": model.y_std,
        },
        "architecture": {
            "model_type": model.model_type,
            "lookback": model.lookback,
            "horizon": model.horizon,
            "fast": model.fast,
            "max_train": model.max_train,
        },
        "training_history": model.training_history_,
    }
    LEGACY.torch.save(payload, checkpoint_path.with_suffix(".pt"))


def fit_predict_model(
    method: str,
    train: WindowData,
    test: WindowData,
    seed: int,
    fast: bool,
    max_deep_train: int,
    checkpoint_path: Path | None = None,
):
    residual_target = True
    y_train = train.target - train.hist[:, -1:]
    if method == "Persistence":
        pred = np.repeat(test.hist[:, -1:], test.target.shape[1], axis=1)
        if checkpoint_path is not None:
            save_model_checkpoint(method=method, checkpoint_path=checkpoint_path, model=None, train=train, seed=seed, fast=fast, max_deep_train=max_deep_train, diagnostics={})
        return pred, [], {}
    if method == "Linear trend":
        pred = LEGACY.predict_baseline("Linear trend", test.hist, test.target.shape[1])
        if checkpoint_path is not None:
            save_model_checkpoint(method=method, checkpoint_path=checkpoint_path, model=None, train=train, seed=seed, fast=fast, max_deep_train=max_deep_train, diagnostics={})
        return pred, [], {}
    if method == "MLP":
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(96, 48),
                activation="relu",
                alpha=1e-4,
                max_iter=80 if fast else 160,
                early_stopping=True,
                random_state=seed,
            ),
        )
        y_fit = y_train.ravel() if y_train.shape[1] == 1 else y_train
        model.fit(train.features, y_fit)
        pred_raw = np.asarray(model.predict(test.features), dtype=np.float32)
        if pred_raw.ndim == 1:
            pred_raw = pred_raw.reshape(-1, 1)
        pred = pred_raw + test.hist[:, -1:]
        diagnostics = {"best_epoch": getattr(model[-1], "n_iter_", -1), "epochs_ran": getattr(model[-1], "n_iter_", -1)}
        if checkpoint_path is not None:
            save_model_checkpoint(method=method, checkpoint_path=checkpoint_path, model=model, train=train, seed=seed, fast=fast, max_deep_train=max_deep_train, diagnostics=diagnostics)
        return pred, [], diagnostics

    model = LEGACY.TorchSequenceRegressor(
        method,
        train.hist.shape[1],
        train.target.shape[1],
        seed,
        fast=fast,
        max_train=max_deep_train,
    )
    model.fit(train.features, y_train)
    pred = model.predict(test.features) + test.hist[:, -1:]
    if checkpoint_path is not None:
        save_model_checkpoint(method=method, checkpoint_path=checkpoint_path, model=model, train=train, seed=seed, fast=fast, max_deep_train=max_deep_train, diagnostics=model.training_diagnostics_)
    return pred, model.training_history_, model.training_diagnostics_


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    return {
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(math.sqrt(mean_squared_error(yt, yp))),
        "R2": float(r2_score(yt, yp)),
    }


def append_csv(path: Path, rows: list[dict[str, object]], columns: list[str] | None = None) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if columns is not None:
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False, encoding="utf-8-sig")


def aggregate_with_std(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for vals, sub in df.groupby(keys, sort=False):
        if not isinstance(vals, tuple):
            vals = (vals,)
        row = dict(zip(keys, vals))
        for m in ["MAE", "RMSE", "R2"]:
            row[m] = f"{sub[m].mean():.4f}±{sub[m].std(ddof=0):.4f}"
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_std"] = sub[m].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_rows(fold: int, lookback: int, horizon: int, method: str, test: WindowData, pred: np.ndarray) -> list[dict[str, object]]:
    rows = []
    groups = [
        ("fault_type", test.fault_type),
        ("pct", test.pct),
        ("severity", test.severity),
    ]
    for group_type, values in groups:
        for value in sorted(set(values.tolist()), key=lambda x: str(x)):
            mask = values == value
            if not np.any(mask):
                continue
            row = {
                "Fold": fold,
                "Lookback": lookback,
                "Horizon": horizon,
                "Method": method,
                "Group Type": group_type,
                "Group Value": f"{float(value):.2f}" if group_type == "pct" else str(value),
                "Test Windows": int(mask.sum()),
            }
            row.update(metrics(test.target[mask], pred[mask]))
            rows.append(row)
    return rows


def plot_metric_lines(summary_numeric: pd.DataFrame, out_base: Path, primary_lookback: int) -> None:
    sub = summary_numeric[summary_numeric["Lookback"] == primary_lookback].copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for method in METHOD_ORDER:
        m = sub[sub["Method"] == method].sort_values("Horizon")
        if m.empty:
            continue
        for ax, metric in zip(axes, ["MAE_mean", "RMSE_mean", "R2_mean"]):
            ax.plot(m["Horizon"], m[metric], marker="o", lw=1.7, label=method, color=METHOD_COLORS.get(method))
    for ax, ylabel in zip(axes, ["MAE", "RMSE", "R2"]):
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].set_title(f"Forecasting error (lookback={primary_lookback})")
    axes[1].set_title("Future trajectory reconstruction")
    axes[2].set_title("Explained variance")
    axes[2].legend(fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_example_case(example: dict[str, object], out_base: Path) -> None:
    hist = example["hist"]
    target = example["target"]
    predictions = example["predictions"]
    scenario = example["scenario"]
    start = example["start"]
    full_series = example.get("full_series")
    fault_start = example.get("fault_start")
    fault_end = example.get("fault_end")
    lookback = len(hist)
    horizon = len(target)
    t_hist = np.arange(start, start + lookback)
    t_future = np.arange(start + lookback, start + lookback + horizon)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 6.2),
        gridspec_kw={"height_ratios": [1.0, 1.35]},
    )

    ax = axes[0]
    if full_series is not None:
        full_series = np.asarray(full_series, dtype=np.float32)
        ax.plot(np.arange(len(full_series)), full_series, color="#4B5563", lw=0.85, alpha=0.85, label="Full DSHI trajectory")
        lo, hi = np.nanpercentile(full_series, [1.0, 99.5])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pad = 0.08 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
    if isinstance(fault_start, (int, np.integer)) and isinstance(fault_end, (int, np.integer)) and fault_start >= 0:
        ax.axvspan(int(fault_start), int(fault_end), color="#FEE2E2", alpha=0.55, label="Injected degradation period")
    ax.axvspan(t_hist[0], t_hist[-1], color="#DBEAFE", alpha=0.75, label="Input history")
    ax.axvspan(t_future[0], t_future[-1], color="#FED7AA", alpha=0.75, label="Forecast target")
    ax.set_ylabel("DSHI")
    ax.set_title("Selected cross-date degradation-state forecasting case")
    ax.grid(alpha=0.2)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper right")

    ax = axes[1]
    ax.axvspan(t_hist[0], t_hist[-1], color="#DBEAFE", alpha=0.55)
    ax.axvspan(t_future[0], t_future[-1], color="#FED7AA", alpha=0.55)
    ax.plot(t_hist, hist, color="#111827", lw=1.8, label="Observed history")
    ax.plot(t_future, target, color="#000000", lw=2.4, label="Ground truth future")
    for method, pred in predictions.items():
        width = 2.0 if method not in {"Persistence", "Linear trend"} else 1.4
        alpha = 0.95 if method not in {"Persistence", "Linear trend"} else 0.45
        ax.plot(
            t_future,
            pred,
            lw=width,
            alpha=alpha,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method),
        )
    ax.axvline(start + lookback - 0.5, color="#6B7280", lw=1.0, ls=":")
    ax.set_xlabel("Time step in injected sequence")
    ax.set_ylabel("Dual-Space Health Index")
    ax.text(
        0.01,
        0.02,
        f"{scenario} | start={start} | lookback={lookback} | horizon={horizon}",
        transform=ax.transAxes,
        fontsize=8,
        color="#4B5563",
    )
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_paper_case_pair(
    examples: list[dict[str, object]],
    out_base: Path,
) -> None:
    """Render two fixed Exp7 cases as a compact paper-ready qualitative figure."""
    if len(examples) != 2:
        raise ValueError("The paper case-pair figure requires exactly two examples.")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    fig = plt.figure(figsize=(7.2, 3.55), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, wspace=0.16)
    legend_handles = []
    legend_labels = []

    for panel_idx, example in enumerate(examples):
        sub = outer[panel_idx].subgridspec(2, 1, height_ratios=[0.23, 1.0], hspace=0.035)
        ax_context = fig.add_subplot(sub[0])
        ax_detail = fig.add_subplot(sub[1], sharex=ax_context)

        hist = np.asarray(example["hist"], dtype=float)
        target = np.asarray(example["target"], dtype=float)
        predictions = example["predictions"]
        start = int(example["start"])
        lookback = len(hist)
        horizon = len(target)
        t_hist = np.arange(start, start + lookback)
        t_future = np.arange(start + lookback, start + lookback + horizon)
        x_lo = max(0, start - 120)
        x_hi = start + lookback + horizon + 40

        full_series = np.asarray(example["full_series"], dtype=float)
        local_x = np.arange(x_lo, min(x_hi, len(full_series)))
        ax_context.plot(local_x, full_series[local_x], color="#59636E", lw=0.85, zorder=1)
        ax_context.axvspan(t_hist[0], t_hist[-1], color="#DCEAF7", alpha=0.72, zorder=0)
        ax_context.axvspan(t_future[0], t_future[-1], color="#FCE3C5", alpha=0.72, zorder=0)
        ax_context.axvline(t_hist[0], color="#6B7280", lw=0.55, ls=":", zorder=2)
        ax_context.axvline(t_future[-1], color="#6B7280", lw=0.55, ls=":", zorder=2)
        ax_context.set_xlim(x_lo, x_hi)
        ax_context.set_ylabel("DSHI" if panel_idx == 0 else "", fontsize=7.5)
        ax_context.tick_params(axis="both", labelsize=6.5, length=2.1, pad=1.3)
        ax_context.tick_params(axis="x", labelbottom=False)
        ax_context.spines[["top", "right"]].set_visible(False)
        ax_context.grid(axis="y", color="#D9D9D9", lw=0.35, alpha=0.5)

        h_hist, = ax_detail.plot(t_hist, hist, color="#111827", lw=1.45, label="Observed history", zorder=4)
        h_truth, = ax_detail.plot(t_future, target, color="#000000", lw=1.8, label="Ground truth", zorder=5)
        selected_methods = ["TCN", "xLSTM-Transformer", "xLSTM-Tuned"]
        plotted = {"Observed history": h_hist, "Ground truth": h_truth}
        for method in selected_methods:
            if method not in predictions:
                continue
            h_pred, = ax_detail.plot(
                t_future,
                np.asarray(predictions[method], dtype=float),
                color=METHOD_COLORS[method],
                lw=1.55 if method != "xLSTM-Tuned" else 2.1,
                label=METHOD_LABELS[method],
                zorder=6 if method == "xLSTM-Tuned" else 4,
            )
            plotted[METHOD_LABELS[method]] = h_pred

        ax_detail.axvspan(t_hist[0], t_hist[-1], color="#DCEAF7", alpha=0.58, zorder=0)
        ax_detail.axvspan(t_future[0], t_future[-1], color="#FCE3C5", alpha=0.58, zorder=0)
        ax_detail.axvline(start + lookback - 0.5, color="#6B7280", lw=0.8, ls=":", zorder=2)
        ax_detail.set_xlim(x_lo, x_hi)
        y_all = np.concatenate([hist, target] + [np.asarray(v, dtype=float) for v in predictions.values()])
        y_pad = max(0.04, 0.08 * float(np.ptp(y_all)))
        ax_detail.set_ylim(float(np.min(y_all) - y_pad), float(np.max(y_all) + y_pad))
        ax_detail.set_xlabel("Time step", fontsize=8, labelpad=2)
        if panel_idx == 0:
            ax_detail.set_ylabel("Dual-Space Health Index", fontsize=8)
        else:
            ax_detail.set_ylabel("")
        ax_detail.tick_params(axis="both", labelsize=6.7, length=2.1, pad=1.3)
        ax_detail.spines[["top", "right"]].set_visible(False)
        ax_detail.grid(color="#D9D9D9", lw=0.45, alpha=0.6)

        panel = chr(ord("a") + panel_idx)
        fault_label = str(example["fault_type"]).replace("_", " ").title()
        ax_context.text(
            0.0,
            1.03,
            f"({panel}) {fault_label}: pct={float(example['pct']) * 100:.0f}%, severity={int(example['severity'])}",
            transform=ax_context.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )
        if not legend_handles:
            ordered_labels = ["Observed history", "Ground truth", "TCN", "xLSTM-Transformer", "GLM-xLSTM-T"]
            legend_handles = [plotted[label] for label in ordered_labels if label in plotted]
            legend_labels = [label for label in ordered_labels if label in plotted]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=5,
        frameon=False,
        fontsize=6.8,
        handlelength=1.8,
        columnspacing=1.2,
    )
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_base.with_name(out_base.name + "_600dpi").with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.03, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def collect_case_candidates(
    test: "WindowData",
    predictions: dict[str, np.ndarray],
    dshi: dict[str, np.ndarray],
    fault_bounds: dict[str, tuple[int, int, int]],
    *,
    per_fold_limit: int,
    smooth_window: int,
) -> list[dict[str, object]]:
    """Return accurate but nontrivial, non-boundary DSHI forecasting cases."""
    ours = predictions.get("xLSTM-Tuned")
    if ours is None or len(ours) == 0:
        return []

    error = np.mean(np.abs(test.target - ours), axis=1)
    span = np.ptp(test.target, axis=1)
    net_change = np.abs(test.target[:, -1] - test.target[:, 0])
    max_jump = np.max(np.abs(np.diff(test.target, axis=1)), axis=1)
    mask = (test.pct >= 0.05) & (test.severity >= 4)
    if np.any(mask):
        span_floor = np.quantile(span[mask], 0.50)
        delta_floor = np.quantile(net_change[mask], 0.40)
        jump_ceiling = np.quantile(max_jump[mask], 0.80)
        mask &= (span >= span_floor) & (net_change >= delta_floor) & (max_jump <= jump_ceiling)
    if not np.any(mask):
        mask = np.ones(len(error), dtype=bool)

    candidate_indices = np.flatnonzero(mask)
    candidate_indices = candidate_indices[np.argsort(error[candidate_indices])]
    required_indices: list[int] = []
    for scenario_name, start_value in _paper_case_specs():
        matches = np.flatnonzero((test.scenario.astype(str) == scenario_name) & (test.start == start_value))
        if len(matches):
            required_indices.append(int(matches[0]))
    ordered_indices = list(dict.fromkeys(required_indices + candidate_indices.tolist()))
    rows: list[dict[str, object]] = []
    selected_methods = [m for m in ["TCN", "xLSTM-Transformer", "xLSTM-Tuned"] if m in predictions]
    for pos in ordered_indices[: max(per_fold_limit, len(required_indices), 1)]:
        scenario = str(test.scenario[pos])
        rows.append(
            {
                "hist": test.hist[pos].copy(),
                "target": test.target[pos].copy(),
                "predictions": {m: predictions[m][pos].copy() for m in selected_methods},
                "scenario": scenario,
                "start": int(test.start[pos]),
                "full_series": causal_smooth(dshi[scenario], smooth_window),
                "fault_start": fault_bounds[scenario][1],
                "fault_end": fault_bounds[scenario][2],
                "fault_type": str(test.fault_type[pos]),
                "severity": int(test.severity[pos]),
                "pct": float(test.pct[pos]),
                "case_mae": float(error[pos]),
            }
        )
    return rows


def save_case_bank(
    candidates: list[dict[str, object]],
    out_dir: Path,
    *,
    max_plots: int,
    paper_case_pair: bool = False,
) -> None:
    if max_plots <= 0 or not candidates:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, object]] = []
    used_fault_types: set[str] = set()
    for item in sorted(candidates, key=lambda x: float(x["case_mae"])):
        fault_type = str(item["fault_type"])
        if fault_type not in used_fault_types:
            selected.append(item)
            used_fault_types.add(fault_type)
        if len(selected) >= max_plots:
            break
    if len(selected) < max_plots:
        selected_ids = {id(item) for item in selected}
        for item in sorted(candidates, key=lambda x: float(x["case_mae"])):
            if id(item) not in selected_ids:
                selected.append(item)
                selected_ids.add(id(item))
            if len(selected) >= max_plots:
                break

    rows = []
    for rank, item in enumerate(selected, start=1):
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(item["scenario"]))
        out_base = out_dir / f"case_{rank:02d}_{safe_name}_start{item['start']}"
        plot_example_case(item, out_base)
        case_data_path = out_base.with_name(out_base.name + "_data").with_suffix(".npz")
        payload = {
            "hist": np.asarray(item["hist"], dtype=np.float32),
            "target": np.asarray(item["target"], dtype=np.float32),
            "full_series": np.asarray(item["full_series"], dtype=np.float32),
            "start": np.asarray([int(item["start"])], dtype=np.int64),
            "fault_start": np.asarray([int(item["fault_start"])], dtype=np.int64),
            "fault_end": np.asarray([int(item["fault_end"])], dtype=np.int64),
            "metadata_json": np.asarray(
                [
                    json.dumps(
                        {
                            "scenario": str(item["scenario"]),
                            "fault_type": str(item["fault_type"]),
                            "severity": int(item["severity"]),
                            "pct": float(item["pct"]),
                            "case_mae": float(item["case_mae"]),
                        },
                        ensure_ascii=False,
                    )
                ]
            ),
        }
        for method, pred in item["predictions"].items():
            payload[f"prediction__{method}"] = np.asarray(pred, dtype=np.float32)
        np.savez_compressed(case_data_path, **payload)
        rows.append(
            {
                "rank": rank,
                "scenario": item["scenario"],
                "fault_type": item["fault_type"],
                "severity": item["severity"],
                "pct": item["pct"],
                "start": item["start"],
                "glm_xlstm_t_mae": item["case_mae"],
                "figure": str(out_base.with_suffix(".png")),
                "data": str(case_data_path),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "case_selection_index.csv", index=False, encoding="utf-8-sig")
    if paper_case_pair:
        pair = []
        for scenario_name, start_value in _paper_case_specs():
            match = next(
                (
                    item
                    for item in candidates
                    if str(item["scenario"]) == scenario_name and int(item["start"]) == start_value
                ),
                None,
            )
            if match is not None:
                pair.append(match)
        if len(pair) == 2:
            plot_paper_case_pair(pair, out_dir / "exp7_fig_f21a_two_selected_cases")
        else:
            raise RuntimeError("Could not recover both fixed paper-approved cases for the paper pair.")


def load_case_data(path: Path) -> dict[str, object]:
    """Load a saved case package for re-rendering without model retraining."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"][0]))
        predictions = {
            key.removeprefix("prediction__"): data[key].copy()
            for key in data.files
            if key.startswith("prediction__")
        }
        return {
            "hist": data["hist"].copy(),
            "target": data["target"].copy(),
            "full_series": data["full_series"].copy(),
            "start": int(data["start"][0]),
            "fault_start": int(data["fault_start"][0]),
            "fault_end": int(data["fault_end"][0]),
            "predictions": predictions,
            **meta,
        }


def rerender_paper_case_pair(case_dir: Path) -> None:
    """Re-render the selected paper pair using saved case arrays only."""
    packages = [load_case_data(path) for path in sorted(case_dir.glob("case_*_data.npz"))]
    pair = []
    for scenario_name, start_value in _paper_case_specs():
        item = next((row for row in packages if str(row["scenario"]) == scenario_name and int(row["start"]) == start_value), None)
        if item is None:
            raise RuntimeError(f"Saved case package missing fixed paper case: {scenario_name} start={start_value}")
        pair.append(item)
    plot_paper_case_pair(pair, case_dir / "exp7_fig_f21a_two_selected_cases")


def predict_from_torch_checkpoint(checkpoint_path: Path, features: np.ndarray) -> np.ndarray:
    """Replay a saved deep forecaster without calling fit()."""
    payload = LEGACY.torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch = payload["architecture"]
    normalization = payload["normalization"]
    method = str(arch["model_type"])
    tuned = method == "xLSTM-Tuned"
    hidden = 96 if tuned else 64
    dropout = 0.15 if tuned else 0.1
    aux_mean = normalization.get("aux_mean")
    aux_dim = int(aux_mean.shape[1]) if aux_mean is not None else 0
    model = LEGACY.SequenceForecasterNet(
        method,
        int(arch["lookback"]),
        int(arch["horizon"]),
        hidden=hidden,
        aux_dim=aux_dim,
        dropout=dropout,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    lookback = int(arch["lookback"])
    x_all = np.asarray(features, dtype=np.float32)
    x = (x_all[:, :lookback] - float(normalization["x_mean"])) / float(normalization["x_std"])
    aux = x_all[:, lookback:]
    if aux.shape[1] and aux_mean is not None:
        aux = (aux - aux_mean) / normalization["aux_std"]
    outputs = []
    with LEGACY.torch.no_grad():
        for offset in range(0, len(x), 2048):
            bx = LEGACY.torch.from_numpy(x[offset:offset + 2048])
            bax = LEGACY.torch.from_numpy(aux[offset:offset + 2048].astype(np.float32))
            outputs.append(model(bx, bax).numpy())
    raw = np.vstack(outputs) * float(normalization["y_std"]) + float(normalization["y_mean"])
    return raw.astype(np.float32)


def replay_fixed_paper_case_pair(case_dir: Path, checkpoint_dir: Path, dshi_tag: str) -> None:
    """Recreate the two approved cases from saved weights, without retraining."""
    scenarios = list_fault_scenarios(COMMON.device)
    grid = verify_complete_date_grid(scenarios)
    folds = make_date_folds(grid["dates"], 3)
    dshi = compute_or_load_dshi(scenarios, COMMON.device, tag=dshi_tag, refresh=False)
    fault_bounds = read_fault_bounds(scenarios)
    test = build_window_data(
        scenarios,
        dshi,
        fault_bounds,
        set(folds[0]),
        lookback=200,
        horizon=150,
        slide=10,
        smooth_window=5,
        max_windows_per_scenario=12,
        sample_mode="event",
        seed=42 + 10_000 + 1 + 200 * 100 + 150,
    )
    predictions: dict[str, np.ndarray] = {}
    for method in ["TCN", "xLSTM-Transformer", "xLSTM-Tuned"]:
        checkpoint = checkpoint_dir / f"fold01_L200_H150_{method}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        predictions[method] = predict_from_torch_checkpoint(checkpoint, test.features) + test.hist[:, -1:]

    cases = []
    for scenario_name, start_value in _paper_case_specs():
        matches = np.flatnonzero((test.scenario.astype(str) == scenario_name) & (test.start == start_value))
        if not len(matches):
            raise RuntimeError(f"The fixed paper case is not in the replayed test windows: {scenario_name}, start={start_value}")
        pos = int(matches[0])
        cases.append(
            {
                "hist": test.hist[pos].copy(),
                "target": test.target[pos].copy(),
                "predictions": {method: pred[pos].copy() for method, pred in predictions.items()},
                "scenario": scenario_name,
                "start": start_value,
                "full_series": causal_smooth(dshi[scenario_name], 5),
                "fault_start": fault_bounds[scenario_name][1],
                "fault_end": fault_bounds[scenario_name][2],
                "fault_type": str(test.fault_type[pos]),
                "severity": int(test.severity[pos]),
                "pct": float(test.pct[pos]),
                "case_mae": float(np.mean(np.abs(test.target[pos] - predictions["xLSTM-Tuned"][pos]))),
            }
        )
    case_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(cases, start=1):
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(item["scenario"]))
        data_path = case_dir / f"case_paper_{index:02d}_{safe_name}_start{item['start']}_data.npz"
        payload = {
            "hist": np.asarray(item["hist"], dtype=np.float32),
            "target": np.asarray(item["target"], dtype=np.float32),
            "full_series": np.asarray(item["full_series"], dtype=np.float32),
            "start": np.asarray([int(item["start"])], dtype=np.int64),
            "fault_start": np.asarray([int(item["fault_start"])], dtype=np.int64),
            "fault_end": np.asarray([int(item["fault_end"])], dtype=np.int64),
            "metadata_json": np.asarray([json.dumps({key: item[key] for key in ["scenario", "fault_type", "severity", "pct", "case_mae"]})]),
            **{f"prediction__{method}": np.asarray(pred, dtype=np.float32) for method, pred in item["predictions"].items()},
        }
        np.savez_compressed(data_path, **payload)
    plot_paper_case_pair(cases, case_dir / "exp7_fig_f21a_two_selected_cases")


def run(args: argparse.Namespace) -> None:
    PATHS.ensure_dirs()
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.tag).strip("_")
    out_dir = PATHS.exp7_figures_dir
    prefix = f"exp7_{tag}"
    detail_path = out_dir / f"{prefix}_detail_sequence_forecasting.csv"
    subgroup_path = out_dir / f"{prefix}_subgroup_detail_sequence_forecasting.csv"
    history_path = out_dir / f"{prefix}_deep_training_history.csv"
    diag_path = out_dir / f"{prefix}_training_diagnostics.csv"
    summary_path = out_dir / f"{prefix}_table_t7_sequence_forecasting.csv"
    summary_numeric_path = out_dir / f"{prefix}_table_t7_sequence_forecasting_numeric.csv"
    subgroup_summary_path = out_dir / f"{prefix}_subgroup_summary_sequence_forecasting.csv"
    meta_path = out_dir / f"{prefix}_metadata.json"
    fig_path = out_dir / f"{prefix}_fig_f20_horizon_lines"
    example_path = out_dir / f"{prefix}_fig_f21_prediction_example"
    checkpoint_dir = PATHS.outputs_dir / "exp7_checkpoints" / tag
    checkpoint_manifest_path = checkpoint_dir / "checkpoint_manifest.csv"

    scenarios = list_fault_scenarios(COMMON.device, max_dates=args.fast_dates if args.fast else None)
    grid = verify_complete_date_grid(scenarios)
    if grid["missing_combinations"]:
        raise RuntimeError(f"Incomplete date grid; first missing rows: {grid['missing_combinations'][:5]}")
    dates = grid["dates"]
    folds = make_date_folds(dates, args.folds if not args.fast else 2)

    print("=" * 78, flush=True)
    print("AACE-CM-Exp7 v3: cross-date degradation-state trajectory forecasting", flush=True)
    print(f"device={COMMON.device} scenarios={len(scenarios)} dates={len(dates)} folds={len(folds)}", flush=True)
    valid_grid = [(lb, hz) for lb in args.lookbacks for hz in args.horizons if hz <= lb]
    skipped_grid = [(lb, hz) for lb in args.lookbacks for hz in args.horizons if hz > lb]
    print(f"lookbacks={args.lookbacks} horizons={args.horizons}", flush=True)
    print(f"valid lookback-horizon pairs={len(valid_grid)}; skipped horizon>lookback pairs={len(skipped_grid)}", flush=True)
    print("=" * 78, flush=True)
    sac.print_cache_status()

    active_methods = args.methods or METHOD_ORDER
    unknown_methods = sorted(set(active_methods) - set(METHOD_ORDER))
    if unknown_methods:
        raise ValueError(f"Unknown methods requested: {unknown_methods}")
    print(f"methods={active_methods}", flush=True)

    dshi_tag = args.dshi_cache_tag or tag
    dshi = compute_or_load_dshi(scenarios, COMMON.device, tag=dshi_tag, refresh=args.refresh_dshi_cache)
    fault_bounds = read_fault_bounds(scenarios)

    completed: set[tuple[int, int, int, str]] = set()
    if args.resume and detail_path.exists():
        old = pd.read_csv(detail_path)
        for _, row in old.iterrows():
            completed.add((int(row["Fold"]), int(row["Lookback"]), int(row["Horizon"]), str(row["Method"])))
        print(f"Resume enabled: {len(completed)} completed model runs found.", flush=True)

    example_payload = None
    case_candidates: list[dict[str, object]] = []
    for fold_idx, test_dates in enumerate(folds, start=1):
        if args.fold_indices and fold_idx not in args.fold_indices:
            continue
        train_dates = set(dates) - set(test_dates)
        print(f"Fold {fold_idx}/{len(folds)} train_dates={len(train_dates)} test_dates={sorted(test_dates)}", flush=True)
        for lookback in args.lookbacks:
            for horizon in args.horizons:
                if horizon > lookback:
                    print(f"  skip lookback={lookback} horizon={horizon} (horizon > lookback)", flush=True)
                    continue
                if args.resume and all((fold_idx, lookback, horizon, method) in completed for method in active_methods):
                    print(f"  skip lookback={lookback} horizon={horizon} (all methods completed)", flush=True)
                    continue
                print(f"  lookback={lookback} horizon={horizon}", flush=True)
                train = build_window_data(
                    scenarios,
                    dshi,
                    fault_bounds,
                    train_dates,
                    lookback,
                    horizon,
                    args.slide_step,
                    args.smooth_window,
                    args.max_windows_per_scenario,
                    args.sample_mode,
                    args.seed + fold_idx + lookback * 100 + horizon,
                )
                test = build_window_data(
                    scenarios,
                    dshi,
                    fault_bounds,
                    set(test_dates),
                    lookback,
                    horizon,
                    args.slide_step,
                    args.smooth_window,
                    args.max_test_windows_per_scenario,
                    args.sample_mode,
                    args.seed + 10_000 + fold_idx + lookback * 100 + horizon,
                )
                print(f"    train_windows={len(train.hist)} test_windows={len(test.hist)}", flush=True)
                if len(train.hist) == 0 or len(test.hist) == 0:
                    continue

                example_predictions = {}
                for method in active_methods:
                    key = (fold_idx, lookback, horizon, method)
                    if key in completed:
                        continue
                    checkpoint_stem = checkpoint_dir / f"fold{fold_idx:02d}_L{lookback}_H{horizon}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', method)}"
                    pred, history, diag = fit_predict_model(
                        method,
                        train,
                        test,
                        seed=args.seed + fold_idx + lookback + horizon,
                        fast=args.fast,
                        max_deep_train=args.max_deep_train if not args.fast else min(args.max_deep_train, 3000),
                        checkpoint_path=checkpoint_stem,
                    )
                    checkpoint_file = next((path for path in [checkpoint_stem.with_suffix(ext) for ext in [".pt", ".pkl", ".json"]] if path.exists()), None)
                    append_csv(
                        checkpoint_manifest_path,
                        [
                            {
                                "Fold": fold_idx,
                                "Lookback": lookback,
                                "Horizon": horizon,
                                "Method": method,
                                "Checkpoint": str(checkpoint_file) if checkpoint_file is not None else "",
                                "Seed": args.seed + fold_idx + lookback + horizon,
                                "Feature Dim": int(train.features.shape[1]),
                            }
                        ],
                    )
                    m = metrics(test.target, pred)
                    detail_row = {
                        "Fold": fold_idx,
                        "Train Dates": ";".join(sorted(train_dates)),
                        "Test Dates": ";".join(sorted(test_dates)),
                        "Lookback": lookback,
                        "Horizon": horizon,
                        "Method": method,
                        "Train Windows": len(train.hist),
                        "Test Windows": len(test.hist),
                    }
                    detail_row.update(m)
                    append_csv(detail_path, [detail_row])
                    append_csv(subgroup_path, subgroup_rows(fold_idx, lookback, horizon, method, test, pred))
                    if history:
                        hist_rows = [
                            {
                                "Fold": fold_idx,
                                "Lookback": lookback,
                                "Horizon": horizon,
                                "Method": method,
                                **row,
                            }
                            for row in history
                        ]
                        append_csv(history_path, hist_rows)
                    if diag:
                        append_csv(
                            diag_path,
                            [
                                {
                                    "Fold": fold_idx,
                                    "Lookback": lookback,
                                    "Horizon": horizon,
                                    "Method": method,
                                    **diag,
                                }
                            ],
                            columns=DIAG_COLUMNS,
                        )
                    if fold_idx == 1 and lookback == args.primary_lookback and horizon == args.primary_horizon:
                        example_predictions[method] = pred
                    print(f"      {method}: MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} R2={m['R2']:.4f}", flush=True)

                if example_payload is None and example_predictions:
                    # Pick a visually interpretable case: visible future dynamics,
                    # moderate prediction error, and no dominant spike in history.
                    ref_method = "xLSTM-Transformer" if "xLSTM-Transformer" in example_predictions else list(example_predictions)[-1]
                    candidate_mask = (test.pct >= 0.05) & (test.severity >= 5)
                    if not np.any(candidate_mask):
                        candidate_mask = np.ones(len(test.target), dtype=bool)
                    candidate_idx = np.flatnonzero(candidate_mask)
                    target_q90 = np.percentile(test.target[candidate_idx], 90, axis=1)
                    target_q10 = np.percentile(test.target[candidate_idx], 10, axis=1)
                    future_span = target_q90 - target_q10
                    future_delta = np.abs(test.target[candidate_idx, -1] - test.target[candidate_idx, 0])
                    hist_q99 = np.percentile(test.hist[candidate_idx], 99, axis=1)
                    hist_q01 = np.percentile(test.hist[candidate_idx], 1, axis=1)
                    hist_span = hist_q99 - hist_q01
                    case_error = np.mean(np.abs(test.target[candidate_idx] - example_predictions[ref_method][candidate_idx]), axis=1)

                    def zscore(x: np.ndarray) -> np.ndarray:
                        s = float(np.std(x))
                        if s < 1e-8:
                            return np.zeros_like(x, dtype=np.float32)
                        return (x - float(np.mean(x))) / s

                    error_center_penalty = np.abs(zscore(case_error))
                    score = 1.2 * zscore(future_span) + 0.8 * zscore(future_delta) - 0.5 * zscore(hist_span) - 0.2 * error_center_penalty
                    pos = int(candidate_idx[int(np.argmax(score))])
                    example_payload = {
                        "hist": test.hist[pos],
                        "target": test.target[pos],
                        "predictions": {
                            k: v[pos]
                            for k, v in example_predictions.items()
                            if k in {"Persistence", "Linear trend", "MLP", "TCN", ref_method}
                        },
                        "scenario": str(test.scenario[pos]),
                        "start": int(test.start[pos]),
                        "full_series": causal_smooth(dshi[str(test.scenario[pos])], args.smooth_window),
                        "fault_start": fault_bounds[str(test.scenario[pos])][1],
                        "fault_end": fault_bounds[str(test.scenario[pos])][2],
                    }
                if (
                    args.case_bank_size > 0
                    and lookback == args.primary_lookback
                    and horizon == args.primary_horizon
                    and "xLSTM-Tuned" in example_predictions
                ):
                    case_candidates.extend(
                        collect_case_candidates(
                            test,
                            example_predictions,
                            dshi,
                            fault_bounds,
                            per_fold_limit=max(args.case_bank_size * 4, 12),
                            smooth_window=args.smooth_window,
                        )
                    )

    detail = pd.read_csv(detail_path)
    summary = aggregate_with_std(detail, ["Lookback", "Horizon", "Method"])
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_numeric_path, index=False, encoding="utf-8-sig")

    if subgroup_path.exists():
        subgroup = pd.read_csv(subgroup_path)
        subgroup_summary = aggregate_with_std(subgroup, ["Lookback", "Horizon", "Method", "Group Type", "Group Value"])
        subgroup_summary.to_csv(subgroup_summary_path, index=False, encoding="utf-8-sig")

    plot_metric_lines(summary, fig_path, args.primary_lookback)
    if example_payload is not None:
        plot_example_case(example_payload, example_path)
    if args.case_bank_size > 0:
        save_case_bank(
            case_candidates,
            out_dir / f"{prefix}_prediction_case_bank",
            max_plots=args.case_bank_size,
            paper_case_pair=args.paper_case_pair,
        )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "Exp7 cross-date degradation-state trajectory forecasting",
                "tag": tag,
                "device": COMMON.device,
                "task": "past degradation-state trajectory window -> future degradation-state trajectory",
                "representation": "Dual-Space Health Index from fixed Exp3/Exp4 VAE+DBSCAN model",
                "split": "3-fold grouped by real healthy-background date",
                "date_grid": grid,
                "lookbacks": args.lookbacks,
                "horizons": args.horizons,
                "valid_lookback_horizon_pairs": valid_grid,
                "skipped_lookback_horizon_pairs": skipped_grid,
                "slide_step": args.slide_step,
                "smooth_window": args.smooth_window,
                "sample_mode": args.sample_mode,
                "max_windows_per_scenario": args.max_windows_per_scenario,
                "max_test_windows_per_scenario": args.max_test_windows_per_scenario,
                "max_deep_train": args.max_deep_train,
                "models": active_methods,
                "dshi_cache_tag": dshi_tag,
                "metrics": ["MAE", "RMSE", "R2"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    for path in [
        detail_path,
        subgroup_path,
        history_path,
        diag_path,
        summary_path,
        summary_numeric_path,
        subgroup_summary_path,
        meta_path,
        fig_path.with_suffix(".png"),
        fig_path.with_suffix(".pdf"),
        example_path.with_suffix(".png"),
        example_path.with_suffix(".pdf"),
    ]:
        if path.exists():
            shutil.copy2(path, PATHS.figures_dir / path.name)

    print(f"Saved: {summary_path}", flush=True)
    print(f"Saved: {subgroup_summary_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp7 v3 cross-date trajectory forecasting")
    parser.add_argument("--tag", default="sequence_forecasting_v3")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--fast-dates", type=int, default=6)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--lookbacks", type=parse_int_list, default="50,80,100,150,200")
    parser.add_argument("--horizons", type=parse_int_list, default="10,20,30,50,100,150")
    parser.add_argument("--methods", type=parse_str_list, default=None)
    parser.add_argument("--dshi-cache-tag", default=None)
    parser.add_argument("--primary-lookback", type=int, default=200)
    parser.add_argument("--primary-horizon", type=int, default=50)
    parser.add_argument("--slide-step", type=int, default=10)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--sample-mode", choices=["uniform", "event"], default="event")
    parser.add_argument("--max-windows-per-scenario", type=int, default=24)
    parser.add_argument("--max-test-windows-per-scenario", type=int, default=12)
    parser.add_argument("--max-deep-train", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refresh-dshi-cache", action="store_true")
    parser.add_argument("--case-bank-size", type=int, default=0)
    parser.add_argument("--paper-case-pair", action="store_true")
    parser.add_argument("--fold-indices", type=parse_int_list, default=None)
    parser.add_argument("--rerender-paper-case-pair", default=None, help="Existing case-bank directory; re-render figures without training.")
    parser.add_argument("--replay-fixed-paper-case-pair", default=None, help="Case-bank directory to rebuild from saved checkpoints without training.")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory containing saved Exp7 model checkpoints.")
    args = parser.parse_args()
    if args.rerender_paper_case_pair:
        rerender_paper_case_pair(Path(args.rerender_paper_case_pair))
        return
    if args.replay_fixed_paper_case_pair:
        if not args.checkpoint_dir or not args.dshi_cache_tag:
            parser.error("--replay-fixed-paper-case-pair requires --checkpoint-dir and --dshi-cache-tag")
        replay_fixed_paper_case_pair(
            Path(args.replay_fixed_paper_case_pair),
            Path(args.checkpoint_dir),
            str(args.dshi_cache_tag),
        )
        return
    run(args)


if __name__ == "__main__":
    main()
