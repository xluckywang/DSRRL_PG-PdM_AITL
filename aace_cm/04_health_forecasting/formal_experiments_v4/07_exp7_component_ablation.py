"""Controlled component ablations for DSHI trajectory forecasting."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ORIGINAL = HERE / "07_exp7_sequence_forecasting.py"
MODEL = HERE / "07_exp7_dshi_forecast_ratio.py"
OUTPUTS = HERE / "outputs" / "exp7_figures"
PREFIX = "exp7_sequence_forecasting_v4_tuned_full"
METADATA = OUTPUTS / f"{PREFIX}_metadata.json"
DIAGNOSTICS = OUTPUTS / f"{PREFIX}_training_diagnostics.csv"
REFERENCE = HERE / "figures" / "exp7_sequence_forecasting_v4_full_with_v3_baselines_detail_clean.csv"
CACHE_ROOT = OUTPUTS / "archive_old_dshi_cache_before_exp6_weights_20260703"
LABEL_ROOT = ROOT / "aace_cm" / "04-fault_data_LEGACY_vibration_only_20260628" / "BA1"
SUPPLEMENT_METADATA = OUTPUTS / "exp7_sequence_forecasting_v4_casebank_20260713_metadata.json"
SUPPLEMENT_LABELS = ROOT / "aace_cm" / "04-temp_fault_data" / "BA1"
LOOKBACK = 200
BASE_SEED = 42
HORIZONS = (10, 50, 150)
VARIANTS = ("full", "no_local", "no_gate", "no_stats")
METRICS = ("MAE", "RMSE", "R2")


def load_exp7():
    spec = importlib.util.spec_from_file_location("exp7_frozen_sequence", ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ZeroLocal(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden

    def forward(self, x):
        return x.new_zeros((x.shape[0], self.hidden, x.shape[-1]))


class OpenGate(nn.Module):
    def forward(self, x):
        return torch.ones_like(x)


class ZeroStats(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.width = hidden // 2

    def forward(self, x):
        return x.new_zeros((x.shape[0], self.width))


@contextmanager
def model_variant(legacy, variant: str):
    original = legacy.SequenceForecasterNet
    if variant == "full":
        yield
        return

    class AblatedForecaster(original):
        def __init__(self, *args, hidden, **kwargs):
            super().__init__(*args, hidden=hidden, **kwargs)
            # Keep initialization and the fusion head fixed; neutralize one path.
            if variant == "no_local":
                self.local_stem = ZeroLocal(hidden)
            elif variant == "no_gate":
                self.memory_gate = OpenGate()
            elif variant == "no_stats":
                if self.aux_encoder is None:
                    raise ValueError("The reference model has no statistics branch")
                self.aux_encoder = ZeroStats(hidden)
            else:
                raise ValueError(f"Unknown variant: {variant}")

    legacy.SequenceForecasterNet = AblatedForecaster
    try:
        yield
    finally:
        legacy.SequenceForecasterNet = original


def file_hash(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def smooth_locked_trace(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    # Preserve the original float32 cumulative sum and scalar division dtype.
    sums = np.cumsum(np.insert(values, 0, 0.0))
    end = np.arange(1, len(sums))
    start = np.maximum(0, end - window)
    dtype = np.asarray(np.float32(1) / 3).dtype
    return np.divide(sums[1:] - sums[start], end - start, dtype=dtype).astype(np.float32)


def load_locked_inputs(exp7, config, folds, selected_folds, smoke, supplement):
    stem = f"{config['dshi_cache_tag']}_dshi_cache_{config['device']}"
    cache_root = OUTPUTS if supplement else CACHE_ROOT
    label_root = SUPPLEMENT_LABELS if supplement else LABEL_ROOT
    cache, index = cache_root / f"{stem}.npz", cache_root / f"{stem}_meta.csv"
    # The original loader sorts parameter filenames, not stripped scenario names.
    rows = pd.read_csv(index).sort_values("scenario", key=lambda names: names + "_params.json")
    grid = config["date_grid"]
    if (
        rows.scenario.duplicated().any()
        or rows.array_key.duplicated().any()
        or len(rows) != grid["num_scenarios"]
        or sorted(rows.date.unique()) != grid["dates"]
    ):
        raise ValueError("DSHI index does not match the recorded date grid")
    if supplement:
        weights = rows[["effective_w_latent", "effective_w_reconstruction", "effective_w_residual"]]
        if set(rows.score_family) != {"exp6_latent_residual_dshi"} or not np.allclose(
            weights.to_numpy(), [0.94, 0, 0.06]
        ):
            raise ValueError("Supplemental DSHI index has unexpected Exp6 scoring weights")

    if smoke:
        test_dates = folds[selected_folds[0] - 1]
        dates = {min(test_dates), min(set(grid["dates"]) - test_dates)}
        rows = rows[rows.date.isin(dates) & (rows.pct == 0.1) & (rows.severity == 5)]
        if len(rows) != len(dates) * len(grid["fault_types"]):
            raise ValueError("Smoke grid does not cover each fault family on both dates")

    scenarios, arrays = [], {}
    with np.load(cache, allow_pickle=False) as archive:
        if len(archive.files) != grid["num_scenarios"]:
            raise ValueError("Archived DSHI array count does not match the frozen grid")
        for row in rows.itertuples(index=False):
            labels = label_root / f"{row.scenario}_labels.parquet"
            if not labels.is_file():
                raise FileNotFoundError(labels)
            arr = archive[row.array_key].astype(np.float32, copy=False)
            if arr.shape != (row.length,) or not np.isfinite(arr).all():
                raise ValueError(f"Invalid DSHI array for {row.scenario}")
            smoothed = smooth_locked_trace(arr, config["smooth_window"])
            if smoke and not np.array_equal(smoothed, exp7.causal_smooth(arr, config["smooth_window"])):
                raise ValueError(f"Vectorized smoothing changed archived DSHI: {row.scenario}")
            arrays[row.scenario] = smoothed
            scenarios.append(exp7.ScenarioInfo(
                row.scenario, row.date, row.fault_type, row.pct, row.severity,
                label_root / f"{row.scenario}_data.parquet", labels,
            ))

    if not smoke and exp7.verify_complete_date_grid(scenarios) != grid:
        raise ValueError("DSHI scenario grid differs from the recorded configuration")
    bounds = exp7.read_fault_bounds(scenarios)
    for sc in scenarios:
        length, first, last = bounds[sc.name]
        if length != len(arrays[sc.name]) or not (0 <= first < last <= length):
            raise ValueError(f"Fault labels do not match archived DSHI: {sc.name}")
    return cache, index, scenarios, arrays, bounds


def input_statistics(train, seed, budget):
    x = train.features
    y = train.target - train.hist[:, -1:]
    if len(x) > budget:
        chosen = np.random.default_rng(seed).choice(len(x), size=budget, replace=False)
        x, y = x[chosen], y[chosen]
    return {
        "x_mean": float(x[:, :LOOKBACK].mean()),
        "x_std": float(x[:, :LOOKBACK].std() + 1e-6),
        "y_mean": float(y.mean()),
        "y_std": float(y.std() + 1e-6),
        "aux_dim": x.shape[1] - LOOKBACK,
    }


def audit_historical(fold, horizon, train, test, stats, reference, diagnostics):
    mask = (reference.Fold == fold) & (reference.Lookback == LOOKBACK) & (reference.Horizon == horizon)
    matches = reference.loc[mask]
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen metric row for fold={fold}, H={horizon}")
    row = matches.iloc[0]
    for side, windows in (("Train", train), ("Test", test)):
        if (
            set(windows.date) != set(row[f"{side} Dates"].split(";"))
            or len(windows.hist) != row[f"{side} Windows"]
        ):
            raise ValueError(f"{side} dates or windows differ from V2.0: fold={fold}, H={horizon}")
    historical = diagnostics.loc[
        (diagnostics.Fold == fold)
        & (diagnostics.Lookback == LOOKBACK)
        & (diagnostics.Horizon == horizon), list(stats)
    ]
    if historical.empty or not np.allclose(
        historical.to_numpy(), list(stats.values()), atol=2e-6, rtol=2e-6
    ):
        raise ValueError(
            f"Training input statistics differ from frozen diagnostics: fold={fold}, "
            f"H={horizon}; reconstructed={stats}; archived={historical.to_dict('records')}"
        )
    return {f"historical_{k}": float(row[k]) for k in METRICS}


def save_results(out, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "per_fold.csv", index=False)
    counts = frame.groupby(["Variant", "Horizon"]).size().rename("folds")
    means = frame.groupby(["Variant", "Horizon"])[list(METRICS)].mean()
    counts.to_frame().join(means).to_csv(out / "summary.csv")
    base = frame.loc[frame.Variant.eq("full"), ["Fold", "Horizon", *METRICS]]
    paired = frame.merge(base, on=["Fold", "Horizon"], suffixes=("", "_full"))
    paired = paired.loc[paired.Variant.ne("full")].copy()
    for metric in METRICS:
        paired[f"{metric}_delta"] = paired[metric] - paired[f"{metric}_full"]
    paired.to_csv(out / "paired_vs_full.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "audit", "formal"), required=True)
    parser.add_argument("--data-protocol", choices=("historical", "supplement"), default="historical")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--fold-indices", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    exp7 = load_exp7()
    supplement = args.data_protocol == "supplement"
    metadata_path = SUPPLEMENT_METADATA if supplement else METADATA
    cfg = json.loads(metadata_path.read_text(encoding="utf-8"))
    dates = cfg["date_grid"]["dates"]
    folds = exp7.make_date_folds(dates, 3)
    smoke = args.mode == "smoke"
    if len(set(args.horizons)) != len(args.horizons) or not set(args.horizons) <= set(HORIZONS):
        parser.error("Use unique horizons from the planned ablation grid (10, 50, 150)")
    if len(set(args.fold_indices)) != len(args.fold_indices) or not set(args.fold_indices) <= {1, 2, 3}:
        parser.error("Use unique fold indices from 1 to 3")
    if smoke and (len(args.fold_indices) != 1 or len(args.horizons) != 1):
        parser.error("Smoke mode requires one fold and one horizon")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    print(f"Loading DSHI and injection labels ({args.data_protocol})", flush=True)
    cache, index, scenarios, dshi, bounds = load_locked_inputs(
        exp7, cfg, folds, args.fold_indices, smoke, supplement
    )
    manifest = {
        "mode": args.mode, "command": [sys.executable, *sys.argv],
        "data_protocol": args.data_protocol,
        "lookback": LOOKBACK, "horizons": args.horizons, "folds": args.fold_indices,
        "scenarios": len(scenarios), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": str(torch.__version__), "seed_base": BASE_SEED,
        "variants": list(VARIANTS),
        "metadata": str(metadata_path), "dshi_cache": str(cache),
        "label_root": str(scenarios[0].label_path.parent), "index": str(index),
        "source_sha256": {str(p): file_hash(p) for p in (Path(__file__), ORIGINAL, MODEL, metadata_path, index)},
        "checkpoint_selection": "same seeded validation-window split within the development dates",
        "window_settings": {k: cfg[k] for k in (
            "slide_step", "smooth_window", "sample_mode", "max_windows_per_scenario",
            "max_test_windows_per_scenario", "max_deep_train"
        )},
    }
    if not smoke:
        manifest["dshi_sha256"] = file_hash(cache)
        if not supplement:
            manifest.update(diagnostics=str(DIAGNOSTICS), reference=str(REFERENCE))
    (out / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame([
        {"scenario": sc.name, "date": sc.date, "type": sc.fault_type, "severity": sc.severity,
         "ratio": sc.pct, "length": bounds[sc.name][0], "fault_start": bounds[sc.name][1],
         "fault_end_exclusive": bounds[sc.name][2], "label_path": str(sc.label_path)}
        for sc in scenarios
    ]).to_csv(out / "scenario_manifest.csv", index=False)
    if not (smoke or supplement):
        ref = pd.read_csv(REFERENCE).loc[lambda frame: frame.Method.eq("xLSTM-Tuned")]
        diag = pd.read_csv(DIAGNOSTICS).loc[lambda frame: frame.Method.eq("xLSTM-Tuned")]
    rows, audit_rows = [], []
    for fold in args.fold_indices:
        held_out = folds[fold - 1]
        for horizon in args.horizons:
            print(f"Preparing fold={fold}, L={LOOKBACK}, H={horizon}, mode={args.mode}", flush=True)
            common = (scenarios, dshi, bounds)
            window = (LOOKBACK, horizon, cfg["slide_step"], 1)
            seed = BASE_SEED + fold + LOOKBACK + horizon
            window_seed = BASE_SEED + fold + LOOKBACK * 100 + horizon
            train = exp7.build_window_data(
                *common, set(dates) - held_out, *window, cfg["max_windows_per_scenario"],
                cfg["sample_mode"], window_seed
            )
            test = exp7.build_window_data(
                *common, held_out, *window, cfg["max_test_windows_per_scenario"],
                cfg["sample_mode"], window_seed + 10_000
            )
            settings = {
                "Fold": fold, "Lookback": LOOKBACK, "Horizon": horizon, "Seed": seed,
                "Train Dates": ";".join(sorted(set(train.date))),
                "Test Dates": ";".join(sorted(set(test.date))),
                "Train Windows": len(train.hist), "Test Windows": len(test.hist),
            }
            train_dates, test_dates = set(train.date), set(test.date)
            if not train_dates or not test_dates or train_dates & test_dates:
                raise ValueError("Development and test windows must use disjoint, nonempty date sets")
            if not smoke and (train_dates != set(dates) - held_out or test_dates != held_out):
                raise ValueError("At least one date is missing from the constructed windows")
            stats = input_statistics(train, seed, cfg["max_deep_train"])
            if not (smoke or supplement):
                stats.update(audit_historical(fold, horizon, train, test, stats, ref, diag))
                print("  Dates, window counts, and input statistics match V2.0.", flush=True)
            audit_rows.append({**settings, **stats})
            pd.DataFrame(audit_rows).to_csv(out / "input_audit.csv", index=False)
            if args.mode == "audit":
                continue
            for variant in VARIANTS:
                started = perf_counter()
                ckpt = out / "checkpoints" / f"fold{fold}_L{LOOKBACK}_H{horizon}_{variant}"
                print(f"  Training {variant}, {len(train.hist)} development and {len(test.hist)} test windows", flush=True)
                with model_variant(exp7.LEGACY, variant):
                    pred, history, details = exp7.fit_predict_model(
                        "xLSTM-Tuned", train, test, seed, smoke, cfg["max_deep_train"], ckpt
                    )
                if pred.shape != test.target.shape or not np.isfinite(pred).all():
                    raise ValueError(f"Invalid predictions from {variant}")
                metric = exp7.metrics(test.target, pred)
                rows.append({
                    **settings, "Variant": variant, **metric, **details,
                    "elapsed_sec": round(perf_counter() - started, 3), "checkpoint": str(ckpt.with_suffix(".pt")),
                })
                pd.DataFrame(history).to_csv(out / f"{ckpt.name}_history.csv", index=False)
                save_results(out, rows)
                print(f"  {variant}: {metric}", flush=True)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
