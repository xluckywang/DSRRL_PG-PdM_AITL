"""R2-6 fixed-threshold replay of the frozen EXP3 configuration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch


ROOT = Path(os.environ.get("EXP3_ROOT", Path(__file__).resolve().parents[1])).resolve()
LOCK = Path(os.environ.get("EXP3_LOCK_DIR", Path(__file__).resolve().parent))
DAILY = ROOT / "data/monitoring/line_1"
LABELS = ROOT / "data/labels/line_1"
SCENARIOS = ROOT / "data/scenarios/line_1"
FEATURES = ["X", "Y", "Z", "T"]
SEEDS = (42, 43, 44, 45, 46)
FAULTS = ("crack", "imbalance", "mechanical_looseness", "thermal")
LEVELS = (1, 3, 6, 9)
FRACTIONS = (1, 5, 10)
RADIUS = 30
DELAY = 33
ALARM_BUDGET = 0.01
STATES = {"normal": (2,), "transitions": (1, 3, 4, 5), "stopped": (0,)}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_locked():
    file = LOCK / "exp3_model_selection.py"
    spec = importlib.util.spec_from_file_location("locked_exp3", file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod, sys.modules["config"]


def protocol(config):
    train = config.get_train_dates()
    development = config.get_development_dates()
    test = config.get_test_dates()
    if (len(train), len(development), len(test)) != (29, 24, 5):
        raise ValueError("Unexpected cohort sizes")
    if len(set(train + development + test)) != 58:
        raise ValueError("Record IDs must be unique across and within split roles")
    return train, development, test


def read_day(date):
    raw = pd.read_csv(DAILY / f"{date}_consolidated_data.csv", usecols=["Timestamp", *FEATURES])
    labels = pd.read_csv(LABELS / f"{date}_labels.csv")
    if not np.array_equal(labels["index"].to_numpy(), np.arange(len(raw))):
        raise ValueError(f"{date}: semantic labels not row-aligned")
    ts = pd.to_datetime(raw.pop("Timestamp")).to_numpy(dtype="datetime64[s]")
    if np.isnat(ts).any() or np.any(np.diff(ts).astype("int64") <= 0):
        raise ValueError(f"{date}: timestamps not strictly increasing")
    states = labels.state.to_numpy(dtype=np.int8)
    if not np.isin(states, range(6)).all():
        raise ValueError(f"{date}: invalid operating state")
    return ts, raw[FEATURES].to_numpy(dtype=np.float32), states


def spans(ts, observed):
    """Yield contiguous runs of genuinely observed four-channel records."""
    indices = np.flatnonzero(observed)
    if not len(indices):
        return
    boundaries = np.flatnonzero(
        (np.diff(indices) != 1) | (np.diff(ts[indices]).astype("int64") != 1)
    ) + 1
    for run in np.split(indices, boundaries):
        if len(run) >= 2 * DELAY + 1:
            yield int(run[0]), int(run[-1]) + 1


def smooth_locked(x):
    # The archived np.mean call reduces both axes and broadcasts the scalar.
    means = x.sum(axis=1, dtype=np.float64)
    sums = np.r_[0.0, np.cumsum(means)]
    pos = np.arange(len(x))
    left, right = np.maximum(pos - RADIUS, 0), np.minimum(pos + RADIUS + 1, len(x))
    avg = (sums[right] - sums[left]) / ((right - left) * x.shape[1])
    return np.repeat(avg.astype(x.dtype)[:, None], x.shape[1], axis=1)


def infer(mod, model, scaled):
    window = mod._build_sliding_windows(scaled, 7)
    z, recon, err = [], [], []
    for i in range(0, len(scaled), 8192):
        x, w = scaled[i:i + 8192], window[i:i + 8192]
        z.append(mod._encode(model, x, w, "Conv1DAE"))
        r, e = mod._recon(model, x, w, "Conv1DAE")
        recon.append(x - r)
        err.append(e)
    return smooth_locked(np.concatenate(z)), np.concatenate(err), smooth_locked(np.concatenate(recon))


def fit(mod, train, seed):
    parts = []
    for date in train:
        x, y = mod._load_monitoring_day("line_1", date)
        parts.append(x[y == 2].astype(np.float32))
    healthy = np.concatenate(parts)
    cap = mod.EXP3.max_train_healthy
    if len(healthy) > cap:
        healthy = healthy[np.random.default_rng(42).choice(len(healthy), size=cap, replace=False)]
    scaler = StandardScaler().fit(healthy)
    scaled = scaler.transform(healthy).astype(np.float32)
    windows = mod._build_sliding_windows(scaled, 7)
    model, loss = mod._train_ae(scaled, windows, "Conv1DAE", 8, 100, seed)
    z, e, r = infer(mod, model, scaled)
    latent = mod._build_baseline(z, "DBSCAN", seed)
    residual = mod._build_baseline(r, "DBSCAN", seed)
    refs = [mod._dist_to_baseline(z, latent, "DBSCAN"), e, mod._dist_to_baseline(r, residual, "DBSCAN")]
    stats = [v for a in refs for v in (float(np.mean(a)), float(np.std(a) + 1e-12))]
    return model, scaler, latent, residual, stats, loss


def score(mod, fitted, x, ranges):
    model, scaler, latent, residual, stat, _ = fitted
    scores = np.full(len(x), np.nan, dtype=np.float32)
    for start, stop in ranges:
        z, e, r = infer(mod, model, scaler.transform(x[start:stop]).astype(np.float32))
        a = mod._dist_to_baseline(z, latent, "DBSCAN")
        b = mod._dist_to_baseline(r, residual, "DBSCAN")
        scores[start + DELAY:stop - DELAY] = mod._score_ours_ab(
            a, e, b, stat[0], stat[1], stat[2], stat[3], stat[4], stat[5]
        )[DELAY:-DELAY]
    return scores


def read_scenario(name, ts, day_states):
    x = pd.read_parquet(SCENARIOS / f"{name}_data.parquet", columns=["Timestamp", *FEATURES])
    y = pd.read_parquet(SCENARIOS / f"{name}_labels.parquet", columns=["fault_label", "state"])
    if not np.array_equal(pd.to_datetime(x["Timestamp"]).to_numpy(dtype="datetime64[s]"), ts):
        raise ValueError(f"{name}: scenario timestamps differ from background")
    if not np.array_equal(y.state.to_numpy(), day_states) or not y.fault_label.isin([0, 1]).all():
        raise ValueError(f"{name}: scenario labels inconsistent with background")
    values = x[FEATURES].to_numpy(dtype=np.float32)
    return values, y.fault_label.to_numpy(dtype=bool)


def count_row(score_vec, fault, subset, threshold):
    eligible = np.isfinite(score_vec) & subset
    alarm = eligible & (score_vec > threshold)
    return dict(
        total=int(subset.sum()), scored=int(eligible.sum()),
        fault_total=int((subset & fault).sum()), fault_scored=int((eligible & fault).sum()),
        tp=int((alarm & fault).sum()), fp=int((alarm & ~fault).sum()),
        fn=int((eligible & ~alarm & fault).sum()), tn=int((eligible & ~alarm & ~fault).sum()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    mod, config = load_locked()
    train, dev, test = protocol(config)
    # Check the optimized replay against the archived smoothing implementation.
    sample = np.random.default_rng(42).normal(size=(200, 8)).astype(np.float32)
    np.testing.assert_allclose(smooth_locked(sample), mod._window_smooth(sample, RADIUS), atol=1e-7)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    train_inputs = [p for d in train for p in (
        DAILY / f"{d}_preprocessed.csv", LABELS / f"{d}_labels.csv")]
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in train_inputs}
    manifest = dict(seed=args.seed, train=train, development=dev, test=test,
                    nominal_replay_delay_seconds=DELAY, threshold_rule="normal-development 0.99 higher quantile; alarm iff score > threshold",
                    fault_types=FAULTS, severity=LEVELS, injection_percent=FRACTIONS, low_severity=[1, 3],
                    model_driver_sha256=sha(LOCK / "exp3_model_selection.py"),
                    config_sha256=sha(LOCK / "config.py"),
                    driver_sha256=sha(Path(__file__)), python=sys.version, torch=str(torch.__version__),
                    train_input_sha256=inputs)
    (out / "protocol.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    dates = {}
    for d in [*dev, *test]:
        ts, x, state = read_day(d)
        observed = np.isfinite(x).all(axis=1)
        ranges = list(spans(ts, observed))
        dates[d] = ts, x, state, observed, ranges
    cases = [(d, f, p, v, f"{d}_{f}_pct{p}_sev{v}") for d in test
             for f in FAULTS for p in FRACTIONS for v in LEVELS]
    audit = []
    for d, f, p, v, name in cases:
        ts, _, state, observed, _ = dates[d]
        x, y = read_scenario(name, ts, state)
        if not y.any():
            raise ValueError(f"{name}: no annotated injection")
        bad = observed & ~np.isfinite(x).all(axis=1)
        audit.append(dict(date=d, mechanism=f, injection_pct=p, severity=v, scenario=name,
                          fault_rows=int(y.sum()), total_rows=len(y),
                          invalid_augmented_rows_on_observed_background=int(bad.sum())))
    pd.DataFrame(audit).to_csv(out / "scenario_manifest.csv", index=False)
    pd.DataFrame([
        dict(date=d, split="development" if d in dev else "test", rows=len(ts),
             complete_sensor_rows=int(obs.sum()), scored_context_rows=sum(b-a-2*DELAY for a,b in ranges))
        for d, (ts, _, _, obs, ranges) in dates.items()
    ]).to_csv(out / "date_coverage.csv", index=False)
    print(f"Validated {len(train)} training / {len(dev)} development / {len(test)} test dates; {len(cases)} scenarios", flush=True)
    if args.audit_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("The EXP3 fitting driver requires a CUDA-enabled PyTorch environment")
    torch.set_num_threads(4)
    mod._set_device(0)
    fitted = fit(mod, train, args.seed)
    model, scaler, latent, residual, stat, loss = fitted
    torch.save(model.state_dict(), out / "model.pt")
    np.savez(out / "reference.npz", mean=scaler.mean_, scale=scaler.scale_, latent=latent,
             residual=residual, stats=stat)
    dev_scores = []
    for d in dev:
        _, x, state, _, ranges = dates[d]
        vals = score(mod, fitted, x, ranges)
        dev_scores.append(vals[(state == 2) & np.isfinite(vals)])
    arr = np.concatenate(dev_scores)
    if not len(arr):
        raise ValueError("No eligible development data")
    threshold = float(np.quantile(arr, 1.0 - ALARM_BUDGET, method="higher"))
    (out / "threshold.json").write_text(json.dumps(dict(threshold=threshold,
        stable_development_n=len(arr), development_fpr=float((arr > threshold).mean()),
        train_loss=loss), indent=2), encoding="utf-8")
    print(f"Threshold frozen: {threshold:.6f}, development n={len(arr)}", flush=True)
    rows, pooled = [], {"all": [], "low": []}
    score_dir = out / "scores"
    score_dir.mkdir()
    for d in test:
        ts, x, state, _, ranges = dates[d]
        vals = score(mod, fitted, x, ranges)
        labels = np.zeros(len(x), dtype=bool)
        np.savez_compressed(score_dir / f"{d}_measured.npz", timestamp=ts, state=state,
                            fault=labels, score=vals)
        for group, states in STATES.items():
            rows.append(dict(date=d, scenario="measured", group=group,
                             **count_row(vals, labels, np.isin(state, states), threshold)))
    for d, f, p, v, name in cases:
        ts, _, state, obs, _ = dates[d]
        x, fault = read_scenario(name, ts, state)
        vals = score(mod, fitted, x, spans(ts, obs & np.isfinite(x).all(axis=1)))
        np.savez_compressed(score_dir / f"{name}.npz", timestamp=ts, state=state,
                            fault=fault, score=vals)
        subset = fault | (state == 2)
        valid = subset & np.isfinite(vals)
        row = dict(date=d, scenario=name, mechanism=f, injection_pct=p, severity=v, group="constructed",
                   **count_row(vals, fault, subset, threshold))
        rows.append(row)
        pair = vals[valid], fault[valid]
        pooled["all"].append(pair)
        if v in (1, 3):
            pooled["low"].append(pair)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "per_case_counts.csv", index=False)
    metrics = []
    for group, records in pooled.items():
        scores, targets = [np.concatenate(items) for items in zip(*records)]
        tp, fp = np.sum((scores > threshold) & targets), np.sum((scores > threshold) & ~targets)
        selected = frame.loc[frame.group.eq("constructed") & (
            frame.severity.isin((1, 3)) if group == "low" else True)]
        metrics.append(dict(group=group, auc_roc=roc_auc_score(targets, scores),
                            precision=tp/(tp+fp) if tp+fp else np.nan,
                            recall=tp/targets.sum(), fpr=fp/(~targets).sum(),
                            coverage=selected.scored.sum()/selected.total.sum(),
                            fault_coverage=selected.fault_scored.sum()/selected.fault_total.sum(),
                            tp=int(tp), fp=int(fp), fn=int(targets.sum()-tp), tn=int((~targets).sum()-fp)))
    for group in STATES:
        sums = frame.loc[frame.group.eq(group), ["total", "scored", "fp", "tn"]].sum()
        metrics.append(dict(group=f"measured_{group}", fpr=sums.fp/sums.scored if sums.scored else np.nan,
                            coverage=sums.scored/sums.total, **sums.to_dict()))
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
