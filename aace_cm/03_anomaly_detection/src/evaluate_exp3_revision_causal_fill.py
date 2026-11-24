"""Evaluate the saved Exp3 replay checkpoint with short causal gap filling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch

import evaluate_exp3_revision_alarm as replay


MAX_GAP_SECONDS = 5


def dense_input(ts, x):
    # Forward fill per channel; a missing value never reads a later sample.
    elapsed = (ts - ts[0]).astype(np.int64)
    dense_ts = ts[0] + np.arange(elapsed[-1] + 1).astype("timedelta64[s]")
    dense = np.full((len(dense_ts), x.shape[1]), np.nan, dtype=np.float32)
    dense[elapsed] = np.where(np.isfinite(x), x, np.nan)
    filled = pd.DataFrame(dense).ffill(limit=MAX_GAP_SECONDS).to_numpy()
    return elapsed, dense_ts, filled


def scores_at_observations(mod, fitted, ts, values):
    index, dense_ts, filled = dense_input(ts, values)
    return replay.score(mod, fitted, filled, replay.spans(dense_ts, np.isfinite(filled).all(axis=1)))[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    mod, config = replay.load_locked()
    train, dev, test = replay.protocol(config)
    protocol = json.loads((args.fit / "protocol.json").read_text(encoding="utf-8"))
    if any(protocol[k] != v for k, v in (("train", train), ("development", dev), ("test", test))):
        raise ValueError("Checkpoint and evaluation date manifests disagree")
    mod._set_device(0)
    model = mod._build_ae("Conv1DAE", 4, 8, [], []).to(mod.DEVICE)
    model.load_state_dict(torch.load(args.fit / "model.pt", map_location=mod.DEVICE, weights_only=True))
    with np.load(args.fit / "reference.npz") as ref:
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_ = ref["mean"], ref["scale"]
        fitted = model, scaler, ref["latent"], ref["residual"], ref["stats"], None
    args.out.mkdir(parents=True, exist_ok=False)
    score_dir = args.out / "scores"
    score_dir.mkdir()
    protocol.update(short_causal_fill_seconds=MAX_GAP_SECONDS, checkpoint=str(args.fit.resolve()),
                    checkpoint_sha256=replay.sha(args.fit / "model.pt"),
                    evaluation_driver_sha256=replay.sha(Path(__file__)))
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    parts = []
    for d in dev:
        ts, x, states = replay.read_day(d)
        vals = scores_at_observations(mod, fitted, ts, x)
        parts.append(vals[(states == 2) & np.isfinite(vals)])
    calibration = np.concatenate(parts)
    if not len(calibration):
        raise ValueError("No eligible development observations")
    threshold = float(np.quantile(calibration, 1.0 - replay.ALARM_BUDGET, method="higher"))
    (args.out / "threshold.json").write_text(json.dumps(dict(
        threshold=threshold, stable_development_n=len(calibration),
        development_fpr=float((calibration > threshold).mean()),
    ), indent=2), encoding="utf-8")
    print(f"Threshold frozen: {threshold:.6f}; {len(calibration)} development scores", flush=True)

    rows, pooled = [], {"all": [], "low": []}
    for d in test:
        ts, raw, state = replay.read_day(d)
        missing = ~np.isfinite(raw)
        vals = scores_at_observations(mod, fitted, ts, raw)
        negatives = np.zeros(len(raw), dtype=bool)
        np.savez_compressed(score_dir / f"{d}_measured.npz", timestamp=ts, state=state,
                            fault=negatives, score=vals)
        for group, states in replay.STATES.items():
            rows.append(dict(date=d, scenario="measured", group=group,
                             **replay.count_row(vals, negatives, np.isin(state, states), threshold)))
        for fault in replay.FAULTS:
            for pct in replay.FRACTIONS:
                for severity in replay.LEVELS:
                    name = f"{d}_{fault}_pct{pct}_sev{severity}"
                    x, y = replay.read_scenario(name, ts, state)
                    x[missing] = np.nan
                    vals = scores_at_observations(mod, fitted, ts, x)
                    subset = y | (state == 2)
                    valid = subset & np.isfinite(vals)
                    rows.append(dict(date=d, scenario=name, mechanism=fault, injection_pct=pct,
                                     severity=severity, group="constructed",
                                     **replay.count_row(vals, y, subset, threshold)))
                    pair = vals[valid], y[valid]
                    pooled["all"].append(pair)
                    if severity in (1, 3):
                        pooled["low"].append(pair)
                    np.savez_compressed(score_dir / f"{name}.npz", timestamp=ts, state=state,
                                        fault=y, score=vals)
        print(f"Completed test date {d}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "per_case_counts.csv", index=False)
    metrics = []
    for group, records in pooled.items():
        scores, y = [np.concatenate(items) for items in zip(*records)]
        pred = scores > threshold
        tp, fp = int(np.sum(pred & y)), int(np.sum(pred & ~y))
        n_pos, n_neg = int(y.sum()), int((~y).sum())
        subset = frame.loc[frame.group.eq("constructed") & (
            frame.severity.isin((1, 3)) if group == "low" else True)]
        metrics.append(dict(group=group, auc_roc=roc_auc_score(y, scores),
                            precision=tp/(tp+fp) if tp+fp else np.nan, recall=tp/n_pos, fpr=fp/n_neg,
                            coverage=subset.scored.sum()/subset.total.sum(),
                            fault_coverage=subset.fault_scored.sum()/subset.fault_total.sum(),
                            tp=tp, fp=fp, fn=n_pos-tp, tn=n_neg-fp))
    for group in replay.STATES:
        sums = frame.loc[frame.group.eq(group), ["total", "scored", "fp", "tn"]].sum()
        metrics.append(dict(group=f"measured_{group}", fpr=sums.fp/sums.scored if sums.scored else np.nan,
                            coverage=sums.scored/sums.total, **sums.to_dict()))
    summary = pd.DataFrame(metrics)
    summary.to_csv(args.out / "metrics.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
