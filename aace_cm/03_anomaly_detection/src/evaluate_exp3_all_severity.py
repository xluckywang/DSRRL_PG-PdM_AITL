"""Extend the locked Exp3 alarm replay to all ten existing injection levels."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch

import evaluate_exp3_revision_alarm as replay
import evaluate_exp3_revision_causal_fill as causal


LEVELS = tuple(range(1, 11))
FIELDS = ("auc_roc", "precision", "recall", "fpr")


def collect(args):
    seed = args.seed
    previous = args.baseline / f"seed_{seed}_causal_fill"
    manifest = json.loads((previous / "protocol.json").read_text(encoding="utf-8"))
    cutoff = json.loads((previous / "threshold.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == seed
    assert replay.sha(Path(replay.__file__)) == manifest["driver_sha256"]
    assert replay.sha(Path(causal.__file__)) == manifest["evaluation_driver_sha256"]
    fit = args.baseline / ("seed_42" if seed == 42 else f"seed_{seed}_fit")
    assert replay.sha(fit / "model.pt") == manifest["checkpoint_sha256"]
    mod, config = replay.load_locked()
    train, dev, test = replay.protocol(config)
    assert (train, dev, test) == (manifest["train"], manifest["development"], manifest["test"])
    assert manifest["short_causal_fill_seconds"] == causal.MAX_GAP_SECONDS
    assert manifest["severity"] == list(replay.LEVELS)
    assert manifest["injection_percent"] == list(replay.FRACTIONS)
    assert manifest["fault_types"] == list(replay.FAULTS)
    torch.set_num_threads(args.threads)
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    mod.DEVICE = torch.device("cuda:0" if args.cuda else "cpu")
    model = mod._build_ae("Conv1DAE", 4, 8, [], []).to(mod.DEVICE)
    model.load_state_dict(torch.load(fit / "model.pt", map_location=mod.DEVICE, weights_only=True))
    with np.load(fit / "reference.npz") as ref:
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_ = ref["mean"], ref["scale"]
        fitted = model, scaler, ref["latent"], ref["residual"], ref["stats"], None

    threshold = cutoff["threshold"]
    archived = pd.read_csv(previous / "per_case_counts.csv").set_index("scenario")
    days = {date: replay.read_day(date) for date in test}
    args.out.mkdir(parents=True, exist_ok=False)
    score_dir = args.out / "new_scores"
    score_dir.mkdir()
    rows, metrics = [], []
    source_hashes = {}
    for mechanism in replay.FAULTS:
        for level in LEVELS:
            pairs, counts = [], []
            for date, (ts, raw, state) in days.items():
                for pct in replay.FRACTIONS:
                    name = f"{date}_{mechanism}_pct{pct}_sev{level}"
                    values, fault = replay.read_scenario(name, ts, state)
                    if not fault.any():
                        raise ValueError(f"{name}: no injection labels")
                    if level in replay.LEVELS:
                        path = previous / "scores" / f"{name}.npz"
                        with np.load(path) as old:
                            for key, actual in (("timestamp", ts), ("state", state), ("fault", fault)):
                                np.testing.assert_array_equal(old[key], actual)
                            score = old["score"]
                        source = "archived"
                    else:
                        values[~np.isfinite(raw)] = np.nan
                        score = causal.scores_at_observations(mod, fitted, ts, values)
                        np.savez_compressed(score_dir / f"{name}.npz", timestamp=ts, state=state,
                                            fault=fault, score=score)
                        source = "new"
                    subset = fault | (state == 2)
                    valid = subset & np.isfinite(score)
                    result = replay.count_row(score, fault, subset, threshold)
                    tn, fp, fn, tp = confusion_matrix(
                        fault[valid], score[valid] > threshold, labels=[False, True]).ravel()
                    assert (tp, fp, fn, tn) == tuple(result[k] for k in ("tp", "fp", "fn", "tn")), name
                    if source == "archived":
                        for key, value in result.items():
                            assert value == archived.loc[name, key], (seed, name, key)
                    counts.append(result)
                    pairs.append((score[valid], fault[valid]))
                    rows.append(dict(seed=seed, date=date, scenario=name, mechanism=mechanism,
                                     level=level, injection_pct=pct, source=source, **result))
                    source_hashes[name] = replay.sha(
                        path if source == "archived" else score_dir / f"{name}.npz")
            assert len(pairs) == 15
            score, fault = [np.concatenate(part) for part in zip(*pairs)]
            sums = {key: sum(row[key] for row in counts) for key in counts[0]}
            tp, fp, fn, tn = (sums[k] for k in ("tp", "fp", "fn", "tn"))
            assert tp + fn and fp + tn and tp + fn + fp + tn == len(score)
            metrics.append(dict(seed=seed, mechanism=mechanism, level=level,
                                auc_roc=roc_auc_score(fault, score),
                                precision=tp / (tp + fp) if tp + fp else np.nan,
                                recall=tp / (tp + fn), fpr=fp / (fp + tn),
                                fault_coverage=sums["fault_scored"] / sums["fault_total"],
                                **sums))
        print(f"Seed {seed}: completed {mechanism} (levels 1–10)", flush=True)
    assert len(rows) == 600 and len(metrics) == 40 and len(source_hashes) == 600
    pd.DataFrame(rows).to_csv(args.out / "scenario_counts.csv", index=False)
    pd.DataFrame(metrics).to_csv(args.out / "by_seed.csv", index=False)
    (args.out / "protocol.json").write_text(json.dumps(dict(
        seed=seed, source_run=str(previous.resolve()), test=test,
        development=dev, train=train, original_threshold=cutoff,
        original_model_sha256=manifest["checkpoint_sha256"],
        original_reference_sha256=replay.sha(fit / "reference.npz"),
        original_replay_sha256=manifest["evaluation_driver_sha256"],
        original_model_driver_sha256=manifest["driver_sha256"],
        score_archive_sha256=source_hashes, device=str(mod.DEVICE),
        torch=str(torch.__version__), python=sys.version,
    ), indent=2), encoding="utf-8")


def summarize(args):
    frames, cases, thresholds = [], [], []
    old = pd.read_csv(args.baseline / "multiseed_severity/by_seed_fault_and_level.csv")
    coverage = None
    counts = ["total", "scored", "fault_total", "fault_scored"]
    for seed in replay.SEEDS:
        directory = args.runs / f"seed_{seed}"
        table = pd.read_csv(directory / "by_seed.csv")
        scenarios = pd.read_csv(directory / "scenario_counts.csv")
        manifest = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
        threshold = json.loads((args.baseline / f"seed_{seed}_causal_fill/threshold.json").read_text(encoding="utf-8"))
        assert manifest["original_threshold"] == threshold
        assert len(scenarios) == 600 and not scenarios.scenario.duplicated().any()
        assert scenarios.groupby(["mechanism", "level"]).size().eq(15).all()
        assert scenarios.source.value_counts().to_dict() == {"new": 360, "archived": 240}
        assert len(table) == 40 and not table.duplicated(["mechanism", "level"]).any()
        assert set(table.level) == set(LEVELS) and set(table.mechanism) == set(replay.FAULTS)
        original = table.loc[table.level.isin(replay.LEVELS)].set_index(["mechanism", "level"]).sort_index()
        expected = old.loc[old.seed.eq(seed)].rename(columns={"severity": "level"})
        expected = expected.set_index(["mechanism", "level"]).sort_index()
        for k in ["precision", "recall", "fpr", "tp", "fp", "fn", "tn"]:
            np.testing.assert_allclose(original[k], expected[k], rtol=0, atol=1e-14)
        same = scenarios.set_index("scenario").sort_index()[counts]
        if coverage is None:
            coverage = same
        else:
            pd.testing.assert_frame_equal(coverage, same)
        thresholds.append(dict(seed=seed, **threshold, new_score_device=manifest["device"]))
        frames.append(table)
        cases.append(scenarios)
    table = pd.concat(frames, ignore_index=True)
    table["mechanism"] = pd.Categorical(table.mechanism, categories=replay.FAULTS, ordered=True)
    table = table.sort_values(["mechanism", "level", "seed"])
    aggregates = table.groupby(["mechanism", "level"], observed=True)[[*FIELDS, "fault_coverage"]].agg(
        ["mean", "std", "count"])
    aggregates.columns = [f"{k}_{s}" for k, s in aggregates.columns]
    aggregates = aggregates.reset_index()
    assert len(aggregates) == 40 and aggregates[[f"{k}_count" for k in FIELDS]].eq(5).all().all()
    table.to_csv(args.runs / "by_seed.csv", index=False, encoding="utf-8-sig")
    aggregates.to_csv(args.runs / "by_fault_and_level.csv", index=False, encoding="utf-8-sig")
    pd.concat(cases).to_csv(args.runs / "scenario_counts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(thresholds).to_csv(args.runs / "thresholds.csv", index=False, encoding="utf-8-sig")
    source = dict(summary=json.loads(aggregates.to_json(orient="records", double_precision=15)),
                  seeds=json.loads(table.to_json(orient="records", double_precision=15)),
                  thresholds=thresholds)
    (args.runs / "workbook_input.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print("Verified 40 fault/level groups, 200 seed-level rows and 3,000 case rows.")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(required=True)
    worker = commands.add_parser("seed")
    worker.add_argument("--seed", required=True, type=int, choices=replay.SEEDS)
    worker.add_argument("--baseline", required=True, type=Path)
    worker.add_argument("--out", required=True, type=Path)
    worker.add_argument("--cuda", action="store_true")
    worker.add_argument("--threads", type=int, default=3)
    worker.set_defaults(run=collect)
    aggregate = commands.add_parser("summary")
    aggregate.add_argument("--baseline", required=True, type=Path)
    aggregate.add_argument("--runs", required=True, type=Path)
    aggregate.set_defaults(run=summarize)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
