"""Recover ratio-stratified fixed-threshold metrics from the locked EXP3 replay."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


FAULTS = {
    "crack": "Crack", "imbalance": "Imbalance",
    "mechanical_looseness": "Mechanical looseness", "thermal": "Thermal degradation",
}
METRICS = {"precision": "Precision", "recall": "Recall", "fpr": "False-alarm rate"}
RATIOS, LEVELS, SEEDS = (1, 5, 10), tuple(range(1, 11)), tuple(range(42, 47))
COUNTS = ["tp", "fp", "fn", "tn"]
PALETTE = ["#e3eef6", "#d0e4f1", "#b8d7e9", "#9bc6dd", "#80b5d2",
           "#65a2c5", "#4e8db4", "#3d77a0", "#30618b", "#234c73"]


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def evaluate(score, label, cutoff):
    hit = score > cutoff
    tp = int((hit & label).sum())
    fp = int((hit & ~label).sum())
    positive = int(label.sum())
    negative = len(label) - positive
    assert positive and negative
    return dict(tp=tp, fp=fp, fn=positive-tp, tn=negative-fp,
                auc_roc=roc_auc_score(label, score),
                precision=tp/(tp+fp) if tp+fp else np.nan, recall=tp/positive, fpr=fp/negative)


def collect(fusion, out, local=False):
    plan_file = fusion / ("plan_v2/plan.json" if local else "plan/plan.json")
    plan, selection = read_json(plan_file), read_json(fusion / "frozen/selection.json")
    assert sha(plan_file) == selection["plan_sha"]
    assert selection["candidates"]["locked"]["mode"] == "locked"
    archive = pd.read_csv(fusion / "results/by_seed.csv").query("candidate == 'locked' and role == 'test'")
    archive = archive.set_index(["seed", "mechanism", "level"])
    assert len(archive) == 200 and archive.index.is_unique
    rows, checks = [], []
    for seed in SEEDS:
        if local:
            earlier = fusion.parent / f"r2_6_20261109/seed_{seed}_causal_fill"
            complete = fusion.parent / f"all_severity_20261109/seed_{seed}"
            protocol = read_json(complete / "protocol.json")
            assert protocol["test"] == plan["test"]
            assert protocol["original_model_sha256"] == plan["fits"][str(seed)]["model"]
            assert protocol["original_reference_sha256"] == plan["fits"][str(seed)]["reference"]
        else:
            folder = fusion / f"test_{seed}"
            protocol = read_json(folder / "verified.json")
            assert protocol["seed"] == seed and protocol["plan_sha"] == sha(plan_file)
            assert protocol["count"] == 600 and len(plan["test"]) == 5
        cutoff = np.float32(selection["cutoffs"][str(seed)]["locked"])
        for mechanism in FAULTS:
            for level in LEVELS:
                by_ratio = []
                for pct in RATIOS:
                    by_date = []
                    for day in plan["test"]:
                        case = f"{day}_{mechanism}_pct{pct}_sev{level}.npz"
                        if local:
                            folder = earlier / "scores" if level in (1, 3, 6, 9) else complete / "new_scores"
                        with np.load(folder / case) as data:
                            if local:
                                label, score = data["fault"].astype(bool), data["score"]
                                keep = (label | (data["state"] == 2)) & np.isfinite(score)
                                score, label = score[keep], label[keep]
                            else:
                                score, label = data["x"][:, -1], data["y"].astype(bool)
                        assert np.isfinite(score).all()
                        by_date.append((score, label))
                    score, label = (np.concatenate(v) for v in zip(*by_date))
                    by_ratio.append((score, label))
                    rows.append(dict(seed=seed, mechanism=mechanism, level=level, pct=pct,
                                     dates=len(by_date), cutoff=float(cutoff),
                                     **evaluate(score, label, cutoff)))
                score, label = (np.concatenate(v) for v in zip(*by_ratio))
                result = evaluate(score, label, cutoff)
                expected = archive.loc[(seed, mechanism, level)]
                diff = {f"{k}_difference": result[k]-expected[k] for k in [*COUNTS,"auc_roc"]}
                for k in ("precision", "recall", "fpr"):
                    diff[f"{k}_difference"] = result[k]-expected[k]
                assert diff["tp_difference"] + diff["fn_difference"] == 0, (seed, mechanism, level, diff)
                assert diff["fp_difference"] + diff["tn_difference"] == 0, (seed, mechanism, level, diff)
                if not local:
                    assert all(diff[f"{k}_difference"] == 0 for k in COUNTS), (seed, mechanism, level, diff)
                    assert abs(diff["auc_roc_difference"]) < 1e-12, (seed, mechanism, level, diff)
                checks.append(dict(seed=seed, mechanism=mechanism, level=level, **diff))
        print(f"Seed {seed}: checked reaggregated counts against the Table Y archive.", flush=True)
    rows, checks = pd.DataFrame(rows), pd.DataFrame(checks)
    assert len(rows) == 600 and len(checks) == 200
    out.mkdir(parents=True, exist_ok=False)
    rows.to_csv(out / "ratio_metrics_by_seed.csv", index=False)
    checks.to_csv(out / "table_y_reaggregation_check.csv", index=False)
    receipt = {"plan_sha": sha(plan_file), "archive_sha": sha(fusion / "results/by_seed.csv"),
               "selection_sha": sha(fusion / "frozen/selection.json"),
               "rows_sha": sha(out / "ratio_metrics_by_seed.csv"),
               "checks_sha": sha(out / "table_y_reaggregation_check.csv"), "local_scores": local}
    (out / "verified.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def draw_metric_grid(stats, metric, axes, *, closed=False, compact=False, column_titles=True):
    from matplotlib.ticker import MaxNLocator

    if metric == "fpr":
        observed_max = (stats[f"{metric}_mean"] + stats[f"{metric}_sd"]).max()
        ytop = float(np.ceil((1.38 if compact else 1.28)*observed_max*2)/2)
        ticks = MaxNLocator(nbins=4).tick_values(0, ytop/1.12)
        ticks = ticks[ticks <= ytop*.95]
    else:
        ytop, ticks = (135 if compact else 126), [0, 25, 50, 75, 100]
    for i, pct in enumerate(RATIOS):
        for j, (fault, title) in enumerate(FAULTS.items()):
            ax = axes[i, j]
            subset = stats[(stats.pct == pct) & (stats.mechanism == fault)].set_index("level").loc[list(LEVELS)]
            v, sd = (subset[f"{metric}_{k}"].to_numpy(dtype=float) for k in ("mean", "sd"))
            if closed:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("#48515a")
                    spine.set_linewidth(.6)
            else:
                ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#e6ebef", lw=.46)
            ax.set_axisbelow(True)
            ax.set(xlim=(.48,10.52), ylim=(0,ytop), xticks=LEVELS, yticks=ticks)
            ax.tick_params(labelsize=6.8, length=2.4, width=.55, pad=1.5)
            ax.tick_params(labelbottom=i == len(RATIOS)-1, labelleft=j == 0)
            valid = np.isfinite(v)
            x = np.asarray(LEVELS)[valid]
            color = np.asarray(PALETTE)[valid].tolist()
            ax.bar(x, v[valid], width=.75, color=color, edgecolor="#506777", linewidth=.42)
            uncertain = valid & np.isfinite(sd)
            ax.errorbar(np.asarray(LEVELS)[uncertain], v[uncertain],
                        yerr=[np.minimum(v[uncertain],sd[uncertain]),sd[uncertain]],
                        fmt="none", color="#292f35", elinewidth=.55, capsize=1.15, capthick=.55)
            for level, value, s in zip(LEVELS, v, sd):
                label = f"{value:.2f}" if np.isfinite(value) else "NA"
                y = value + (s if np.isfinite(s) else 0) if np.isfinite(value) else 0
                ax.annotate(label, (level, y), xytext=(0,1.4), textcoords="offset points",
                            rotation=90, ha="center", va="bottom",
                            fontsize=5.6 if compact else 5.9, color="#263442")
            if i == 0 and column_titles:
                ax.set_title(title, fontsize=8.7, pad=3 if compact else 4.5)
            if j == 0:
                ax.text(-.32, .5, f"{pct}%", transform=ax.transAxes, ha="center", va="center",
                        fontsize=8.5, weight="bold")


def plot_metric(stats, metric, out, export_figure, font, provenance, label=None):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(180/25.4, 106/25.4), dpi=300)
    fig.subplots_adjust(left=.123, right=.984, bottom=.131, top=.902, hspace=.24, wspace=.17)
    draw_metric_grid(stats, metric, axes)
    label = label or METRICS[metric]
    fig.supylabel(f"{label} (%)", x=.015, fontsize=9.5)
    fig.supxlabel("Injected severity level", y=.048, fontsize=9.5)
    export_figure(fig, out / f"EXP3_{metric}_ratio_severity", dpi=600,
                  provenance={**provenance, "font": font, "metric": label,
                              "labels": "Bar labels are means; whiskers show sample SD, lower bound clipped at zero."})
    plt.close(fig)


def main(root, data, out):
    from plotting import export_figure, publication_context

    receipt = read_json(data / "verified.json")
    for k, path in (
        ("plan", root / "plan_v2/plan.json"),
        ("archive", root / "results/by_seed.csv"),
        ("selection", root / "frozen/selection.json"),
        ("rows", data / "ratio_metrics_by_seed.csv"),
        ("checks", data / "table_y_reaggregation_check.csv"),
    ):
        assert sha(path) == receipt[f"{k}_sha"], path
    rows = pd.read_csv(data / "ratio_metrics_by_seed.csv")
    checks = pd.read_csv(data / "table_y_reaggregation_check.csv")
    assert len(checks) == 200 and len(rows) == 600
    assert (checks.tp_difference + checks.fn_difference).eq(0).all()
    assert (checks.fp_difference + checks.tn_difference).eq(0).all()
    diffs = checks[[f"{k}_difference" for k in [*METRICS, "auc_roc"]]].abs().max() * 100
    print("Maximum per-seed pooled differences from Table Y (percentage points):", diffs.to_dict())
    if receipt["local_scores"]:
        assert diffs.max() < .1, "Local cached scores diverge materially from Table Y"
    else:
        assert checks[[f"{k}_difference" for k in COUNTS]].eq(0).all().all()
        assert checks.auc_roc_difference.abs().max() < 1e-12
    keys = ["mechanism", "pct", "level"]
    assert rows.groupby(keys).size().eq(5).all()
    metrics = [*METRICS, "auc_roc"]
    stats = (rows.assign(**{k: rows[k]*100 for k in metrics}).groupby(keys)[metrics]
             .agg(["mean", "std", "count"]))
    stats.columns = [f"{k}_{'sd' if stat == 'std' else stat}" for k, stat in stats.columns]
    stats = stats.reset_index()
    assert len(stats) == 120
    out.mkdir(parents=True, exist_ok=False)
    rows.to_csv(out / "ratio_metrics_by_seed.csv", index=False)
    stats.to_csv(out / "ratio_metrics_summary_pct.csv", index=False)
    checks.to_csv(out / "table_y_reaggregation_check.csv", index=False)
    provenance = {
        "data": str(root.resolve()), "score": "Locked original dual-space score; original injection amplitude.",
        "local_replay_scores": receipt["local_scores"],
        "max_pooled_table_y_difference_percentage_points": diffs.to_dict(),
        "calibration": "Original locked-score threshold from eight development dates, separately per seed.",
        "aggregation": "Within each fault, level, ratio and seed, pool eligible positive and stable-operation "
                       "negative windows from the same five playback dates. Plot mean and sample SD over seeds.",
        "undefined_precision": "No predicted positives yields undefined precision, not zero. Use available seeds.",
        "ratio_vs_severity": "Rows are injection ratio; columns are mechanism; bar color is severity, not amplitude multiplier.",
    }
    with publication_context(font_family="Times New Roman", base_size=8) as font:
        for metric in METRICS:
            plot_metric(stats, metric, out, export_figure, font, provenance)
    print("Verified matched positive/negative window counts for all 200 Table Y groups.")
    print(f"Undefined precision: {rows.precision.isna().sum()} seed groups; "
          f"{stats.precision_count.eq(0).sum()} figure cells.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    mode = p.add_subparsers(dest="mode", required=True)
    for name, fields in (("collect", ("root", "out")), ("plot", ("root", "data", "out"))):
        sub = mode.add_parser(name)
        for key in fields:
            sub.add_argument(f"--{key}", required=True, type=Path)
        if name == "collect":
            sub.add_argument("--local", action="store_true")
    a = p.parse_args()
    if a.mode == "collect":
        collect(a.root, a.out, a.local)
    else:
        main(a.root, a.data, a.out)

