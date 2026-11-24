"""Extract and plot the ratio-stratified frozen condition-calibrated EXP3 results."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


KEYS = ["seed", "mechanism", "level"]
COUNTS = ["tp", "fp", "fn", "tn"]
METRICS = {"auc_roc": "AUC-ROC", "precision": "Precision", "recall": "Recall", "fpr": "False-alarm rate"}
GROUPED_LEVELS = (1, 3, 5, 7, 9)
RATIO_SEVERITY_COLORS = {
    1: ("#D6E5F1", "#AECBE2", "#7EACCE", "#4F88B5", "#2B6497"),
    5: ("#F3E8D3", "#E8D0A3", "#D5B070", "#B98B3F", "#936722"),
    10: ("#EFDFE9", "#E1BCD4", "#CB94B7", "#AE6898", "#864575"),
}
BLUE_RATIO_SEVERITY_COLORS = {
    1: ("#DDF0F6", "#B5DDEA", "#83C2D9", "#489EBD", "#1B7498"),
    5: ("#D2E3F3", "#A9CCE9", "#74ADD9", "#407EBE", "#24599B"),
    10: ("#D0D8ED", "#A1B0D8", "#7489BF", "#4F64A5", "#30427D"),
}
BLUE_RATIO_COLORS = {1: "#D8EDF7", 5: "#70B2D5", 10: "#3F67A4"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rates(frame):
    f = frame.copy()
    f["precision"] = f.tp / (f.tp + f.fp)
    f["recall"] = f.tp / (f.tp + f.fn)
    f["fpr"] = f.fp / (f.fp + f.tn)
    return f


def evaluate(parts, threshold):
    s, y = (np.concatenate(v) for v in zip(*parts))
    pred = s > threshold
    return dict(tp=int((y & pred).sum()), fp=int((~y & pred).sum()),
                fn=int((y & ~pred).sum()), tn=int((~y & ~pred).sum()), auc_roc=roc_auc_score(y, s))


def collect(root, out):
    import exp3_context_revision as ctx
    import evaluate_exp3_revision_alarm as replay
    from audit_exp3_paired_imbalance import build_scorer

    plan_path = root / "plan/plan.json"
    freeze_path = root / "context_frozen/selection.json"
    source_path = root / "context_results/by_seed.csv"
    inc_path = root / "frozen/selection.json"
    plan, inc, frozen = (read_json(p) for p in (plan_path, inc_path, freeze_path))
    plan_hash = sha(plan_path)
    assert inc["plan_sha"] == frozen["plan_sha"] == plan_hash
    assert frozen["incumbent_sha"] == sha(inc_path) and frozen["driver_sha"] == sha(Path(ctx.__file__))
    name = frozen["selected"]
    options = {name: frozen["options"][name]}
    incumbent = {inc["selected"]: inc["candidates"][inc["selected"]]}
    ref = pd.read_csv(source_path).query("candidate == @name and role == 'test'").set_index(KEYS).sort_index()
    assert len(ref) == 200 and ref.index.is_unique
    rows, combined = [], []
    for seed in replay.SEEDS:
        dev, test = root / f"dev_{seed}", root / f"test_{seed}"
        devctx, testctx = root / "context_dev", root / "context_test"
        for folder in (dev, test, devctx, testctx):
            assert read_json(folder / "verified.json")["plan_sha"] == plan_hash
        xs, zs = [], []
        for date in sorted(plan["calibration"]):
            x, = ctx.read(dev / f"cal_{date}.npz")
            z, = ctx.read(devctx / f"cal_{date}.npz")
            assert len(x) == len(z)
            xs.append(x)
            zs.append(z)
        assert len(xs) == 8
        score = build_scorer(np.concatenate(xs), np.concatenate(zs), incumbent, options)
        threshold = np.float32(frozen["cutoffs"][str(seed)][name])
        cases = pd.read_csv(test / "cases.csv")
        assert len(cases) == 600 and set(cases.day) == set(plan["test"])
        for (fault, level), group in cases.groupby(["mechanism", "level"]):
            pooled = []
            for pct, subgroup in group.groupby("pct"):
                assert set(subgroup.day) == set(plan["test"]) and len(subgroup) == 5
                parts = []
                for case in subgroup.itertuples(index=False):
                    x, y = ctx.read(test / f"{case.scenario}.npz")
                    z, y2 = ctx.read(testctx / f"{case.scenario}.npz")
                    np.testing.assert_array_equal(y, y2)
                    assert len(y) == case.eligible and y.sum() == case.positive
                    parts.append((score(x, z)[:, 0], y))
                result = evaluate(parts, threshold)
                rows.append(dict(seed=seed, mechanism=fault, level=level, pct=pct,
                                 dates=len(parts), cutoff=float(threshold), **result))
                pooled.extend(parts)
            combined.append(dict(seed=seed, mechanism=fault, level=level, **evaluate(pooled, threshold)))
        print(f"Seed {seed}: {len(cases)} archived scenarios", flush=True)
    frame, pooled = rates(pd.DataFrame(rows)), rates(pd.DataFrame(combined)).set_index(KEYS).sort_index()
    assert len(frame) == 600
    pd.testing.assert_index_equal(pooled.index, ref.index)
    pd.testing.assert_frame_equal(pooled[COUNTS], ref[COUNTS])
    np.testing.assert_allclose(pooled[list(METRICS)], ref[list(METRICS)], rtol=0, atol=1e-12)
    out.mkdir(parents=True, exist_ok=False)
    frame.to_csv(out / "ratio_metrics_by_seed.csv", index=False)
    pooled.to_csv(out / "pooled_archive_check.csv")
    (out / "verified.json").write_text(json.dumps({
        "plan": plan_hash, "selection": sha(freeze_path), "archive": sha(source_path),
        "ratio_metrics": sha(out / "ratio_metrics_by_seed.csv"),
        "pooled": sha(out / "pooled_archive_check.csv"), "score": name,
        "matched_groups": len(pooled),
    }, indent=2), encoding="utf-8")
    print("Exact archived confusion counts and matching AUC for all 200 pooled groups.")


def add_ratio_severity_key(fig, ratios, palette, ink):
    from matplotlib.patches import Rectangle

    columns = .476 + .036 * np.arange(len(GROUPED_LEVELS))
    fig.text(columns.mean(), .985, "Severity level", ha="center", va="top", fontsize=7.5, color=ink)
    for x, level in zip(columns, GROUPED_LEVELS):
        fig.text(x, .961, str(level), ha="center", va="center", fontsize=7.2, color=ink)
    fig.text(.410, .923, "Injection ratio", ha="right", va="center", fontsize=7.5, color=ink)
    for k, pct in enumerate(ratios):
        y = .941 - .018 * k
        fig.text(.449, y, f"{pct}%", ha="right", va="center", fontsize=7.2, color=ink)
        for x, color in zip(columns, palette[pct]):
            fig.add_artist(Rectangle((x - .014, y - .006), .028, .012, transform=fig.transFigure,
                                     facecolor=color, edgecolor=ink, linewidth=.35))


def grouped_plate(stats, *, palette=None, ratio_colors=None, seed_rows=None, compact_seeds=False,
                  auc_min=0, fpr_max=3):
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch
    from report_r26_ratio_bars import FAULTS, RATIOS

    levels = GROUPED_LEVELS
    colors = tuple(ratio_colors[pct] for pct in RATIOS) if ratio_colors else ("#86ADD0", "#6BA89A", "#D99879")
    ink = "#303942"
    show_seeds = seed_rows is not None
    width, left = (174, .071) if compact_seeds else (180, .105)
    fig, axes = plt.subplots(4, 4, sharex=True, sharey="row",
                             figsize=(width / 25.4, 150 / 25.4), dpi=300)
    top = .954 if show_seeds else (.864 if palette else .890)
    if ratio_colors:
        top = .916
    fig.subplots_adjust(left=left, right=.988, bottom=.081, top=top, wspace=.125, hspace=.14)
    data = stats.set_index(["mechanism", "pct", "level"], verify_integrity=True)
    shown = stats[stats.level.isin(levels)]
    assert len(shown) == len(FAULTS) * len(RATIOS) * len(levels)
    if show_seeds:
        seed_ids = sorted(seed_rows.seed.unique())
        assert len(seed_ids) == 5
        seed_offsets = np.linspace(-.105, .105, len(seed_ids))
        seed_data = (seed_rows[seed_rows.level.isin(levels)]
                     .set_index(["mechanism", "pct", "level", "seed"], verify_integrity=True).sort_index())
    positions = np.arange(len(levels))
    annotations = []
    row_labels = []
    point_count = 0
    overflow_count = 0
    for i, (metric, name) in enumerate(METRICS.items()):
        mean, sd = f"{metric}_mean", f"{metric}_sd"
        capped = compact_seeds and metric == "fpr"
        lower = auc_min if metric == "auc_roc" else 0
        upper = (shown[mean] + shown[sd]).max()
        if show_seeds:
            upper = max(upper, 100 * seed_data[metric].max())
        if capped:
            limit, yticks = fpr_max, np.arange(fpr_max + 1)
        elif metric == "fpr":
            limit = float(np.ceil(upper / .74 * 2) / 2)
            yticks = np.arange(0, limit, 1)
        elif lower:
            limit, yticks = max(115, np.ceil(upper + 12)), np.arange(lower, 101, 10)
            assert (shown[mean] - shown[sd]).min() >= lower
            if show_seeds:
                assert 100 * seed_data[metric].min() >= lower
        else:
            limit, yticks = max(135, np.ceil(upper / .78)), [0, 25, 50, 75, 100]
        for j, (fault, title) in enumerate(FAULTS.items()):
            ax = axes[i, j]
            panel_overflow = False
            for k, (pct, color) in enumerate(zip(RATIOS, colors)):
                values, errors = data.loc[(fault, pct), [mean, sd]].loc[list(levels)].to_numpy().T
                assert np.isfinite(values).all() and np.isfinite(errors).all()
                x = positions + (k - 1) * .267
                fills = palette[pct] if palette else (color,) * len(levels)
                outline = not show_seeds or compact_seeds
                bars = ax.bar(x, values, width=.245, color=fills,
                              edgecolor=ink if outline else "none", linewidth=.38 if outline else 0,
                              yerr=[np.minimum(values, errors), errors],
                              error_kw=dict(ecolor=ink, elinewidth=.48, capsize=1, capthick=.48))
                np.testing.assert_array_equal([bar.get_height() for bar in bars], values)
                np.testing.assert_array_equal([bar.get_facecolor() for bar in bars], [to_rgba(c) for c in fills])
                label_heights = values + errors
                if show_seeds:
                    points = 100 * (seed_data.loc[(fault, pct), metric].unstack("seed")
                                    .loc[list(levels), seed_ids].to_numpy())
                    np.testing.assert_allclose(points.mean(axis=1), values, rtol=0, atol=1e-12)
                    np.testing.assert_allclose(points.std(axis=1, ddof=1), errors, rtol=0, atol=1e-12)
                    point_x = x[:, None] + seed_offsets
                    visible = points <= limit if capped else np.ones_like(points, dtype=bool)
                    dots = ax.scatter(point_x[visible], points[visible], s=1.5,
                                      facecolors="#101722" if ratio_colors else "#182A3C",
                                      edgecolors="none" if ratio_colors else "white",
                                      linewidths=0 if ratio_colors else .16,
                                      zorder=4, clip_on=False)
                    np.testing.assert_array_equal(dots.get_offsets()[:, 1], points[visible])
                    if ratio_colors:
                        assert not len(dots.get_edgecolors())
                    point_count += len(dots.get_offsets())
                    if not visible.all():
                        markers = ax.scatter(point_x[~visible], np.full((~visible).sum(), limit),
                                             s=8, marker="^", color="#182A3C", linewidths=0,
                                             zorder=5, clip_on=False)
                        overflow_count += len(markers.get_offsets())
                        panel_overflow = True
                    label_heights = np.maximum(label_heights, points.max(axis=1))
                for xx, value, height in zip(x, values, label_heights):
                    # Move the mean inside the bar if its outside label would run into the axis limit.
                    inside = capped and height > limit * .86
                    annotations.append(ax.annotate(f"{value:.2f}", (xx, .12 if inside else height),
                                                   xytext=(0, 1.4 if show_seeds else 1.1),
                                                   textcoords="offset points", ha="center", va="bottom",
                                                   rotation=90, fontsize=5.35, color="white" if inside else ink))
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(ink)
                spine.set_linewidth(.55)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#e5e9ed", lw=.42)
            ax.set(xlim=(-.47, len(levels) - .53), ylim=(lower, limit), yticks=yticks)
            ax.set_xticks(positions, labels=levels)
            ax.tick_params(labelsize=6.7, length=2.0, width=.5, pad=1.6,
                           labelleft=j == 0, labelbottom=i == len(METRICS) - 1)
            if i == 0:
                ax.set_title(title, fontsize=8.3, fontweight="bold", pad=3.8)
            if panel_overflow:
                ax.text(.035, .94, f"Triangles: seeds above {fpr_max}%", transform=ax.transAxes,
                        va="top", fontsize=6.2, color=ink)
        box = axes[i, 0].get_position()
        row_labels.append(fig.text(.022, (box.y0 + box.y1) / 2, f"{name} (%)", ha="center", va="center",
                                   fontsize=8.2, weight="bold", rotation=90))
    if ratio_colors:
        handles = [Patch(facecolor=color, edgecolor=ink, linewidth=.4, label=f"{pct}% injection")
                   for pct, color in zip(RATIOS, colors)]
        legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=((left + .988) / 2, .985),
                            ncols=3, fontsize=7.8, frameon=False, borderaxespad=.2,
                            handlelength=1.2, handletextpad=.45, columnspacing=1.4)
    elif palette and not show_seeds:
        add_ratio_severity_key(fig, RATIOS, palette, ink)
    elif not show_seeds:
        handles = [Patch(facecolor=color, edgecolor=ink, linewidth=.4, label=f"{pct}%")
                   for pct, color in zip(RATIOS, colors)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.547, .981), ncols=3,
                   title="Injection ratio", title_fontsize=7.7, fontsize=7.8, frameon=False,
                   handlelength=1.1, handletextpad=.42, columnspacing=1.5, labelspacing=.2)
    fig.supxlabel("Injected severity level", x=(left + .988) / 2, y=.024, fontsize=8.2)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if ratio_colors:
        title_top = max(ax.title.get_window_extent(renderer).y1 for ax in axes[0])
        assert legend.get_window_extent(renderer).y0 - title_top >= fig.dpi / 25.4
    elif palette and not show_seeds:
        key_bottom = min(patch.get_window_extent(renderer).y0 for patch in fig.artists)
        title_top = max(ax.title.get_window_extent(renderer).y1 for ax in axes[0])
        assert key_bottom - title_top >= fig.dpi / 25.4
    for label in annotations:
        area = label.get_window_extent(renderer)
        assert label.axes.bbox.contains(area.x0, area.y0) and label.axes.bbox.contains(area.x1, area.y1), label
    if compact_seeds:
        assert all(ax.get_ylim()[0] == auc_min for ax in axes[0])
        assert all(ax.get_ylim() == (0, fpr_max) for ax in axes[-1])
        for ax, label in zip(axes[:, 0], row_labels):
            tick_left = min(tick.get_window_extent(renderer).x0 for tick in ax.get_yticklabels())
            assert tick_left - label.get_window_extent(renderer).x1 >= fig.dpi / 25.4
    assert len(fig.axes) == 16 and len(annotations) == 240
    if show_seeds:
        assert point_count + overflow_count == len(annotations) * len(seed_ids)
        assert overflow_count == ((100 * seed_data.fpr > fpr_max).sum() if compact_seeds else 0)
        assert not fig.artists and len(fig.legends) == int(ratio_colors is not None)
        print(f"Verified {point_count} seed dots and {overflow_count} overflow markers against the source values.")
    return fig


def plot(root, data, out, vertical_closed=False, compact=False, grouped_style=None,
         auc_min=0, fpr_max=3, case_counts=None):
    from report_r26_ratio_bars import draw_metric_grid, plot_metric
    from plotting import export_figure, publication_context
    from matplotlib import pyplot as plt

    receipt = read_json(data / "verified.json")
    for key, path in (
        ("plan", root / "plan/plan.json"),
        ("selection", root / "context_frozen/selection.json"),
        ("archive", root / "context_results/by_seed.csv"),
        ("ratio_metrics", data / "ratio_metrics_by_seed.csv"),
        ("pooled", data / "pooled_archive_check.csv"),
    ):
        assert sha(path) == receipt[key], path
    assert receipt["matched_groups"] == 200
    rows = pd.read_csv(data / "ratio_metrics_by_seed.csv")
    condition = ["mechanism", "pct", "level"]
    assert len(rows) == 600 and rows.groupby(condition).seed.nunique().eq(5).all()
    assert rows.groupby(condition).size().eq(5).all()
    if case_counts is not None:
        archived_counts = (pd.read_csv(case_counts).query("calibration_budget == .01")
                           .groupby(KEYS+["pct"])[COUNTS].sum().sort_index())
        pd.testing.assert_frame_equal(rows.set_index(KEYS+["pct"])[COUNTS].sort_index(), archived_counts)
    if not rows[list(METRICS)].notna().all().all():
        raise ValueError("The plot requires defined metrics for each seed and condition. Check the input and ratio table.")
    assert rows.dates.eq(5).all()
    stats = rows.assign(**{m: 100*rows[m] for m in METRICS}).groupby(condition)[list(METRICS)].agg(["mean","std"])
    stats.columns = [f"{m}_{'sd' if key == 'std' else key}" for m, key in stats.columns]
    stats = stats.reset_index()
    assert len(stats) == 120
    out.mkdir(parents=True, exist_ok=False)
    stats.to_csv(out / "ratio_metrics_summary_pct.csv", index=False)
    rows.to_csv(out / "ratio_metrics_by_seed.csv", index=False)
    provenance = {
        "score": receipt["score"],
        "receipt": receipt,
        "conditions": "Each large panel contains a 3 x 4 grid: 1%, 5%, 10% injection ratio by four fault types.",
        "uncertainty": "Five-seed mean with sample SD; lower whisker stops at zero.",
        "aggregation": "Pool the eligible windows across the same five dates within each seed, ratio, "
                       "mechanism and level; calculate metrics, then summarize seeds. AUC is computed from "
                       "the scores and labels, not averaged from scenario AUC or inferred from the counts.",
    }
    print("Verified 120 conditions and 600 seed-by-ratio rows against the collection receipt.")
    with publication_context(font_family="Times New Roman", base_size=8) as font:
        if grouped_style:
            stem = "EXP3_condition_calibrated_1x_grouped_4x4"
            palette = None
            ratio_colors = BLUE_RATIO_COLORS if grouped_style == "blue_seeds_fixed" else None
            compact_seeds = grouped_style in ("blue_seeds_compact", "blue_seeds_fixed")
            show_seeds = grouped_style == "blue_seeds" or compact_seeds
            if grouped_style == "bivariate":
                stem += "_ratio_severity"
                palette = RATIO_SEVERITY_COLORS
            elif grouped_style == "blue" or show_seeds:
                stem += "_" + grouped_style
                if not ratio_colors:
                    palette = BLUE_RATIO_SEVERITY_COLORS
            if compact_seeds and (auc_min or fpr_max != 3):
                stem += f"_auc{auc_min}_fpr{fpr_max}"
            fig = grouped_plate(stats, palette=palette, ratio_colors=ratio_colors, seed_rows=rows if show_seeds else None,
                                compact_seeds=compact_seeds, auc_min=auc_min, fpr_max=fpr_max)
            color_notes = {}
            if palette:
                color_notes = {"color_encoding": "Hue encodes injection ratio; darker fills encode higher severity.",
                               "palette_by_injection_ratio": palette, "palette_level_order": GROUPED_LEVELS}
            elif ratio_colors:
                color_notes = {"color_encoding": "One fixed blue per injection ratio; severity is encoded by position only.",
                               "palette_by_injection_ratio": ratio_colors, "seed_marker_outline": False}
            if show_seeds:
                color_notes["points"] = "One dot per archived seed result; deterministic horizontal offsets only."
            if compact_seeds:
                color_notes["auc_axis_min_pct"] = auc_min
                color_notes["fpr_axis_max_pct"] = fpr_max
                color_notes["fpr_display"] = (f"Axis fixed at 0 to {fpr_max}%; triangles mark any seed values "
                                               "above the axis limit. Statistics use all seeds.")
            export_figure(fig, out / stem, dpi=600,
                          provenance={**provenance, "font": font, "layout": "4 metric rows x 4 fault columns",
                                      "conditions": "Levels 1, 3, 5, 7, 9; bars ordered 1%, 5%, 10% within each group.",
                                      **color_notes})
            plt.close(fig)
            print("Rendered 16 closed panels with 15 labeled bars per panel.")
            return
        if compact:
            fig = plt.figure(figsize=(180/25.4, 315/25.4), dpi=300)
            plate = fig.add_gridspec(4, 1, left=.108, right=.984, bottom=.062, top=.936, hspace=.17)
            left, title_y, title_size, offset_y, xlabel_y = .108, .988, 10.5, .006, .022
            stem = "EXP3_condition_calibrated_1x_vertical_compact"
        elif vertical_closed:
            fig = plt.figure(figsize=(180/25.4, 425/25.4), dpi=300)
            plate = fig.add_gridspec(4, 1, left=.108, right=.984, bottom=.041, top=.947, hspace=.28)
            left, title_y, title_size, offset_y, xlabel_y = .108, .987, 11.5, .011, .013
            stem = "EXP3_condition_calibrated_1x_vertical_closed"
        else:
            for metric, label in METRICS.items():
                plot_metric(stats, metric, out, export_figure, font, provenance, label=label)
            fig = plt.figure(figsize=(360/25.4, 225/25.4), dpi=300)
            plate = fig.add_gridspec(2, 2, left=.064, right=.984, bottom=.089, top=.899, wspace=.20, hspace=.43)
            left, title_y, title_size, offset_y, xlabel_y = .064, .974, 15, .027, .028
            stem = "EXP3_condition_calibrated_1x_combined"
        fig.text(left, title_y, "Condition-calibrated score | Original injection amplitude (1x)",
                 fontsize=title_size, weight="bold", va="top")
        for n, (metric, label) in enumerate(METRICS.items()):
            cell = plate[n, 0] if vertical_closed else plate[n//2, n%2]
            box = cell.get_position(fig)
            sub = cell.subgridspec(3,4,hspace=.13 if compact else .24,wspace=.17)
            axes = np.empty((3,4), dtype=object)
            for i,j in np.ndindex(axes.shape):
                shared = {} if i == j == 0 else dict(sharex=axes[0,0], sharey=axes[0,0])
                axes[i,j] = fig.add_subplot(sub[i,j], **shared)
            draw_metric_grid(stats, metric, axes, closed=vertical_closed, compact=compact,
                             column_titles=(not compact or n == 0))
            x = box.x0 if vertical_closed else box.x0-.026
            y = box.y1 + (.019 if compact and n == 0 else offset_y)
            fig.text(x, y, f"({chr(97+n)}) {label} (%)", fontsize=10.5 if compact else 11.5, weight="bold")
            if not compact or n == len(METRICS)-1:
                fig.text((box.x0+box.x1)/2, box.y0-xlabel_y, "Injected severity level", fontsize=8.8, ha="center")
        fig.text(left, .012 if vertical_closed else .023,
                 "Rows: injection ratio. Bar labels: five-seed means (%). Whiskers: sample standard deviation.",
                 fontsize=7.8 if vertical_closed else 8.5, color="#4b5763")
        assert len(fig.axes) == 48
        if vertical_closed:
            assert all(spine.get_visible() for ax in fig.axes for spine in ax.spines.values())
        export_figure(fig, out / stem, dpi=600, provenance={**provenance, "font": font,
                      "layout": ("Compact vertical closed" if compact else "4 vertically stacked metric groups, closed axes"
                                 if vertical_closed else "2 x 2")})
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    for mode, keys in (("collect", ("root", "out")), ("plot", ("root", "data", "out"))):
        c = sub.add_parser(mode)
        for key in keys:
            c.add_argument(f"--{key}", required=True, type=Path)
        if mode == "plot":
            c.add_argument("--case-counts", type=Path, help="Optional independent scenario-count audit.")
            layout = c.add_mutually_exclusive_group()
            layout.add_argument("--vertical-closed", action="store_true")
            layout.add_argument("--compact", action="store_true", help="Use a more compact vertical, closed layout.")
            layout.add_argument("--grouped", dest="grouped_style", action="store_const", const="plain",
                                help="Use four metric rows and four fault columns.")
            layout.add_argument("--grouped-bivariate", dest="grouped_style", action="store_const", const="bivariate",
                                help="Use grouped bars colored by ratio and severity.")
            layout.add_argument("--grouped-blue", dest="grouped_style", action="store_const", const="blue",
                                help="Use the 15-shade blue palette for grouped bars.")
            layout.add_argument("--grouped-blue-seeds", dest="grouped_style", action="store_const", const="blue_seeds",
                                help="Use borderless blue bars with individual seed dots and no color key.")
            layout.add_argument("--grouped-blue-seeds-compact", dest="grouped_style", action="store_const",
                                const="blue_seeds_compact", help="Use outlined blue bars, tighter labels, and a 3% FPR limit.")
            layout.add_argument("--grouped-blue-seeds-fixed", dest="grouped_style", action="store_const",
                                const="blue_seeds_fixed", help="Use three fixed ratio colors and unoutlined seed dots.")
            c.add_argument("--auc-min", choices=(0, 50), type=int, default=0)
            c.add_argument("--fpr-max", choices=(3, 4), type=int, default=3)
    a = p.parse_args()
    if a.mode == "collect":
        collect(a.root, a.out)
    else:
        if (a.auc_min or a.fpr_max != 3) and a.grouped_style not in ("blue_seeds_compact", "blue_seeds_fixed"):
            p.error("Axis limits require --grouped-blue-seeds-compact or --grouped-blue-seeds-fixed.")
        plot(a.root, a.data, a.out, a.vertical_closed or a.compact, a.compact, a.grouped_style,
             auc_min=a.auc_min, fpr_max=a.fpr_max, case_counts=a.case_counts)
