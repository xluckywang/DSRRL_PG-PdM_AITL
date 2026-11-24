#!/usr/bin/env python
"""Build the formal AACE-CM Exp8 method-comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METHOD_NAMES = {
    "ae_residual_only": "Healthy-AE residual clustering",
    "paired_contrast_trajectory": "Paired contrast + trajectory",
    "paired_contrast_temperature_coupling": (
        "Paired contrast + temperature/coupling"
    ),
    "paired_contrast_full": "Paired contrast + full descriptors",
    "paired_contrast_plus_ae": "Paired contrast + AE residual",
    "paired_contrast_only": "Proposed-TFRC",
}
METHOD_ORDER = list(METHOD_NAMES)
METRICS = ["Accuracy", "Balanced Accuracy", "Macro-F1", "MCC"]


def markdown_table(frame: pd.DataFrame) -> str:
    best = {metric: frame[metric].max() for metric in METRICS}
    lines = [
        "| Method | Accuracy | Balanced Accuracy | Macro-F1 | MCC |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        values = []
        for metric, value in zip(METRICS, row[1:]):
            rendered = f"{value:.4f}"
            if abs(value - best[metric]) < 1e-12:
                rendered = f"**{rendered}**"
            values.append(rendered)
        method = f"**{row.Method}**" if row.Method == "Proposed-TFRC" else row.Method
        lines.append(f"| {method} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def latex_table(frame: pd.DataFrame) -> str:
    best = {metric: frame[metric].max() for metric in METRICS}
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Accuracy & Balanced Accuracy & Macro-F1 & MCC \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        values = []
        for metric, value in zip(METRICS, row[1:]):
            rendered = f"{value:.4f}"
            if abs(value - best[metric]) < 1e-12:
                rendered = rf"\textbf{{{rendered}}}"
            values.append(rendered)
        method = (
            r"\textbf{Proposed-TFRC}"
            if row.Method == "Proposed-TFRC"
            else row.Method.replace("+", r"$+$")
        )
        lines.append(f"{method} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = pd.read_csv(args.summary).set_index("representation")
    missing = set(METHOD_ORDER) - set(source.index)
    if missing:
        raise ValueError(f"Missing formal comparison variants: {sorted(missing)}")

    rows = []
    for representation in METHOD_ORDER:
        row = source.loc[representation]
        rows.append(
            {
                "Method": METHOD_NAMES[representation],
                "Accuracy": float(row["one_record_accuracy"]),
                "Balanced Accuracy": float(
                    row["one_record_balanced_accuracy"]
                ),
                "Macro-F1": float(row["one_record_macro_f1"]),
                "MCC": float(row["one_record_mcc"]),
            }
        )
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "aace_exp8_method_comparison.csv", index=False)
    (args.output_dir / "aace_exp8_method_comparison.md").write_text(
        markdown_table(frame), encoding="utf-8"
    )
    (args.output_dir / "aace_exp8_method_comparison.tex").write_text(
        latex_table(frame), encoding="utf-8"
    )
    print(markdown_table(frame))


if __name__ == "__main__":
    main()
