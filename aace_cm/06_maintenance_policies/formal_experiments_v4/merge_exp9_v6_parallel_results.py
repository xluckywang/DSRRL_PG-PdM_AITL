from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
OUTPUTS = THIS_DIR / "outputs"
PREFIX = "exp9_qrdqn_v6_model_comparison_shard_"
OUT_DIR = OUTPUTS / "exp9_qrdqn_v6_model_comparison_parallel_merged"

RULE_METHODS = {
    "Run-to-failure",
    "Periodic PM",
    "Fixed threshold 0.7",
    "Fixed threshold 1.0",
    "Greedy risk policy",
    "DMT heuristic",
}

ORDER = [
    "Run-to-failure",
    "DQN",
    "Double-DQN",
    "Dueling-DQN",
    "Double-Dueling-DQN",
    "SCORER-DQN",
    "LSTM-DQN",
    "HiLSTM-Attn-DQN",
    "QR-DQN",
    "LSTM-QR-DQN",
    "HiLSTM-Attn-QR-DQN",
    "PPO",
    "SAC",
    "Periodic PM",
    "Fixed threshold 0.7",
    "Fixed threshold 1.0",
    "Greedy risk policy",
    "DMT heuristic",
]


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    columns = list(df.columns)
    str_df = df.fillna("").astype(str)
    widths = {
        col: max(len(str(col)), *(len(v) for v in str_df[col].tolist()))
        for col in columns
    }

    def row(values: list[str]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[col]) for col, v in zip(columns, values)) + " |"

    with open(path, "w", encoding="utf-8") as f:
        f.write(row([str(c) for c in columns]))
        f.write("\n")
        f.write("| " + " | ".join("-" * widths[col] for col in columns) + " |")
        f.write("\n")
        for _, record in str_df.iterrows():
            f.write(row([record[col] for col in columns]))
            f.write("\n")
        f.write("\n")


def main() -> None:
    shard_dirs = sorted(p for p in OUTPUTS.glob(f"{PREFIX}*") if p.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"No shard output directories found under {OUTPUTS} with prefix {PREFIX}")

    rows = []
    metadata = {}
    for shard_dir in shard_dirs:
        table_path = shard_dir / "exp9_v6_model_comparison_table.csv"
        meta_path = shard_dir / "exp9_v6_metadata.json"
        if not table_path.exists():
            raise FileNotFoundError(f"Missing shard table: {table_path}")
        table = pd.read_csv(table_path)
        table.insert(0, "Shard", shard_dir.name.replace("exp9_qrdqn_v6_model_comparison_", ""))
        rows.append(table)
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                metadata[shard_dir.name] = json.load(f)

    all_rows = pd.concat(rows, ignore_index=True)
    model_rows = all_rows[~all_rows["Method"].isin(RULE_METHODS)].copy()
    rule_rows = all_rows[all_rows["Method"].isin(RULE_METHODS)].drop_duplicates("Method", keep="first").copy()
    merged = pd.concat([rule_rows, model_rows], ignore_index=True)
    order_map = {name: i for i, name in enumerate(ORDER)}
    merged["_order"] = merged["Method"].map(order_map).fillna(999)
    merged = merged.sort_values(["_order", "Method"]).drop(columns=["_order"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_DIR / "exp9_v6_parallel_merged_table.csv", index=False, encoding="utf-8-sig")
    _write_markdown_table(merged, OUT_DIR / "exp9_v6_parallel_merged_table.md")
    with open(OUT_DIR / "exp9_v6_parallel_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "shard_dirs": [str(p) for p in shard_dirs],
                "merged_rows": int(len(merged)),
                "metadata_by_shard": metadata,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Merged {len(shard_dirs)} shards -> {OUT_DIR}")


if __name__ == "__main__":
    main()
