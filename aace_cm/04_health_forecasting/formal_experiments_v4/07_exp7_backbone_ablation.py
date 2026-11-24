"""Run single-module ablations of the original Exp7 forecasting backbone."""

from contextlib import contextmanager
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "07_exp7_component_ablation.py"
HORIZONS = (10, 20, 30, 50, 100, 150)
VARIANTS = ("full", "no_gate", "no_lstm", "no_transformer")
SOURCE_HASHES = {
    "07_exp7_component_ablation.py": "c583031d3b9030b68a749c91d6d904fd915548d3c37c83ba8beed737ee7c21e4",
    "07_exp7_sequence_forecasting.py": "bd06123a528f11c0ced2b5de6a8716b4c28b3b48f155c8b317af6672614d372c",
    "07_exp7_dshi_forecast_ratio.py": "c5c0404d9c3cc821d33bad704e65ae687b2b31f53a2b73f99c9530332554b3bb",
}
def supplement_contract():
    """Read fingerprints from a trusted local contract, not the distributed code."""
    path = Path(os.environ.get("AACE_EXP7_PRIVATE_CONTRACT", HERE.parent / "local_inputs" / "exp7_contract.json"))
    if not path.is_file():
        raise FileNotFoundError(f"Supply the separate audited Exp7 input contract: {path}")
    with path.open(encoding="utf-8-sig") as handle:
        contract = json.load(handle)
    if not contract["files"]:
        raise ValueError("The local contract must include metadata and index fingerprints.")
    return contract


class BypassRecurrent(nn.Module):
    def forward(self, sequence):
        return sequence, None


def backbone_variant(original_variant):
    @contextmanager
    def variant_context(legacy, variant):
        if variant in {"full", "no_gate"}:
            with original_variant(legacy, variant):
                yield
            return

        original = legacy.SequenceForecasterNet

        class Forecaster(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if self.model_type != "xLSTM-Tuned" or not isinstance(self.xlstm, nn.LSTM):
                    raise ValueError("Unexpected recurrent backbone")
                if variant == "no_lstm":
                    self.xlstm = BypassRecurrent()
                elif variant == "no_transformer":
                    self.transformer = nn.Identity()
                else:
                    raise ValueError(f"Unknown backbone variant: {variant}")

        legacy.SequenceForecasterNet = Forecaster
        try:
            yield
        finally:
            legacy.SequenceForecasterNet = original

    return variant_context


def check_hash(path, expected):
    with path.open("rb") as file:
        actual = hashlib.file_digest(file, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Source or input hash mismatch: {path}")


def check_bounds(bounds, expected_hash):
    digest = hashlib.sha256()
    for scenario, (length, first, end) in sorted(bounds.items()):
        digest.update(f"{scenario},{length},{first},{end}\n".encode("ascii"))
    if len(bounds) != 5800 or digest.hexdigest() != expected_hash:
        raise ValueError("Fault boundaries differ from the audited supplemental input")


def summarize(output, horizons, folds):
    rows = pd.read_csv(output / "per_fold.csv")
    index = ["Fold", "Horizon", "Variant"]
    expected = {(f, h, v) for f in folds for h in horizons for v in VARIANTS}
    actual = list(rows[index].itertuples(index=False, name=None))
    metrics = ["MAE", "RMSE", "R2"]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete or duplicate result grid")
    if not np.isfinite(rows[metrics].to_numpy()).all():
        raise ValueError("Nonfinite metrics in the result grid")
    if not all(Path(name).is_file() for name in rows["checkpoint"]):
        raise ValueError("Missing fitted checkpoints")
    for _, cell in rows.groupby(["Fold", "Horizon"]):
        if not (cell[["Seed", "Train Dates", "Test Dates", "Train Windows", "Test Windows"]].nunique() == 1).all():
            raise ValueError("Ablations do not share the same windows and seed")
    baseline = rows.loc[rows.Variant.eq("full"), ["Fold", "Horizon", *metrics]]
    paired = rows.merge(baseline, on=["Fold", "Horizon"], suffixes=("", "_full"), validate="many_to_one")
    for m in ("MAE", "RMSE"):
        if not (paired[m + "_full"] > 0).all():
            raise ValueError(f"Nonpositive baseline for {m}")
        paired[m + "_change_pct"] = 100 * (paired[m] / paired[m + "_full"] - 1)
    paired["R2_change"] = paired["R2"] - paired["R2_full"]
    changes = ["MAE_change_pct", "RMSE_change_pct", "R2_change"]
    fold_means = paired.groupby(["Variant", "Fold"])[changes].mean()
    fold_means.to_csv(output / "overall_per_fold.csv")
    overall = fold_means.groupby("Variant")[changes].agg(["mean", "std"])
    overall.columns = ["_".join(column) for column in overall.columns]
    overall.reindex(VARIANTS).to_csv(output / "overall_summary.csv")
    by_horizon = rows.groupby(["Variant", "Horizon"])[metrics].agg(["mean", "std"])
    by_horizon.columns = ["_".join(column) for column in by_horizon.columns]
    by_horizon.to_csv(output / "horizon_mean_sd.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "audit", "formal"), required=True)
    parser.add_argument("--data-protocol", choices=("historical", "supplement"), required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--fold-indices", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-memory-cap", type=float)
    args = parser.parse_args()
    if args.cuda_memory_cap is not None:
        cap = args.cuda_memory_cap
        if not 0 < cap < 1 or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            parser.error("The CUDA cap requires exactly one visible GPU and a fraction in (0, 1)")
        free, _ = torch.cuda.mem_get_info(0)
        if free < 12 * 1024**3:
            parser.error(f"Expected 12 GiB free on the selected GPU, got {free / 1024**3:.2f} GiB")
        torch.cuda.set_per_process_memory_fraction(cap, 0)

    for name, digest in SOURCE_HASHES.items():
        check_hash(HERE / name, digest)
    original_main = runpy.run_path(str(RUNNER))["main"]
    scope = original_main.__globals__
    original_load = scope["load_locked_inputs"]

    if args.data_protocol == "supplement":
        contract = supplement_contract()
        for name, digest in contract["files"].items():
            check_hash(scope["OUTPUTS"] / name, digest)
        tag = "sequence_forecasting_v4_casebank_20260713"
        check_hash(scope["OUTPUTS"] / f"{tag}_dshi_cache_BA1.npz", contract["cache_sha256"])

    def load_checked(*load_args, **kwargs):
        inputs = original_load(*load_args, **kwargs)
        if args.data_protocol == "supplement" and args.mode != "smoke":
            check_bounds(inputs[-1], contract["bounds_sha256"])
            print("Locked supplemental cache and 5,800 label boundaries verified.", flush=True)
        return inputs

    scope.update(HORIZONS=HORIZONS, VARIANTS=VARIANTS,
                 model_variant=backbone_variant(scope["model_variant"]), load_locked_inputs=load_checked)
    launcher_args = sys.argv
    # Forward only flags supported by the frozen runner.
    sys.argv = [launcher_args[0], "--mode", args.mode, "--data-protocol", args.data_protocol,
                "--output-dir", str(args.output_dir), "--horizons", *map(str, args.horizons),
                "--fold-indices", *map(str, args.fold_indices)]
    try:
        original_main()
    finally:
        sys.argv = launcher_args
    manifest = args.output_dir / "run.json"
    info = json.loads(manifest.read_text(encoding="utf-8"))
    info["command"] = [sys.executable, *launcher_args]
    info["source_sha256"][str(Path(__file__).resolve())] = scope["file_hash"](Path(__file__))
    info["runtime"] = {"torch_cuda": torch.version.cuda, "numpy": np.__version__, "pandas": pd.__version__}
    info["ablations"] = {
        "full": "unchanged xLSTM-Tuned architecture; two standard PyTorch LSTM layers",
        "no_gate": "independent memory gate replaced with ones; native LSTM gates unchanged",
        "no_lstm": "projected sequence passed directly to the independent gate",
        "no_transformer": "gated recurrent sequence passed directly to the last-position readout",
    }
    manifest.write_text(json.dumps(info, indent=2), encoding="utf-8")
    if args.mode == "formal":
        summarize(args.output_dir, args.horizons, args.fold_indices)


if __name__ == "__main__":
    main()
