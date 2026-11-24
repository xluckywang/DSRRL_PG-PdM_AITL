#!/usr/bin/env python
"""Build a current-weight Exp7 DSHI cache for the formal Exp9/Exp10 scenarios."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent


def _load_module(name: str, filename: str):
    path = THIS_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="exp10_v4_formal_dshi")
    parser.add_argument("--device", default="BA1")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    exp9 = _load_module("exp9_scenario_selector", "09_exp9_qrdqn_v6_model_comparison.py")
    # py312 currently combines pandas with a pyarrow build that may attempt to
    # unregister the same extension twice when Exp7 and Exp9 share a process.
    exp9._patch_pyarrow_for_pandas()
    exp7 = _load_module("exp7_dshi_cache_builder", "07_exp7_sequence_forecasting.py")

    selected_names = {
        name for name, _, _ in exp9._select_balanced_params(fast=args.fast, device=args.device)
    }
    all_scenarios = exp7.list_fault_scenarios(args.device)
    scenarios = [scenario for scenario in all_scenarios if scenario.name in selected_names]
    found = {scenario.name for scenario in scenarios}
    missing = sorted(selected_names - found)
    if missing:
        raise RuntimeError(
            f"Cannot build DSHI cache: found {len(found)}/{len(selected_names)} selected scenarios; "
            f"first missing: {missing[:5]}"
        )

    print(
        f"[DSHI cache] tag={args.tag} device={args.device} scenarios={len(scenarios)} "
        f"mode={'fast' if args.fast else 'formal'}",
        flush=True,
    )
    trajectories = exp7.compute_or_load_dshi(
        scenarios,
        args.device,
        tag=args.tag,
        refresh=args.refresh,
    )
    if set(trajectories) != selected_names:
        missing = sorted(selected_names - set(trajectories))
        raise RuntimeError(f"DSHI cache verification failed; first missing: {missing[:5]}")
    print(f"[DSHI cache] verified coverage={len(trajectories)}/{len(selected_names)}", flush=True)


if __name__ == "__main__":
    main()
