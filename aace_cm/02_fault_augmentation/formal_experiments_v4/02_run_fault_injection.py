"""Exp2 unified fault-injection runner.

This is now the single entry point for Exp2 fault injection. It can generate:

1. vibration-channel fault data via aace_cm/fault_injector.py
2. temperature-channel fault data via 02_run_temperature_fault_injection.py

By default it uses the second half of the Exp0-audited date set, i.e. the
last 29 days of the 58-day cohort, for fault injection/evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
VW_DIR = ROOT_DIR / "aace_cm"
DATA_DIR = ROOT_DIR / "DATA" / "Data preprocessing" / "AACE-CM" / "Per_second"
EXP0_RECOMMENDED_DATES = (
    THIS_DIR / "outputs" / "exp0_data_audit" / "recommended_dates.csv"
)
DEFAULT_TEMP_OUT_DIR = VW_DIR / "04-temp_fault_data"

sys.path.insert(0, str(VW_DIR))
sys.path.insert(0, str(THIS_DIR))

from config import COMMON, PATHS, get_exp0_recommended_eval_dates, get_frozen_all_dates, get_frozen_eval_dates  # noqa: E402
from fault_injector import SyntheticFaultDataset  # noqa: E402
from aace_config import FaultInjectionConfig, SensorConfig  # noqa: E402


FAULT_TYPES = list(FaultInjectionConfig.FAULT_TYPES)
GRID_PCTS = list(FaultInjectionConfig.GRID_PERCENTAGES)
GRID_SEVS = list(FaultInjectionConfig.GRID_SEVERITY_LEVELS)
DEFAULT_DEVICES = (COMMON.device,)


def _parse_csv_floats(value: str | None, default: Sequence[float]) -> list[float]:
    if not value:
        return list(default)
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_ints(value: str | None, default: Sequence[int]) -> list[int]:
    if not value:
        return list(default)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_strings(value: str | None, default: Sequence[str]) -> list[str]:
    if not value:
        return list(default)
    return [x.strip() for x in value.split(",") if x.strip()]


def _filter_eligible_dates(min_rows: int) -> tuple[list[str], list[tuple[str, int]]]:
    eligible: list[str] = []
    skipped: list[tuple[str, int]] = []
    for path in sorted(DATA_DIR.glob("*_preprocessed.csv")):
        with path.open(encoding="utf-8", errors="ignore") as fh:
            n_rows = sum(1 for _ in fh) - 1
        date = path.name.replace("_preprocessed.csv", "")
        if n_rows >= min_rows:
            eligible.append(date)
        else:
            skipped.append((date, n_rows))
    return eligible, skipped


def _load_exp0_recommended_dates(path: Path = EXP0_RECOMMENDED_DATES) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Exp0 recommended date file not found: {path}")
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"Exp0 recommended date file must contain a 'date' column: {path}")
    return df["date"].astype(str).tolist()


def _exp0_recommended_eval_dates() -> list[str]:
    dates = get_exp0_recommended_eval_dates()
    if dates:
        return dates
    all_dates = _load_exp0_recommended_dates()
    return all_dates[len(all_dates) // 2:]


def _select_dates(args: argparse.Namespace, eligible: list[str]) -> list[str]:
    eligible_set = set(eligible)
    if args.dates:
        requested = _parse_csv_strings(args.dates, [])
        dates = [d for d in requested if d in eligible_set]
        missing = sorted(set(requested) - set(dates))
        if missing:
            print(f"  requested dates not eligible: {missing}")
    elif args.date_mode == "exp0_recommended_eval":
        dates = [d for d in _exp0_recommended_eval_dates() if d in eligible_set]
    elif args.date_mode == "exp0_recommended":
        dates = [d for d in _load_exp0_recommended_dates() if d in eligible_set]
    elif args.date_mode == "frozen_eval":
        frozen = set(get_frozen_eval_dates())
        dates = [d for d in eligible if d in frozen]
    elif args.date_mode == "frozen_all":
        frozen = set(get_frozen_all_dates())
        dates = [d for d in eligible if d in frozen]
    elif args.date_mode == "all":
        dates = list(eligible)
    elif args.date_mode == "latest":
        n = args.num_days if args.num_days > 0 else len(eligible)
        dates = list(eligible[-n:])
    elif args.date_mode == "first":
        n = args.num_days if args.num_days > 0 else len(eligible)
        dates = list(eligible[:n])
    else:
        raise ValueError(f"Unknown date mode: {args.date_mode}")

    if args.limit_days:
        dates = dates[: args.limit_days]
    return dates


def _inject_vibration_for_device(
    device: str,
    dates: list[str],
    out_dir: Path,
    fault_types: list[str],
    pcts: list[float],
    severities: list[int],
    dry: bool,
) -> int:
    dev_out = out_dir / device
    dev_out.mkdir(parents=True, exist_ok=True)
    generator = SyntheticFaultDataset(str(dev_out))
    count = 0

    for date in dates:
        source_path = DATA_DIR / f"{date}_preprocessed.csv"
        labels_path = PATHS.vw_labels_dir / device / f"{date}_labels.csv"
        if not source_path.exists() or not labels_path.exists():
            print(f"  [skip] {device} {date}: missing source or labels")
            continue
        scenario_count = len(fault_types) * len(pcts) * len(severities)
        if dry:
            print(f"  [plan:vibration] {device} {date}: {scenario_count} scenarios")
        else:
            print(f"  [vibration] {device} {date} ...", flush=True)
            generator.generate_grid(
                str(source_path),
                str(labels_path),
                bearing=device,
                percentages=pcts,
                severity_levels=severities,
                fault_types=fault_types,
            )
        count += scenario_count
    return count


def _load_temperature_module():
    module_path = THIS_DIR / "02_run_temperature_fault_injection.py"
    spec = importlib.util.spec_from_file_location("exp2_temperature_fault_injection", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load temperature injection module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_temperature_injection(
    args: argparse.Namespace,
    dates: list[str],
    devices: list[str],
    fault_types: list[str],
    pcts: list[float],
    severities: list[int],
) -> None:
    module = _load_temperature_module()
    temp_args = SimpleNamespace(
        dry=args.dry,
        overwrite=args.overwrite,
        out_dir=str(Path(args.temperature_out_dir)),
        vibration_base_dir=str(Path(args.vibration_out_dir)),
        devices=",".join(devices),
        fault_types=",".join(fault_types),
        pcts=",".join(str(x) for x in pcts),
        severities=",".join(str(x) for x in severities),
        dates=",".join(dates),
        date_mode=args.date_mode,
        num_days=0,
        limit_days=0,
        min_rows=args.min_rows,
        clean_output=False,
    )
    module.run(temp_args)


def _clean_output_tree(out_dir: Path, devices: Sequence[str]) -> None:
    """Remove old Exp2 scenario files for selected devices to avoid stale-date contamination."""
    patterns = ("*_data.parquet", "*_labels.parquet", "*_params.json")
    removed = 0
    for device in devices:
        device_dir = out_dir / device
        if not device_dir.exists():
            continue
        for pattern in patterns:
            for path in device_dir.glob(pattern):
                path.unlink()
                removed += 1
    print(f"  cleaned old scenario files under {out_dir}: {removed}")


def main(args: argparse.Namespace) -> None:
    devices = _parse_csv_strings(args.devices, DEFAULT_DEVICES)
    fault_types = _parse_csv_strings(args.fault_types, FAULT_TYPES)
    pcts = _parse_csv_floats(args.pcts, GRID_PCTS)
    severities = _parse_csv_ints(args.severities, GRID_SEVS)

    eligible, skipped = _filter_eligible_dates(args.min_rows)
    dates = _select_dates(args, eligible)
    if not dates:
        print("[error] No dates selected for injection.")
        return

    vibration_out = Path(args.vibration_out_dir)
    temperature_out = Path(args.temperature_out_dir)
    per_day = len(fault_types) * len(pcts) * len(severities)
    modalities = ["vibration", "temperature"] if args.modalities == "both" else [args.modalities]

    print("=" * 78)
    print("Exp2 unified fault injection")
    print(f"  source: {DATA_DIR}")
    print(f"  labels: {PATHS.vw_labels_dir}")
    print(f"  date_mode: {args.date_mode}")
    print(f"  dates: {len(dates)} ({dates[0]} .. {dates[-1]})")
    print(f"  devices: {devices}")
    print(f"  modalities: {modalities}")
    print(f"  grid: {len(fault_types)} fault types x {len(pcts)} pct x {len(severities)} severity = {per_day}/device/day")
    print(f"  vibration output: {vibration_out}")
    print(f"  temperature output: {temperature_out}")
    if skipped:
        print(f"  skipped by min_rows: {len(skipped)}")
    print("=" * 78)

    total_planned = len(devices) * len(dates) * per_day * len(modalities)
    print(f"  total planned scenarios across selected modalities: {total_planned}")

    if args.clean_output and not args.dry:
        if "vibration" in modalities:
            _clean_output_tree(vibration_out, devices)
        if "temperature" in modalities:
            _clean_output_tree(temperature_out, devices)

    if "vibration" in modalities:
        vibration_out.mkdir(parents=True, exist_ok=True)
        vib_total = 0
        for device in devices:
            vib_total += _inject_vibration_for_device(
                device=device,
                dates=dates,
                out_dir=vibration_out,
                fault_types=fault_types,
                pcts=pcts,
                severities=severities,
                dry=args.dry,
            )
        print(f"[vibration] planned/generated scenarios: {vib_total}")

    if "temperature" in modalities:
        _run_temperature_injection(args, dates, devices, fault_types, pcts, severities)

    if not args.dry:
        print("\nDone.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp2 unified vibration + temperature fault injection")
    parser.add_argument("--dry", action="store_true", help="Show generation plan only")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite temperature scenario files")
    parser.add_argument("--clean-output", action="store_true", help="Delete existing scenario files for selected devices before generation")
    parser.add_argument(
        "--modalities",
        default="both",
        choices=["both", "vibration", "temperature"],
        help="Which Exp2 modalities to generate",
    )
    parser.add_argument("--devices", default=",".join(DEFAULT_DEVICES), help="Comma-separated devices")
    parser.add_argument("--fault-types", default=",".join(FAULT_TYPES), help="Comma-separated fault types")
    parser.add_argument("--pcts", default=None, help="Comma-separated injection percentages")
    parser.add_argument("--severities", default=None, help="Comma-separated severity levels")
    parser.add_argument("--dates", default=None, help="Comma-separated dates; overrides --date-mode")
    parser.add_argument(
        "--date-mode",
        default="exp0_recommended_eval",
        choices=["exp0_recommended_eval", "exp0_recommended", "frozen_eval", "frozen_all", "all", "latest", "first"],
        help="Date selection strategy when --dates is not provided",
    )
    parser.add_argument("--num-days", type=int, default=0, help="Number of days for latest/first modes")
    parser.add_argument("--limit-days", type=int, default=0, help="Optional cap on selected dates")
    parser.add_argument("--min-rows", type=int, default=SensorConfig.MIN_ROWS, help="Minimum rows per source day")
    parser.add_argument("--vibration-out-dir", default=str(PATHS.exp2_fault_data_dir), help="Vibration fault-data output directory")
    parser.add_argument("--temperature-out-dir", default=str(DEFAULT_TEMP_OUT_DIR), help="Temperature fault-data output directory")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
