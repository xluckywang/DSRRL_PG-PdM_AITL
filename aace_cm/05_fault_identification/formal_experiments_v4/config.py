"""Data paths for Exp8. All inputs are supplied separately from this package."""

import csv
import os
from pathlib import Path
from types import SimpleNamespace


INPUT_ROOT = Path(
    os.environ.get("EXP8_INPUT_ROOT", Path(__file__).resolve().parents[1] / "inputs")
).expanduser()
COMMON = SimpleNamespace(device="BA1")
EXP3 = SimpleNamespace(normal_state_id=2)


def get_feature_cols(device):
    return [f"{device}_{axis}" for axis in ("X", "Y", "Z", "T")]


def get_feature_data_file(device, date):
    return INPUT_ROOT / "healthy" / f"{date}_preprocessed.csv"


def get_label_file(device, date):
    return INPUT_ROOT / "operating_states" / device / f"{date}_labels.csv"


def get_active_fault_data_dir():
    return INPUT_ROOT / "fault_scenarios"


def get_exp0_recommended_train_dates():
    cohort = INPUT_ROOT / "recommended_dates.csv"
    with cohort.open(encoding="utf-8-sig", newline="") as file:
        dates = [
            row["date"].strip()
            for row in csv.DictReader(file)
            if (row.get("date") or "").strip()
            and (row.get("recommended_for_both_devices") or "true").strip().lower()
            in {"true", "1", "yes", "y"}
        ]
    return dates[: len(dates) // 2]
