"""Public, data-free configuration for the EXP3 reproduction code."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


PACKAGE_ROOT = Path(os.environ.get("EXP3_ROOT", Path(__file__).resolve().parents[1])).resolve()
SPLIT_FILE = Path(os.environ.get("EXP3_SPLITS", PACKAGE_ROOT / "config" / "splits.json"))


@dataclass(frozen=True)
class Paths:
    root_dir: Path = PACKAGE_ROOT
    data_dir: Path = root_dir / "data"
    monitoring_dir: Path = data_dir / "monitoring"
    labels_dir: Path = data_dir / "labels"
    scenarios_dir: Path = data_dir / "scenarios"
    outputs_dir: Path = root_dir / "outputs"
    exp3_grid_dir: Path = outputs_dir / "exp3_grid_search"

    def ensure_dirs(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.exp3_grid_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CommonConfig:
    device: str = "line_1"
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    test_ratio: float = 0.5


@dataclass(frozen=True)
class Exp3Config:
    ae_structures_extended: tuple[str, ...] = (
        "SimpleAE", "FlexibleAE", "DeepAE", "Conv1DAE", "VAE", "MemAE"
    )
    distance_metrics: tuple[str, ...] = ("euclidean", "mahalanobis")
    baseline_methods: tuple[str, ...] = (
        "KMeans_K3", "KMeans_K5", "GMM_K3", "GMM_K5", "DBSCAN", "LOF"
    )
    window_radii: tuple[int, ...] = (0, 10, 20, 30)
    epochs: tuple[int, ...] = (50, 100, 200)
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    conv1d_window_size: int = 7
    conv1d_channels: tuple[int, ...] = (32, 64, 128)
    vae_beta: float = 1.0
    memae_memory_size: int = 128
    dbscan_eps: float = 0.30
    dbscan_min_samples: int = 5
    dbscan_max_points: int = 20_000
    normal_state_id: int = 2
    early_stopping_patience: int = 15
    early_stopping_threshold: float = 1e-4
    fast_num_days: int = 3
    fast_window_radii: tuple[int, ...] = (0,)
    fast_epochs: tuple[int, ...] = (50,)
    fast_ae_structures: tuple[str, ...] = ("FlexibleAE", "Conv1DAE")
    fast_baseline_methods: tuple[str, ...] = ("KMeans_K3", "LOF")
    fast_distance_metrics: tuple[str, ...] = ("euclidean", "mahalanobis")
    fast_seeds: tuple[int, ...] = (42,)
    full_evaluation: bool = False
    eval_severity_levels: tuple[int, ...] = (1, 3, 6, 9)
    eval_pct_values: tuple[float, ...] = (0.01, 0.05, 0.10)
    max_scenarios_per_combination: int = 1
    scenario_sampling_seed: int = 20260628
    max_train_healthy: int = 200_000
    max_scorable_points: int = 10_000


PATHS = Paths()
COMMON = CommonConfig()
EXP3 = Exp3Config()


def _splits() -> dict:
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Missing split manifest: {SPLIT_FILE}. Copy config/splits.example.json "
            "to config/splits.json and replace the aliases with local record IDs."
        )
    splits = json.loads(SPLIT_FILE.read_text(encoding="utf-8-sig"))
    for role in ("train", "development", "test"):
        ids = splits[role]
        if not isinstance(ids, list) or any(
            not isinstance(record_id, str) or not re.fullmatch("[A-Za-z0-9_-]+", record_id)
            for record_id in ids
        ):
            raise ValueError(f"Invalid record IDs in split {role}")
    return splits


def get_train_dates() -> List[str]:
    return list(_splits()["train"])


def get_development_dates() -> List[str]:
    return list(_splits()["development"])


def get_test_dates() -> List[str]:
    return list(_splits()["test"])


def get_exp0_recommended_eval_dates() -> List[str]:
    return get_development_dates() + get_test_dates()


def get_exp2_generation_dates() -> List[str]:
    return get_exp0_recommended_eval_dates()


def train_dates_set() -> set[str]:
    return set(get_train_dates())


def get_feature_cols(device: str | None = None) -> List[str]:
    return ["X", "Y", "Z", "T"]


def get_feature_data_file(device: str, record_id: str) -> Path:
    return PATHS.monitoring_dir / device / f"{record_id}_preprocessed.csv"


def get_label_file(device: str, record_id: str) -> Path:
    return PATHS.labels_dir / device / f"{record_id}_labels.csv"


def get_active_fault_data_dir() -> Path:
    return PATHS.scenarios_dir
