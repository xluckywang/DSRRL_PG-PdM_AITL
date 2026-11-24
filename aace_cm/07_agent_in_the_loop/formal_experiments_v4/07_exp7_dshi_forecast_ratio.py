"""
AACE-CM-Exp7: DSHI sequence forecasting with scarce-fault ratio ablation.

This experiment follows the evaluation spirit of Wang et al. (2026)
SyntheITS: compare multiple predictors under different proportions of
fault/degradation samples added to a base training set.

Task definition
---------------
Given a historical Dual-Space Health Index (DSHI) window, predict the
future DSHI sequence. This is a numeric reconstruction/regression task,
not RUL estimation and not event classification.

Training-ratio setting
----------------------
P1-P5 correspond to adding 0%, 25%, 50%, 75%, and 100% of available
fault/degradation windows into a healthy-window training pool.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from config import COMMON, EXP7, PATHS, get_active_fault_data_dir, get_feature_cols, load_exp3_best_config
import shared_ae_cache as sac


RATIO_LEVELS = (0.0, 0.25, 0.50, 0.75, 1.00)
RATIO_LABELS = {
    0.0: "P1-0%",
    0.25: "P2-25%",
    0.50: "P3-50%",
    0.75: "P4-75%",
    1.00: "P5-100%",
}

METHOD_ORDER = [
    "Persistence",
    "Linear trend",
    "Ridge",
    "HistGBDT",
    "Random forest",
    "MLP",
    "LSTM",
    "GRU",
    "CNN-LSTM",
    "TCN",
    "Transformer Encoder",
    "Transformer-LSTM",
    "xLSTM-Transformer",
    "xLSTM-Tuned",
]

METHOD_COLORS = {
    "Persistence": "#8C8C8C",
    "Linear trend": "#54A24B",
    "Ridge": "#4C78A8",
    "HistGBDT": "#F58518",
    "Random forest": "#E45756",
    "MLP": "#B279A2",
    "LSTM": "#7F3C8D",
    "GRU": "#11A579",
    "CNN-LSTM": "#3969AC",
    "TCN": "#F2B701",
    "Transformer Encoder": "#6B6ECF",
    "Transformer-LSTM": "#E73F74",
    "xLSTM-Transformer": "#17BECF",
    "xLSTM-Tuned": "#0097A7",
}


@dataclass
class Scenario:
    name: str
    date: str
    fault_type: str
    pct: float
    severity: int
    x: np.ndarray
    labels: np.ndarray


@dataclass
class WindowSet:
    hist: np.ndarray
    target: np.ndarray
    features: np.ndarray
    is_fault_window: np.ndarray
    scenario: np.ndarray
    date: np.ndarray
    fault_type: np.ndarray
    severity: np.ndarray
    pct: np.ndarray
    start: np.ndarray


def _vib_cols(device: str) -> list[str]:
    return get_feature_cols(device)


def _read_fault_data(data_path: Path, lab_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return pd.read_parquet(str(data_path)), pd.read_parquet(str(lab_path))
    except Exception:
        data_csv = data_path.with_suffix(".csv")
        lab_csv = lab_path.with_suffix(".csv")
        if data_csv.exists() and lab_csv.exists():
            return pd.read_csv(str(data_csv)), pd.read_csv(str(lab_csv))
        raise


def load_fault_scenarios(device: str, max_per_type: int | None = None) -> list[Scenario]:
    device_dir = get_active_fault_data_dir() / device
    if not device_dir.exists():
        raise RuntimeError(f"Fault scenario directory not found: {device_dir}")

    by_type: dict[str, list[str]] = {}
    for pf in sorted(device_dir.glob("*_params.json")):
        base = pf.name.replace("_params.json", "")
        m = re.match(r"^(\d{4}-\d{2}-\d{2})_(.+?)_(pct\d+)_(sev\d+)$", base)
        if not m:
            continue
        by_type.setdefault(m.group(2), []).append(base)

    scenarios: list[Scenario] = []
    vib = _vib_cols(device)
    for fault_type, names in sorted(by_type.items()):
        chosen = names if max_per_type is None else names[:max_per_type]
        for base in chosen:
            m = re.match(r"^(\d{4}-\d{2}-\d{2})_(.+?)_(pct\d+)_(sev\d+)$", base)
            if not m:
                continue
            date, ftype, pct_str, sev_str = m.groups()
            pct = int(pct_str.replace("pct", "")) / 100.0
            severity = int(sev_str.replace("sev", ""))

            data_path = device_dir / f"{base}_data.parquet"
            lab_path = device_dir / f"{base}_labels.parquet"
            if not data_path.exists() or not lab_path.exists():
                continue
            df, lf = _read_fault_data(data_path, lab_path)
            n = min(len(df), len(lf))
            assert all(c in df.columns for c in vib), (
                f"Missing fault feature columns in {base}: "
                f"{[c for c in vib if c not in df.columns]}"
            )
            scenarios.append(
                Scenario(
                    name=base,
                    date=date,
                    fault_type=ftype,
                    pct=pct,
                    severity=severity,
                    x=df[vib].values[:n].astype(np.float32),
                    labels=lf["fault_label"].values[:n].astype(int),
                )
            )
    return scenarios


def compute_scenario_dshi(scenarios: list[Scenario], device: str) -> dict[str, np.ndarray]:
    ae, scaler, pool_latent, pool_residual, info = sac.load_best_vae()
    ref_stats = sac.get_health_reference_stats(
        ae, scaler, pool_latent, pool_residual,
        device=device,
        window_w=int(info.get("w", 0)),
    )
    out: dict[str, np.ndarray] = {}
    for sc in scenarios:
        dshi = sac.compute_dual_space_health_index(
            sc.x,
            ae,
            scaler,
            pool_latent,
            pool_residual,
            x_is_normalized=False,
            ref_stats=ref_stats,
            device=device,
            window_w=int(info.get("w", 0)),
        )
        out[sc.name] = np.nan_to_num(dshi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out


def _feature_from_history(hist: np.ndarray) -> np.ndarray:
    h = hist.astype(np.float32)
    t = np.arange(len(h), dtype=np.float32)
    slope = float(np.polyfit(t, h, deg=1)[0]) if len(h) >= 2 else 0.0
    accel = float(np.polyfit(t, h, deg=2)[0]) if len(h) >= 3 else 0.0
    diffs = np.diff(h)
    stats = np.array(
        [
            h[-1],
            h.mean(),
            h.std(),
            h.min(),
            h.max(),
            h.max() - h.min(),
            slope,
            accel,
            diffs[-1] if len(diffs) else 0.0,
            diffs.mean() if len(diffs) else 0.0,
            np.maximum(diffs, 0).mean() if len(diffs) else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([h, stats]).astype(np.float32)


def build_windows(
    scenarios: list[Scenario],
    dshi_by_scenario: dict[str, np.ndarray],
    lookback: int,
    horizon: int,
    slide_step: int,
    smooth_window: int = 1,
) -> WindowSet:
    hist_rows = []
    target_rows = []
    feature_rows = []
    fault_flags = []
    scenario_names = []
    dates = []
    fault_types = []
    severities = []
    pcts = []
    starts = []

    for sc in scenarios:
        dshi = smooth_series(dshi_by_scenario[sc.name], smooth_window)
        labels = sc.labels
        max_start = len(dshi) - lookback - horizon
        if max_start <= 0:
            continue
        for start in range(0, max_start + 1, slide_step):
            h = dshi[start:start + lookback]
            y = dshi[start + lookback:start + lookback + horizon]
            future_labels = labels[start + lookback:start + lookback + horizon]
            hist_rows.append(h)
            target_rows.append(y)
            feature_rows.append(_feature_from_history(h))
            fault_flags.append(int(future_labels.max() > 0))
            scenario_names.append(sc.name)
            dates.append(sc.date)
            fault_types.append(sc.fault_type)
            severities.append(sc.severity)
            pcts.append(sc.pct)
            starts.append(start)

    return WindowSet(
        hist=np.asarray(hist_rows, dtype=np.float32),
        target=np.asarray(target_rows, dtype=np.float32),
        features=np.asarray(feature_rows, dtype=np.float32),
        is_fault_window=np.asarray(fault_flags, dtype=bool),
        scenario=np.asarray(scenario_names, dtype=object),
        date=np.asarray(dates, dtype=object),
        fault_type=np.asarray(fault_types, dtype=object),
        severity=np.asarray(severities, dtype=int),
        pct=np.asarray(pcts, dtype=np.float32),
        start=np.asarray(starts, dtype=int),
    )


def make_date_folds(dates: np.ndarray, n_folds: int) -> list[set[str]]:
    unique_dates = sorted(set(map(str, dates)))
    n = max(2, min(int(n_folds), len(unique_dates)))
    return [{d for i, d in enumerate(unique_dates) if i % n == fold} for fold in range(n)]


def make_scenario_folds(scenarios: np.ndarray, n_folds: int) -> list[set[str]]:
    unique_scenarios = sorted(set(map(str, scenarios)))
    n = max(2, min(int(n_folds), len(unique_scenarios)))
    return [{s for i, s in enumerate(unique_scenarios) if i % n == fold} for fold in range(n)]


def smooth_series(x: np.ndarray, window: int) -> np.ndarray:
    """Causal moving average used only for Exp7 forecasting targets.

    Each smoothed point uses only current/past DSHI values, so the history
    window cannot leak future information into the forecasting input.
    """
    w = int(window)
    if w <= 1:
        return x.astype(np.float32)
    x = x.astype(np.float32)
    out = np.empty_like(x, dtype=np.float32)
    csum = np.cumsum(np.insert(x, 0, 0.0)).astype(np.float64)
    for i in range(len(x)):
        start = max(0, i - w + 1)
        out[i] = float((csum[i + 1] - csum[start]) / (i - start + 1))
    return out.astype(np.float32)


def build_stratification_keys(windows: WindowSet, idx: np.ndarray, include_start: bool = False) -> np.ndarray:
    keys = []
    for i in idx:
        base = f"{windows.fault_type[i]}|sev{windows.severity[i]}|pct{windows.pct[i]:.2f}"
        if include_start:
            # Coarse temporal bins keep examples distributed across a scenario.
            base += f"|bin{int(windows.start[i]) // 1000}"
        keys.append(base)
    return np.asarray(keys, dtype=object)


def stratified_choice(idx: np.ndarray, keys: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Choose n indices while keeping fault type/severity/pct groups represented."""
    idx = np.asarray(idx, dtype=int)
    if n <= 0 or len(idx) == 0:
        return np.asarray([], dtype=int)
    if len(idx) <= n:
        out = idx.copy()
        rng.shuffle(out)
        return out

    selected = []
    unique_keys = sorted(set(map(str, keys)))
    per_group = max(1, n // max(1, len(unique_keys)))
    leftovers = []
    for key in unique_keys:
        group = idx[np.asarray(keys, dtype=object) == key]
        if len(group) == 0:
            continue
        take = min(len(group), per_group)
        selected.extend(rng.choice(group, size=take, replace=False).tolist())
        if len(group) > take:
            leftovers.extend(np.setdiff1d(group, np.asarray(selected, dtype=int), assume_unique=False).tolist())

    selected = np.asarray(selected, dtype=int)
    if len(selected) < n:
        remaining = np.setdiff1d(idx, selected, assume_unique=False)
        add_n = min(n - len(selected), len(remaining))
        if add_n > 0:
            selected = np.concatenate([selected, rng.choice(remaining, size=add_n, replace=False)])
    if len(selected) > n:
        selected = rng.choice(selected, size=n, replace=False)
    rng.shuffle(selected)
    return selected.astype(int)


def sampled_train_indices(
    windows: WindowSet,
    train_mask: np.ndarray,
    fault_ratio: float,
    seed: int,
    *,
    max_healthy: int,
    max_fault_full: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    train_idx = np.where(train_mask)[0]
    healthy = train_idx[~windows.is_fault_window[train_idx]]
    fault = train_idx[windows.is_fault_window[train_idx]]

    if len(healthy) > max_healthy:
        healthy = stratified_choice(
            healthy,
            build_stratification_keys(windows, healthy, include_start=True),
            max_healthy,
            rng,
        )

    fault_full = min(len(fault), max_fault_full)
    fault_n = int(round(fault_full * float(fault_ratio)))
    if fault_n > 0:
        fault_pool = stratified_choice(
            fault,
            build_stratification_keys(windows, fault),
            fault_full,
            rng,
        )
        fault_chosen = stratified_choice(
            fault_pool,
            build_stratification_keys(windows, fault_pool),
            fault_n,
            rng,
        )
        chosen = np.concatenate([healthy, fault_chosen])
    else:
        chosen = healthy
    rng.shuffle(chosen)
    return chosen.astype(int)


def sampled_test_indices(
    windows: WindowSet,
    test_mask: np.ndarray,
    seed: int,
    *,
    max_healthy: int,
    max_fault: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    test_idx = np.where(test_mask)[0]
    healthy = test_idx[~windows.is_fault_window[test_idx]]
    fault = test_idx[windows.is_fault_window[test_idx]]
    if len(healthy) > max_healthy:
        healthy = stratified_choice(
            healthy,
            build_stratification_keys(windows, healthy, include_start=True),
            max_healthy,
            rng,
        )
    if len(fault) > max_fault:
        fault = stratified_choice(
            fault,
            build_stratification_keys(windows, fault),
            max_fault,
            rng,
        )
    chosen = np.concatenate([healthy, fault])
    rng.shuffle(chosen)
    return chosen.astype(int)


class SequenceForecasterNet(nn.Module):
    def __init__(self, model_type: str, lookback: int, horizon: int, hidden: int = 64,
                 aux_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.model_type = model_type
        self.horizon = horizon
        self.aux_dim = int(aux_dim)
        self.aux_encoder = (
            nn.Sequential(nn.Linear(self.aux_dim, hidden // 2), nn.ReLU(), nn.Dropout(dropout))
            if self.aux_dim > 0 else None
        )
        head_in = hidden + (hidden // 2 if self.aux_dim > 0 else 0)
        if model_type == "LSTM":
            self.backbone = nn.LSTM(1, hidden, num_layers=1, batch_first=True, dropout=0.0)
            self.head = nn.Sequential(nn.LayerNorm(head_in), nn.Dropout(dropout), nn.Linear(head_in, horizon))
        elif model_type == "GRU":
            self.backbone = nn.GRU(1, hidden, num_layers=1, batch_first=True, dropout=0.0)
            self.head = nn.Sequential(nn.LayerNorm(head_in), nn.Dropout(dropout), nn.Linear(head_in, horizon))
        elif model_type == "CNN-LSTM":
            self.conv = nn.Sequential(
                nn.Conv1d(1, hidden // 2, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden // 2, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.backbone = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(head_in), nn.Dropout(dropout), nn.Linear(head_in, horizon))
        elif model_type == "TCN":
            self.tcn = nn.Sequential(
                nn.Conv1d(1, hidden, kernel_size=3, padding=1, dilation=1),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
                nn.ReLU(),
            )
            self.head = nn.Sequential(nn.LayerNorm(head_in), nn.Dropout(dropout), nn.Linear(head_in, horizon))
        elif model_type == "Transformer-LSTM":
            n_heads = 4
            self.input_projection = nn.Linear(1, hidden)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=n_heads,
                dim_feedforward=hidden * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.backbone = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(head_in), nn.Dropout(dropout), nn.Linear(head_in, horizon))
        elif model_type == "Transformer Encoder":
            n_heads = 4
            self.input_projection = nn.Linear(1, hidden)
            self.position_embedding = nn.Parameter(torch.zeros(1, lookback, hidden))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=n_heads,
                dim_feedforward=hidden * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.head = nn.Sequential(nn.LayerNorm(head_in), nn.Dropout(dropout), nn.Linear(head_in, horizon))
        elif model_type == "xLSTM-Transformer":
            n_heads = 4
            self.input_projection = nn.Linear(1, hidden)
            self.xlstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
            self.memory_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=n_heads,
                dim_feedforward=hidden * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.head = nn.Sequential(nn.LayerNorm(head_in), nn.Dropout(dropout), nn.Linear(head_in, horizon))
        elif model_type == "xLSTM-Tuned":
            n_heads = 4
            aux_out = hidden // 2 if self.aux_dim > 0 else 0
            self.local_stem = nn.Sequential(
                nn.Conv1d(1, hidden // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden // 2, hidden, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.input_projection = nn.Linear(1, hidden)
            self.position_embedding = nn.Parameter(torch.zeros(1, lookback, hidden))
            self.xlstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, dropout=dropout)
            self.memory_gate = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.Sigmoid())
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=n_heads,
                dim_feedforward=hidden * 3,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            tuned_head_in = hidden * 2 + aux_out
            self.head = nn.Sequential(
                nn.LayerNorm(tuned_head_in),
                nn.Dropout(dropout),
                nn.Linear(tuned_head_in, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, horizon),
            )
        else:
            raise ValueError(f"Unknown sequence model: {model_type}")

    def _with_aux(self, rep: torch.Tensor, aux: torch.Tensor | None) -> torch.Tensor:
        if self.aux_encoder is not None and aux is not None:
            return torch.cat([rep, self.aux_encoder(aux)], dim=1)
        return rep

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        # x: [batch, lookback]
        x_seq = x.unsqueeze(-1)
        if self.model_type in {"LSTM", "GRU"}:
            out, _ = self.backbone(x_seq)
            return self.head(self._with_aux(out[:, -1, :], aux))
        if self.model_type == "CNN-LSTM":
            z = self.conv(x_seq.transpose(1, 2)).transpose(1, 2)
            out, _ = self.backbone(z)
            return self.head(self._with_aux(out[:, -1, :], aux))
        if self.model_type == "TCN":
            z = self.tcn(x_seq.transpose(1, 2))[:, :, -1]
            return self.head(self._with_aux(z, aux))
        if self.model_type == "Transformer Encoder":
            z = self.input_projection(x_seq) + self.position_embedding[:, :x_seq.size(1), :]
            z = self.transformer(z)
            return self.head(self._with_aux(z[:, -1, :], aux))
        if self.model_type == "xLSTM-Transformer":
            z = self.input_projection(x_seq)
            z, _ = self.xlstm(z)
            z = self.transformer(z * self.memory_gate(z))
            return self.head(self._with_aux(z[:, -1, :], aux))
        if self.model_type == "xLSTM-Tuned":
            local = self.local_stem(x_seq.transpose(1, 2))
            local_rep = 0.5 * local[:, :, -1] + 0.5 * local.mean(dim=2)
            z = self.input_projection(x_seq) + self.position_embedding[:, :x_seq.size(1), :]
            z, _ = self.xlstm(z)
            z = self.transformer(z * self.memory_gate(z))
            rep = torch.cat([local_rep, z[:, -1, :]], dim=1)
            return self.head(self._with_aux(rep, aux))
        z = self.input_projection(x_seq)
        z = self.transformer(z)
        out, _ = self.backbone(z)
        return self.head(self._with_aux(out[:, -1, :], aux))


class TorchSequenceRegressor:
    """Small sklearn-like wrapper for DSHI multi-step sequence forecasting."""

    def __init__(self, model_type: str, lookback: int, horizon: int, seed: int,
                 fast: bool = False, max_train: int = 20_000):
        self.model_type = model_type
        self.lookback = lookback
        self.horizon = horizon
        self.seed = seed
        self.fast = fast
        self.max_train = int(max_train)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x_mean = 0.0
        self.x_std = 1.0
        self.y_mean = 0.0
        self.y_std = 1.0
        self.model: SequenceForecasterNet | None = None
        self.training_history_: list[dict[str, float]] = []
        self.training_diagnostics_: dict[str, float | int | str] = {}

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        x_all = np.asarray(x_train, dtype=np.float32)
        x = x_all[:, :self.lookback]
        aux = x_all[:, self.lookback:]
        y = np.asarray(y_train, dtype=np.float32)
        if len(x) > self.max_train:
            chosen = rng.choice(np.arange(len(x)), size=self.max_train, replace=False)
            x = x[chosen]
            aux = aux[chosen]
            y = y[chosen]

        self.x_mean = float(x.mean())
        self.x_std = float(x.std() + 1e-6)
        self.aux_mean = aux.mean(axis=0, keepdims=True).astype(np.float32) if aux.shape[1] else None
        self.aux_std = (aux.std(axis=0, keepdims=True) + 1e-6).astype(np.float32) if aux.shape[1] else None
        self.y_mean = float(y.mean())
        self.y_std = float(y.std() + 1e-6)
        x = (x - self.x_mean) / self.x_std
        if aux.shape[1]:
            aux = (aux - self.aux_mean) / self.aux_std
        y = (y - self.y_mean) / self.y_std

        order = rng.permutation(len(x))
        val_n = max(64, int(0.15 * len(order))) if len(order) >= 500 else max(1, int(0.2 * len(order)))
        val_idx = order[:val_n]
        train_idx = order[val_n:]
        if len(train_idx) == 0:
            train_idx = order
            val_idx = order[:0]

        tuned = self.model_type == "xLSTM-Tuned"
        batch_size = 384 if tuned and not self.fast else (512 if not self.fast else 256)
        train_ds = TensorDataset(
            torch.from_numpy(x[train_idx]),
            torch.from_numpy(aux[train_idx].astype(np.float32)),
            torch.from_numpy(y[train_idx]),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        if len(val_idx):
            val_x = torch.from_numpy(x[val_idx]).to(self.device)
            val_aux = torch.from_numpy(aux[val_idx].astype(np.float32)).to(self.device)
            val_y = torch.from_numpy(y[val_idx]).to(self.device)
            val_y_raw = y[val_idx] * self.y_std + self.y_mean
        else:
            val_x = val_aux = val_y = val_y_raw = None

        hidden = 48 if (self.fast and tuned) else (32 if self.fast else (96 if tuned else 64))
        dropout = 0.15 if tuned else 0.1
        self.model = SequenceForecasterNet(
            self.model_type,
            self.lookback,
            self.horizon,
            hidden=hidden,
            aux_dim=aux.shape[1],
            dropout=dropout,
        ).to(self.device)
        lr = 8e-4 if tuned else 1e-3
        weight_decay = 5e-4 if tuned else 1e-4
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        max_epochs = 16 if self.fast else (90 if tuned else 40)
        patience = 6 if self.fast else (18 if tuned else 8)
        warmup_epochs = 3 if self.fast else (10 if tuned else 0)
        if tuned:
            def lr_lambda(epoch_idx: int) -> float:
                if epoch_idx < warmup_epochs:
                    return float(epoch_idx + 1) / float(max(1, warmup_epochs))
                denom = max(1, max_epochs - warmup_epochs)
                progress = min(1.0, float(epoch_idx - warmup_epochs) / float(denom))
                return 0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * progress))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=3)
        base_loss_fn = nn.SmoothL1Loss(reduction="none")

        def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            loss = base_loss_fn(pred, target)
            if tuned and loss.ndim == 2 and loss.shape[1] > 1:
                weights = torch.linspace(1.0, 1.35, steps=loss.shape[1], device=loss.device)
                loss = loss * weights.unsqueeze(0)
            return loss.mean()
        best_state = None
        best_val = float("inf")
        best_epoch = -1
        stale = 0
        self.training_history_ = []

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            train_losses = []
            for bx, bax, by in train_loader:
                bx = bx.to(self.device)
                bax = bax.to(self.device)
                by = by.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(bx, bax)
                loss = loss_fn(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            if val_x is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(self.model(val_x, val_aux), val_y).detach().cpu())
            else:
                val_loss = float(np.mean(train_losses))
            if tuned:
                scheduler.step()
            else:
                scheduler.step(val_loss)
            lr = float(optimizer.param_groups[0]["lr"])
            self.training_history_.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(train_losses)),
                    "val_loss": val_loss,
                    "lr": lr,
                    "train_size": int(len(train_idx)),
                    "val_size": int(len(val_idx)),
                    "x_mean": self.x_mean,
                    "x_std": self.x_std,
                    "y_mean": self.y_mean,
                    "y_std": self.y_std,
                }
            )
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.training_diagnostics_ = {
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "epochs_ran": int(len(self.training_history_)),
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "aux_dim": int(aux.shape[1]),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "dropout": float(dropout),
            "warmup_epochs": int(warmup_epochs),
        }
        if val_x is not None and val_y_raw is not None:
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(val_x, val_aux).detach().cpu().numpy() * self.y_std + self.y_mean
            self.training_diagnostics_["val_mae_raw"] = float(mean_absolute_error(val_y_raw.reshape(-1), val_pred.reshape(-1)))
            self.training_diagnostics_["val_rmse_raw"] = float(math.sqrt(mean_squared_error(val_y_raw.reshape(-1), val_pred.reshape(-1))))
        return self

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TorchSequenceRegressor is not fitted.")
        x_all = np.asarray(x_test, dtype=np.float32)
        x = x_all[:, :self.lookback]
        aux = x_all[:, self.lookback:]
        x = (x - self.x_mean) / self.x_std
        if aux.shape[1] and self.aux_mean is not None and self.aux_std is not None:
            aux = (aux - self.aux_mean) / self.aux_std
        self.model.eval()
        preds = []
        batch_size = 2048
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                bx = torch.from_numpy(x[start:start + batch_size]).to(self.device)
                bax = torch.from_numpy(aux[start:start + batch_size].astype(np.float32)).to(self.device)
                pred = self.model(bx, bax).detach().cpu().numpy()
                preds.append(pred)
        out = np.vstack(preds)
        return (out * self.y_std + self.y_mean).astype(np.float32)


def make_models(
    seed: int,
    lookback: int,
    horizon: int,
    fast: bool = False,
    include_heavy: bool = False,
    max_deep_train: int = 20_000,
) -> dict[str, object]:
    n_estimators = 40 if fast else 80
    gb_iter = 30 if fast else 80
    mlp_iter = 80 if fast else 120
    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "HistGBDT": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=gb_iter,
                learning_rate=0.06,
                l2_regularization=0.05,
                random_state=seed,
            )
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=1,
        ),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(96, 48),
                activation="relu",
                alpha=1e-4,
                max_iter=mlp_iter,
                early_stopping=True,
                random_state=seed,
            ),
        ),
        "LSTM": TorchSequenceRegressor("LSTM", lookback, horizon, seed + 11, fast=fast, max_train=max_deep_train),
        "GRU": TorchSequenceRegressor("GRU", lookback, horizon, seed + 13, fast=fast, max_train=max_deep_train),
        "CNN-LSTM": TorchSequenceRegressor("CNN-LSTM", lookback, horizon, seed + 17, fast=fast, max_train=max_deep_train),
        "TCN": TorchSequenceRegressor("TCN", lookback, horizon, seed + 19, fast=fast, max_train=max_deep_train),
        "Transformer Encoder": TorchSequenceRegressor("Transformer Encoder", lookback, horizon, seed + 21, fast=fast, max_train=max_deep_train),
        "Transformer-LSTM": TorchSequenceRegressor("Transformer-LSTM", lookback, horizon, seed + 23, fast=fast, max_train=max_deep_train),
        "xLSTM-Transformer": TorchSequenceRegressor("xLSTM-Transformer", lookback, horizon, seed + 29, fast=fast, max_train=max_deep_train),
    }
    return models


def predict_baseline(method: str, hist: np.ndarray, horizon: int) -> np.ndarray:
    if method == "Persistence":
        return np.repeat(hist[:, -1:], horizon, axis=1)
    if method == "Linear trend":
        out = np.zeros((len(hist), horizon), dtype=np.float32)
        t = np.arange(hist.shape[1], dtype=np.float32)
        future_t = np.arange(hist.shape[1], hist.shape[1] + horizon, dtype=np.float32)
        for i, h in enumerate(hist):
            slope, intercept = np.polyfit(t, h, deg=1)
            out[i] = intercept + slope * future_t
        return out
    raise KeyError(method)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(math.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp))
    # Trend error: compare future slope, not only pointwise values.
    t = np.arange(y_true.shape[1], dtype=np.float32)
    true_slopes = np.array([np.polyfit(t, row, deg=1)[0] for row in y_true])
    pred_slopes = np.array([np.polyfit(t, row, deg=1)[0] for row in y_pred])
    slope_mae = float(mean_absolute_error(true_slopes, pred_slopes))
    final_mae = float(mean_absolute_error(y_true[:, -1], y_pred[:, -1]))
    return {"mae": mae, "rmse": rmse, "r2": r2, "slope_mae": slope_mae, "final_mae": final_mae}


def aggregate_results(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ratio_label in [RATIO_LABELS[r] for r in RATIO_LEVELS]:
        for method in METHOD_ORDER:
            sub = detail[(detail["Ratio"] == ratio_label) & (detail["Method"] == method)]
            if sub.empty:
                continue
            rows.append(
                {
                    "Ratio": ratio_label,
                    "Method": method,
                    "MAE": f"{sub['MAE'].mean():.4f}±{sub['MAE'].std(ddof=0):.4f}",
                    "RMSE": f"{sub['RMSE'].mean():.4f}±{sub['RMSE'].std(ddof=0):.4f}",
                    "R2": f"{sub['R2'].mean():.4f}±{sub['R2'].std(ddof=0):.4f}",
                    "Slope MAE": f"{sub['Slope MAE'].mean():.5f}±{sub['Slope MAE'].std(ddof=0):.5f}",
                    "Final-step MAE": f"{sub['Final-step MAE'].mean():.4f}±{sub['Final-step MAE'].std(ddof=0):.4f}",
                }
            )
    return pd.DataFrame(rows)


def aggregate_subgroup_results(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if detail.empty:
        return pd.DataFrame()
    group_cols = ["Group Type", "Group Value", "Ratio", "Method"]
    for keys, sub in detail.groupby(group_cols, sort=False):
        group_type, group_value, ratio_label, method = keys
        rows.append(
            {
                "Group Type": group_type,
                "Group Value": group_value,
                "Ratio": ratio_label,
                "Method": method,
                "Test Windows": int(round(sub["Test Windows"].mean())),
                "MAE": f"{sub['MAE'].mean():.4f}±{sub['MAE'].std(ddof=0):.4f}",
                "RMSE": f"{sub['RMSE'].mean():.4f}±{sub['RMSE'].std(ddof=0):.4f}",
                "R2": f"{sub['R2'].mean():.4f}±{sub['R2'].std(ddof=0):.4f}",
            }
        )
    return pd.DataFrame(rows)


def subgroup_metric_rows(
    windows: WindowSet,
    test_idx: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    *,
    fold_idx: int,
    ratio_label: str,
    ratio: float,
    method: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if len(test_idx) == 0:
        return rows

    local_fault = windows.is_fault_window[test_idx]
    if not np.any(local_fault):
        return rows

    fault_positions = np.where(local_fault)[0]
    group_specs = [
        ("fault_type", windows.fault_type[test_idx]),
        ("severity", windows.severity[test_idx]),
        ("pct", windows.pct[test_idx]),
    ]
    for group_type, values in group_specs:
        for value in sorted(set(values[fault_positions].tolist()), key=lambda x: str(x)):
            mask = local_fault & (values == value)
            positions = np.where(mask)[0]
            if len(positions) == 0:
                continue
            metrics = regression_metrics(y_test[positions], y_pred[positions])
            if group_type == "pct":
                group_value = f"{float(value):.2f}"
            else:
                group_value = str(value)
            rows.append(
                {
                    "Fold": fold_idx,
                    "Group Type": group_type,
                    "Group Value": group_value,
                    "Ratio": ratio_label,
                    "Fault Ratio": ratio,
                    "Method": method,
                    "Test Windows": int(len(positions)),
                    "MAE": metrics["mae"],
                    "RMSE": metrics["rmse"],
                    "R2": metrics["r2"],
                }
            )
    return rows


def plot_ratio_lines(numeric: pd.DataFrame, out_base: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ratios = [RATIO_LABELS[r] for r in RATIO_LEVELS]
    x = np.arange(len(ratios))
    for method in METHOD_ORDER:
        sub = numeric[numeric["Method"] == method].set_index("Ratio").reindex(ratios)
        if sub.empty or sub["MAE_mean"].isna().all():
            continue
        axes[0].plot(x, sub["MAE_mean"], marker="o", lw=1.6, label=method, color=METHOD_COLORS.get(method))
        axes[1].plot(x, sub["RMSE_mean"], marker="o", lw=1.6, label=method, color=METHOD_COLORS.get(method))
    for ax, ylabel in zip(axes, ["MAE", "RMSE"]):
        ax.set_xticks(x)
        ax.set_xticklabels(ratios, rotation=25, ha="right")
        ax.set_xlabel("Fault-window proportion added to training")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].set_title("DSHI forecasting error under scarce-fault ratios")
    axes[1].set_title("Future-sequence reconstruction error")
    axes[1].legend(fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_prediction_case(
    hist: np.ndarray,
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    out_path: Path,
    *,
    scenario: str,
    start: int,
    title: str = "Example of degradation progression forecasting",
) -> None:
    t_hist = np.arange(len(hist))
    t_future = np.arange(len(hist), len(hist) + len(target))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.axvspan(t_hist[0], t_hist[-1], color="#F3F4F6", alpha=0.9, label="Input history")
    ax.axvspan(t_future[0], t_future[-1], color="#FDECEC", alpha=0.7, label="Forecast horizon")
    ax.plot(t_hist, hist, color="#333333", lw=1.8, label="Observed DSHI history")
    ax.plot(t_future, target, color="#000000", lw=2.2, label="Ground-truth future DSHI")
    for method, pred in predictions.items():
        width = 2.0 if method not in {"Persistence", "Linear trend"} else 1.4
        alpha = 0.95 if method not in {"Persistence", "Linear trend"} else 0.55
        ax.plot(t_future, pred, lw=width, alpha=alpha, label=method, color=METHOD_COLORS.get(method))
    ax.axvline(len(hist) - 0.5, color="#777777", lw=1.0, ls=":")
    ax.set_xlabel("Relative time step")
    ax.set_ylabel("Dual-Space Health Index")
    ax.set_title(title)
    ax.text(
        0.01,
        0.02,
        f"{scenario} | start={start}",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_representative_prediction(
    windows: WindowSet,
    predictions: dict[str, np.ndarray],
    test_indices: np.ndarray,
    out_base: Path,
    preferred_method: str,
) -> None:
    if len(test_indices) == 0 or preferred_method not in predictions:
        return
    cand_positions = np.where(windows.is_fault_window[test_indices])[0]
    if len(cand_positions) == 0:
        cand_positions = np.arange(len(test_indices))
    per_case_mae = np.mean(
        np.abs(windows.target[test_indices[cand_positions]] - predictions[preferred_method][cand_positions]),
        axis=1,
    )
    median_rank = np.argsort(per_case_mae)[len(per_case_mae) // 2]
    pos = int(cand_positions[median_rank])
    local = int(test_indices[pos])
    selected_predictions = {
        m: p[pos]
        for m, p in predictions.items()
        if m in {"Persistence", "Linear trend", "HistGBDT", preferred_method}
    }
    plot_prediction_case(
        windows.hist[local],
        windows.target[local],
        selected_predictions,
        out_base,
        scenario=str(windows.scenario[local]),
        start=int(windows.start[local]),
    )


def plot_prediction_bank(
    windows: WindowSet,
    predictions: dict[str, np.ndarray],
    test_indices: np.ndarray,
    out_dir: Path,
    preferred_method: str,
    max_plots: int,
) -> list[dict[str, object]]:
    if len(test_indices) == 0 or preferred_method not in predictions:
        return []
    rows = []
    cand_positions = np.where(windows.is_fault_window[test_indices])[0]
    if len(cand_positions) == 0:
        cand_positions = np.arange(len(test_indices))
    errors = np.mean(
        np.abs(windows.target[test_indices[cand_positions]] - predictions[preferred_method][cand_positions]),
        axis=1,
    )
    # Plot from low to high error so the folder can be skimmed quickly.
    order = cand_positions[np.argsort(errors)]
    if max_plots > 0:
        order = order[:max_plots]
    for rank, pos in enumerate(order, start=1):
        local = int(test_indices[int(pos)])
        selected_predictions = {
            m: p[int(pos)]
            for m, p in predictions.items()
            if m in {"Persistence", "Linear trend", "HistGBDT", preferred_method}
        }
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(windows.scenario[local]))
        out_path = out_dir / f"{rank:05d}_{safe_name}_start{int(windows.start[local])}"
        plot_prediction_case(
            windows.hist[local],
            windows.target[local],
            selected_predictions,
            out_path,
            scenario=str(windows.scenario[local]),
            start=int(windows.start[local]),
            title="DSHI degradation progression forecasting candidate",
        )
        rows.append(
            {
                "rank_by_mae": rank,
                "scenario": str(windows.scenario[local]),
                "date": str(windows.date[local]),
                "fault_type": str(windows.fault_type[local]),
                "severity": int(windows.severity[local]),
                "pct": float(windows.pct[local]),
                "start": int(windows.start[local]),
                "preferred_method": preferred_method,
                "case_mae": float(np.mean(np.abs(windows.target[local] - predictions[preferred_method][int(pos)]))),
                "figure": str(out_path.with_suffix(".png")),
            }
        )
    return rows


def run_exp7_forecast_ratio(
    fast: bool = False,
    n_folds: int = 5,
    max_healthy: int = 120_000,
    max_fault_full: int = 30_000,
    horizon_override: int | None = None,
    max_test_healthy: int = 30_000,
    max_test_fault: int = 10_000,
    include_heavy: bool = False,
    smooth_window: int = 5,
    max_deep_train: int = 20_000,
    plot_all_examples: bool = False,
    max_example_plots: int = 200,
    example_ratio: str = "P5-100%",
    preferred_example_method: str = "MLP",
    output_tag: str = "v2",
    residual_target: bool = True,
) -> None:
    device = COMMON.device
    lookback = int(EXP7.lookback)
    horizon = int(horizon_override) if horizon_override is not None else int(EXP7.lookforward)
    slide_step = int(EXP7.slide_step)
    seed = int(COMMON.seeds[0])
    max_per_type = int(EXP7.fast_num_scenarios) if fast else None

    out_dir = PATHS.exp7_figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    PATHS.figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print(f"AACE-CM-Exp7: DSHI sequence forecasting ratio ablation [{'fast' if fast else 'formal'}]", flush=True)
    print(
        f"device={device} lookback={lookback} horizon={horizon} slide={slide_step} "
        f"smooth_window={smooth_window}",
        flush=True,
    )
    print("=" * 72, flush=True)

    best_cfg = load_exp3_best_config(required=False)
    if best_cfg:
        print(
            "Exp3/4 cached model: "
            f"AE={best_cfg.get('ae_structure')} cluster={best_cfg.get('baseline_method')} "
            f"window={best_cfg.get('window_w')}",
            flush=True,
        )
    sac.print_cache_status()

    scenarios = load_fault_scenarios(device, max_per_type=max_per_type)
    if fast and len(scenarios) > 8:
        scenarios = scenarios[:8]
    print(f"Loaded scenarios: {len(scenarios)}", flush=True)

    print("Computing DSHI trajectories...", flush=True)
    dshi_by_scenario = compute_scenario_dshi(scenarios, device=device)
    windows = build_windows(scenarios, dshi_by_scenario, lookback, horizon, slide_step, smooth_window=smooth_window)
    print(
        f"Windows: {len(windows.hist)} fault_future={int(windows.is_fault_window.sum())} "
        f"rate={windows.is_fault_window.mean():.2%}",
        flush=True,
    )
    if len(windows.hist) == 0:
        raise RuntimeError("No Exp7 windows generated.")

    folds = (
        make_scenario_folds(windows.scenario, n_folds=2)
        if fast
        else make_date_folds(windows.date, n_folds=n_folds)
    )
    detail_rows = []
    subgroup_rows = []
    numeric_rows = []
    deep_history_rows = []
    deep_diagnostic_rows = []
    representative_predictions: dict[str, np.ndarray] | None = None
    representative_test_idx: np.ndarray | None = None
    example_index_rows: list[dict[str, object]] = []

    for fold_idx, held_dates in enumerate(folds, start=1):
        test_mask = (
            np.isin(windows.scenario, list(held_dates))
            if fast
            else np.isin(windows.date, list(held_dates))
        )
        train_mask = ~test_mask
        test_idx = np.where(test_mask)[0]
        if len(test_idx) == 0:
            continue
        if not fast:
            test_idx = sampled_test_indices(
                windows,
                test_mask,
                seed=seed + 10_000 + fold_idx,
                max_healthy=max_test_healthy,
                max_fault=max_test_fault,
            )
        x_test_feat = windows.features[test_idx]
        y_test = windows.target[test_idx]
        hist_test = windows.hist[test_idx]

        print(
            f"Fold {fold_idx}/{len(folds)} held_dates={sorted(held_dates)} "
            f"test={len(test_idx)} fault_test={int(windows.is_fault_window[test_idx].sum())}",
            flush=True,
        )

        for ratio in RATIO_LEVELS:
            ratio_label = RATIO_LABELS[ratio]
            train_idx = sampled_train_indices(
                windows,
                train_mask,
                ratio,
                seed=seed + fold_idx + int(ratio * 1000),
                max_healthy=max_healthy if not fast else 4_000,
                max_fault_full=max_fault_full if not fast else 1_000,
            )
            x_train_feat = windows.features[train_idx]
            y_train = windows.target[train_idx]
            if residual_target:
                y_train_fit = y_train - windows.hist[train_idx, -1:]
            else:
                y_train_fit = y_train
            print(
                f"  {ratio_label}: train={len(train_idx)} "
                f"fault_train={int(windows.is_fault_window[train_idx].sum())}",
                flush=True,
            )

            preds: dict[str, np.ndarray] = {
                "Persistence": predict_baseline("Persistence", hist_test, horizon),
                "Linear trend": predict_baseline("Linear trend", hist_test, horizon),
            }

            for method, model in make_models(
                seed + fold_idx,
                lookback=lookback,
                horizon=horizon,
                fast=fast,
                include_heavy=(include_heavy or fast),
                max_deep_train=max_deep_train if not fast else min(max_deep_train, 4_000),
            ).items():
                model.fit(x_train_feat, y_train_fit)
                if hasattr(model, "training_history_"):
                    for row in getattr(model, "training_history_", []):
                        deep_history_rows.append(
                            {
                                "Fold": fold_idx,
                                "Ratio": ratio_label,
                                "Fault Ratio": ratio,
                                "Method": method,
                                **row,
                            }
                        )
                    diag = getattr(model, "training_diagnostics_", {})
                    if diag:
                        deep_diagnostic_rows.append(
                            {
                                "Fold": fold_idx,
                                "Ratio": ratio_label,
                                "Fault Ratio": ratio,
                                "Method": method,
                                **diag,
                            }
                        )
                pred_fit = np.asarray(model.predict(x_test_feat), dtype=np.float32)
                if residual_target:
                    pred_fit = pred_fit + hist_test[:, -1:]
                preds[method] = pred_fit

            for method in METHOD_ORDER:
                if method not in preds:
                    continue
                m = regression_metrics(y_test, preds[method])
                detail_rows.append(
                    {
                        "Fold": fold_idx,
                        "Held Dates": ";".join(sorted(held_dates)),
                        "Ratio": ratio_label,
                        "Fault Ratio": ratio,
                        "Method": method,
                        "Train Windows": len(train_idx),
                        "Fault Train Windows": int(windows.is_fault_window[train_idx].sum()),
                        "Test Windows": len(test_idx),
                        "Fault Test Windows": int(windows.is_fault_window[test_idx].sum()),
                        "MAE": m["mae"],
                        "RMSE": m["rmse"],
                        "R2": m["r2"],
                        "Slope MAE": m["slope_mae"],
                        "Final-step MAE": m["final_mae"],
                    }
                )
                subgroup_rows.extend(
                    subgroup_metric_rows(
                        windows,
                        test_idx,
                        y_test,
                        preds[method],
                        fold_idx=fold_idx,
                        ratio_label=ratio_label,
                        ratio=ratio,
                        method=method,
                    )
                )
            if representative_predictions is None and ratio_label == example_ratio:
                representative_predictions = preds
                representative_test_idx = test_idx

            if plot_all_examples and ratio_label == example_ratio:
                bank_dir = out_dir / "exp7_prediction_case_bank" / f"fold{fold_idx:02d}_{ratio_label}"
                example_index_rows.extend(
                    plot_prediction_bank(
                        windows,
                        preds,
                        test_idx,
                        bank_dir,
                        preferred_method=preferred_example_method if preferred_example_method in preds else "HistGBDT",
                        max_plots=max_example_plots,
                    )
                )

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        raise RuntimeError("No Exp7 forecast-ratio folds evaluated.")
    subgroup_detail = pd.DataFrame(subgroup_rows)
    subgroup_summary = aggregate_subgroup_results(subgroup_detail)

    summary = aggregate_results(detail)
    for ratio_label in [RATIO_LABELS[r] for r in RATIO_LEVELS]:
        for method in METHOD_ORDER:
            sub = detail[(detail["Ratio"] == ratio_label) & (detail["Method"] == method)]
            if sub.empty:
                continue
            numeric_rows.append(
                {
                    "Ratio": ratio_label,
                    "Method": method,
                    "MAE_mean": sub["MAE"].mean(),
                    "RMSE_mean": sub["RMSE"].mean(),
                    "R2_mean": sub["R2"].mean(),
                    "Slope_MAE_mean": sub["Slope MAE"].mean(),
                    "Final_step_MAE_mean": sub["Final-step MAE"].mean(),
                }
            )
    numeric = pd.DataFrame(numeric_rows)

    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(output_tag)).strip("_")
    prefix = f"exp7_{tag}_fast" if fast else f"exp7_{tag}"
    summary_path = out_dir / f"{prefix}_table_t6_forecast_ratio.csv"
    numeric_path = out_dir / f"{prefix}_table_t6_forecast_ratio_numeric.csv"
    detail_path = out_dir / f"{prefix}_detail_forecast_ratio.csv"
    subgroup_detail_path = out_dir / f"{prefix}_subgroup_detail_forecast_ratio.csv"
    subgroup_summary_path = out_dir / f"{prefix}_subgroup_summary_forecast_ratio.csv"
    meta_path = out_dir / f"{prefix}_forecast_ratio_metadata.json"
    deep_history_path = out_dir / f"{prefix}_deep_training_history.csv"
    deep_diag_path = out_dir / f"{prefix}_deep_training_diagnostics.csv"
    fig_path = out_dir / f"{prefix}_fig_f18_forecast_ratio_lines"
    example_path = out_dir / f"{prefix}_fig_f19_forecast_example"
    example_index_path = out_dir / f"{prefix}_prediction_case_bank_index.csv"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    numeric.to_csv(numeric_path, index=False, encoding="utf-8-sig")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    if not subgroup_detail.empty:
        subgroup_detail.to_csv(subgroup_detail_path, index=False, encoding="utf-8-sig")
    if not subgroup_summary.empty:
        subgroup_summary.to_csv(subgroup_summary_path, index=False, encoding="utf-8-sig")
    if deep_history_rows:
        pd.DataFrame(deep_history_rows).to_csv(deep_history_path, index=False, encoding="utf-8-sig")
    if deep_diagnostic_rows:
        pd.DataFrame(deep_diagnostic_rows).to_csv(deep_diag_path, index=False, encoding="utf-8-sig")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "Exp7 DSHI sequence forecasting ratio ablation",
                "experiment_version": "v2_causal_smoothing_aux_features",
                "task": "history DSHI window -> future DSHI sequence",
                "health_index": sac.HEALTH_INDEX_NAME,
                "lookback": lookback,
                "horizon": horizon,
                "slide_step": slide_step,
                "forecast_target_smoothing_window": smooth_window,
                "smoothing_mode": "causal moving average",
                "learning_target": "future_dshi_minus_current_dshi" if residual_target else "future_dshi",
                "ratios": {RATIO_LABELS[r]: r for r in RATIO_LEVELS},
                "num_scenarios": len(scenarios),
                "num_windows": int(len(windows.hist)),
                "fault_future_windows": int(windows.is_fault_window.sum()),
                "fold_mode": "date-group",
                "n_folds": len(folds),
                "max_healthy_train_windows": max_healthy if not fast else 4_000,
                "max_fault_full_windows": max_fault_full if not fast else 1_000,
                "max_test_healthy_windows": max_test_healthy if not fast else "all",
                "max_test_fault_windows": max_test_fault if not fast else "all",
                "include_heavy_models": bool(include_heavy or fast),
                "max_deep_train_windows": max_deep_train if not fast else min(max_deep_train, 4_000),
                "deep_model_aux_features": "same history-statistics appended to feature baselines",
                "train_sampling": "healthy windows stratified by scenario/time bin; fault windows stratified by fault_type/severity/pct",
                "test_sampling": "healthy windows stratified by scenario/time bin; fault windows stratified by fault_type/severity/pct",
                "subgroup_analysis": "fault test windows grouped by fault_type, severity, and pct with MAE/RMSE/R2",
                "inspired_by": "Wang et al. 2026 SyntheITS P1-P5 ratio-ablation evaluation style",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    plot_ratio_lines(numeric, fig_path)
    if representative_predictions is not None and representative_test_idx is not None:
        preferred = preferred_example_method if preferred_example_method in representative_predictions else "HistGBDT"
        plot_representative_prediction(windows, representative_predictions, representative_test_idx, example_path, preferred)
    if example_index_rows:
        pd.DataFrame(example_index_rows).to_csv(example_index_path, index=False, encoding="utf-8-sig")

    if not fast:
        for path in [
            summary_path,
            numeric_path,
            detail_path,
            subgroup_detail_path,
            subgroup_summary_path,
            meta_path,
            deep_history_path,
            deep_diag_path,
            fig_path.with_suffix(".png"),
            fig_path.with_suffix(".pdf"),
            example_path.with_suffix(".png"),
            example_path.with_suffix(".pdf"),
            example_index_path,
        ]:
            if path.exists():
                shutil.copy2(path, PATHS.figures_dir / path.name)

    print("\nExp7 DSHI forecast-ratio summary:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nSaved: {summary_path}", flush=True)
    print(f"Saved: {detail_path}", flush=True)
    if not subgroup_summary.empty:
        print(f"Saved: {subgroup_summary_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="AACE-CM-Exp7: DSHI sequence forecasting ratio ablation")
    parser.add_argument("--fast", action="store_true", help="run a small validation subset")
    parser.add_argument("--formal", action="store_true", help="run the formal configuration")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-healthy", type=int, default=120000)
    parser.add_argument("--max-fault-full", type=int, default=30000)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max-test-healthy", type=int, default=30000)
    parser.add_argument("--max-test-fault", type=int, default=10000)
    parser.add_argument("--include-heavy", action="store_true", help="include slow multi-output gradient boosting in formal mode")
    parser.add_argument("--smooth-window", type=int, default=5, help="moving-average window for Exp7 forecasting target")
    parser.add_argument("--max-deep-train", type=int, default=20000, help="maximum training windows per deep sequence model")
    parser.add_argument("--plot-all-examples", action="store_true", help="plot a bank of test prediction cases")
    parser.add_argument("--max-example-plots", type=int, default=200, help="maximum prediction-case figures; 0 means all sampled test cases")
    parser.add_argument("--example-ratio", type=str, default="P5-100%", help="ratio level used for representative/case-bank plots")
    parser.add_argument("--preferred-example-method", type=str, default="MLP")
    parser.add_argument("--output-tag", type=str, default="v2", help="output filename tag to avoid overwriting prior Exp7 runs")
    parser.add_argument("--direct-target", action="store_true", help="train models to predict future DSHI directly instead of residual increments")
    args = parser.parse_args()
    run_exp7_forecast_ratio(
        fast=args.fast and not args.formal,
        n_folds=args.folds,
        max_healthy=args.max_healthy,
        max_fault_full=args.max_fault_full,
        horizon_override=args.horizon,
        max_test_healthy=args.max_test_healthy,
        max_test_fault=args.max_test_fault,
        include_heavy=args.include_heavy,
        smooth_window=args.smooth_window,
        max_deep_train=args.max_deep_train,
        plot_all_examples=args.plot_all_examples,
        max_example_plots=args.max_example_plots,
        example_ratio=args.example_ratio,
        preferred_example_method=args.preferred_example_method,
        output_tag=args.output_tag,
        residual_target=not args.direct_target,
    )


if __name__ == "__main__":
    main()
