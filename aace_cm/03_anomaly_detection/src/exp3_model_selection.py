"""
EXP3: AE+基准参数网格搜索 v2
================================
对应论文 C2（为 exp8 打基础的异常检测框架）。

升级内容（v2 相比 v1）：
  ① 新增 AE 架构：Conv1DAE（时序感知）、VAE（概率分布）、MemAE（记忆增强）
  ② 新增距离度量：Mahalanobis（方向敏感）
  ③ 新增基准方法：LOF（局部密度异常检测）
  ④ Ours-A / Ours-B 在后处理阶段统一计算，不再作为独立搜索维度
  ⑤ Conv1DAE 需要时序窗口数据（window=7），在数据预处理阶段构造

搜索网格维度（5维）：
  AE结构:     SimpleAE / FlexibleAE / DeepAE / Conv1DAE / VAE / MemAE
  距离度量:   euclidean / mahalanobis
  基准方法:   KMeans_K3 / KMeans_K5 / GMM_K3 / GMM_K5 / DBSCAN / LOF
  隐向量维度: 4 / 8
  α权重:      0.3 / 0.5 / 0.7

运行：
  python src/exp3_model_selection.py --fast
  python src/exp3_model_selection.py --formal
  python src/exp3_model_selection.py --ae_filter Conv1DAE,VAE --formal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN, KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import (roc_auc_score, roc_curve,
                             precision_score, recall_score, f1_score,
                             average_precision_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from scipy.linalg import solve_triangular

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from config import (
    COMMON,
    EXP3,
    PATHS,
    get_active_fault_data_dir,
    get_exp2_generation_dates,
    get_feature_cols,
    get_feature_data_file,
    get_label_file,
    get_train_dates,
    train_dates_set,
)

DEVICE = torch.device("cpu")


def _set_device(gpu_id: int) -> None:
    global DEVICE
    DEVICE = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


# ── 随机种子 ─────────────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class Exp3DataBundle:
    x_train_healthy: np.ndarray       # (N_train, 3) 健康数据（标准化后）
    train_labels: np.ndarray           # (N_train,) 全1（NORMAL）

    daily_x: List[np.ndarray]          # List[(Mi, 3)] 每天的原始特征（标准化前）
    daily_y: List[np.ndarray]          # List[(Mi,)] 每天的 state 标签
    daily_dates: List[str]

    # 时序窗口数据（仅 Conv1DAE 使用）
    x_train_windowed: np.ndarray       # (N_train, window_size, 3)

    fault_x: List[np.ndarray]          # List[(Mi, 3)] 故障数据
    fault_labels: List[np.ndarray]      # List[(Mi,)] 0=健康段, 1=故障段
    fault_scenarios: List[str]
    fault_meta: List[dict]


# ── AE 模型类 ────────────────────────────────────────────────────────────────

class FlexibleAutoEncoder(nn.Module):
    """MLP 自编码器（原有）。"""
    def __init__(self, input_dim: int, encoder_dims: List[int], decoder_dims: List[int]):
        super().__init__()
        enc_layers, prev = [], input_dim
        for d in encoder_dims:
            enc_layers.extend([nn.Linear(prev, d), nn.ReLU()])
            prev = d
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers, prev = [], prev
        for d in decoder_dims:
            dec_layers.extend([nn.Linear(prev, d), nn.ReLU()])
            prev = d
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class Conv1DAutoEncoder(nn.Module):
    """
    1D-CNN 自编码器：每次编码一个时序窗口，捕获局部时序模式。
    输入: (batch, window_size, 3)  — 例如 (batch, 7, 3)
    输出: (batch, 3)
    """
    def __init__(self, window_size: int = 7, channels: Tuple[int, int, int] = (32, 64, 128),
                 latent_dim: int = 16, input_dim: int = 3):
        super().__init__()
        c1, c2, c3 = channels
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(c1), nn.ReLU(),
            nn.MaxPool1d(2),                           # 7→3
            nn.Conv1d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(c2), nn.ReLU(),
            nn.MaxPool1d(2),                           # 3→1
            nn.Flatten(),
        )
        # 展平后维度: c2 * 1 = c2
        self.fc_enc = nn.Linear(c2, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, c2)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose1d(c2, c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(c1), nn.ReLU(),
            nn.Upsample(size=3, mode='linear'),       # 1→3
            nn.ConvTranspose1d(c1, input_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(input_dim), nn.ReLU(),
            nn.Upsample(size=window_size, mode='linear'),  # 3→7
        )
        self._window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_size, 3) → permute → (batch, 3, window_size)
        h = self.encoder(x.permute(0, 2, 1))
        z = self.fc_enc(h)
        h2 = self.fc_dec(z).unsqueeze(-1)   # (batch, c2, 1)
        recon = self.decoder(h2)
        return recon.permute(0, 2, 1)       # (batch, window_size, 3)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.encoder(x.permute(0, 2, 1))
            return self.fc_enc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h2 = self.fc_dec(z).unsqueeze(-1)
        recon = self.decoder(h2)
        return recon.permute(0, 2, 1)


class VAutoEncoder(nn.Module):
    """
    变分自编码器（VAE）：latent 空间是概率分布 (μ, logσ²)，
    能更鲁棒地捕捉"正常状态的分布边界"。
    """
    def __init__(self, input_dim: int, encoder_dims: List[int], decoder_dims: List[int],
                 latent_dim: int, beta: float = 1.0):
        super().__init__()
        enc_layers, prev = [], input_dim
        for d in encoder_dims:
            enc_layers.extend([nn.Linear(prev, d), nn.ReLU()])
            prev = d
        self.encoder_mean = nn.Sequential(*enc_layers, nn.Linear(prev, latent_dim))
        self.encoder_logvar = nn.Sequential(*enc_layers, nn.Linear(prev, latent_dim))

        dec_layers, prev = [], latent_dim
        for d in decoder_dims:
            dec_layers.extend([nn.Linear(prev, d), nn.ReLU()])
            prev = d
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.beta = beta
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder_mean(x)

    def encode_latent_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            mu = self.encoder_mean(x)
            logvar = self.encoder_logvar(x)
            return mu, logvar


class MemoryAutoEncoder(nn.Module):
    """
    记忆增强自编码器（MemAE）：
    - Memory Bank 存储正常模式的原型向量
    - 推理时用 Query 从 Memory Bank 中检索最相似的原型
    - 重构 = 原型的加权组合，重构失败程度 = 异常分数
    """
    def __init__(self, input_dim: int, encoder_dims: List[int], decoder_dims: List[int],
                 memory_size: int = 128, latent_dim: int = 16):
        super().__init__()
        enc_layers, prev = [], input_dim
        for d in encoder_dims:
            enc_layers.extend([nn.Linear(prev, d), nn.ReLU()])
            prev = d
        self.encoder = nn.Sequential(*enc_layers)
        self.latent_dim = latent_dim or prev

        self.memory_keys = nn.Parameter(
            torch.randn(memory_size, self.latent_dim),
            requires_grad=True
        )
        nn.init.kaiming_normal_(self.memory_keys.data)

        self.query_proj = nn.Linear(self.latent_dim, memory_size)

        dec_layers, prev = [], self.latent_dim
        for d in decoder_dims:
            dec_layers.extend([nn.Linear(prev, d), nn.ReLU()])
            prev = d
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.memory_size = memory_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        attn = self.query_proj(z)
        attn_weights = F.softmax(attn, dim=-1)

        retrieved = attn_weights @ self.memory_keys
        recon = self.decoder(retrieved)
        return recon, z, attn_weights

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def _build_ae(name: str, input_dim: int, latent_dim: int,
               encoder_dims: List[int], decoder_dims: List[int]) -> nn.Module:
    """工厂函数：根据名称构造 AE 实例。"""
    if name == "SimpleAE":
        enc_d, dec_d = [8], [8]
        return FlexibleAutoEncoder(input_dim, enc_d, dec_d)
    if name == "FlexibleAE":
        enc_d, dec_d = [16, 8, 4], [8, 16]
        return FlexibleAutoEncoder(input_dim, enc_d, dec_d)
    if name == "DeepAE":
        enc_d, dec_d = [32, 16, 8], [16, 32]
        return FlexibleAutoEncoder(input_dim, enc_d, dec_d)
    if name == "Conv1DAE":
        return Conv1DAutoEncoder(
            window_size=EXP3.conv1d_window_size,
            channels=EXP3.conv1d_channels,
            latent_dim=latent_dim,
            input_dim=input_dim,
        )
    if name == "VAE":
        enc_d = [16, 8, latent_dim]
        dec_d = [8, 16]
        return VAutoEncoder(input_dim, enc_d, dec_d, latent_dim,
                            beta=EXP3.vae_beta)
    if name == "MemAE":
        enc_d = [32, 16, latent_dim]
        dec_d = [16, 32]
        return MemoryAutoEncoder(input_dim, enc_d, dec_d,
                                 memory_size=EXP3.memae_memory_size,
                                 latent_dim=latent_dim)
    raise ValueError(f"Unknown AE: {name}")


# ── 数据加载 ─────────────────────────────────────────────────────────────────

def _vib_cols(device: str) -> List[str]:
    return get_feature_cols(device)


def _load_monitoring_dates(device: str) -> List[str]:
    dates = []
    for date in get_train_dates():
        if get_feature_data_file(device, date).exists() and get_label_file(device, date).exists():
            dates.append(date)
    return sorted(dates)


def _load_monitoring_day(device: str, date: str) -> Tuple[np.ndarray, np.ndarray]:
    cols = _vib_cols(device)
    feat_df = pd.read_csv(get_feature_data_file(device, date))
    assert all(c in feat_df.columns for c in cols), (
        f"Missing feature columns for {device} {date}: "
        f"{[c for c in cols if c not in feat_df.columns]}"
    )
    feat_df = feat_df[cols]
    lab_df = pd.read_csv(get_label_file(device, date))
    # 删除含 NaN 的行
    nan_mask = feat_df.isna().any(axis=1)
    if nan_mask.sum() > 0:
        feat_df = feat_df.dropna()
        lab_df = lab_df.loc[feat_df.index]
    state_col = "state" if "state" in lab_df.columns else "label"
    n = min(len(feat_df), len(lab_df))
    return feat_df.iloc[:n].to_numpy(), lab_df[state_col].iloc[:n].astype(int).to_numpy()


def _load_exp2_scenarios(device: str) -> List[Dict]:
    import re
    device_dir = get_active_fault_data_dir() / device
    if not device_dir.exists():
        return []
    scenarios = []
    for pf in sorted(device_dir.glob("*_params.json")):
        base = pf.name.replace("_params.json", "")
        m = re.match(r"^([A-Za-z0-9_-]+?)_(crack|imbalance|mechanical_looseness|thermal)_(pct\d+)_(sev\d+)$", base)
        if not m:
            continue
        date_str, fault_type, pct_str, sev_str = m.groups()
        try:
            pct = int(pct_str.replace("pct", "")) / 100.0
            sev = int(sev_str.replace("sev", ""))
        except ValueError:
            continue
        scenarios.append({
            "name": base, "date": date_str, "fault_type": fault_type,
            "pct": pct, "severity": sev,
            "data_file": str(device_dir / f"{base}_data.parquet"),
            "labels_file": str(device_dir / f"{base}_labels.parquet"),
        })
    return scenarios


def _build_sliding_windows(x: np.ndarray, window_size: int) -> np.ndarray:
    """将 (N, 3) 的时序数据转为 (N, window_size, 3) 的窗口数据。"""
    n = len(x)
    pad_left = window_size // 2
    pad_right = window_size // 2
    padded = np.pad(x, ((pad_left, pad_right), (0, 0)), mode="edge")
    windows = np.zeros((n, window_size, x.shape[1]), dtype=np.float32)
    for i in range(n):
        windows[i] = padded[i:i + window_size]
    return windows


def _build_dataset(fast: bool = False, full_eval: bool = False) -> Exp3DataBundle:
    device = COMMON.device
    all_dates = _load_monitoring_dates(device)
    if not all_dates:
        raise RuntimeError(f"未找到 {device} 的特征与语义标签")
    train_set = train_dates_set()
    dates = [d for d in all_dates if d in train_set]
    if fast:
        dates = dates[:EXP3.fast_num_days]

    vib = _vib_cols(device)

    daily_x, daily_y, daily_dates = [], [], []
    all_healthy: List[np.ndarray] = []

    for d in dates:
        x, y = _load_monitoring_day(device, d)
        daily_x.append(x.astype(np.float32))
        daily_y.append(y.astype(np.int32))
        daily_dates.append(d)
        mask = (y == EXP3.normal_state_id)
        if mask.sum() > 0:
            all_healthy.append(x[mask])

    if not all_healthy:
        raise RuntimeError(f"未找到 NORMAL 状态（state_id={EXP3.normal_state_id}）")

    x_train_healthy_full = np.vstack(all_healthy).astype(np.float32)

    # 下采样加速（上限从 config 可配置）
    if EXP3.max_train_healthy > 0 and len(x_train_healthy_full) > EXP3.max_train_healthy:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_train_healthy_full), size=EXP3.max_train_healthy, replace=False)
        x_train_healthy = x_train_healthy_full[idx]
    else:
        x_train_healthy = x_train_healthy_full

    train_labels = np.ones(len(x_train_healthy), dtype=np.int32)

    # 构造时序窗口数据（用于 Conv1DAE）
    x_windowed = _build_sliding_windows(x_train_healthy, EXP3.conv1d_window_size)

    # 故障数据
    scenarios = _load_exp2_scenarios(device)
    # ── 可配置的评估场景选择 ──────────────────────────────────────────────────
    # 按 config 中的 eval_severity_levels 和 eval_pct_values 筛选，
    # 再从每种 (fault_type, severity, pct) 组合中均匀选不同日期的场景。
    if not fast:
        valid_sevs = set(EXP3.eval_severity_levels)
        valid_pcts = set(EXP3.eval_pct_values)
        # 按 (fault_type, severity, pct) 分组，每组再按日期均匀采样
        by_combo: Dict[Tuple[str, int, float], list] = {}
        for s in scenarios:
            if s["severity"] not in valid_sevs or s["pct"] not in valid_pcts:
                continue
            key = (s["fault_type"], s["severity"], s["pct"])
            by_combo.setdefault(key, []).append(s)

        selected = []
        for combo, clist in sorted(by_combo.items()):
            # 从不同日期中均匀采样（保证多样性）
            clist_sorted = sorted(clist, key=lambda x: x["date"])
            n = min(len(clist_sorted), EXP3.max_scenarios_per_combination)
            seed_payload = (
                f"{EXP3.scenario_sampling_seed}|{combo[0]}|"
                f"{combo[1]}|{combo[2]:.8f}"
            ).encode("utf-8")
            combo_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "little") % (2 ** 32)
            combo_rng = np.random.default_rng(combo_seed)
            chosen_idx = np.sort(combo_rng.choice(len(clist_sorted), size=n, replace=False))
            selected.extend(clist_sorted[int(i)] for i in chosen_idx)
        scenarios = selected
    elif len(scenarios) > 0:
        # fast 模式：每种故障类型各取 1 个场景（尽量不同日期）
        seen, selected = set(), []
        for s in sorted(scenarios, key=lambda x: x["date"]):
            if s["fault_type"] not in seen:
                selected.append(s)
                seen.add(s["fault_type"])
            if len(seen) >= 4:
                break
        scenarios = selected

    fault_x_list, fault_labels_list, scenario_names, fault_meta = [], [], [], []
    max_scorable = EXP3.max_scorable_points if not full_eval else 0
    rng = np.random.default_rng(777)

    for sc in scenarios:
        try:
            df = pd.read_parquet(sc["data_file"], engine="fastparquet")
            lf = (pd.read_parquet(sc["labels_file"], engine="fastparquet")
                  if sc["labels_file"].endswith(".parquet") else pd.read_csv(sc["labels_file"]))
            assert all(c in df.columns for c in vib), (
                f"Missing fault feature columns in {sc['name']}: "
                f"{[c for c in vib if c not in df.columns]}"
            )
            vib_df = df[vib].copy()
            # 删除含 NaN 的行（故障数据中偶尔有极少数损坏点）
            nan_mask = vib_df.isna().any(axis=1)
            if nan_mask.sum() > 0:
                print(f"[警告] {sc['name']} 有 {nan_mask.sum()} 行含 NaN，已删除", flush=True)
                vib_df = vib_df.dropna()
                lf = lf.loc[vib_df.index]
            x_fault_full = vib_df.values.astype(np.float32)
            fl_full = (lf["fault_label"].values[:len(x_fault_full)].astype(np.float32)
                       if "fault_label" in lf.columns
                       else np.zeros(len(x_fault_full), np.float32))
            state_full = (lf["state"].values[:len(x_fault_full)].astype(np.int32)
                          if "state" in lf.columns else None)
            eval_dropped = 0
            if (getattr(EXP3, "filter_eval_negative_to_normal_state", True)
                    and state_full is not None):
                eval_mask = (
                    (fl_full == 1)
                    | ((fl_full == 0) & (state_full == EXP3.normal_state_id))
                )
                eval_dropped = int((~eval_mask).sum())
                x_fault_full = x_fault_full[eval_mask]
                fl_full = fl_full[eval_mask]
                state_full = state_full[eval_mask]
            if len(np.unique(fl_full.astype(int))) < 2:
                print(f"[警告] {sc['name']} 评估过滤后缺少正/负类，已跳过 "
                      f"(kept={len(fl_full)}, dropped={eval_dropped})",
                      flush=True)
                continue
            if max_scorable > 0 and len(x_fault_full) > max_scorable:
                # 保守：对所有AE类型都做窗口安全检查（conv1d_window_size=7）
                x_fault, fl = _stratified_sequential_sample(
                    x_fault_full, fl_full,
                    max_total=max_scorable,
                    rng=rng,
                )
            else:
                x_fault, fl = x_fault_full, fl_full
            fault_x_list.append(x_fault)
            fault_labels_list.append(fl)
            scenario_names.append(sc["name"])
            fault_meta.append({
                "name": sc["name"], "date": sc["date"],
                "fault_type": sc["fault_type"],
                "severity": sc["severity"], "pct": sc["pct"],
                "eval_kept": int(len(x_fault_full)),
                "eval_dropped_non_normal_negative": eval_dropped,
            })
        except Exception as e:
            print(f"[警告] 加载 {sc['name']} 失败: {e}")
            continue

    if not fault_x_list:
        raise RuntimeError("Exp2 故障数据为空，请先运行 02_exp2_fault_injection.py")

    print(f"[Exp3] 健康样本={len(x_train_healthy)} | 天数={len(daily_x)} | "
          f"故障场景={len(fault_x_list)} | 窗口={EXP3.conv1d_window_size}", flush=True)

    return Exp3DataBundle(
        x_train_healthy=x_train_healthy,
        train_labels=train_labels,
        daily_x=daily_x, daily_y=daily_y, daily_dates=daily_dates,
        x_train_windowed=x_windowed,
        fault_x=fault_x_list,
        fault_labels=fault_labels_list,
        fault_scenarios=scenario_names,
        fault_meta=fault_meta,
    )


# ── AE 训练 ──────────────────────────────────────────────────────────────────

def _ae_forward(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """根据模型类型执行 forward，返回 (recon_x, ...)。"""
    if isinstance(model, VAutoEncoder):
        recon, mu, logvar = model(x)
        return recon, mu, logvar
    if isinstance(model, MemoryAutoEncoder):
        recon, z, attn = model(x)
        return recon, z, attn
    if isinstance(model, Conv1DAutoEncoder):
        recon = model(x)
        return recon,
    recon = model(x)
    return recon,


def _train_ae(
    x: np.ndarray,
    x_windowed: Optional[np.ndarray],
    ae_name: str,
    latent_dim: int,
    epochs: int,
    seed: int,
) -> Tuple[nn.Module, float]:
    """
    训练 AE，返回 (模型, 最终损失)。
    VAE 使用 ELBO loss：MSE + β * KL divergence。
    MemAE 使用 MSE + attention entropy 正则。
    """
    _set_seed(seed)
    input_dim = x.shape[1]
    enc_d, dec_d = _ae_structure(ae_name, latent_dim)
    model = _build_ae(ae_name, input_dim, latent_dim, enc_d, dec_d)
    model.to(DEVICE)
    print(f"[Train] {ae_name} lat={latent_dim} ep={epochs} seed={seed} start", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=EXP3.learning_rate,
                            weight_decay=EXP3.weight_decay)

    x_tensor = (torch.tensor(x_windowed, dtype=torch.float32).to(DEVICE)
                 if ae_name == "Conv1DAE"
                 else torch.tensor(x, dtype=torch.float32).to(DEVICE))
    effective_batch = min(EXP3.batch_size, len(x_tensor))
    loader = torch.utils.data.DataLoader(
        x_tensor, batch_size=effective_batch, shuffle=True, drop_last=True
    )

    best_loss, best_state, patience_counter = float("inf"), None, 0
    t0 = time.perf_counter()
    print(f"[Train] {ae_name} lat={latent_dim} ep={epochs} seed={seed} "
          f"batches={len(loader)} device={DEVICE}", flush=True)

    for ep in range(epochs):
        model.train()
        epoch_losses = []
        for xb in loader:
            xb = xb.to(DEVICE)
            opt.zero_grad()

            if isinstance(model, VAutoEncoder):
                recon, mu, logvar = model(xb)
                recon_loss = F.mse_loss(recon, xb)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + model.beta * kl_loss
            elif isinstance(model, MemoryAutoEncoder):
                recon, z, attn = model(xb)
                recon_loss = F.mse_loss(recon, xb)
                entropy = torch.mean(-attn * torch.log(attn + 1e-8))
                loss = recon_loss + 0.01 * entropy
            else:
                recon = model(xb)
                loss = F.mse_loss(recon, xb)

            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        mean_loss = np.mean(epoch_losses)
        elapsed = time.perf_counter() - t0
        improved = mean_loss < best_loss * (1 - EXP3.early_stopping_threshold)
        if improved:
            best_loss = mean_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"[Train] {ae_name} ep={ep+1}/{epochs} loss={mean_loss:.6f} "
              f"pat={patience_counter}/{EXP3.early_stopping_patience} "
              f"elapsed={elapsed:.0f}s", flush=True)

        if patience_counter >= EXP3.early_stopping_patience:
            print(f"[Train] {ae_name} early stop at ep {ep+1}", flush=True)
            break

        # 超时保护（正式模式每个 job 最多 10 分钟）
        if elapsed > 600 and not EXP3.full_evaluation:
            print(f"[Train] {ae_name} timeout at {elapsed:.0f}s", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_loss)


def _ae_structure(name: str, latent_dim: int) -> Tuple[List[int], List[int]]:
    """返回 (encoder_dims, decoder_dims)。"""
    if name == "SimpleAE":
        return [8], [8]
    if name == "FlexibleAE":
        return [16, 8, 4], [8, 16]
    if name == "DeepAE":
        return [32, 16, 8], [16, 32]
    if name == "Conv1DAE":
        return [], []
    if name == "VAE":
        return [16, 8, latent_dim], [8, 16]
    if name == "MemAE":
        return [32, 16, latent_dim], [16, 32]
    raise ValueError(f"Unknown AE: {name}")


def _stratified_sequential_sample(
    x_full: np.ndarray,
    fl_full: np.ndarray,
    max_total: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    时序分层顺序采样（替代随机均匀采样）。

    设计原则：
    ① 保留时序顺序，不打乱——Conv1DAE 和时序平滑依赖时间依赖性
    ② 正常段和故障段都要有样本，比例与原数据一致
    ③ 段内均匀采样（避免只采边界）
    ④ 无放回采样

    Args:
        x_full:    (N, 3) 原始振动数据
        fl_full:   (N,)  故障标签（0=正常, 1=故障）
        max_total: 最大采样总数
        rng:       随机数生成器

    Returns:
        (x_sample, fl_sample) — 时序有序，无重复
    """
    N = len(x_full)
    if N <= max_total:
        return x_full, fl_full

    # Step 1: 找出所有连续段的起止索引
    labels = fl_full.astype(int)
    boundaries = np.diff(labels, prepend=labels[0])
    segment_starts = np.where(boundaries != 0)[0]
    segment_ends = np.append(segment_starts[1:], N)

    segments = list(zip(segment_starts, segment_ends))
    # 合并相邻同标签段（diff 会在每个段内部跳变点触发，导致过碎）
    merged = []
    for start, end in segments:
        if not merged or merged[-1][1] != start or merged[-1][2] != labels[start]:
            merged.append([start, end, labels[start]])
        else:
            merged[-1][1] = end
    segments = [(s, e, l) for s, e, l in merged]

    # Step 2: 计算每段应采多少（按长度比例分层）
    seg_lengths = np.array([e - s for s, e, _ in segments])
    seg_labels = np.array([l for _, _, l in segments])

    # 目标：总采样 max_total，段内按比例分配（向下取整保下限）
    raw_counts = (seg_lengths / seg_lengths.sum() * max_total).astype(int)
    # 保证每段至少 1 个（如果段长度 > 0）
    raw_counts = np.maximum(raw_counts, 1)
    # 调整：总数可能超过 max_total，逐个从最短段扣
    while raw_counts.sum() > max_total:
        idx = int(np.argmin(raw_counts / np.maximum(seg_lengths, 1)))
        if raw_counts[idx] <= 1:
            break
        raw_counts[idx] -= 1
    # 可能还不够（所有段都被压到1个），直接截断
    if raw_counts.sum() > max_total:
        raw_counts[raw_counts.sum() - max_total:] -= 1

    # Step 3: 每段内均匀采样（避免只采段首/段尾）
    selected_indices = []
    for (start, end, label), count in zip(segments, raw_counts):
        seg_len = end - start
        if count >= seg_len:
            idx_in_seg = np.arange(start, end)
        else:
            # 均匀位置 + 小随机扰动（保持顺序）
            positions = np.linspace(0, seg_len - 1, count)
            jitter = rng.uniform(-0.4, 0.4, size=count)
            raw_idx = np.round(positions + jitter).astype(int)
            raw_idx = np.clip(raw_idx, 0, seg_len - 1)
            idx_in_seg = start + np.sort(np.unique(raw_idx))
        selected_indices.append(idx_in_seg)

    # 合并所有段（已经是时序有序）
    all_idx = np.concatenate(selected_indices)

    return x_full[all_idx], fl_full[all_idx]


# ── 推理（编码 / 重构）───────────────────────────────────────────────────────

@torch.no_grad()
@torch.no_grad()
def _encode(model: nn.Module, x: np.ndarray, x_windowed: Optional[np.ndarray],
            ae_name: str) -> np.ndarray:
    model.eval()
    if ae_name == "Conv1DAE":
        if x_windowed is None:
            xw = _build_sliding_windows(x, EXP3.conv1d_window_size)
        else:
            xw = x_windowed
        xw_t = torch.tensor(xw, dtype=torch.float32).to(DEVICE)
        z = model.encode(xw_t).cpu().numpy()
    else:
        x_t = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        z = model.encode(x_t).cpu().numpy()
    # 防御性：推理时不应有 NaN
    if np.any(np.isnan(z)):
        raise ValueError(f"{ae_name} encode() 返回 NaN，shape={z.shape}，输入x有NaN={np.any(np.isnan(x))}")
    return z


@torch.no_grad()
def _recon(model: nn.Module, x: np.ndarray, x_windowed: Optional[np.ndarray],
           ae_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (recon_x, recon_err_per_sample)。
    Conv1DAE: 重构是窗口形式 (N, 7, 3)，取中间帧作为点重构，残差取 RMS 平均到 (N, 3)。
    """
    model.eval()
    if ae_name == "Conv1DAE" and x_windowed is not None:
        xw_t = torch.tensor(x_windowed, dtype=torch.float32).to(DEVICE)
        recon_window = model(xw_t).cpu().numpy()       # (N, 7, 3)
        center = x_windowed.shape[1] // 2
        # 取窗口中心点作为"该时刻的重构"
        recon_point = recon_window[:, center, :]         # (N, 3)
        # 残差 RMS 按时间维度平均
        residual_sq = (recon_window - x_windowed) ** 2   # (N, 7, 3)
        recon_err = np.sqrt(np.mean(residual_sq, axis=1).mean(axis=1))  # (N,)
    else:
        x_t = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        out = model(x_t)
        recon_t = out[0].cpu() if isinstance(out, tuple) else out.cpu()
        recon_point = recon_t.numpy()
        recon_err = np.mean((recon_point - x) ** 2, axis=1)  # (N,)
    return recon_point, recon_err


# ── 时序平滑 ────────────────────────────────────────────────────────────────

def _window_smooth(z: np.ndarray, radius: int) -> np.ndarray:
    """带半径 radius 的滑动均值。clamp 保证不越界，兼容小数据集（< radius 行）。"""
    if radius <= 0:
        return z
    n = len(z)
    out = np.empty_like(z)
    half = radius
    for i in range(n):
        l = max(0, i - half)
        r = min(n, i + half + 1)
        out[i] = np.mean(z[l:r])
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = z[bad]
    return out


# ── 基线池 / 基准 ────────────────────────────────────────────────────────────

def _build_baseline(z: np.ndarray, method: str, seed: int) -> object:
    """
    工厂函数：返回不同的基准对象（可以是聚类模型、LOF 等）。
    返回值根据 method 不同而不同，后续用 _dist_to_baseline 统一计算距离。
    """
    if method.startswith("KMeans_K"):
        k = int(method.split("_K")[1])
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km.fit(z)
        return km  # .cluster_centers_ → (k, d)

    if method.startswith("GMM_K"):
        k = int(method.split("_K")[1])
        try:
            gmm = GaussianMixture(n_components=k, random_state=seed,
                                   reg_covar=1e-4, n_init=5)
            gmm.fit(z)
        except ValueError:
            gmm = GaussianMixture(n_components=min(k, 2), random_state=seed,
                                   reg_covar=1e-3)
            gmm.fit(z)
        return gmm  # .means_ → (k, d)

    if method == "DBSCAN":
        z_fit = z
        if len(z) > EXP3.dbscan_max_points:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(z), size=EXP3.dbscan_max_points, replace=False)
            z_fit = z[idx]
        db = DBSCAN(eps=EXP3.dbscan_eps, min_samples=EXP3.dbscan_min_samples)
        db.fit(z_fit)
        centers: List[np.ndarray] = []
        for c in sorted(set(db.labels_)):
            if c == -1:
                continue
            centers.append(z_fit[db.labels_ == c].mean(axis=0))
        if not centers:
            return z.mean(axis=0, keepdims=True)
        return np.vstack(centers)

    if method == "LOF":
        # LOF 在大规模训练集上 O(n²) 构建 BallTree，必须采样
        lof_max_train = 10000
        rng = np.random.default_rng(seed)
        z_fit = z if len(z) <= lof_max_train else z[rng.choice(len(z), size=lof_max_train, replace=False)]
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination='auto')
        lof.fit(z_fit)
        return lof

    raise ValueError(f"Unknown baseline method: {method}")


def _dist_to_baseline(z: np.ndarray, baseline: object,
                       method: str) -> np.ndarray:
    """
    统一接口：对任意基准方法，计算每个样本的异常分数。
    返回 (N,) 数组，越大越异常。
    """
    if method.startswith("KMeans_K"):
        centers = baseline.cluster_centers_
        diff = z[:, None, :] - centers[None, :, :]
        return np.min(np.sqrt(np.sum(diff * diff, axis=2)), axis=1)

    if method.startswith("GMM_K"):
        means = baseline.means_
        covars = baseline.covariances_
        dists = np.zeros((len(z), len(means)))
        for k in range(len(means)):
            cov = covars[k] + 1e-4 * np.eye(covars[k].shape[0])
            try:
                L = np.linalg.cholesky(cov)
                diff_k = z - means[k]
                solved = solve_triangular(L, diff_k.T, check_finite=False)
                dists[:, k] = np.sum(solved.T ** 2, axis=1)
                if np.any(np.isnan(dists[:, k])) or np.any(np.isinf(dists[:, k])):
                    dists[:, k] = np.sum((z - means[k]) ** 2, axis=1)
            except (np.linalg.LinAlgError, ValueError):
                dists[:, k] = np.sum((z - means[k]) ** 2, axis=1)
        return np.sqrt(np.min(dists, axis=1))

    if method == "DBSCAN":
        centers = baseline if isinstance(baseline, np.ndarray) else np.array([baseline])
        diff = z[:, None, :] - centers[None, :, :]
        return np.min(np.sqrt(np.sum(diff * diff, axis=2)), axis=1)

    if method == "LOF":
        # LOF 的 score_samples 返回负的 LOF 值（越大越异常）
        # sklearn LocalOutlierFactor.negative_outlier_factor_ 就是 LOF 本身
        # novelty=True 时 score_samples 返回 -LOF，转换为正数异常分数
        raw = baseline.score_samples(z)
        return -raw  # 转为越大越异常

    raise ValueError(f"Unknown baseline method: {method}")


def _build_covariance(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建协方差矩阵（用于 Mahalanobis 距离）。
    返回 (mean, precision) — precision = Σ⁻¹
    """
    try:
        ec = EmpiricalCovariance().fit(z)
        precision = ec.precision_
        if np.any(np.isnan(precision)) or np.any(np.isinf(precision)):
            raise np.linalg.LinAlgError("EmpiricalCovariance precision has NaN/Inf")
    except np.linalg.LinAlgError:
        cov = np.cov(z.T) + 1e-4 * np.eye(z.shape[1])
        precision = np.linalg.inv(cov)
    mean = np.mean(z, axis=0)
    return mean, precision


def _mahalanobis(z: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """计算每个样本到 mean 的马氏距离。"""
    diff = z - mean
    left = diff @ precision
    mahal = np.sqrt(np.sum(left * diff, axis=1))
    # 安全网：若有 NaN/Inf 则降级为欧氏距离
    if np.any(np.isnan(mahal)) or np.any(np.isinf(mahal)):
        mahal = np.sqrt(np.sum(diff ** 2, axis=1))
    return mahal


# ── Ours 评分函数 ────────────────────────────────────────────────────────────

def _score_ours_a(
    dist_lat: np.ndarray, recon_err: np.ndarray,
    ref_lat_mean: float, ref_lat_std: float,
    ref_recon_mean: float, ref_recon_std: float,
    alpha: float,
) -> np.ndarray:
    """Ours-A: α * dist_latent + (1-α) * recon_err，z-score 归一化后加权。"""
    dist_lat_n = (dist_lat - ref_lat_mean) / (ref_lat_std + 1e-12)
    recon_n = (recon_err - ref_recon_mean) / (ref_recon_std + 1e-12)
    return alpha * dist_lat_n + (1 - alpha) * recon_n


def _score_ours_b(
    dist_lat: np.ndarray, dist_res: np.ndarray,
    ref_lat_mean: float, ref_lat_std: float,
    ref_res_mean: float, ref_res_std: float,
    alpha: float,
) -> np.ndarray:
    """Ours-B: α * dist_latent + (1-α) * dist_residual，z-score 归一化后加权。"""
    dist_lat_n = (dist_lat - ref_lat_mean) / (ref_lat_std + 1e-12)
    dist_res_n = (dist_res - ref_res_mean) / (ref_res_std + 1e-12)
    return alpha * dist_lat_n + (1 - alpha) * dist_res_n


def _score_ours_ab(
    dist_lat: np.ndarray, recon_err: np.ndarray, dist_res: np.ndarray,
    ref_lat_mean: float, ref_lat_std: float,
    ref_recon_mean: float, ref_recon_std: float,
    ref_res_mean: float, ref_res_std: float,
) -> np.ndarray:
    """
    Ours-A+B: Ours-A 和 Ours-B 的等权平均，
    让 latent 距离和残差距离同时贡献信息。
    """
    s_a = _score_ours_a(dist_lat, recon_err,
                         ref_lat_mean, ref_lat_std, ref_recon_mean, ref_recon_std, alpha=0.5)
    s_b = _score_ours_b(dist_lat, dist_res,
                         ref_lat_mean, ref_lat_std, ref_res_mean, ref_res_std, alpha=0.5)
    return 0.5 * s_a + 0.5 * s_b


def _score_metadata(variant_name: str) -> Dict[str, object]:
    """Paper-facing score metadata. Avoid writing the ambiguous legacy alpha field."""
    if variant_name == "Ours-A":
        return {
            "ours_variant": "Ours-A",
            "effective_w_latent": 0.30,
            "effective_w_reconstruction": 0.70,
            "effective_w_residual": 0.00,
            "score_family": "historical_latent_reconstruction",
        }
    if variant_name == "Ours-B":
        return {
            "ours_variant": "Ours-B",
            "effective_w_latent": 0.30,
            "effective_w_reconstruction": 0.00,
            "effective_w_residual": 0.70,
            "score_family": "historical_latent_residual",
        }
    return {
        "ours_variant": "dual_space",
        "effective_w_latent": 0.50,
        "effective_w_reconstruction": 0.25,
        "effective_w_residual": 0.25,
        "score_family": "dual_space_health_index",
    }


def _normalize_score_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize old/new result rows to the paper-facing score metadata columns."""
    if df is None or df.empty or "ours_variant" not in df.columns:
        return df
    df = df.copy()
    df = df.drop(columns=["alpha"], errors="ignore")
    meta = df["ours_variant"].map(_score_metadata).apply(pd.Series)
    for col in meta.columns:
        df[col] = meta[col].values
    return df


# ── 评估指标 ────────────────────────────────────────────────────────────────

def _fpr_at_tpr95(y: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, scores)
    valid = np.where(tpr >= 0.95)[0]
    return float(np.min(fpr[valid])) if len(valid) > 0 else float("nan")


def _calc_metrics(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(score)
    if valid.sum() == 0:
        return {"separation_ratio": float("nan"), "auc_roc": float("nan"),
                "fpr_95tpr": float("nan"), "pr_auc": float("nan"),
                "precision_10pct": float("nan"), "recall_10pct": float("nan"),
                "f1_threshold": float("nan")}
    y, score = y[valid], score[valid]
    unique = np.unique(y)
    if len(unique) < 2:
        return {"separation_ratio": float("nan"), "auc_roc": float("nan"),
                "fpr_95tpr": float("nan"), "pr_auc": float("nan"),
                "precision_10pct": float("nan"), "recall_10pct": float("nan"),
                "f1_threshold": float("nan")}

    hm, hf = y == 0, y == 1
    # sep（分离比）：故障样本均值/健康样本均值。因 z-score 归一化后量纲不确定，不适合跨架构比较，故移除。
    auc = float(roc_auc_score(y, score))
    fpr95 = _fpr_at_tpr95(y, score)
    pr_auc = float(average_precision_score(y, score))

    # Precision/Recall@10%：取分数最高的10%作为正类，看命中率
    n_top = max(1, int(len(score) * 0.10))
    top_k_idx = np.argsort(score)[-n_top:]
    top_k_y = y[top_k_idx]
    prec_at_10 = float(top_k_y.mean())  # top10%中真正故障比例
    rec_at_10 = float(top_k_y.sum() / max(1, y.sum()))  # 故障中被top10%覆盖的比例

    # 最佳 threshold（F1）
    fpr_arr, tpr_arr, thresholds = roc_curve(y, score)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    prec_arr = (tpr_arr * n_pos) / np.maximum(tpr_arr * n_pos + fpr_arr * n_neg, 1e-12)
    rec_arr = tpr_arr
    f1_arr = 2 * prec_arr * rec_arr / np.maximum(prec_arr + rec_arr, 1e-12)
    best_f1 = float(np.nanmax(f1_arr))

    return {
        "separation_ratio": float("nan"),  # 已废弃：跨架构量纲不可比
        "auc_roc": auc,
        "fpr_95tpr": fpr95,
        "pr_auc": pr_auc,
        "precision_10pct": prec_at_10,
        "recall_10pct": rec_at_10,
        "f1_threshold": best_f1,
    }


# ── 图表生成 ───────────────────────────────────────────────────────────────

def _make_figures(df: pd.DataFrame) -> None:
    dv = df[np.isfinite(df["auc_roc"])].copy()
    # sep 过滤已移除（该指标已废弃）

    if dv.empty:
        return

    # F6: AE × 基准热力图（AUC）
    pivot_auc = dv.pivot_table(index="ae_structure", columns="baseline_method",
                               values="auc_roc", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot_auc, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax,
                vmin=0.5, vmax=1.0, cbar_kws={"label": "AUC-ROC"})
    ax.set_title("Figure F6: AE × Baseline — AUC-ROC (mean)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PATHS.exp3_grid_dir / "exp3_fig_f6_heatmap_auc.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Exp3] F6 → {PATHS.exp3_grid_dir / 'exp3_fig_f6_heatmap_auc.png'}")

    # F6b: 按距离度量分组
    pivot_dist = dv.pivot_table(
        index="ae_structure", columns="distance_metric",
        values="auc_roc", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot_dist, annot=True, fmt=".3f", cmap="coolwarm", ax=ax,
                vmin=0.5, vmax=1.0, cbar_kws={"label": "AUC-ROC"})
    ax.set_title("Figure F6b: AE × Distance Metric — AUC-ROC (mean)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PATHS.exp3_grid_dir / "exp3_fig_f6b_heatmap_dist.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # F7: Ours 策略对比
    pivot_ours = dv.pivot_table(index="ae_structure", columns="ours_variant",
                                values="auc_roc", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot_ours, annot=True, fmt=".3f", cmap="viridis", ax=ax,
                vmin=0.5, vmax=1.0, cbar_kws={"label": "AUC-ROC"})
    ax.set_title("Figure F7: AE × Ours Variant — AUC-ROC (mean)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PATHS.exp3_grid_dir / "exp3_fig_f7_heatmap_ours.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # F8: 基准方法对比箱线图
    methods = sorted(dv["baseline_method"].unique())
    data_auc = [dv[dv["baseline_method"] == m]["auc_roc"].dropna().values for m in methods]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data_auc, tick_labels=methods, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Figure F8: Baseline Methods — AUC-ROC Distribution", fontsize=12)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.4, 1.05)
    plt.tight_layout()
    plt.savefig(PATHS.exp3_grid_dir / "exp3_fig_f8_boxplot.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Exp3] F8 → {PATHS.exp3_grid_dir / 'exp3_fig_f8_boxplot.png'}")


# ── 分段并行合并 ────────────────────────────────────────────────────────────

def _merge_segments() -> None:
    """从所有 GPU 段中间文件合并出完整结果（跑完后执行一次）"""
    out_dir = PATHS.exp3_grid_dir
    # 找所有中间文件
    segments = sorted(out_dir.glob("exp3_segment_*.csv"))
    if not segments:
        print("[Exp3] 未找到 segment 文件，请先在各 GPU 上跑完实验")
        return
    print(f"[Exp3] 找到 {len(segments)} 个段文件，开始合并...")

    dfs = []
    for seg in segments:
        seg_df = pd.read_csv(seg)
        seg_tag = seg.stem.replace("exp3_segment_", "")
        seg_df["_segment"] = seg_tag
        dfs.append(seg_df)
        print(f"  + {seg.name} ({len(seg_df)} rows)")

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[Exp3] 合并完成，共 {len(df_all)} rows")

    # 分离合并结果和 per-scenario 结果
    if "_segment" in df_all.columns and "scenario_name" in df_all.columns:
        df_sc_all = df_all[df_all["scenario_name"].notna() & (df_all["scenario_name"] != "")].copy()
        df_grid_all = df_all[df_all["scenario_name"].isna() | (df_all["scenario_name"] == "")].drop(
            columns=["_segment"], errors="ignore")
    else:
        df_sc_all = pd.DataFrame()
        df_grid_all = df_all.drop(columns=["_segment"], errors="ignore")

    # 重新调用 _save_final 完成所有聚合
    _save_final(df_grid_all.to_dict("records"), df_sc_all.to_dict("records"))


# ── 保存函数 ────────────────────────────────────────────────────────────────

def _save_final(rows, scenario_rows=None) -> None:
    # 支持两种调用方式：_save_final(list, list) 或 _save_final(df, df)
    if isinstance(rows, pd.DataFrame):
        df = rows
    else:
        df = pd.DataFrame(rows)
    if scenario_rows is None:
        df_sc = pd.DataFrame()
    elif isinstance(scenario_rows, pd.DataFrame):
        df_sc = scenario_rows
    else:
        df_sc = pd.DataFrame(scenario_rows)
    df = _normalize_score_metadata_columns(df)
    df_sc = _normalize_score_metadata_columns(df_sc)

    # ── 全量合并结果 ──────────────────────────────────────────
    df.to_csv(PATHS.exp3_grid_dir / "exp3_grid_results.csv",
              index=False, float_format="%.6f")

    # ── Per-scenario 详细结果（最细粒度）───────────────────────
    if len(df_sc) > 0:
        df_sc.to_csv(PATHS.exp3_grid_dir / "exp3_scenario_results.csv",
                     index=False, float_format="%.6f")
        print(f"[Exp3] scenario_results.csv → {len(df_sc)} rows", flush=True)

        agg_cols = dict(
            sc_auc_mean=("sc_auc_roc", "mean"), sc_auc_std=("sc_auc_roc", "std"),
            # sc_sep_mean 已废弃（跨架构量纲不可比）
            sc_fpr_mean=("sc_fpr_95tpr", "mean"),
            sc_pr_auc_mean=("sc_pr_auc", "mean"),
            sc_prec_top10pct_mean=("sc_precision_top10pct", "mean"),
            sc_rec_top10pct_mean=("sc_recall_top10pct", "mean"),
            sc_f1_best_mean=("sc_f1_best", "mean"),
            n_scenarios=("sc_auc_roc", "count"),
        )

        # ── 按 fault_type 聚合 ──────────────────────────────
        by_ft = df_sc.groupby(["ae_structure", "baseline_method", "distance_metric",
                               "window_w", "ours_variant", "fault_type"],
                              as_index=False).agg(**agg_cols).sort_values(
                                  "sc_auc_mean", ascending=False)
        by_ft.to_csv(PATHS.exp3_grid_dir / "exp3_by_fault_type.csv",
                     index=False, float_format="%.6f")

        # ── 按 pct 聚合 ─────────────────────────────────────
        by_pct = df_sc.groupby(["ae_structure", "baseline_method", "distance_metric",
                                 "window_w", "ours_variant", "pct"],
                                as_index=False).agg(**agg_cols).sort_values(
                                    "sc_auc_mean", ascending=False)
        by_pct.to_csv(PATHS.exp3_grid_dir / "exp3_by_pct.csv",
                      index=False, float_format="%.6f")

        # ── 按 severity 聚合 ────────────────────────────────
        by_sev = df_sc.groupby(["ae_structure", "baseline_method", "distance_metric",
                                 "window_w", "ours_variant", "severity"],
                                as_index=False).agg(**agg_cols).sort_values(
                                    "sc_auc_mean", ascending=False)
        by_sev.to_csv(PATHS.exp3_grid_dir / "exp3_by_severity.csv",
                      index=False, float_format="%.6f")

        # ── 按 fault_type × pct 交叉聚合 ──────────────────
        by_ft_pct = df_sc.groupby(["ae_structure", "baseline_method", "distance_metric",
                                    "window_w", "ours_variant", "fault_type", "pct"],
                                   as_index=False).agg(**agg_cols).sort_values(
                                       "sc_auc_mean", ascending=False)
        by_ft_pct.to_csv(PATHS.exp3_grid_dir / "exp3_by_fault_type_pct.csv",
                         index=False, float_format="%.6f")

        print(f"[Exp3] 场景聚合完成 | by_fault_type={len(by_ft)} | "
              f"by_pct={len(by_pct)} | by_severity={len(by_sev)} | "
              f"by_ft_pct={len(by_ft_pct)}", flush=True)

    # ── 合并结果：按配置聚合（去 seed）────────────────────────
    gcols = ["ae_structure", "baseline_method", "distance_metric",
             "latent_dim", "window_w", "epochs", "ours_variant",
             "effective_w_latent", "effective_w_reconstruction",
             "effective_w_residual", "score_family"]
    summary = (
        df.groupby(gcols, as_index=False)
        .agg(
            # sep_mean/sep_std 已废弃（跨架构量纲不可比）
            auc_mean=("auc_roc", "mean"), auc_std=("auc_roc", "std"),
            fpr95_mean=("fpr_95tpr", "mean"), fpr95_std=("fpr_95tpr", "std"),
            loss_mean=("train_loss", "mean"), loss_std=("train_loss", "std"),
            time_mean=("train_sec", "mean"), time_std=("train_sec", "std"),
            n_runs=("seed", "count"),
        )
        .sort_values("auc_mean", ascending=False)
    )
    summary.to_csv(PATHS.exp3_grid_dir / "exp3_grid_summary.csv",
                   index=False, float_format="%.6f")

    # Top5（按 AUC）
    top5 = summary.head(5).copy()
    top5.insert(0, "rank", range(1, len(top5) + 1))
    top5.to_csv(PATHS.exp3_grid_dir / "exp3_top5_configs.csv",
                index=False, float_format="%.6f")

    best = top5.iloc[0].to_dict() if len(top5) > 0 else {}
    with open(PATHS.exp3_grid_dir / "exp3_best_config.json", "w",
              encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    _make_figures(df)
    print(f"[Exp3] 完成 | grid → exp3_grid_results.csv | "
          f"scenarios → exp3_scenario_results.csv ({len(df_sc)} rows) | "
          f"best → exp3_best_config.json", flush=True)


# ── 主循环 ──────────────────────────────────────────────────────────────────

def _segment_key_columns() -> List[str]:
    return ["ae_structure", "latent_dim", "epochs", "seed"]


def _segment_job_key_columns() -> List[str]:
    return (_segment_key_columns()
            + ["baseline_method", "distance_metric", "window_w"])


def _segment_row_frame(rows: List[dict], scenario_rows: List[dict],
                       segment_id: str) -> pd.DataFrame:
    df_seg = _normalize_score_metadata_columns(pd.DataFrame(rows))
    df_sc_seg = _normalize_score_metadata_columns(pd.DataFrame(scenario_rows))
    for col in ["scenario_name", "date", "fault_type", "pct", "severity",
                "sc_auc_roc", "sc_separation_ratio", "sc_fpr_95tpr",
                "sc_pr_auc", "sc_precision_top10pct", "sc_recall_top10pct",
                "sc_f1_best", "sc_n_total", "sc_n_fault", "sc_n_healthy",
                "sc_fault_pct"]:
        if col not in df_seg.columns:
            df_seg[col] = None
    df_seg["_segment"] = segment_id
    df_sc_seg["_segment"] = segment_id
    return pd.concat([df_seg, df_sc_seg], ignore_index=True)


def _load_completed_segment_tasks(seg_file: Path,
                                  expected_rows_per_train_task: int) -> set:
    if not seg_file.exists() or seg_file.stat().st_size == 0:
        return set()
    key_cols = _segment_key_columns()
    try:
        df_done = pd.read_csv(seg_file, usecols=lambda c: c in key_cols)
    except Exception as exc:
        print(f"[Exp3] warning: cannot read resume file {seg_file.name}: {exc}",
              flush=True)
        return set()
    if df_done.empty or any(c not in df_done.columns for c in key_cols):
        return set()
    counts = df_done.groupby(key_cols).size()
    completed = set()
    for key, count in counts.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        if int(count) >= expected_rows_per_train_task:
            completed.add((str(key_tuple[0]), int(key_tuple[1]),
                           int(key_tuple[2]), int(key_tuple[3])))
    return completed


def _load_completed_segment_jobs(seg_file: Path,
                                 expected_rows_per_job: int) -> set:
    if not seg_file.exists() or seg_file.stat().st_size == 0:
        return set()
    key_cols = _segment_job_key_columns()
    try:
        df_done = pd.read_csv(seg_file, usecols=lambda c: c in key_cols)
    except Exception as exc:
        print(f"[Exp3] warning: cannot read resume jobs from "
              f"{seg_file.name}: {exc}", flush=True)
        return set()
    if df_done.empty or any(c not in df_done.columns for c in key_cols):
        return set()
    counts = df_done.groupby(key_cols).size()
    completed = set()
    for key, count in counts.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        if int(count) >= expected_rows_per_job:
            completed.add((
                str(key_tuple[0]),
                int(key_tuple[1]),
                int(key_tuple[2]),
                int(key_tuple[3]),
                str(key_tuple[4]),
                str(key_tuple[5]),
                int(key_tuple[6]),
            ))
    return completed


def _sanitize_segment_checkpoint(seg_file: Path,
                                 expected_rows_per_job: int) -> None:
    if not seg_file.exists() or seg_file.stat().st_size == 0:
        return
    key_cols = _segment_job_key_columns()
    try:
        df_done = pd.read_csv(seg_file)
    except Exception as exc:
        print(f"[Exp3] warning: cannot sanitize {seg_file.name}: {exc}",
              flush=True)
        return
    if df_done.empty or any(c not in df_done.columns for c in key_cols):
        return

    kept = []
    dropped_rows = 0
    for _, group in df_done.groupby(key_cols, sort=False):
        if len(group) >= expected_rows_per_job:
            kept.append(group.head(expected_rows_per_job))
            dropped_rows += max(0, len(group) - expected_rows_per_job)
        else:
            dropped_rows += len(group)

    if dropped_rows == 0:
        return
    tmp_file = seg_file.with_name(seg_file.name + ".tmp")
    if kept:
        pd.concat(kept, ignore_index=True).to_csv(
            tmp_file, index=False, float_format="%.6f")
        os.replace(tmp_file, seg_file)
    else:
        if tmp_file.exists():
            tmp_file.unlink()
        seg_file.unlink()
    print(f"[Exp3] sanitized {seg_file.name}: dropped {dropped_rows} "
          "incomplete/duplicate rows", flush=True)


def _append_segment_checkpoint(seg_file: Path, rows: List[dict],
                               scenario_rows: List[dict],
                               segment_id: str) -> int:
    if not rows and not scenario_rows:
        return 0
    df_chunk = _segment_row_frame(rows, scenario_rows, segment_id)
    header = not seg_file.exists() or seg_file.stat().st_size == 0
    df_chunk.to_csv(seg_file, mode="a", header=header, index=False,
                    float_format="%.6f")
    return len(df_chunk)


def run_exp3(fast: bool = False, ae_filter: Optional[List[str]] = None,
              baseline_filter: Optional[List[str]] = None,
              gpu_id: int = 0, segment_tag: str = "",
              shard_index: int = 0, num_shards: int = 1) -> None:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards - 1}], got {shard_index}")
    tag = f"-GPU{gpu_id}" + (f"_{segment_tag}" if segment_tag else "")
    _set_device(gpu_id)
    device_name = (torch.cuda.get_device_name(DEVICE)
                   if DEVICE.type == "cuda" else "CPU")
    visible_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(f"[Exp3{tag}] local_gpu={gpu_id} visible_cuda={visible_cuda} "
          f"device={DEVICE} ({device_name}) | AE过滤={ae_filter} | "
          f"基准过滤={baseline_filter}", flush=True)

    PATHS.ensure_dirs()

    # ── 可复现性记录 ───────────────────────────────────────────────────────
    try:
        import subprocess, datetime
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
            cwd=Path(__file__).resolve().parent
        ).strip()[:8]
    except Exception:
        git_hash = "unknown"

    # ── 数据加载 ─────────────────────────────────────────────────────────
    t_data = time.perf_counter()
    bundle = _build_dataset(fast=fast, full_eval=EXP3.full_evaluation and not fast)
    print(f"[Exp3{tag}] 数据加载完成 {time.perf_counter()-t_data:.1f}s", flush=True)

    # 标准化（在原始点数据上做，Conv1DAE 窗口数据用同一个 scaler）
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(bundle.x_train_healthy).astype(np.float32)

    # 时序窗口数据（用标准化后的数据）
    t_win = time.perf_counter()
    x_train_windowed = _build_sliding_windows(
        x_train_scaled, EXP3.conv1d_window_size
    ).astype(np.float32)
    print(f"[Exp3{tag}] 窗口构建完成 {time.perf_counter()-t_win:.1f}s", flush=True)

    # ── 搜索维度 ─────────────────────────────────────────────────────────────
    if fast:
        ae_list = EXP3.fast_ae_structures
        baseline_list = EXP3.fast_baseline_methods
        dist_list = EXP3.fast_distance_metrics
        latent_dims = [4]
        # Exp3 selects the detector structure. Component-weight sensitivity is handled by Exp6.
        score_weight_profiles = ["historical_latent_reconstruction",
                                 "historical_latent_residual",
                                 "dual_space_health_index"]
        window_radii = EXP3.fast_window_radii
        epochs_list = EXP3.fast_epochs
        seeds = EXP3.fast_seeds
    else:
        ae_list = EXP3.ae_structures_extended
        baseline_list = EXP3.baseline_methods
        dist_list = EXP3.distance_metrics
        latent_dims = [4, 8]
        # Exp3 selects the detector structure. Component-weight sensitivity is handled by Exp6.
        score_weight_profiles = ["historical_latent_reconstruction",
                                 "historical_latent_residual",
                                 "dual_space_health_index"]
        window_radii = EXP3.window_radii
        epochs_list = EXP3.epochs
        seeds = COMMON.seeds

    # 过滤器（仅在精简版之后生效）
    if ae_filter:
        ae_list = [a for a in ae_filter if a in EXP3.ae_structures_extended]
        if not ae_list:
            raise ValueError(f"AE过滤器 {ae_filter} 与已知结构无交集")
    if baseline_filter:
        baseline_list = [b for b in baseline_list if b in baseline_filter]
        if not baseline_list:
            raise ValueError(f"基准过滤器 {baseline_filter} 与已知方法无交集")

    all_train_tasks = [
        (ae_name, latent_dim, epochs, seed)
        for ae_name in ae_list
        for latent_dim in latent_dims
        for epochs in epochs_list
        for seed in seeds
    ]
    train_tasks = [
        task for idx, task in enumerate(all_train_tasks)
        if idx % num_shards == shard_index
    ]
    if not train_tasks:
        raise RuntimeError(
            f"No Exp3 train tasks assigned to shard {shard_index}/{num_shards}. "
            f"Total train tasks={len(all_train_tasks)}."
        )
    train_task_set = set(train_tasks)
    n_train_all = len(all_train_tasks)
    n_train = len(train_tasks)
    n_infer = (len(baseline_list) * len(dist_list) * len(window_radii))
    n_variant = 3  # Ours-A, Ours-B, Ours-A+B
    job_total = n_train * n_infer
    total = job_total * n_variant
    print(f"[Exp3] 训练任务={n_train} | 推理组合={n_infer}×3(Ours变体) | 总结果={total}", flush=True)
    print(f"[Exp3] train_tasks={n_train}/{n_train_all} | infer_combinations={n_infer} | result_rows={total}", flush=True)
    if num_shards > 1:
        print(f"[Exp3{tag}] shard={shard_index}/{num_shards} assigned_train_tasks={n_train}", flush=True)
    print(f"[Exp3] AE={ae_list}", flush=True)
    print(f"[Exp3] 基准={baseline_list}", flush=True)
    print(f"[Exp3] 距离={dist_list}", flush=True)

    # ── 可复现性记录（所有搜索维度已确定后生成）────────────────────────
    segment_id = segment_tag if segment_tag else "gpu0"
    seg_file = PATHS.exp3_grid_dir / f"exp3_segment_{segment_id}.csv"
    expected_rows_per_job = n_variant * (1 + len(bundle.fault_meta))
    _sanitize_segment_checkpoint(seg_file, expected_rows_per_job)
    completed_jobs = _load_completed_segment_jobs(seg_file, expected_rows_per_job)
    train_job_counts = defaultdict(int)
    for job_key in completed_jobs:
        train_job_counts[job_key[:4]] += 1
    completed_train_tasks = {
        train_key for train_key, count in train_job_counts.items()
        if count >= n_infer
    }
    if completed_jobs:
        print(f"[Exp3{tag}] resume: found {len(completed_jobs)} "
              f"completed jobs in {seg_file.name}", flush=True)
    if completed_train_tasks:
        print(f"[Exp3{tag}] resume: found {len(completed_train_tasks)} "
              f"completed train tasks in {seg_file.name}", flush=True)

    try:
        import subprocess as _subprocess, datetime as _datetime
        _git_hash = _subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=_subprocess.DEVNULL, text=True,
            cwd=Path(__file__).resolve().parent
        ).strip()[:8]
    except Exception:
        _git_hash = "unknown"

    _record = {
        "exp": "Exp3",
        "git_hash": _git_hash,
        "timestamp": _datetime.datetime.now().isoformat(),
        "fast": fast,
        "ae_filter": ae_filter,
        "baseline_filter": baseline_filter,
        "seeds": list(seeds),
        "ae_structures": ae_list,
        "baseline_methods": baseline_list,
        "distance_metrics": dist_list,
        "latent_dims": latent_dims,
        "score_weight_profiles": score_weight_profiles,
        "window_radii": window_radii,
        "epochs_list": epochs_list,
        "scenario_sampling": "seeded_random_by_fault_severity_pct",
        "scenario_sampling_seed": EXP3.scenario_sampling_seed,
        "max_scenarios_per_combination": EXP3.max_scenarios_per_combination,
        "eval_severity_levels": list(EXP3.eval_severity_levels),
        "eval_pct_values": list(EXP3.eval_pct_values),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "n_train_tasks_all": n_train_all,
        "assigned_train_tasks": [
            {
                "ae_structure": task[0],
                "latent_dim": task[1],
                "epochs": task[2],
                "seed": task[3],
            }
            for task in train_tasks
        ],
        "n_train_tasks": n_train,
        "n_infer_per_task": n_infer,
        "n_total_results": total,
    }
    record_tag = segment_tag if segment_tag else f"gpu{gpu_id}"
    record_name = (
        f"exp3_run_record_{record_tag}.json"
        if segment_tag or num_shards > 1 else
        "exp3_run_record.json"
    )
    with open(PATHS.exp3_grid_dir / record_name, "w", encoding="utf-8") as _rf:
        json.dump(_record, _rf, ensure_ascii=False, indent=2)
    print(f"[Exp3{tag}] run record -> {record_name}  git={_git_hash}", flush=True)
    print(f"[Exp3{tag}] 可复现记录 → exp3_run_record.json  git={_git_hash}", flush=True)

    rows: List[dict] = []
    scenario_rows: List[dict] = []
    job = 0

    for ae_name in ae_list:
        is_conv = (ae_name == "Conv1DAE")

        for latent_dim in latent_dims:
            for epochs in epochs_list:
                for seed in seeds:
                    if (ae_name, latent_dim, epochs, seed) not in train_task_set:
                        continue
                    train_key = (ae_name, int(latent_dim), int(epochs), int(seed))
                    if train_key in completed_train_tasks:
                        job += n_infer
                        print(f"[Exp3{tag}] resume skip train task {train_key} "
                              f"job={job}/{job_total}", flush=True)
                        continue
                    t0 = time.perf_counter()

                    # ── 阶段1：训练AE（耗时部分，每个AE只跑一次）───────────────
                    model, train_loss = _train_ae(
                        x_train_scaled, x_train_windowed if is_conv else None,
                        ae_name, latent_dim, epochs, seed,
                    )
                    model.to(DEVICE)
                    model.eval()

                    # ── 预计算：一次性编码所有数据（共享给所有推理组合）────────
                    z_train = _encode(model, x_train_scaled,
                                       x_train_windowed if is_conv else None, ae_name)
                    recon_train, recon_err_train = _recon(
                        model, x_train_scaled,
                        x_train_windowed if is_conv else None, ae_name
                    )
                    r_train = (x_train_scaled - recon_train).astype(np.float32)
                    elapsed_train = time.perf_counter() - t0

                    # ── 预计算：一次性编码所有故障场景（每个AE只算一次）────────
                    cache_zf, cache_rf, cache_recon_f, cache_labels = [], [], [], []
                    for xf_raw, fl in zip(bundle.fault_x, bundle.fault_labels):
                        xf = scaler.transform(xf_raw).astype(np.float32)
                        xf_w = (_build_sliding_windows(xf, EXP3.conv1d_window_size).astype(np.float32)
                                if is_conv else None)
                        zf = _encode(model, xf, xf_w, ae_name)
                        recon_xf, recon_f = _recon(model, xf, xf_w, ae_name)
                        rf = (xf - recon_xf).astype(np.float32)
                        cache_zf.append(zf); cache_rf.append(rf)
                        cache_recon_f.append(recon_f); cache_labels.append(fl.astype(int))
                    print(f"[Exp3{tag}] {ae_name} 缓存完成({len(cache_zf)}个场景)，推理...", flush=True)

                    # ── 阶段2：批量推理（纯计算，共享z/r）─────────────────────
                    t_infer = time.perf_counter()
                    for dist_metric in dist_list:
                        for baseline_method in baseline_list:
                            for w in window_radii:
                                job_key = (
                                    ae_name, int(latent_dim), int(epochs), int(seed),
                                    baseline_method, dist_metric, int(w)
                                )
                                if job_key in completed_jobs:
                                    job += 1
                                    if job % 50 == 0 or job == 1:
                                        print(f"[Exp3{tag}] resume skip job "
                                              f"{job}/{job_total} {job_key}",
                                              flush=True)
                                    continue
                                rows_start = len(rows)
                                scenario_rows_start = len(scenario_rows)
                                # 对每个 (baseline, dist, window) 组合建基准
                                z_train_w = _window_smooth(z_train, w)
                                r_train_w = _window_smooth(r_train, w)
                                b_lat = _build_baseline(z_train_w, baseline_method, seed)
                                b_res = _build_baseline(r_train_w, baseline_method, seed)
                                m_lat, p_lat = _build_covariance(z_train_w)
                                m_res, p_res = _build_covariance(r_train_w)

                                # 训练集参考统计（用于z-score归一化）
                                if baseline_method == "LOF":
                                    ref_lat = np.ones(len(z_train_w))
                                else:
                                    ref_lat = _dist_to_baseline(z_train_w, b_lat, baseline_method)
                                ref_res = _dist_to_baseline(r_train_w, b_res, baseline_method)
                                ref_lat_mean = float(np.mean(ref_lat))
                                ref_lat_std = float(np.std(ref_lat) + 1e-12)
                                ref_res_mean = float(np.mean(ref_res))
                                ref_res_std = float(np.std(ref_res) + 1e-12)
                                ref_recon_mean = float(np.mean(recon_err_train))
                                ref_recon_std = float(np.std(recon_err_train) + 1e-12)

                                # ── 对所有故障场景批量评分 ─────────────────────────
                                all_scores_a, all_scores_b, all_scores_ab = [], [], []
                                all_labels = []

                                for i, (xf_raw, fl) in enumerate(
                                        zip(bundle.fault_x, bundle.fault_labels)):
                                    xf = scaler.transform(xf_raw).astype(np.float32)

                                    if is_conv:
                                        xf_w = _build_sliding_windows(
                                            xf, EXP3.conv1d_window_size).astype(np.float32)
                                    else:
                                        xf_w = None

                                    zf = _encode(model, xf, xf_w, ae_name)
                                    zf_w = _window_smooth(zf, w)
                                    recon_xf, recon_f = _recon(model, xf, xf_w, ae_name)
                                    rf = (xf - recon_xf).astype(np.float32)
                                    rf_w = _window_smooth(rf, w)

                                    try:
                                        if dist_metric == "euclidean":
                                            if baseline_method == "LOF":
                                                dl = -b_lat.score_samples(zf_w)
                                                dr = -b_res.score_samples(rf_w)
                                            else:
                                                dl = _dist_to_baseline(zf_w, b_lat, baseline_method)
                                                dr = _dist_to_baseline(rf_w, b_res, baseline_method)
                                        else:  # mahalanobis
                                            dl = _mahalanobis(zf_w, m_lat, p_lat)
                                            dr = _mahalanobis(rf_w, m_res, p_res)
                                    except Exception as e:
                                        # LOF/Mahalanobis 对某些数据可能数值不稳定，跳过该场景
                                        print(f"[Exp3{tag}] ⚠ 场景{i}({bundle.fault_meta[i].get('name','?')}) "
                                              f"{baseline_method}/{dist_metric} 失败: {e}，跳过", flush=True)
                                        continue

                                    # Historical A/B branches are kept as internal comparisons.
                                    # The paper-facing fused branch is recorded as dual_space
                                    # with explicit effective component weights.
                                    alpha = 0.3
                                    sa = _score_ours_a(dl, recon_f,
                                                       ref_lat_mean, ref_lat_std,
                                                       ref_recon_mean, ref_recon_std,
                                                       alpha)
                                    sb = _score_ours_b(dl, dr,
                                                       ref_lat_mean, ref_lat_std,
                                                       ref_res_mean, ref_res_std,
                                                       alpha)
                                    sab = _score_ours_ab(dl, recon_f, dr,
                                                         ref_lat_mean, ref_lat_std,
                                                         ref_recon_mean, ref_recon_std,
                                                         ref_res_mean, ref_res_std)
                                    all_scores_a.extend(sa.tolist())
                                    all_scores_b.extend(sb.tolist())
                                    all_scores_ab.extend(sab.tolist())
                                    all_labels.extend(fl.astype(int).tolist())

                                    # ── Per-scenario 详细指标 ──────────────────────────
                                    sc_meta = bundle.fault_meta[i]
                                    y_sc = fl.astype(int)
                                    for variant_name, sc_scores in [
                                        ("Ours-A", sa), ("Ours-B", sb), ("Ours-A+B", sab)
                                    ]:
                                        sc_metrics = _calc_metrics(y_sc, sc_scores)
                                        scenario_rows.append({
                                            "seed": seed,
                                            "ae_structure": ae_name,
                                            "baseline_method": baseline_method,
                                            "distance_metric": dist_metric,
                                            "latent_dim": latent_dim,
                                            "window_w": w,
                                            "epochs": epochs,
                                            **_score_metadata(variant_name),
                                            # ── 场景元信息 ────────────────────────────────
                                            "scenario_name": sc_meta.get("name", ""),
                                            "date": sc_meta.get("date", ""),
                                            "fault_type": sc_meta.get("fault_type", ""),
                                            "pct": sc_meta.get("pct", 0.0),
                                            "severity": sc_meta.get("severity", 0),
                                            # ── 场景级指标 ──────────────────────────────
                                            "sc_separation_ratio": float("nan"),  # 已废弃
                                            "sc_auc_roc": sc_metrics["auc_roc"],
                                            "sc_fpr_95tpr": sc_metrics["fpr_95tpr"],
                                            "sc_pr_auc": sc_metrics["pr_auc"],
                                            "sc_precision_top10pct": sc_metrics["precision_10pct"],
                                            "sc_recall_top10pct": sc_metrics["recall_10pct"],
                                            "sc_f1_best": sc_metrics["f1_threshold"],
                                            # ── 样本统计 ────────────────────────────────
                                            "sc_n_total": int(len(y_sc)),
                                            "sc_n_fault": int((y_sc == 1).sum()),
                                            "sc_n_healthy": int((y_sc == 0).sum()),
                                            "sc_fault_pct": float((y_sc == 1).mean()),
                                        })

                                # ── 计算指标并记录 ────────────────────────────────
                                y_all = np.array(all_labels, dtype=int)
                                for variant_name, scores in [
                                    ("Ours-A", all_scores_a),
                                    ("Ours-B", all_scores_b),
                                    ("Ours-A+B", all_scores_ab),
                                ]:
                                    metrics = _calc_metrics(y_all, np.array(scores))
                                    # 汇总所有场景的 fault_type/pct/severity 列表
                                    all_fault_types = [m.get("fault_type", "") for m in bundle.fault_meta]
                                    all_pcts = [m.get("pct", 0.0) for m in bundle.fault_meta]
                                    all_sevs = [m.get("severity", 0) for m in bundle.fault_meta]
                                    all_dates = [m.get("date", "") for m in bundle.fault_meta]
                                    rows.append({
                                        "seed": seed,
                                        "ae_structure": ae_name,
                                        "baseline_method": baseline_method,
                                        "distance_metric": dist_metric,
                                        "latent_dim": latent_dim,
                                        "window_w": w,
                                        "epochs": epochs,
                                        **_score_metadata(variant_name),
                                        # ── 场景元信息 ─────────────────────────────────
                                        "n_scenarios": len(bundle.fault_meta),
                                        "fault_types": "|".join(sorted(set(all_fault_types))),
                                        "pct_values": "|".join(str(p) for p in sorted(set(all_pcts))),
                                        "severity_values": "|".join(str(s) for s in sorted(set(all_sevs))),
                                        "dates": "|".join(sorted(set(all_dates))),
                                        # ── 合并指标 ───────────────────────────────────
                                        "separation_ratio": float("nan"),  # 已废弃
                                        "auc_roc": metrics["auc_roc"],
                                        "fpr_95tpr": metrics["fpr_95tpr"],
                                        "pr_auc": metrics["pr_auc"],
                                        "precision_top10pct": metrics["precision_10pct"],
                                        "recall_top10pct": metrics["recall_10pct"],
                                        "f1_best": metrics["f1_threshold"],
                                        # ── 训练统计 ───────────────────────────────────
                                        "train_loss": train_loss,
                                        "train_sec": elapsed_train,
                                        "infer_sec": time.perf_counter() - t_infer,
                                        "n_healthy_samples": len(x_train_scaled),
                                        "n_fault_samples": int((y_all == 1).sum()),
                                        "n_healthy_in_test": int((y_all == 0).sum()),
                                    })

                                job += 1
                                if job % 50 == 0 or job == 1:
                                    print(f"[Exp3{tag}] job {job}/{job_total} "
                                          f"ae={ae_name} bl={baseline_method} "
                                          f"dist={dist_metric} w={w}", flush=True)

                                chunk_rows = rows[rows_start:]
                                chunk_scenario_rows = scenario_rows[scenario_rows_start:]
                                written = _append_segment_checkpoint(
                                    seg_file, chunk_rows, chunk_scenario_rows,
                                    segment_id)
                                completed_jobs.add(job_key)
                                train_job_counts[train_key] += 1
                                if train_job_counts[train_key] >= n_infer:
                                    completed_train_tasks.add(train_key)
                                del rows[rows_start:]
                                del scenario_rows[scenario_rows_start:]
                                print(f"[Exp3{tag}] checkpoint job {job}/{job_total} "
                                      f"{job_key} -> {seg_file.name} "
                                      f"(+{written} rows)", flush=True)

    print(f"[Exp3{tag}] 所有job完成，确认分片文件...", flush=True)
    tail_rows = _append_segment_checkpoint(
        seg_file, rows, scenario_rows, segment_id)
    if tail_rows:
        print(f"[Exp3{tag}] final checkpoint append (+{tail_rows} rows)",
              flush=True)
    if seg_file.exists():
        try:
            saved_rows = sum(1 for _ in open(seg_file, "r", encoding="utf-8")) - 1
        except UnicodeDecodeError:
            saved_rows = sum(1 for _ in open(seg_file, "r", encoding="gbk",
                                             errors="ignore")) - 1
        print(f"[Exp3{tag}] 段文件已保存 → {seg_file.name} ({saved_rows} rows)",
              flush=True)
    else:
        print(f"[Exp3{tag}] warning: segment file was not created: {seg_file}",
              flush=True)
    print(f"[Exp3{tag}] 完成! 跑完后执行 --merge 合并所有段文件", flush=True)


# ── 命令行入口 ───────────────────────────────────────────────────────────────

def _parse_csv_ints(value: str) -> List[int]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        raise ValueError("--gpus must contain at least one GPU id")
    return [int(v) for v in items]


def run_exp3_parallel(
    fast: bool,
    ae_filter: Optional[List[str]],
    baseline_filter: Optional[List[str]],
    gpus: List[int],
    auto_merge: bool = True,
) -> None:
    if not gpus:
        raise ValueError("No GPUs specified for Exp3 parallel launch")

    PATHS.ensure_dirs()
    script = Path(__file__).resolve()
    mode_arg = "--fast" if fast else "--formal"
    workers = []

    print(f"[Exp3] launching {len(gpus)} workers on physical GPUs: {gpus}", flush=True)
    for shard_index, physical_gpu in enumerate(gpus):
        segment = f"gpu{physical_gpu}_shard{shard_index}of{len(gpus)}"
        cmd = [
            sys.executable,
            str(script),
            mode_arg,
            "--gpu_id", "0",
            "--segment", segment,
            "--shard_index", str(shard_index),
            "--num_shards", str(len(gpus)),
        ]
        if ae_filter:
            cmd.extend(["--ae_filter", ",".join(ae_filter)])
        if baseline_filter:
            cmd.extend(["--baseline_filter", ",".join(baseline_filter)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        env["PYTHONUNBUFFERED"] = "1"
        log_path = PATHS.exp3_grid_dir / f"exp3_segment_{segment}.log"
        log_fh = open(log_path, "w", encoding="utf-8")
        print(f"[Exp3] start {segment}: {' '.join(cmd)}", flush=True)
        print(f"[Exp3] log  {segment}: {log_path}", flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=str(script.parent),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        workers.append((segment, physical_gpu, proc, log_fh, log_path))

    failures = []
    for segment, physical_gpu, proc, log_fh, log_path in workers:
        return_code = proc.wait()
        log_fh.close()
        if return_code != 0:
            failures.append((segment, physical_gpu, return_code, log_path))
            print(f"[Exp3] FAILED {segment} gpu={physical_gpu} rc={return_code} log={log_path}", flush=True)
        else:
            print(f"[Exp3] DONE   {segment} gpu={physical_gpu} log={log_path}", flush=True)

    if failures:
        msg = "\n".join(
            f"{segment} gpu={gpu} rc={rc} log={log_path}"
            for segment, gpu, rc, log_path in failures
        )
        raise RuntimeError(f"Exp3 parallel workers failed:\n{msg}")

    if auto_merge:
        _merge_segments()
    else:
        print("[Exp3] auto merge disabled; run with --merge after all segments finish.", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EXP3: AE+基准网格搜索 v2（支持 Conv1DAE/VAE/MemAE + Mahalanobis + LOF）"
    )
    p.add_argument("--fast", action="store_true", help="快速验证")
    p.add_argument("--formal", action="store_true", help="正式全网格")
    p.add_argument("--gpu_id", type=int, default=0, help="GPU ID（默认0）")
    p.add_argument("--ae_filter", type=str, default=None,
                   help="逗号分隔的AE结构，如 Conv1DAE,VAE")
    p.add_argument("--baseline_filter", type=str, default=None,
                   help="逗号分隔的基准方法，如 KMeans_K3,LOF")
    p.add_argument("--segment", type=str, default="",
                   help="并行分段标签（区分各GPU中间文件）")
    p.add_argument("--merge", action="store_true",
                   help="从所有 segment 中间文件合并出最终结果（不跑实验）")
    p.add_argument("--parallel", action="store_true",
                   help="Launch one Exp3 worker per GPU and shard train tasks automatically.")
    p.add_argument("--gpus", type=str, default=None,
                   help="Comma-separated physical GPU ids for --parallel, e.g. 0,1,2,3.")
    p.add_argument("--shard_index", type=int, default=0,
                   help="Internal worker shard index. Used by --parallel child processes.")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Internal worker shard count. Used by --parallel child processes.")
    p.add_argument("--no_auto_merge", action="store_true",
                   help="With --parallel, do not merge segment CSV files after workers finish.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge:
        _merge_segments()
        return
    ae_filter = None
    if args.ae_filter:
        ae_filter = [a.strip() for a in args.ae_filter.split(",")]
    baseline_filter = None
    if args.baseline_filter:
        baseline_filter = [b.strip() for b in args.baseline_filter.split(",")]
    if args.parallel:
        if args.gpus:
            gpus = _parse_csv_ints(args.gpus)
        elif torch.cuda.is_available():
            gpus = list(range(torch.cuda.device_count()))
        else:
            raise RuntimeError("--parallel requires --gpus when CUDA is not visible")
        run_exp3_parallel(
            fast=not args.formal,
            ae_filter=ae_filter,
            baseline_filter=baseline_filter,
            gpus=gpus,
            auto_merge=not args.no_auto_merge,
        )
        return
    run_exp3(
        fast=not args.formal,
        ae_filter=ae_filter,
        baseline_filter=baseline_filter,
        gpu_id=args.gpu_id,
        segment_tag=args.segment,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()

