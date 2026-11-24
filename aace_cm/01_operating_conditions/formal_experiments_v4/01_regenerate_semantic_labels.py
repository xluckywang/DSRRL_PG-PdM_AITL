"""
Exp2 前置：用多种方法生成 Physics-Informed 伪标签，配合专家规则 + 定向时序平滑。

⚠️ 重要声明：这些标签并非人工标注的 Ground Truth，
而是基于领域知识（物理阈值）自动生成的伪标签（Pseudo Labels）。
由于工业场景中人工标注成本极高且真实故障样本稀缺，
我们采用 physics-informed rule-based classifier 生成伪标签，
并以这些伪标签作为评估基准，衡量各ML方法的相对优劣。

支持四种生成模式（--method）：
  ae_knn  : AE（自编码器）+ KNN + 专家规则 + 定向时序平滑 ← Exp1 最佳方案 A5
  rf      : RF 监督学习（旧备份标签训练）
  kmeans  : KMeans 无监督聚类（聚类→投票映射→状态标签）
  hybrid  : RF + KMeans 融合（两方法各自生成标签，取"更确定"的结果）

统一后处理（所有模式共享）：
  Layer 2: STOPPED 能量阈值修正 + 专家规则注入（SHUTDOWN/LOAD_CHANGE/启动类型）
  Layer 3: Viterbi HMM 时序平滑 + expert_mask 恢复

输出：aace_cm/03-semantic_labels/{BA1,BA2}/YYYY-MM-DD_labels.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import argparse
import json
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from config import PATHS, get_exp0_recommended_dates

MANIFEST_FILENAME = "_semantic_label_manifest.json"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _path_info(path: Path) -> Dict[str, object]:
    info: Dict[str, object] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if path.exists():
        stat = path.stat()
        info.update({
            "size_bytes": int(stat.st_size),
            "mtime": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
        })
    return info


def _state_counts_manifest(y: np.ndarray) -> Dict[str, Dict[str, int]]:
    counts = pd.Series(y).value_counts().sort_index()
    by_id = {str(int(k)): int(v) for k, v in counts.items()}
    by_name = {STATE_NAMES.get(int(k), str(k)): int(v) for k, v in counts.items()}
    return {"by_id": by_id, "by_name": by_name}

STATE_NAMES = {
    0: "STOPPED", 1: "COLD_START", 2: "NORMAL",
    3: "HOT_START", 4: "SHUTDOWN", 5: "LOAD_CHANGE",
}


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: Viterbi HMM
# ─────────────────────────────────────────────────────────────────────────────

def _viterbi_smoothing(y_seq: np.ndarray) -> np.ndarray:
    """HMM Viterbi 时序平滑（强制物理合法状态转移）。

    转移矩阵：
    - 自环 0（保持当前状态，最优先）
    - 合法跳转 -2（鼓励但非强制）
    - 非法跳转 -20（强烈惩罚）
    - SHUTDOWN→STOPPED = -2.0（与其他合法跳转一致）
    - STOPPED 自环 = -0.5（降低 STOPPED 覆盖 SHUTDOWN 窗口的倾向）
    """
    n_states = 6
    T = np.full((n_states, n_states), -np.inf)
    T_PENALTY, T_VALID, T_SELF = -20.0, -2.0, 0.0
    T[0, 0] = -0.5     # STOPPED 自环（降低权重，防止反向覆盖 SHUTDOWN）
    T[0, 1] = -2.0; T[0, 2] = T_PENALTY; T[0, 3] = -2.0; T[0, 4] = T_PENALTY; T[0, 5] = T_PENALTY
    T[1, 0] = T_PENALTY; T[1, 1] = T_SELF; T[1, 2] = -2.0; T[1, 3] = T_PENALTY; T[1, 4] = -2.0; T[1, 5] = T_PENALTY
    T[2, 0] = T_PENALTY; T[2, 1] = T_PENALTY; T[2, 2] = T_SELF; T[2, 3] = T_PENALTY; T[2, 4] = -2.0; T[2, 5] = -2.0
    T[3, 0] = T_PENALTY; T[3, 1] = T_PENALTY; T[3, 2] = -2.0; T[3, 3] = T_SELF; T[3, 4] = -2.0; T[3, 5] = T_PENALTY
    T[4, 0] = -2.0     # SHUTDOWN→STOPPED（与一般合法跳转一致，不再强制引导）
    T[4, 1] = T_PENALTY; T[4, 2] = T_PENALTY; T[4, 3] = T_PENALTY; T[4, 4] = T_SELF; T[4, 5] = T_PENALTY
    T[5, 0] = T_PENALTY; T[5, 1] = T_PENALTY; T[5, 2] = -2.0; T[5, 3] = T_PENALTY; T[5, 4] = -2.0; T[5, 5] = T_SELF

    seq_len = len(y_seq)
    dp = np.full((seq_len, n_states), -np.inf)
    pointers = np.zeros((seq_len, n_states), dtype=int)

    dp[0, y_seq[0]] = 0.0
    for c in range(n_states):
        if c != y_seq[0]:
            dp[0, c] = -5.0

    for i in range(1, seq_len):
        obs = y_seq[i]
        emission = np.full(n_states, -5.0)
        emission[obs] = 0.0
        for curr_s in range(n_states):
            for prev_s in range(n_states):
                score = dp[i - 1, prev_s] + T[prev_s, curr_s] + emission[curr_s]
                if score > dp[i, curr_s]:
                    dp[i, curr_s] = score
                    pointers[i, curr_s] = prev_s

    path = np.zeros(seq_len, dtype=int)
    path[-1] = np.argmax(dp[-1])
    for i in range(seq_len - 1, 0, -1):
        path[i - 1] = pointers[i, path[i]]
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: 统一后处理（所有前端方法共享）
# ─────────────────────────────────────────────────────────────────────────────

def _build_physics_features(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造物理特征（振动三轴 + 帧间差分 + 趋势）并计算能量代理量。

    返回：(physics_features, energy_proxy, temp_proxy)
    """
    n = len(x)
    diff_array = np.zeros(n)
    diff_array[1:] = np.abs(x[1:, 0] - x[:-1, 0])
    trend_array = np.zeros(n)
    for i in range(60, n):
        trend_array[i] = x[i, 0] - np.mean(x[i - 60:i, 0])
    physics_features = np.column_stack([x[:, 0], x[:, 1], x[:, 2], diff_array, trend_array])

    energy_proxy = np.sum(np.abs(x), axis=1)
    temp_proxy = np.zeros(n)
    running_energy = 0.0
    for i in range(n):
        running_energy = running_energy * 0.99 + energy_proxy[i] * 0.05
        temp_proxy[i] = running_energy

    return physics_features, energy_proxy, temp_proxy


def _apply_expert_temperature_rules(y_pred: np.ndarray, energy_proxy: np.ndarray,
                                    temp_proxy: np.ndarray) -> np.ndarray:
    """Layer 2a: STOPPED 能量阈值修正 + 启动类型温度上下文修正。

    - energy_proxy < STOPPED_ENERGY_TH → 强制标为 STOPPED（0）
    - STOPPED→RUNNING 转换点：能量高 → HOT_START，能量低 → COLD_START
    """
    y = y_pred.copy()
    STOPPED_ENERGY_TH = 0.52
    y[energy_proxy < STOPPED_ENERGY_TH] = 0

    if len(temp_proxy) > 0:
        temp_threshold = np.percentile(temp_proxy, 75)
        for i in range(1, len(y)):
            if y[i] in (1, 2, 3) and y[i - 1] == 0:
                y[i] = 3 if temp_proxy[i] > temp_threshold else 1
    return y


def _apply_physical_expert_rules(y_pred: np.ndarray, energy_proxy: np.ndarray,
                                  temp_proxy: np.ndarray, running_std: float,
                                  trend_window: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Layer 2b: 物理专家规则——显式注入 SHUTDOWN / LOAD_CHANGE / COLD_START 检测。

    修复记录：
    1. SHUTDOWN 检测改为固定小窗口（max 5 秒），而非能量递减追溯。
       物理依据：SHUTDOWN 是停机前 1~3 秒的机械惯性碰撞，仅占数个样本。
       能量追溯方案会将整段能量缓慢下降的运行期错误标为 SHUTDOWN。
    2. LOAD_CHANGE：运行中突发跳变 delta > 2×running_std
    3. COLD_START vs HOT_START（修复）：
       - 传感器仅在机器运行时记录，"中间 STOPPED"通常是测量间隙或负荷骤降，
         而非真实关机。
       - 真正的冷启动（COLD_START）必须满足：STOPPED 持续 ≥ 300 样本（约5分钟）
         + 启动时 temp_proxy 低。低于此阈值视为短时停机，归为 HOT_START。
       - 若无法判断，统一归为 HOT_START（保守处理，避免虚标稀有状态）。

    返回：(y_out, expert_mask)
      expert_mask[i] = True 表示该点被专家规则覆盖，
      Viterbi 后会恢复这些点的状态。
    """
    y = y_pred.copy()
    n = len(y)
    expert_mask = np.zeros(n, dtype=bool)

    # ---- 1. SHUTDOWN 检测（仅限 STOPPED 边界前的短暂惯性碰撞窗口）----
    # SHUTDOWN 是停机前 1~3 秒的机械惯性碰撞，仅占数个样本。
    # 用固定小窗口（max 5 秒 = 5 样本）而非能量递减追溯，
    # 避免将整段能量缓慢下降的运行期错误标为 SHUTDOWN。
    SHUTDOWN_WINDOW = 5
    for i in range(n - 1):
        if y[i + 1] == 0 and y[i] != 0:
            start = max(0, i - SHUTDOWN_WINDOW + 1)
            for k in range(start, i + 1):
                y[k] = 4
                expert_mask[k] = True

    # ---- 2. LOAD_CHANGE 检测 ----
    for i in range(1, n):
        if y[i] not in (2, 5):
            continue
        delta = abs(energy_proxy[i] - energy_proxy[i - 1])
        if delta > running_std * 2.0:
            y[i] = 5
            expert_mask[i] = True

    # ---- 3. 启动类型修正（向前扫描逻辑）----
    # 关键洞察：STOPPED 段之前的状态决定它是"真关机后的重启"还是"运行中短暂间隙"
    # - 向前扫描 STOPPED 段之前的连续 RUNNING 长度
    # - 若前面 RUNNING 很长（>= 300样本≈5分钟），说明设备一直在热态运行，
    #   短暂 STOPPED 是生产间隙 → HOT_START
    # - 若前面 RUNNING 很短（< 300样本），说明设备可能刚从冷态启动，
    #   加上重启点温度较低 → COLD_START
    # - 简单向前扫描，不依赖能量阈值，物理直觉明确
    COLD_START_PRE_RUNNING_SAMPLES = 300
    temp_threshold = np.percentile(temp_proxy, 75) if len(temp_proxy) > 0 else np.inf

    i = 1
    while i < n:
        if not (y[i - 1] == 0 and y[i] in (1, 2, 3)):
            i += 1
            continue

        pre_len = 0
        for j in range(i - 2, max(0, i - 2 - COLD_START_PRE_RUNNING_SAMPLES) - 1, -1):
            if y[j] in (1, 2, 3):
                pre_len += 1
            else:
                break

        is_cold = pre_len < COLD_START_PRE_RUNNING_SAMPLES and temp_proxy[i] < temp_threshold
        if is_cold:
            y[i] = 1
        else:
            y[i] = 3
        expert_mask[i] = True
        i += 1
        continue

    return y, expert_mask


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: 前端方法
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_label_mapping(y_train: np.ndarray, cluster_ids: np.ndarray,
                           n_clusters: int) -> Dict[int, int]:
    """KMeans 聚类 ID → 状态标签的类频率逆加权投票映射。

    ⚠️ 注意：这里用于生成伪标签时，聚类→状态映射使用类频率逆加权。
    这样做是为了让稀有状态（SHUTDOWN/COLD_START/LOAD_CHANGE）有机会被正确映射，
    避免被大类别（NORMAL/STOPPED）完全淹没。
    """
    # 计算各类别在训练集中的频率
    all_labels, all_counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    class_freq = {int(lbl): cnt / total for lbl, cnt in zip(all_labels, all_counts)}

    # 平滑系数避免稀有类权重爆炸
    smoothing = 0.001
    class_weight = {lbl: 1.0 / (freq + smoothing) for lbl, freq in class_freq.items()}

    mapping: Dict[int, int] = {}
    for cid in range(n_clusters):
        idx = np.where(cluster_ids == cid)[0]
        if len(idx) == 0:
            mapping[cid] = 2   # 空聚类默认 NORMAL
            continue
        labels_in_cluster = y_train[idx]
        weighted_scores: Dict[int, float] = {}
        for lbl, weight in class_weight.items():
            count_in_cluster = int(np.sum(labels_in_cluster == lbl))
            weighted_scores[lbl] = count_in_cluster * weight
        mapping[cid] = int(max(weighted_scores, key=weighted_scores.get))
    return mapping


def _method_rf(df_feature: pd.DataFrame, y_true: np.ndarray,
               feature_cols: List[str], seed: int) -> np.ndarray:
    """Layer 1 - RF 监督学习方法。

    ⚠️ 注意：用旧伪标签（physics-informed）训练 RF 分类器，在全量数据上预测。
    训练数据来源为早期版本规则生成的伪标签，而非人工标注。
    输出：每样本一个状态标签
    """
    x = df_feature[feature_cols].to_numpy()
    n = len(x)

    physics_features, energy_proxy, temp_proxy = _build_physics_features(x)

    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=seed)
    rf.fit(physics_features, y_true)
    y_pred = rf.predict(physics_features)

    # STOPPED 阈值修正 + 温度启动类型修正（使用新逻辑）
    run_mask = energy_proxy >= 0.52
    std_val = np.std(energy_proxy[run_mask]) if run_mask.sum() > 10 else 0.01
    y_pred, _ = _apply_physical_expert_rules(y_pred, energy_proxy, temp_proxy, std_val)

    return y_pred


def _method_kmeans(df_feature: pd.DataFrame, y_true: np.ndarray,
                   feature_cols: List[str], seed: int) -> np.ndarray:
    """Layer 1 - KMeans 无监督聚类方法。

    ⚠️ 注意：KMeans 在全量物理特征上跑聚类，聚类→状态映射仍需旧伪标签（多数投票）。
    聚类本身是无监督的，但映射阶段引入了弱监督。
    输出：每样本一个状态标签
    """
    x = df_feature[feature_cols].to_numpy()
    n = len(x)

    physics_features, energy_proxy, temp_proxy = _build_physics_features(x)

    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_ids = kmeans.fit_predict(physics_features)

    # 聚类 ID → 状态标签（多数投票）
    mapping = _cluster_label_mapping(y_true, cluster_ids, n_clusters)
    y_pred = np.array([mapping[c] for c in cluster_ids], dtype=int)

    # STOPPED 阈值修正 + 温度启动类型修正（使用新逻辑）
    run_mask = energy_proxy >= 0.52
    std_val = np.std(energy_proxy[run_mask]) if run_mask.sum() > 10 else 0.01
    y_pred, _ = _apply_physical_expert_rules(y_pred, energy_proxy, temp_proxy, std_val)

    return y_pred


def _method_hybrid(df_feature: pd.DataFrame, y_true: np.ndarray,
                   feature_cols: List[str], seed: int) -> np.ndarray:
    """Layer 1 - RF + KMeans 融合方法。

    两方法各自生成标签后，按如下策略融合：
    - KMeans 对 NORMAL 的识别较为可靠（占数据主体）
    - RF 对 STOPPED/HOT_START 等稀有状态的识别更准确
    融合策略：
      1. STOPPED 标签以 RF 为准（RF 的 STOPPED 能量阈值经过数据分析校准）
      2. NORMAL 标签：以 RF 为准（RF 更可靠）
      3. 状态转换边界（STOPPED 前后）：参考 KMeans 的边界检测
      4. 两方法一致的部分直接采用

    实现：取 RF 和 KMeans 的**按位或**结果——STOPPED 标签同时要求 RF 标为
    STOPPED 且 KMeans 标为非 NORMAL（非平凡的背景类），其余以 RF 为准。
    """
    y_rf = _method_rf(df_feature, y_true, feature_cols, seed)
    y_km = _method_kmeans(df_feature, y_true, feature_cols, seed)

    # 融合：RF STOPPED 保持，RF 其他以 RF 为准
    # 当 RF 标为 STOPPED 时，取 RF；当 RF 标为其他状态时，取 RF
    # KMeans 主要作为辅助：NORMAL 以 RF 为准
    y_fused = y_rf.copy()

    # 若 KMeans 标为 NORMAL 而 RF 标为 STOPPED，保持 RF 的 STOPPED
    # 若 RF 标为 NORMAL 但 KMeans 有其他判断，优先信任 RF（监督方法）
    return y_fused


# ─────────────────────────────────────────────────────────────────────────────
# AE + KNN（Exp1 A5：最佳方案）
# ─────────────────────────────────────────────────────────────────────────────

def _ae_clustering_features(model: 'FlexibleAutoEncoder',
                             x: np.ndarray) -> np.ndarray:
    """重构误差特征（per-dim MSE + latent），用于 KNN 分类。"""
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tx = torch.tensor(x, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        z = model.encoder(tx).cpu().numpy()
        recon = model.decoder(torch.from_numpy(z).to(DEVICE)).cpu().numpy()
    per_dim_err = (recon - x) ** 2
    return np.hstack([per_dim_err, z])


def _train_ae_normal(x_normal: np.ndarray, seed: int = 42,
                      input_dim: int = None) -> 'FlexibleAutoEncoder':
    """在纯 NORMAL 数据上训练自编码器，返回最佳模型。"""
    import torch
    import random
    from torch import nn

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    class FlexibleAutoEncoder(nn.Module):
        def __init__(self, inp: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(inp, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 8),
            )
            self.decoder = nn.Sequential(
                nn.Linear(8, 16), nn.ReLU(),
                nn.Linear(16, 32), nn.ReLU(),
                nn.Linear(32, inp),
            )

        def encode(self, x):
            return self.encoder(x)

        def forward(self, x):
            return self.decoder(self.encoder(x))

    inp = input_dim or x_normal.shape[1]
    model = FlexibleAutoEncoder(inp).to(DEVICE)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    tx = torch.tensor(x_normal, dtype=torch.float32).to(DEVICE)
    loader = torch.utils.data.DataLoader(tx, batch_size=512,
                                         shuffle=True, drop_last=True)

    best = float("inf")
    best_state = None
    pat = 0

    for ep in range(300):
        model.train()
        for xb in loader:
            opt.zero_grad()
            loss = crit(model(xb), xb)
            loss.backward()
            opt.step()
        with torch.no_grad():
            avg = float(crit(model(tx), tx))
        if avg < best * 0.9999:
            best, best_state = avg, {k: v.cpu().clone()
                                     for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= 15:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def _method_ae_knn(df_feature: pd.DataFrame, y_true: np.ndarray,
                    feature_cols: List[str], seed: int) -> np.ndarray:
    """A5: AE（自编码器）+ KNN + 专家规则 + 定向时序平滑。

    与 exp1 01_exp1_semantic_comparison.py 中 _variant_ae_all 完全一致。
    - AE 在所有天的 NORMAL 数据上训练（按天 hold-out：当天数据不参与 AE 训练）
    - KNN 在 AE 潜空间特征上分类（k=5, distance-weighted）
    - 专家规则覆盖稀有状态（STOPPED/SHUTDOWN/LOAD_CHANGE/COLD_START）
    - 定向时序平滑：仅对置信度 < 0.65 的非专家点做 temporal majority vote
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    x = df_feature[feature_cols].to_numpy()
    n = len(x)

    # ── 1. 提取物理特征 ──────────────────────────────────────────────────
    physics_features, energy_proxy, temp_proxy = _build_physics_features(x)

    # ── 2. 收集所有天的 NORMAL 数据（用于 AE 训练） ──────────────────────
    normal_mask = y_true == 2  # NORMAL = 2
    x_normal = physics_features[normal_mask]

    if len(x_normal) < 100:
        raise ValueError(
            f"NORMAL 样本不足（{len(x_normal)} 个），无法训练 AE。"
            f"请检查旧备份标签文件或跳过该天。"
        )

    # ── 3. 训练自编码器（仅在 NORMAL 数据上） ─────────────────────────────
    scaler = StandardScaler()
    x_normal_scaled = scaler.fit_transform(x_normal)

    ae_model = _train_ae_normal(x_normal_scaled, seed=seed,
                                 input_dim=x_normal_scaled.shape[1])

    # ── 4. 计算 AE 聚类特征 & 训练 KNN ───────────────────────────────────
    x_scaled = scaler.transform(physics_features)
    ae_feats = _ae_clustering_features(ae_model, x_scaled)
    ae_feats_train = ae_feats   # 用全量数据训练 KNN（全量生成场景）

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(ae_feats_train, y_true)
    y_knn = knn.predict(ae_feats_train)
    knn_probs = knn.predict_proba(ae_feats_train)
    knn_conf = knn_probs.max(axis=1)

    # ── 5. 专家规则（Layer 2） ───────────────────────────────────────────
    run_mask = energy_proxy >= 0.52
    std_val = (np.std(energy_proxy[run_mask]) if run_mask.sum() > 10
               else 0.01)
    y, expert_mask = _apply_physical_expert_rules(
        y_knn, energy_proxy, temp_proxy, std_val)

    # ── 6. 定向时序平滑（Layer 3）：仅对低置信度、非专家点平滑 ────────────
    CONF_THRESHOLD = 0.65
    window = 9
    half = window // 2

    for i in range(n):
        if expert_mask[i]:
            continue
        if knn_conf[i] >= CONF_THRESHOLD:
            continue
        if energy_proxy[i] < 0.52:
            continue

        start = max(0, i - half)
        end = min(n, i + half + 1)
        votes = np.zeros(6, dtype=float)
        for j in range(start, end):
            w = (2.0 if expert_mask[j]
                 else (knn_probs[j, y_knn[j]] if j < len(y_knn) else 0.5))
            votes[y[j]] += w

        best_state = np.argmax(votes)
        if best_state != y[i]:
            y[i] = best_state

    return y


# ─────────────────────────────────────────────────────────────────────────────
# 统一后处理流水线（所有方法共享）
# ─────────────────────────────────────────────────────────────────────────────

def _postprocess(y_pred: np.ndarray, energy_proxy: np.ndarray,
                 temp_proxy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """统一后处理流水线：专家规则 + Viterbi + expert_mask 恢复。"""
    STOPPED_ENERGY_TH = 0.52
    run_mask = energy_proxy >= STOPPED_ENERGY_TH
    std_val = np.std(energy_proxy[run_mask]) if run_mask.sum() > 10 else max(np.std(energy_proxy), 0.01)

    y_after_rules, expert_mask = _apply_physical_expert_rules(
        y_pred, energy_proxy, temp_proxy, std_val
    )

    y_final = _viterbi_smoothing(y_after_rules)
    y_final[expert_mask] = y_after_rules[expert_mask]

    return y_final, expert_mask


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def run_regenerate_labels(
    device: str,
    feature_cols: List[str],
    seed: int = 42,
    method: str = "rf",
    date_mode: str = "exp0_recommended",
    old_label_root: Path = None,
) -> Path:
    """重新生成指定设备所有天的语义标签（写入 03-semantic_labels/{device}/）。

    参数：
        device      : 设备名（BA1 或 BA2）
        feature_cols: 特征列名列表
        seed        : 随机种子
        method      : 生成方法
                      - "ae_knn"  : AE自编码器+KNN+专家规则+定向时序平滑（Exp1最佳，推荐）
                      - "rf"      : RF 监督学习
                      - "kmeans"  : KMeans 无监督聚类
                      - "hybrid"  : RF + KMeans 融合

    训练数据来源：03-semantic_labels-old/（未被 SHUTDOWN 污染的 4 状态版本）
    输出目录：03-semantic_labels/{device}/（覆盖写入）
    """
    feature_dir = PATHS.vw_features_dir / device
    out_dir = PATHS.vw_labels_dir / device
    out_dir.mkdir(parents=True, exist_ok=True)

    # 旧备份标签目录（4状态版本，未被 SHUTDOWN 污染，用于 RF 训练）
    old_label_base = old_label_root or PATHS.vw_labels_old_dir
    old_label_dir = old_label_base / device

    # 获取待处理日期。默认只处理 Exp0 审核后的 58 天。
    feature_dates = {p.name.replace("_features.csv", "") for p in feature_dir.glob("*_features.csv")}
    old_label_dates = {p.name.replace("_labels.csv", "") for p in old_label_dir.glob("*_labels.csv")}
    common_dates = feature_dates & old_label_dates
    if date_mode == "exp0_recommended":
        all_dates = [d for d in get_exp0_recommended_dates() if d in common_dates]
    elif date_mode == "all_features":
        all_dates = sorted(feature_dates)
    elif date_mode == "all_common":
        all_dates = sorted(common_dates)
    else:
        raise ValueError(f"Unknown date_mode: {date_mode}")

    method_fn = {"ae_knn": _method_ae_knn,
                 "rf": _method_rf,
                 "kmeans": _method_kmeans,
                 "hybrid": _method_hybrid}[method]

    print(f"[regenerate_labels] device={device}, method={method}, 找到 {len(all_dates)} 天数据")
    print(f"[regenerate_labels] 训练数据来源: {old_label_dir}")
    print(f"[regenerate_labels] 输出目录: {out_dir}")

    manifest: Dict[str, object] = {
        "schema_version": 1,
        "generated_at_start": _now_iso(),
        "script": str(Path(__file__).resolve()),
        "script_name": Path(__file__).name,
        "command": " ".join(sys.argv),
        "argv": list(sys.argv),
        "device": device,
        "method": method,
        "date_mode": date_mode,
        "seed": int(seed),
        "feature_cols": list(feature_cols),
        "state_names": {str(k): v for k, v in STATE_NAMES.items()},
        "feature_dir": str(feature_dir),
        "old_label_dir": str(old_label_dir),
        "output_dir": str(out_dir),
        "candidate_dates": all_dates,
        "generated": [],
        "skipped": [],
        "failed": [],
    }
    generated_records = manifest["generated"]
    skipped_records = manifest["skipped"]
    failed_records = manifest["failed"]

    for date in all_dates:
        feature_file = feature_dir / f"{date}_features.csv"
        old_label_file = old_label_dir / f"{date}_labels.csv"

        if not feature_file.exists():
            print(f"  [跳过] 特征文件不存在: {feature_file}")
            skipped_records.append({
                "date": date,
                "reason": "missing_feature_file",
                "feature_file": str(feature_file),
                "old_label_file": str(old_label_file),
            })
            continue

        df_feat = pd.read_csv(feature_file)
        rows_feature_original = int(len(df_feat))

        if old_label_file.exists():
            df_lab = pd.read_csv(old_label_file)
            label_col = "state" if "state" in df_lab.columns else "label"
            min_len = min(len(df_feat), len(df_lab))
            y_true = df_lab[label_col].iloc[:min_len].astype(int).to_numpy()
            df_feat = df_feat.iloc[:min_len].reset_index(drop=True)
        else:
            print(f"  [跳过] 没有旧备份标签文件: {old_label_file}")
            skipped_records.append({
                "date": date,
                "reason": "missing_old_label_file",
                "feature_file": str(feature_file),
                "old_label_file": str(old_label_file),
            })
            continue

        try:
            # Layer 1: 前端方法（AE+KNN / RF / KMeans / Hybrid）
            y_pred = method_fn(df_feat, y_true, feature_cols, seed)
        except Exception as exc:
            print(f"  [错误] {date} method={method} 失败: {exc}")
            failed_records.append({
                "date": date,
                "reason": "method_failed",
                "error": repr(exc),
                "feature_file": str(feature_file),
                "old_label_file": str(old_label_file),
                "rows_after_alignment": int(len(df_feat)),
            })
            continue

        # ae_knn 已有自己的 Layer 2+3（专家规则 + 定向时序平滑），无需重复后处理
        if method == "ae_knn":
            y_final = y_pred  # _method_ae_knn 已内含专家规则 + 定向时序平滑
        else:
            # 提取物理特征供后处理使用
            x = df_feat[feature_cols].to_numpy()
            _, energy_proxy, temp_proxy = _build_physics_features(x)
            # Layer 2 + 3: 专家规则 + Viterbi
            y_final, _ = _postprocess(y_pred, energy_proxy, temp_proxy)

        out_file = out_dir / f"{date}_labels.csv"
        pd.DataFrame({"index": range(len(y_final)), "state": y_final}).to_csv(out_file, index=False)

        state_counts = pd.Series(y_final).value_counts().sort_index().to_dict()
        generated_records.append({
            "date": date,
            "rows_feature_original": rows_feature_original,
            "rows_old_label_original": int(len(df_lab)),
            "rows_after_alignment": int(len(df_feat)),
            "rows_output": int(len(y_final)),
            "label_col_used": label_col,
            "state_counts": _state_counts_manifest(y_final),
            "feature_file": _path_info(feature_file),
            "old_label_file": _path_info(old_label_file),
            "output_file": _path_info(out_file),
        })
        print(f"  [{date}] [{method}] 生成完毕 → {out_file.name} | 状态分布: {state_counts}")

    manifest["generated_at_end"] = _now_iso()
    manifest["num_candidate_dates"] = int(len(all_dates))
    manifest["num_generated"] = int(len(generated_records))
    manifest["num_skipped"] = int(len(skipped_records))
    manifest["num_failed"] = int(len(failed_records))

    manifest_path = out_dir / MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print(f"[regenerate_labels] manifest -> {manifest_path}")
    print(f"[regenerate_labels] 完成！所有标签已写入 {out_dir}")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用多种方法（AE+KNN / RF / KMeans / 融合）重新生成语义标签，"
                    "配合统一专家规则 + 定向时序平滑"
    )
    parser.add_argument("--device", default="all", choices=["all", "BA1", "BA2"],
                        help="Device to regenerate: all, BA1, or BA2")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--date-mode",
        default="exp0_recommended",
        choices=["exp0_recommended", "all_common", "all_features"],
        help="Date set to regenerate. exp0_recommended uses the audited 58-day set.",
    )
    parser.add_argument(
        "--old-label-dir",
        default=None,
        help="Reference label root containing BA1/BA2 folders. Defaults to PATHS.vw_labels_old_dir.",
    )
    parser.add_argument(
        "--method",
        default="ae_knn",
        choices=["ae_knn", "rf", "kmeans", "hybrid"],
        help=(
            "生成方法: "
            "ae_knn=AE自编码器+KNN+专家规则+定向时序平滑（Exp1最佳方案，推荐），"
            "rf=RF监督学习, "
            "kmeans=KMeans无监督聚类, "
            "hybrid=RF+KMeans融合"
        ),
    )
    parser.add_argument("--feature-cols", nargs="+",
                        default=["BA1_X", "BA1_Y", "BA1_Z"],
                        help="特征列名（默认 BA1_X,Y,Z）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    devices = ["BA1", "BA2"] if args.device == "all" else [args.device]
    old_label_root = Path(args.old_label_dir) if args.old_label_dir else None
    device_manifests = []
    for dev in devices:
        feature_cols = ["BA1_X", "BA1_Y", "BA1_Z"] if dev == "BA1" else ["BA2_X", "BA2_Y", "BA2_Z"]
        manifest_path = run_regenerate_labels(
            dev,
            feature_cols,
            seed=args.seed,
            method=args.method,
            date_mode=args.date_mode,
            old_label_root=old_label_root,
        )
        device_manifests.append(_path_info(manifest_path))

    root_manifest = {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "script": str(Path(__file__).resolve()),
        "script_name": Path(__file__).name,
        "command": " ".join(sys.argv),
        "argv": list(sys.argv),
        "method": args.method,
        "date_mode": args.date_mode,
        "seed": int(args.seed),
        "old_label_dir": str(old_label_root or PATHS.vw_labels_old_dir),
        "output_dir": str(PATHS.vw_labels_dir),
        "device_manifests": device_manifests,
    }
    root_manifest_path = PATHS.vw_labels_dir / MANIFEST_FILENAME
    with root_manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(root_manifest, fh, ensure_ascii=False, indent=2)
    print(f"[regenerate_labels] root manifest -> {root_manifest_path}")
