"""
故障注入模块

使用KDE-SS-Gen-Framework物理模型进行故障信号注入。

故障类型：
1. crack       - Paris疲劳裂纹扩展模型
2. thermal     - Arrhenius热退化模型
3. mechanical_looseness - Archard轴承磨损模型
4. imbalance   - 动不平衡模型

注入策略：
- 仅在NORMAL状态段注入
- 使用物理模型直接修改原始DataFrame
- 返回 (modified_df, binary_labels)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

_VENDOR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "formal_experiments_v4", "_vendor_py312"))
if os.path.isdir(_VENDOR_DIR) and _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

# 导入配置
from aace_config import Paths, SensorConfig, FaultInjectionConfig, SemanticConfig

# 导入KDE物理模型
Paths.add_kde_to_path()
from anomaly_generators import PhysicsAnomalyGenerator


def _patch_pyarrow_for_pandas() -> None:
    try:
        import pyarrow as pa
        import pyarrow.lib as palib
    except Exception:
        return
    original = getattr(pa, "unregister_extension_type", None)
    if original is None or getattr(original, "_vw_fault_safe", False):
        return

    def safe_unregister(name):
        try:
            return original(name)
        except palib.ArrowKeyError as exc:
            if "No type extension" in str(exc):
                return None
            raise

    safe_unregister._vw_fault_safe = True
    pa.unregister_extension_type = safe_unregister


RUNNING_ENERGY_TH = 0.52
LOAD_CHANGE_CONTEXT = 120
MAX_LOAD_CHANGE_BRIDGE = 300


def _segments_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    segments: List[Tuple[int, int]] = []
    start = prev = int(idx[0])
    for raw_idx in idx[1:]:
        cur = int(raw_idx)
        if cur != prev + 1:
            segments.append((start, prev + 1))
            start = cur
        prev = cur
    segments.append((start, prev + 1))
    return segments


def _derive_running_load_change_mask(
    df: pd.DataFrame,
    state_labels: np.ndarray,
    bearing: str,
) -> np.ndarray:
    vib_cols = [f"{bearing}_{axis}" for axis in ("X", "Y", "Z") if f"{bearing}_{axis}" in df.columns]
    if len(vib_cols) < 3:
        return np.zeros(len(state_labels), dtype=bool)
    x = df[vib_cols].iloc[:len(state_labels)].to_numpy(dtype=float)
    energy = np.sum(np.abs(x), axis=1)
    stopped = np.isin(state_labels, (SemanticConfig.STOPPED, SemanticConfig.SHUTDOWN))
    high_energy = energy >= RUNNING_ENERGY_TH
    out = np.zeros(len(state_labels), dtype=bool)
    running_states = (SemanticConfig.COLD_START, SemanticConfig.NORMAL,
                      SemanticConfig.HOT_START, SemanticConfig.LOAD_CHANGE)
    for start, end in _segments_from_mask(stopped):
        candidate = np.zeros(len(state_labels), dtype=bool)
        candidate[start:end] = high_energy[start:end]
        if not candidate[start:end].any():
            continue
        pre_s = max(0, start - LOAD_CHANGE_CONTEXT)
        post_e = min(len(state_labels), end + LOAD_CHANGE_CONTEXT)
        before_y = state_labels[pre_s:start]
        after_y = state_labels[end:post_e]
        before_e = energy[pre_s:start]
        after_e = energy[end:post_e]
        before_running = (
            np.isin(before_y, running_states).any()
            or (len(before_e) > 0 and np.mean(before_e >= RUNNING_ENERGY_TH) >= 0.35)
        )
        after_running = (
            np.isin(after_y, running_states).any()
            or (len(after_e) > 0 and np.mean(after_e >= RUNNING_ENERGY_TH) >= 0.35)
        )
        if before_running and after_running:
            out |= candidate
    return out


def _merge_segments_across_continuity(
    segments: List[Tuple[int, int]],
    continuity_mask: np.ndarray,
    max_bridge_gap: int = MAX_LOAD_CHANGE_BRIDGE,
) -> List[Tuple[int, int]]:
    if not segments:
        return []
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = segments[0]
    for next_s, next_e in segments[1:]:
        gap = next_s - cur_e
        can_bridge = (
            0 < gap <= max_bridge_gap
            and bool(np.all(continuity_mask[cur_e:next_s]))
        )
        if can_bridge:
            cur_e = next_e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = next_s, next_e
    merged.append((cur_s, cur_e))
    return merged


class FaultInjector:
    """
    故障注入器 — 封装 KDE-SS-Gen-Framework 物理模型

    对每个轴承座(BA1/BA2)独立创建 PhysicsAnomalyGenerator，
    根据语义标签仅在 NORMAL 状态段注入故障。
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.last_clamp_report = None
        # 为每个轴承座创建独立的generator
        self.generators = {}
        for bearing in SensorConfig.BEARING_NAMES:
            self.generators[bearing] = PhysicsAnomalyGenerator(
                sensor_name=bearing,
                random_state=random_state
            )

    def inject_single(
        self,
        df: pd.DataFrame,
        bearing: str,
        fault_type: str = 'mechanical_looseness',
        severity: float = 1.0,
        start_ratio: float = 0.3,
        duration_ratio: float = 0.5,
        **kwargs
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        在单个轴承座上注入故障

        Args:
            df: 包含传感器列的DataFrame
            bearing: 轴承座名称 ('BA1' 或 'BA2')
            fault_type: 'crack'/'thermal'/'mechanical_looseness'/'imbalance'/'cascade'/'mixed'
            severity: 严重程度
            start_ratio: 故障起点位置比例
            duration_ratio: 故障持续比例

        Returns:
            (modified_df, labels) — labels: 0=正常, 1=故障
        """
        if bearing not in self.generators:
            raise ValueError(f"未知轴承座: {bearing}. 可选: {list(self.generators.keys())}")

        # 钳位 severity 防止物理模型输出爆炸
        capped_severity = self._cap_severity(fault_type, severity)

        # 为不平衡模型注入增强参数
        inject_kwargs = dict(kwargs)
        if fault_type == 'imbalance':
            for k, v in FaultInjectionConfig.IMBALANCE_PARAMS.items():
                inject_kwargs.setdefault(k, v)
        elif fault_type == 'crack':
            for k, v in FaultInjectionConfig.CRACK_PARAMS.items():
                inject_kwargs.setdefault(k, v)
        elif fault_type == 'thermal':
            for k, v in FaultInjectionConfig.THERMAL_PARAMS.items():
                inject_kwargs.setdefault(k, v)

        gen = self.generators[bearing]

        # 记录注入前的数据范围（用于后处理钳位）
        vib_cols = [f"{bearing}_{ax}" for ax in ['X', 'Y', 'Z']]
        existing_cols = [c for c in vib_cols if c in df.columns]
        pre_stats = {c: (df[c].min(), df[c].max(), df[c].std()) for c in existing_cols}

        result_df, labels = gen.generate_scenario(
            df, scenario=fault_type, severity=capped_severity,
            start_ratio=start_ratio, duration_ratio=duration_ratio,
            **inject_kwargs
        )

        # 后处理：钳位异常值，并保存摘要用于网格日志
        result_df, clamp_report = self._post_clamp(
            result_df,
            pre_stats,
            labels,
            fault_type=fault_type,
            severity=capped_severity,
            bearing=bearing,
        )
        self.last_clamp_report = clamp_report

        return result_df, labels

    @staticmethod
    def _cap_severity(fault_type: str, severity: float) -> float:
        """根据故障类型钳位 severity"""
        cap = FaultInjectionConfig.SEVERITY_CAPS.get(fault_type, 1.5)
        if severity > cap:
            print(f"  severity {severity:.2f} 超过 {fault_type} 上限 {cap}，已钳位")
        return min(severity, cap)

    @staticmethod
    def _post_clamp(
        df: pd.DataFrame,
        pre_stats: dict,
        labels: np.ndarray,
        fault_type: str,
        severity: float,
        bearing: str,
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """钳位注入后的异常值，并返回可读的场景级统计信息。"""
        max_ratio = FaultInjectionConfig.POST_INJECTION_MAX_RATIO
        adaptive_strength = FaultInjectionConfig.POST_INJECTION_ADAPTIVE_STRENGTH
        verbose = FaultInjectionConfig.POST_INJECTION_CLIP_VERBOSE

        cap = FaultInjectionConfig.SEVERITY_CAPS.get(fault_type, 1.5)
        severity_ratio = min(1.0, float(severity) / max(cap, 1e-12))
        clip_scale = 1.0 + adaptive_strength * severity_ratio

        details = []
        total_points = 0
        clipped_points = 0
        max_overshoot_ratio = 1.0

        for col, (vmin, vmax, vstd) in pre_stats.items():
            if col not in df.columns:
                continue

            series = df[col].to_numpy()
            data_range = max(abs(vmax), abs(vmin), vstd * 5)
            clip_bound = data_range * max_ratio * clip_scale
            before_abs = np.abs(series)
            clipped_mask = before_abs > clip_bound
            clipped_count = int(clipped_mask.sum())
            before_max = float(before_abs.max()) if len(before_abs) else 0.0

            df[col] = df[col].clip(-clip_bound, clip_bound)
            after_max = float(df[col].abs().max()) if len(df[col]) else 0.0

            total_points += len(series)
            clipped_points += clipped_count
            overshoot_ratio = (before_max / clip_bound) if clip_bound > 0 else 1.0
            max_overshoot_ratio = max(max_overshoot_ratio, overshoot_ratio)

            if clipped_count > 0:
                details.append({
                    'col': col,
                    'before_max': before_max,
                    'after_max': after_max,
                    'clip_bound': float(clip_bound),
                    'clipped_count': clipped_count,
                    'clipped_ratio': clipped_count / max(1, len(series)),
                    'overshoot_ratio': overshoot_ratio,
                })
                if verbose:
                    print(
                        f"  {col}: 钳位 {before_max:.2f} → {after_max:.2f} "
                        f"(上限 {clip_bound:.2f}, clipped={clipped_count}/{len(series)})"
                    )

        report = {
            'bearing': bearing,
            'fault_type': fault_type,
            'severity': float(severity),
            'severity_ratio': float(severity_ratio),
            'clip_scale': float(clip_scale),
            'n_fault_samples': int(np.sum(labels > 0)),
            'total_points': int(total_points),
            'clipped_points': int(clipped_points),
            'clipped_ratio': float(clipped_points / max(1, total_points)),
            'clip_cols': int(len(details)),
            'max_overshoot_ratio': float(max_overshoot_ratio),
            'details': details,
        }

        return df, report

    def inject_in_segments(
        self,
        df: pd.DataFrame,
        state_labels: np.ndarray,
        bearing: str,
        fault_type: str = 'mechanical_looseness',
        severity: float = 1.0,
        injectable_states: List[int] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        在指定状态段注入故障

        Args:
            df: 完整DataFrame
            state_labels: 每行的状态标签（来自语义分类器）
            bearing: 轴承座名称
            fault_type: 故障类型
            severity: 严重程度
            injectable_states: 可注入的状态列表，默认用配置

        Returns:
            (modified_df, fault_labels)
        """
        from semantic_classifier import StateLabels

        if injectable_states is None:
            injectable_states = FaultInjectionConfig.INJECTABLE_STATES

        # 找到可注入的段
        inject_mask = np.isin(state_labels, injectable_states)
        inject_indices = np.where(inject_mask)[0]

        if len(inject_indices) == 0:
            print(f"警告: 没有可注入的状态段 (states={injectable_states})，跳过注入")
            return df.copy(), np.zeros(len(df), dtype=int)

        # 在可注入的连续段中选最长的
        segments = self._find_segments(inject_indices)
        longest = max(segments, key=lambda s: s[1] - s[0])
        seg_start, seg_end = longest

        print(f"  注入段: [{seg_start}:{seg_end}] ({seg_end - seg_start} 样本)")

        # 提取段数据进行注入
        seg_df = df.iloc[seg_start:seg_end].reset_index(drop=True)

        injected_seg, seg_labels = self.inject_single(
            seg_df, bearing, fault_type, severity, **kwargs
        )

        # 合并回完整DataFrame
        result_df = df.copy()
        for col in injected_seg.columns:
            if col in result_df.columns:
                result_df.iloc[seg_start:seg_end, result_df.columns.get_loc(col)] = injected_seg[col].values

        # 完整标签
        full_labels = np.zeros(len(df), dtype=int)
        full_labels[seg_start:seg_end] = seg_labels

        return result_df, full_labels

    def inject_normal_only(
        self,
        df: pd.DataFrame,
        state_labels: np.ndarray,
        bearing: str,
        fault_type: str = 'mechanical_looseness',
        severity: float = 1.0,
        **kwargs
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """向后兼容：仅在NORMAL段注入（调用 inject_in_segments）"""
        from semantic_classifier import StateLabels
        return self.inject_in_segments(
            df, state_labels, bearing, fault_type, severity,
            injectable_states=[StateLabels.NORMAL], **kwargs
        )

    # 最小段长度（行）：短于该值的段被跳过，避免在停机碎片刻意注入
    _MIN_INJECT_SEG_LEN = 200

    def inject_percentage_of_normal(
        self,
        df: pd.DataFrame,
        state_labels: np.ndarray,
        bearing: str,
        fault_type: str,
        severity: float,
        percentage: float,
        rng: np.random.Generator = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        按 percentage 比例在 NORMAL 状态段中随机位置注入故障。

        注入位置约束（鲁棒性设计）：
          - 注入起点只能在 NORMAL 数据中间 10%~80% 范围内随机选择
          - 前 10% NORMAL 数据：保留为健康基准（用于对比）
          - 后 20% NORMAL 数据：保留以确保注满量达到 percentage（故障最高 20%）
          - 超过 80% 之后的 NORMAL 不用于起始注入，保证注入后有足够空间完成目标量

        如果 NORMAL 数据总量不足，退化为尽可能满足约束。

        Args:
            df: 完整 DataFrame
            state_labels: 每行的语义状态标签（与 df 同行对应）
            bearing: 轴承座名称
            fault_type: 故障类型
            severity: 物理模型 severity
            percentage: 注入比例（0.01 = 总 NORMAL 行数的 1%）
            rng: numpy 随机数生成器（用于可复现的随机起点选择）
            **kwargs: 透传给 inject_single 的额外参数

        Returns:
            (modified_df, fault_labels)
            df.attrs["_fault_injection_info"] 包含注入信息（含 rng_state 用于复现）
        """
        if rng is None:
            rng = np.random.default_rng(self.random_state)

        n_total = len(df)
        normal_indices = np.where(state_labels == SemanticConfig.NORMAL)[0]
        if len(normal_indices) == 0:
            print(f"  警告: 没有 NORMAL 数据，跳过")
            return df.copy(), np.zeros(n_total, dtype=int)

        n_normal = len(normal_indices)
        n_inject_total = max(1, int(n_normal * percentage))

        # ── 把所有 NORMAL 索引拼成虚拟连续序列 ─────────────────────────
        # 在这条"虚拟序列"的 10%~80% 区间随机选起点，向后连续填目标量
        # 再把虚拟位置映射回原始索引
        n_inject_total = max(1, int(n_normal * percentage))

        normal_flat = normal_indices  # 已排序的 1D 数组
        global_inj_start = int(len(normal_flat) * 0.10)   # 虚拟序列前10%之后
        global_inj_end = int(len(normal_flat) * 0.80)     # 虚拟序列后20%之前
        available = global_inj_end - global_inj_start

        if available < n_inject_total:
            # 全局可注范围不够，退化：在虚拟序列中间随机选起点向左填
            blk_start_flat = int(rng.integers(0, len(normal_flat) - n_inject_total + 1))
        else:
            # 在虚拟可注范围内随机选起点
            blk_start_flat = int(rng.integers(global_inj_start, global_inj_end - n_inject_total + 1))
        blk_end_flat = blk_start_flat + n_inject_total

        # 把虚拟区间映射回原始索引，找出原始位置上的故障区间
        fault_mask = np.zeros(n_total, dtype=bool)
        for pos in range(blk_start_flat, blk_end_flat):
            orig_idx = normal_flat[pos]
            fault_mask[orig_idx] = True

        # 合并为连续的原始索引区间
        injected_segments_detail = _segments_from_mask(fault_mask)
        derived_load_change = _derive_running_load_change_mask(df, state_labels, bearing)
        continuity_mask = (
            (state_labels == SemanticConfig.NORMAL)
            | (state_labels == SemanticConfig.LOAD_CHANGE)
            | derived_load_change
        )
        injected_segments_detail = _merge_segments_across_continuity(
            injected_segments_detail,
            continuity_mask,
        )
        fault_mask = np.zeros(n_total, dtype=bool)
        for blk_start, blk_end in injected_segments_detail:
            fault_mask[blk_start:blk_end] = True

        total_injected = sum(e - s for (s, e) in injected_segments_detail)

        result_df = df.copy()
        full_labels = np.zeros(n_total, dtype=int)
        for (blk_start, blk_end) in injected_segments_detail:
            full_labels[blk_start:blk_end] = 1

        # ── 调用 inject_single 注入（整块 start=0, duration=1）──────────
        for (blk_start, blk_end) in injected_segments_detail:
            blk_len = blk_end - blk_start
            if blk_len < 1:
                continue
            blk_df = df.iloc[blk_start:blk_end].reset_index(drop=True)
            injected_seg, _ = self.inject_single(
                blk_df, bearing, fault_type, severity,
                start_ratio=0.0, duration_ratio=1.0, **kwargs
            )
            for col in injected_seg.columns:
                if col.startswith(f"{bearing}_") and col in result_df.columns:
                    result_df.iloc[blk_start:blk_end,
                                   result_df.columns.get_loc(col)] = injected_seg[col].values

        total_injected = sum(e - s for (s, e) in injected_segments_detail)

        # 保存注入信息和随机状态（用于复现）
        rng_state_for_replay = rng.bit_generator.state.copy()
        result_df.attrs["_fault_injection_info"] = {
            "n_inject_total": int(n_inject_total),
            "total_injected": int(total_injected),
            "n_normal": int(n_normal),
            "injected_segments": injected_segments_detail,
            "n_derived_load_change": int(derived_load_change.sum()),
            "continuity_states": ["NORMAL", "LOAD_CHANGE", "derived_running_load_change"],
            "max_load_change_bridge": int(MAX_LOAD_CHANGE_BRIDGE),
            "rng_state": str(rng_state_for_replay),
        }

        # 打印摘要
        segs_str = '; '.join(f"[{s}-{e}]({e - s})" for (s, e) in injected_segments_detail)
        print(f"  注入: 目标={n_inject_total} 实际={total_injected} "
              f"NORMAL={n_normal} pct={percentage*100:.1f}% "
              f"多段数={len(injected_segments_detail)} "
              f"各段: {segs_str}")

        return result_df, full_labels


    @staticmethod
    def _find_segments(indices: np.ndarray) -> List[Tuple[int, int]]:
        """找到连续索引段"""
        if len(indices) == 0:
            return []
        segments = []
        start = indices[0]
        prev = indices[0]
        for idx in indices[1:]:
            if idx != prev + 1:
                segments.append((start, prev + 1))
                start = idx
            prev = idx
        segments.append((start, prev + 1))
        return segments


class SyntheticFaultDataset:
    """合成故障数据集生成器"""

    def __init__(self, output_dir: str = None, random_state: int = 42):
        self.output_dir = output_dir or Paths.FAULT_DATA_DIR
        self.injector = FaultInjector(random_state=random_state)
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_from_raw_data(
        self,
        data_path: str,
        labels_path: str = None,
        bearing: str = 'BA1',
        fault_types: List[str] = None,
        n_scenarios: int = 4,
        severity_range: Tuple[float, float] = None
    ):
        """
        从原始/特征数据生成故障数据集

        Args:
            data_path: 数据文件路径（含BA1/BA2传感器列）
            labels_path: 语义标签路径（可选，用于分段注入）
            bearing: 目标轴承座
            fault_types: 故障类型列表
            n_scenarios: 场景数量
            severity_range: 严重程度范围（默认用配置）
        """
        if fault_types is None:
            fault_types = FaultInjectionConfig.FAULT_TYPES
        if severity_range is None:
            severity_range = FaultInjectionConfig.FAULT_SEVERITY_RANGE

        df = pd.read_csv(data_path)
        print(f"加载数据: {data_path} ({len(df)} 行)")

        # 加载状态标签（如果有）
        state_labels = None
        if labels_path and os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
            if 'label' in labels_df.columns:
                state_labels = labels_df['label'].values
            elif 'state' in labels_df.columns:
                state_labels = labels_df['state'].values

        rng = np.random.default_rng(self.injector.random_state)

        for i, fault_type in enumerate(fault_types):
            for j in range(max(1, n_scenarios // len(fault_types))):
                severity = rng.uniform(*severity_range)
                scenario_name = f"{bearing}_{fault_type}_{j+1}"
                print(f"\n场景: {scenario_name} (severity={severity:.2f})")

                if state_labels is not None:
                    injected_df, fault_labels = self.injector.inject_in_segments(
                        df, state_labels, bearing, fault_type, severity
                    )
                else:
                    injected_df, fault_labels = self.injector.inject_single(
                        df, bearing, fault_type, severity,
                        start_ratio=rng.uniform(0.1, 0.4),
                        duration_ratio=rng.uniform(0.3, 0.5)
                    )

                # 保存
                out_data = os.path.join(self.output_dir, f"{scenario_name}_data.csv")
                _patch_pyarrow_for_pandas()
                injected_df.to_parquet(out_data.replace('.csv', '.parquet'), index=False)

                out_labels = os.path.join(self.output_dir, f"{scenario_name}_labels.csv")
                _patch_pyarrow_for_pandas()
                pd.DataFrame({'fault_label': fault_labels}).to_parquet(out_labels.replace('.csv', '.parquet'), index=False)

                params = {
                    'bearing': bearing, 'fault_type': fault_type,
                    'severity': float(severity), 'n_fault_samples': int(fault_labels.sum())
                }
                out_params = os.path.join(self.output_dir, f"{scenario_name}_params.json")
                with open(out_params, 'w') as f:
                    json.dump(params, f, indent=2)

                print(f"  故障样本: {int(fault_labels.sum())}/{len(fault_labels)}")

    def generate_grid(
        self,
        data_path: str,
        labels_path: str,
        bearing: str = None,
        percentages: List[float] = None,
        severity_levels: List[int] = None,
        fault_types: List[str] = None,
    ):
        """
        网格化故障注入: 百分比 × 力度 × 故障类型

        当指定 bearing 时，仅注入该设备的 X/Y/Z 列。
        物理模型已考虑 X/Y/Z 三轴方向的差异化效应。

        Output naming: {file_stem}_{fault_type}_pct{%}_sev{n}_data.csv

        Args:
            data_path: 原始 _preprocessed.csv 文件路径
            labels_path: 对应的语义标签 _labels.csv 路径
            bearing: 目标设备（如 'BA1'），None 则注入所有设备
            percentages: 注入比例列表（占NORMAL数据）
            severity_levels: 力度等级列表（1-10）
            fault_types: 故障类型列表
        """
        if percentages is None:
            percentages = FaultInjectionConfig.GRID_PERCENTAGES
        if severity_levels is None:
            severity_levels = FaultInjectionConfig.GRID_SEVERITY_LEVELS
        if fault_types is None:
            fault_types = FaultInjectionConfig.FAULT_TYPES

        df = pd.read_csv(data_path)

        # 加载语义标签
        labels_df = pd.read_csv(labels_path)
        state_col = 'label' if 'label' in labels_df.columns else 'state'
        state_labels = labels_df[state_col].values[:len(df)]

        n_normal = int((state_labels == SemanticConfig.NORMAL).sum())
        file_stem = os.path.splitext(os.path.basename(data_path))[0].replace('_preprocessed', '')

        total = len(fault_types) * len(percentages) * len(severity_levels)
        print(f"\n网格注入: {file_stem}")
        print(f"  NORMAL行数: {n_normal}/{len(df)}")
        print(f"  网格: {len(fault_types)}类型 × {len(percentages)}比例 × {len(severity_levels)}力度 = {total} 场景")

        # ── 统一随机数生成器：每天用文件路径做 hashseed，
        # ── 同一文件内同一 (fault_type, pct, sev) 仍可复现 ─────────
        rng_seed = hash(data_path) & 0xFFFFFFFF
        rng = np.random.default_rng(rng_seed)
        print(f"  [RNG seed=0x{rng_seed:08X}]")

        count = 0
        max_sev = FaultInjectionConfig.GRID_MAX_SEVERITY

        # 决定要注入的设备列表
        target_bearings = [bearing] if bearing else SensorConfig.BEARING_NAMES

        for fault_type in fault_types:
            severity_cap = FaultInjectionConfig.SEVERITY_CAPS.get(fault_type, 1.5)

            for pct in percentages:
                for sev_level in severity_levels:
                    shape = FaultInjectionConfig.SEVERITY_SCALING.get(fault_type, "linear")
                    norm = sev_level / max_sev  # 归一化到 [0, 1]
                    if shape == "sqrt":
                        physics_severity = severity_cap * (norm ** 0.5)
                    elif shape == "log":
                        # 对数升压：极微弱时几乎无感知，随 sev 增加逐渐显现
                        physics_severity = severity_cap * (0.01 + np.log1p(norm * 9) / np.log1p(9))
                    else:
                        physics_severity = severity_cap * norm

                    scenario_name = f"{file_stem}_{fault_type}_pct{int(pct*100)}_sev{sev_level}"

                    # 故障注入（累积多轴承）：
                    # inject_percentage_of_normal 内部用 df.copy() 生成副本再修改并返回，
                    # 不会原地修改传入的 df，所以每次都要用返回值更新 work_df
                    combined_labels = np.zeros(len(df), dtype=int)
                    scenario_clip_reports = []
                    last_inj_info = None
                    work_df = df.copy()  # 累积所有轴承注入结果的工作副本

                    for b in target_bearings:
                        inj, labels = self.injector.inject_percentage_of_normal(
                            work_df, state_labels, b, fault_type,
                            physics_severity, pct, rng=rng
                        )
                        work_df = inj  # 接收返回值：用当前轴承注入结果更新工作副本
                        if self.injector.last_clamp_report is not None:
                            scenario_clip_reports.append(self.injector.last_clamp_report)
                        combined_labels = np.maximum(combined_labels, labels)
                        last_inj_info = work_df.attrs.get("_fault_injection_info", None)

                    # 精简保存：work_df 已是多轴承注入后的完整数据
                    lean_df = pd.DataFrame({
                        'Timestamp': work_df['Timestamp'].values,
                        'time_idx': work_df['time_idx'].values,
                    })
                    for b in target_bearings:
                        for ax in ['X', 'Y', 'Z']:
                            col = f"{b}_{ax}"
                            if col in work_df.columns:
                                lean_df[col] = work_df[col].values

                    out_data = os.path.join(self.output_dir, f"{scenario_name}_data.parquet")
                    _patch_pyarrow_for_pandas()
                    lean_df.to_parquet(out_data, index=False)

                    out_labels = os.path.join(self.output_dir, f"{scenario_name}_labels.csv")
                    _patch_pyarrow_for_pandas()
                    pd.DataFrame({
                        'fault_label': combined_labels,
                        'state': state_labels[:len(combined_labels)],
                    }).to_parquet(out_labels.replace('.csv', '.parquet'), index=False)

                    out_params = os.path.join(self.output_dir, f"{scenario_name}_params.json")
                    params_dict = {
                        'bearing': ','.join(target_bearings),
                        'fault_type': fault_type,
                        'percentage': pct,
                        'severity_level': sev_level,
                        'physics_severity': float(physics_severity),
                        'n_fault_samples': int(combined_labels.sum()),
                        'source_file': os.path.basename(data_path),
                        'rng_seed': int(rng_seed),
                    }
                    # 记录注入随机种子和注入位置信息（用于复现）
                    if last_inj_info is not None:
                        params_dict['injected_segments'] = [
                            [int(s), int(e)] for (s, e) in last_inj_info.get('injected_segments', [])
                        ]
                        params_dict['n_inject_total'] = last_inj_info.get('n_inject_total', 0)
                        params_dict['total_injected'] = last_inj_info.get('total_injected', 0)
                        params_dict['rng_state'] = last_inj_info.get('rng_state', '')
                    with open(out_params, 'w') as f:
                        json.dump(params_dict, f, indent=2)

                    total_clipped_points = int(sum(r['clipped_points'] for r in scenario_clip_reports))
                    total_points = int(sum(r['total_points'] for r in scenario_clip_reports))
                    clip_cols = int(sum(r['clip_cols'] for r in scenario_clip_reports))
                    max_overshoot = max((r['max_overshoot_ratio'] for r in scenario_clip_reports), default=1.0)
                    clip_ratio = total_clipped_points / max(1, total_points)
                    fault_ratio = combined_labels.mean() if len(combined_labels) > 0 else 0.0

                    print(
                        f"  [{count+1:03d}/{total}] {fault_type:12s} pct{int(pct*100):>2d} "
                        f"sev{sev_level:>2d} phys={physics_severity:.3f}/{severity_cap:.3f} "
                        f"fault={int(combined_labels.sum())}/{len(combined_labels)} ({fault_ratio*100:.2f}%) "
                        f"clipped={total_clipped_points}/{total_points} ({clip_ratio*100:.3f}%) "
                        f"clip_cols={clip_cols} max_over={max_overshoot:.2f}x"
                    )

                    count += 1
                    if count % 50 == 0:
                        print(f"  进度: {count}/{total}")

        print(f"  完成: {count} 个场景 → {self.output_dir}")


def visualize_fault_injection(
    original_df: pd.DataFrame,
    injected_df: pd.DataFrame,
    fault_labels: np.ndarray,
    bearing: str,
    title: str,
    output_path: str
):
    """
    可视化故障注入效果

    Args:
        original_df: 原始DataFrame
        injected_df: 注入后DataFrame
        fault_labels: 故障标签
        bearing: 轴承座名称
        title: 图表标题
        output_path: 输出路径
    """
    col_x = f"{bearing}_X"
    if col_x not in original_df.columns:
        print(f"列 {col_x} 不存在，跳过可视化")
        return

    orig = original_df[col_x].values
    injected = injected_df[col_x].values

    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    axes[0].plot(orig, 'b-', alpha=0.7, linewidth=0.5)
    axes[0].set_title('原始数据', fontsize=12)
    axes[0].set_ylabel('振动值')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(injected, 'r-', alpha=0.7, linewidth=0.5)
    fault_indices = np.where(fault_labels > 0)[0]
    if len(fault_indices) > 0:
        axes[1].axvspan(fault_indices[0], fault_indices[-1], alpha=0.2, color='red', label='故障区域')
    axes[1].set_title('注入故障后', fontsize=12)
    axes[1].set_ylabel('振动值')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    diff = injected - orig
    axes[2].plot(diff, 'g-', alpha=0.7, linewidth=0.5)
    axes[2].set_title('故障信号（差值）', fontsize=12)
    axes[2].set_xlabel('样本索引')
    axes[2].set_ylabel('幅值')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存图表: {output_path}")


def create_synthetic_dataset(
    data_dir: str = None,
    output_dir: str = None,
    n_scenarios: int = 4
):
    """
    创建合成故障数据集

    Args:
        data_dir: 数据目录
        output_dir: 输出目录
        n_scenarios: 每个轴承的场景数
    """
    if data_dir is None:
        data_dir = Paths.DATA_DIR
    if output_dir is None:
        output_dir = Paths.FAULT_DATA_DIR

    os.makedirs(output_dir, exist_ok=True)

    data_files = [f for f in os.listdir(data_dir) if f.endswith('_preprocessed.csv')]
    if not data_files:
        print("没有找到数据文件")
        return

    print(f"找到 {len(data_files)} 个数据文件")

    generator = SyntheticFaultDataset(output_dir)

    for data_file in data_files[:3]:
        print(f"\n处理: {data_file}")
        data_path = os.path.join(data_dir, data_file)

        for bearing in SensorConfig.BEARING_NAMES:
            generator.generate_from_raw_data(
                data_path,
                bearing=bearing,
                n_scenarios=n_scenarios
            )

    print(f"\n合成数据集已保存到: {output_dir}")


def main():
    """主函数"""
    print("="*60)
    print("AACE-CM数据集故障注入模块（KDE物理模型）")
    print("="*60)

    Paths.ensure_dirs()
    create_synthetic_dataset()

    print("\n故障注入完成！")
    print(f"输出目录: {Paths.FAULT_DATA_DIR}")


if __name__ == "__main__":
    main()
