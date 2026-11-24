"""
运行状态语义识别模块 — 6状态分类器

基于振动信号识别设备的运行状态：
- STOPPED (0):     停机（振动 < 阈值，背景噪声 ~0.08-0.15）
- COLD_START (1):  冷启动（振动从高往下降，零件磨合）
- NORMAL (2):      正常运行（振动稳定在正常范围）
- HOT_START (3):   热启动（振动从低往上升，短停恢复）
- SHUTDOWN (4):    停机过程（振动突然大幅下降）
- LOAD_CHANGE (5): 负载变换（振动突发跳变后持续）

核心设计依据（来自 docs/语义识别状态分析.md）：
- 仅使用振动信号进行分类，温度不参与聚类
- COLD_START vs HOT_START 通过振动趋势方向区分
- 1Hz采样下启停转换通常只有1-3个样本
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple
from collections import deque

from aace_config import Paths, SensorConfig, SemanticConfig


class StateLabels:
    """状态标签定义"""
    STOPPED = 0
    COLD_START = 1
    NORMAL = 2
    HOT_START = 3
    SHUTDOWN = 4
    LOAD_CHANGE = 5

    LABELS = {0: 'STOPPED', 1: 'COLD_START', 2: 'NORMAL',
              3: 'HOT_START', 4: 'SHUTDOWN', 5: 'LOAD_CHANGE'}
    LABEL_NAMES = ['STOPPED', 'COLD_START', 'NORMAL',
                   'HOT_START', 'SHUTDOWN', 'LOAD_CHANGE']


class RuleBasedClassifier:
    """
    基于振动信号规则的6状态分类器

    分类逻辑（按优先级）：
    1. STOPPED:     振动 < IDLE_THRESHOLD 且持续
    2. SHUTDOWN:    帧间差大幅下降（running → idle 方向）
    3. 启动检测:     帧间差大幅上升（idle → running 方向）
       - COLD_START: 启动后振动趋势下降（磨合过程）
       - HOT_START:  启动后振动趋势上升或稳定（短停恢复）
    4. LOAD_CHANGE: 运行中出现突发跳变（但仍在运行范围）
    5. NORMAL:      稳定运行（默认）
    """

    def __init__(self):
        self.cfg = SemanticConfig
        # 历史缓冲区
        self.history = deque(maxlen=self.cfg.TREND_WINDOW)
        self.prev_state = None
        self.steps_since_start = 0  # 距离上次启动的步数

    def reset(self):
        """重置内部状态"""
        self.history.clear()
        self.prev_state = None
        self.steps_since_start = 0

    def classify_single(self, value: float) -> int:
        """
        分类单个时刻的状态

        Args:
            value: 参考传感器的振动幅值（如 BA1_X）

        Returns:
            状态标签（0-5）
        """
        self.history.append(value)
        h = list(self.history)

        # --- 帧间差 ---
        delta = abs(h[-1] - h[-2]) if len(h) >= 2 else 0.0
        signed_delta = (h[-1] - h[-2]) if len(h) >= 2 else 0.0

        # --- 规则1: STOPPED ---
        if value < self.cfg.IDLE_THRESHOLD:
            self.steps_since_start = 0
            state = StateLabels.STOPPED
            self.prev_state = state
            return state

        # --- 规则2: SHUTDOWN（运行→停机方向的突降） ---
        if (signed_delta < -self.cfg.TRANSIENT_JUMP and
                self.prev_state in (StateLabels.NORMAL, StateLabels.LOAD_CHANGE,
                                    StateLabels.COLD_START, StateLabels.HOT_START, None)):
            state = StateLabels.SHUTDOWN
            self.prev_state = state
            return state

        # --- 规则3: 启动检测（停机→运行方向的突升） ---
        if (signed_delta > self.cfg.TRANSIENT_JUMP and
                self.prev_state in (StateLabels.STOPPED, StateLabels.SHUTDOWN, None)):
            self.steps_since_start = 1
            # 刚启动，还不知道趋势方向，先标为 HOT_START（后续由趋势修正）
            state = StateLabels.HOT_START
            self.prev_state = state
            return state

        # --- 规则3b: 启动后趋势判定（前60秒内） ---
        if (self.steps_since_start > 0 and
                self.steps_since_start <= self.cfg.TREND_WINDOW):
            self.steps_since_start += 1

            if len(h) >= 5:
                recent = h[-5:]
                trend = recent[-1] - recent[0]

                if trend < -self.cfg.TREND_THRESHOLD:
                    # 振动从高往下降 = 冷启动（零件磨合）
                    state = StateLabels.COLD_START
                    self.prev_state = state
                    return state
                else:
                    # 振动持平或上升 = 热启动（短停恢复）
                    state = StateLabels.HOT_START
                    self.prev_state = state
                    return state

        # 超出启动窗口，重置计数
        if self.steps_since_start > self.cfg.TREND_WINDOW:
            self.steps_since_start = 0

        # --- 规则4: LOAD_CHANGE（运行中突发跳变） ---
        if (delta > self.cfg.TRANSIENT_JUMP and
                self.prev_state == StateLabels.NORMAL):
            state = StateLabels.LOAD_CHANGE
            self.prev_state = state
            return state

        # --- 规则5: NORMAL（默认稳定运行） ---
        state = StateLabels.NORMAL
        self.prev_state = state
        return state

    def classify_batch(self, values) -> List[int]:
        """
        批量分类整个序列

        Args:
            values: 振动幅值序列（numpy array/list）

        Returns:
            状态标签列表
        """
        self.reset()
        return [self.classify_single(float(v)) for v in values]


class MLStateClassifier:
    """
    基于阈值+时序梯度的6状态分类器（v2）

    修复原 KMeans 方法的严重缺陷：
    KMeans(k=2) 在以运行数据为主的文件中，会将运行数据自身分成
    "低能量"和"高能量"两簇，导致 >95% 数据被误判为 STOPPED。

    新方法：
    1. 时序间隙感知（不跨时间间隙计算梯度）
    2. 多设备阈值检测 STOPPED（BA1_X 和 BA2_X 都低于阈值）
    3. 短片段平滑（消除 1-2 秒噪声翻转）
    4. 边界转换检测（SHUTDOWN / STARTUP）
    5. 启动类型判定（COLD_START vs HOT_START 通过趋势方向）
    6. 运行中突变检测（LOAD_CHANGE）
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    @staticmethod
    def _compute_vibration_intensity(df: pd.DataFrame, vibration_cols: List[str]) -> pd.Series:
        """计算振动总强度 = sqrt(sum(x²))"""
        xyz_cols = [c for c in vibration_cols if '_X' in c or '_Y' in c or '_Z' in c]
        if not xyz_cols:
            return pd.Series(np.zeros(len(df)), index=df.index)
        vibration = df[xyz_cols].fillna(0)
        return np.sqrt((vibration ** 2).sum(axis=1))

    @staticmethod
    def _smooth_short_segments(labels: np.ndarray, min_length: int = 3) -> np.ndarray:
        """
        平滑短片段：将长度 < min_length 的孤立片段替换为周围状态。
        例：[NORMAL, STOPPED, NORMAL] → [NORMAL, NORMAL, NORMAL]
        """
        result = labels.copy()
        n = len(result)
        i = 0
        while i < n:
            current = result[i]
            j = i
            while j < n and result[j] == current:
                j += 1
            seg_len = j - i
            if seg_len < min_length and i > 0 and j < n:
                if result[i - 1] == result[j]:
                    result[i:j] = result[i - 1]
            i = j
        return result

    def classify_dataframe(self, df: pd.DataFrame, vibration_cols: List[str] = None,
                           device: str = None) -> List[int]:
        """
        对完整DataFrame进行6状态分类（逐样本级别）

        当指定 device 时，仅使用该设备的振动列进行分类。
        BA1 和 BA2 是独立设备，各自拥有独立的运行状态。

        Args:
            df: 包含振动传感器列的DataFrame
            vibration_cols: 振动列名列表（指定 device 时自动从 device 推导）
            device: 设备名称（如 'BA1'），指定后仅用该设备的列

        Returns:
            逐样本的状态标签列表
        """
        if device is not None:
            vibration_cols = [c for c in SensorConfig.get_device_sensors(device)
                              if c in df.columns]
        if vibration_cols is None:
            vibration_cols = [c for c in SensorConfig.VIBRATION_SENSORS if c in df.columns]

        if not vibration_cols:
            raise ValueError("DataFrame 中没有可用的振动列")

        n = len(df)
        labels = np.full(n, StateLabels.NORMAL, dtype=int)

        # ---- Step 1: 时序间隙检测 ----
        if 'time_idx' in df.columns:
            time_diffs = df['time_idx'].diff().fillna(1).values
        else:
            time_diffs = np.ones(n)

        # ---- Step 2: STOPPED 检测 ----
        # 使用振动列中的 X 轴判定（单设备模式只检查该设备）
        ref_sensors = [c for c in vibration_cols if c.endswith('_X')]
        if ref_sensors:
            ref_vals = df[ref_sensors].fillna(0).values
            stopped_mask = np.all(ref_vals < SemanticConfig.IDLE_THRESHOLD, axis=1)
            labels[stopped_mask] = StateLabels.STOPPED
        else:
            ref_vals = df[vibration_cols[0]].fillna(0).values
            labels[ref_vals < SemanticConfig.IDLE_THRESHOLD] = StateLabels.STOPPED

        # ---- Step 3: 短片段平滑 ----
        labels = self._smooth_short_segments(labels, min_length=3)

        # ---- 计算振动强度（用于转换检测） ----
        intensity = self._compute_vibration_intensity(df, vibration_cols).values
        running_mask = labels == StateLabels.NORMAL
        if running_mask.sum() > 10:
            running_std = np.std(intensity[running_mask])
        else:
            running_std = max(np.std(intensity), 0.01)

        # ---- Step 4: SHUTDOWN 检测 ----
        # 运行段的最后 1-2 个样本（紧接 STOPPED 之前）
        for i in range(n - 1):
            if labels[i] != StateLabels.STOPPED and labels[i + 1] == StateLabels.STOPPED:
                labels[i] = StateLabels.SHUTDOWN
                # 前一个样本如果有显著下降也标为 SHUTDOWN
                if (i > 0 and labels[i - 1] == StateLabels.NORMAL
                        and time_diffs[i] <= 5):
                    delta = abs(intensity[i] - intensity[i - 1])
                    if delta > running_std * 0.5:
                        labels[i - 1] = StateLabels.SHUTDOWN

        # ---- Step 5: STARTUP 检测 ----
        # STOPPED → RUNNING 边界处，判定 COLD_START vs HOT_START
        i = 0
        while i < n:
            if labels[i] == StateLabels.STOPPED:
                i += 1
                continue
            # 前一个有效样本是 STOPPED 且无大时间间隙
            if i > 0 and labels[i - 1] == StateLabels.STOPPED and time_diffs[i] <= 5:
                # 收集前方连续运行段用于趋势判定
                future_vals = []
                for j in range(i, min(i + SemanticConfig.TREND_WINDOW, n)):
                    if labels[j] == StateLabels.STOPPED or labels[j] == StateLabels.SHUTDOWN:
                        break
                    if j > i and time_diffs[j] > 5:
                        break
                    future_vals.append(intensity[j])

                if len(future_vals) >= 5:
                    trend = np.polyfit(range(len(future_vals)), future_vals, 1)[0]
                    start_type = (StateLabels.COLD_START if trend < -SemanticConfig.TREND_THRESHOLD
                                  else StateLabels.HOT_START)
                else:
                    start_type = StateLabels.HOT_START

                # 标记启动段（最多 TREND_WINDOW 秒）
                propagated = 0
                for j in range(i, min(i + SemanticConfig.TREND_WINDOW, n)):
                    if labels[j] in (StateLabels.STOPPED, StateLabels.SHUTDOWN):
                        break
                    if j > i and time_diffs[j] > 5:
                        break
                    if labels[j] == StateLabels.NORMAL:
                        labels[j] = start_type
                    propagated += 1

                i += max(propagated, 1)
            else:
                i += 1

        # ---- Step 6: LOAD_CHANGE 检测 ----
        for i in range(1, n):
            if labels[i] != StateLabels.NORMAL:
                continue
            if time_diffs[i] > 5:
                continue
            if labels[i - 1] not in (StateLabels.NORMAL, StateLabels.LOAD_CHANGE):
                continue
            delta = abs(intensity[i] - intensity[i - 1])
            if delta > running_std * 2.0:
                labels[i] = StateLabels.LOAD_CHANGE

        # =====================================================================
        # HYBRID SOTA INJECTION (Expert Knowledge + Viterbi HMM Smoothing)
        # =====================================================================
        labels_arr = np.array(labels)
        
        # 1. Compute physical proxies (Energy and Temperature)
        if vibration_cols:
            vibration = df[vibration_cols].fillna(0).values
            energy_proxy = np.sum(np.abs(vibration), axis=1)
        else:
            energy_proxy = np.zeros(n)
            
        temp_proxy = np.zeros(n)
        running_energy = 0
        for i in range(n):
            running_energy = running_energy * 0.99 + energy_proxy[i] * 0.05
            temp_proxy[i] = running_energy

        # 2. Rule 1: Temperature & Abs Energy logic
        zero_mask = (energy_proxy < 0.05)
        labels_arr[zero_mask] = StateLabels.STOPPED
        
        # Rule 2: Startup Temperature contextualization
        if n > 0:
            temp_percentile = np.percentile(temp_proxy, 75)
            for i in range(1, n):
                if labels_arr[i] in [StateLabels.COLD_START, StateLabels.NORMAL, StateLabels.HOT_START] and labels_arr[i-1] == StateLabels.STOPPED:
                    if temp_proxy[i] > temp_percentile:
                        labels_arr[i] = StateLabels.HOT_START
                    else:
                        labels_arr[i] = StateLabels.COLD_START

        # 3. HMM Viterbi State Transition Smoothing
        n_states = 6
        T = np.full((n_states, n_states), -np.inf)
        T_PENALTY = -20.0
        T_VALID = -2.0
        T_SELF = 0.0
        
        T[0, 0] = T_SELF; T[0, 1] = T_VALID; T[0, 2] = T_PENALTY; T[0, 3] = T_VALID; T[0, 4] = T_PENALTY; T[0, 5] = T_PENALTY;
        T[1, 0] = T_PENALTY; T[1, 1] = T_SELF; T[1, 2] = T_VALID; T[1, 3] = T_PENALTY; T[1, 4] = T_VALID; T[1, 5] = T_PENALTY;
        T[2, 0] = T_PENALTY; T[2, 1] = T_PENALTY; T[2, 2] = T_SELF; T[2, 3] = T_PENALTY; T[2, 4] = T_VALID; T[2, 5] = T_VALID;
        T[3, 0] = T_PENALTY; T[3, 1] = T_PENALTY; T[3, 2] = T_VALID; T[3, 3] = T_SELF; T[3, 4] = T_VALID; T[3, 5] = T_PENALTY;
        T[4, 0] = T_VALID; T[4, 1] = T_PENALTY; T[4, 2] = T_PENALTY; T[4, 3] = T_PENALTY; T[4, 4] = T_SELF; T[4, 5] = T_PENALTY;
        T[5, 0] = T_PENALTY; T[5, 1] = T_PENALTY; T[5, 2] = T_VALID; T[5, 3] = T_PENALTY; T[5, 4] = T_VALID; T[5, 5] = T_SELF;
        
        dp = np.full((n, n_states), -np.inf)
        pointers = np.zeros((n, n_states), dtype=int)
        
        if n > 0:
            dp[0, labels_arr[0]] = 0.0
            for c in range(n_states):
                if c != labels_arr[0]: dp[0, c] = -5.0
                
            for i in range(1, n):
                obs = labels_arr[i]
                emission = np.full(n_states, -5.0)
                emission[obs] = 0.0
                
                for curr_s in range(n_states):
                    for prev_s in range(n_states):
                        score = dp[i-1, prev_s] + T[prev_s, curr_s] + emission[curr_s]
                        if score > dp[i, curr_s]:
                            dp[i, curr_s] = score
                            pointers[i, curr_s] = prev_s
                            
            path = np.zeros(n, dtype=int)
            path[-1] = np.argmax(dp[-1])
            for i in range(n-1, 0, -1):
                path[i-1] = pointers[i, path[i]]
                
            # 4. Short segment fragmentation filter
            min_len = 20
            start_idx = 0
            cur_val = path[0]
            for i in range(1, n):
                if path[i] != cur_val:
                    length = i - start_idx
                    if length < min_len:
                        if start_idx > 0:
                            path[start_idx:i] = path[start_idx-1]
                        else:
                            for j in range(n - i):
                                if path[i+j] != path[i]:
                                    path[start_idx:i] = path[i]
                                    break
                    start_idx = i
                    cur_val = path[i]
                    
            return path.tolist()
            
        return labels_arr.tolist()


class HybridStateClassifier:
    """
    混合状态分类器 — 默认使用ML分类器（阈值 + 时序梯度 + 多设备检测）

    如果ML分类失败则回退到规则分类器。
    """

    def __init__(self, use_ml: bool = True):
        self.rule_classifier = RuleBasedClassifier()
        self.ml_classifier = MLStateClassifier() if use_ml else None
        self.use_ml = use_ml

    def classify(self, value: float) -> Tuple[int, str]:
        """分类单个值，返回 (标签, 分类器名称)"""
        state = self.rule_classifier.classify_single(value)
        return state, 'Rule'

    def classify_batch(self, values) -> List[int]:
        """批量分类（单列，回退到规则方法）"""
        return self.rule_classifier.classify_batch(values)

    def classify_dataframe(self, df: pd.DataFrame, vibration_cols: List[str] = None,
                           device: str = None) -> List[int]:
        """
        基于完整DataFrame的多通道ML分类

        Args:
            df: 包含振动传感器列的DataFrame
            vibration_cols: 振动列名列表
            device: 设备名称（如 'BA1'），指定后仅用该设备的列

        Returns:
            状态标签列表（每行一个）
        """
        if self.use_ml and self.ml_classifier is not None:
            try:
                return self.ml_classifier.classify_dataframe(df, vibration_cols, device=device)
            except Exception as e:
                print(f"ML分类器失败 ({e})，回退到规则方法")

        # 回退：规则方法
        if device:
            ref = f'{device}_X'
        else:
            ref = SemanticConfig.REFERENCE_SENSOR
        if ref in df.columns:
            return self.rule_classifier.classify_batch(df[ref].values)
        return self.rule_classifier.classify_batch(df.iloc[:, 0].values)


def classify_dataframe(df: pd.DataFrame,
                       ref_sensor: str = None,
                       use_ml: bool = True) -> pd.DataFrame:
    """
    对整个DataFrame进行状态分类

    Args:
        df: 包含传感器列的DataFrame
        ref_sensor: 参考振动传感器列名（规则方法用），默认 SemanticConfig.REFERENCE_SENSOR
        use_ml: 是否使用ML分类器

    Returns:
        添加了 'state' 和 'state_name' 列的DataFrame
    """
    classifier = HybridStateClassifier(use_ml=use_ml)

    if use_ml:
        labels = classifier.classify_dataframe(df)
    else:
        ref = ref_sensor or SemanticConfig.REFERENCE_SENSOR
        if ref not in df.columns:
            raise ValueError(f"参考传感器 {ref} 不在DataFrame列中")
        labels = classifier.classify_batch(df[ref].values)

    result = df.copy()
    result['state'] = labels
    result['state_name'] = [StateLabels.LABELS[l] for l in labels]
    return result


def label_data_interactively(features_df: pd.DataFrame, output_path: str = None):
    """
    自动生成初始标签（使用ML分类器）

    Args:
        features_df: 特征DataFrame
        output_path: 标签保存路径

    Returns:
        标签列表
    """
    print("\n" + "="*60)
    print("自动生成初始标签（ML分类器）...")
    print("="*60)

    print(f"数据点数: {len(features_df)}")

    classifier = HybridStateClassifier(use_ml=True)
    labels = classifier.classify_dataframe(features_df)

    # 统计标签分布
    label_counts = pd.Series(labels).value_counts()
    print("\n标签分布:")
    for label, count in sorted(label_counts.items()):
        print(f"  {StateLabels.LABELS[label]}: {count} ({count/len(labels)*100:.1f}%)")

    # 保存标签
    if output_path:
        labels_df = pd.DataFrame({
            'index': range(len(labels)),
            'label': labels,
            'label_name': [StateLabels.LABELS[l] for l in labels]
        })
        labels_df.to_csv(output_path, index=False)
        print(f"\n标签已保存: {output_path}")

    return labels


def process_features_for_classification(features_file: str) -> pd.DataFrame:
    """
    处理特征文件用于分类

    Args:
        features_file: 特征文件路径

    Returns:
        处理后的DataFrame
    """
    print(f"加载特征文件: {features_file}")
    df = pd.read_csv(features_file)

    print(f"特征数: {len(df)}")
    print(f"列数: {len(df.columns)}")

    return df


def visualize_classification(features_df: pd.DataFrame, labels: List[int], output_dir: str):
    """
    可视化分类结果

    Args:
        features_df: 特征DataFrame
        labels: 标签列表
        output_dir: 输出目录
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    os.makedirs(output_dir, exist_ok=True)

    # 获取参考列
    ref = SemanticConfig.REFERENCE_SENSOR
    rms_col = ref if ref in features_df.columns else features_df.columns[0]

    # 颜色映射
    colors = {
        StateLabels.STOPPED: 'gray',
        StateLabels.COLD_START: 'blue',
        StateLabels.NORMAL: 'green',
        StateLabels.HOT_START: 'orange',
        StateLabels.SHUTDOWN: 'red',
        StateLabels.LOAD_CHANGE: 'purple'
    }

    fig, ax = plt.subplots(figsize=(15, 6))

    # 绘制时间序列，按状态着色
    for state in StateLabels.LABELS.keys():
        mask = np.array(labels) == state
        indices = np.where(mask)[0]
        if len(indices) > 0:
            ax.scatter(
                indices,
                features_df.iloc[indices][rms_col],
                c=colors[state],
                label=StateLabels.LABELS[state],
                alpha=0.6,
                s=10
            )

    ax.set_xlabel('样本索引', fontsize=12)
    ax.set_ylabel('振动值', fontsize=12)
    ax.set_title('运行状态分类结果', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, 'state_classification.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存图表: {output_path}")

    # 绘制状态分布饼图
    fig, ax = plt.subplots(figsize=(8, 8))

    label_counts = pd.Series(labels).value_counts()
    sizes = [label_counts.get(i, 0) for i in range(6)]
    labels_pie = [StateLabels.LABELS[i] for i in range(6)]
    colors_pie = [colors[i] for i in range(6)]

    # 过滤掉0值
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels_pie, colors_pie) if s > 0]
    if non_zero:
        sizes, labels_pie, colors_pie = zip(*non_zero)
        ax.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax.set_title('状态分布', fontsize=14)

    output_path = os.path.join(output_dir, 'state_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存图表: {output_path}")


def classify_all_features(output_dir: str = None):
    """
    对所有特征文件进行分类

    Args:
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = Paths.SEMANTIC_LABELS_DIR

    os.makedirs(output_dir, exist_ok=True)

    # 查找特征文件
    features_dir = Paths.FEATURES_DIR
    feature_files = [f for f in os.listdir(features_dir) if f.endswith('_features.csv')]

    if not feature_files:
        print("没有找到特征文件，请先运行特征提取")
        return

    print(f"\n找到 {len(feature_files)} 个特征文件")

    classifier = HybridStateClassifier()

    all_labels = []

    for feature_file in feature_files:
        print(f"\n处理: {feature_file}")

        # 加载特征
        features_path = os.path.join(features_dir, feature_file)
        features_df = process_features_for_classification(features_path)

        # 分类
        labels = label_data_interactively(
            features_df,
            output_path=os.path.join(output_dir, feature_file.replace('_features.csv', '_labels.csv'))
        )

        # 可视化
        visualize_classification(
            features_df,
            labels,
            output_dir
        )

        all_labels.extend(labels)

    print(f"\n总计处理 {len(all_labels)} 个样本")

    # 总体统计
    label_counts = pd.Series(all_labels).value_counts()
    print("\n总体状态分布:")
    for label, count in sorted(label_counts.items()):
        print(f"  {StateLabels.LABELS[label]}: {count} ({count/len(all_labels)*100:.1f}%)")


def main():
    """主函数"""
    print("="*60)
    print("AACE-CM数据集语义识别模块")
    print("="*60)

    # 确保目录存在
    Paths.ensure_dirs()

    # 对所有特征文件进行分类
    classify_all_features(Paths.SEMANTIC_LABELS_DIR)

    print("\n语义识别完成！")
    print(f"标签保存在: {Paths.SEMANTIC_LABELS_DIR}")


if __name__ == "__main__":
    main()
