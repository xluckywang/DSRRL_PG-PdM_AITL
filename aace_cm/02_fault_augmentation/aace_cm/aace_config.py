"""
AACE-CM configuration for physics-based fault augmentation

基于Per_second数据（1Hz采样率）
架构：直接传感器读数 → 自编码器 → 残差距离 → RL决策
故障注入：KDE-SS-Gen-Framework物理模型（Paris/Arrhenius/Archard/Imbalance）
"""

import os
import sys

# ==================== 路径配置 ====================
class Paths:
    # 基础路径
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = os.path.join(BASE_DIR, "DATA", "Data preprocessing", "AACE-CM", "Per_second")
    EXP_DIR = os.path.join(BASE_DIR, "aace_cm")
    OUTPUT_DIR = os.path.join(EXP_DIR, "05-outputs")

    # KDE-SS-Gen-Framework路径（物理模型所在位置）
    KDE_FRAMEWORK_DIR = os.path.join(BASE_DIR, "KDE-SS-Gen-Framework")

    # 特征和标签目录
    FEATURES_DIR = os.path.join(EXP_DIR, "02-features")
    SEMANTIC_LABELS_DIR = os.path.join(EXP_DIR, "03-semantic_labels")
    FAULT_DATA_DIR = os.path.join(EXP_DIR, "04-fault_data")

    @classmethod
    def device_dir(cls, base_dir: str, device: str) -> str:
        """返回设备子目录路径，如 features/BA1"""
        return os.path.join(base_dir, device)

    @classmethod
    def ensure_dirs(cls):
        """确保所有目录存在（含设备子目录）"""
        for d in [cls.OUTPUT_DIR, cls.FEATURES_DIR,
                  cls.SEMANTIC_LABELS_DIR, cls.FAULT_DATA_DIR]:
            os.makedirs(d, exist_ok=True)
            for dev in SensorConfig.BEARING_NAMES:
                os.makedirs(os.path.join(d, dev), exist_ok=True)

    @classmethod
    def add_kde_to_path(cls):
        """将KDE-SS-Gen-Framework加入Python路径"""
        if cls.KDE_FRAMEWORK_DIR not in sys.path:
            sys.path.insert(0, cls.KDE_FRAMEWORK_DIR)


# ==================== 传感器配置 ====================
class SensorConfig:
    """
    传感器配置

    本项目仅使用 BA1/BA2 两个电机的数据进行方法验证。
    温度列（BA1_T, BA2_T）仅在语义识别阶段辅助理解状态，
    不参与后续聚类/自编码器/RL等分析。
    """

    # 本项目使用的传感器（BA1 + BA2, 含温度）
    ALL_SENSORS = [
        'BA1_X', 'BA1_Y', 'BA1_Z', 'BA1_T',
        'BA2_X', 'BA2_Y', 'BA2_Z', 'BA2_T',
    ]

    # 轴承座名称（用于KDE-SS-Gen物理模型的sensor_name参数）
    BEARING_NAMES = ['BA1', 'BA2']

    # 振动传感器（用于聚类/特征提取/自编码器，不含温度）
    VIBRATION_SENSORS = ['BA1_X', 'BA1_Y', 'BA1_Z', 'BA2_X', 'BA2_Y', 'BA2_Z']

    # 温度传感器（仅用于语义理解，不参与聚类分析）
    TEMP_SENSORS = ['BA1_T', 'BA2_T']

    # 特征列 = 仅振动传感器（直接原始读数，无滑动窗口）
    FEATURE_COLUMNS = VIBRATION_SENSORS

    # 数据量过滤：少于4小时(14400秒=14400行)的文件跳过
    MIN_ROWS = 14400

    @classmethod
    def get_device_sensors(cls, device: str) -> list:
        """获取指定设备的振动列名，如 BA1 → ['BA1_X','BA1_Y','BA1_Z']"""
        return [f'{device}_X', f'{device}_Y', f'{device}_Z']


# ==================== 语义识别配置 ====================
class SemanticConfig:
    """
    语义识别配置 - 6种运行状态

    基于 aace_cm/docs/语义识别状态分析.md 的数据分析结论。
    核心区分方法：振动信号趋势方向。
    - COLD_START: 振动从高往下降（零件磨合→稳定）
    - HOT_START:  振动从低往上升（短停后恢复）
    温度仅用于辅助语义理解，不参与聚类。
    """

    # 6状态定义
    STATES = ['STOPPED', 'COLD_START', 'NORMAL', 'HOT_START', 'SHUTDOWN', 'LOAD_CHANGE']

    # 状态编码
    STOPPED = 0       # 停机：振动幅值低（背景噪声 ~0.08-0.15）
    COLD_START = 1    # 冷启动：振动从高往下降（零件公差磨合）
    NORMAL = 2        # 正常运行：振动稳定在正常范围
    HOT_START = 3     # 热启动：振动从低往上升（短停后恢复）
    SHUTDOWN = 4      # 停机过程：振动突然大幅下降
    LOAD_CHANGE = 5   # 负载变换：振动突发跳变

    STATE_ENCODING = {s: i for i, s in enumerate(STATES)}
    STATE_LABELS = {i: s for i, s in enumerate(STATES)}

    # ---------- 阈值参数（基于 BA1_X 实际数据校准） ----------
    # BA1_X: idle ≈ 0.08-0.15, running ≈ 0.25-0.30
    IDLE_THRESHOLD = 0.18       # 振动幅值 < 此值判为 STOPPED
    NORMAL_MIN = 0.20           # 正常运行下限
    NORMAL_MAX = 0.40           # 正常运行上限
    TRANSIENT_JUMP = 0.15       # 帧间差 > 此值判为瞬态（启停/负载变化）

    # 趋势判定（区分 COLD_START vs HOT_START）
    TREND_WINDOW = 60           # 趋势分析窗口（秒）
    TREND_THRESHOLD = 0.05      # 趋势变化 > 5% 才判定方向

    # 用于分类的参考传感器
    REFERENCE_SENSOR = 'BA1_X'

    # 温度阈值（仅用于语义理解，不用于聚类）
    TEMP_COLD_THRESHOLD = 25.0  # 低于此值可能是冷启动
    TEMP_HOT_THRESHOLD = 28.0   # 高于此值可能是热启动


# ==================== 故障注入配置 ====================
class FaultInjectionConfig:
    """
    故障注入配置 - 基于KDE-SS-Gen-Framework物理模型

    4个物理模型：
    - crack:        Paris疲劳裂纹扩展（da/dN = C·ΔK^m）
    - thermal:      Arrhenius热退化（k = A·e^(-Ea/RT)）
    - mechanical_looseness: Archard轴承磨损（V = k·W·s/H）
    - imbalance:    转子不平衡（F = m·e·ω²）

    2个组合场景：
    - cascade: 级联故障（不平衡→磨损→裂纹）
    - mixed:   混合故障（多种故障并发）
    """

    # 支持的故障场景
    FAULT_TYPES = ['crack', 'thermal', 'mechanical_looseness', 'imbalance']
    COMPOSITE_SCENARIOS = ['cascade', 'mixed']
    ALL_SCENARIOS = FAULT_TYPES + COMPOSITE_SCENARIOS

    # 故障注入参数
    FAULT_SEVERITY_RANGE = (0.3, 1.0)
    FAULT_START_RATIO_RANGE = (0.1, 0.7)
    FAULT_DURATION_RATIO = 0.3

    # 每种故障类型的 severity 上限（severity > cap 时钳位到 cap）
    # 基于 ISO 10816 + 工业文献校准（详见 formal_experiments_v3/config.py 文档）
    SEVERITY_CAPS = {
        'crack': 1.0,            # 裂纹：severity 直接控制裂纹尺寸比例(0→100%)，
                                   # 2026-05-08: 从0.3提高到1.0，使裂纹可扩展到完整范围
        'thermal': 1.0,           # 热退化：2026-05-08 从0.5翻倍，增加振动效应区分度
        'mechanical_looseness': 1.0,  # 松动：是天然放大器，需要较大 cap
        'imbalance': 0.5,         # 不平衡：physics_severity 直接缩放振动幅值
    }

    # severity_level (1~10) → physics_severity 的缩放形状
    # linear:  线性（用于 crack, imbalance）
    # sqrt:    平方根（用于 mechanical_looseness — 低 sev 放大快，高 sev 饱和）
    # log:     对数升压（用于 thermal — 极微弱，早期几乎无感知）
    # 2026-05-08: crack 改为 linear，确保 physics_severity 能在 severity=10 时达到 cap=1.0
    #             使 inject sigmoid 从 ~1.1x (sev=1) 到 ~2.0x (sev=10) 全范围工作
    SEVERITY_SCALING = {
        'crack': "linear",
        'thermal': "log",
        'mechanical_looseness': "sqrt",
        'imbalance': "linear",
    }

    # 不平衡模型参数。
    # 1 Hz downsampled data cannot preserve the real 1x RPM waveform, so the
    # injector maps F=m*e*omega^2 to an observable RMS-level offset.  The
    # earlier conservative values (0.002 kg, 0.001 m) made BA1_X/Y/Z deltas
    # nearly zero after downsampling, so keep the physics-model nominal scale.
    IMBALANCE_PARAMS = {
        'imbalance_mass': 0.010,   # kg
        'eccentricity': 0.005,     # m
    }

    # 裂纹模型参数（Paris 公式控制裂纹扩展速度，摩擦振动噪声由 physics_models.py 注入）
    CRACK_PARAMS = {
        'initial_crack_size': 0.2,    # 初始裂纹尺寸（mm）
        'critical_crack_size': 4.0,   # 临界断裂尺寸（mm）
    }

    # 热退化模型参数（Arrhenius 方程）
    # temp_increase_rate 单位：°C/步（1Hz=1秒），需足够小避免温升远超真实范围
    # 真实轴承热故障：通常在 10~30 分钟内升温 5~20°C
    THERMAL_PARAMS = {
        'temp_increase_rate': 0.005,  # °C/步（severity=1 时末尾约 2°C 温升，合理）
    }

    # 注入后异常值钳位：故障值不超过原始范围的 MAX_RATIO 倍
    # 2026-04-09: 从 3.0 提高到 8.0，避免高 severity 的故障被过度压缩，
    # 导致 crack/imbalance/thermal 各 severity 级别无法区分
    # 新钳位上限 = data_range × 8.0 × (1 + 2.0 × severity_ratio)
    POST_INJECTION_MAX_RATIO = 8.0
    # 是否打印每个通道的钳位明细（False时仅打印场景级摘要）
    POST_INJECTION_CLIP_VERBOSE = False
    # 钳位上限随严重度放宽系数，避免高力度被同一上限“压平”
    # clip_scale = 1 + POST_INJECTION_ADAPTIVE_STRENGTH * severity_ratio
    # severity_ratio in [0, 1]，其中 1 表示达到该故障类型的 SEVERITY_CAP
    POST_INJECTION_ADAPTIVE_STRENGTH = 2.0

    # 注入策略：在 NORMAL + 启停过渡段注入
    INJECT_ONLY_IN_NORMAL = False
    INJECTABLE_STATES = [0, 1, 2, 3, 4]  # STOPPED, COLD_START, NORMAL, HOT_START, SHUTDOWN

    # ---- 网格化注入参数 ----
    # 注入比例: NORMAL数据中被注入故障的连续行占比
    GRID_PERCENTAGES = [0.01, 0.02, 0.05, 0.10, 0.20]
    # 故障力度等级 1-10（满分10映射为该故障类型的 SEVERITY_CAP）
    GRID_SEVERITY_LEVELS = list(range(1, 11))
    GRID_MAX_SEVERITY = 10


# ==================== 故障检测配置 ====================
class DetectionConfig:
    """故障检测配置"""

    # 自编码器配置
    ENCODER_LAYERS = [32, 16, 8]
    DECODER_LAYERS = [16, 32]
    LATENT_DIM = 8
    DROPOUT = 0.1

    # 训练配置
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    # 基线池配置
    BASELINE_WINDOW_RADIUS = 5  # ±5秒（可根据实验调整）
    BASELINE_N_CLUSTERS = 3

    # 训练时排除的运行状态（仅 NORMAL=2 参与自编码器训练和基线池构建，其他状态各自训练独立模型）
    EXCLUDE_STATES_FROM_TRAINING = [0, 1, 3, 4, 5]  # 除 NORMAL(2) 外全部排除

    # 检测阈值
    DETECTION_THRESHOLD_PERCENTILE = 95  # 使用95分位数作为阈值


# ==================== 强化学习配置 ====================
class RLConfig:
    """强化学习维护决策配置（复用IMS配置）"""

    # 状态空间配置
    STATE_DIM = 5
    STATE_FEATURES = [
        'baseline_distance',    # 当前异常分数
        'distance_trend',       # 异常趋势
        'time_since_last',      # 距离上次维护的时间步
        'distance_variance',    # 最近N步的方差
        'cumulative_risk'       # 累积风险
    ]

    # 趋势窗口大小
    TREND_WINDOW = 5

    # 动作空间配置
    ACTION_DIM = 4
    ACTIONS = [
        'continue',      # 0: 继续运行
        'inspect',       # 1: 安排检查
        'preventive',    # 2: 预防性维护
        'immediate'      # 3: 立即停机维护
    ]

    # 奖励函数配置（利润模型）
    REVENUE_PER_STEP = 1.0
    COST_INSPECT = 5.0
    COST_PREVENTIVE = 50.0
    COST_IMMEDIATE = 200.0
    COST_FAILURE = 10000.0

    # 响应充分性系数
    RESPONSE_EFFECTIVENESS = {
        0: 0.0,   # Continue
        1: 0.3,   # Inspect
        2: 0.7,   # Preventive
        3: 1.0,   # Immediate
    }

    # DQN配置
    HIDDEN_DIMS = [128, 128, 64]
    LR = 1e-3
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    TAU = 0.005
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    TARGET_UPDATE = 100

    # 故障阈值基础值
    FAILURE_THRESHOLD_BASE = 100.0

    # 训练轮数
    MAX_EPISODES = 500
    MAX_STEPS_PER_EPISODE = 1000


# ==================== 全局配置 ====================
class Config:
    """全局配置类"""

    paths = Paths
    sensors = SensorConfig
    semantic = SemanticConfig
    fault_injection = FaultInjectionConfig
    detection = DetectionConfig
    rl = RLConfig

    @classmethod
    def summary(cls):
        """打印配置摘要"""
        print("="*80)
        print("AACE-CM数据集配置摘要")
        print("="*80)
        print(f"\n[路径]")
        print(f"  数据目录: {cls.paths.DATA_DIR}")
        print(f"  输出目录: {cls.paths.OUTPUT_DIR}")
        print(f"  特征目录: {cls.paths.FEATURES_DIR}")

        print(f"\n[传感器]")
        print(f"  振动传感器: {cls.sensors.VIBRATION_SENSORS}")
        print(f"  温度传感器（仅语义理解）: {cls.sensors.TEMP_SENSORS}")

        print(f"\n[特征工程]")
        print(f"  模式: 直接传感器读数（无滑动窗口）")
        print(f"  特征列: {cls.sensors.FEATURE_COLUMNS}")

        print(f"\n[语义识别]")
        print(f"  状态类别: {cls.semantic.STATES}")

        print(f"\n[故障注入]")
        print(f"  故障类型: {cls.fault_injection.FAULT_TYPES}")

        print("="*80)


if __name__ == "__main__":
    Config.summary()
    Paths.ensure_dirs()
