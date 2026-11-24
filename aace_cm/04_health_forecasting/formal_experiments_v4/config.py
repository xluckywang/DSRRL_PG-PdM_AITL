"""formal_experiments_v3 鍏ㄥ眬閰嶇疆銆?
澶嶇敤 aace_cm 鐨勬暟鎹笌璇箟妯″潡璺緞锛岀粺涓绠＄悊 v3 鐨勮緭鍏ヨ緭鍑虹洰褰曘?
鏁呴殰娉ㄥ叆鍙傛暟璇存槑锛堝弬鑰冨伐涓氭枃鐚笌 ISO 10816锛夛細
  - crack:        Paris 鐤插姵瑁傜汗銆傛棭鏈熸瀬寰急锛?.1~1.5脳 baseline锛夛紝BPFO/BPFI 棰戠巼鍑虹幇鍚庢墠鏄庢樉銆?                  ISO 10816 涓嶅鐏垫晱鈥斺旇绾圭殑鍐插嚮鐗规у湪 10~1000 Hz 缁煎悎鑳介噺閲岃娣规病锛?                  闇楂橀鍖呯粶鍒嗘瀽锛坘urtosis銆乧rest factor锛夋垨澹板彂灏勬墠鑳芥彁鍓嶆鍑恒?                  鎵浠ヨ绾圭殑 sev1 搴旇闈炲父杞诲井锛屼富瑕佷綋鐜板湪缁熻鐗瑰緛鑰岄潪鑲夌溂鍙鲸鐨勫箙鍊笺?  - imbalance:    杞瓙涓嶅钩琛°?脳 RPM 棰戠巼鍗犱紭銆傝交寰笉骞宠　 鈫?3~8脳 baseline锛堝 Zone A/B 鏈哄櫒锛夈?                  涓瓑涓嶅钩琛?鈫?杩涘叆 Zone C锛?~7 mm/s锛夈傛斁澶у櫒 = m路e路蠅虏锛岀嚎鎬х缉鏀俱?  - mechanical:   鏈烘鏉惧姩銆傚己闈炵嚎鎬ф斁澶у櫒锛屾湰韬笉浜х敓鍔涙簮浣嗘斁澶т竴鍒囧凡鏈夋尟鍔ㄣ?                  杞诲井鏉惧姩 鈫?3~5脳 baseline锛涗弗閲嶆澗鍔?鈫?10~20脳锛堝彲杈?Zone D锛夈?                  鐗瑰緛锛?脳銆?脳銆?脳 楂樻璋愭尝锛涙櫄鏈熷嚭鐜?0.5脳 鍒嗛銆?  - thermal:      鐑鍖?鐑笉瀵逛腑銆傞殢娓╁害閫愭笎鎭跺寲锛?脳+2脳 RPM 娣峰悎棰戣氨銆?                  鍗囨俯 鈫?鎸姩缂撳崌 2~5脳 baseline銆備笌 misalignment 鍗忓悓浣滅敤銆?
骞呭害鍙傝冿紙Group 2 鏈哄櫒锛?5~300 kW锛屽垰鎬у熀搴э紝ISO 10816锛夛細
  Zone A: 0~1.12 mm/s  鑹ソ
  Zone B: 1.12~2.8 mm/s  鍙帴鍙?  Zone C: 2.8~7.1 mm/s  瀵嗗垏鐩戞帶
  Zone D: >7.1 mm/s     绔嬪嵆澶勭悊
  瀵逛簬鍘熷 baseline 鏋佷綆锛?0.3 mm/s锛夌殑鏈哄櫒锛岃繘鍏?Zone D 闇瑕佺害 20~25脳 澧炲箙銆?"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    root_dir: Path = ROOT_DIR
    v3_dir: Path = root_dir / "formal_experiments_v3"
    v4_dir: Path = root_dir / "formal_experiments_v4"
    vw_dir: Path = root_dir / "aace_cm"

    # AACE-CM 璇箟浼爣绛剧洰褰曪紙Physics-Informed 6鐘舵佷吉鏍囩锛岀敤浜庝笅娓稿疄楠岃缁冿級
    # 杈撳嚭鐩綍锛歏W-experiments/03-semantic_labels/
    # 鈿狅笍 娉ㄦ剰锛氳繖浜涙槸浼爣绛撅紙Pseudo Labels锛夛紝闈炰汉宸ユ爣娉?Ground Truth
    vw_labels_dir: Path = vw_dir / "03-semantic_labels"

    # AACE-CM 鎸姩鐗瑰緛鐩綍
    vw_features_dir: Path = vw_dir / "02-features"
    aace_per_second_dir: Path = root_dir / "DATA" / "Data preprocessing" / "AACE-CM" / "Per_second"

    # 鏃у浠界洰褰曪紙4 鐘舵佺増鏈紝鏈 SHUTDOWN 姹℃煋锛屼粎鐢ㄤ簬 01_regenerate_semantic_labels.py 璁粌 RF锛?    # WARNING: 浠?01_regenerate_semantic_labels.py 浣跨敤姝よ矾寰勶紱鍏朵粬鎵鏈夊疄楠岀粺涓鐢?vw_labels_dir
    vw_labels_old_dir: Path = Path(str(root_dir).replace("#_00_FINAL/aace_cm", "#_00_FINAL")) / "03-semantic_labels-old"

    outputs_dir: Path = v4_dir / "outputs"
    figures_dir: Path = v4_dir / "figures"
    exp0_data_audit_dir: Path = outputs_dir / "exp0_data_audit"
    exp0_recommended_dates_csv: Path = exp0_data_audit_dir / "recommended_dates.csv"
    exp1_semantic_dir: Path = outputs_dir / "exp1_semantic"
    exp1_figures_dir: Path = outputs_dir / "exp1_figures"
    exp2_fault_data_dir: Path = outputs_dir / "exp2_fault_injected_data"
    exp2_vibration_fault_data_dir: Path = exp2_fault_data_dir
    exp2_temperature_fault_data_dir: Path = vw_dir / "04-temp_fault_data"
    exp2_legacy_vw_fault_data_dir: Path = vw_dir / "04-fault_data"
    exp2_figures_dir: Path = outputs_dir / "exp2_figures"
    exp3_grid_dir: Path = outputs_dir / "exp3_grid_search"
    exp3_server_grid_dir: Path = outputs_dir / "exp3_grid_search_server_20260701"
    exp4_figures_dir: Path = outputs_dir / "exp4_figures"
    exp5_figures_dir: Path = outputs_dir / "exp5_figures"
    exp6_dir: Path = outputs_dir / "exp6_sensitivity"
    exp7_figures_dir: Path = outputs_dir / "exp7_figures"
    exp8_figures_dir: Path = outputs_dir / "exp8_figures"
    exp9_rl_decision_dir: Path = outputs_dir / "exp9_rl_decision"
    exp9_llm_tuning_dir: Path = outputs_dir / "exp9_llm_tuning"
    exp9_figures_dir: Path = outputs_dir / "exp9_figures"
    exp10_dir: Path = outputs_dir / "exp10_ims_validation"
    ims_dir: Path = outputs_dir / "ims_validation"

    def ensure_dirs(self) -> None:
        dirs = [
            self.outputs_dir,
            self.figures_dir,
            self.exp0_data_audit_dir,
            self.exp1_semantic_dir,
            self.exp1_figures_dir,
            self.exp2_fault_data_dir,
            self.exp2_temperature_fault_data_dir,
            self.exp2_figures_dir,
            self.exp3_grid_dir,
            self.exp3_server_grid_dir,
            self.exp4_figures_dir,
            self.exp5_figures_dir,
            self.exp6_dir,
            self.exp7_figures_dir,
            self.exp8_figures_dir,
            self.exp9_rl_decision_dir,
            self.exp9_llm_tuning_dir,
            self.exp9_figures_dir,
            self.exp10_dir,
            self.ims_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CommonConfig:
    device: str = "BA1"
    seeds: List[int] = (42, 43, 44, 45, 46)
    test_ratio: float = 0.5


@dataclass(frozen=True)
class DataAlignmentConfig:
    # Mainline after 2026-06-26: PCHIP-filled source data and temperature-aware
    # fault scenarios. Existing scripts remain compatible until they opt in to
    # the helper functions below.
    active_fault_modality: str = "temperature"
    active_feature_mode: str = "vibration_temperature"
    active_healthy_source: str = "per_second"
    exp2_date_mode: str = "exp0_recommended_eval"


@dataclass(frozen=True)
class DSHIConfig:
    # Final paper-facing health-index weights selected by Exp6 F18.
    # The Exp3 best_config JSON is preserved as the historical model-selection
    # record; downstream Exp5-Exp11 should use these final two-space weights.
    final_w_latent: float = 0.94
    final_w_reconstruction: float = 0.0
    final_w_residual: float = 0.06
    final_score_family: str = "exp6_latent_residual_dshi"
    final_weight_source: str = (
        "Exp6 dual-space weight sensitivity; mean per-scenario AUC optimum"
    )


@dataclass(frozen=True)
class Exp1Config:
    feature_cols: List[str] = ("BA1_X", "BA1_Y", "BA1_Z", "BA1_T")
    n_clusters: int = 6
    baseline_hidden: tuple = (128, 64)
    max_iter: int = 120

    # 蹇熼獙璇佹ā寮?    fast_num_days: int = 8
    figure_random_seed: int = 2026
    figure_num_days: int = 10


@dataclass(frozen=True)
class Exp2Config:
    # 妯℃嫙鏁呴殰娉ㄥ叆鍙傛暟
    fault_types: tuple = ('crack', 'thermal', 'mechanical_looseness', 'imbalance')
    snr_levels: tuple = (0, 5, 10)
    severity_levels: tuple = (2, 4, 6, 8, 10)  # 鍧囧寑瑕嗙洊 1~10锛圴8.1 鏇存柊锛?    fast_combinations: int = 2  # 蹇熸ā寮忎笅姣忕鏁呴殰鍙敓鎴怤绉嶇粍鍚?

@dataclass(frozen=True)
class Exp3Config:
    # 鈹鈹 AE 鏋舵瀯鎼滅储缁村害 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    # MLP 鍩虹嚎
    ae_structures: List[str] = ("SimpleAE", "FlexibleAE", "DeepAE")
    # 鏃跺簭鎰熺煡锛堟柊澧烇級
    ae_structures_extended: List[str] = ("SimpleAE", "FlexibleAE", "DeepAE",
                                          "Conv1DAE", "VAE", "MemAE")

    # 鈹鈹 璺濈搴﹂噺鎼滅储缁村害锛堟柊澧烇級鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    #   euclidean:  娆ф皬璺濈锛堝熀鍑嗭級
    #   mahalanobis: 椹皬璺濈锛堟柟鍚戞晱鎰燂紝鑷姩鏍规嵁姝ｅ父鏁版嵁鍗忔柟宸姞鏉冿級
    distance_metrics: List[str] = ("euclidean", "mahalanobis")

    # 鈹鈹 姝ｅ父鍩哄噯琛ㄧず鎼滅储缁村害 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    # KMeans / GMM: 鐢ㄨ仛绫讳腑蹇冧唬琛ㄦ甯?    # DBSCAN:        鐢ㄥ瘑搴﹀尯鍩熺殑璐ㄥ績浠ｈ〃姝ｅ父
    # LOF:           鐢ㄥ眬閮ㄥ瘑搴︽瘮鍊间綔涓哄紓甯稿垎鏁板熀鍑嗭紙涓嶆槸鑱氱被锛屾槸寮傚父妫娴嬫柟娉曪級
    baseline_methods: List[str] = ("KMeans_K3", "KMeans_K5", "GMM_K3", "GMM_K5",
                                   "DBSCAN", "LOF")
    # 鍏煎鏃у悕锛堥儴鍒嗕唬鐮佸彲鑳藉紩鐢?cluster_methods锛?    cluster_methods: List[str] = ("KMeans_K3", "KMeans_K5", "DBSCAN", "GMM")

    # 鈹鈹 鏃跺簭骞虫粦绐楀彛 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    window_radii: List[int] = (0, 10, 20, 30)
    epochs: List[int] = (50, 100, 200)

    batch_size: int = 512
    learning_rate: float = 1e-3

    # 鈹鈹 Conv1DAE 涓撶敤鍙傛暟 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    conv1d_window_size: int = 7        # 鏃堕棿绐楀彛澶у皬锛埪?姝ラ偦鍩燂級
    conv1d_channels: List[int] = (32, 64, 128)  # Conv1D 閫氶亾鏁?
    # 鈹鈹 VAE 涓撶敤鍙傛暟 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    vae_beta: float = 1.0              # 尾-VAE 瓒呭弬鏁帮紙KL loss 鏉冮噸锛?
    # 鈹鈹 MemAE 涓撶敤鍙傛暟 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    memae_memory_size: int = 128       # Memory Bank 涓師鍨嬪悜閲忕殑鏁伴噺
    memae_sparsity: float = 0.1        # 璁板繂绋鐤忓害锛堟瘡姝ユ洿鏂扮殑璁板繂姣斾緥锛?
    # 鈹鈹 浼樺寲鍣ㄤ笌璁粌 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    optimizer: str = "adam"
    weight_decay: float = 1e-5                    # L2 姝ｅ垯锛岄槻姝㈣繃鎷熷悎鍒版亽绛夋槧灏?
    # 鈹鈹 Bug Fix锛欰dam 鏇夸唬 SGD 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    # SGD 瀵?AE 閲嶆瀯浠诲姟鏀舵暃鏋佹參锛屼笖鏋佹槗闄峰叆鎭掔瓑鏄犲皠锛堟墍鏈夊眰鏉冮噸鈮?锛夈?    # Adam with weight_decay 瀵?AE 浠诲姟鏇撮瞾妫掞紝鐩稿悓 epoch 涓嬮噸鏋勮川閲忔樉钁楁洿楂樸?
    # 鈹鈹 GMM 缁勪欢鏁?鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    gmm_components_k3: int = 3
    gmm_components_k5: int = 5

    # 鈹鈹 Bug Fix锛欴BSCAN 瀛愰噰鏍蜂笂闄愬ぇ骞呮彁楂?鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    # DBSCAN prototype-pool parameters.
    dbscan_eps: float = 0.3
    dbscan_min_samples: int = 5                  # 闄嶄綆 min_samples 閫傚簲浣庣淮绌洪棿
    dbscan_max_points: int = 20000

    # 鈹鈹 璇勫垎铻嶅悎鍙傛暟 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    # Legacy scalar-score alpha kept for compatibility; dual-space scoring uses
    # effective weights from the active Exp3 best config.
    score_alpha: float = 0.7

    # 璇箟闂ㄦ帶锛氫粎瀵?NORMAL 鐘舵佸仛妫娴嬶紱鍏朵綑鐘舵佺洿鎺ヨ烦杩囥?    detect_only_normal: bool = True
    normal_state_id: int = 2

    # 鈹鈹 Early Stopping 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    # 鐩戞帶璁粌闆嗛噸鏋勮宸紝杩炵画 patience 涓?epoch 鏃犳敼鍠勫垯鍋滄
    early_stopping_patience: int = 15
    early_stopping_threshold: float = 1e-4        # 鐩稿鏀瑰杽闃堝?
    # 鈹鈹 蹇熼獙璇佸弬鏁?鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    fast_num_days: int = 3
    fast_window_radii: List[int] = (0,)
    fast_epochs: List[int] = (50,)
    # 蹇熸ā寮忚鐩栵細鏂?AE锛圕onv1DAE锛? 鍩虹嚎 MLP锛團lexibleAE锛?    fast_ae_structures: List[str] = ("FlexibleAE", "Conv1DAE")
    fast_baseline_methods: List[str] = ("KMeans_K3", "LOF")
    fast_distance_metrics: List[str] = ("euclidean", "mahalanobis")
    fast_seeds: List[int] = (42,)

    # 鈹鈹 璇勪及绮掑害鎺у埗 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    # full_evaluation=True: 璇勪及鍏ㄩ儴鍦烘櫙 鈫?杈撳嚭 per-scenario + per-fault-type CSV
    # full_evaluation=False: 浠呴噰鏍峰皯閲忓満鏅?鈫?浠呰緭鍑烘暣浣撹仛鍚堟寚鏍囷紙閫熷害鏇村揩锛?    # 娉ㄦ剰锛氱綉鏍兼悳绱㈤樁娈靛簲璁句负 False锛屽叏閲忚瘎浼扮暀缁欏悗缁垎鏋愯剼鏈?    full_evaluation: bool = False

    # 鈹鈹 璇勪及鍦烘櫙閫夋嫨锛堝彲閰嶇疆鍖栵級鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    # 璇勪及鏃朵粠 exp2 鐢熸垚鐨勬晠闅滃満鏅綉鏍间腑閫夋嫨鍝簺缁勫悎鏉ヨ瘎浼?    #
    # eval_severity_levels: 閫夊摢浜?severity 绛夌骇锛堜粠 GRID_SEVERITY_LEVELS=1~10 涓夛級
    #   浣犺姹傦細1, 3, 6, 9锛堜綆/涓綆/涓珮/楂橈紝瑕嗙洊 1~10 鑼冨洿锛?    eval_severity_levels: List[int] = (1, 3, 6, 9)
    #
    # eval_pct_values: 閫夊摢浜涙敞鍏ユ瘮渚嬶紙浠?GRID_PERCENTAGES=[0.01,0.02,0.05,0.10,0.20] 涓夛級
    #   浣犺姹傦細1%, 5%, 10%锛堜綆/涓?楂橈級
    eval_pct_values: List[float] = (0.01, 0.05, 0.10)
    #
    # max_scenarios_per_combination: 姣忕 (fault_type, severity, pct) 缁勫悎鏈澶氶夊嚑涓満鏅?    #   BA1/BA2 鍚勬湁涓涓棩鏈熸睜锛屼粠涓夋嫨涓嶅悓鏃ユ湡鐨勫満鏅鍔犲鏍锋?    #   璁句负 1 = 姣忕粍鍚堝彧閫?涓棩鏈熺殑鍦烘櫙锛堟?4绫诲瀷脳4sev脳3pct脳1=48涓満鏅級
    #   璁句负 2 = 姣忕粍鍚堥?涓棩鏈燂紙鎬?4脳4脳3脳2=96涓満鏅級
    max_scenarios_per_combination: int = 1
    # Seeded random representative-date sampling for Exp3 model selection.
    scenario_sampling_seed: int = 20260628
    #
    # max_train_healthy: AE 璁粌鐢ㄥ仴搴锋暟鎹笅閲囨牱涓婇檺锛?=涓嶉檺鍒讹紝鐢ㄥ叏閮級
    #   褰撳墠 ~444K Normal 鐐规潵鑷?3澶╂暟鎹?    #   璁句负 200_000 = 淇濈暀绾?45%锛屾瘮鍘熸潵 50K 鏇村厖鍒?    #   璁句负 0 = 涓嶇敤涓嬮噰鏍凤紝鐢ㄥ叏閮?~444K 鐐?    max_train_healthy: int = 200_000
    #
    # max_scorable_points: 姣忓満鏅瘎浼版椂鏈澶氶噰鏍峰灏戜釜鐐癸紙0=鍏ㄩ儴锛?    #   鏃у?00鈫掗殢鏈哄潎鍖閲囨牱锛屾晠闅滆澶皯锛坧ct=1%浠厏3涓級锛岃瘎浼颁笉鍙潬
    #   鏃у?000鈫掓椂搴忓垎灞傞噰鏍凤紝閮ㄥ垎pct=1%鍦烘櫙鏁呴殰琛屼粛鍋忓皯(~32涓?
    #   鏂板?0000鈫掓椂搴忓垎灞傞『搴忛噰鏍凤紙瑙?_stratified_sequential_sample锛夛細
    #     - 淇濈暀鏃跺簭椤哄簭锛屼笉鎵撲贡
    #     - 姝ｅ父娈?鏁呴殰娈垫寜闀垮害姣斾緥鍒嗗眰閲囨牱
    #     - 娈靛唴鍧囧寑閲囨牱锛堜笉鍙噰杈圭晫锛?    #     - pct=1%鏈鍧忓満鏅? 113鏁呴殰琛屸啋閲囨牱绾?3涓紙vs鏃?00鈫掍粎绾?涓級
    #     - pct=5%鍦烘櫙: 鏁呴殰琛屼粠绾?2鈫掔害145锛屾彁鍗囨槑鏄?    max_scorable_points: int = 10000

    # Evaluation mask for injected fault detection.
    # Positive samples are injected fault points (fault_label=1).
    # Negative samples should match the AE training definition: un-injected
    # points whose semantic state equals normal_state_id. Un-injected points
    # from other semantic states are natural/non-normal operating states and
    # are excluded from Exp3 detection metrics.
    filter_eval_negative_to_normal_state: bool = True

@dataclass(frozen=True)
class Exp7Config:
    # 鈹鈹 鏃跺簭棰勬祴绐楀彛鍙傛暟 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    # lookback: 鐪嬪墠澶氬皯姝ュ巻鍙诧紙鍗曚綅锛氭锛?姝?1绉扏1Hz锛?    lookback: int = 30
    # lookforward: 棰勬祴鏈潵澶氬皯姝ワ紙椤跺垔寤鸿 5~15 姝ワ紝杩欓噷鐢?10 姝?鈮?10 绉掞級
    lookforward: int = 10
    # 婊戝姩閲囨牱姝ラ暱锛堣秺灏忛噰鏍疯秺瀵嗭紝鏁版嵁瓒婂浣嗚绠楄秺鎱級
    slide_step: int = 5

    # 鈹鈹 鏁呴殰娉ㄥ叆鍙傛暟锛堜笌 EXP2 淇濇寔涓鑷达級鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    # Severity 绾у埆锛?,4,6,8,10 鍧囧寑瑕嗙洊 1~10 鑼冨洿
    severity_levels: tuple = (2, 4, 6, 8, 10)
    # 娉ㄥ叆姣斾緥锛堝崰 NORMAL 鏁版嵁闀垮害鐨勬瘮渚嬶紝瀵瑰簲 load_fault_scenarios 涓殑 pct 鍊硷級
    # 娉ㄦ剰锛歴["pct"] 瀛樼殑鏄皬鏁板舰寮忥紙濡?0.05, 0.10, 0.20锛夛紝涓嶆槸鐧惧垎姣旀暟瀛?    inject_pcts: tuple = (0.05, 0.10, 0.20)

    # 鈹鈹 闃堝艰绠楃瓥鐣?鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    # 鍔ㄦ侀槇鍊硷細鍙栨墍鏈夊満鏅?baseline_distance 鐨勭 P 鐧惧垎浣嶆暟
    # P 瓒婂ぇ闃堝艰秺楂?鈫?姝ｆ牱鏈秺灏戯紱P 瓒婂皬闃堝艰秺浣?鈫?姝ｆ牱鏈秺澶?    # 椤跺垔寤鸿 50~70锛岃繖閲岀敤 60锛堜繚璇佺害 40% 绐楀彛瓒婄嚎锛屼究浜庡钩琛★級
    threshold_percentile: float = 60.0

    # 鈹鈹 璁粌绛栫暐 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    # 鏄惁浣跨敤鐣欎竴娉曪紙Leave-One-Scenario-Out锛夛細姣忎釜鍦烘櫙杞祦浣滀负娴嬭瘯闆?    # True = 闆舵牱鏈硾鍖栭獙璇侊紙鏇翠弗鏍硷級锛汧alse = 闅忔満鍒掑垎锛堝揩閫熼獙璇侊級
    leave_one_out: bool = True
    # 璁粌闆嗕腑 safe 涓?transition 绐楀彛鐨勭洰鏍囨瘮渚嬶紙balanced = 50/50锛?    # 鑻?safe 绐楀彛澶瀵艰嚧绫诲埆涓嶅钩琛★紝鍙皟楂樻鍊煎亸鍚戞鏍锋湰
    positive_ratio: float = 0.5
    # 姣忎釜鍦烘櫙鏈澶氶噰鏍风殑绐楀彛鏁帮紙閬垮厤澶ф暟鎹泦鏄惧瓨婧㈠嚭锛?=涓嶉檺鍒讹級
    max_windows_per_scenario: int = 0
    # 璁粌 epochs
    train_epochs: int = 30

    # 鈹鈹 蹇熼獙璇佹ā寮?鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    fast_num_scenarios: int = 4   # 蹇熸ā寮忓彧鐢?4 涓満鏅紙1 绉嶆晠闅滅被鍨?脳 4 severity锛?

# 鈹鈹 鏁呴殰娉ㄥ叆鍏ㄥ眬閰嶇疆 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
# 鏈厤缃悓鏃剁敤浜?formal_experiments_v3锛堣鍙栧凡鐢熸垚鏁版嵁锛夊拰
# aace_cm/exp2锛堢敓鎴愭柊鏁呴殰鏁版嵁锛夈?#
# 璁捐鍘熷垯锛堝弬鑰?ISO 10816 + 宸ヤ笟鏂囩尞锛夛細
#   SEVERITY_AMPLITUDE_PROFILES锛歿fault_type: {severity_level: amplitude_multiplier}}
#   amplitude_multiplier = 鏁呴殰娈?RMS / 鍚屼竴鍦烘櫙 NORMAL 娈?RMS
#     1.0 = 涓庢甯告棤宸紓
#     1.5 = 杞诲井锛堣绾规棭鏈?/ 鐑鍖栧垵鏈燂級
#     3.0 = 涓瓑锛堟澗鍔ㄨ交搴?/ 涓嶅钩琛¤交搴︼級
#     8.0 = 涓ラ噸锛堟澗鍔ㄩ噸搴?/ 涓嶅钩琛￠噸搴?鈫?Zone C 杈圭晫锛?#    15.0+ = 鏋佷弗閲嶏紙鈫?Zone D锛?#
#   SEVERITY_CAPS锛氭帶鍒舵敞鍏ュ姏搴︿笂闄愶紙severity > cap 鏃堕挸浣嶅埌 cap锛?#     crack:        鍋忎綆 (0.3)  鈥?瑁傜汗闈犻珮棰戠壒寰佽岄潪澶ф尟骞咃紱鏃╂湡搴斿嚑涔庝笉鍙
#     imbalance:    涓瓑 (0.5)  鈥?绾挎х缉鏀撅紝涓瓑鍔涘害鍗冲彲杈?Zone C
#     thermal:      涓瓑 (0.5)  鈥?鐑鍖栭潬娓╁害婕傜Щ鑰岄潪骞呭硷紱鎸姩澧炲姞鏄瑕佺殑
#     mechanical:   鍋忛珮 (1.0)  鈥?鏉惧姩鏄ぉ鐒舵斁澶у櫒锛岄渶瑕佹洿澶?cap 鎵嶈兘鍙戞尌
#
#   SEVERITY_SCALING锛氭帶鍒?severity_level(1~10) 鈫?physics_severity 鐨勬槧灏勫舰鐘?#     linear:    绾挎э紙鐢ㄤ簬 imbalance锛?#     sqrt:      sqrt 鍗囧帇锛堢敤浜?mechanical 鈥?浣?sev 鏀惧ぇ蹇紝楂?sev 楗卞拰锛?#     log:       log 鍗囧帇锛堢敤浜?thermal/crack 鈥?鏋佸井寮憋紝鏃╂湡搴斿嚑涔庝笉鍙锛?#
#   CLAMP_RATIOS锛氬悗娉ㄥ叆閽充綅涓婇檺 = data_range 脳 ratio
#     ratio 瓒婂ぇ 鈫?鍏佽宄板艰秺澶э紝浣嗗お澶т細鍘嬩笉浣忓紓甯稿?
@dataclass(frozen=True)
class FaultInjectionConfig:
    # 鈹鈹 鏁呴殰绫诲瀷鍒楄〃 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    fault_types: Tuple[str, ...] = ('crack', 'thermal', 'mechanical_looseness', 'imbalance')

    # 鈹鈹 骞呭害鍒嗗竷锛坰everity 1~10 鈫?amplitude multiplier锛夆攢鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    # 鏍煎紡锛歿鏁呴殰绫诲瀷: {sev_level: rms_multiplier}}
    # sev 1 = 杞诲害锛宻ev 10 = 閲嶅害
    # 浣跨敤鏂瑰紡锛歱hysics_severity = SEVERITY_CAPS[fault] 脳 AMPLITUDE_PROFILES[fault][sev]
    SEVERITY_AMPLITUDE_PROFILES: Dict[str, Dict[int, float]] = None

    # 鈹鈹 鍔涘害涓婇檺锛堜紶鍏ョ墿鐞嗘ā鍨嬬殑 severity 鐨勬渶澶у硷級鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    SEVERITY_CAPS: Dict[str, float] = None

    # 鈹鈹 severity_level(1~10) 鈫?physics_severity 鐨勭缉鏀惧舰鐘?鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    # linear:   绾挎э紙鐢ㄤ簬鐗╃悊涓婄嚎鎬х缉鏀剧殑鏁呴殰锛?    # sqrt:     骞虫柟鏍癸紙鐢ㄤ簬鏈烘鏉惧姩 鈥?浣?sev 鏃舵斁澶у揩锛岄珮 sev 蹇熼ケ鍜岋級
    # log:      瀵规暟鍗囧帇锛堢敤浜庤绾?鐑鍖?鈥?鏋佸井寮憋紝鏃╂湡鍑犱箮鏃犳劅鐭ワ級
    SEVERITY_SCALING: Dict[str, str] = None

    # 鈹鈹 鍚庢敞鍏ラ挸浣嶅弬鏁?鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    # clip_bound = data_range 脳 CLAMP_BASE_RATIO 脳 (1 + severity_ratio 脳 ADAPTIVE_STRENGTH)
    CLAMP_BASE_RATIO: float = 3.0   # 鍩虹閽充綅鍊嶇巼
    ADAPTIVE_STRENGTH: float = 2.0   # severity 瓒婂ぇ鍔涘害鐨勯挸浣嶄笂闄愯秺楂?
    # 鈹鈹 娉ㄥ叆姣斾緥锛堟晠闅滅獥鍙ｅ崰鎬?NORMAL 鏁版嵁闀垮害鐨勬瘮渚嬶級鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    GRID_PERCENTAGES: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20)
    GRID_SEVERITY_LEVELS: Tuple[int, ...] = tuple(range(1, 11))

    def __post_init__(self):
        if self.SEVERITY_AMPLITUDE_PROFILES is None:
            object.__setattr__(self, "SEVERITY_AMPLITUDE_PROFILES", {
                "crack": {
                    1: 1.05, 2: 1.10, 3: 1.18, 4: 1.28, 5: 1.40,
                    6: 1.55, 7: 1.75, 8: 2.00, 9: 2.40, 10: 3.00,
                },
                "imbalance": {
                    1: 1.50, 2: 2.00, 3: 2.50, 4: 3.00, 5: 4.00,
                    6: 5.00, 7: 6.00, 8: 8.00, 9: 10.0, 10: 12.0,
                },
                "mechanical_looseness": {
                    1: 3.00, 2: 3.80, 3: 4.50, 4: 5.20, 5: 6.00,
                    6: 7.20, 7: 8.50, 8: 10.0, 9: 12.0, 10: 15.0,
                },
                "thermal": {
                    1: 1.10, 2: 1.20, 3: 1.35, 4: 1.55, 5: 1.80,
                    6: 2.10, 7: 2.50, 8: 3.00, 9: 3.80, 10: 5.00,
                },
            })
        if self.SEVERITY_CAPS is None:
            object.__setattr__(self, "SEVERITY_CAPS", {
                "crack": 1.0,
                "thermal": 1.0,
                "imbalance": 0.5,
                "mechanical_looseness": 1.0,
            })
        if self.SEVERITY_SCALING is None:
            object.__setattr__(self, "SEVERITY_SCALING", {
                "crack": "linear",
                "imbalance": "linear",
                "thermal": "log",
                "mechanical_looseness": "sqrt",
            })
    def physics_severity(self, fault_type: str, sev_level: int) -> float:
        """Map integer severity level (1-10) to the physics-model severity value."""
        cap = self.SEVERITY_CAPS.get(fault_type, 1.0)
        norm = sev_level / 10.0
        shape = self.SEVERITY_SCALING.get(fault_type, "linear")
        if shape == "sqrt":
            phys = norm ** 0.5
        elif shape == "log":
            phys = 0.01 + np.log1p(norm * 9) / np.log1p(9)
        else:
            phys = norm
        return cap * phys

    def amplitude_multiplier(self, fault_type: str, sev_level: int) -> float:
        """Return the RMS amplitude multiplier for a fault type and severity level."""
        return self.SEVERITY_AMPLITUDE_PROFILES.get(fault_type, {}).get(sev_level, 1.0)


import os
import numpy as np  # 鐢ㄤ簬 physics_severity 璁＄畻


# 鈹鈹 璁粌鏁版嵁鏃ユ湡鍒掑垎宸ュ叿 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
# The restricted historical cohort must be supplied locally.
def _load_frozen_cohort() -> Tuple[List[str], List[str]]:
    path = Path(os.environ.get("AACE_FROZEN_SPLIT", ROOT_DIR / "local_inputs" / "frozen_split.json"))
    if not path.is_file():
        raise FileNotFoundError(
            f"Supply a local train/eval manifest for the historical fallback: {path}. "
            "The manuscript cohort uses the separate vetted-date manifest."
        )
    with path.open(encoding="utf-8-sig") as handle:
        split = json.load(handle)
    train, evaluation = split["train"], split["eval"]
    if not all(isinstance(group, list) and group and all(isinstance(day, str) for day in group)
               for group in (train, evaluation)):
        raise ValueError("The frozen split requires nonempty train and eval lists of date strings.")
    if len(set(train + evaluation)) != len(train) + len(evaluation):
        raise ValueError("The frozen split contains duplicate or shared dates.")
    return train, evaluation

_PER_SECOND_DIR = ROOT_DIR / "DATA" / "Data preprocessing" / "AACE-CM" / "Per_second"

def _load_dynamic_healthy_dates() -> List[str]:
    """Load dynamically discovered healthy dates from Per_second when needed."""
    if not _PER_SECOND_DIR.exists():
        return []  # 闄嶇骇锛氫笉闄愬埗鏃ユ湡
    rows = {}
    for f in os.listdir(_PER_SECOND_DIR):
        if not f.endswith('_preprocessed.csv'):
            continue
        path = _PER_SECOND_DIR / f
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                n = sum(1 for _ in fh) - 1
            if n >= 14400:
                rows[f.replace('_preprocessed.csv', '')] = n
        except Exception:
            continue
    if not rows:
        return []
    sorted_dates = sorted(rows.keys())
    n_half = len(sorted_dates) // 2
    return sorted_dates[:n_half]


def _load_healthy_dates() -> List[str]:
    """Return the first-half healthy training dates for v4 experiments."""
    dates = get_exp0_recommended_train_dates()
    return dates if dates else _load_frozen_cohort()[0]


def get_train_dates() -> List[str]:
    """Return healthy training dates for AE/health-baseline training."""
    return _load_healthy_dates()


def get_frozen_eval_dates() -> List[str]:
    """Return frozen evaluation dates."""
    return _load_frozen_cohort()[1]


def get_frozen_all_dates() -> List[str]:
    """Return all frozen v4 dates."""
    return sum(_load_frozen_cohort(), [])


def get_exp0_recommended_dates(path: Path = None) -> List[str]:
    """Return Exp0-vetted dates that are row-aligned, labeled, and usable for Exp2."""
    csv_path = path or PATHS.exp0_recommended_dates_csv
    if not csv_path.exists():
        return []
    dates: List[str] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            date = (row.get("date") or "").strip()
            if not date:
                continue
            both_ok = (row.get("recommended_for_both_devices") or "True").strip().lower()
            if both_ok in {"true", "1", "yes", "y"}:
                dates.append(date)
    return dates


def _split_first_train_second_eval(dates: List[str]) -> Tuple[List[str], List[str]]:
    """Split an ordered cohort into first-half training and second-half injection/eval."""
    if not dates:
        return [], []
    mid = len(dates) // 2
    return dates[:mid], dates[mid:]


def get_exp0_recommended_train_dates() -> List[str]:
    """Return the first half of Exp0-vetted dates for healthy training."""
    train, _ = _split_first_train_second_eval(get_exp0_recommended_dates())
    return train


def get_exp0_recommended_eval_dates() -> List[str]:
    """Return the second half of Exp0-vetted dates for Exp2 injection/evaluation."""
    _, eval_dates = _split_first_train_second_eval(get_exp0_recommended_dates())
    return eval_dates


def get_experiment_dates() -> List[str]:
    """Return the full Exp0-vetted 58-day experiment cohort."""
    dates = get_exp0_recommended_dates()
    return dates if dates else get_frozen_all_dates()


def get_exp2_generation_dates(date_mode: str = None) -> List[str]:
    """Return the date cohort that Exp2-style injection scripts should use."""
    mode = (date_mode or DATA_ALIGN.exp2_date_mode).lower()
    if mode in {"exp0_recommended_eval", "exp0_eval", "eval", "injection"}:
        dates = get_exp0_recommended_eval_dates()
        return dates if dates else get_frozen_eval_dates()
    if mode in {"exp0_recommended_train", "exp0_train"}:
        dates = get_exp0_recommended_train_dates()
        return dates if dates else get_train_dates()
    if mode == "exp0_recommended":
        dates = get_exp0_recommended_dates()
        return dates if dates else get_frozen_all_dates()
    if mode == "frozen_eval":
        return get_frozen_eval_dates()
    if mode == "frozen_all":
        return get_frozen_all_dates()
    if mode == "train":
        return get_train_dates()
    raise ValueError(f"Unknown date_mode: {date_mode}")


def get_fault_data_dir(modality: str) -> Path:
    """Map a fault-data modality name to the canonical directory."""
    key = modality.lower().replace("-", "_")
    if key in {"temperature", "temp", "vibration_temperature", "thermal_aware"}:
        return PATHS.exp2_temperature_fault_data_dir
    if key in {"vibration", "vib"}:
        return PATHS.exp2_vibration_fault_data_dir
    if key in {"legacy", "legacy_vibration", "vw_legacy"}:
        return PATHS.exp2_legacy_vw_fault_data_dir
    raise ValueError(f"Unknown fault-data modality: {modality}")


def get_active_fault_data_dir(modality: str = None) -> Path:
    """Return the fault-data directory selected for downstream experiments."""
    return get_fault_data_dir(modality or DATA_ALIGN.active_fault_modality)


def get_exp3_results_dir() -> Path:
    """Return the active Exp3 result directory used by downstream experiments."""
    server_best = PATHS.exp3_server_grid_dir / "exp3_best_config.json"
    if server_best.exists():
        return PATHS.exp3_server_grid_dir
    return PATHS.exp3_grid_dir


def get_exp3_best_config_file() -> Path:
    """Return the active Exp3 best-config JSON path."""
    return get_exp3_results_dir() / "exp3_best_config.json"


def load_exp3_best_config(required: bool = False) -> dict:
    """Load the active Exp3 best config for Exp4-Exp11 downstream reuse."""
    cfg_path = get_exp3_best_config_file()
    if not cfg_path.exists():
        if required:
            raise FileNotFoundError(f"Active Exp3 best config not found: {cfg_path}")
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("ae_structure", "Conv1DAE")
    cfg.setdefault("baseline_method", "DBSCAN")
    cfg.setdefault("cluster_method", cfg.get("baseline_method", "DBSCAN"))
    cfg.setdefault("distance_metric", "euclidean")
    cfg.setdefault("latent_dim", 8)
    cfg.setdefault("window_w", 30)
    cfg.setdefault("epochs", 100)
    cfg.setdefault("seed", COMMON.seeds[0])
    cfg.setdefault("ours_variant", "dual_space")
    cfg.setdefault("score_family", "dual_space_health_index")
    cfg.setdefault("effective_w_latent", 0.50)
    cfg.setdefault("effective_w_reconstruction", 0.25)
    cfg.setdefault("effective_w_residual", 0.25)
    cfg["cluster_method"] = cfg.get("baseline_method", cfg.get("cluster_method", "DBSCAN"))
    cfg["_source_path"] = str(cfg_path)
    return cfg


def get_final_dshi_weights() -> Tuple[float, float, float]:
    """Return final paper-facing DSHI weights as latent/reconstruction/residual."""
    return (
        float(DSHI.final_w_latent),
        float(DSHI.final_w_reconstruction),
        float(DSHI.final_w_residual),
    )


def get_final_dshi_config() -> dict:
    """Return final DSHI metadata used by downstream Exp5-Exp11 scripts."""
    w_latent, w_recon, w_residual = get_final_dshi_weights()
    return {
        "score_family": DSHI.final_score_family,
        "effective_w_latent": w_latent,
        "effective_w_reconstruction": w_recon,
        "effective_w_residual": w_residual,
        "weight_source": DSHI.final_weight_source,
    }


def get_feature_cols(device: str = None, mode: str = None) -> List[str]:
    """Return canonical model input columns for a device and feature mode."""
    dev = device or COMMON.device
    key = (mode or DATA_ALIGN.active_feature_mode).lower().replace("-", "_")
    vibration = [f"{dev}_X", f"{dev}_Y", f"{dev}_Z"]
    temperature = [f"{dev}_T"]
    if key in {"vibration", "vib"}:
        return vibration
    if key in {"temperature", "temp"}:
        return temperature
    if key in {"vibration_temperature", "vib_temp", "temperature_vibration", "all"}:
        return vibration + temperature
    raise ValueError(f"Unknown feature mode: {mode}")


def get_feature_data_file(device: str, date: str, mode: str = None) -> Path:
    """Return the healthy-day feature CSV for a device/date under the feature mode."""
    key = (mode or DATA_ALIGN.active_feature_mode).lower().replace("-", "_")
    if key in {"vibration_temperature", "vib_temp", "temperature_vibration", "temperature", "temp", "all"}:
        return PATHS.aace_per_second_dir / f"{date}_preprocessed.csv"
    if key in {"vibration", "vib"}:
        return PATHS.vw_features_dir / device / f"{date}_features.csv"
    raise ValueError(f"Unknown feature mode: {mode}")


def get_label_file(device: str, date: str) -> Path:
    """Return the semantic label CSV for a device/date."""
    return PATHS.vw_labels_dir / device / f"{date}_labels.csv"


def get_required_source_cols(device: str = None) -> List[str]:
    """Return source columns required for temperature-aware AACE-CM experiments."""
    dev = device or COMMON.device
    return ["Timestamp", "time_idx", f"{dev}_T", f"{dev}_X", f"{dev}_Y", f"{dev}_Z"]


def is_frozen_experiment_date(date: str) -> bool:
    """Return whether a date belongs to the frozen v4 experiment cohort."""
    return date in get_frozen_all_dates()


def is_train_date(date: str) -> bool:
    """Return whether a date belongs to the healthy training split."""
    train = _load_healthy_dates()
    return date in train if train else True  # 闄嶇骇锛氭壘涓嶅埌鏃堕粯璁ゆ斁琛?

# 鈹鈹 鍏ㄥ眬鍗曚緥锛堟ā鍧楀姞杞芥椂璁＄畻涓娆★級鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
_TRAIN_DATES_SET: set = set()   # 寤惰繜鍒濆鍖?

def train_dates_set() -> set:
    """Return the lazily loaded healthy training-date set."""
    global _TRAIN_DATES_SET
    if not _TRAIN_DATES_SET:
        _TRAIN_DATES_SET = set(_load_healthy_dates())
    return _TRAIN_DATES_SET


PATHS = Paths()
COMMON = CommonConfig()
DATA_ALIGN = DataAlignmentConfig()
DSHI = DSHIConfig()
EXP1 = Exp1Config()
EXP2 = Exp2Config()
EXP3 = Exp3Config()
EXP7 = Exp7Config()
FAULT_INJECT = FaultInjectionConfig()
