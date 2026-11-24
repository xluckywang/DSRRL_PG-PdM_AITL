"""
shared_ae_cache.py
==================
所有 exp5–exp11 实验共享的 VAE checkpoint 加载器。

所有实验统一从以下文件加载训练好的 VAE（不再各自重训）：
    outputs/exp4_figures/best_vae/best_vae.pt

该文件由 04_exp4_train_and_save_best.py 一次性生成，包含：
    ae_state_dict, scaler_mean_, scaler_scale_, scaler_n_features_in_,
    pool_latent, pool_residual,
    ae_structure, latent_dim, alpha, w, cluster_method, epochs, seed, loss
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ── 动态构造路径（向上两级找到 formal_experiments_v4/） ──────────────────────
_CACHE_PKG_DIR = Path(__file__).resolve().parent  # shared_ae_cache.py 所在目录
_EXP4_DIR = _CACHE_PKG_DIR                        # shared_ae_cache.py 在 formal_experiments_v4/ 下
_EXP4_FIG_DIR = _EXP4_DIR / "outputs" / "exp4_figures"
_BEST_VAE_DIR = _EXP4_FIG_DIR / "best_vae"
_BEST_VAE_PATH = _BEST_VAE_DIR / "best_vae.pt"


# ── 懒加载缓存（避免重复读文件） ─────────────────────────────────────────────
_cached_ckpt: dict | None = None
_cached_ref_stats: dict[tuple, dict] = {}

HEALTH_INDEX_NAME = "Dual-Space Health Index"
_FALLBACK_FINAL_DSHI_WEIGHTS = (0.94, 0.0, 0.06)


def get_final_dshi_weights() -> tuple[float, float, float]:
    """Return final Exp6-selected DSHI weights as latent/reconstruction/residual."""
    try:
        from config import get_final_dshi_weights as _cfg_weights
        weights = _cfg_weights()
    except Exception:
        weights = _FALLBACK_FINAL_DSHI_WEIGHTS
    return tuple(float(v) for v in weights)


def get_final_dshi_config() -> dict:
    """Return final Exp6-selected DSHI metadata."""
    try:
        from config import get_final_dshi_config as _cfg
        return dict(_cfg())
    except Exception:
        w_latent, w_recon, w_residual = get_final_dshi_weights()
        return {
            "score_family": "exp6_latent_residual_dshi",
            "effective_w_latent": w_latent,
            "effective_w_reconstruction": w_recon,
            "effective_w_residual": w_residual,
            "weight_source": "Exp6 dual-space weight sensitivity",
        }


def _load_raw_ckpt() -> dict:
    global _cached_ckpt
    if _cached_ckpt is not None:
        return _cached_ckpt
    if not _BEST_VAE_PATH.exists():
        raise FileNotFoundError(
            f"VAE 缓存不存在: {_BEST_VAE_PATH}\n"
            "请先运行: python 04_exp4_train_and_save_best.py"
        )
    import torch
    ckpt = torch.load(_BEST_VAE_PATH, map_location="cpu", weights_only=False)
    _cached_ckpt = ckpt
    return ckpt


def get_best_vae_info() -> dict:
    """返回 checkpoint 的元信息（不加载模型）。"""
    ckpt = _load_raw_ckpt()
    final_dshi = get_final_dshi_config()
    return {
        "ae_structure": ckpt.get("ae_structure"),
        "latent_dim":   int(ckpt.get("latent_dim", 4)),
        "alpha":        float(ckpt.get("alpha", 0.3)),
        "w":            int(ckpt.get("window_w", 0)),
        "cluster_method": ckpt.get("cluster_method", "DBSCAN"),
        "epochs":       int(ckpt.get("epochs_trained", 200)),
        "seed":         int(ckpt.get("seed", 42)),
        "loss":         float(ckpt.get("final_loss", 0.0)),
        "path":         str(_BEST_VAE_PATH),
        "health_index_name": HEALTH_INDEX_NAME,
        "feature_cols": ckpt.get("feature_cols"),
        "score_family": final_dshi["score_family"],
        "effective_w_latent": final_dshi["effective_w_latent"],
        "effective_w_reconstruction": final_dshi["effective_w_reconstruction"],
        "effective_w_residual": final_dshi["effective_w_residual"],
        "weight_source": final_dshi["weight_source"],
    }


def load_best_vae() -> tuple:
    """
    加载训练好的 VAE + scaler + 基线池。

    Returns
    -------
    (ae, scaler, pool_latent, pool_residual, info)
        ae              : 已是 eval() 模式，可直接用于 encode/reconstruct
        scaler          : sklearn StandardScaler，已 fit 过
        pool_latent     : ndarray (K, latent_dim)   — 健康 latent 聚类中心
        pool_residual   : ndarray (K, n_features)  — 健康重构残差聚类中心
        info            : dict，元信息（ae_structure, latent_dim, alpha, …）

    用法示例
    --------
    >>> ae, scaler, pool_latent, pool_residual, info = load_best_vae()
    >>> x_norm = scaler.transform(x_raw)
    >>> with torch.no_grad():
    ...     z = ae.encode(torch.tensor(x_norm, dtype=torch.float32)).numpy()
    """
    import torch
    import pickle
    from sklearn.preprocessing import StandardScaler

    ckpt = _load_raw_ckpt()

    # ── 重建模型 ─────────────────────────────────────────────────────────────
    import importlib
    sys.path.insert(0, str(_EXP4_DIR))
    exp4_mod = importlib.import_module("04_exp4_train_and_save_best")
    _ae_dims = getattr(exp4_mod, "_ae_dims")
    _build_ae = getattr(exp4_mod, "_build_ae")
    VAE = getattr(exp4_mod, "VAE")
    Conv1DAutoEncoder = getattr(exp4_mod, "Conv1DAutoEncoder", None)

    ae_struct  = ckpt["ae_structure"]
    latent_dim = int(ckpt["latent_dim"])
    n_feat     = int(ckpt["scaler_n_features_in"])
    try:
        from config import COMMON, get_feature_cols
        active_cols = get_feature_cols(COMMON.device)
        assert n_feat == len(active_cols), (
            f"VAE checkpoint input_dim={n_feat}, but active feature columns are "
            f"{len(active_cols)}: {active_cols}. Re-run 04_exp4_train_and_save_best.py "
            "after data alignment."
        )
    except ImportError:
        pass

    if ae_struct == "Conv1DAE":
        if Conv1DAutoEncoder is None:
            raise RuntimeError("Conv1DAE checkpoint found, but Conv1DAutoEncoder is unavailable.")
        ae = Conv1DAutoEncoder(
            window_size=int(ckpt.get("conv1d_window_size", 7)),
            channels=tuple(ckpt.get("conv1d_channels", (16, 8, 4))),
            latent_dim=latent_dim,
            input_dim=n_feat,
        )
    elif ae_struct == "VAE":
        # 兼容新旧 checkpoint（新版有 encoder_dims，旧版从 _ae_dims 推导）
        enc_dims = ckpt.get("encoder_dims") or [16, 8, 4]
        dec_dims = ckpt.get("decoder_dims") or [8, 16]
        ae = VAE(
            input_dim=n_feat,
            encoder_dims=enc_dims,
            decoder_dims=dec_dims,
            latent_dim=latent_dim,
        )
    else:
        # FlexibleAE/SimpleAE 用 _ae_dims 映射表（与 exp4 保存时一致）
        enc_d, dec_d = _ae_dims(ae_struct, latent_dim)
        ae = _build_ae(ae_struct, n_feat, latent_dim)

    ae.load_state_dict(ckpt["ae_state_dict"])
    ae.eval()

    # ── 重建 scaler ─────────────────────────────────────────────────────────
    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.n_features_in_ = n_feat
    scaler.n_samples_seen_ = int(ckpt.get("scaler_n_samples_seen", len(scaler.mean_)))

    # ── 基线池 ─────────────────────────────────────────────────────────────
    pool_latent   = ckpt["pool_latent"]
    pool_residual = ckpt["pool_residual"]

    info = get_best_vae_info()
    return ae, scaler, pool_latent, pool_residual, info


def _window_smooth_vec(z: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return z
    n = len(z)
    out = np.empty_like(z)
    for i in range(n):
        l = max(0, i - radius)
        r = min(n, i + radius + 1)
        out[i] = z[l:r].mean(axis=0)
    return out


def _dist_to_pool(x: np.ndarray, pool: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - pool[None, :, :]
    return np.min(np.sqrt(np.sum(diff ** 2, axis=2)), axis=1)


def _build_sliding_windows(x: np.ndarray, window_size: int) -> np.ndarray:
    n = len(x)
    pad_left = window_size // 2
    pad_right = window_size // 2
    padded = np.pad(x, ((pad_left, pad_right), (0, 0)), mode="edge")
    windows = np.zeros((n, window_size, x.shape[1]), dtype=np.float32)
    for i in range(n):
        windows[i] = padded[i:i + window_size]
    return windows


def encode_array(ae, x_norm: np.ndarray, batch_size: int = 10000) -> np.ndarray:
    """Encode normalized samples with the cached AE/VAE."""
    import torch

    device = next(ae.parameters()).device
    if getattr(ae, "_is_conv1d_ae", False):
        window_size = int(getattr(ae, "window_size", 7))
        x_model = _build_sliding_windows(x_norm, window_size)
    else:
        x_model = x_norm
    out = []
    ae.eval()
    with torch.no_grad():
        for i in range(0, len(x_model), batch_size):
            xb = torch.tensor(x_model[i:i + batch_size], dtype=torch.float32, device=device)
            if hasattr(ae, "encode_latent_params"):
                z, _ = ae.encode_latent_params(xb)
            else:
                z = ae.encode(xb)
            out.append(z.detach().cpu().numpy())
    return np.vstack(out).astype(np.float32)


def reconstruct_array(ae, x_norm: np.ndarray, batch_size: int = 10000) -> np.ndarray:
    """Deterministically reconstruct normalized samples with the cached AE/VAE."""
    import torch

    device = next(ae.parameters()).device
    is_conv = getattr(ae, "_is_conv1d_ae", False)
    if is_conv:
        window_size = int(getattr(ae, "window_size", 7))
        x_model = _build_sliding_windows(x_norm, window_size)
    else:
        x_model = x_norm
    out = []
    ae.eval()
    with torch.no_grad():
        for i in range(0, len(x_model), batch_size):
            xb = torch.tensor(x_model[i:i + batch_size], dtype=torch.float32, device=device)
            if hasattr(ae, "encode_latent_params"):
                z, _ = ae.encode_latent_params(xb)
                recon = ae.decode(z)
            else:
                recon = ae(xb)
                if isinstance(recon, tuple):
                    recon = recon[0]
            out.append(recon.detach().cpu().numpy())
    recon_all = np.vstack(out).astype(np.float32)
    if is_conv:
        center = int(getattr(ae, "window_size", 7)) // 2
        return recon_all[:, center, :].astype(np.float32)
    return recon_all


def compute_dual_space_components(
    x: np.ndarray,
    ae=None,
    scaler=None,
    pool_latent: np.ndarray | None = None,
    pool_residual: np.ndarray | None = None,
    *,
    x_is_normalized: bool = False,
    window_w: int | None = None,
) -> dict:
    """Return latent, reconstruction and residual-space distances from best_vae.pt."""
    if ae is None or scaler is None or pool_latent is None or pool_residual is None:
        ae, scaler, pool_latent, pool_residual, info = load_best_vae()
    else:
        info = get_best_vae_info()

    w = int(info.get("w", 0) if window_w is None else window_w)
    x_norm = x.astype(np.float32) if x_is_normalized else scaler.transform(x).astype(np.float32)
    x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    z = encode_array(ae, x_norm)
    z_s = _window_smooth_vec(z, w)
    recon = reconstruct_array(ae, x_norm)
    residual = (x_norm - recon).astype(np.float32)
    recon_err = np.mean((x_norm - recon) ** 2, axis=1).astype(np.float32)
    dist_latent = _dist_to_pool(z_s, pool_latent).astype(np.float32)
    dist_residual = _dist_to_pool(residual, pool_residual).astype(np.float32)

    return {
        "x_norm": x_norm,
        "z": z,
        "z_smooth": z_s,
        "recon": recon,
        "residual": residual,
        "recon_err": recon_err,
        "dist_latent": dist_latent,
        "dist_residual": dist_residual,
    }


def load_training_healthy_raw(device: str = "BA1") -> np.ndarray:
    """Load the same healthy-only AACE-CM training background used by Exp4."""
    import pandas as pd
    from config import (
        COMMON,
        EXP3,
        get_feature_cols,
        get_feature_data_file,
        get_label_file,
        get_train_dates,
        train_dates_set,
    )

    device = device or COMMON.device
    train_set = train_dates_set()
    dates = [
        d for d in get_train_dates()
        if get_feature_data_file(device, d).exists() and get_label_file(device, d).exists()
    ]
    dates = [d for d in dates if d in train_set]
    if not dates:
        raise RuntimeError("No healthy training dates found for shared VAE cache.")

    feature_cols = get_feature_cols(device)
    chunks = []
    for d in dates:
        feat_df = pd.read_csv(get_feature_data_file(device, d))
        assert all(c in feat_df.columns for c in feature_cols), (
            f"Missing feature columns for {device} {d}: "
            f"{[c for c in feature_cols if c not in feat_df.columns]}"
        )
        lab_df = pd.read_csv(get_label_file(device, d))
        state_col = "state" if "state" in lab_df.columns else "label"
        n = min(len(feat_df), len(lab_df))
        mask = lab_df[state_col].iloc[:n].values == EXP3.normal_state_id
        chunks.append(feat_df[feature_cols].iloc[:n].values[mask])
    return np.vstack(chunks).astype(np.float32)


def get_health_reference_stats(
    ae=None,
    scaler=None,
    pool_latent: np.ndarray | None = None,
    pool_residual: np.ndarray | None = None,
    *,
    device: str = "BA1",
    window_w: int | None = None,
) -> dict:
    """Healthy-only z-score references for the Dual-Space Health Index."""
    global _cached_ref_stats
    if ae is None or scaler is None or pool_latent is None or pool_residual is None:
        ae, scaler, pool_latent, pool_residual, info = load_best_vae()
    else:
        info = get_best_vae_info()

    w = int(info.get("w", 0) if window_w is None else window_w)
    key = (str(device), w)
    if key in _cached_ref_stats:
        return _cached_ref_stats[key]

    x_healthy = load_training_healthy_raw(device=device)
    comp = compute_dual_space_components(
        x_healthy, ae, scaler, pool_latent, pool_residual,
        x_is_normalized=False, window_w=w,
    )
    _cached_ref_stats[key] = {
        "dist_latent_mean": float(comp["dist_latent"].mean()),
        "dist_latent_std": float(comp["dist_latent"].std()),
        "recon_err_mean": float(comp["recon_err"].mean()),
        "recon_err_std": float(comp["recon_err"].std()),
        "dist_residual_mean": float(comp["dist_residual"].mean()),
        "dist_residual_std": float(comp["dist_residual"].std()),
        "window_w": w,
        "device": str(device),
    }
    return _cached_ref_stats[key]


def compute_dual_space_health_index(
    x: np.ndarray,
    ae=None,
    scaler=None,
    pool_latent: np.ndarray | None = None,
    pool_residual: np.ndarray | None = None,
    *,
    x_is_normalized: bool = False,
    ref_stats: dict | None = None,
    device: str = "BA1",
    window_w: int | None = None,
    weights: tuple[float, float, float] | None = None,
    return_components: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    Compute the paper-facing Dual-Space Health Index.

    By default this uses the final Exp6-selected two-space weights:
    latent=0.94, reconstruction=0.00, residual=0.06.
    """
    if ae is None or scaler is None or pool_latent is None or pool_residual is None:
        ae, scaler, pool_latent, pool_residual, _ = load_best_vae()
    comp = compute_dual_space_components(
        x, ae, scaler, pool_latent, pool_residual,
        x_is_normalized=x_is_normalized, window_w=window_w,
    )
    if ref_stats is None:
        ref_stats = get_health_reference_stats(
            ae, scaler, pool_latent, pool_residual,
            device=device, window_w=window_w,
        )

    dl_n = (comp["dist_latent"] - ref_stats["dist_latent_mean"]) / (
        ref_stats["dist_latent_std"] + 1e-12
    )
    re_n = (comp["recon_err"] - ref_stats["recon_err_mean"]) / (
        ref_stats["recon_err_std"] + 1e-12
    )
    dr_n = (comp["dist_residual"] - ref_stats["dist_residual_mean"]) / (
        ref_stats["dist_residual_std"] + 1e-12
    )
    if weights is None:
        weights = get_final_dshi_weights()
    w_latent, w_recon, w_residual = weights
    total = float(w_latent + w_recon + w_residual)
    if abs(total) < 1e-12:
        raise ValueError("Dual-Space Health Index weights must not sum to zero.")
    score = (
        (w_latent / total) * dl_n
        + (w_recon / total) * re_n
        + (w_residual / total) * dr_n
    ).astype(np.float32)

    if not return_components:
        return score

    comp_out = dict(comp)
    comp_out.update({
        "dist_latent_z": dl_n.astype(np.float32),
        "recon_err_z": re_n.astype(np.float32),
        "dist_residual_z": dr_n.astype(np.float32),
        "health_index": score,
        "weights": tuple(float(v) / total for v in weights),
        "ref_stats": ref_stats,
        "health_index_name": HEALTH_INDEX_NAME,
    })
    return score, comp_out


def compute_residual_windows_from_raw(
    x_raw: np.ndarray,
    window_radius: int,
    ae=None,
    scaler=None,
) -> np.ndarray:
    """Create temporal windows from cached-VAE point residuals."""
    if ae is None or scaler is None:
        ae, scaler, _, _, _ = load_best_vae()
    x_norm = scaler.transform(x_raw).astype(np.float32)
    recon = reconstruct_array(ae, x_norm)
    residual = (x_norm - recon).astype(np.float32)
    w = int(window_radius)
    if len(residual) <= 2 * w:
        return residual.reshape(1, -1).astype(np.float32)
    windows = [residual[t - w:t + w + 1, :].flatten() for t in range(w, len(residual) - w)]
    return np.asarray(windows, dtype=np.float32)


def print_cache_status() -> bool:
    """打印缓存状态，返回是否命中。"""
    if _BEST_VAE_PATH.exists():
        info = get_best_vae_info()
        print(f"  [VAE 缓存命中] {info['path']}", flush=True)
        print(f"  -> AE={info['ae_structure']}  latent={info['latent_dim']}"
              f"  loss={info['loss']:.4f}"
              f"  α={info['alpha']}  w={info['w']}"
              f"  cluster={info['cluster_method']}", flush=True)
        return True
    else:
        print(f"  [VAE 缓存未命中] 请先运行: python 04_exp4_train_and_save_best.py", flush=True)
        return False


# ── 兼容性别名 ──────────────────────────────────────────────────────────────
load_cached_vae = load_best_vae
