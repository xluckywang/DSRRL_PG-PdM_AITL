"""
AACE-CM-Exp4: 璁粌骞朵繚瀛樻渶浣抽厤缃?VAE + 鍩虹嚎姹?
==============================================
浠?exp3_best_config.json 璇诲彇鏈浼橀厤缃紙VAE / latent_dim=4 / DBSCAN锛夛紝
涓娆℃ц缁冿紝灏嗘墍鏈変骇鐗╀繚瀛樺埌 outputs/best_vae/銆?

浜х墿锛堟墍鏈夊疄楠屽叡浜級锛?
  outputs/best_vae/best_vae.pt       鈥?VAE 鏉冮噸 + scaler + 鍩虹嚎姹?+ 鍏冧俊鎭?
  outputs/best_vae/best_vae_info.txt 鈥?璁粌鎽樿

杩愯锛?
  python 04_exp4_train_and_save_best.py
  python 04_exp4_train_and_save_best.py --force   # 寮哄埗閲嶈
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from config import (
    COMMON,
    EXP3,
    PATHS,
    get_exp2_generation_dates,
    get_feature_cols,
    get_feature_data_file,
    get_label_file,
    get_final_dshi_config,
    get_train_dates,
    load_exp3_best_config,
    train_dates_set,
)


def _vib_cols(device: str) -> List[str]:
    return get_feature_cols(device)


'''
# Legacy pre-server loader disabled. It pointed to outputs/exp3_grid_search and
# defaulted to the old VAE/three-axis configuration.
def _load_best_config() -> dict:
    cfg_path = PATHS.exp3_grid_dir / "exp3_best_config.json"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.setdefault("latent_dim", 4)
        cfg.setdefault("epochs", 200)
        cfg.setdefault("seed", 42)
        cfg.setdefault("window_w", 0)
        cfg.setdefault("alpha", 0.3)
        cfg.setdefault("distance_metric", "euclidean")
        cfg.setdefault("baseline_method", "DBSCAN")
        cfg["cluster_method"] = cfg.get("baseline_method")
        print(f"[TrainBest] 浠?{cfg_path.name} 鍔犺浇閰嶇疆: "
              f"AE={cfg['ae_structure']} latent={cfg['latent_dim']} "
              f"method={cfg.get('baseline_method')} epochs={cfg['epochs']}")
        return cfg
    print("[TrainBest] 鏈壘鍒?exp3_best_config.json锛屼娇鐢ㄩ粯璁ら厤缃紙VAE / latent=4 / DBSCAN锛?)
    return {
        "ae_structure": "VAE", "latent_dim": 4, "epochs": 200,
        "seed": 42, "window_w": 0, "alpha": 0.3,
        "baseline_method": "DBSCAN", "cluster_method": "DBSCAN",
        "distance_metric": "euclidean",
    }


'''
def _load_best_config() -> dict:
    cfg = load_exp3_best_config(required=False)
    if cfg:
        cfg.update(get_final_dshi_config())
        cfg.setdefault("latent_dim", 8)
        cfg.setdefault("epochs", 100)
        cfg.setdefault("seed", 42)
        cfg.setdefault("window_w", 30)
        cfg.setdefault("alpha", 0.3)
        cfg.setdefault("distance_metric", "euclidean")
        cfg.setdefault("baseline_method", "DBSCAN")
        cfg["cluster_method"] = cfg.get("baseline_method", cfg.get("cluster_method", "DBSCAN"))
        print(
            f"[TrainBest] load Exp3 best config: {cfg.get('_source_path', 'unknown')} "
            f"AE={cfg['ae_structure']} latent={cfg['latent_dim']} "
            f"method={cfg.get('baseline_method')} epochs={cfg['epochs']}"
        )
        return cfg
    print("[TrainBest] Exp3 best config not found; using formal fallback (Conv1DAE / latent=8 / DBSCAN)")
    return {
        "ae_structure": "Conv1DAE", "latent_dim": 8, "epochs": 100,
        "seed": 42, "window_w": 30, "alpha": 0.3,
        "baseline_method": "DBSCAN", "cluster_method": "DBSCAN",
        "distance_metric": "euclidean", "ours_variant": "dual_space",
        **get_final_dshi_config(),
    }


def _ae_dims(name: str, latent_dim: int) -> Tuple[List[int], List[int]]:
    return {
        "SimpleAE":    ([8], [8]),
        "FlexibleAE":  ([16, 8, 4], [8, 16]),
        "DeepAE":      ([32, 16, 8], [16, 32]),
        "VAE":         ([16, 8, latent_dim], [8, 16]),
        "Conv1DAE":    ([latent_dim], [latent_dim]),
    }[name]


class FlexibleAE(nn.Module):
    def __init__(self, input_dim, enc_dims, dec_dims):
        super().__init__()
        enc, prev = [], input_dim
        for d in enc_dims:
            enc += [nn.Linear(prev, d), nn.ReLU()]; prev = d
        self.encoder = nn.Sequential(*enc)
        dec, prev = [], enc_dims[-1] if enc_dims else input_dim
        for d in dec_dims:
            dec += [nn.Linear(prev, d), nn.ReLU()]; prev = d
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class VAE(nn.Module):
    def __init__(self, input_dim, encoder_dims, decoder_dims, latent_dim, beta=1.0):
        super().__init__()
        enc_layers, prev = [], input_dim
        for d in encoder_dims:
            enc_layers.extend([nn.Linear(prev, d), nn.ReLU()]); prev = d
        self.encoder_mean = nn.Sequential(*enc_layers, nn.Linear(prev, latent_dim))
        self.encoder_logvar = nn.Sequential(*enc_layers, nn.Linear(prev, latent_dim))
        dec_layers = []
        prev = latent_dim
        for d in decoder_dims:
            dec_layers.extend([nn.Linear(prev, d), nn.ReLU()]); prev = d
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)
        self.beta = beta

    def encode(self, x):
        with torch.no_grad():
            return self.encoder_mean(x)

    def forward(self, x):
        mu = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(z), mu, logvar


class Conv1DAutoEncoder(nn.Module):
    def __init__(self, window_size: int, channels: Tuple[int, int, int],
                 latent_dim: int, input_dim: int):
        super().__init__()
        c1, c2, _ = channels
        self.input_dim = input_dim
        self.window_size = window_size
        self._is_conv1d_ae = True
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        self.fc_enc = nn.Linear(c2, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, c2)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose1d(c2, c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.Upsample(size=3, mode="linear"),
            nn.ConvTranspose1d(c1, input_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Upsample(size=window_size, mode="linear"),
        )

    def forward(self, x):
        h = self.encoder(x.permute(0, 2, 1))
        z = self.fc_enc(h)
        h2 = self.fc_dec(z).unsqueeze(-1)
        recon = self.decoder(h2)
        return recon.permute(0, 2, 1)

    def encode(self, x):
        with torch.no_grad():
            h = self.encoder(x.permute(0, 2, 1))
            return self.fc_enc(h)

    def decode(self, z):
        h2 = self.fc_dec(z).unsqueeze(-1)
        recon = self.decoder(h2)
        return recon.permute(0, 2, 1)


def _build_sliding_windows(x: np.ndarray, window_size: int) -> np.ndarray:
    n = len(x)
    pad_left = window_size // 2
    pad_right = window_size // 2
    padded = np.pad(x, ((pad_left, pad_right), (0, 0)), mode="edge")
    windows = np.zeros((n, window_size, x.shape[1]), dtype=np.float32)
    for i in range(n):
        windows[i] = padded[i:i + window_size]
    return windows


def _build_ae(name: str, input_dim: int, latent_dim: int) -> nn.Module:
    enc_d, dec_d = _ae_dims(name, latent_dim)
    if name in ("SimpleAE", "FlexibleAE", "DeepAE"):
        return FlexibleAE(input_dim, enc_d, dec_d)
    if name == "Conv1DAE":
        return Conv1DAutoEncoder(
            window_size=EXP3.conv1d_window_size,
            channels=EXP3.conv1d_channels,
            latent_dim=latent_dim,
            input_dim=input_dim,
        )
    if name == "VAE":
        return VAE(input_dim, enc_d, dec_d, latent_dim, beta=EXP3.vae_beta)
    raise ValueError(f"Unknown AE: {name}")


def _train_ae(x_norm: np.ndarray, struct: str, latent_dim: int,
               epochs: int, seed: int) -> Tuple[nn.Module, List[float], int]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = _build_ae(struct, x_norm.shape[1], latent_dim)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=EXP3.learning_rate,
                             weight_decay=EXP3.weight_decay)
    criterion = nn.MSELoss()
    train_x = (
        _build_sliding_windows(x_norm, EXP3.conv1d_window_size)
        if struct == "Conv1DAE" else
        x_norm
    )
    tx = torch.tensor(train_x, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        tx, batch_size=min(EXP3.batch_size, len(tx)),
        shuffle=True, drop_last=True)

    best_loss, best_state, pat = float("inf"), None, 0
    losses = []
    for ep in range(epochs):
        ep_loss = 0
        for xb in loader:
            opt.zero_grad()
            if isinstance(model, VAE):
                recon, mu, logvar = model(xb)
                recon_loss = criterion(recon, xb)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + model.beta * kl_loss
            else:
                loss = criterion(model(xb), xb)
            loss.backward(); opt.step()
            ep_loss += loss.item()
        mean_loss = ep_loss / len(loader)
        losses.append(float(mean_loss))
        if mean_loss < best_loss * (1 - EXP3.early_stopping_threshold):
            best_loss = mean_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= EXP3.early_stopping_patience:
            print(f"[TrainBest] Early stopping @ epoch {ep+1}  loss={mean_loss:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, losses, len(losses)


@torch.no_grad()
def _encode(model: nn.Module, x_norm: np.ndarray) -> np.ndarray:
    if getattr(model, "_is_conv1d_ae", False):
        xw = _build_sliding_windows(x_norm, EXP3.conv1d_window_size)
        return model.encode(torch.tensor(xw, dtype=torch.float32)).numpy()
    return model.encode(torch.tensor(x_norm, dtype=torch.float32)).numpy()


@torch.no_grad()
def _recon_and_residual(model: nn.Module, x_norm: np.ndarray):
    if getattr(model, "_is_conv1d_ae", False):
        xw = _build_sliding_windows(x_norm, EXP3.conv1d_window_size)
        tensor_in = torch.tensor(xw, dtype=torch.float32)
        recon_w = model(tensor_in).detach().numpy()
        center = EXP3.conv1d_window_size // 2
        recon = recon_w[:, center, :]
        return recon, x_norm - recon
    tensor_in = torch.tensor(x_norm, dtype=torch.float32)
    output = model(tensor_in)
    if isinstance(output, tuple):
        recon = output[0]  # VAE: (recon, mu, logvar)
    else:
        recon = output        # AE: recon
    recon = recon.detach().numpy()
    return recon, x_norm - recon


def _window_smooth(z: np.ndarray, w: int) -> np.ndarray:
    if w <= 0:
        return z
    pad = w // 2
    out = np.zeros_like(z)
    for i in range(len(z)):
        s, e = max(0, i - pad), min(len(z), i + pad + 1)
        out[i] = z[s:e].mean(axis=0)
    return out


def _build_baseline_pool(z: np.ndarray, method: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(z) > EXP3.dbscan_max_points:
        idx = rng.choice(len(z), EXP3.dbscan_max_points, replace=False)
        z_fit = z[idx]
    else:
        z_fit = z
    if method == "KMeans_K3":
        km = KMeans(n_clusters=3, random_state=seed, n_init=10); km.fit(z_fit)
        return km.cluster_centers_
    if method == "KMeans_K5":
        km = KMeans(n_clusters=5, random_state=seed, n_init=10); km.fit(z_fit)
        return km.cluster_centers_
    if method == "DBSCAN":
        db = DBSCAN(eps=EXP3.dbscan_eps, min_samples=EXP3.dbscan_min_samples, n_jobs=-1)
        db.fit(z_fit)
        centers = []
        for label in set(db.labels_):
            if label == -1:
                continue
            mask = db.labels_ == label
            centers.append(z_fit[mask].mean(axis=0))
        return np.array(centers) if centers else z_fit[:1]
    if method == "GMM":
        gmm = GaussianMixture(n_components=min(5, len(z_fit)),
                               random_state=seed, n_init=5)
        gmm.fit(z_fit)
        return gmm.means_
    return z_fit[:5]


# 鈹鈹 涓诲嚱鏁?鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="寮哄埗閲嶈锛堝拷鐣ョ紦瀛橈級")
    args = parser.parse_args()

    BEST_VAE_DIR = PATHS.exp4_figures_dir / "best_vae"
    BEST_VAE_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = BEST_VAE_DIR / "best_vae.pt"
    info_path = BEST_VAE_DIR / "best_vae_info.txt"

    if ckpt_path.exists() and not args.force:
        print(f"\n[TrainBest] 缂撳瓨宸插瓨鍦? {ckpt_path.name}")
        print("[TrainBest] Use --force to retrain the active Exp3 best AE.")
        return

    print("\n" + "=" * 65)
    print("AACE-CM-Exp4: Train and cache the active Exp3 best AE")
    print("=" * 65)

    # 鈹鈹 Step 1: 鍔犺浇鍋ュ悍鏁版嵁 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    print("\n[1/5] 鍔犺浇鍋ュ悍鏁版嵁...", flush=True)
    device = COMMON.device
    all_dates = [
        d for d in get_train_dates()
        if get_feature_data_file(device, d).exists() and get_label_file(device, d).exists()
    ]
    train_dates = [d for d in all_dates if d in train_dates_set()]
    vib = _vib_cols(device)
    all_x, all_y = [], []
    for d in train_dates:
        feat_df = pd.read_csv(get_feature_data_file(device, d))
        assert all(c in feat_df.columns for c in vib), (
            f"Missing feature columns for {device} {d}: "
            f"{[c for c in vib if c not in feat_df.columns]}"
        )
        feat_df = feat_df[vib]
        lab_df = pd.read_csv(get_label_file(device, d))
        state_col = "state" if "state" in lab_df.columns else "label"
        n = min(len(feat_df), len(lab_df))
        all_x.append(feat_df[vib].iloc[:n].values)
        all_y.append(lab_df[state_col].iloc[:n].values)
    x_healthy_raw = np.vstack(all_x).astype(np.float32)
    y_all = np.concatenate(all_y)
    x_healthy_raw = x_healthy_raw[y_all == EXP3.normal_state_id]
    print(f"  healthy samples: {x_healthy_raw.shape} | train days: {len(train_dates)}")

    # 鈹鈹 Step 2: 鏍囧噯鍖?鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    print("\n[2/5] 鏍囧噯鍖?..", flush=True)
    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x_healthy_raw)
    print(f"  scaler: mean.shape={scaler.mean_.shape}  n_features={scaler.n_features_in_}")

    # 鈹鈹 Step 3: 璁粌 VAE 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    print("\n[3/5] Training active Exp3 best AE...", flush=True)
    cfg = _load_best_config()
    model, losses, epochs_trained = _train_ae(
        x_norm,
        struct=cfg["ae_structure"],
        latent_dim=int(cfg["latent_dim"]),
        epochs=int(cfg["epochs"]),
        seed=int(cfg["seed"]),
    )
    final_loss = losses[-1]
    print(f"  璁粌瀹屾垚: epochs={epochs_trained}  final_loss={final_loss:.6f}")

    # 鈹鈹 Step 4: 鏋勫缓鍩虹嚎姹?鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    print("\n[4/5] 鏋勫缓鍩虹嚎姹?..", flush=True)
    z_all = _encode(model, x_norm)
    _, residual_all = _recon_and_residual(model, x_norm)
    w = int(cfg.get("window_w", 0))
    z_smooth = _window_smooth(z_all, w)
    r_smooth = _window_smooth(residual_all, w)
    cluster_method = cfg.get("cluster_method", "DBSCAN")
    seed = int(cfg.get("seed", 42))
    pool_latent = _build_baseline_pool(z_smooth, cluster_method, seed)
    pool_residual = _build_baseline_pool(r_smooth, cluster_method, seed)
    print(f"  z_pool: {pool_latent.shape}  r_pool: {pool_residual.shape}  method={cluster_method}")

    # 鈹鈹 Step 5: 淇濆瓨 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    print("\n[5/5] 淇濆瓨 checkpoint...", flush=True)
    final_dshi = get_final_dshi_config()
    ckpt = {
        # 妯″瀷
        "ae_state_dict": model.state_dict(),
        "ae_structure": cfg["ae_structure"],
        "latent_dim": int(cfg["latent_dim"]),
        "encoder_dims": None if cfg["ae_structure"] == "Conv1DAE" else _ae_dims(cfg["ae_structure"], int(cfg["latent_dim"]))[0],
        "decoder_dims": None if cfg["ae_structure"] == "Conv1DAE" else _ae_dims(cfg["ae_structure"], int(cfg["latent_dim"]))[1],
        "conv1d_window_size": EXP3.conv1d_window_size,
        "conv1d_channels": tuple(EXP3.conv1d_channels),
        # Scaler
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "scaler_n_features_in": scaler.n_features_in_,
        "feature_cols": list(vib),
        "feature_data_source": "get_feature_data_file",
        # 鍩虹嚎姹?
        "pool_latent": pool_latent,
        "pool_residual": pool_residual,
        # 瓒呭弬鏁?
        "alpha": float(cfg.get("alpha", 0.3)),
        "window_w": int(cfg.get("window_w", 0)),
        "cluster_method": cluster_method,
        "distance_metric": cfg.get("distance_metric", "euclidean"),
        "health_index_name": "Dual-Space Health Index",
        "score_family": final_dshi["score_family"],
        "effective_w_latent": final_dshi["effective_w_latent"],
        "effective_w_reconstruction": final_dshi["effective_w_reconstruction"],
        "effective_w_residual": final_dshi["effective_w_residual"],
        "weight_source": final_dshi["weight_source"],
        # 鍏冧俊鎭?
        "epochs_trained": epochs_trained,
        "final_loss": float(final_loss),
        "seed": int(cfg.get("seed", 42)),
        "train_date": str(pd.Timestamp.now().date()),
        "n_healthy_samples": len(x_healthy_raw),
        "n_train_days": len(train_dates),
        "exp3_rank": cfg.get("rank", 1),
    }
    torch.save(ckpt, ckpt_path)
    print(f"  -> {ckpt_path}")

    # 淇濆瓨鏂囨湰鎽樿
    info_lines = [
        f"AACE-CM-Exp4 Active Exp3 Best AE Checkpoint",
        f"{'='*50}",
        f"AE Structure : {cfg['ae_structure']}",
        f"Latent Dim   : {cfg['latent_dim']}",
        f"Cluster Meth : {cluster_method}",
        f"Window W     : {cfg.get('window_w', 0)}",
        f"Alpha        : {cfg.get('alpha', 0.3)}",
        f"Seed         : {cfg.get('seed', 42)}",
        f"Epochs Train : {epochs_trained} (early stopped)",
        f"Final Loss   : {final_loss:.6f}",
        f"Healthy Pts  : {len(x_healthy_raw)}",
        f"Train Days   : {len(train_dates)}",
        f"Pool Latent  : {pool_latent.shape}",
        f"Pool Residual: {pool_residual.shape}",
        f"Health Index : Dual-Space Health Index",
        f"DSHI Weights : latent={final_dshi['effective_w_latent']:.2f}, "
        f"reconstruction={final_dshi['effective_w_reconstruction']:.2f}, "
        f"residual={final_dshi['effective_w_residual']:.2f}",
        f"Weight Source: {final_dshi['weight_source']}",
        f"Train Date   : {ckpt['train_date']}",
        f"{'='*50}",
    ]
    info_path.write_text("\n".join(info_lines))
    print(f"  -> {info_path}")

    print(f"\n{'='*65}")
    print(f"Done. All experiments can now load this VAE without retraining.")
    print(f"  Location: {ckpt_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
