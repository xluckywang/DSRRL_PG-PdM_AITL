"""TemporalAE routines shared by the formal Exp8 representation preparation."""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalAE(nn.Module):
    """Temporal Autoencoder with flattened window input."""
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

    def get_residual(self, x):
        return x - self.forward(x)


def train_temporal_ae(
    X_windows: np.ndarray,
    latent_dim: int = 8,
    epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
    device=None,
) -> Tuple[TemporalAE, StandardScaler]:
    """Train TemporalAE on healthy data."""
    if device is None:
        device = DEVICE
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_windows).astype(np.float32)

    ae = TemporalAE(input_dim=X_scaled.shape[1], latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    MAX_TRAIN = 80000
    if len(X_scaled) > MAX_TRAIN:
        idx = np.random.RandomState(seed).choice(len(X_scaled), MAX_TRAIN, replace=False)
        X_scaled = X_scaled[idx]

    tx = torch.from_numpy(X_scaled)
    dataset = torch.utils.data.TensorDataset(tx)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

    ae.train()
    best_loss = float("inf"); best_state = None; pat = 0
    for epoch in range(epochs):
        ep_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            recon = ae(xb)
            loss = criterion(recon, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(xb)
        mean_l = ep_loss / len(dataset)
        if mean_l < best_loss * (1 - 1e-4):
            best_loss = mean_l
            best_state = {k: v.clone() for k, v in ae.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= 15:
            print(f"    TemporalAE early stop @ epoch {epoch+1}", flush=True)
            break
    if best_state:
        ae.load_state_dict(best_state)
    ae.eval()
    return ae, scaler


def compute_residual_windows(
    ae: TemporalAE,
    scaler: StandardScaler,
    X_windows: np.ndarray,
    device=None,
) -> np.ndarray:
    """Compute residual windows from AE."""
    if device is None:
        device = DEVICE
    ae.eval()
    X_scaled = scaler.transform(X_windows).astype(np.float32)
    with torch.no_grad():
        tx = torch.from_numpy(X_scaled).to(device)
        residual = ae.get_residual(tx).cpu().numpy()
    return residual
