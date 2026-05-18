"""
DeepFractal Model — full prediction framework (Section 2 + Section 3.2).

Architecture (Fig 1):
  Input  →  Multi-scale Fractal Feature Extraction  (16×16, 32×32, 64×64)
         →  Feature Extraction MLP per scale
         →  FractalFeatureFusion  (MSA + HTD)
         →  Prediction head (3 FC layers: 128→64→32→1)
         →  VAE latent bottleneck (optional, for LVAE loss)

Paper training settings (Section 3.2):
  - Batch size   : 64
  - Learning rate: 0.0001  (Adam, β₁=0.9, β₂=0.999, ε=1e-8)
  - Epochs       : 200  (early stopping patience=20)
  - Dropout      : 0.3 after first two hidden layers
  - Batch norm   : after each hidden layer
  - Split        : 60% train / 20% val / 20% test
  - Normalisation: Z-score  x' = (x-μ)/σ  (Eq. 32)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fusion import FractalFeatureFusion
from loss import FractalLoss


# ---------------------------------------------------------------------------
# Per-scale feature encoder  (MLP: in_dim → 128 → 64 → feature_dim)
# ---------------------------------------------------------------------------

class ScaleEncoder(nn.Module):
    def __init__(self, in_dim: int, feature_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# VAE bottleneck  (for LVAE latent distribution alignment)
# ---------------------------------------------------------------------------

class VAEBottleneck(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 16):
        super().__init__()
        self.mu_head      = nn.Linear(in_dim, latent_dim)
        self.log_var_head = nn.Linear(in_dim, latent_dim)
        self.decode       = nn.Linear(latent_dim, in_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu      = self.mu_head(x)
        log_var = self.log_var_head(x)
        # Reparameterisation trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z   = mu + eps * std
        return self.decode(z), mu, log_var


# ---------------------------------------------------------------------------
# Prediction head  (128 → 64 → 32 → 1, paper Section 3.2)
# ---------------------------------------------------------------------------

class PredictionHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),   # linear output for regression
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# DeepFractal  — full model
# ---------------------------------------------------------------------------

class DeepFractal(nn.Module):
    """
    Stock price prediction model based on multi-scale fractals and deep learning.

    Parameters
    ----------
    fractal_dim   : number of fractal features per scale (default 10)
    n_scales      : number of window scales (default 3: 16, 32, 64)
    feature_dim   : internal feature dimension per scale encoder
    tucker_rank   : Tucker decomposition rank in HTD
    latent_dim    : VAE latent dimension
    dropout       : dropout rate
    use_vae       : whether to include the VAE bottleneck
    """

    def __init__(
        self,
        fractal_dim:  int   = 10,
        n_scales:     int   = 3,
        feature_dim:  int   = 64,
        tucker_rank:  int   = 8,
        latent_dim:   int   = 16,
        dropout:      float = 0.3,
        use_vae:      bool  = True,
    ):
        super().__init__()
        self.n_scales  = n_scales
        self.use_vae   = use_vae

        # Per-scale encoder: fractal_dim → feature_dim
        self.encoders = nn.ModuleList([
            ScaleEncoder(fractal_dim, feature_dim, dropout) for _ in range(n_scales)
        ])

        # Multi-scale fusion (MSA + HTD)
        self.fusion = FractalFeatureFusion(
            n_scales=n_scales,
            feature_dim=feature_dim,
            tucker_rank=tucker_rank,
            output_dim=feature_dim,
        )

        # Optional VAE bottleneck
        if use_vae:
            self.vae = VAEBottleneck(feature_dim, latent_dim)

        # Prediction head
        self.head = PredictionHead(feature_dim, dropout)

    def forward(self, x_scales: list[torch.Tensor]) -> dict:
        """
        Parameters
        ----------
        x_scales : list of n_scales tensors, each (B, fractal_dim)

        Returns
        -------
        dict with keys: 'pred', 'mu' (optional), 'log_var' (optional)
        """
        encoded = [enc(x) for enc, x in zip(self.encoders, x_scales)]
        fused   = self.fusion(encoded)

        mu, log_var = None, None
        if self.use_vae:
            fused, mu, log_var = self.vae(fused)

        pred = self.head(fused)
        return {"pred": pred, "mu": mu, "log_var": log_var}


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def z_score_normalize(
    arr: np.ndarray,
    mean: float | None = None,
    std: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Z-score normalisation (Eq. 32)."""
    if mean is None:
        mean = arr.mean()
    if std is None:
        std  = arr.std() + 1e-12
    return (arr - mean) / std, float(mean), float(std)


def build_dataset(
    scale_features: dict[int, np.ndarray],
    targets: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio:   float = 0.2,
) -> dict:
    """
    Align multi-scale features to the shortest scale, z-score normalise,
    and split 60/20/20 (paper Section 3.2).

    Returns dict with train/val/test TensorDatasets and normalisation params.
    """
    # Align to shortest
    min_len = min(f.shape[0] for f in scale_features.values())
    aligned = {w: f[-min_len:] for w, f in scale_features.items()}
    targets = targets[-min_len:]

    # Normalise features per scale (fit on train only)
    train_end = int(min_len * train_ratio)
    val_end   = int(min_len * (train_ratio + val_ratio))

    norm_params = {}
    norm_feats  = {}
    for w, feat in aligned.items():
        tr_mean = feat[:train_end].mean(axis=0)
        tr_std  = feat[:train_end].std(axis=0) + 1e-12
        norm_feats[w] = (feat - tr_mean) / tr_std
        norm_params[w] = (tr_mean, tr_std)

    # Normalise targets
    tgt_mean = targets[:train_end].mean()
    tgt_std  = targets[:train_end].std() + 1e-12
    tgt_norm = (targets - tgt_mean) / tgt_std
    norm_params["target"] = (tgt_mean, tgt_std)

    def _make_ds(start, end):
        xs = [
            torch.tensor(norm_feats[w][start:end], dtype=torch.float32)
            for w in sorted(norm_feats)
        ]
        y  = torch.tensor(tgt_norm[start:end], dtype=torch.float32)
        return TensorDataset(*xs, y)

    return {
        "train": _make_ds(0, train_end),
        "val":   _make_ds(train_end, val_end),
        "test":  _make_ds(val_end, min_len),
        "norm_params": norm_params,
        "scales": sorted(norm_feats.keys()),
    }


def train(
    model:       DeepFractal,
    dataset:     dict,
    epochs:      int   = 200,
    batch_size:  int   = 64,
    lr:          float = 1e-4,
    patience:    int   = 20,
    lambda_mse:  float = 1.0,
    lambda_renyi:float = 0.1,
    lambda_hurst:float = 0.05,
    lambda_mfs:  float = 0.05,
    lambda_vae:  float = 0.01,
    use_fractal: bool  = True,
    device:      str   = "cpu",
    seed:        int   = 42,
    verbose:     bool  = True,
) -> dict:
    """
    Train the DeepFractal model (paper Section 3.2 settings).

    Returns dict with train/val loss histories and best model state.
    """
    torch.manual_seed(seed)
    model = model.to(device)

    criterion = FractalLoss(
        lambda_mse=lambda_mse,
        lambda_renyi=lambda_renyi,
        lambda_hurst=lambda_hurst,
        lambda_mfs=lambda_mfs,
        lambda_vae=lambda_vae,
        use_fractal=use_fractal,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience // 2, factor=0.5)

    n_scales = len(dataset["scales"])
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(dataset["val"],   batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        ep_loss = 0.0
        for batch in train_loader:
            x_scales = [batch[i].to(device) for i in range(n_scales)]
            y_true   = batch[n_scales].to(device)

            optimizer.zero_grad()
            out = model(x_scales)
            losses = criterion(y_true, out["pred"], out["mu"], out["log_var"])
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += losses["total"].item() * y_true.shape[0]

        ep_loss /= len(dataset["train"])
        train_losses.append(ep_loss)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_scales = [batch[i].to(device) for i in range(n_scales)]
                y_true   = batch[n_scales].to(device)
                out      = model(x_scales)
                losses   = criterion(y_true, out["pred"], out["mu"], out["log_var"])
                val_loss += losses["total"].item() * y_true.shape[0]
        val_loss /= len(dataset["val"])
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs}  train={ep_loss:.5f}  val={val_loss:.5f}  best={best_val:.5f}")

        if no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"train_losses": train_losses, "val_losses": val_losses, "best_val": best_val}


def evaluate(
    model:    DeepFractal,
    dataset:  dict,
    device:   str = "cpu",
) -> dict:
    """
    Evaluate on test set. Returns MAE, MAPE, MSE, RMSE, R² (Eqs. 27-31)
    in the original (de-normalised) price scale.
    """
    model.eval()
    n_scales = len(dataset["scales"])
    test_loader = DataLoader(dataset["test"], batch_size=256, shuffle=False)

    tgt_mean, tgt_std = dataset["norm_params"]["target"]
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            x_scales = [batch[i].to(device) for i in range(n_scales)]
            y_true   = batch[n_scales].to(device)
            out      = model(x_scales)
            all_true.append(y_true.cpu().numpy())
            all_pred.append(out["pred"].cpu().numpy())

    y_true = np.concatenate(all_true) * tgt_std + tgt_mean
    y_pred = np.concatenate(all_pred) * tgt_std + tgt_mean

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))))
    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2   = float(1 - ss_res / (ss_tot + 1e-12))

    return {
        "MAE":  mae,
        "MAPE": mape,
        "MSE":  mse,
        "RMSE": rmse,
        "R2":   r2,
        "y_true": y_true,
        "y_pred": y_pred,
    }
