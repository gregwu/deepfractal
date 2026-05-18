"""
Baseline sequence models from Table 1 of the paper.

  RNN       — vanilla recurrent network [31]
  LSTM      — long short-term memory    [32]
  GRU       — gated recurrent unit      [32]
  ALSTM     — attention LSTM            [33]
  VMD-LSTM  — variational mode decomposition + LSTM [34]

All models share:
  - Input: (B, seq_len, input_dim)  raw close-price windows (normalised)
  - Output: (B,)  single-step price prediction
  - Training: same Adam settings as DeepFractal (lr=1e-4, β₁=0.9, β₂=0.999)
  - Early stopping: patience=20, max 200 epochs
  - Loss: MSE  (baselines do not use fractal loss)

VMD-LSTM uses PyTorch-based VMD decomposition (no external package needed).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Sequence dataset builder
# ---------------------------------------------------------------------------

def make_sequence_dataset(
    close: np.ndarray,
    seq_len: int = 60,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> dict:
    """
    Build sliding-window sequences from closing prices.
    X: (N, seq_len)  input window (normalised)
    y: (N,)          next-step close (normalised)
    Split 60/20/20 chronologically; Z-score on train stats.
    """
    n = len(close)
    xs, ys = [], []
    for i in range(seq_len, n):
        xs.append(close[i - seq_len: i])
        ys.append(close[i])
    X = np.array(xs, dtype=np.float32)
    y = np.array(ys, dtype=np.float32)

    N = len(y)
    tr = int(N * train_ratio)
    va = int(N * (train_ratio + val_ratio))

    # Normalise on train statistics
    x_mean, x_std = X[:tr].mean(), X[:tr].std() + 1e-12
    y_mean, y_std = y[:tr].mean(), y[:tr].std() + 1e-12

    X_norm = (X - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    def _ds(a, b):
        Xt = torch.tensor(X_norm[a:b]).unsqueeze(-1)   # (N, seq_len, 1)
        yt = torch.tensor(y_norm[a:b])
        return TensorDataset(Xt, yt)

    return {
        "train": _ds(0, tr),
        "val":   _ds(tr, va),
        "test":  _ds(va, N),
        "norm":  (y_mean, y_std),
        "n_train": tr, "n_val": va - tr, "n_test": N - va,
    }


# ---------------------------------------------------------------------------
# RNN
# ---------------------------------------------------------------------------

class RNNModel(nn.Module):
    def __init__(self, input_dim: int = 1, hidden: int = 64, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden, n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.fc  = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_dim: int = 1, hidden: int = 64, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------

class GRUModel(nn.Module):
    def __init__(self, input_dim: int = 1, hidden: int = 64, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.fc  = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------------------
# ALSTM — Attention LSTM [33]
# Additive attention over all hidden states before the output FC.
# ---------------------------------------------------------------------------

class ALSTMModel(nn.Module):
    def __init__(self, input_dim: int = 1, hidden: int = 64, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm   = nn.LSTM(input_dim, hidden, n_layers, batch_first=True,
                              dropout=dropout if n_layers > 1 else 0.0)
        self.attn_w = nn.Linear(hidden, 1)
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                          # (B, T, H)
        scores  = self.attn_w(out).squeeze(-1)         # (B, T)
        weights = F.softmax(scores, dim=1)             # (B, T)
        context = (out * weights.unsqueeze(-1)).sum(1) # (B, H)
        return self.fc(self.drop(context)).squeeze(-1)


# ---------------------------------------------------------------------------
# VMD-LSTM — Variational Mode Decomposition + LSTM [34]
#
# VMD decomposes the input window into K band-limited modes using an
# iterative optimisation in the frequency domain (Dragomiretskiy 2014).
# Each mode is fed through a shared LSTM; outputs are summed.
# ---------------------------------------------------------------------------

def _vmd(signal: np.ndarray, K: int = 3, alpha: float = 2000.0,
         tau: float = 0.0, tol: float = 1e-7, max_iter: int = 300) -> np.ndarray:
    """
    Variational Mode Decomposition.
    Returns modes of shape (K, len(signal)).
    """
    T = len(signal)
    fs = 1.0 / T
    freqs = np.arange(T) / T
    freqs[freqs >= 0.5] -= 1.0

    f_hat = np.fft.fft(signal)

    omega = np.zeros(K)                       # centre frequencies
    for k in range(K):
        omega[k] = (0.5 / K) * k

    u_hat = np.zeros((K, T), dtype=complex)
    lam   = np.zeros(T, dtype=complex)

    for _ in range(max_iter):
        omega_old = omega.copy()
        for k in range(K):
            # Update u_hat_k
            sum_other = sum(u_hat[j] for j in range(K) if j != k)
            num = f_hat - sum_other + lam / 2
            denom = 1 + alpha * (freqs - omega[k]) ** 2
            u_hat[k] = num / denom

            # Update omega_k (centre frequency)
            pos_mask = freqs > 0
            u_sq = np.abs(u_hat[k, pos_mask]) ** 2
            omega[k] = np.dot(freqs[pos_mask], u_sq) / (np.sum(u_sq) + 1e-12)

        # Dual ascent
        lam += tau * (np.sum(u_hat, axis=0) - f_hat)

        if np.max(np.abs(omega - omega_old)) < tol:
            break

    modes = np.real(np.fft.ifft(u_hat, axis=1))
    return modes


def _vmd_batch(X: np.ndarray, K: int = 3) -> np.ndarray:
    """Apply VMD to each row of X (B, T) → (B, T, K)."""
    B, T = X.shape
    out = np.zeros((B, T, K), dtype=np.float32)
    for i in range(B):
        try:
            modes = _vmd(X[i], K=K)     # (K, T)
            out[i] = modes.T            # (T, K)
        except Exception:
            out[i, :, 0] = X[i]        # fallback: raw signal as mode 0
    return out


class VMDLSTMModel(nn.Module):
    """VMD decomposes each window into K modes; a shared LSTM processes all."""

    def __init__(self, K: int = 3, hidden: int = 64, n_layers: int = 2,
                 dropout: float = 0.3, precomputed: bool = False):
        super().__init__()
        self.K           = K
        self.precomputed = precomputed  # if True, input is already (B,T,K)
        self.lstm = nn.LSTM(K, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.attn_w = nn.Linear(hidden, 1)
        self.fc     = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, K) — multi-mode input
        out, _ = self.lstm(x)
        scores  = self.attn_w(out).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = (out * weights.unsqueeze(-1)).sum(1)
        return self.fc(context).squeeze(-1)


# ---------------------------------------------------------------------------
# Generic training loop for all baseline models
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))))
    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    r2   = float(1 - ss_res / ss_tot)
    return {"MAE": mae, "MAPE": mape, "MSE": mse, "RMSE": rmse, "R2": r2}


def train_baseline(
    model: nn.Module,
    dataset: dict,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-4,
    patience: int = 20,
    device: str = "cpu",
    seed: int = 42,
    vmd_k: int = 3,
    is_vmd: bool = False,
    close_train: np.ndarray | None = None,
) -> dict:
    """
    Train a baseline model. For VMD-LSTM, pre-computes VMD decomposition.
    Returns evaluation metrics on the test set (de-normalised).
    """
    torch.manual_seed(seed)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience // 2, factor=0.5)

    y_mean, y_std = dataset["norm"]

    # For VMD-LSTM: pre-decompose all windows into modes
    if is_vmd:
        def _rebuild_loader(ds, shuffle):
            Xs = ds.tensors[0].numpy().squeeze(-1)   # (N, T)
            ys = ds.tensors[1]
            Xv = torch.tensor(_vmd_batch(Xs, K=vmd_k))
            return DataLoader(TensorDataset(Xv, ys), batch_size=batch_size, shuffle=shuffle)
        train_loader = _rebuild_loader(dataset["train"], True)
        val_loader   = _rebuild_loader(dataset["val"],   False)
        test_loader  = _rebuild_loader(dataset["test"],  False)
    else:
        train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(dataset["val"],   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(dataset["test"],  batch_size=batch_size, shuffle=False)

    best_val, best_state, no_improve = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(yb)
        val_loss /= len(dataset["val"])
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    # Evaluate on test set
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            all_pred.append(model(xb.to(device)).cpu().numpy())
            all_true.append(yb.numpy())

    y_pred = np.concatenate(all_pred) * y_std + y_mean
    y_true = np.concatenate(all_true) * y_std + y_mean
    return _compute_metrics(y_true, y_pred)
