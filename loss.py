"""
Multi-Scale Fractal Loss Function — Section 2.3 of the paper.

LFractal = λ₁·LRnyi + λ₂·LHurst + λ₃·LMFS + λ₄·LVAE    (Eq. 26)

Components:
  LMSE      — mean square error baseline                       (Eq. 17)
  LLogCosh  — log-cosh error (robust to outliers)             (Eq. 18)
  LRnyi     — generalized Rényi entropy error                  (Eq. 19)
  LHölder   — Hölder regularity constraint                     (Eq. 20-21)
  LHurst    — Hurst long-range dependency constraint           (Eq. 22)
  LMFS      — multifractal spectrum deviation                  (Eq. 23-24)
  LVAE      — KL divergence for latent distribution alignment  (Eq. 25)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Base error terms
# ---------------------------------------------------------------------------

def loss_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Standard MSE (Eq. 17)."""
    return F.mse_loss(y_pred, y_true)


def loss_log_cosh(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Log-Cosh error — robust to outliers (Eq. 18)."""
    diff = y_pred - y_true
    return torch.mean(torch.log(torch.cosh(diff + 1e-12)))


# ---------------------------------------------------------------------------
# Fractal loss terms
# ---------------------------------------------------------------------------

def loss_renyi(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    q: float = 2.0,
) -> torch.Tensor:
    """
    Generalized Rényi entropy error (Eq. 19).

    L_Rnyi = 1/(1-q) * log( Σ P_t^q )
    P_t = |y_t - ŷ_t|^q / Σ_j |y_j - ŷ_j|^q

    When q→1 degenerates to Shannon entropy.
    """
    if abs(q - 1.0) < 1e-6:
        # Shannon entropy limit
        abs_err = torch.abs(y_true - y_pred) + 1e-12
        p = abs_err / abs_err.sum()
        return -torch.sum(p * torch.log(p))

    abs_err = torch.abs(y_true - y_pred) ** q + 1e-12
    p = abs_err / abs_err.sum()
    return (1.0 / (1.0 - q)) * torch.log(torch.sum(p ** q) + 1e-12)


def _holder_exponent(series: torch.Tensor) -> torch.Tensor:
    """
    Approximate Hölder exponent H_q(y) for a 1-D tensor (Eq. 20).

    Estimated as the log-log slope of max increment vs scale.
    Returns a scalar tensor.
    """
    n = series.shape[0]
    if n < 4:
        return torch.tensor(0.5, device=series.device, dtype=series.dtype)

    log_scales, log_max_inc = [], []
    for scale in [max(2, n // k) for k in [2, 4, 8] if n // k >= 2]:
        increments = torch.abs(series[scale:] - series[:-scale])
        if increments.numel() == 0:
            continue
        log_scales.append(torch.log(torch.tensor(float(scale), device=series.device)))
        log_max_inc.append(torch.log(increments.max() + 1e-12))

    if len(log_scales) < 2:
        return torch.tensor(0.5, device=series.device, dtype=series.dtype)

    ls = torch.stack(log_scales)
    lm = torch.stack(log_max_inc)
    # slope via least squares
    ls_mean = ls.mean()
    lm_mean = lm.mean()
    slope = ((ls - ls_mean) * (lm - lm_mean)).sum() / ((ls - ls_mean) ** 2).sum().clamp(min=1e-12)
    return slope


def loss_holder(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Hölder regularity constraint (Eq. 21).
    Penalises mismatch between Hölder exponents of true and predicted series.

    L_Hölder = Σ_t |H_q(y_t) - H_q(ŷ_t)|²
    Applied batch-wise (each sample in the batch is one time series).
    """
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(0)
        y_pred = y_pred.unsqueeze(0)
    total = torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)
    for i in range(y_true.shape[0]):
        h_true = _holder_exponent(y_true[i])
        h_pred = _holder_exponent(y_pred[i])
        total = total + (h_true - h_pred) ** 2
    return total / y_true.shape[0]


def loss_hurst(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Hurst long-range dependency constraint (Eq. 22).
    Same formula as Hölder — both measure fractal regularity of the sequence.

    L_Hurst = Σ_t |H_q(y_t) - H_q(ŷ_t)|²
    """
    return loss_holder(y_true, y_pred)


def _mfs_scalar(series: torch.Tensor, q_vals: torch.Tensor) -> torch.Tensor:
    """
    Compute multifractal spectrum scalars for a 1-D series (differentiable approx).
    Uses log-moment method: τ(q) ≈ slope of log(Σ μ_i^q) vs log(scale).

    Returns Dq vector (same length as q_vals).
    """
    n = series.shape[0]
    scales = [max(2, n // k) for k in [4, 8] if n // k >= 2]
    if len(scales) < 2:
        return torch.zeros_like(q_vals)

    Dq_list = []
    for q in q_vals:
        log_inv_f, log_mu_q = [], []
        for f in scales:
            nb = n // f
            if nb < 1:
                continue
            segs = series[:nb * f].reshape(nb, f)
            mu = segs.abs().mean(dim=1) + 1e-12
            mu = mu / mu.sum()
            if torch.abs(q - 1.0) < 1e-6:
                lmu = -(mu * torch.log(mu)).sum()
            else:
                lmu = torch.log((mu ** q).sum() + 1e-12)
            log_inv_f.append(torch.log(torch.tensor(1.0 / f, device=series.device)))
            log_mu_q.append(lmu)

        if len(log_inv_f) < 2 or torch.abs(q - 1.0) < 1e-6:
            Dq_list.append(torch.tensor(1.0, device=series.device, dtype=series.dtype))
            continue

        lf = torch.stack(log_inv_f)
        lm = torch.stack(log_mu_q)
        lf_mean = lf.mean()
        lm_mean = lm.mean()
        slope = ((lf - lf_mean) * (lm - lm_mean)).sum() / ((lf - lf_mean) ** 2).sum().clamp(1e-12)
        Dq_list.append(slope / (q - 1.0))

    return torch.stack(Dq_list)


def loss_mfs(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    q_vals: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Multifractal spectrum deviation loss (Eq. 24).

    L_MFS = Σ_q |MFS_q(y) - MFS_q(ŷ)|²
    """
    if q_vals is None:
        q_vals = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=y_true.device, dtype=y_true.dtype)

    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(0)
        y_pred = y_pred.unsqueeze(0)

    total = torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)
    for i in range(y_true.shape[0]):
        dq_true = _mfs_scalar(y_true[i], q_vals)
        dq_pred = _mfs_scalar(y_pred[i], q_vals)
        total = total + ((dq_true - dq_pred) ** 2).sum()
    return total / y_true.shape[0]


def loss_vae(
    mu: torch.Tensor,
    log_var: torch.Tensor,
) -> torch.Tensor:
    """
    VAE KL divergence for latent space regularisation (Eq. 25).

    L_VAE = D_KL( q(Z|X) || p(Z) ) = -0.5 * Σ (1 + log_var - μ² - exp(log_var))
    """
    kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return kl


# ---------------------------------------------------------------------------
# Combined fractal loss (Eq. 26)
# ---------------------------------------------------------------------------

class FractalLoss(nn.Module):
    """
    LFractal = λ₁·LRnyi + λ₂·LHurst + λ₃·LMFS + λ₄·LVAE  (Eq. 26)

    Also includes MSE and Log-Cosh as the base error (the paper adds these
    on top of the fractal constraints).

    Parameters
    ----------
    lambda_mse    : weight for MSE base error
    lambda_logcosh: weight for Log-Cosh base error
    lambda_renyi  : λ₁
    lambda_hurst  : λ₂
    lambda_mfs    : λ₃
    lambda_vae    : λ₄
    q_renyi       : q parameter for Rényi entropy (default 2.0)
    use_fractal   : if False, only MSE + LogCosh (ablation baseline)
    """

    def __init__(
        self,
        lambda_mse:     float = 1.0,
        lambda_logcosh: float = 0.1,
        lambda_renyi:   float = 0.1,
        lambda_hurst:   float = 0.05,
        lambda_mfs:     float = 0.05,
        lambda_vae:     float = 0.01,
        q_renyi:        float = 2.0,
        use_fractal:    bool  = True,
    ):
        super().__init__()
        self.lambda_mse     = lambda_mse
        self.lambda_logcosh = lambda_logcosh
        self.lambda_renyi   = lambda_renyi
        self.lambda_hurst   = lambda_hurst
        self.lambda_mfs     = lambda_mfs
        self.lambda_vae     = lambda_vae
        self.q_renyi        = q_renyi
        self.use_fractal    = use_fractal

        q_vals = torch.linspace(-5.0, 5.0, 11)   # match feature extraction q range
        self.register_buffer("q_vals", q_vals)

    def forward(
        self,
        y_true:  torch.Tensor,
        y_pred:  torch.Tensor,
        mu:      torch.Tensor | None = None,
        log_var: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        y_true  : (B,) or (B, T) ground-truth prices
        y_pred  : same shape, predicted prices
        mu      : (B, latent_dim) VAE mean     (optional)
        log_var : (B, latent_dim) VAE log-var  (optional)

        Returns
        -------
        dict with individual loss components and 'total'
        """
        losses = {}

        losses["mse"]     = self.lambda_mse     * loss_mse(y_true, y_pred)
        losses["logcosh"] = self.lambda_logcosh  * loss_log_cosh(y_true, y_pred)

        if self.use_fractal:
            losses["renyi"]   = self.lambda_renyi  * loss_renyi(y_true, y_pred, q=self.q_renyi)
            losses["hurst"]   = self.lambda_hurst   * loss_hurst(y_true, y_pred)
            losses["mfs"]     = self.lambda_mfs     * loss_mfs(y_true, y_pred, q_vals=self.q_vals)
        else:
            losses["renyi"]  = torch.tensor(0.0, device=y_true.device)
            losses["hurst"]  = torch.tensor(0.0, device=y_true.device)
            losses["mfs"]    = torch.tensor(0.0, device=y_true.device)

        if mu is not None and log_var is not None:
            losses["vae"] = self.lambda_vae * loss_vae(mu, log_var)
        else:
            losses["vae"] = torch.tensor(0.0, device=y_true.device)

        losses["total"] = sum(losses.values())
        return losses
