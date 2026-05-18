"""
Multi-Scale Fractal Feature Fusion — Section 2.2 of the paper.

Two fusion mechanisms (Eqs. 12-16):
  1. MSA  — Multi-Scale Attention:  α_f = softmax(W_att Z_l),  F_MSA = Σ α_l Z_l
  2. HTD  — Higher-order Tensor Decomposition (Tucker):  F ≈ C ×₁ U₁ ×₂ U₂

Final fused feature (Eq. 16):
    F_final = [F_weighted, F_MSA, F_HTD]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleAttention(nn.Module):
    """
    Multi-Scale Attention (MSA) — Eq. 12.

    Takes L scale-level feature tensors of shape (B, d) and produces a
    single weighted sum (B, d).

    α_l = softmax(W_att Z_l)
    F_MSA = Σ_l α_l Z_l
    """

    def __init__(self, n_scales: int, feature_dim: int):
        super().__init__()
        self.n_scales    = n_scales
        self.feature_dim = feature_dim
        # W_att projects each scale feature to a scalar attention score
        self.W_att = nn.Linear(feature_dim, 1, bias=True)

    def forward(self, scale_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        scale_features : list of L tensors, each (B, feature_dim)

        Returns
        -------
        (B, feature_dim)
        """
        # Stack → (B, L, feature_dim)
        Z = torch.stack(scale_features, dim=1)
        # Score each scale: (B, L, 1) → (B, L)
        scores = self.W_att(Z).squeeze(-1)
        alpha  = F.softmax(scores, dim=1)          # (B, L)
        # Weighted sum: (B, L, 1) * (B, L, d) → (B, d)
        fused  = (alpha.unsqueeze(-1) * Z).sum(dim=1)
        return fused


class ChannelAttention(nn.Module):
    """
    Channel-wise attention applied to a single feature tensor (B, d).
    Computes per-channel weights via a small MLP.
    """

    def __init__(self, feature_dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, feature_dim // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d) → (B, d)"""
        return x * self.mlp(x)


class TuckerFusion(nn.Module):
    """
    Higher-order Tensor Decomposition (HTD) via Tucker decomposition — Eqs. 13-15.

    Constructs a tensor F ∈ R^{L×d} from the L scale features, then approximates
    F ≈ C ×₁ U₁ ×₂ U₂  using learned factor matrices.

    The output is the flattened Tucker reconstruction, projecting back to output_dim.
    """

    def __init__(self, n_scales: int, feature_dim: int, rank: int = 8, output_dim: int | None = None):
        super().__init__()
        self.n_scales    = n_scales
        self.feature_dim = feature_dim
        self.rank        = rank
        self.output_dim  = output_dim or feature_dim

        # Factor matrices U1 ∈ R^{L×r}, U2 ∈ R^{d×r}
        self.U1 = nn.Linear(n_scales, rank, bias=False)
        self.U2 = nn.Linear(feature_dim, rank, bias=False)

        # Core tensor parameters C ∈ R^{r×r}
        self.C  = nn.Parameter(torch.randn(rank, rank) * 0.01)

        # Project Tucker output back to output_dim
        self.proj = nn.Linear(rank * rank, self.output_dim)

    def forward(self, scale_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        scale_features : list of L tensors, each (B, feature_dim)

        Returns
        -------
        (B, output_dim)
        """
        B = scale_features[0].shape[0]
        # Stack → (B, L, d)
        F_tensor = torch.stack(scale_features, dim=1)

        # Mode-1 product: ×₁ U₁ → (B, r, d)
        # F_tensor: (B, L, d)  →  transpose(1,2): (B, d, L)
        m1 = self.U1(F_tensor.transpose(1, 2))   # (B, d, r)
        m1 = m1.transpose(1, 2)                   # (B, r, d)

        # Mode-2 product: ×₂ U₂ → (B, r, r)
        m2 = self.U2(m1)                          # (B, r, r)

        # Core contraction: element-wise with C (broadcast)
        core = m2 * self.C.unsqueeze(0)           # (B, r, r)

        # Flatten and project → (B, output_dim)
        out = self.proj(core.reshape(B, -1))
        return out


class FractalFeatureFusion(nn.Module):
    """
    Complete feature fusion module (Eqs. 12-16).

    Combines:
      - Channel attention on each scale (F_weighted)
      - Multi-scale attention F_MSA
      - Tucker HTD F_HTD

    Final output: F_final = concat([F_weighted_mean, F_MSA, F_HTD])
    """

    def __init__(
        self,
        n_scales: int,
        feature_dim: int,
        tucker_rank: int = 8,
        output_dim: int | None = None,
    ):
        super().__init__()
        self.n_scales    = n_scales
        self.feature_dim = feature_dim
        self.output_dim  = output_dim or feature_dim

        self.chan_attn = nn.ModuleList([
            ChannelAttention(feature_dim) for _ in range(n_scales)
        ])
        self.msa   = MultiScaleAttention(n_scales, feature_dim)
        self.htd   = TuckerFusion(n_scales, feature_dim, rank=tucker_rank, output_dim=feature_dim)

        # Project concatenated [F_weighted, F_MSA, F_HTD] → output_dim
        self.final_proj = nn.Linear(feature_dim * 3, self.output_dim)
        self.norm       = nn.LayerNorm(self.output_dim)

    def forward(self, scale_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        scale_features : list of n_scales tensors, each (B, feature_dim)

        Returns
        -------
        (B, output_dim)
        """
        # Channel-attention weighted features per scale
        weighted = [self.chan_attn[i](z) for i, z in enumerate(scale_features)]

        # F_weighted: mean across scales (B, feature_dim)
        f_weighted = torch.stack(weighted, dim=1).mean(dim=1)

        # F_MSA: attention-weighted sum (B, feature_dim)
        f_msa = self.msa(weighted)

        # F_HTD: Tucker decomposition output (B, feature_dim)
        f_htd = self.htd(weighted)

        # Concatenate and project (Eq. 16)
        f_cat   = torch.cat([f_weighted, f_msa, f_htd], dim=-1)
        f_final = self.norm(self.final_proj(f_cat))
        return f_final
