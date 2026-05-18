"""
Fractal Feature Extraction — Section 2.1 of the paper.

Three feature channels extracted from a stock price time series:
  1. Generalized Hurst exponent  Hq  (Eq. 5)   — long-range memory
  2. High-order fractal dimension FDq (Eq. 7-8) — self-similar complexity
  3. Multifractal spectrum       MFSq (Eq. 9-11)— local scale inhomogeneity

All three are computed over a sliding window so the output has the same
length as the input minus (window - 1).

Paper parameters:
  q  ∈ [-5, 5]   (fractal order)
  α, β           (Hurst control weights; α=0, β=2 → classic Hurst)
  γ              (NSIM nonlinear scale exponent)
  θ              (GFRFT kernel angle, default π/4)
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


# ---------------------------------------------------------------------------
# 2.1.1  Generalized Hurst exponent  Hq  (Eqs. 1-5)
# ---------------------------------------------------------------------------

def _rs_statistic(x: np.ndarray) -> float:
    """Classic R/S statistic for a sub-series x."""
    n = len(x)
    if n < 2:
        return np.nan
    mean_x = x.mean()
    y = np.cumsum(x - mean_x)           # cumulative deviation (Eq. 2)
    r = y.max() - y.min()               # range  (Eq. 3)
    s = np.std(x, ddof=0)              # std    (Eq. 3)
    return r / s if s > 1e-12 else np.nan


def hurst_rs(series: np.ndarray, min_n: int = 8) -> float:
    """
    Classic R/S Hurst exponent: regress log(R/S) ~ H*log(n). (Eq. 4)
    """
    n = len(series)
    ns, rs_vals = [], []
    for sub_n in [max(min_n, n // k) for k in range(2, max(3, n // min_n + 1))]:
        if sub_n < min_n:
            continue
        rs_list = [
            _rs_statistic(series[i: i + sub_n])
            for i in range(0, n - sub_n + 1, sub_n)
        ]
        rs_list = [v for v in rs_list if not np.isnan(v) and v > 0]
        if rs_list:
            ns.append(sub_n)
            rs_vals.append(np.mean(rs_list))
    if len(ns) < 2:
        return 0.5
    log_n = np.log(ns)
    log_rs = np.log(rs_vals)
    h, _ = np.polyfit(log_n, log_rs, 1)
    return float(np.clip(h, 0.0, 1.0))


def generalized_hurst(
    series: np.ndarray,
    q: float = 2.0,
    alpha: float = 0.0,
    beta: float = 2.0,
    min_n: int = 8,
) -> float:
    """
    Generalized Hurst exponent Hq^(α,β) corrected by Rényi entropy (Eq. 5).

    H_q^(α,β) = lim_{n→∞}  log( Σ P_i^α |X(i+n)-X(i)|^β ) / (q * log n)

    where P_i = |X(i)|^α / Σ|X(j)|^α  (fractal measure).
    When α=0, β=2 → classic Hurst.
    """
    n = len(series)
    if n < min_n * 2:
        return hurst_rs(series, min_n)

    # Fractal measure P_i
    abs_x = np.abs(series) + 1e-12
    if alpha == 0:
        p = np.ones(n) / n
    else:
        p_raw = abs_x ** alpha
        p = p_raw / p_raw.sum()

    log_ns, log_vals = [], []
    for sub_n in range(min_n, n // 2, max(1, (n // 2 - min_n) // 8)):
        increments = np.abs(series[sub_n:] - series[:-sub_n]) ** beta
        weighted = (p[:len(increments)] ** alpha) * increments
        val = weighted.sum()
        if val > 1e-15:
            log_ns.append(np.log(sub_n))
            log_vals.append(np.log(val))

    if len(log_ns) < 2:
        return hurst_rs(series, min_n)

    slope, _ = np.polyfit(log_ns, log_vals, 1)
    hq = slope / q if abs(q) > 1e-6 else hurst_rs(series, min_n)
    return float(np.clip(hq, 0.0, 1.5))


# ---------------------------------------------------------------------------
# 2.1.2  High-order fractal dimension  FDq  (Eqs. 6-8)
# ---------------------------------------------------------------------------

def fractal_dimension_q(
    series: np.ndarray,
    q: float = 2.0,
    scales: Sequence[int] | None = None,
) -> float:
    """
    Non-integer order fractal dimension FDq via box-counting (Eq. 7):

    FDq = lim_{f→0}  log( Σ_i (N_i(f)/N_total(f))^q ) / ((q-1)*log(1/f))
    """
    n = len(series)
    if scales is None:
        scales = [max(2, n // k) for k in [2, 4, 8, 16] if n // k >= 2]
        scales = sorted(set(scales))

    log_inv_f, log_sum = [], []
    for f in scales:
        n_boxes = max(1, n // f)
        counts = []
        for i in range(n_boxes):
            seg = series[i * f: (i + 1) * f]
            if len(seg) == 0:
                continue
            rng = seg.max() - seg.min()
            counts.append(rng + 1e-12)
        total = sum(counts)
        if total < 1e-15:
            continue
        probs = np.array(counts) / total
        if abs(q - 1) < 1e-6:
            # Shannon entropy (q→1 limit)
            s = -np.sum(probs * np.log(probs + 1e-15))
            log_sum.append(s)
        else:
            log_sum.append(np.log(np.sum(probs ** q) + 1e-15))
        log_inv_f.append(np.log(1.0 / f))

    if len(log_inv_f) < 2:
        return 1.5  # fallback

    if abs(q - 1) < 1e-6:
        slope, _ = np.polyfit(log_inv_f, log_sum, 1)
        return float(np.clip(slope, 0.0, 2.0))
    else:
        slope, _ = np.polyfit(log_inv_f, log_sum, 1)
        return float(np.clip(slope / (q - 1), 0.0, 2.0))


def fractal_dimension_gks(
    series: np.ndarray,
    q: float = 2.0,
    scales: Sequence[int] | None = None,
) -> float:
    """
    Generalized Kolmogorov-Smirnov corrected FD (Eq. 8):

    FDq^GKS = lim  log( Σ_i |P_i(f) - P_i(2f)|^q ) / ((q-1)*log(1/f))
    """
    n = len(series)
    if scales is None:
        scales = [max(2, n // k) for k in [4, 8, 16] if n // k >= 2]
        scales = sorted(set(scales))

    log_inv_f, log_sum = [], []
    for f in scales:
        f2 = f * 2
        if f2 > n:
            continue
        boxes_f  = max(1, n // f)
        boxes_f2 = max(1, n // f2)

        def _probs(nb, step):
            cnts = []
            for i in range(nb):
                seg = series[i * step: (i + 1) * step]
                cnts.append(seg.max() - seg.min() + 1e-12 if len(seg) else 1e-12)
            t = sum(cnts)
            return np.array(cnts) / t

        p1 = _probs(boxes_f, f)
        p2 = _probs(boxes_f2, f2)
        # align lengths
        min_len = min(len(p1), len(p2))
        diff = np.abs(p1[:min_len] - p2[:min_len]) ** q
        val = np.sum(diff)
        if val > 1e-15 and abs(q - 1) > 1e-6:
            log_sum.append(np.log(val))
            log_inv_f.append(np.log(1.0 / f))

    if len(log_inv_f) < 2:
        return fractal_dimension_q(series, q, scales)

    slope, _ = np.polyfit(log_inv_f, log_sum, 1)
    return float(np.clip(slope / (q - 1), 0.0, 2.0))


# ---------------------------------------------------------------------------
# 2.1.3  Multifractal spectrum  MFSq  (Eqs. 9-11)
# ---------------------------------------------------------------------------

def _gfrft_transform(series: np.ndarray, theta: float = np.pi / 4) -> np.ndarray:
    """
    Discrete approximation of generalized fractional Fourier transform (GFRFT).
    Kernel: K_θ(t,τ) = e^{iπtτ tanθ}  (Eq. 9, paper text).
    We use a chirp-multiplication approximation.
    """
    n = len(series)
    t = np.arange(n, dtype=float)
    chirp = np.exp(1j * np.pi * t ** 2 * np.tan(theta) / n)
    return np.fft.fft(series * chirp) * chirp


def multifractal_spectrum(
    series: np.ndarray,
    q_vals: Sequence[float] | None = None,
    scales: Sequence[int] | None = None,
    theta: float = np.pi / 4,
    gamma: float = 0.5,
) -> dict:
    """
    Compute multifractal spectrum MFSq (Eqs. 9-11).

    Returns dict with:
        'Dq'    : generalized dimension spectrum  (Eq. 9)
        'Dq_ft' : GFRFT-corrected Dq             (Eq. 10)
        'Dq_ns' : NSIM-corrected Dq              (Eq. 11)
        'alpha' : Hölder exponent  α = dDq/dq
        'f_alpha': singularity spectrum f(α) = qα - Dq
        'width' : spectral width  max(α) - min(α)  (key scalar feature)
        'height': max f(α)
    """
    n = len(series)
    if q_vals is None:
        q_vals = np.linspace(-5, 5, 21)
    q_vals = np.array(q_vals)

    if scales is None:
        scales = [max(2, n // k) for k in [4, 8, 16] if n // k >= 2]
        scales = sorted(set(scales))

    # Compute local measures μ_i at each scale
    def _measures(f):
        nb = max(1, n // f)
        segs = [series[i * f:(i + 1) * f] for i in range(nb) if len(series[i * f:(i + 1) * f]) > 0]
        amps = np.array([np.abs(s).mean() + 1e-12 for s in segs])
        return amps / amps.sum()

    Dq_list, Dq_ft_list, Dq_ns_list = [], [], []

    for q in q_vals:
        log_inv_f, log_mu_q, log_mu_ft_q, log_mu_ns_q = [], [], [], []
        for f in scales:
            mu = _measures(f)
            if abs(q - 1) < 1e-6:
                lmu = -np.sum(mu * np.log(mu + 1e-15))
            else:
                lmu = np.log(np.sum(mu ** q) + 1e-15)

            # GFRFT correction (Eq. 10): apply fractional Fourier to measures
            mu_ft = np.abs(_gfrft_transform(mu, theta))
            mu_ft = mu_ft / (mu_ft.sum() + 1e-12)
            if abs(q - 1) < 1e-6:
                lmu_ft = -np.sum(mu_ft * np.log(mu_ft + 1e-15))
            else:
                lmu_ft = np.log(np.sum(mu_ft ** q) + 1e-15)

            # NSIM correction (Eq. 11): (μ^γ - μ^(γ+1))^q
            mu_ns = np.abs(mu ** gamma - mu ** (gamma + 1)) + 1e-12
            if abs(q - 1) < 1e-6:
                lmu_ns = -np.sum(mu_ns * np.log(mu_ns + 1e-15))
            else:
                lmu_ns = np.log(np.sum(mu_ns ** q) + 1e-15)

            log_inv_f.append(np.log(1.0 / f))
            log_mu_q.append(lmu)
            log_mu_ft_q.append(lmu_ft)
            log_mu_ns_q.append(lmu_ns)

        if len(log_inv_f) < 2 or abs(q - 1) < 1e-6:
            Dq_list.append(1.0)
            Dq_ft_list.append(1.0)
            Dq_ns_list.append(1.0)
        else:
            denom = (q - 1)
            Dq_list.append(float(np.polyfit(log_inv_f, log_mu_q, 1)[0] / denom))
            Dq_ft_list.append(float(np.polyfit(log_inv_f, log_mu_ft_q, 1)[0] / denom))
            Dq_ns_list.append(float(np.polyfit(log_inv_f, log_mu_ns_q, 1)[0] / denom))

    Dq      = np.array(Dq_list)
    Dq_ft   = np.array(Dq_ft_list)
    Dq_ns   = np.array(Dq_ns_list)

    # Hölder exponent α = d(q*Dq)/dq and singularity spectrum f(α) = q*α - Dq
    tau_q   = q_vals * Dq
    alpha   = np.gradient(tau_q, q_vals)
    f_alpha = q_vals * alpha - Dq

    return {
        "Dq":      Dq,
        "Dq_ft":   Dq_ft,
        "Dq_ns":   Dq_ns,
        "alpha":   alpha,
        "f_alpha": f_alpha,
        "width":   float(alpha.max() - alpha.min()),
        "height":  float(f_alpha.max()),
        "q_vals":  q_vals,
    }


# ---------------------------------------------------------------------------
# Sliding-window feature extraction  (multi-scale: 16×16, 32×32, 64×64)
# ---------------------------------------------------------------------------

def extract_fractal_features(
    close: np.ndarray,
    window: int = 64,
    q_hurst: float = 2.0,
    q_fd: float = 2.0,
    alpha_h: float = 0.0,
    beta_h: float = 2.0,
    gamma: float = 0.5,
    theta: float = np.pi / 4,
    q_mfs: Sequence[float] | None = None,
    step: int = 1,
) -> np.ndarray:
    """
    Slide a window over the price series and compute the fractal feature vector
    at each step.

    Returns ndarray of shape (T, F) where T = (len(close)-window)//step + 1
    and F = number of fractal features per window.

    Feature vector per window (10 features):
      [0]  Hq      — generalized Hurst exponent
      [1]  FDq     — box-counting fractal dimension
      [2]  FDq_gks — GKS-corrected fractal dimension
      [3]  MFS_width    — multifractal spectrum width  max(α)-min(α)
      [4]  MFS_height   — max f(α)
      [5]  MFS_D0       — generalized dimension at q=0
      [6]  MFS_D2       — generalized dimension at q=2 (correlation dim)
      [7]  MFS_Dq_ft_width — GFRFT-corrected spectral width
      [8]  MFS_Dq_ns_width — NSIM-corrected spectral width
      [9]  RS_hurst      — classic R/S Hurst (baseline)
    """
    n = len(close)
    results = []

    if q_mfs is None:
        q_mfs = np.linspace(-5, 5, 11)

    positions = range(window, n + 1, step)
    for end in positions:
        seg = close[end - window: end]

        hq      = generalized_hurst(seg, q=q_hurst, alpha=alpha_h, beta=beta_h)
        fdq     = fractal_dimension_q(seg, q=q_fd)
        fdq_gks = fractal_dimension_gks(seg, q=q_fd)
        mfs     = multifractal_spectrum(seg, q_vals=q_mfs, theta=theta, gamma=gamma)

        # Find Dq at specific q values
        q_arr = mfs["q_vals"]
        idx0 = int(np.argmin(np.abs(q_arr - 0)))
        idx2 = int(np.argmin(np.abs(q_arr - 2)))
        D0 = float(mfs["Dq"][idx0])
        D2 = float(mfs["Dq"][idx2])

        ft_width = float(mfs["Dq_ft"].max() - mfs["Dq_ft"].min())
        ns_width = float(mfs["Dq_ns"].max() - mfs["Dq_ns"].min())

        rs_h = hurst_rs(seg)

        row = np.array([
            hq, fdq, fdq_gks,
            mfs["width"], mfs["height"],
            D0, D2,
            ft_width, ns_width,
            rs_h,
        ], dtype=np.float32)
        results.append(row)

    if not results:
        return np.empty((0, 10), dtype=np.float32)
    return np.stack(results)


def extract_multiscale_features(
    close: np.ndarray,
    windows: tuple[int, ...] = (16, 32, 64),
    **kwargs,
) -> dict[int, np.ndarray]:
    """
    Extract fractal features at multiple scales (paper uses 16×16, 32×32, 64×64).
    Returns dict {window_size: feature_array}.
    The arrays will have different lengths; they are aligned to the shortest.
    """
    feats = {}
    for w in windows:
        feats[w] = extract_fractal_features(close, window=w, **kwargs)
    return feats
