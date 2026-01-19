"""
Similarity metrics for comparing model weights.

Provides:
- Cosine similarity
- Relative L2 distance (symmetric)
- Linear CKA (Centered Kernel Alignment)
- Spectral similarity (eigenvalue comparison)
- Frobenius norm utilities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from scipy import linalg


@dataclass
class MetricResult:
    """Container for a single metric computation result."""

    value: float
    metric_name: str
    n_params: int
    notes: Optional[str] = None


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Compute cosine similarity between two tensors.
    
    Tensors are flattened and compared as vectors.
    
    Args:
        a: First tensor
        b: Second tensor
        eps: Small constant for numerical stability
        
    Returns:
        Cosine similarity in range [-1, 1]
    """
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()

    a_norm = torch.norm(a_flat) + eps
    b_norm = torch.norm(b_flat) + eps

    cosine = torch.dot(a_flat, b_flat) / (a_norm * b_norm)
    return float(torch.clamp(cosine, -1.0, 1.0))


def rel_l2_symmetric(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Compute symmetric relative L2 distance.
    
    Uses the average of norms as reference, making the metric symmetric:
    rel_l2(a, b) == rel_l2(b, a)
    
    Args:
        a: First tensor
        b: Second tensor
        eps: Small constant for numerical stability
        
    Returns:
        Relative L2 distance (0 = identical, larger = more different)
    """
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()

    diff_norm = torch.norm(a_flat - b_flat)
    avg_norm = (torch.norm(a_flat) + torch.norm(b_flat)) / 2 + eps

    return float(diff_norm / avg_norm)


def frobenius_norm(a: torch.Tensor) -> float:
    """Compute Frobenius norm of a tensor."""
    return float(torch.norm(a.flatten().float()))


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """
    Compute Hilbert-Schmidt Independence Criterion.
    
    Uses the unbiased estimator from Song et al. (2012).
    """
    n = K.shape[0]
    if n < 4:
        return 0.0

    # Center the kernel matrices
    K_centered = K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()
    L_centered = L - L.mean(axis=0, keepdims=True) - L.mean(axis=1, keepdims=True) + L.mean()

    # Compute HSIC
    return float(np.sum(K_centered * L_centered) / ((n - 1) ** 2))


def linear_cka(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """
    Compute Linear Centered Kernel Alignment (CKA).
    
    CKA measures similarity of representations that is invariant to
    orthogonal transformations and isotropic scaling.
    
    For weight matrices, this compares the structure of learned
    representations rather than raw parameter values.
    
    Reference:
        Kornblith et al. (2019) "Similarity of Neural Network Representations Revisited"
    
    Args:
        a: First tensor (will be reshaped to 2D if needed)
        b: Second tensor (same shape as a)
        eps: Small constant for numerical stability
        
    Returns:
        CKA similarity in range [0, 1]
    """
    # Reshape to 2D matrices: [n, d]
    a_2d = a.float().reshape(a.shape[0], -1) if a.dim() > 1 else a.float().unsqueeze(0)
    b_2d = b.float().reshape(b.shape[0], -1) if b.dim() > 1 else b.float().unsqueeze(0)

    # Convert to numpy for stable centering math
    X = a_2d.numpy()
    Y = b_2d.numpy()
    n = X.shape[0]

    # Linear kernels
    K = X @ X.T
    L = Y @ Y.T

    def _center_gram_unbiased(G: np.ndarray) -> np.ndarray:
        """Unbiased Gram matrix centering (Kornblith et al., 2019).

        This reduces positive bias of linear CKA on finite samples.
        """
        G = G.copy()
        if G.shape[0] < 4:
            return G * 0.0
        np.fill_diagonal(G, 0.0)
        means = G.sum(axis=0) / (n - 2)
        means = means - means.sum() / (2 * (n - 1))
        G = G - means[None, :] - means[:, None]
        np.fill_diagonal(G, 0.0)
        return G

    Kc = _center_gram_unbiased(K)
    Lc = _center_gram_unbiased(L)

    # CKA = <Kc, Lc>_F / (||Kc||_F ||Lc||_F)
    hsic_xy = float(np.sum(Kc * Lc))
    hsic_xx = float(np.sum(Kc * Kc))
    hsic_yy = float(np.sum(Lc * Lc))
    denom = np.sqrt(hsic_xx * hsic_yy) + eps
    return float(np.clip(hsic_xy / denom, 0.0, 1.0))


def spectral_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
    top_k: Optional[int] = None,
    eps: float = 1e-12,
) -> float:
    """
    Compare tensors by their spectral properties (singular values).
    
    Computes cosine similarity of normalized singular value spectra.
    This captures structural similarity that is invariant to rotations.
    
    Args:
        a: First tensor (will be reshaped to 2D)
        b: Second tensor
        top_k: Only compare top k singular values (None = all)
        eps: Small constant for numerical stability
        
    Returns:
        Spectral similarity in range [0, 1]
    """
    # Reshape to 2D
    a_2d = a.float().reshape(a.shape[0], -1) if a.dim() > 1 else a.float().unsqueeze(0)
    b_2d = b.float().reshape(b.shape[0], -1) if b.dim() > 1 else b.float().unsqueeze(0)

    # Compute singular values
    try:
        s_a = torch.linalg.svdvals(a_2d)
        s_b = torch.linalg.svdvals(b_2d)
    except RuntimeError:
        # SVD can fail for ill-conditioned matrices
        return 0.0

    # Truncate if requested
    if top_k is not None:
        s_a = s_a[:top_k]
        s_b = s_b[:top_k]

    # Pad shorter spectrum with zeros
    max_len = max(len(s_a), len(s_b))
    if len(s_a) < max_len:
        s_a = torch.nn.functional.pad(s_a, (0, max_len - len(s_a)))
    if len(s_b) < max_len:
        s_b = torch.nn.functional.pad(s_b, (0, max_len - len(s_b)))

    # Normalize and compute cosine similarity
    s_a_norm = s_a / (torch.norm(s_a) + eps)
    s_b_norm = s_b / (torch.norm(s_b) + eps)

    return float(torch.dot(s_a_norm, s_b_norm))


def effective_rank(a: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Compute effective rank of a tensor.
    
    Effective rank measures the "true dimensionality" of a matrix,
    accounting for the distribution of singular values.
    
    Reference:
        Roy & Vetterli (2007) "The effective rank: A measure of effective dimensionality"
    
    Args:
        a: Input tensor (will be reshaped to 2D)
        eps: Small constant for numerical stability
        
    Returns:
        Effective rank (continuous value)
    """
    a_2d = a.float().reshape(a.shape[0], -1) if a.dim() > 1 else a.float().unsqueeze(0)

    try:
        s = torch.linalg.svdvals(a_2d)
    except RuntimeError:
        return 1.0

    # Normalize singular values to get distribution
    s = s + eps
    s_norm = s / s.sum()

    # Compute entropy
    entropy = -torch.sum(s_norm * torch.log(s_norm))

    # Effective rank = exp(entropy)
    return float(torch.exp(entropy))


def weight_stats(a: torch.Tensor) -> dict:
    """
    Compute various statistics of a weight tensor.
    
    Args:
        a: Input tensor
        
    Returns:
        Dict with mean, std, min, max, sparsity, effective_rank
    """
    a_flat = a.flatten().float()

    return {
        "mean": float(a_flat.mean()),
        "std": float(a_flat.std()),
        "min": float(a_flat.min()),
        "max": float(a_flat.max()),
        "sparsity": float((a_flat.abs() < 1e-8).float().mean()),
        "frobenius": frobenius_norm(a),
        "effective_rank": effective_rank(a),
    }


def compute_all_metrics(
    a: torch.Tensor,
    b: torch.Tensor,
    compute_cka: bool = True,
    compute_spectral: bool = True,
) -> dict:
    """
    Compute all available similarity metrics between two tensors.
    
    Args:
        a: First tensor
        b: Second tensor
        compute_cka: Whether to compute CKA (can be slow for large tensors)
        compute_spectral: Whether to compute spectral similarity
        
    Returns:
        Dict with all metric values
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    result = {
        "cosine": cosine_sim(a, b),
        "rel_l2": rel_l2_symmetric(a, b),
        "n_params": int(a.numel()),
    }

    # CKA can be expensive for large tensors
    if compute_cka and a.numel() < 10_000_000:  # 10M param limit
        result["cka"] = linear_cka(a, b)
    else:
        result["cka"] = None

    # Spectral similarity
    if compute_spectral and a.dim() > 1 and a.numel() < 50_000_000:
        result["spectral"] = spectral_similarity(a, b)
    else:
        result["spectral"] = None

    return result
