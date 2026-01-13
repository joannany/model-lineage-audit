"""
Evidence scoring and statistical inference for lineage detection.

Combines observed similarities with null distributions to compute
evidence scores for model derivation claims.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .compare import CompareResult
from .nulls import NullDistribution

logger = logging.getLogger(__name__)


@dataclass
class LayerEvidence:
    """Evidence scores for a single layer."""

    layer: str
    cosine_obs: float
    cosine_pct: float  # Percentile in null distribution
    cosine_z: float
    rel_l2_obs: float
    rel_l2_pct: float
    rel_l2_z: float
    cka_obs: Optional[float] = None
    cka_pct: Optional[float] = None
    cka_z: Optional[float] = None
    n_params: int = 0


@dataclass
class ModuleEvidence:
    """Evidence scores for a single module type."""

    module: str
    cosine_obs: float
    cosine_pct: float
    cosine_z: float
    rel_l2_obs: float
    rel_l2_pct: float
    rel_l2_z: float
    cka_obs: Optional[float] = None
    cka_pct: Optional[float] = None
    cka_z: Optional[float] = None
    n_params: int = 0


@dataclass
class EvidenceResult:
    """Complete evidence assessment for a model pair."""

    # Summary scores
    global_evidence: Dict[str, float]

    # Per-layer evidence
    by_layer: List[LayerEvidence]

    # Per-module evidence
    by_module: List[ModuleEvidence]

    # Overall assessment
    verdict: str  # "strong_evidence", "moderate_evidence", "weak_evidence", "no_evidence"
    confidence: float  # 0-1 confidence in the verdict

    # Metadata
    meta: Dict[str, Any] = field(default_factory=dict)

    def summary_df(self) -> pd.DataFrame:
        """Convert layer evidence to DataFrame."""
        return pd.DataFrame([
            {
                "layer": e.layer,
                "cosine": e.cosine_obs,
                "cosine_pct": e.cosine_pct,
                "cosine_z": e.cosine_z,
                "rel_l2": e.rel_l2_obs,
                "rel_l2_pct": e.rel_l2_pct,
                "rel_l2_z": e.rel_l2_z,
                "cka": e.cka_obs,
                "cka_pct": e.cka_pct,
                "cka_z": e.cka_z,
                "n_params": e.n_params,
            }
            for e in self.by_layer
        ])

    def __str__(self) -> str:
        lines = [
            f"Evidence Result: {self.verdict.upper()}",
            f"Confidence: {self.confidence:.1%}",
            "",
            "Global Evidence:",
        ]
        for k, v in self.global_evidence.items():
            lines.append(f"  {k}: {v:.3f}")
        return "\n".join(lines)


def compute_evidence_scores(
    comparison: CompareResult,
    null_dist: NullDistribution,
    *,
    high_percentile_threshold: float = 95.0,
    low_percentile_threshold: float = 5.0,
) -> EvidenceResult:
    """
    Compute evidence scores by comparing observed similarities to null distribution.
    
    Args:
        comparison: Result from comparing two models
        null_dist: Null distribution from independent models
        high_percentile_threshold: Percentile above which similarity is suspicious
        low_percentile_threshold: Percentile below which similarity is suspicious (for rel_l2)
        
    Returns:
        EvidenceResult with comprehensive scoring
    """
    by_layer_df = comparison.profiles["by_layer"]
    by_module_df = comparison.profiles["by_module"]

    # Compute layer-level evidence
    layer_evidence: List[LayerEvidence] = []
    for _, row in by_layer_df.iterrows():
        layer = row["layer"]

        # Cosine similarity
        cos_obs = row.get("wmean_cosine", row.get("mean_cosine", 0.0))
        cos_pct = null_dist.percentile("cosine", cos_obs, layer=layer) if cos_obs else 50.0
        cos_z = null_dist.z_score("cosine", cos_obs, layer=layer) if cos_obs else 0.0

        # Relative L2 (lower = more similar)
        rel_obs = row.get("wmean_rel_l2", row.get("mean_rel_l2", 1.0))
        rel_pct = 100 - null_dist.percentile("rel_l2", rel_obs, layer=layer) if rel_obs else 50.0
        rel_z = -null_dist.z_score("rel_l2", rel_obs, layer=layer) if rel_obs else 0.0

        # CKA (if available)
        cka_obs = row.get("wmean_cka", row.get("mean_cka"))
        cka_pct = None
        cka_z = None
        if cka_obs is not None and not np.isnan(cka_obs):
            cka_pct = null_dist.percentile("cka", cka_obs, layer=layer)
            cka_z = null_dist.z_score("cka", cka_obs, layer=layer)

        layer_evidence.append(LayerEvidence(
            layer=layer,
            cosine_obs=cos_obs or 0.0,
            cosine_pct=cos_pct,
            cosine_z=cos_z,
            rel_l2_obs=rel_obs or 1.0,
            rel_l2_pct=rel_pct,
            rel_l2_z=rel_z,
            cka_obs=cka_obs,
            cka_pct=cka_pct,
            cka_z=cka_z,
            n_params=int(row.get("n_params_total", 0)),
        ))

    # Compute module-level evidence
    module_evidence: List[ModuleEvidence] = []
    for _, row in by_module_df.iterrows():
        module = row["module"]

        cos_obs = row.get("wmean_cosine", row.get("mean_cosine", 0.0))
        cos_pct = null_dist.percentile("cosine", cos_obs, module=module) if cos_obs else 50.0
        cos_z = null_dist.z_score("cosine", cos_obs, module=module) if cos_obs else 0.0

        rel_obs = row.get("wmean_rel_l2", row.get("mean_rel_l2", 1.0))
        rel_pct = 100 - null_dist.percentile("rel_l2", rel_obs, module=module) if rel_obs else 50.0
        rel_z = -null_dist.z_score("rel_l2", rel_obs, module=module) if rel_obs else 0.0

        cka_obs = row.get("wmean_cka", row.get("mean_cka"))
        cka_pct = None
        cka_z = None
        if cka_obs is not None and not np.isnan(cka_obs):
            cka_pct = null_dist.percentile("cka", cka_obs, module=module)
            cka_z = null_dist.z_score("cka", cka_obs, module=module)

        module_evidence.append(ModuleEvidence(
            module=module,
            cosine_obs=cos_obs or 0.0,
            cosine_pct=cos_pct,
            cosine_z=cos_z,
            rel_l2_obs=rel_obs or 1.0,
            rel_l2_pct=rel_pct,
            rel_l2_z=rel_z,
            cka_obs=cka_obs,
            cka_pct=cka_pct,
            cka_z=cka_z,
            n_params=int(row.get("n_params_total", 0)),
        ))

    # Compute global evidence
    global_evidence = _compute_global_evidence(
        layer_evidence,
        module_evidence,
        high_percentile_threshold,
    )

    # Determine verdict
    verdict, confidence = _determine_verdict(
        global_evidence,
        layer_evidence,
        high_percentile_threshold,
    )

    return EvidenceResult(
        global_evidence=global_evidence,
        by_layer=layer_evidence,
        by_module=module_evidence,
        verdict=verdict,
        confidence=confidence,
        meta={
            "high_percentile_threshold": high_percentile_threshold,
            "low_percentile_threshold": low_percentile_threshold,
        },
    )


def _compute_global_evidence(
    layer_evidence: List[LayerEvidence],
    module_evidence: List[ModuleEvidence],
    threshold: float,
) -> Dict[str, float]:
    """Compute global summary statistics.

    Notes:
      - We primarily use percentiles (non-parametric) derived from the null distribution.
      - CKA may be missing for large tensors (by design); in that case we simply omit
        CKA-driven global aggregates rather than imputing.
    """
    if not layer_evidence:
        return {}

    # Weighted by parameter count
    total_params = sum(e.n_params for e in layer_evidence) or 1

    # Weighted mean percentiles
    wmean_cos_pct = sum(e.cosine_pct * e.n_params for e in layer_evidence) / total_params
    wmean_rel_pct = sum(e.rel_l2_pct * e.n_params for e in layer_evidence) / total_params

    # Fraction of layers above threshold (cosine)
    above_threshold_cos = sum(1 for e in layer_evidence if e.cosine_pct >= threshold)
    frac_above_cos = above_threshold_cos / len(layer_evidence)

    # Max percentile (strongest signal)
    max_cos_pct = max(e.cosine_pct for e in layer_evidence)

    # Combined z-score (rough aggregate; percentiles are typically more robust)
    cos_zs = [e.cosine_z for e in layer_evidence if e.cosine_z is not None and e.cosine_z != 0]
    combined_cos_z = (sum(cos_zs) / np.sqrt(len(cos_zs))) if cos_zs else 0.0

    # Optional: CKA aggregates (only if we have CKA for at least some layers)
    cka_layers = [e for e in layer_evidence if e.cka_pct is not None and e.n_params > 0]
    wmean_cka_pct = None
    frac_above_cka = None
    max_cka_pct = None
    combined_cka_z = None
    if cka_layers:
        total_params_cka = sum(e.n_params for e in cka_layers) or 1
        wmean_cka_pct = sum(float(e.cka_pct) * e.n_params for e in cka_layers) / total_params_cka
        above_threshold_cka = sum(1 for e in cka_layers if float(e.cka_pct) >= threshold)
        frac_above_cka = above_threshold_cka / len(cka_layers)
        max_cka_pct = max(float(e.cka_pct) for e in cka_layers)
        cka_zs = [float(e.cka_z) for e in cka_layers if e.cka_z is not None and float(e.cka_z) != 0.0]
        combined_cka_z = (sum(cka_zs) / np.sqrt(len(cka_zs))) if cka_zs else 0.0

    out = {
        "weighted_mean_cosine_pct": float(wmean_cos_pct),
        "weighted_mean_rel_l2_pct": float(wmean_rel_pct),
        "fraction_above_threshold": float(frac_above_cos),
        "max_cosine_percentile": float(max_cos_pct),
        "combined_z_score": float(combined_cos_z),
        "n_layers": float(len(layer_evidence)),
        "n_modules": float(len(module_evidence)),
    }
    if wmean_cka_pct is not None:
        out.update({
            "weighted_mean_cka_pct": float(wmean_cka_pct),
            "fraction_above_threshold_cka": float(frac_above_cka or 0.0),
            "max_cka_percentile": float(max_cka_pct or 0.0),
            "combined_cka_z_score": float(combined_cka_z or 0.0),
            "n_layers_with_cka": float(len(cka_layers)),
        })
    return out


def _determine_verdict(
    global_evidence: Dict[str, float],
    layer_evidence: List[LayerEvidence],
    threshold: float,
) -> Tuple[str, float]:
    """
    Determine overall verdict and confidence.

    Heuristic (conservative) rules:
      - Primary signal is cosine percentile consistency across layers.
      - If CKA percentiles are available, they act as a corroborating signal.
        We do NOT treat CKA alone as sufficient for 'strong evidence' because it
        is more sensitive to architectural/normalization choices and may be missing
        for large tensors.
    """
    if not global_evidence:
        return "no_evidence", 0.0

    frac_above_cos = float(global_evidence.get("fraction_above_threshold", 0.0))
    max_cos_pct = float(global_evidence.get("max_cosine_percentile", 50.0))
    wmean_cos_pct = float(global_evidence.get("weighted_mean_cosine_pct", 50.0))

    # Optional CKA aggregates
    wmean_cka_pct = global_evidence.get("weighted_mean_cka_pct", None)
    frac_above_cka = global_evidence.get("fraction_above_threshold_cka", None)

    # Helper: treat CKA as corroboration if present and high
    has_strong_cka = (wmean_cka_pct is not None and float(wmean_cka_pct) >= 99.0) or (
        frac_above_cka is not None and float(frac_above_cka) >= 0.9
    )
    has_moderate_cka = (wmean_cka_pct is not None and float(wmean_cka_pct) >= 95.0) or (
        frac_above_cka is not None and float(frac_above_cka) >= 0.5
    )

    # Strong evidence: most layers above ~99th percentile on cosine,
    # with corroboration from CKA if available.
    if frac_above_cos >= 0.9 and wmean_cos_pct >= 99.0:
        confidence = min(1.0, (wmean_cos_pct - 95.0) / 5.0)
        if wmean_cka_pct is not None:
            confidence = min(1.0, confidence + (0.1 if has_strong_cka else 0.0))
        return "strong_evidence", confidence

    # Moderate evidence: majority above threshold on cosine, or slightly weaker cosine
    # but corroborated by strong CKA.
    if (frac_above_cos >= 0.5 and wmean_cos_pct >= 95.0) or (
        frac_above_cos >= 0.4 and wmean_cos_pct >= 94.0 and has_strong_cka
    ):
        confidence = min(1.0, max(frac_above_cos, 0.5) * (wmean_cos_pct - 90.0) / 10.0)
        if wmean_cka_pct is not None:
            confidence = min(1.0, confidence + (0.05 if has_moderate_cka else 0.0))
        return "moderate_evidence", confidence

    # Weak evidence: some layers suspicious on cosine or one extreme outlier,
    # again allowing corroboration from CKA if cosine is borderline.
    if frac_above_cos >= 0.2 or max_cos_pct >= 99.0 or (has_moderate_cka and wmean_cos_pct >= 92.0):
        confidence = max(frac_above_cos, (max_cos_pct - 95.0) / 5.0)
        if wmean_cka_pct is not None and has_moderate_cka:
            confidence = min(0.7, confidence + 0.05)
        return "weak_evidence", min(confidence, 0.7)

    # No evidence
    confidence = 1.0 - frac_above_cos - (wmean_cos_pct - 50.0) / 100.0
    return "no_evidence", max(0.0, min(1.0, confidence))


def format_evidence_report(evidence: EvidenceResult) -> str:
    """Format evidence result as a human-readable report."""

    def _is_nan(x: object) -> bool:
        try:
            return x is not None and float(x) != float(x)  # NaN check
        except Exception:
            return False
    lines = [
        "=" * 60,
        "MODEL LINEAGE EVIDENCE REPORT",
        "=" * 60,
        "",
        f"VERDICT: {evidence.verdict.replace('_', ' ').upper()}",
        f"Confidence: {evidence.confidence:.1%}",
        "",
        "-" * 60,
        "GLOBAL EVIDENCE SCORES",
        "-" * 60,
    ]

    for key, value in evidence.global_evidence.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.3f}")
        else:
            lines.append(f"  {key}: {value}")

    lines.extend([
        "",
        "-" * 60,
        "PER-LAYER EVIDENCE",
        "-" * 60,
        f"{'Layer':<15} {'CosObs':>7} {'CosPct':>7} {'RelObs':>7} {'RelPct':>7} {'CKAObs':>7} {'CKAPct':>7} {'Params':>12}",
        "-" * 60,
    ])

    for e in evidence.by_layer:
        cka_obs_str = f"{float(e.cka_obs):>7.4f}" if (e.cka_obs is not None and not _is_nan(e.cka_obs)) else "    N/A"
        cka_pct_str = f"{float(e.cka_pct):>6.1f}%" if (e.cka_pct is not None and not _is_nan(e.cka_pct)) else "   N/A "
        lines.append(
            f"{e.layer:<15} {e.cosine_obs:>7.4f} {e.cosine_pct:>6.1f}% "
            f"{e.rel_l2_obs:>7.4f} {e.rel_l2_pct:>6.1f}% "
            f"{cka_obs_str} {cka_pct_str} {e.n_params:>12,}"
        )
    lines.extend([
        "",
        "-" * 60,
        "PER-MODULE EVIDENCE",
        "-" * 60,
        f"{'Module':<15} {'CosObs':>7} {'CosPct':>7} {'RelObs':>7} {'RelPct':>7} {'CKAObs':>7} {'CKAPct':>7} {'Params':>12}",
        "-" * 60,
    ])

    for e in evidence.by_module:
        cka_obs_str = f"{float(e.cka_obs):>7.4f}" if (e.cka_obs is not None and not _is_nan(e.cka_obs)) else "    N/A"
        cka_pct_str = f"{float(e.cka_pct):>6.1f}%" if (e.cka_pct is not None and not _is_nan(e.cka_pct)) else "   N/A "
        lines.append(
            f"{e.module:<15} {e.cosine_obs:>7.4f} {e.cosine_pct:>6.1f}% "
            f"{e.rel_l2_obs:>7.4f} {e.rel_l2_pct:>6.1f}% "
            f"{cka_obs_str} {cka_pct_str} {e.n_params:>12,}"
        )
    lines.extend([
        "",
        "=" * 60,
        "INTERPRETATION GUIDE",
        "-" * 60,
        "- Percentile indicates where observed similarity falls in the",
        "  distribution of independently-trained model pairs.",
        "- 99th percentile means only 1% of independent pairs are this similar.",
        "- Strong evidence requires most layers consistently above 99th percentile.",
        "- High cosine + low rel_l2 + high CKA together strengthen the case.",
        "=" * 60,
    ])

    return "\n".join(lines)
