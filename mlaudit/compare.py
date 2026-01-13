"""
Pairwise model comparison and aggregation.

Compares two model checkpoints tensor-by-tensor and produces
aggregated similarity profiles by layer and module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from .groupings import GROUPINGS, GroupingResult, auto_detect_grouping
from .io import TensorDict, get_common_keys
from .metrics import compute_all_metrics, cosine_sim, linear_cka, rel_l2_symmetric, spectral_similarity

logger = logging.getLogger(__name__)


@dataclass
class CompareConfig:
    """Configuration for model comparison."""

    grouping: str = "auto"
    exclude_embedding: bool = False
    exclude_lm_head: bool = False
    max_keys: Optional[int] = None  # For quick smoke tests
    compute_cka: bool = True
    compute_spectral: bool = True
    progress: bool = True  # Show progress bar


@dataclass
class CompareResult:
    """Result of comparing two models."""

    raw_df: pd.DataFrame
    profiles: Dict[str, pd.DataFrame]
    meta: Dict[str, Any]
    coverage: Dict[str, int] = field(default_factory=dict)

    def global_summary(self) -> Dict[str, float]:
        """
        Compute global summary statistics.
        
        Returns weighted mean of all metrics across all tensors.
        """
        df = self.raw_df
        if df.empty:
            return {}

        weights = df["n_params"].astype(float)
        total_weight = weights.sum()

        summary = {
            "total_params_compared": int(df["n_params"].sum()),
            "n_tensors_compared": len(df),
        }

        for col in ["cosine", "rel_l2", "cka", "spectral"]:
            if col in df.columns:
                valid = df[col].notna()
                if valid.any():
                    w = weights[valid]
                    summary[f"weighted_mean_{col}"] = float((df.loc[valid, col] * w).sum() / w.sum())
                    summary[f"mean_{col}"] = float(df.loc[valid, col].mean())
                    summary[f"min_{col}"] = float(df.loc[valid, col].min())
                    summary[f"max_{col}"] = float(df.loc[valid, col].max())

        return summary


def compare_state_dicts(
    sd_a: TensorDict,
    sd_b: TensorDict,
    cfg: Optional[CompareConfig] = None,
) -> CompareResult:
    """
    Compare two model state dicts tensor-by-tensor.
    
    Args:
        sd_a: First model's state dict
        sd_b: Second model's state dict
        cfg: Comparison configuration
        
    Returns:
        CompareResult with raw metrics and aggregated profiles
    """
    if cfg is None:
        cfg = CompareConfig()

    # Auto-detect grouping if requested
    if cfg.grouping == "auto":
        grouping_name = auto_detect_grouping(list(sd_a.keys()))
        logger.info(f"Auto-detected grouping: {grouping_name}")
    else:
        grouping_name = cfg.grouping

    grouper = GROUPINGS.get(grouping_name, GROUPINGS["default"])

    # Find common keys
    common, a_only, b_only = get_common_keys(sd_a, sd_b, require_same_shape=True)

    coverage = {
        "common": len(common),
        "a_only": len(a_only),
        "b_only": len(b_only),
        "shape_matched": len(common),
    }

    if cfg.max_keys is not None:
        common = common[: cfg.max_keys]

    # Compute metrics for each tensor
    rows: List[Dict[str, Any]] = []
    iterator = tqdm(common, desc="Comparing tensors") if cfg.progress else common

    for key in iterator:
        a = sd_a[key]
        b = sd_b[key]

        g = grouper(key)

        # Apply exclusions
        if cfg.exclude_embedding and g.module == "embedding":
            continue
        if cfg.exclude_lm_head and g.module == "lm_head":
            continue

        metrics = compute_all_metrics(
            a, b,
            compute_cka=cfg.compute_cka,
            compute_spectral=cfg.compute_spectral,
        )

        row = {
            "key": key,
            "layer": g.layer,
            "module": g.module,
            "shape": str(tuple(a.shape)),
            **metrics,
        }
        rows.append(row)

    if not rows:
        raise ValueError(
            "No tensors to compare after filtering. "
            f"Coverage: {coverage}"
        )

    raw_df = pd.DataFrame(rows)
    profiles = aggregate_profiles(raw_df)

    meta = {
        "grouping": grouping_name,
        "exclude_embedding": cfg.exclude_embedding,
        "exclude_lm_head": cfg.exclude_lm_head,
        "compute_cka": cfg.compute_cka,
        "compute_spectral": cfg.compute_spectral,
    }

    return CompareResult(
        raw_df=raw_df,
        profiles=profiles,
        meta=meta,
        coverage=coverage,
    )


def _weighted_agg(g: pd.DataFrame, metric_cols: List[str]) -> pd.Series:
    """Aggregate a group with weighted and unweighted means."""
    result = {}
    weights = g["n_params"].astype(float)
    total_weight = weights.sum()

    for col in metric_cols:
        if col not in g.columns:
            continue
        valid = g[col].notna()
        if valid.any():
            result[f"mean_{col}"] = g.loc[valid, col].mean()
            w = weights[valid]
            result[f"wmean_{col}"] = (g.loc[valid, col] * w).sum() / w.sum()
        else:
            result[f"mean_{col}"] = None
            result[f"wmean_{col}"] = None

    result["n_tensors"] = len(g)
    result["n_params_total"] = int(total_weight)

    return pd.Series(result)


def aggregate_profiles(raw_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Aggregate raw tensor metrics into layer/module profiles.
    
    Returns:
        Dict with keys: by_layer, by_module, by_layer_module
    """
    metric_cols = ["cosine", "rel_l2", "cka", "spectral"]

    # By layer
    by_layer = raw_df.groupby("layer", dropna=False).apply(
        lambda g: _weighted_agg(g, metric_cols)
    ).reset_index()

    # By module
    by_module = raw_df.groupby("module", dropna=False).apply(
        lambda g: _weighted_agg(g, metric_cols)
    ).reset_index()

    # By layer and module
    by_layer_module = raw_df.groupby(["layer", "module"], dropna=False).apply(
        lambda g: _weighted_agg(g, metric_cols)
    ).reset_index()

    # Sort by natural layer order
    def layer_sort_key(x: str) -> Tuple[int, str]:
        if x.startswith("layer_"):
            suffix = x[6:]
            if suffix == "embed":
                return (-1, "")
            if suffix == "final":
                return (999999, "")
            if suffix.isdigit():
                return (int(suffix), "")
        return (999998, x)

    by_layer["_sort"] = by_layer["layer"].apply(lambda x: layer_sort_key(x))
    by_layer = by_layer.sort_values("_sort").drop(columns=["_sort"])

    by_layer_module["_sort"] = by_layer_module["layer"].apply(lambda x: layer_sort_key(x))
    by_layer_module = by_layer_module.sort_values(["_sort", "module"]).drop(columns=["_sort"])

    by_module = by_module.sort_values("module")

    return {
        "by_layer": by_layer.reset_index(drop=True),
        "by_module": by_module.reset_index(drop=True),
        "by_layer_module": by_layer_module.reset_index(drop=True),
    }


def compare_multiple(
    models: Dict[str, TensorDict],
    cfg: Optional[CompareConfig] = None,
) -> pd.DataFrame:
    """
    Compute pairwise comparisons between multiple models.
    
    Args:
        models: Dict mapping model names to state dicts
        cfg: Comparison configuration
        
    Returns:
        DataFrame with pairwise global similarities
    """
    if cfg is None:
        cfg = CompareConfig()
    cfg.progress = False  # Disable per-pair progress

    names = sorted(models.keys())
    rows = []

    total_pairs = len(names) * (len(names) - 1) // 2
    with tqdm(total=total_pairs, desc="Pairwise comparisons") as pbar:
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                result = compare_state_dicts(
                    models[name_a],
                    models[name_b],
                    cfg,
                )
                summary = result.global_summary()
                rows.append({
                    "model_a": name_a,
                    "model_b": name_b,
                    **summary,
                })
                pbar.update(1)

    return pd.DataFrame(rows)
