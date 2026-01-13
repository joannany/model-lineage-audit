"""
Null distribution generation and management.

Builds statistical baselines from independently-trained models
to enable principled provenance detection. Without null distributions,
high similarity could just mean "both are transformers trained on text."

The null distribution captures:
- What cosine similarity looks like between unrelated models
- Variance by layer (early layers often more similar across models)
- Variance by module type (attention vs MLP vs layernorm)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from .compare import CompareConfig, compare_state_dicts
from .io import TensorDict, load_state

logger = logging.getLogger(__name__)


@dataclass
class NullDistribution:
    """
    Statistical null distribution for model similarity metrics.
    
    Built from pairwise comparisons of independently-trained models.
    """

    # Summary statistics by grouping
    by_layer: pd.DataFrame
    by_module: pd.DataFrame
    by_layer_module: pd.DataFrame

    # Raw pairwise data for bootstrapping
    raw_pairs: pd.DataFrame

    # Metadata
    meta: Dict[str, Any] = field(default_factory=dict)

    def percentile(
        self,
        metric: str,
        value: float,
        layer: Optional[str] = None,
        module: Optional[str] = None,
    ) -> float:
        """
        Compute percentile of a value within the null distribution.
        
        Args:
            metric: Metric name (cosine, rel_l2, cka, spectral)
            value: Observed value
            layer: Layer to condition on (optional)
            module: Module to condition on (optional)
            
        Returns:
            Percentile (0-100) of the value in the null distribution
        """
        col = f"wmean_{metric}"

        if layer is not None and module is not None:
            df = self.by_layer_module
            mask = (df["layer"] == layer) & (df["module"] == module)
        elif layer is not None:
            df = self.by_layer
            mask = df["layer"] == layer
        elif module is not None:
            df = self.by_module
            mask = df["module"] == module
        else:
            # Global percentile
            df = self.by_layer
            mask = pd.Series(True, index=df.index)

        subset = df.loc[mask, col].dropna()
        if len(subset) == 0:
            logger.warning(f"No null samples for {layer}/{module}")
            return 50.0  # Neutral if no data

        return float(stats.percentileofscore(subset, value, kind="rank"))

    def z_score(
        self,
        metric: str,
        value: float,
        layer: Optional[str] = None,
        module: Optional[str] = None,
    ) -> float:
        """
        Compute z-score of a value relative to null distribution.
        
        Note: Model weights are not normally distributed, so interpret
        z-scores with caution. Percentiles are often more reliable.
        
        Args:
            metric: Metric name
            value: Observed value
            layer: Layer to condition on (optional)
            module: Module to condition on (optional)
            
        Returns:
            Z-score (positive = more similar than expected)
        """
        col = f"wmean_{metric}"

        if layer is not None and module is not None:
            df = self.by_layer_module
            mask = (df["layer"] == layer) & (df["module"] == module)
        elif layer is not None:
            df = self.by_layer
            mask = df["layer"] == layer
        elif module is not None:
            df = self.by_module
            mask = df["module"] == module
        else:
            df = self.by_layer
            mask = pd.Series(True, index=df.index)

        subset = df.loc[mask, col].dropna()
        if len(subset) < 2:
            return 0.0

        mean = subset.mean()
        std = subset.std()
        if std < 1e-10:
            return 0.0

        return float((value - mean) / std)

    def threshold(
        self,
        metric: str,
        percentile: float,
        layer: Optional[str] = None,
        module: Optional[str] = None,
    ) -> float:
        """
        Get the threshold value for a given percentile.
        
        Args:
            metric: Metric name
            percentile: Target percentile (0-100)
            layer: Layer to condition on (optional)
            module: Module to condition on (optional)
            
        Returns:
            Threshold value
        """
        col = f"wmean_{metric}"

        if layer is not None and module is not None:
            df = self.by_layer_module
            mask = (df["layer"] == layer) & (df["module"] == module)
        elif layer is not None:
            df = self.by_layer
            mask = df["layer"] == layer
        elif module is not None:
            df = self.by_module
            mask = df["module"] == module
        else:
            df = self.by_layer
            mask = pd.Series(True, index=df.index)

        subset = df.loc[mask, col].dropna()
        if len(subset) == 0:
            return 0.5  # Default neutral

        return float(np.percentile(subset, percentile))

    def save(self, path: str | Path) -> None:
        """Save null distribution to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.by_layer.to_csv(path / "by_layer.csv", index=False)
        self.by_module.to_csv(path / "by_module.csv", index=False)
        self.by_layer_module.to_csv(path / "by_layer_module.csv", index=False)
        self.raw_pairs.to_csv(path / "raw_pairs.csv", index=False)

        with open(path / "meta.json", "w") as f:
            json.dump(self.meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "NullDistribution":
        """Load null distribution from directory."""
        path = Path(path)

        by_layer = pd.read_csv(path / "by_layer.csv")
        by_module = pd.read_csv(path / "by_module.csv")
        by_layer_module = pd.read_csv(path / "by_layer_module.csv")
        raw_pairs = pd.read_csv(path / "raw_pairs.csv")

        with open(path / "meta.json") as f:
            meta = json.load(f)

        return cls(
            by_layer=by_layer,
            by_module=by_module,
            by_layer_module=by_layer_module,
            raw_pairs=raw_pairs,
            meta=meta,
        )


def build_null_distribution(
    model_paths: List[str],
    *,
    cfg: Optional[CompareConfig] = None,
    architecture: str = "unknown",
    description: str = "",
) -> NullDistribution:
    """
    Build null distribution from a set of independently-trained models.
    
    Args:
        model_paths: List of paths to model checkpoints
        cfg: Comparison configuration
        architecture: Architecture name for metadata
        description: Description for metadata
        
    Returns:
        NullDistribution built from all pairwise comparisons
    """
    if len(model_paths) < 2:
        raise ValueError("Need at least 2 models to build null distribution")

    if cfg is None:
        cfg = CompareConfig()
    cfg.progress = False

    # Load all models
    logger.info(f"Loading {len(model_paths)} models...")
    models: Dict[str, TensorDict] = {}
    for i, path in enumerate(tqdm(model_paths, desc="Loading models")):
        state = load_state(path)
        models[f"model_{i}"] = state.tensors

    # Compute all pairwise comparisons
    names = sorted(models.keys())
    all_by_layer = []
    all_by_module = []
    all_by_layer_module = []
    pair_summaries = []

    total_pairs = len(names) * (len(names) - 1) // 2
    logger.info(f"Computing {total_pairs} pairwise comparisons...")

    pair_idx = 0
    with tqdm(total=total_pairs, desc="Pairwise comparisons") as pbar:
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                result = compare_state_dicts(
                    models[name_a],
                    models[name_b],
                    cfg,
                )

                # Tag with pair info
                for df in [result.profiles["by_layer"]]:
                    df["pair_idx"] = pair_idx
                    df["model_a"] = name_a
                    df["model_b"] = name_b
                    all_by_layer.append(df.copy())

                for df in [result.profiles["by_module"]]:
                    df["pair_idx"] = pair_idx
                    df["model_a"] = name_a
                    df["model_b"] = name_b
                    all_by_module.append(df.copy())

                for df in [result.profiles["by_layer_module"]]:
                    df["pair_idx"] = pair_idx
                    df["model_a"] = name_a
                    df["model_b"] = name_b
                    all_by_layer_module.append(df.copy())

                summary = result.global_summary()
                summary["pair_idx"] = pair_idx
                summary["model_a"] = name_a
                summary["model_b"] = name_b
                pair_summaries.append(summary)

                pair_idx += 1
                pbar.update(1)

    meta = {
        "n_models": len(model_paths),
        "n_pairs": total_pairs,
        "architecture": architecture,
        "description": description,
        "model_paths": model_paths,
        "config": {
            "grouping": cfg.grouping,
            "exclude_embedding": cfg.exclude_embedding,
            "exclude_lm_head": cfg.exclude_lm_head,
        },
    }

    return NullDistribution(
        by_layer=pd.concat(all_by_layer, ignore_index=True),
        by_module=pd.concat(all_by_module, ignore_index=True),
        by_layer_module=pd.concat(all_by_layer_module, ignore_index=True),
        raw_pairs=pd.DataFrame(pair_summaries),
        meta=meta,
    )


def load_null_distribution(path: str | Path) -> NullDistribution:
    """Load a saved null distribution."""
    return NullDistribution.load(path)


# Pre-built null distributions for common architectures
# These would be populated by running build_null.py on sets of
# independently-trained models
PREBUILT_NULLS: Dict[str, str] = {
    # "llama-7b": "/path/to/llama_7b_null",
    # "mistral-7b": "/path/to/mistral_7b_null",
}


def get_prebuilt_null(architecture: str) -> Optional[NullDistribution]:
    """
    Get a prebuilt null distribution for a common architecture.
    
    Args:
        architecture: Architecture name (e.g., "llama-7b")
        
    Returns:
        NullDistribution if available, None otherwise
    """
    if architecture not in PREBUILT_NULLS:
        return None
    return load_null_distribution(PREBUILT_NULLS[architecture])
