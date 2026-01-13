"""
Report generation utilities.

Produces:
- CSV tables (raw metrics, aggregated profiles)
- Matplotlib/Seaborn plots
- Markdown reports
- Summary statistics
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .compare import CompareResult
from .nulls import NullDistribution
from .scoring import EvidenceResult, format_evidence_report

logger = logging.getLogger(__name__)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 160


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_comparison_tables(
    out_dir: Path,
    result: CompareResult,
) -> List[Path]:
    """Save comparison results as CSV files."""
    paths = []

    # Raw tensor metrics
    raw_path = out_dir / "raw_tensor_metrics.csv"
    result.raw_df.to_csv(raw_path, index=False)
    paths.append(raw_path)

    # Aggregated profiles
    for name, df in result.profiles.items():
        profile_path = out_dir / f"{name}.csv"
        df.to_csv(profile_path, index=False)
        paths.append(profile_path)

    return paths


def _layer_sort_key(layer: str) -> Tuple[int, str]:
    """Sort key for layer names."""
    if layer.startswith("layer_"):
        suffix = layer[6:]
        if suffix == "embed":
            return (-1, "")
        if suffix == "final":
            return (999999, "")
        if suffix.isdigit():
            return (int(suffix), "")
    if layer == "unknown":
        return (999998, "")
    return (999997, layer)


def plot_layer_profile(
    out_dir: Path,
    by_layer: pd.DataFrame,
    title: str = "Layer-wise Similarity Profile",
    metric: str = "wmean_cosine",
    figsize: Tuple[int, int] = (14, 5),
) -> Path:
    """
    Plot layer-wise similarity profile.
    
    Args:
        out_dir: Output directory
        by_layer: DataFrame with layer-level metrics
        title: Plot title
        metric: Column to plot
        figsize: Figure size
        
    Returns:
        Path to saved figure
    """
    df = by_layer.copy()

    # Sort by layer order
    df["_sort"] = df["layer"].apply(_layer_sort_key)
    df = df.sort_values("_sort").reset_index(drop=True)

    x = df["layer"].tolist()
    y = df[metric].tolist()

    fig, ax = plt.subplots(figsize=figsize)

    # Color by value (higher = more similar)
    colors = plt.cm.RdYlGn(np.array(y) / 2 + 0.5)  # Map [-1,1] to [0,1]

    bars = ax.bar(range(len(x)), y, color=colors, edgecolor="gray", linewidth=0.5)

    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=60, ha="right", fontsize=8)
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Baseline (0.5)")
    ax.axhline(y=0.9, color="orange", linestyle="--", alpha=0.5, label="High similarity (0.9)")
    ax.axhline(y=0.99, color="red", linestyle="--", alpha=0.5, label="Very high (0.99)")

    ax.set_xlabel("Layer")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()

    out_path = out_dir / f"layer_profile_{metric}.png"
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


def plot_module_profile(
    out_dir: Path,
    by_module: pd.DataFrame,
    title: str = "Module-wise Similarity Profile",
    metric: str = "wmean_cosine",
    figsize: Tuple[int, int] = (10, 5),
) -> Path:
    """Plot module-wise similarity profile."""
    df = by_module.sort_values("module").reset_index(drop=True)

    x = df["module"].tolist()
    y = df[metric].tolist()

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.RdYlGn(np.array(y) / 2 + 0.5)
    bars = ax.bar(range(len(x)), y, color=colors, edgecolor="gray", linewidth=0.5)

    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=45, ha="right")
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Module Type")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)

    plt.tight_layout()

    out_path = out_dir / f"module_profile_{metric}.png"
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


def plot_heatmap(
    out_dir: Path,
    by_layer_module: pd.DataFrame,
    metric: str = "wmean_cosine",
    figsize: Tuple[int, int] = (12, 8),
) -> Path:
    """Plot layer × module similarity heatmap."""
    # Pivot to matrix form
    df = by_layer_module.copy()
    df["_sort"] = df["layer"].apply(_layer_sort_key)
    df = df.sort_values("_sort")

    pivot = df.pivot(index="layer", columns="module", values=metric)

    # Reorder layers
    layer_order = df.groupby("layer")["_sort"].first().sort_values().index
    pivot = pivot.reindex(layer_order)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn",
        center=0.5,
        vmin=0,
        vmax=1,
        annot=False,
        cbar_kws={"label": metric.replace("_", " ").title()},
    )

    ax.set_xlabel("Module Type")
    ax.set_ylabel("Layer")
    ax.set_title(f"Layer × Module Similarity ({metric})")

    plt.tight_layout()

    out_path = out_dir / f"heatmap_{metric}.png"
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


def plot_distribution_comparison(
    out_dir: Path,
    comparison: CompareResult,
    null_dist: Optional[NullDistribution] = None,
    metric: str = "cosine",
    figsize: Tuple[int, int] = (10, 5),
) -> Path:
    """
    Plot distribution of observed similarities vs null distribution.
    
    Useful for visualizing how the comparison differs from baseline.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Observed distribution
    obs_values = comparison.raw_df[metric].dropna()
    ax.hist(
        obs_values,
        bins=50,
        density=True,
        alpha=0.7,
        label=f"Observed (n={len(obs_values)})",
        color="blue",
    )

    # Null distribution (if provided)
    if null_dist is not None:
        null_values = null_dist.by_layer[f"wmean_{metric}"].dropna()
        ax.hist(
            null_values,
            bins=50,
            density=True,
            alpha=0.5,
            label=f"Null baseline (n={len(null_values)})",
            color="gray",
        )

    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {metric} values")
    ax.legend()

    plt.tight_layout()

    out_path = out_dir / f"distribution_{metric}.png"
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


def write_markdown_report(
    out_dir: Path,
    result: CompareResult,
    model_a: str,
    model_b: str,
    evidence: Optional[EvidenceResult] = None,
) -> Path:
    """Generate comprehensive markdown report."""
    lines = [
        "# Model Lineage Audit Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Input Models",
        "",
        f"- **Model A:** `{model_a}`",
        f"- **Model B:** `{model_b}`",
        "",
        "## Configuration",
        "",
    ]

    for key, value in result.meta.items():
        lines.append(f"- **{key}:** `{value}`")

    lines.extend([
        "",
        "## Coverage",
        "",
    ])

    for key, value in result.coverage.items():
        lines.append(f"- **{key}:** {value:,}")

    # Global summary
    summary = result.global_summary()
    if summary:
        lines.extend([
            "",
            "## Global Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ])
        for key, value in summary.items():
            if isinstance(value, float):
                lines.append(f"| {key} | {value:.6f} |")
            else:
                lines.append(f"| {key} | {value:,} |")

    # Evidence section (if available)
    if evidence is not None:
        lines.extend([
            "",
            "## Evidence Assessment",
            "",
            f"**Verdict:** {evidence.verdict.replace('_', ' ').upper()}",
            "",
            f"**Confidence:** {evidence.confidence:.1%}",
            "",
            "### Global Evidence Scores",
            "",
            "| Score | Value |",
            "|-------|-------|",
        ])
        for key, value in evidence.global_evidence.items():
            if isinstance(value, float):
                lines.append(f"| {key} | {value:.4f} |")
            else:
                lines.append(f"| {key} | {value} |")

    # Top similar layers
    by_layer = result.profiles["by_layer"].copy()
    if "wmean_cosine" in by_layer.columns:
        top_layers = by_layer.nlargest(10, "wmean_cosine")
        lines.extend([
            "",
            "## Top 10 Most Similar Layers",
            "",
            top_layers[["layer", "wmean_cosine", "wmean_rel_l2", "n_tensors", "n_params_total"]]
            .to_markdown(index=False),
        ])

    # Module summary
    by_module = result.profiles["by_module"]
    lines.extend([
        "",
        "## Module Summary",
        "",
        by_module.to_markdown(index=False),
    ])

    # Output files
    lines.extend([
        "",
        "## Output Files",
        "",
        "- `raw_tensor_metrics.csv`: Per-tensor similarity metrics",
        "- `by_layer.csv`: Layer-aggregated metrics",
        "- `by_module.csv`: Module-aggregated metrics",
        "- `by_layer_module.csv`: Layer × module breakdown",
        "- `layer_profile_*.png`: Layer-wise similarity plots",
        "- `module_profile_*.png`: Module-wise similarity plots",
        "- `heatmap_*.png`: Layer × module heatmaps",
        "",
    ])

    # Interpretation guide
    lines.extend([
        "## Interpretation Guide",
        "",
        "### Metrics",
        "",
        "- **Cosine similarity**: Measures angle between weight vectors. 1.0 = identical direction.",
        "- **Relative L2**: Measures relative magnitude of difference. 0.0 = identical.",
        "- **CKA**: Measures structural similarity invariant to scaling/rotation.",
        "- **Spectral**: Compares singular value spectra.",
        "",
        "### What constitutes evidence of derivation?",
        "",
        "1. **Necessary but not sufficient**: High cosine similarity alone doesn't prove derivation.",
        "2. **Compare to null**: Similarity must exceed what's expected from independent training.",
        "3. **Layer patterns matter**: Fine-tuning often shows more divergence in early layers.",
        "4. **Module patterns matter**: Attention may diverge while MLP stays similar (or vice versa).",
        "",
        "### Limitations",
        "",
        "- Weight similarity cannot detect behavioral changes from fine-tuning.",
        "- Models trained on similar data may have high similarity despite being independent.",
        "- Architecture differences (even minor) can cause misleading comparisons.",
        "",
    ])

    # Write file
    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return report_path


def generate_full_report(
    out_dir: str | Path,
    result: CompareResult,
    model_a: str,
    model_b: str,
    null_dist: Optional[NullDistribution] = None,
) -> Dict[str, Path]:
    """
    Generate complete report with all outputs.
    
    Args:
        out_dir: Output directory
        result: Comparison result
        model_a: Path/name of first model
        model_b: Path/name of second model
        null_dist: Optional null distribution for evidence scoring
        
    Returns:
        Dict mapping output type to path
    """
    out_dir = ensure_dir(out_dir)
    outputs: Dict[str, Path] = {}

    # Save CSVs
    csv_paths = save_comparison_tables(out_dir, result)
    outputs["raw_csv"] = csv_paths[0]
    for p in csv_paths[1:]:
        outputs[p.stem] = p

    # Generate plots
    by_layer = result.profiles["by_layer"]
    by_module = result.profiles["by_module"]
    by_layer_module = result.profiles["by_layer_module"]

    title_base = f"{Path(model_a).name} vs {Path(model_b).name}"

    # Layer profiles
    for metric in ["wmean_cosine", "wmean_rel_l2"]:
        if metric in by_layer.columns:
            outputs[f"layer_plot_{metric}"] = plot_layer_profile(
                out_dir, by_layer, f"Layer Profile: {title_base}", metric
            )

    # Module profiles
    for metric in ["wmean_cosine", "wmean_rel_l2"]:
        if metric in by_module.columns:
            outputs[f"module_plot_{metric}"] = plot_module_profile(
                out_dir, by_module, f"Module Profile: {title_base}", metric
            )

    # Heatmaps
    for metric in ["wmean_cosine"]:
        if metric in by_layer_module.columns:
            outputs[f"heatmap_{metric}"] = plot_heatmap(
                out_dir, by_layer_module, metric
            )

    # Distribution plots
    outputs["dist_cosine"] = plot_distribution_comparison(
        out_dir, result, null_dist, "cosine"
    )

    # Evidence scoring (if null distribution provided)
    evidence = None
    if null_dist is not None:
        from .scoring import compute_evidence_scores
        evidence = compute_evidence_scores(result, null_dist)

        # Save evidence report
        evidence_path = out_dir / "EVIDENCE.txt"
        evidence_path.write_text(format_evidence_report(evidence), encoding="utf-8")
        outputs["evidence_report"] = evidence_path

    # Markdown report
    outputs["report"] = write_markdown_report(
        out_dir, result, model_a, model_b, evidence
    )

    logger.info(f"Generated {len(outputs)} output files in {out_dir}")

    return outputs
