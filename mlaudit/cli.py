"""
Command-line interface for model-lineage-audit.

Commands:
- compare: Compare two models and generate similarity report
- build-null: Build null distribution from a set of models
- score: Score a comparison against a null distribution
- info: Show information about a model checkpoint
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two model checkpoints."""
    from .compare import CompareConfig, compare_state_dicts
    from .io import load_state
    from .nulls import load_null_distribution
    from .report import generate_full_report

    logger.info(f"Loading model A: {args.model_a}")
    state_a = load_state(args.model_a, verbose=True)
    logger.info(f"  {state_a.summary()}")

    logger.info(f"Loading model B: {args.model_b}")
    state_b = load_state(args.model_b, verbose=True)
    logger.info(f"  {state_b.summary()}")

    cfg = CompareConfig(
        grouping=args.grouping,
        exclude_embedding=args.exclude_emb,
        exclude_lm_head=args.exclude_head,
        max_keys=args.max_keys,
        compute_cka=not args.no_cka,
        compute_spectral=not args.no_spectral,
        progress=not args.quiet,
    )

    logger.info("Comparing models...")
    result = compare_state_dicts(state_a.tensors, state_b.tensors, cfg)

    # Load null distribution if provided
    null_dist = None
    if args.null_dist:
        logger.info(f"Loading null distribution: {args.null_dist}")
        null_dist = load_null_distribution(args.null_dist)

    logger.info(f"Generating report in: {args.out}")
    outputs = generate_full_report(
        args.out,
        result,
        args.model_a,
        args.model_b,
        null_dist,
    )

    # Print summary
    summary = result.global_summary()
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Tensors compared: {summary.get('n_tensors_compared', 0):,}")
    print(f"Parameters compared: {summary.get('total_params_compared', 0):,}")
    print(f"Weighted mean cosine: {summary.get('weighted_mean_cosine', 0):.6f}")
    print(f"Weighted mean rel_l2: {summary.get('weighted_mean_rel_l2', 0):.6f}")
    if "weighted_mean_cka" in summary and summary["weighted_mean_cka"] is not None:
        print(f"Weighted mean CKA: {summary['weighted_mean_cka']:.6f}")
    print("=" * 60)
    print(f"\nFull report: {outputs['report']}")

    return 0


def cmd_build_null(args: argparse.Namespace) -> int:
    """Build null distribution from multiple models."""
    from .compare import CompareConfig
    from .nulls import build_null_distribution

    model_paths = args.models

    if len(model_paths) < 2:
        logger.error("Need at least 2 models to build null distribution")
        return 1

    logger.info(f"Building null distribution from {len(model_paths)} models")

    cfg = CompareConfig(
        grouping=args.grouping,
        exclude_embedding=args.exclude_emb,
        exclude_lm_head=args.exclude_head,
        compute_cka=not args.no_cka,
        compute_spectral=not args.no_spectral,
        progress=not args.quiet,
    )

    null_dist = build_null_distribution(
        model_paths,
        cfg=cfg,
        architecture=args.architecture,
        description=args.description or "",
    )

    logger.info(f"Saving null distribution to: {args.out}")
    null_dist.save(args.out)

    # Print summary
    print("\n" + "=" * 60)
    print("NULL DISTRIBUTION SUMMARY")
    print("=" * 60)
    print(f"Models: {null_dist.meta['n_models']}")
    print(f"Pairs: {null_dist.meta['n_pairs']}")
    print(f"Architecture: {null_dist.meta['architecture']}")

    # Show distribution statistics
    by_layer = null_dist.by_layer
    if "wmean_cosine" in by_layer.columns:
        cos_mean = by_layer["wmean_cosine"].mean()
        cos_std = by_layer["wmean_cosine"].std()
        print(f"Mean cosine (across layers): {cos_mean:.4f} ± {cos_std:.4f}")

    print("=" * 60)
    print(f"\nNull distribution saved to: {args.out}")

    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Score a comparison against a null distribution."""
    from .compare import CompareConfig, compare_state_dicts
    from .io import load_state
    from .nulls import load_null_distribution
    from .scoring import compute_evidence_scores, format_evidence_report

    logger.info(f"Loading model A: {args.model_a}")
    state_a = load_state(args.model_a)

    logger.info(f"Loading model B: {args.model_b}")
    state_b = load_state(args.model_b)

    logger.info(f"Loading null distribution: {args.null_dist}")
    null_dist = load_null_distribution(args.null_dist)

    cfg = CompareConfig(
        grouping=args.grouping,
        exclude_embedding=args.exclude_emb,
        exclude_lm_head=args.exclude_head,
        compute_cka=True,
        compute_spectral=True,
        progress=not args.quiet,
    )

    logger.info("Comparing models...")
    result = compare_state_dicts(state_a.tensors, state_b.tensors, cfg)

    logger.info("Computing evidence scores...")
    evidence = compute_evidence_scores(result, null_dist)

    # Print report
    print(format_evidence_report(evidence))

    # Save if output specified
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(format_evidence_report(evidence), encoding="utf-8")
        logger.info(f"Evidence report saved to: {args.out}")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show information about a model checkpoint."""
    from .io import load_state
    from .groupings import GROUPINGS, auto_detect_grouping, group_keys

    logger.info(f"Loading: {args.model}")
    state = load_state(args.model, verbose=True)

    print("\n" + "=" * 60)
    print("MODEL CHECKPOINT INFO")
    print("=" * 60)
    print(f"Path: {state.path}")
    print(f"Format: {state.format}")
    print(f"Total tensors: {len(state.tensors):,}")
    print(f"Total parameters: {state.total_params:,}")
    print(f"Skipped keys: {len(state.skipped_keys)}")

    # Auto-detect grouping
    keys = list(state.tensors.keys())
    detected = auto_detect_grouping(keys)
    print(f"\nDetected grouping: {detected}")

    # Show key structure
    grouped = group_keys(keys, detected)
    print(f"\nLayers found: {len(grouped)}")
    print("\nLayer breakdown:")

    for layer in sorted(grouped.keys(), key=lambda x: (0 if x.startswith("layer_") and x[6:].isdigit() else 1, x)):
        modules = grouped[layer]
        n_keys = sum(len(v) for v in modules.values())
        n_params = sum(state.tensors[k].numel() for m in modules.values() for k in m)
        module_names = ", ".join(sorted(modules.keys()))
        print(f"  {layer}: {n_keys} tensors, {n_params:,} params ({module_names})")

    if args.list_keys:
        print("\n" + "-" * 60)
        print("ALL TENSOR KEYS")
        print("-" * 60)
        for key in sorted(keys):
            shape = tuple(state.tensors[key].shape)
            print(f"  {key}: {shape}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        prog="mlaudit",
        description="Model Lineage Audit - forensic toolkit for model similarity analysis",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.2.0",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two model checkpoints and generate similarity report",
    )
    compare_parser.add_argument(
        "--model-a", "-a",
        required=True,
        help="Path to first model checkpoint",
    )
    compare_parser.add_argument(
        "--model-b", "-b",
        required=True,
        help="Path to second model checkpoint",
    )
    compare_parser.add_argument(
        "--out", "-o",
        required=True,
        help="Output directory for report",
    )
    compare_parser.add_argument(
        "--null-dist", "-n",
        help="Path to null distribution for evidence scoring",
    )
    compare_parser.add_argument(
        "--grouping", "-g",
        default="auto",
        choices=["auto", "default", "llama", "gpt2", "bert", "moe"],
        help="Key grouping strategy (default: auto-detect)",
    )
    compare_parser.add_argument(
        "--exclude-emb",
        action="store_true",
        help="Exclude embedding layers",
    )
    compare_parser.add_argument(
        "--exclude-head",
        action="store_true",
        help="Exclude LM head layers",
    )
    compare_parser.add_argument(
        "--no-cka",
        action="store_true",
        help="Skip CKA computation (faster)",
    )
    compare_parser.add_argument(
        "--no-spectral",
        action="store_true",
        help="Skip spectral similarity computation",
    )
    compare_parser.add_argument(
        "--max-keys",
        type=int,
        help="Limit number of keys for quick tests",
    )
    compare_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress bars",
    )
    compare_parser.set_defaults(func=cmd_compare)

    # Build-null command
    null_parser = subparsers.add_parser(
        "build-null",
        help="Build null distribution from multiple independent models",
    )
    null_parser.add_argument(
        "--models", "-m",
        nargs="+",
        required=True,
        help="Paths to model checkpoints (at least 2)",
    )
    null_parser.add_argument(
        "--out", "-o",
        required=True,
        help="Output directory for null distribution",
    )
    null_parser.add_argument(
        "--architecture",
        default="unknown",
        help="Architecture name for metadata",
    )
    null_parser.add_argument(
        "--description",
        help="Description for metadata",
    )
    null_parser.add_argument(
        "--grouping", "-g",
        default="auto",
        choices=["auto", "default", "llama", "gpt2", "bert", "moe"],
        help="Key grouping strategy",
    )
    null_parser.add_argument(
        "--exclude-emb",
        action="store_true",
        help="Exclude embedding layers",
    )
    null_parser.add_argument(
        "--exclude-head",
        action="store_true",
        help="Exclude LM head layers",
    )
    null_parser.add_argument(
        "--no-cka",
        action="store_true",
        help="Skip CKA computation",
    )
    null_parser.add_argument(
        "--no-spectral",
        action="store_true",
        help="Skip spectral similarity computation",
    )
    null_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress bars",
    )
    null_parser.set_defaults(func=cmd_build_null)

    # Score command
    score_parser = subparsers.add_parser(
        "score",
        help="Score a model comparison against a null distribution",
    )
    score_parser.add_argument(
        "--model-a", "-a",
        required=True,
        help="Path to first model checkpoint",
    )
    score_parser.add_argument(
        "--model-b", "-b",
        required=True,
        help="Path to second model checkpoint",
    )
    score_parser.add_argument(
        "--null-dist", "-n",
        required=True,
        help="Path to null distribution",
    )
    score_parser.add_argument(
        "--out", "-o",
        help="Output path for evidence report",
    )
    score_parser.add_argument(
        "--grouping", "-g",
        default="auto",
        choices=["auto", "default", "llama", "gpt2", "bert", "moe"],
        help="Key grouping strategy",
    )
    score_parser.add_argument(
        "--exclude-emb",
        action="store_true",
        help="Exclude embedding layers",
    )
    score_parser.add_argument(
        "--exclude-head",
        action="store_true",
        help="Exclude LM head layers",
    )
    score_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress bars",
    )
    score_parser.set_defaults(func=cmd_score)

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about a model checkpoint",
    )
    info_parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to model checkpoint",
    )
    info_parser.add_argument(
        "--list-keys",
        action="store_true",
        help="List all tensor keys",
    )
    info_parser.set_defaults(func=cmd_info)

    return parser


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        rc = args.func(args)
        sys.exit(rc)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}")
        if "--debug" in sys.argv:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
