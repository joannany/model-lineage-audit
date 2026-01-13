#!/usr/bin/env python3
"""
Build a null distribution from a set of independently trained models.

Usage:
    python scripts/build_null_benchmark.py \
        --models model1.pt model2.pt model3.pt \
        --out nulls/my_architecture \
        --architecture llama-7b

The null distribution enables statistical testing of lineage claims by
establishing baseline similarity between unrelated models.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlaudit.compare import CompareConfig
from mlaudit.nulls import build_null_distribution


def main():
    parser = argparse.ArgumentParser(
        description="Build null distribution from independent models"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        required=True,
        help="Paths to model checkpoints (at least 2)"
    )
    parser.add_argument(
        "--out", "-o",
        required=True,
        help="Output directory for null distribution"
    )
    parser.add_argument(
        "--architecture",
        default="unknown",
        help="Architecture name"
    )
    parser.add_argument(
        "--description",
        default="",
        help="Description"
    )
    parser.add_argument(
        "--grouping",
        default="auto",
        help="Grouping strategy"
    )
    parser.add_argument(
        "--no-cka",
        action="store_true",
        help="Skip CKA computation"
    )
    args = parser.parse_args()

    if len(args.models) < 2:
        print("Error: Need at least 2 models")
        sys.exit(1)

    print(f"Building null distribution from {len(args.models)} models...")

    cfg = CompareConfig(
        grouping=args.grouping,
        compute_cka=not args.no_cka,
    )

    null_dist = build_null_distribution(
        args.models,
        cfg=cfg,
        architecture=args.architecture,
        description=args.description,
    )

    print(f"Saving to {args.out}...")
    null_dist.save(args.out)

    print("\nNull Distribution Summary:")
    print(f"  Models: {len(args.models)}")
    print(f"  Pairs: {null_dist.meta.get('n_pairs', 0)}")
    
    if "wmean_cosine" in null_dist.by_layer.columns:
        print(f"  Mean cosine: {null_dist.by_layer['wmean_cosine'].mean():.4f}")
        print(f"  Std cosine: {null_dist.by_layer['wmean_cosine'].std():.4f}")

    print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
