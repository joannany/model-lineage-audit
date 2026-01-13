#!/usr/bin/env python3
"""
Example script: Compare two model checkpoints.

Usage:
    python scripts/compare_two_models.py modelA.pt modelB.pt --out results/
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlaudit import load_state, compare_state_dicts, CompareConfig
from mlaudit.report import generate_full_report


def main():
    parser = argparse.ArgumentParser(description="Compare two model checkpoints")
    parser.add_argument("model_a", help="Path to first model")
    parser.add_argument("model_b", help="Path to second model")
    parser.add_argument("--out", "-o", default="./results", help="Output directory")
    parser.add_argument("--grouping", "-g", default="auto", help="Grouping strategy")
    parser.add_argument("--no-cka", action="store_true", help="Skip CKA")
    args = parser.parse_args()

    print(f"Loading {args.model_a}...")
    state_a = load_state(args.model_a)
    print(f"  {state_a.summary()}")

    print(f"Loading {args.model_b}...")
    state_b = load_state(args.model_b)
    print(f"  {state_b.summary()}")

    cfg = CompareConfig(
        grouping=args.grouping,
        compute_cka=not args.no_cka,
    )

    print("Comparing...")
    result = compare_state_dicts(state_a.tensors, state_b.tensors, cfg)

    print(f"Generating report in {args.out}...")
    outputs = generate_full_report(args.out, result, args.model_a, args.model_b)

    print("\nSummary:")
    for k, v in result.global_summary().items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print(f"\nReport: {outputs['report']}")


if __name__ == "__main__":
    main()
