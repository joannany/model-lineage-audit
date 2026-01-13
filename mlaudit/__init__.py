"""
Model Lineage Audit (mlaudit)
=============================

A forensic toolkit for detecting model lineage and provenance through
weight similarity analysis, CKA comparison, and statistical benchmarking.

Main components:
- io: Model checkpoint loading (PyTorch, safetensors)
- metrics: Similarity metrics (cosine, CKA, spectral)
- groupings: Architecture-aware tensor grouping
- compare: Pairwise model comparison
- nulls: Null distribution generation for statistical testing
- scoring: Z-score and evidence computation
- report: Report generation (CSV, plots, markdown)
- cli: Command-line interface
"""

__version__ = "0.2.0"
__author__ = "Model Lineage Audit Contributors"

from .io import load_state, LoadedState
from .metrics import cosine_sim, rel_l2_symmetric, linear_cka, spectral_similarity
from .groupings import GROUPINGS, GroupingResult
from .compare import compare_state_dicts, aggregate_profiles, CompareConfig, CompareResult
from .nulls import NullDistribution, build_null_distribution, load_null_distribution
from .scoring import compute_evidence_scores, EvidenceResult

__all__ = [
    # Version
    "__version__",
    # IO
    "load_state",
    "LoadedState",
    # Metrics
    "cosine_sim",
    "rel_l2_symmetric",
    "linear_cka",
    "spectral_similarity",
    # Groupings
    "GROUPINGS",
    "GroupingResult",
    # Compare
    "compare_state_dicts",
    "aggregate_profiles",
    "CompareConfig",
    "CompareResult",
    # Nulls
    "NullDistribution",
    "build_null_distribution",
    "load_null_distribution",
    # Scoring
    "compute_evidence_scores",
    "EvidenceResult",
]
