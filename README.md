# Model Lineage Audit (mlaudit)

An analysis toolkit for assessing evidence of model lineage and provenance through weight similarity analysis.

## Why I Built This

I kept seeing the same problem: teams deploying models with no way to trace lineage when something went wrong. If a flaw is found in a training set or a dependency, how do you identify every model that inherited it? In regulated environments, "we're not sure" isn't an answer. This is my attempt at a practical one.

This is one piece of what I call the "Data DNA" layer—the traceability infrastructure that lets you quarantine a fleet of agents the moment you find a flaw in a shared dependency. You can't quarantine what you can't trace.

## Overview

This project explores one aspect of model lineage: what can (and cannot) be inferred from weight-level similarity. It is designed for investigation and reasoning, not as a standalone authority on provenance.

As AI models proliferate, a critical question emerges: how do you assess where a model came from? `mlaudit` approaches this with statistical rigor—not vibes.

It helps assess the question: **"Is Model B likely to be derived from Model A?"**

This is increasingly important for:
- **AI Safety**: Identifying unauthorized or unintended model reuse
- **IP Protection**: Assessing evidence of lineage in disputes
- **Compliance**: Supporting provenance verification in regulated settings
- **Research**: Understanding fine-tuning dynamics and model evolution

## Features

- **Multi-metric comparison**: Cosine similarity, CKA, spectral analysis, relative L2
- **Architecture-aware grouping**: Layer and module-level analysis
- **Null distribution benchmarking**: Statistical baselines from independent models
- **Evidence scoring**: Contextualized inference with percentiles and z-scores
- **Comprehensive reporting**: CSV outputs, plots, and markdown summaries

## Installation

```bash
git clone https://github.com/joannany/model-lineage-audit.git
cd model-lineage-audit
pip install -e .
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

**Accepted checkpoint formats:** `.safetensors`, `.pt`, `.pth`, `.bin` (PyTorch state_dict).

### Compare Two Models

```bash
mlaudit compare \
  --model-a /path/to/modelA.safetensors \
  --model-b /path/to/modelB.safetensors \
  --out ./results/modelA_vs_modelB
```

This generates:
- `raw_tensor_metrics.csv`: Per-tensor similarity scores
- `by_layer.csv`, `by_module.csv`: Aggregated profiles
- `layer_profile_*.png`: Similarity visualizations
- `REPORT.md`: Human-readable evidence summary

### Build a Null Distribution

To contextualize observed similarity, you can build a baseline from unrelated models of the same architecture:

```bash
mlaudit build-null \
  --models model1.pt model2.pt model3.pt model4.pt \
  --out ./null_distributions/llama_7b \
  --architecture llama-7b
```

### Score Against a Null Distribution

```bash
mlaudit score \
  --model-a /path/to/original.pt \
  --model-b /path/to/suspicious.pt \
  --null-dist ./null_distributions/llama_7b
```

Output:

```
============================================================
MODEL LINEAGE EVIDENCE REPORT
============================================================

EVIDENCE SUMMARY: STRONG (relative to baseline)
Confidence: 94.2%

------------------------------------------------------------
GLOBAL EVIDENCE SCORES
------------------------------------------------------------
  weighted_mean_cosine_pct: 99.234
  fraction_above_threshold: 0.912
  combined_z_score: 4.567
...
```

### Inspect a Checkpoint

```bash
mlaudit info --model /path/to/model.safetensors --list-keys
```

## Python API

```python
from mlaudit import load_state, compare_state_dicts, CompareConfig

state_a = load_state("modelA.safetensors")
state_b = load_state("modelB.safetensors")

result = compare_state_dicts(
    state_a.tensors,
    state_b.tensors,
    CompareConfig(grouping="llama", compute_cka=True)
)

print(result.global_summary())
print(result.profiles["by_layer"])
```

### With Evidence Scoring

```python
from mlaudit import (
    load_state, compare_state_dicts, 
    load_null_distribution, compute_evidence_scores
)

result = compare_state_dicts(state_a.tensors, state_b.tensors)
null_dist = load_null_distribution("./null_distributions/llama_7b")

evidence = compute_evidence_scores(result, null_dist)

print(f"Evidence: {evidence.summary}")
print(f"Confidence: {evidence.confidence:.1%}")

for layer in evidence.by_layer:
    print(
        f"{layer.layer}: cosine={layer.cosine_obs:.4f}, "
        f"percentile={layer.cosine_pct:.1f}%"
    )
```

## Metrics Explained

### Cosine Similarity

Measures the angle between flattened weight vectors. Values range from -1 to 1:
- **1.0**: Identical direction (weights proportional)
- **0.0**: Orthogonal
- **-1.0**: Opposite direction

Results are clamped to [-1, 1] to handle floating-point edge cases with near-identical vectors.

### Relative L2 (Symmetric)

Measures the relative magnitude of difference:

```
rel_l2 = ||a - b|| / ((||a|| + ||b||) / 2)
```

- **0.0**: Identical weights
- **Larger**: More different

### CKA (Centered Kernel Alignment)

Measures structural similarity invariant to scaling and rotation. Useful for comparing representations across architectures. Uses unbiased Gram centering (Kornblith et al., 2019).

### Spectral Similarity

Compares singular value spectra, capturing structural properties invariant to orthogonal transformations.

## Interpretation Guide

### What High Similarity Means

High similarity alone does not prove derivation. It must be interpreted relative to a null distribution:

| Percentile | Interpretation |
|------------|----------------|
| < 50% | Typical for independent models |
| 50–90% | Unusual, worth investigating |
| 90–99% | Unlikely by chance, suggests a relationship |
| > 99% | Very strong evidence of a shared origin |

### Layer-wise Patterns

Different reuse patterns leave different fingerprints:

- **Full fine-tuning**: All layers modified, relative similarity preserved
- **LoRA / adapters**: Base layers identical, adapter layers distinct
- **Pruning**: Increased sparsity, similar structure
- **Quantization**: High cosine despite numerical perturbations

### Module-wise Patterns

- **Attention weights** often diverge more during fine-tuning
- **MLP weights** may remain more stable
- **LayerNorm** parameters are often similar even across independent models

## Limitations

1. **No behavioral analysis**: Weight similarity does not capture output behavior
2. **Architecture dependence**: Only same-architecture models are comparable
3. **Training data effects**: Shared data can induce similarity without derivation
4. **Initialization effects**: Shared initialization can mask divergence early
5. **Quantization artifacts**: Require careful handling
6. **Not a complete safety solution**: This addresses provenance, not runtime behavior

## Project Structure

```
model-lineage-audit/
├── src/mlaudit/
│   ├── __init__.py
│   ├── io.py
│   ├── groupings.py
│   ├── metrics.py
│   ├── compare.py
│   ├── nulls.py
│   ├── scoring.py
│   ├── report.py
│   └── cli.py
├── scripts/
├── tests/
├── pyproject.toml
├── LICENSE
└── README.md
```

## Contributing

Contributions are welcome, particularly around:
- Baseline distributions for common architectures
- Additional similarity metrics
- Activation-based analysis
- Integration with model registries

## License

MIT License — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{jo_2026_mlaudit,
  author = {Jo, Anna},
  title  = {Model Lineage Audit},
  year   = {2026},
  url    = {https://github.com/joannany/model-lineage-audit}
}
```

## Acknowledgments

Inspired by:
- [Kornblith et al., Similarity of Neural Network Representations (2019)](https://arxiv.org/abs/1905.00414)
- Model provenance research in AI safety
- The open-source LLM ecosystem
