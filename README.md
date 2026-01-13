# Model Lineage Audit (mlaudit)

A forensic toolkit for detecting model lineage and provenance through weight similarity analysis.

## Overview

`mlaudit` helps answer the question: **"Is Model B derived from Model A?"**

This is increasingly important for:
- **AI Safety**: Detecting unauthorized model derivatives
- **IP Protection**: Proving model provenance in disputes
- **Compliance**: Verifying model origins for regulatory purposes
- **Research**: Understanding fine-tuning dynamics and model evolution

## Features

- **Multi-metric comparison**: Cosine similarity, CKA, spectral analysis, relative L2
- **Architecture-aware grouping**: Layer and module-level analysis
- **Null distribution benchmarking**: Statistical baselines from independent models
- **Evidence scoring**: Principled inference with percentiles and z-scores
- **Comprehensive reporting**: CSV, plots, markdown reports

## Installation

```bash
git clone https://github.com/joannany/model-lineage-audit.git
cd model-lineage-audit
pip install -e .
```

Or install with dev dependencies:
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
- `REPORT.md`: Human-readable summary

### Build a Null Distribution

To make statistical claims about lineage, you need a baseline of what similarity looks like between **unrelated** models:

```bash
mlaudit build-null \
  --models model1.pt model2.pt model3.pt model4.pt \
  --out ./null_distributions/llama_7b \
  --architecture llama-7b
```

### Score Against Null Distribution

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

VERDICT: STRONG EVIDENCE
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

# Load models
state_a = load_state("modelA.safetensors")
state_b = load_state("modelB.safetensors")

# Compare
result = compare_state_dicts(
    state_a.tensors,
    state_b.tensors,
    CompareConfig(grouping="llama", compute_cka=True)
)

# Get summary
print(result.global_summary())
# {'weighted_mean_cosine': 0.9847, 'weighted_mean_rel_l2': 0.0312, ...}

# Access layer profiles
print(result.profiles["by_layer"])
```

### With Evidence Scoring

```python
from mlaudit import (
    load_state, compare_state_dicts, 
    load_null_distribution, compute_evidence_scores
)

# Compare models
result = compare_state_dicts(state_a.tensors, state_b.tensors)

# Load null distribution
null_dist = load_null_distribution("./null_distributions/llama_7b")

# Compute evidence
evidence = compute_evidence_scores(result, null_dist)

print(f"Verdict: {evidence.verdict}")
print(f"Confidence: {evidence.confidence:.1%}")

# Per-layer evidence
for layer in evidence.by_layer:
    print(f"{layer.layer}: cosine={layer.cosine_obs:.4f}, "
          f"percentile={layer.cosine_pct:.1f}%")
```

## Metrics Explained

### Cosine Similarity
Measures the angle between flattened weight vectors. Values range from -1 to 1:
- **1.0**: Identical direction (weights proportional)
- **0.0**: Orthogonal
- **-1.0**: Opposite direction

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
Compares singular value spectra. Captures structural properties that are invariant to orthogonal transformations.

## Interpretation Guide

### What High Similarity Means

High similarity alone **does not prove derivation**. You must compare against a null distribution:

| Percentile | Interpretation |
|------------|----------------|
| < 50% | Normal range for independent models |
| 50-90% | Somewhat unusual, worth investigating |
| 90-99% | Unlikely by chance, suggests relationship |
| > 99% | Very strong evidence of derivation |

### Layer-wise Patterns

Different derivation methods leave different fingerprints:

- **Full fine-tuning**: All layers modified, but relative similarity preserved
- **LoRA/adapters**: Base layers identical, adapter layers different
- **Pruning**: Increased sparsity, similar structure
- **Quantization**: Systematic perturbation, high cosine despite numerical differences

### Module-wise Patterns

- **Attention weights** often diverge more during fine-tuning
- **MLP weights** may stay more similar
- **LayerNorm** typically has high similarity even between independent models

## Limitations

1. **No behavioral analysis**: Weight similarity doesn't capture output changes
2. **Architecture dependence**: Can only compare same-architecture models
3. **Training data effects**: Models trained on similar data may be similar without derivation
4. **Initialization effects**: Same initialization + different training can look similar early
5. **Quantization artifacts**: Quantized models need careful handling

## Project Structure

```
model-lineage-audit/
├── src/mlaudit/
│   ├── __init__.py      # Package exports
│   ├── io.py            # Checkpoint loading
│   ├── groupings.py     # Architecture-aware key grouping
│   ├── metrics.py       # Similarity metrics (cosine, CKA, etc.)
│   ├── compare.py       # Pairwise comparison
│   ├── nulls.py         # Null distribution building
│   ├── scoring.py       # Evidence scoring
│   ├── report.py        # Report generation
│   └── cli.py           # Command-line interface
├── scripts/             # Example scripts
├── tests/               # Test suite
├── pyproject.toml       # Package configuration
├── LICENSE              # MIT License
└── README.md
```

## Contributing

Contributions welcome! Areas of interest:
- Pre-built null distributions for common architectures
- Additional metrics (e.g., representation similarity)
- Activation-based comparison (beyond weight analysis)
- Integration with model registries

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this tool in research, please cite:

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
- [CKA paper](https://arxiv.org/abs/1905.00414) (Kornblith et al., 2019)
- Model provenance research in AI safety
- Open-source LLM ecosystem
