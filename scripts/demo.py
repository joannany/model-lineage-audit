#!/usr/bin/env python3
"""
Demo script showing mlaudit capabilities with synthetic models.

This creates synthetic "models" to demonstrate the comparison workflow
without needing actual LLM checkpoints.
"""

import tempfile
from pathlib import Path

import torch
import numpy as np

from mlaudit import (
    load_state,
    compare_state_dicts,
    CompareConfig,
    CompareResult,
)
from mlaudit.report import generate_full_report


def create_synthetic_llama_weights(
    n_layers: int = 4,
    hidden_dim: int = 256,
    intermediate_dim: int = 512,
    vocab_size: int = 1000,
    seed: int = 42,
) -> dict:
    """Create synthetic Llama-style weights."""
    torch.manual_seed(seed)
    sd = {}
    
    # Embedding
    sd["model.embed_tokens.weight"] = torch.randn(vocab_size, hidden_dim)
    
    # Transformer layers
    for i in range(n_layers):
        prefix = f"model.layers.{i}"
        sd[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(hidden_dim, hidden_dim)
        sd[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(hidden_dim, hidden_dim)
        sd[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(hidden_dim, hidden_dim)
        sd[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(hidden_dim, hidden_dim)
        sd[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(intermediate_dim, hidden_dim)
        sd[f"{prefix}.mlp.up_proj.weight"] = torch.randn(intermediate_dim, hidden_dim)
        sd[f"{prefix}.mlp.down_proj.weight"] = torch.randn(hidden_dim, intermediate_dim)
        sd[f"{prefix}.input_layernorm.weight"] = torch.randn(hidden_dim)
        sd[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(hidden_dim)
    
    # Final norm and head
    sd["model.norm.weight"] = torch.randn(hidden_dim)
    sd["lm_head.weight"] = torch.randn(vocab_size, hidden_dim)
    
    return sd


def add_noise(sd: dict, noise_scale: float = 0.01) -> dict:
    """Add Gaussian noise to all tensors."""
    return {k: v + torch.randn_like(v) * noise_scale for k, v in sd.items()}


def fine_tune_simulation(sd: dict, affected_layers: list, change_scale: float = 0.1) -> dict:
    """Simulate fine-tuning by modifying specific layers more."""
    result = {}
    for k, v in sd.items():
        # Check if this key belongs to an affected layer
        is_affected = any(f"layers.{layer}" in k for layer in affected_layers)
        if is_affected:
            result[k] = v + torch.randn_like(v) * change_scale
        else:
            result[k] = v + torch.randn_like(v) * 0.001  # Minor noise
    return result


def main():
    print("=" * 70)
    print("Model Lineage Audit - Demo")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Demo 1: Identical models
    # -------------------------------------------------------------------------
    print("\n[Demo 1] Comparing model to itself...")
    
    base_model = create_synthetic_llama_weights(n_layers=8, seed=42)
    
    result = compare_state_dicts(
        base_model,
        base_model,
        CompareConfig(grouping="llama", compute_cka=False, progress=False),
    )
    
    summary = result.global_summary()
    print(f"  Weighted mean cosine: {summary['weighted_mean_cosine']:.6f} (expected: 1.0)")
    print(f"  Weighted mean rel_l2: {summary['weighted_mean_rel_l2']:.6f} (expected: 0.0)")
    
    # -------------------------------------------------------------------------
    # Demo 2: Slightly noisy model (small perturbation)
    # -------------------------------------------------------------------------
    print("\n[Demo 2] Comparing to model with small noise (0.1% scale)...")
    
    noisy_model = add_noise(base_model, noise_scale=0.001)
    
    result = compare_state_dicts(
        base_model,
        noisy_model,
        CompareConfig(grouping="llama", compute_cka=False, progress=False),
    )
    
    summary = result.global_summary()
    print(f"  Weighted mean cosine: {summary['weighted_mean_cosine']:.6f} (expected: ~0.9999)")
    print(f"  Weighted mean rel_l2: {summary['weighted_mean_rel_l2']:.6f}")
    
    # -------------------------------------------------------------------------
    # Demo 3: Fine-tuned model (specific layers modified)
    # -------------------------------------------------------------------------
    print("\n[Demo 3] Comparing to 'fine-tuned' model (layers 4-7 modified)...")
    
    finetuned_model = fine_tune_simulation(base_model, affected_layers=[4, 5, 6, 7], change_scale=0.1)
    
    result = compare_state_dicts(
        base_model,
        finetuned_model,
        CompareConfig(grouping="llama", compute_cka=False, progress=False),
    )
    
    summary = result.global_summary()
    print(f"  Weighted mean cosine: {summary['weighted_mean_cosine']:.6f}")
    print(f"  Weighted mean rel_l2: {summary['weighted_mean_rel_l2']:.6f}")
    
    print("\n  Layer-wise breakdown:")
    by_layer = result.profiles["by_layer"]
    for _, row in by_layer.iterrows():
        print(f"    {row['layer']:<15}: cosine={row['wmean_cosine']:.4f}, rel_l2={row['wmean_rel_l2']:.4f}")
    
    # -------------------------------------------------------------------------
    # Demo 4: Independent models (different seeds)
    # -------------------------------------------------------------------------
    print("\n[Demo 4] Comparing independently trained models (different seeds)...")
    
    independent_model = create_synthetic_llama_weights(n_layers=8, seed=999)
    
    result = compare_state_dicts(
        base_model,
        independent_model,
        CompareConfig(grouping="llama", compute_cka=False, progress=False),
    )
    
    summary = result.global_summary()
    print(f"  Weighted mean cosine: {summary['weighted_mean_cosine']:.6f} (expected: ~0)")
    print(f"  Weighted mean rel_l2: {summary['weighted_mean_rel_l2']:.6f}")
    
    # -------------------------------------------------------------------------
    # Demo 5: Full report generation
    # -------------------------------------------------------------------------
    print("\n[Demo 5] Generating full report...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save models
        model_a_path = Path(tmpdir) / "base_model.pt"
        model_b_path = Path(tmpdir) / "finetuned_model.pt"
        torch.save(base_model, model_a_path)
        torch.save(finetuned_model, model_b_path)
        
        # Load and compare
        state_a = load_state(model_a_path)
        state_b = load_state(model_b_path)
        
        result = compare_state_dicts(
            state_a.tensors,
            state_b.tensors,
            CompareConfig(grouping="llama", progress=False),
        )
        
        # Generate report
        report_dir = output_dir / "finetuning_comparison"
        outputs = generate_full_report(
            report_dir,
            result,
            str(model_a_path),
            str(model_b_path),
        )
        
        print(f"  Report generated at: {report_dir}")
        print(f"  Files created:")
        for name, path in outputs.items():
            print(f"    - {name}: {path.name}")
    
    # -------------------------------------------------------------------------
    # Demo 6: Metrics deep dive
    # -------------------------------------------------------------------------
    print("\n[Demo 6] Metrics deep dive on single tensor...")
    
    from mlaudit.metrics import compute_all_metrics
    
    # Compare a single attention weight
    key = "model.layers.0.self_attn.q_proj.weight"
    metrics = compute_all_metrics(
        base_model[key],
        finetuned_model[key],
        compute_cka=True,
        compute_spectral=True,
    )
    
    print(f"  Tensor: {key}")
    print(f"  Shape: {tuple(base_model[key].shape)}")
    print(f"  Cosine similarity: {metrics['cosine']:.6f}")
    print(f"  Relative L2: {metrics['rel_l2']:.6f}")
    if metrics['cka'] is not None:
        print(f"  CKA: {metrics['cka']:.6f}")
    if metrics['spectral'] is not None:
        print(f"  Spectral similarity: {metrics['spectral']:.6f}")
    
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print(f"\nCheck {output_dir} for generated reports.")


if __name__ == "__main__":
    main()
