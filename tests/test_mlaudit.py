"""
Test suite for model-lineage-audit.

Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
import tempfile
import json

from mlaudit.io import load_state, get_common_keys, LoadedState, TensorDict
from mlaudit.metrics import (
    cosine_sim,
    rel_l2_symmetric,
    linear_cka,
    spectral_similarity,
    effective_rank,
    compute_all_metrics,
)
from mlaudit.groupings import (
    default_grouping,
    llama_grouping,
    gpt2_grouping,
    GroupingResult,
    auto_detect_grouping,
    group_keys,
)
from mlaudit.compare import compare_state_dicts, CompareConfig, aggregate_profiles
from mlaudit.nulls import NullDistribution, build_null_distribution
from mlaudit.scoring import compute_evidence_scores, EvidenceResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_tensors():
    """Create simple test tensors."""
    torch.manual_seed(42)
    return {
        "a": torch.randn(100, 100),
        "b": torch.randn(100, 100),
        "identical_a": torch.randn(50, 50),
        "identical_b": None,  # Will be set to copy of identical_a
        "scaled": None,  # Will be scaled version
    }

@pytest.fixture
def simple_tensors_filled(simple_tensors):
    """Fill in derived tensors."""
    simple_tensors["identical_b"] = simple_tensors["identical_a"].clone()
    simple_tensors["scaled"] = simple_tensors["identical_a"] * 2.5
    return simple_tensors


@pytest.fixture
def mock_state_dict_llama():
    """Create a mock Llama-style state dict."""
    torch.manual_seed(123)
    sd = {}
    
    # Embedding
    sd["model.embed_tokens.weight"] = torch.randn(32000, 4096)
    
    # 4 transformer layers
    for i in range(4):
        prefix = f"model.layers.{i}"
        sd[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(4096, 4096)
        sd[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(4096, 4096)
        sd[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(4096, 4096)
        sd[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(4096, 4096)
        sd[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(11008, 4096)
        sd[f"{prefix}.mlp.up_proj.weight"] = torch.randn(11008, 4096)
        sd[f"{prefix}.mlp.down_proj.weight"] = torch.randn(4096, 11008)
        sd[f"{prefix}.input_layernorm.weight"] = torch.randn(4096)
        sd[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(4096)
    
    # Final norm and head
    sd["model.norm.weight"] = torch.randn(4096)
    sd["lm_head.weight"] = torch.randn(32000, 4096)
    
    return sd


@pytest.fixture
def mock_state_dict_gpt2():
    """Create a mock GPT-2 style state dict."""
    torch.manual_seed(456)
    sd = {}
    
    # Embeddings
    sd["transformer.wte.weight"] = torch.randn(50257, 768)
    sd["transformer.wpe.weight"] = torch.randn(1024, 768)
    
    # 4 transformer blocks
    for i in range(4):
        prefix = f"transformer.h.{i}"
        sd[f"{prefix}.ln_1.weight"] = torch.randn(768)
        sd[f"{prefix}.ln_1.bias"] = torch.randn(768)
        sd[f"{prefix}.attn.c_attn.weight"] = torch.randn(768, 2304)
        sd[f"{prefix}.attn.c_proj.weight"] = torch.randn(768, 768)
        sd[f"{prefix}.ln_2.weight"] = torch.randn(768)
        sd[f"{prefix}.ln_2.bias"] = torch.randn(768)
        sd[f"{prefix}.mlp.c_fc.weight"] = torch.randn(768, 3072)
        sd[f"{prefix}.mlp.c_proj.weight"] = torch.randn(3072, 768)
    
    # Final layer norm
    sd["transformer.ln_f.weight"] = torch.randn(768)
    sd["transformer.ln_f.bias"] = torch.randn(768)
    
    return sd


# ============================================================================
# Test: Metrics
# ============================================================================

class TestCosineSimMetric:
    def test_identical_tensors(self, simple_tensors_filled):
        a = simple_tensors_filled["identical_a"]
        b = simple_tensors_filled["identical_b"]
        sim = cosine_sim(a, b)
        assert sim == pytest.approx(1.0, abs=1e-6)
    
    def test_scaled_tensors(self, simple_tensors_filled):
        """Cosine should be 1.0 for scaled versions."""
        a = simple_tensors_filled["identical_a"]
        scaled = simple_tensors_filled["scaled"]
        sim = cosine_sim(a, scaled)
        assert sim == pytest.approx(1.0, abs=1e-6)
    
    def test_orthogonal_tensors(self):
        """Orthogonal tensors should have cosine ~0."""
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        sim = cosine_sim(a, b)
        assert sim == pytest.approx(0.0, abs=1e-6)
    
    def test_opposite_tensors(self):
        """Opposite tensors should have cosine -1."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = -a
        sim = cosine_sim(a, b)
        assert sim == pytest.approx(-1.0, abs=1e-6)
    
    def test_different_shapes_via_flatten(self):
        """Should work with tensors of different shapes but same numel."""
        a = torch.randn(10, 10)
        b = a.reshape(100)
        sim = cosine_sim(a, b)
        assert sim == pytest.approx(1.0, abs=1e-6)


class TestRelL2Metric:
    def test_identical_tensors(self, simple_tensors_filled):
        a = simple_tensors_filled["identical_a"]
        b = simple_tensors_filled["identical_b"]
        dist = rel_l2_symmetric(a, b)
        assert dist == pytest.approx(0.0, abs=1e-6)
    
    def test_symmetry(self, simple_tensors_filled):
        """rel_l2_symmetric(a, b) should equal rel_l2_symmetric(b, a)."""
        a = simple_tensors_filled["a"]
        b = simple_tensors_filled["b"]
        dist_ab = rel_l2_symmetric(a, b)
        dist_ba = rel_l2_symmetric(b, a)
        assert dist_ab == pytest.approx(dist_ba, abs=1e-6)
    
    def test_scaled_tensors_not_zero(self, simple_tensors_filled):
        """Scaled tensors should have non-zero rel_l2."""
        a = simple_tensors_filled["identical_a"]
        scaled = simple_tensors_filled["scaled"]
        dist = rel_l2_symmetric(a, scaled)
        assert dist > 0.1  # Should be notable difference


class TestCKAMetric:
    def test_identical_tensors(self):
        torch.manual_seed(42)
        a = torch.randn(32, 64)
        b = a.clone()
        cka = linear_cka(a, b)
        assert cka == pytest.approx(1.0, abs=1e-4)
    
    def test_random_tensors_low_cka(self):
        """Random tensors should have CKA closer to 0."""
        torch.manual_seed(42)
        a = torch.randn(32, 64)
        torch.manual_seed(999)
        b = torch.randn(32, 64)
        cka = linear_cka(a, b)
        assert cka < 0.5  # Should be relatively low


class TestSpectralSimilarity:
    def test_identical_tensors(self):
        torch.manual_seed(42)
        a = torch.randn(32, 64)
        b = a.clone()
        sim = spectral_similarity(a, b)
        assert sim == pytest.approx(1.0, abs=1e-4)
    
    def test_scaled_tensors(self):
        """Spectral similarity should be 1.0 for scaled tensors."""
        torch.manual_seed(42)
        a = torch.randn(32, 64)
        b = a * 3.0
        sim = spectral_similarity(a, b)
        assert sim == pytest.approx(1.0, abs=1e-4)


class TestEffectiveRank:
    def test_identity_matrix(self):
        """Identity matrix should have effective rank close to dimension."""
        a = torch.eye(10)
        rank = effective_rank(a)
        assert rank == pytest.approx(10.0, abs=0.5)
    
    def test_rank_one_matrix(self):
        """Rank-1 matrix should have effective rank ~1."""
        a = torch.randn(10, 1) @ torch.randn(1, 10)
        rank = effective_rank(a)
        assert rank < 2.0


# ============================================================================
# Test: Groupings
# ============================================================================

class TestGroupings:
    def test_llama_embedding(self):
        result = llama_grouping("model.embed_tokens.weight")
        assert result.layer == "layer_embed"
        assert result.module == "embedding"
    
    def test_llama_attention(self):
        result = llama_grouping("model.layers.5.self_attn.q_proj.weight")
        assert result.layer == "layer_5"
        assert result.module == "attention"
    
    def test_llama_mlp(self):
        result = llama_grouping("model.layers.10.mlp.gate_proj.weight")
        assert result.layer == "layer_10"
        assert result.module == "mlp"
    
    def test_llama_layernorm(self):
        result = llama_grouping("model.layers.3.input_layernorm.weight")
        assert result.layer == "layer_3"
        assert result.module == "layernorm"
    
    def test_gpt2_embedding(self):
        result = gpt2_grouping("transformer.wte.weight")
        assert result.layer == "layer_embed"
        assert result.module == "embedding"
    
    def test_gpt2_attention(self):
        result = gpt2_grouping("transformer.h.7.attn.c_attn.weight")
        assert result.layer == "layer_7"
        assert result.module == "attention"
    
    def test_auto_detect_llama(self, mock_state_dict_llama):
        keys = list(mock_state_dict_llama.keys())
        detected = auto_detect_grouping(keys)
        assert detected == "llama"
    
    def test_auto_detect_gpt2(self, mock_state_dict_gpt2):
        keys = list(mock_state_dict_gpt2.keys())
        detected = auto_detect_grouping(keys)
        assert detected == "gpt2"
    
    def test_group_keys(self, mock_state_dict_llama):
        keys = list(mock_state_dict_llama.keys())
        grouped = group_keys(keys, "llama")
        
        # Should have layer_embed, layer_0 through layer_3, layer_final
        assert "layer_embed" in grouped
        assert "layer_0" in grouped
        assert "layer_3" in grouped
        assert "layer_final" in grouped


# ============================================================================
# Test: Compare
# ============================================================================

class TestCompare:
    def test_identical_models(self, mock_state_dict_llama):
        """Comparing model to itself should give cosine=1.0."""
        result = compare_state_dicts(
            mock_state_dict_llama,
            mock_state_dict_llama,
            CompareConfig(grouping="llama", compute_cka=False, compute_spectral=False, progress=False)
        )
        
        summary = result.global_summary()
        assert summary["weighted_mean_cosine"] == pytest.approx(1.0, abs=1e-6)
        assert summary["weighted_mean_rel_l2"] == pytest.approx(0.0, abs=1e-6)
    
    def test_modified_model(self, mock_state_dict_llama):
        """Slightly modified model should have high but <1 cosine."""
        sd_modified = {k: v + torch.randn_like(v) * 0.01 for k, v in mock_state_dict_llama.items()}
        
        result = compare_state_dicts(
            mock_state_dict_llama,
            sd_modified,
            CompareConfig(grouping="llama", compute_cka=False, compute_spectral=False, progress=False)
        )
        
        summary = result.global_summary()
        assert 0.9 < summary["weighted_mean_cosine"] < 1.0
        assert summary["weighted_mean_rel_l2"] > 0
    
    def test_profiles_structure(self, mock_state_dict_llama):
        """Test that profiles have expected structure."""
        result = compare_state_dicts(
            mock_state_dict_llama,
            mock_state_dict_llama,
            CompareConfig(grouping="llama", compute_cka=False, progress=False)
        )
        
        assert "by_layer" in result.profiles
        assert "by_module" in result.profiles
        assert "by_layer_module" in result.profiles
        
        by_layer = result.profiles["by_layer"]
        assert "layer" in by_layer.columns
        assert "wmean_cosine" in by_layer.columns
        assert "n_tensors" in by_layer.columns
    
    def test_exclude_embedding(self, mock_state_dict_llama):
        """Test exclusion of embedding layers."""
        result_with = compare_state_dicts(
            mock_state_dict_llama,
            mock_state_dict_llama,
            CompareConfig(grouping="llama", exclude_embedding=False, progress=False)
        )
        
        result_without = compare_state_dicts(
            mock_state_dict_llama,
            mock_state_dict_llama,
            CompareConfig(grouping="llama", exclude_embedding=True, progress=False)
        )
        
        # Should have fewer tensors when excluding embedding
        assert len(result_without.raw_df) < len(result_with.raw_df)


# ============================================================================
# Test: IO
# ============================================================================

class TestIO:
    def test_save_and_load_torch(self, mock_state_dict_llama):
        """Test saving and loading PyTorch checkpoint."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(mock_state_dict_llama, f.name)
            state = load_state(f.name)
            
            assert len(state.tensors) == len(mock_state_dict_llama)
            assert state.format == "torch"
            
            # Check a specific tensor
            key = "model.layers.0.self_attn.q_proj.weight"
            assert torch.allclose(state.tensors[key], mock_state_dict_llama[key])
    
    def test_save_and_load_wrapped(self, mock_state_dict_llama):
        """Test loading checkpoint with wrapper dict."""
        wrapped = {"state_dict": mock_state_dict_llama, "epoch": 10}
        
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(wrapped, f.name)
            state = load_state(f.name)
            
            assert len(state.tensors) == len(mock_state_dict_llama)
    
    def test_get_common_keys(self, mock_state_dict_llama):
        """Test finding common keys between state dicts."""
        sd_subset = {k: v for i, (k, v) in enumerate(mock_state_dict_llama.items()) if i < 10}
        
        common, a_only, b_only = get_common_keys(mock_state_dict_llama, sd_subset)
        
        assert len(common) == 10
        assert len(b_only) == 0
        assert len(a_only) == len(mock_state_dict_llama) - 10


# ============================================================================
# Test: Null Distribution
# ============================================================================

class TestNullDistribution:
    def test_save_and_load(self, mock_state_dict_llama):
        """Test saving and loading null distribution."""
        # Create simple null distribution
        by_layer = pd.DataFrame({
            "layer": ["layer_0", "layer_1"],
            "wmean_cosine": [0.3, 0.35],
            "wmean_rel_l2": [1.2, 1.1],
        })
        by_module = pd.DataFrame({
            "module": ["attention", "mlp"],
            "wmean_cosine": [0.32, 0.28],
            "wmean_rel_l2": [1.15, 1.25],
        })
        by_layer_module = pd.DataFrame({
            "layer": ["layer_0", "layer_0"],
            "module": ["attention", "mlp"],
            "wmean_cosine": [0.31, 0.29],
            "wmean_rel_l2": [1.18, 1.22],
        })
        raw_pairs = pd.DataFrame({
            "pair_idx": [0],
            "weighted_mean_cosine": [0.3],
        })
        
        null_dist = NullDistribution(
            by_layer=by_layer,
            by_module=by_module,
            by_layer_module=by_layer_module,
            raw_pairs=raw_pairs,
            meta={"n_models": 2, "n_pairs": 1},
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            null_dist.save(tmpdir)
            loaded = NullDistribution.load(tmpdir)
            
            assert len(loaded.by_layer) == 2
            assert loaded.meta["n_models"] == 2
    
    def test_percentile(self):
        """Test percentile computation."""
        by_layer = pd.DataFrame({
            "layer": ["layer_0"] * 100,
            "wmean_cosine": np.linspace(0, 1, 100),
        })
        
        null_dist = NullDistribution(
            by_layer=by_layer,
            by_module=pd.DataFrame(),
            by_layer_module=pd.DataFrame(),
            raw_pairs=pd.DataFrame(),
        )
        
        # 0.5 should be around 50th percentile
        pct = null_dist.percentile("cosine", 0.5, layer="layer_0")
        assert 45 < pct < 55
        
        # 0.95 should be around 95th percentile
        pct = null_dist.percentile("cosine", 0.95, layer="layer_0")
        assert 90 < pct < 100


# ============================================================================
# Test: CLI (smoke tests)
# ============================================================================

class TestCLI:
    def test_parser_compare(self):
        from mlaudit.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "compare",
            "--model-a", "a.pt",
            "--model-b", "b.pt",
            "--out", "output",
        ])
        assert args.model_a == "a.pt"
        assert args.model_b == "b.pt"
        assert args.out == "output"
    
    def test_parser_build_null(self):
        from mlaudit.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "build-null",
            "--models", "a.pt", "b.pt", "c.pt",
            "--out", "null_output",
            "--architecture", "llama-7b",
        ])
        assert len(args.models) == 3
        assert args.architecture == "llama-7b"
    
    def test_parser_info(self):
        from mlaudit.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "info",
            "--model", "model.pt",
            "--list-keys",
        ])
        assert args.model == "model.pt"
        assert args.list_keys is True


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    def test_full_pipeline(self, mock_state_dict_llama):
        """Test full comparison pipeline."""
        # Create slightly modified version
        sd_modified = {k: v + torch.randn_like(v) * 0.001 for k, v in mock_state_dict_llama.items()}
        
        # Compare
        result = compare_state_dicts(
            mock_state_dict_llama,
            sd_modified,
            CompareConfig(grouping="llama", progress=False)
        )
        
        # Should have high similarity (small noise)
        summary = result.global_summary()
        assert summary["weighted_mean_cosine"] > 0.99
        
        # Profiles should exist
        assert len(result.profiles["by_layer"]) > 0
        assert len(result.profiles["by_module"]) > 0
