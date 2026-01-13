"""
Architecture-aware tensor key grouping.

Provides strategies to group model parameter keys by:
- Layer index (layer_0, layer_1, ..., layer_N)
- Module type (embedding, attention, mlp, layernorm, etc.)

This enables meaningful aggregation of similarity metrics at both
the layer and module level.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# Type alias for grouping functions
GroupingFunc = Callable[[str], "GroupingResult"]


@dataclass(frozen=True)
class GroupingResult:
    """Result of grouping a tensor key."""

    layer: str  # e.g., "layer_0", "layer_12", "unknown"
    module: str  # e.g., "embedding", "attention", "mlp", "layernorm"


# Regex patterns for common architectures
LAYER_PATTERNS = [
    # Llama / Mistral / most HuggingFace models
    re.compile(r"\.layers\.(\d+)\."),
    # GPT-2 / GPT-Neo
    re.compile(r"\.h\.(\d+)\."),
    # BERT / RoBERTa
    re.compile(r"\.encoder\.layer\.(\d+)\."),
    # T5
    re.compile(r"\.block\.(\d+)\."),
    # General fallback
    re.compile(r"\.(\d+)\."),
]


def _extract_layer_index(key: str) -> Optional[int]:
    """Extract layer index from a key using common patterns."""
    for pattern in LAYER_PATTERNS:
        match = pattern.search(key)
        if match:
            return int(match.group(1))
    return None


def _classify_module(key: str) -> str:
    """
    Classify a tensor key into a module category.
    
    Categories:
    - embedding: Token/position embeddings
    - attention: Self-attention weights (q, k, v, o projections)
    - mlp: Feed-forward network weights
    - layernorm: Layer normalization parameters
    - moe_router: Mixture-of-experts routing
    - lm_head: Language model output head
    - other: Unclassified
    """
    lk = key.lower()

    # Embedding layers
    if any(x in lk for x in ["embed", "wte", "wpe", "tok_embed", "position_embed"]):
        return "embedding"

    # Output head
    if any(x in lk for x in ["lm_head", "output_proj", "classifier"]):
        return "lm_head"

    # Layer normalization
    if any(x in lk for x in ["layernorm", "layer_norm", ".ln", "rmsnorm", "rms_norm", "norm."]):
        return "layernorm"

    # Attention components
    if any(x in lk for x in ["attn", "attention", "self_attn", ".q_proj", ".k_proj", ".v_proj", ".o_proj"]):
        return "attention"

    # MLP / FFN components
    if any(x in lk for x in ["mlp", "ffn", "feed_forward", ".up_proj", ".down_proj", ".gate_proj", "fc1", "fc2"]):
        return "mlp"

    # MoE components
    if any(x in lk for x in ["router", "gate", "moe", "expert"]):
        return "moe"

    return "other"


def default_grouping(key: str) -> GroupingResult:
    """
    Default grouping strategy that works across architectures.
    
    Attempts to extract layer index and classify module type.
    """
    layer_idx = _extract_layer_index(key)
    layer = f"layer_{layer_idx}" if layer_idx is not None else "unknown"
    module = _classify_module(key)
    return GroupingResult(layer=layer, module=module)


def llama_grouping(key: str) -> GroupingResult:
    """
    Grouping optimized for Llama-family models.
    
    Expected key patterns:
    - model.embed_tokens.weight
    - model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    - model.layers.{i}.mlp.{gate,up,down}_proj.weight
    - model.layers.{i}.input_layernorm.weight
    - model.layers.{i}.post_attention_layernorm.weight
    - model.norm.weight
    - lm_head.weight
    """
    lk = key.lower()

    # Extract layer index
    layer_match = re.search(r"\.layers\.(\d+)\.", key)
    if layer_match:
        layer = f"layer_{layer_match.group(1)}"
    elif "embed" in lk:
        layer = "layer_embed"
    elif "lm_head" in lk or (key == "model.norm.weight"):
        layer = "layer_final"
    else:
        layer = "unknown"

    # Classify module
    module = _classify_module(key)

    return GroupingResult(layer=layer, module=module)


def gpt2_grouping(key: str) -> GroupingResult:
    """
    Grouping optimized for GPT-2 / GPT-Neo models.
    
    Expected key patterns:
    - transformer.wte.weight
    - transformer.wpe.weight
    - transformer.h.{i}.attn.*
    - transformer.h.{i}.mlp.*
    - transformer.h.{i}.ln_1.*, transformer.h.{i}.ln_2.*
    - transformer.ln_f.*
    """
    lk = key.lower()

    # Extract layer index
    layer_match = re.search(r"\.h\.(\d+)\.", key)
    if layer_match:
        layer = f"layer_{layer_match.group(1)}"
    elif "wte" in lk or "wpe" in lk:
        layer = "layer_embed"
    elif "ln_f" in lk:
        layer = "layer_final"
    else:
        layer = "unknown"

    module = _classify_module(key)

    return GroupingResult(layer=layer, module=module)


def bert_grouping(key: str) -> GroupingResult:
    """
    Grouping optimized for BERT-family models.
    
    Expected key patterns:
    - bert.embeddings.*
    - bert.encoder.layer.{i}.attention.*
    - bert.encoder.layer.{i}.intermediate.*
    - bert.encoder.layer.{i}.output.*
    - bert.pooler.*
    """
    lk = key.lower()

    layer_match = re.search(r"\.layer\.(\d+)\.", key)
    if layer_match:
        layer = f"layer_{layer_match.group(1)}"
    elif "embedding" in lk:
        layer = "layer_embed"
    elif "pooler" in lk:
        layer = "layer_final"
    else:
        layer = "unknown"

    # BERT-specific module classification
    if "embedding" in lk:
        module = "embedding"
    elif "attention" in lk:
        module = "attention"
    elif "intermediate" in lk or "output" in lk:
        module = "mlp"
    elif "layernorm" in lk or "layer_norm" in lk:
        module = "layernorm"
    elif "pooler" in lk:
        module = "lm_head"
    else:
        module = "other"

    return GroupingResult(layer=layer, module=module)


def moe_grouping(key: str) -> GroupingResult:
    """
    Grouping for Mixture-of-Experts models (Mixtral, etc.).
    
    Adds expert-level granularity.
    """
    base = llama_grouping(key)

    # Check for expert index
    expert_match = re.search(r"\.experts\.(\d+)\.", key)
    if expert_match:
        expert_idx = expert_match.group(1)
        module = f"{base.module}_expert_{expert_idx}"
        return GroupingResult(layer=base.layer, module=module)

    return base


# Registry of available grouping strategies
GROUPINGS: Dict[str, GroupingFunc] = {
    "default": default_grouping,
    "llama": llama_grouping,
    "gpt2": gpt2_grouping,
    "bert": bert_grouping,
    "moe": moe_grouping,
}


def auto_detect_grouping(keys: List[str]) -> str:
    """
    Attempt to auto-detect the best grouping strategy based on key patterns.
    
    Args:
        keys: List of tensor keys from a model
        
    Returns:
        Name of the recommended grouping strategy
    """
    sample = " ".join(keys[:100]).lower()

    if ".layers." in sample and ("llama" in sample or "self_attn" in sample):
        return "llama"
    if ".h." in sample and ("transformer" in sample or "wte" in sample):
        return "gpt2"
    if ".encoder.layer." in sample and "bert" in sample:
        return "bert"
    if ".experts." in sample:
        return "moe"

    return "default"


def group_keys(
    keys: List[str],
    grouping: str | GroupingFunc = "default",
) -> Dict[str, Dict[str, List[str]]]:
    """
    Group tensor keys by layer and module.
    
    Args:
        keys: List of tensor keys
        grouping: Grouping strategy name or function
        
    Returns:
        Nested dict: {layer: {module: [keys]}}
    """
    if isinstance(grouping, str):
        grouper = GROUPINGS.get(grouping, GROUPINGS["default"])
    else:
        grouper = grouping

    result: Dict[str, Dict[str, List[str]]] = {}

    for key in keys:
        g = grouper(key)
        if g.layer not in result:
            result[g.layer] = {}
        if g.module not in result[g.layer]:
            result[g.layer][g.module] = []
        result[g.layer][g.module].append(key)

    return result
