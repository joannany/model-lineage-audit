"""
Model checkpoint loading utilities.

Supports:
- PyTorch checkpoints (.pt, .pth, .bin)
- Safetensors format (.safetensors)
- Various checkpoint wrapper formats (state_dict, model, etc.)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch

logger = logging.getLogger(__name__)

# Type aliases
TensorDict = Dict[str, torch.Tensor]

# Optional safetensors support
try:
    from safetensors.torch import load_file as safetensors_load_file

    HAS_SAFETENSORS = True
except ImportError:
    safetensors_load_file = None  # type: ignore
    HAS_SAFETENSORS = False


@dataclass
class LoadedState:
    """Container for loaded model state with metadata."""

    path: str
    tensors: TensorDict
    format: str
    original_keys: int = 0
    filtered_keys: int = 0
    skipped_keys: List[str] = field(default_factory=list)
    total_params: int = 0

    def __post_init__(self) -> None:
        if self.total_params == 0:
            self.total_params = sum(t.numel() for t in self.tensors.values())

    def summary(self) -> str:
        """Return a summary string of the loaded state."""
        return (
            f"LoadedState(path='{Path(self.path).name}', "
            f"format='{self.format}', "
            f"tensors={len(self.tensors)}, "
            f"params={self.total_params:,}, "
            f"skipped={len(self.skipped_keys)})"
        )


# Allowed floating-point dtypes for comparison
FLOAT_DTYPES: Set[torch.dtype] = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def _unwrap_checkpoint(obj: Any) -> Dict[str, Any]:
    """
    Unwrap common checkpoint wrapper formats to get the raw state dict.
    
    Handles:
    - Direct state dicts
    - {"state_dict": ...}
    - {"model": ...}
    - {"module": ...} (DDP wrappers)
    - Nested combinations
    """
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict, got {type(obj).__name__}")

    # Try common wrapper keys in order of likelihood
    wrapper_keys = ["state_dict", "model", "module", "model_state_dict", "net"]

    for key in wrapper_keys:
        if key in obj and isinstance(obj[key], dict):
            candidate = obj[key]
            # Check if it looks like a tensor dict
            if any(hasattr(v, "shape") for v in candidate.values()):
                logger.debug(f"Unwrapped checkpoint using key '{key}'")
                return candidate

    # If no wrapper found, assume it's already a state dict
    return obj


def _load_torch_checkpoint(path: str) -> tuple[TensorDict, List[str]]:
    """
    Load a PyTorch checkpoint file.
    
    Returns:
        Tuple of (tensor_dict, skipped_keys)
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = _unwrap_checkpoint(obj)

    tensors: TensorDict = {}
    skipped: List[str] = []

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            skipped.append(key)
            continue

        tensor = value.detach().cpu()

        # Skip non-floating-point tensors
        if tensor.dtype not in FLOAT_DTYPES:
            skipped.append(key)
            continue

        # Ensure contiguous memory layout for stable operations
        tensors[key] = tensor.contiguous()

    if not tensors:
        raise ValueError(
            f"No floating-point tensors found in checkpoint. "
            f"Skipped {len(skipped)} keys."
        )

    return tensors, skipped


def _load_safetensors(path: str) -> tuple[TensorDict, List[str]]:
    """
    Load a safetensors file.
    
    Returns:
        Tuple of (tensor_dict, skipped_keys)
    """
    if safetensors_load_file is None:
        raise RuntimeError(
            "safetensors is not installed. Install with: pip install safetensors"
        )

    raw_tensors = safetensors_load_file(path, device="cpu")

    tensors: TensorDict = {}
    skipped: List[str] = []

    for key, tensor in raw_tensors.items():
        tensor = tensor.detach().cpu()

        if tensor.dtype not in FLOAT_DTYPES:
            skipped.append(key)
            continue

        tensors[key] = tensor.contiguous()

    if not tensors:
        raise ValueError(
            f"No floating-point tensors found in safetensors file. "
            f"Skipped {len(skipped)} keys."
        )

    return tensors, skipped


def load_state(
    path: str | Path,
    *,
    verbose: bool = False,
) -> LoadedState:
    """
    Load model state from a checkpoint file.
    
    Args:
        path: Path to the checkpoint file
        verbose: If True, log detailed loading information
        
    Returns:
        LoadedState containing tensors and metadata
        
    Raises:
        FileNotFoundError: If the path doesn't exist
        ValueError: If the checkpoint format is unsupported or contains no tensors
    """
    path_str = str(path)

    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Checkpoint not found: {path_str}")

    ext = os.path.splitext(path_str)[1].lower()

    # Dispatch based on extension
    if ext == ".safetensors":
        tensors, skipped = _load_safetensors(path_str)
        fmt = "safetensors"
    elif ext in {".pt", ".pth", ".bin", ".ckpt"}:
        tensors, skipped = _load_torch_checkpoint(path_str)
        fmt = "torch"
    else:
        # Try torch as fallback
        logger.warning(f"Unknown extension '{ext}', attempting PyTorch load")
        try:
            tensors, skipped = _load_torch_checkpoint(path_str)
            fmt = "torch(?)"
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint with unknown extension: {e}")

    original_keys = len(tensors) + len(skipped)

    state = LoadedState(
        path=path_str,
        tensors=tensors,
        format=fmt,
        original_keys=original_keys,
        filtered_keys=len(tensors),
        skipped_keys=skipped,
    )

    if verbose:
        logger.info(state.summary())

    return state


def get_common_keys(
    sd_a: TensorDict,
    sd_b: TensorDict,
    *,
    require_same_shape: bool = True,
) -> tuple[List[str], List[str], List[str]]:
    """
    Find common keys between two state dicts.
    
    Args:
        sd_a: First state dict
        sd_b: Second state dict
        require_same_shape: If True, only include keys with matching shapes
        
    Returns:
        Tuple of (common_keys, a_only_keys, b_only_keys)
        If require_same_shape is True, common_keys only includes shape-matched keys
    """
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())

    common = keys_a & keys_b
    a_only = sorted(keys_a - keys_b)
    b_only = sorted(keys_b - keys_a)

    if require_same_shape:
        common_matched = []
        for key in sorted(common):
            if sd_a[key].shape == sd_b[key].shape:
                common_matched.append(key)
        return common_matched, a_only, b_only

    return sorted(common), a_only, b_only
