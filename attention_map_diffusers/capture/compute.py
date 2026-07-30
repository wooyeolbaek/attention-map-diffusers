from __future__ import annotations

from typing import Optional, Sequence, Union


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "attention-map-diffusers requires torch for attention capture. "
            "Install the package dependencies before running model demos."
        ) from exc
    return torch


def compute_attention_probs(
    query,
    key,
    *,
    attention_mask=None,
    scale: Optional[float] = None,
    chunk_size: int = 1024,
    heads: Union[str, Sequence[int]] = "all",
    key_indices: Optional[Sequence[int]] = None,
):
    torch = require_torch()
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must have shape [batch, heads, tokens, head_dim].")

    scale_factor = scale if scale is not None else query.shape[-1] ** -0.5
    key_t = key.float().transpose(-2, -1)
    output_key_index = None
    if key_indices is not None:
        output_key_index = torch.tensor(key_indices, device=query.device, dtype=torch.long)
    chunks = []
    for start in range(0, query.shape[-2], chunk_size):
        end = min(start + chunk_size, query.shape[-2])
        logits = torch.matmul(query[:, :, start:end].float(), key_t) * scale_factor
        if attention_mask is not None:
            logits = logits + attention_mask[..., start:end, :]
        probabilities = torch.softmax(logits, dim=-1).to(query.dtype)
        if output_key_index is not None:
            probabilities = probabilities.index_select(-1, output_key_index)
        # Reduce before concatenating query chunks. For the common heads="mean"
        # path this avoids retaining an H-times larger result tensor.
        chunks.append(reduce_heads(probabilities, heads))
    return torch.cat(chunks, dim=-2)


def reduce_heads(tensor, heads: Union[str, Sequence[int]] = "mean"):
    if heads == "all":
        return tensor
    if heads == "mean":
        return tensor.mean(dim=1, keepdim=True)
    torch = require_torch()
    index = torch.tensor([int(head) for head in heads], device=tensor.device)
    return tensor.index_select(1, index)


def output_head_count(heads: Union[str, Sequence[int]], total_heads: int) -> int:
    """Return the head dimension retained in a captured attention tensor."""
    if heads == "mean":
        return 1
    if heads == "all":
        return int(total_heads)
    return len(heads)


def tensor_to_device(tensor, offload: Optional[str], dtype: Optional[str] = None):
    if tensor is None:
        return None
    torch = require_torch()
    tensor = tensor.detach()
    if dtype is not None:
        dtype_value = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        tensor = tensor.to(dtype=dtype_value)
    if offload is not None:
        tensor = tensor.to(offload)
    return tensor
