from __future__ import annotations

import os
from typing import Any, Dict

from .attention_capture import AttentionCapture


attn_maps: Dict[Any, Dict[str, Any]] = {}
_legacy_capture = None


def init_pipeline(
    pipeline,
    *,
    relations="image->text",
    mode="exact",
    heads="all",
    offload="cpu",
    query_indices=None,
    key_indices=None,
    tokens=None,
    **kwargs,
):
    """Backward-compatible pipeline initialization.

    The original project used global monkey patches and a global ``attn_maps``
    dict. This function now installs an ``AttentionCapture`` instance on the
    pipeline and keeps the old global dict updated after generation through
    ``save_attention_maps``.

    New code should prefer:

    ``with AttentionCapture(pipe, relations=[...]) as cap: ...; maps = cap.compute()``
    """

    global _legacy_capture
    capture = AttentionCapture(
        pipeline,
        relations=relations,
        mode=mode,
        heads=heads,
        offload=offload,
        query_indices=query_indices,
        key_indices=key_indices,
        tokens=tokens,
        **kwargs,
    )
    capture.install()
    _legacy_capture = capture
    setattr(pipeline, "_attention_map_capture", capture)
    return pipeline


def get_pipeline_attention_maps(pipeline, relation: str = "image->text"):
    capture = getattr(pipeline, "_attention_map_capture", None)
    if capture is None:
        raise RuntimeError("Pipeline was not initialized with init_pipeline or AttentionCapture.")
    maps = capture.compute()
    legacy = maps.to_legacy(relation=relation)
    attn_maps.clear()
    attn_maps.update(legacy)
    return maps


def restore_pipeline(pipeline) -> None:
    global _legacy_capture
    capture = getattr(pipeline, "_attention_map_capture", None)
    if capture is not None:
        capture.restore()
        if _legacy_capture is capture:
            _legacy_capture = None
        delattr(pipeline, "_attention_map_capture")


def process_token(token, startofword):
    if "</w>" in token:
        token = token.replace("</w>", "")
        if startofword:
            token = "<" + token + ">"
        else:
            token = "-" + token + ">"
            startofword = True
    elif token not in ["<|startoftext|>", "<|endoftext|>"]:
        if startofword:
            token = "<" + token + "-"
            startofword = False
        else:
            token = "-" + token + "-"
    return _safe_filename_token(token), startofword


def save_attention_image(attn_map, tokens, batch_dir, to_pil):
    startofword = True
    for i, (token, attention) in enumerate(zip(tokens, attn_map[: len(tokens)])):
        token, startofword = process_token(token, startofword)
        image = _normalize_for_image(attention)
        to_pil(image.to("cpu", dtype=_torch().float32)).save(os.path.join(batch_dir, f"{i}-{token}.png"))


def save_attention_maps(
    maps,
    tokenizer=None,
    prompts=None,
    base_dir="attn_maps",
    unconditional=True,
    relation: str = "image->text",
):
    if maps is attn_maps and not maps and _legacy_capture is not None:
        computed = _legacy_capture.compute()
        legacy = computed.to_legacy(relation=relation)
        attn_maps.update(legacy)
        computed.save(
            base_dir,
            tokenizer=tokenizer,
            prompts=prompts,
            relation=relation,
            unconditional=unconditional,
        )
        return legacy

    if hasattr(maps, "to_legacy"):
        legacy = maps.to_legacy(relation=relation)
        maps.save(
            base_dir,
            tokenizer=tokenizer,
            prompts=prompts,
            relation=relation,
            unconditional=unconditional,
        )
        return legacy

    if not maps:
        raise ValueError("No attention maps were captured.")

    torch = _torch()
    functional = _torch_functional()
    to_pil = _to_pil_image()

    token_groups = _token_groups(tokenizer, prompts)
    os.makedirs(base_dir, exist_ok=True)

    first_map = list(list(maps.values())[0].values())[0]
    total_attn_map = _prepare_legacy_map(first_map, unconditional)
    total_attn_map = total_attn_map.permute(0, 3, 1, 2)
    total_attn_map = torch.zeros_like(total_attn_map)
    total_attn_map_shape = total_attn_map.shape[-2:]
    total_attn_map_number = 0

    for timestep, layers in maps.items():
        timestep_dir = os.path.join(base_dir, f"{timestep}")
        os.makedirs(timestep_dir, exist_ok=True)

        for layer, attn_map in layers.items():
            layer_dir = os.path.join(timestep_dir, f"{_safe_filename_token(layer)}")
            os.makedirs(layer_dir, exist_ok=True)

            attn_map = _prepare_legacy_map(attn_map, unconditional)
            resized_attn_map = functional.interpolate(
                attn_map.permute(0, 3, 1, 2),
                size=total_attn_map_shape,
                mode="bilinear",
                align_corners=False,
            )
            total_attn_map += resized_attn_map
            total_attn_map_number += 1

            for batch, (tokens, attn) in enumerate(zip(token_groups, attn_map.permute(0, 3, 1, 2))):
                batch_dir = os.path.join(layer_dir, f"batch-{batch}")
                os.makedirs(batch_dir, exist_ok=True)
                save_attention_image(attn, tokens, batch_dir, to_pil)

    if total_attn_map_number:
        total_attn_map /= total_attn_map_number

    for batch, (attention_map, tokens) in enumerate(zip(total_attn_map, token_groups)):
        batch_dir = os.path.join(base_dir, f"batch-{batch}")
        os.makedirs(batch_dir, exist_ok=True)
        save_attention_image(attention_map, tokens, batch_dir, to_pil)

    return maps


def save_attention_record_visuals(
    records,
    tokenizer=None,
    prompts=None,
    base_dir="attn_maps",
    unconditional=False,
):
    """Compatibility wrapper around the relation-aware visualizer."""
    records = [record for record in records if record.data is not None]
    if not records:
        return None
    from .visualization import AttentionVisualizer

    return AttentionVisualizer().save(
        records,
        base_dir,
        tokenizer=tokenizer,
        prompts=prompts,
        relations=sorted({record.relation for record in records}),
        unconditional=unconditional,
    )


def _prepare_legacy_map(attn_map, unconditional: bool):
    attn_map = attn_map.sum(1)
    if attn_map.ndim == 5 and attn_map.shape[1] == 1:
        attn_map = attn_map.squeeze(1)
    if unconditional:
        attn_map = attn_map.chunk(2)[1]
    return attn_map


def _token_groups(tokenizer, prompts):
    if tokenizer is None or prompts is None:
        return [["token"]]
    token_ids = tokenizer(prompts)["input_ids"]
    token_ids = token_ids if token_ids and isinstance(token_ids[0], list) else [token_ids]
    return [tokenizer.convert_ids_to_tokens(token_id) for token_id in token_ids]


def _normalize_for_image(tensor):
    tensor = tensor.detach()
    torch = _torch()
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    tensor = tensor - tensor.min()
    denominator = tensor.max().clamp_min(1e-8)
    return tensor / denominator


def _safe_filename_token(value: str) -> str:
    value = str(value)
    for char in '/\\:*?"<>|':
        value = value.replace(char, "_")
    return value


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Saving attention maps requires torch.") from exc
    return torch


def _torch_functional():
    try:
        import torch.nn.functional as functional
    except ImportError as exc:
        raise ImportError("Saving attention maps requires torch.") from exc
    return functional


def _to_pil_image():
    try:
        from torchvision.transforms import ToPILImage
    except ImportError as exc:
        raise ImportError("Saving attention map PNG files requires torchvision.") from exc
    return ToPILImage()
