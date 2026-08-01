from __future__ import annotations

from dataclasses import fields
from importlib import metadata as importlib_metadata
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .capture.collector import AttentionCollector, AttentionMaps
from .capture.config import CaptureConfig
from .capture.context import (
    CaptureRuntimeContext,
    infer_grid_from_hidden_states,
    infer_grid_from_img_ids,
    infer_video_grid_from_hidden_states,
    pop_context,
    push_context,
)
from .capture.compute import require_torch
from .processors import make_capture_processor


class AttentionCapture:
    """Context manager for relation-aware Diffusers attention-map capture.

    The default ``mode='fast'`` keeps Diffusers' SDPA/Flash Attention path for
    generation and stores q/k tensors for post-hoc attention-map recomputation.
    ``mode='exact'`` materializes selected maps during the attention call and
    is intended for small or heavily selected maps. Only the final denoising
    timestep is retained by default to bound capture memory.
    """

    def __init__(
        self,
        pipeline,
        *,
        relations="image->text",
        layers="all",
        timesteps="last",
        heads="mean",
        mode="fast",
        offload="cpu",
        dtype=None,
        query_indices=None,
        key_indices=None,
        relation_query_indices=None,
        relation_key_indices=None,
        tokens=None,
        chunk_size=1024,
        max_map_elements=128_000_000,
        max_capture_bytes=8 * 1024**3,
        allow_large_maps=False,
        metadata=None,
    ):
        self.pipeline = pipeline
        self.config = CaptureConfig(
            relations=relations,
            layers=layers,
            timesteps=timesteps,
            heads=heads,
            mode=mode,
            offload=offload,
            dtype=dtype,
            query_indices=query_indices,
            key_indices=key_indices,
            relation_query_indices=relation_query_indices,
            relation_key_indices=relation_key_indices,
            tokens=tokens,
            chunk_size=chunk_size,
            max_map_elements=max_map_elements,
            max_capture_bytes=max_capture_bytes,
            allow_large_maps=allow_large_maps,
        )
        self.metadata = _capture_metadata(pipeline)
        self.text_conditioning = _text_conditioning_schema(pipeline)
        self.metadata["text_conditioning"] = self.text_conditioning
        self.metadata.update(dict(metadata or {}))
        self.metadata["capture_config"] = {
            item.name: getattr(self.config, item.name) for item in fields(self.config)
        }
        self.collector = AttentionCollector(self.config, metadata=self.metadata)
        self._processors: List[Tuple[Any, Any]] = []
        self._hooks: List[Any] = []
        self._installed = False
        self._last_denoiser_timestep: Optional[float] = None
        self._denoiser_call_index = -1
        self.coverage: Dict[str, Any] = {}

    def __enter__(self) -> "AttentionCapture":
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.restore()

    @property
    def records(self):
        return self.collector.records

    def clear(self) -> None:
        self.collector.clear()
        self._last_denoiser_timestep = None
        self._denoiser_call_index = -1

    def compute(self, compute_device=None) -> AttentionMaps:
        return self.collector.compute(compute_device=compute_device)

    def save(self, base_dir: str, **kwargs):
        compute_device = kwargs.pop("compute_device", None)
        return self.compute(compute_device=compute_device).save(base_dir, **kwargs)

    def install(self):
        require_torch()
        if self._installed:
            return self.pipeline
        self.clear()
        try:
            self._install_processors()
            self._install_context_hooks()
            self._installed = True
        except Exception:
            self.restore()
            raise
        return self.pipeline

    def restore(self) -> None:
        for handle in reversed(self._hooks):
            handle.remove()
        self._hooks.clear()

        for module, original_processor in reversed(self._processors):
            _set_processor(module, original_processor)
            if getattr(module, "_attention_map_capture_owner", None) is self:
                delattr(module, "_attention_map_capture_owner")
        self._processors.clear()
        self._installed = False

    def _install_processors(self) -> None:
        components = _pipeline_model_components(self.pipeline)
        processor_counts: Dict[str, int] = {}
        unsupported = []
        captured_layers = []
        for component_name, component in components:
            for name, module in component.named_modules():
                processor = getattr(module, "processor", None)
                if processor is None:
                    continue
                processor_name = processor.__class__.__name__
                processor_counts[processor_name] = processor_counts.get(processor_name, 0) + 1
                owner = getattr(module, "_attention_map_capture_owner", None)
                if owner is not None and owner is not self:
                    raise RuntimeError(
                        "This pipeline already has an active AttentionCapture. "
                        "Nested capture on the same pipeline is not supported."
                    )
                layer_id = f"{component_name}.{name}" if name else component_name
                capture_processor = make_capture_processor(processor, self.collector, layer_id)
                if capture_processor is None:
                    unsupported.append({"layer": layer_id, "processor": processor_name})
                    continue
                self._processors.append((module, processor))
                _set_processor(module, capture_processor)
                setattr(module, "_attention_map_capture_owner", self)
                captured_layers.append(layer_id)
        self.coverage = {
            "attention_modules": sum(processor_counts.values()),
            "captured_modules": len(captured_layers),
            "processor_counts": processor_counts,
            "captured_layers": captured_layers,
            "unsupported": unsupported,
        }
        self.metadata["attention_coverage"] = self.coverage
        self.collector.metadata["attention_coverage"] = self.coverage
        if unsupported:
            details = ", ".join(
                f"{item['layer']} ({item['processor']})" for item in unsupported[:8]
            )
            if len(unsupported) > 8:
                details += f", ... and {len(unsupported) - 8} more"
            raise RuntimeError(
                "Attention capture coverage is incomplete; unsupported denoiser processor(s): "
                + details
            )
        if not captured_layers:
            raise RuntimeError("No attention processor modules were found in the pipeline denoiser.")

    def _install_context_hooks(self) -> None:
        for component_name, component in _pipeline_model_components(self.pipeline):
            self._register_context_hooks(component, component_name, is_denoiser=True)
            for name, module in component.named_modules():
                if not name:
                    continue
                class_name = module.__class__.__name__
                if class_name in {
                    "Transformer2DModel",
                    "SD3Transformer2DModel",
                    "FluxTransformer2DModel",
                    "Flux2Transformer2DModel",
                    "SanaTransformer2DModel",
                    "ZImageTransformer2DModel",
                    "PRXTransformer2DModel",
                    "CogVideoXTransformer3DModel",
                    "HunyuanVideoTransformer3DModel",
                    "HunyuanVideo15Transformer3DModel",
                    "WanTransformer3DModel",
                }:
                    self._register_context_hooks(module, f"{component_name}.{name}" if name else component_name)

    def _register_context_hooks(self, module, layer_id: str, *, is_denoiser: bool = False) -> None:
        def pre_hook(mod, args, kwargs):
            context = _context_from_forward(
                mod,
                args,
                kwargs,
                layer_id,
                text_conditioning=self.text_conditioning,
            )
            if is_denoiser:
                self._annotate_denoiser_call(context, mod)
            if context.timestep is not None:
                self.collector.set_step(timestep=context.timestep)
            token = push_context(context)
            setattr(mod, "_attention_map_capture_token", token)

        def post_hook(mod, args, kwargs, output):
            token = getattr(mod, "_attention_map_capture_token", None)
            if token is not None:
                pop_context(token)
                setattr(mod, "_attention_map_capture_token", None)

        self._hooks.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        self._hooks.append(
            module.register_forward_hook(post_hook, with_kwargs=True, always_call=True)
        )

    def _annotate_denoiser_call(self, context: CaptureRuntimeContext, module) -> None:
        timestep = context.timestep
        if timestep != self._last_denoiser_timestep:
            self._last_denoiser_timestep = timestep
            self._denoiser_call_index = 0
        else:
            self._denoiser_call_index += 1
        context.metadata["denoiser_call_index"] = self._denoiser_call_index

        if module.__class__.__name__ != "HunyuanVideo15Transformer3DModel":
            return
        guider = getattr(self.pipeline, "guider", None)
        enabled = bool(getattr(guider, "_enabled", False))
        conditions = int(getattr(guider, "num_conditions", 1) or 1)
        context.metadata["guidance_role"] = _guidance_role(
            self._denoiser_call_index,
            enabled=enabled,
            conditions=conditions,
        )


def _pipeline_model_components(pipeline) -> Iterable[Tuple[str, Any]]:
    for name in ("unet", "transformer"):
        component = getattr(pipeline, name, None)
        if component is not None:
            yield name, component


def _set_processor(module, processor) -> None:
    if hasattr(module, "set_processor"):
        module.set_processor(processor)
    else:
        module.processor = processor


def _context_from_forward(
    module,
    args,
    kwargs: Dict[str, Any],
    layer_id: str,
    text_conditioning: Optional[Dict[str, Any]] = None,
) -> CaptureRuntimeContext:
    class_name = module.__class__.__name__
    hidden_states = _first_present(kwargs, "sample", "hidden_states")
    if hidden_states is None and args:
        hidden_states = args[0]

    timestep = _first_present(kwargs, "timestep")
    if timestep is None and len(args) > 1 and class_name == "UNet2DConditionModel":
        timestep = args[1]

    patch_size = getattr(getattr(module, "config", None), "patch_size", None)
    image_grid = None
    metadata = {"context_layer": layer_id, "context_class": class_name}
    text_components = _text_components_for_forward(
        class_name,
        kwargs,
        text_conditioning,
    )
    if text_components:
        metadata["text_components"] = text_components

    video_classes = {
        "CogVideoXTransformer3DModel",
        "HunyuanVideoTransformer3DModel",
        "HunyuanVideo15Transformer3DModel",
        "WanTransformer3DModel",
    }
    if class_name in video_classes:
        config = getattr(module, "config", None)
        frames_first = class_name == "CogVideoXTransformer3DModel"
        spatial_patch = getattr(config, "patch_size", 1)
        temporal_patch = getattr(config, "patch_size_t", 1)
        if class_name == "WanTransformer3DModel":
            patch_tuple = tuple(spatial_patch)
            temporal_patch, spatial_patch = patch_tuple[0], patch_tuple[1]
        spatial_patch = int(spatial_patch or 1)
        temporal_patch = int(temporal_patch or 1)
        video_grid = infer_video_grid_from_hidden_states(
            hidden_states,
            patch_size=spatial_patch,
            temporal_patch_size=temporal_patch,
            frames_first=frames_first,
            ceil_temporal=frames_first and getattr(config, "patch_size_t", None) is not None,
        )
        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        metadata["input_video_shape"] = (
            tuple(int(value) for value in hidden_states.shape)
            if hidden_states is not None and hasattr(hidden_states, "shape")
            else None
        )
        return CaptureRuntimeContext(
            timestep=_to_float(timestep),
            video_grid=video_grid,
            video_tokens=video_grid.spatial_tokens if video_grid is not None else None,
            text_tokens=_shape_dim(encoder_hidden_states, 1),
            text_components=text_components,
            metadata=metadata,
        )

    if class_name in {"FluxTransformer2DModel", "Flux2Transformer2DModel"}:
        image_grid = infer_grid_from_img_ids(kwargs.get("img_ids"))
        if image_grid is None:
            image_grid = infer_grid_from_hidden_states(hidden_states)
        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        return CaptureRuntimeContext(
            timestep=_to_float(timestep),
            image_grid=image_grid,
            image_tokens=_shape_dim(hidden_states, 1),
            text_tokens=_shape_dim(encoder_hidden_states, 1),
            text_components=text_components,
            metadata=metadata,
        )

    if class_name == "ZImageTransformer2DModel":
        x = kwargs.get("x", hidden_states)
        cap_feats = kwargs.get("cap_feats")
        if cap_feats is None and len(args) > 2:
            cap_feats = args[2]
        patch_size = int(kwargs.get("patch_size", 2))
        f_patch_size = int(kwargs.get("f_patch_size", 1))
        image = x[0] if isinstance(x, (list, tuple)) and x else None
        caption = cap_feats[0] if isinstance(cap_feats, (list, tuple)) and cap_feats else None
        image_tokens = None
        if image is not None and hasattr(image, "shape") and len(image.shape) == 4:
            frames = int(image.shape[-3]) // f_patch_size
            grid_height = int(image.shape[-2]) // patch_size
            grid_width = int(image.shape[-1]) // patch_size
            original_tokens = frames * grid_height * grid_width
            image_tokens = original_tokens + (-original_tokens) % 32
            # Padded tokens have no spatial position. Native generation sizes
            # are divisible by 32; retain the grid only when it is exact.
            if image_tokens == original_tokens:
                from .capture.records import TokenGrid

                image_grid = TokenGrid(
                    frames=frames,
                    height=grid_height,
                    width=grid_width,
                    patch_size=patch_size,
                )
        text_tokens = _shape_dim(caption, 0)
        if text_tokens is not None:
            text_tokens += (-text_tokens) % 32
        if timestep is None:
            timestep = kwargs.get("t")
        if timestep is None and len(args) > 1:
            timestep = args[1]
        return CaptureRuntimeContext(
            timestep=_to_float(timestep),
            image_grid=image_grid,
            image_tokens=image_tokens,
            text_tokens=text_tokens,
            text_components=text_components,
            metadata=metadata,
        )

    if class_name == "PRXTransformer2DModel":
        if hidden_states is not None and hasattr(hidden_states, "shape") and len(hidden_states.shape) == 4:
            from .capture.records import TokenGrid

            patch_size = int(patch_size or getattr(module, "patch_size", 1))
            grid_height = int(hidden_states.shape[-2]) // patch_size
            grid_width = int(hidden_states.shape[-1]) // patch_size
            image_grid = TokenGrid(
                height=grid_height,
                width=grid_width,
                patch_size=patch_size,
            )
            image_tokens = grid_height * grid_width
        else:
            image_tokens = _shape_dim(hidden_states, 1)
        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        return CaptureRuntimeContext(
            timestep=_to_float(timestep),
            image_grid=image_grid,
            image_tokens=image_tokens,
            text_tokens=_shape_dim(encoder_hidden_states, 1),
            text_components=text_components,
            metadata=metadata,
        )

    if class_name in {"SD3Transformer2DModel", "SanaTransformer2DModel"}:
        image_grid = infer_grid_from_hidden_states(hidden_states, patch_size=patch_size)
    elif class_name in {"UNet2DConditionModel", "Transformer2DModel"}:
        image_grid = infer_grid_from_hidden_states(hidden_states)

    encoder_hidden_states = kwargs.get("encoder_hidden_states")
    return CaptureRuntimeContext(
        timestep=_to_float(timestep),
        image_grid=image_grid,
        image_tokens=_shape_dim(hidden_states, 1),
        text_tokens=_shape_dim(encoder_hidden_states, 1),
        text_components=text_components,
        metadata=metadata,
    )


def _text_conditioning_schema(pipeline) -> Dict[str, Any]:
    transformer = getattr(pipeline, "transformer", None)
    denoiser_name = (
        transformer.__class__.__name__
        if transformer is not None
        else getattr(getattr(pipeline, "unet", None), "__class__", type(None)).__name__
    )

    def component(name, tokenizer_attribute, *, length=None, fusion="sequence"):
        item = {
            "name": name,
            "tokenizer_attribute": tokenizer_attribute,
            "fusion": fusion,
        }
        if length is not None:
            item["length"] = int(length)
        return item

    if denoiser_name == "SD3Transformer2DModel":
        clip_length = _tokenizer_length(getattr(pipeline, "tokenizer", None)) or 77
        return {
            "strategy": "token-concatenated",
            "components": [
                component(
                    "clip-l+clip-g-fused",
                    "tokenizer",
                    length=clip_length,
                    fusion="feature-fused",
                ),
                component("t5", "tokenizer_3"),
            ],
            "notes": [
                "CLIP-L and CLIP-G are fused on the feature axis and cannot be separated into probabilities.",
                "The fused CLIP sequence and T5 sequence are concatenated on the token axis.",
            ],
        }
    if denoiser_name == "HunyuanVideo15Transformer3DModel":
        return {
            "strategy": "mask-reordered-token-concatenated",
            "components": [
                component("byt5", "tokenizer_2"),
                component("mllm", "tokenizer"),
            ],
            "non_text_components": ["reference-image"],
        }
    if denoiser_name == "FluxTransformer2DModel":
        return {
            "strategy": "single-attention-sequence",
            "components": [component("t5", "tokenizer_2")],
            "non_attention_conditioning": ["clip-pooled"],
        }
    if denoiser_name == "Flux2Transformer2DModel":
        return {
            "strategy": "single-attention-sequence",
            "components": [component("qwen3", "tokenizer")],
        }
    if denoiser_name == "CogVideoXTransformer3DModel":
        return {
            "strategy": "single-attention-sequence",
            "components": [component("t5", "tokenizer")],
        }
    if denoiser_name == "WanTransformer3DModel":
        return {
            "strategy": "single-attention-sequence",
            "components": [component("umt5", "tokenizer")],
        }
    if denoiser_name == "ZImageTransformer2DModel":
        return {
            "strategy": "single-attention-sequence",
            "components": [component("qwen", "tokenizer")],
        }
    if denoiser_name == "SanaTransformer2DModel":
        return {
            "strategy": "single-attention-sequence",
            "components": [component("gemma", "tokenizer")],
        }
    if denoiser_name == "PRXTransformer2DModel":
        return {
            "strategy": "single-attention-sequence",
            "components": [component("text", "tokenizer")],
        }
    if getattr(pipeline, "tokenizer_2", None) is not None:
        return {
            "strategy": "feature-fused",
            "components": [
                component(
                    "clip-l+clip-g-fused",
                    "tokenizer",
                    fusion="feature-fused",
                )
            ],
            "notes": [
                "The two CLIP outputs share token positions and are concatenated on the feature axis."
            ],
        }
    return {
        "strategy": "single-attention-sequence",
        "components": [component("clip", "tokenizer")],
    }


def _text_components_for_forward(
    class_name: str,
    kwargs: Dict[str, Any],
    schema: Optional[Dict[str, Any]],
):
    if not schema:
        return None
    if class_name == "HunyuanVideo15Transformer3DModel":
        return _hunyuan15_text_components(kwargs, schema)

    components = schema.get("components", [])
    if len(components) <= 1:
        return None

    hidden_states = kwargs.get("encoder_hidden_states")
    batch = _shape_dim(hidden_states, 0)
    total = _shape_dim(hidden_states, 1)
    if batch is None or total is None:
        return None
    result = []
    start = 0
    for index, spec in enumerate(components):
        length = spec.get("length")
        end = total if length is None or index == len(components) - 1 else min(total, start + int(length))
        indices = list(range(start, end))
        item = dict(spec)
        item["indices_by_batch"] = [indices for _ in range(batch)]
        item["token_indices_by_batch"] = [list(range(len(indices))) for _ in range(batch)]
        result.append(item)
        start = end
    return result or None


def _hunyuan15_text_components(kwargs: Dict[str, Any], schema: Dict[str, Any]):
    torch = require_torch()
    mllm = kwargs.get("encoder_hidden_states")
    byt5 = kwargs.get("encoder_hidden_states_2")
    mllm_mask = kwargs.get("encoder_attention_mask")
    byt5_mask = kwargs.get("encoder_attention_mask_2")
    image_embeds = kwargs.get("image_embeds")
    if any(value is None for value in (mllm, byt5, mllm_mask, byt5_mask, image_embeds)):
        return None

    batch = int(mllm.shape[0])
    image_length = int(image_embeds.shape[1])
    is_t2v = bool(torch.all(image_embeds == 0))
    component_specs = {item["name"]: item for item in schema.get("components", [])}
    output = {
        name: {**spec, "indices_by_batch": [], "token_indices_by_batch": []}
        for name, spec in component_specs.items()
    }
    for batch_index in range(batch):
        mllm_active = torch.nonzero(mllm_mask[batch_index].bool(), as_tuple=False).flatten().tolist()
        byt5_active = torch.nonzero(byt5_mask[batch_index].bool(), as_tuple=False).flatten().tolist()
        valid_image = 0 if is_t2v else image_length
        byt5_start = valid_image
        mllm_start = byt5_start + len(byt5_active)
        output["byt5"]["indices_by_batch"].append(
            list(range(byt5_start, byt5_start + len(byt5_active)))
        )
        output["byt5"]["token_indices_by_batch"].append(byt5_active)
        output["mllm"]["indices_by_batch"].append(
            list(range(mllm_start, mllm_start + len(mllm_active)))
        )
        output["mllm"]["token_indices_by_batch"].append(mllm_active)
    return [output[name] for name in ("byt5", "mllm") if name in output]


def _tokenizer_length(tokenizer) -> Optional[int]:
    value = getattr(tokenizer, "model_max_length", None)
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if 0 < value < 1_000_000 else None


def _first_present(mapping: Dict[str, Any], *names: str):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _guidance_role(call_index: int, *, enabled: bool, conditions: int) -> str:
    if not enabled or conditions <= 1:
        return "conditional"
    if call_index == 0:
        return "conditional"
    if call_index == 1:
        return "unconditional"
    return f"guidance-{call_index}"


def _shape_dim(tensor, index: int) -> Optional[int]:
    if tensor is None or not hasattr(tensor, "shape") or len(tensor.shape) <= index:
        return None
    return int(tensor.shape[index])


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "flatten"):
        value = value.flatten()[0]
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _capture_metadata(pipeline) -> Dict[str, Any]:
    result: Dict[str, Any] = {"pipeline_class": pipeline.__class__.__name__}
    for component_name, component in _pipeline_model_components(pipeline):
        result[f"{component_name}_class"] = component.__class__.__name__
    for candidate in (pipeline, getattr(pipeline, "transformer", None), getattr(pipeline, "unet", None)):
        config = getattr(candidate, "config", None)
        model_id = getattr(config, "_name_or_path", None)
        if model_id:
            result["model_id"] = str(model_id)
            break
    try:
        result["attention_map_diffusers_version"] = importlib_metadata.version(
            "attention-map-diffusers"
        )
    except importlib_metadata.PackageNotFoundError:
        result["attention_map_diffusers_version"] = "source"
    try:
        result["diffusers_version"] = importlib_metadata.version("diffusers")
    except importlib_metadata.PackageNotFoundError:
        pass
    try:
        result["torch_version"] = importlib_metadata.version("torch")
    except importlib_metadata.PackageNotFoundError:
        pass
    return result
