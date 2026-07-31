from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from .capture.collector import AttentionCollector, AttentionMaps
from .capture.config import CaptureConfig
from .capture.context import CaptureRuntimeContext, infer_grid_from_hidden_states, infer_grid_from_img_ids, pop_context, push_context
from .capture.compute import require_torch
from .processors import make_capture_processor


class AttentionCapture:
    """Context manager for relation-aware Diffusers attention-map capture.

    The default ``mode='fast'`` keeps Diffusers' SDPA/Flash Attention path for
    generation and stores q/k tensors for post-hoc attention-map recomputation.
    ``mode='exact'`` and ``mode='stream'`` materialize selected maps during the
    attention call and are intended for small or heavily selected maps.
    """

    def __init__(
        self,
        pipeline,
        *,
        relations="image->text",
        layers="all",
        timesteps="all",
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
        allow_large_maps=False,
        capture_unconditional=True,
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
            allow_large_maps=allow_large_maps,
            capture_unconditional=capture_unconditional,
        )
        self.collector = AttentionCollector(self.config)
        self._processors: List[Tuple[Any, Any]] = []
        self._hooks: List[Any] = []
        self._installed = False

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

    def compute(self) -> AttentionMaps:
        return self.collector.compute()

    def save(self, base_dir: str, **kwargs):
        return self.compute().save(base_dir, **kwargs)

    def install(self):
        require_torch()
        if self._installed:
            return self.pipeline
        self.collector.clear()
        self._install_processors()
        self._install_context_hooks()
        self._installed = True
        return self.pipeline

    def restore(self) -> None:
        for handle in reversed(self._hooks):
            handle.remove()
        self._hooks.clear()

        for module, original_processor in reversed(self._processors):
            _set_processor(module, original_processor)
        self._processors.clear()
        self._installed = False

    def _install_processors(self) -> None:
        components = _pipeline_model_components(self.pipeline)
        for component_name, component in components:
            for name, module in component.named_modules():
                processor = getattr(module, "processor", None)
                if processor is None:
                    continue
                layer_id = f"{component_name}.{name}" if name else component_name
                capture_processor = make_capture_processor(processor, self.collector, layer_id)
                if capture_processor is None:
                    continue
                self._processors.append((module, processor))
                _set_processor(module, capture_processor)

    def _install_context_hooks(self) -> None:
        for component_name, component in _pipeline_model_components(self.pipeline):
            self._register_context_hooks(component, component_name)
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
                }:
                    self._register_context_hooks(module, f"{component_name}.{name}" if name else component_name)

    def _register_context_hooks(self, module, layer_id: str) -> None:
        def pre_hook(mod, args, kwargs):
            context = _context_from_forward(mod, args, kwargs, layer_id)
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
        self._hooks.append(module.register_forward_hook(post_hook, with_kwargs=True))


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


def _context_from_forward(module, args, kwargs: Dict[str, Any], layer_id: str) -> CaptureRuntimeContext:
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
        metadata=metadata,
    )


def _first_present(mapping: Dict[str, Any], *names: str):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


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
