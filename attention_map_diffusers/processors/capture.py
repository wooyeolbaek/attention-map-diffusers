from __future__ import annotations

from typing import Any, Dict, Optional

from attention_map_diffusers.capture.compute import (
    output_head_count,
    reduce_heads,
    require_torch,
    tensor_to_device,
)
from attention_map_diffusers.capture.config import normalize_indices
from attention_map_diffusers.capture.context import current_context
from attention_map_diffusers.capture.records import AttentionRecord, ModalityRange, TokenGrid, infer_2d_grid, make_layout


def _require_diffusers_attention():
    try:
        from diffusers.models.attention_processor import Attention
    except ImportError as exc:
        raise ImportError(
            "attention-map-diffusers requires diffusers for model capture. "
            "Install diffusers before using AttentionCapture with a pipeline."
        ) from exc
    return Attention


class BaseCaptureProcessor:
    def __init__(self, collector, layer_id: str, original_processor=None):
        self.collector = collector
        self.layer_id = layer_id
        self.original_processor = original_processor

    def _capture(self, **kwargs) -> None:
        self.collector.capture_qk(layer_id=self.layer_id, **kwargs)


class CaptureAttnProcessor2_0(BaseCaptureProcessor):
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        torch = require_torch()
        import torch.nn.functional as F

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        input_height = input_width = None
        if input_ndim == 4:
            batch_size, channel, input_height, input_width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, input_height * input_width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        attention_type = "self" if encoder_hidden_states is None else "cross"
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        layout = _standard_layout(
            query_tokens=query.shape[-2],
            key_tokens=key.shape[-2],
            attention_type=attention_type,
            input_height=input_height,
            input_width=input_width,
        )
        context = current_context()
        self._capture(
            attention_type=attention_type,
            query=query,
            key=key,
            layout=layout,
            attention_mask=attention_mask,
            timestep=context.timestep,
            metadata=context.metadata,
        )

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, input_height, input_width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class CaptureAttnProcessor(CaptureAttnProcessor2_0):
    pass


class CaptureJointAttnProcessor2_0(BaseCaptureProcessor):
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        *args,
        **kwargs,
    ):
        torch = require_torch()
        import torch.nn.functional as F

        batch_size = hidden_states.shape[0]
        residual = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        image_tokens = query.shape[-2]
        text_tokens = 0

        if encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_key = encoder_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_value = encoder_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)
            text_tokens = encoder_query.shape[-2]

        context = current_context()
        image_grid = context.image_grid or infer_2d_grid(image_tokens)
        layout = _joint_layout_image_text(image_tokens, text_tokens, image_grid)
        self._capture(
            attention_type="joint",
            query=query,
            key=key,
            layout=layout,
            attention_mask=attention_mask,
            timestep=context.timestep,
            metadata=context.metadata,
        )

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


class CaptureFluxAttnProcessor2_0(BaseCaptureProcessor):
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        **kwargs,
    ):
        torch = require_torch()
        import torch.nn.functional as F

        batch_size = hidden_states.shape[0]
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        image_tokens = query.shape[-2]
        text_tokens = 0

        if encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_key = encoder_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_value = encoder_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=2)
            key = torch.cat([encoder_key, key], dim=2)
            value = torch.cat([encoder_value, value], dim=2)
            text_tokens = encoder_query.shape[-2]

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        context = current_context()
        image_grid = context.image_grid or infer_2d_grid(image_tokens)
        if encoder_hidden_states is not None:
            layout = _joint_layout_text_image(text_tokens, image_tokens, image_grid)
            self._capture(
                attention_type="joint",
                query=query,
                key=key,
                layout=layout,
                attention_mask=attention_mask,
                timestep=context.timestep,
                metadata=context.metadata,
            )

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        return hidden_states


class CaptureSanaAttnProcessor2_0(BaseCaptureProcessor):
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        require_torch()
        import torch.nn.functional as F

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        attention_type = "self" if encoder_hidden_states is None else "cross"
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        layout = _standard_layout(
            query_tokens=query.shape[-2],
            key_tokens=key.shape[-2],
            attention_type=attention_type,
        )
        context = current_context()
        self._capture(
            attention_type=attention_type,
            query=query,
            key=key,
            layout=layout,
            attention_mask=attention_mask,
            timestep=context.timestep,
            metadata=context.metadata,
        )

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class CaptureSanaLinearAttnProcessor2_0(BaseCaptureProcessor):
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        torch = require_torch()
        import torch.nn.functional as F

        original_dtype = hidden_states.dtype

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        batch_size = hidden_states.shape[0]
        head_dim = key.shape[-1] // attn.heads
        query_for_map = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_for_map = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        query_for_map = F.relu(query_for_map)
        key_for_map = F.relu(key_for_map)
        self._capture_linear_map(attn, query_for_map, key_for_map)

        query = query.transpose(1, 2).unflatten(1, (attn.heads, -1))
        key = key.transpose(1, 2).unflatten(1, (attn.heads, -1)).transpose(2, 3)
        value = value.transpose(1, 2).unflatten(1, (attn.heads, -1))

        query = F.relu(query)
        key = F.relu(key)

        query, key, value = query.float(), key.float(), value.float()

        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1.0)
        scores = torch.matmul(value, key)
        hidden_states = torch.matmul(scores, query)

        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + 1e-15)
        hidden_states = hidden_states.flatten(1, 2).transpose(1, 2)
        hidden_states = hidden_states.to(original_dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if original_dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states

    def _capture_linear_map(self, attn, query, key) -> None:
        if "image->image" not in self.collector.config.relations:
            return
        if not self.collector.enabled_for(self.layer_id):
            return

        torch = require_torch()
        context = current_context()
        image_grid = context.image_grid or infer_2d_grid(query.shape[-2])
        query_selection = self.collector.config.query_indices
        if self.collector.config.relation_query_indices is not None:
            query_selection = self.collector.config.relation_query_indices.get("image->image", query_selection)
        query_indices = normalize_indices(query_selection, query.shape[-2], image_grid)
        key_selection = (
            self.collector.config.key_indices
            if self.collector.config.key_indices is not None
            else self.collector.config.tokens
        )
        if self.collector.config.relation_key_indices is not None:
            key_selection = self.collector.config.relation_key_indices.get("image->image", key_selection)
        key_indices = normalize_indices(key_selection, key.shape[-2], image_grid)

        selected_query = query
        if query_indices is not None:
            selected_query = selected_query.index_select(-2, torch.tensor(query_indices, device=query.device))

        output_key_count = key.shape[-2] if key_indices is None else len(key_indices)
        map_elements = (
            selected_query.shape[0]
            * output_head_count(self.collector.config.heads, selected_query.shape[1])
            * selected_query.shape[-2]
            * output_key_count
        )
        if map_elements > self.collector.config.max_map_elements and not self.collector.config.allow_large_maps:
            raise RuntimeError(
                f"Refusing to materialize {map_elements:,} linear attention values for "
                f"{self.layer_id} image->image."
            )

        # Use one numerical path for numerator and denominator. Computing the
        # denominator through a separate reduction can diverge from CUDA's
        # matmul accumulation for near-degenerate FP16 rows. Softmax attention
        # has the same full-key normalization rule, so key selection happens
        # only after the complete linear-attention row is normalized.
        full_numerator = torch.matmul(selected_query.float(), key.float().transpose(-2, -1))
        normalizer = full_numerator.sum(dim=-1, keepdim=True).clamp_min(1e-15)
        data = full_numerator / normalizer
        if key_indices is not None:
            key_index = torch.tensor(key_indices, device=data.device, dtype=torch.long)
            data = data.index_select(-1, key_index)
        data = reduce_heads(data, self.collector.config.heads).to(query.dtype)
        data = tensor_to_device(data, self.collector.config.offload, self.collector.config.dtype)

        metadata = dict(context.metadata)
        metadata["attention_kernel"] = "relu_linear"
        if query_indices is not None:
            metadata["query_indices"] = tuple(query_indices)
        if key_indices is not None:
            metadata["key_indices"] = tuple(key_indices)

        self.collector.records.append(
            AttentionRecord(
                layer_id=self.layer_id,
                relation="image->image",
                attention_type="linear-self",
                q_range=(0, query.shape[-2]),
                k_range=(0, key.shape[-2]),
                q_shape=image_grid,
                k_shape=image_grid,
                heads=attn.heads,
                dtype=str(query.dtype).replace("torch.", ""),
                device=str(query.device),
                step_index=self.collector.step_index,
                timestep=context.timestep,
                data=data,
                metadata=metadata,
            )
        )


SUPPORTED_PROCESSOR_NAMES = {
    "AttnProcessor": CaptureAttnProcessor,
    "AttnProcessor2_0": CaptureAttnProcessor2_0,
    "JointAttnProcessor2_0": CaptureJointAttnProcessor2_0,
    "FluxAttnProcessor": CaptureFluxAttnProcessor2_0,
    "FluxAttnProcessor2_0": CaptureFluxAttnProcessor2_0,
    "SanaAttnProcessor2_0": CaptureSanaAttnProcessor2_0,
    "SanaLinearAttnProcessor2_0": CaptureSanaLinearAttnProcessor2_0,
}


def make_capture_processor(original_processor, collector, layer_id: str):
    cls = SUPPORTED_PROCESSOR_NAMES.get(original_processor.__class__.__name__)
    if cls is None:
        return None
    return cls(collector=collector, layer_id=layer_id, original_processor=original_processor)


def _standard_layout(
    *,
    query_tokens: int,
    key_tokens: int,
    attention_type: str,
    input_height: Optional[int] = None,
    input_width: Optional[int] = None,
):
    context = current_context()
    if input_height is not None and input_width is not None:
        image_grid = TokenGrid(height=int(input_height), width=int(input_width))
    else:
        image_grid = context.image_grid or infer_2d_grid(query_tokens)

    if attention_type == "self":
        image_range = ModalityRange("image", 0, query_tokens, image_grid)
        return make_layout([image_range], [image_range])

    return make_layout(
        [ModalityRange("image", 0, query_tokens, image_grid)],
        [ModalityRange("text", 0, key_tokens, None)],
    )


def _joint_layout_image_text(image_tokens: int, text_tokens: int, image_grid: Optional[TokenGrid]):
    image = ModalityRange("image", 0, image_tokens, image_grid)
    text = ModalityRange("text", image_tokens, image_tokens + text_tokens, None)
    ranges = [image]
    if text_tokens:
        ranges.append(text)
    return make_layout(ranges, ranges)


def _joint_layout_text_image(text_tokens: int, image_tokens: int, image_grid: Optional[TokenGrid]):
    text = ModalityRange("text", 0, text_tokens, None)
    image = ModalityRange("image", text_tokens, text_tokens + image_tokens, image_grid)
    ranges = [text, image] if text_tokens else [ModalityRange("image", 0, image_tokens, image_grid)]
    return make_layout(ranges, ranges)
