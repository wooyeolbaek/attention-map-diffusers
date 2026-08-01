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

from .base import BaseCaptureProcessor


def _require_diffusers_attention():
    try:
        from diffusers.models.attention_processor import Attention
    except ImportError as exc:
        raise ImportError(
            "attention-map-diffusers requires diffusers for model capture. "
            "Install diffusers before using AttentionCapture with a pipeline."
        ) from exc
    return Attention


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
            layer_id=self.layer_id,
        )
        context = current_context()
        capture_metadata = dict(context.metadata)
        if _is_text_self_attention(self.layer_id, attention_type):
            capture_metadata["text_components"] = _local_text_components(
                context,
                component_name="mllm",
                batch_size=int(query.shape[0]),
                token_count=int(query.shape[-2]),
            )
            capture_metadata["text_attention_stage"] = "token-refiner"
        self._capture(
            attention_type=attention_type,
            query=query,
            key=key,
            layout=layout,
            attention_mask=attention_mask,
            timestep=context.timestep,
            metadata=capture_metadata,
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


class CaptureFlux2AttnProcessor(BaseCaptureProcessor):
    """Capture FLUX.2 double-stream attention without changing its backend."""

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        torch = require_torch()
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.embeddings import apply_rotary_emb

        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        image_tokens = int(query.shape[1])
        text_tokens = 0

        if encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
            encoder_key = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
            encoder_value = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)
            text_tokens = int(encoder_query.shape[1])
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        if encoder_hidden_states is not None:
            context = current_context()
            self._capture(
                attention_type="joint",
                query=query.transpose(1, 2),
                key=key.transpose(1, 2),
                layout=_joint_layout_text_image(
                    text_tokens,
                    image_tokens,
                    context.image_grid or infer_2d_grid(image_tokens),
                ),
                attention_mask=attention_mask,
                timestep=context.timestep,
                metadata=context.metadata,
            )

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=getattr(self.original_processor, "_attention_backend", None),
            parallel_config=getattr(self.original_processor, "_parallel_config", None),
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [text_tokens, hidden_states.shape[1] - text_tokens], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


class CaptureFlux2ParallelSelfAttnProcessor(BaseCaptureProcessor):
    """Capture FLUX.2 single-stream attention and preserve fused MLP output."""

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        torch = require_torch()
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.embeddings import apply_rotary_emb

        projected = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            projected,
            [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor],
            dim=-1,
        )
        query, key, value = qkv.chunk(3, dim=-1)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        context = current_context()
        total_tokens = int(query.shape[1])
        text_tokens = int(context.text_tokens or 0)
        if text_tokens > total_tokens:
            text_tokens = 0
        image_tokens = total_tokens - text_tokens
        self._capture(
            attention_type="joint",
            query=query.transpose(1, 2),
            key=key.transpose(1, 2),
            layout=_joint_layout_text_image(
                text_tokens,
                image_tokens,
                context.image_grid or infer_2d_grid(image_tokens),
            ),
            attention_mask=attention_mask,
            timestep=context.timestep,
            metadata=context.metadata,
        )

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=getattr(self.original_processor, "_attention_backend", None),
            parallel_config=getattr(self.original_processor, "_parallel_config", None),
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        return attn.to_out(torch.cat([hidden_states, mlp_hidden_states], dim=-1))


class CaptureZSingleStreamAttnProcessor(BaseCaptureProcessor):
    """Capture Z-Image refiner and unified-sequence attention."""

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        freqs_cis=None,
    ):
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if freqs_cis is not None:
            query = _apply_z_image_rotary(query, freqs_cis)
            key = _apply_z_image_rotary(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        context = current_context()
        token_count = int(query.shape[1])
        if ".noise_refiner." in self.layer_id:
            image_grid = context.image_grid or infer_2d_grid(token_count)
            image = ModalityRange("image", 0, token_count, image_grid)
            layout = make_layout([image], [image])
            attention_type = "self"
        elif ".context_refiner." in self.layer_id or ".siglip_refiner." in self.layer_id:
            text = ModalityRange("text", 0, token_count, None)
            layout = make_layout([text], [text])
            attention_type = "self"
        else:
            image_tokens = int(context.image_tokens or 0)
            text_tokens = int(context.text_tokens or 0)
            if image_tokens + text_tokens != token_count:
                image_tokens = min(image_tokens, token_count)
                text_tokens = token_count - image_tokens
            layout = _joint_layout_image_text(
                image_tokens,
                text_tokens,
                context.image_grid or infer_2d_grid(image_tokens),
            )
            attention_type = "joint"

        self._capture(
            attention_type=attention_type,
            query=query.transpose(1, 2),
            key=key.transpose(1, 2),
            layout=layout,
            attention_mask=attention_mask,
            timestep=context.timestep,
            metadata=context.metadata,
        )
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=getattr(self.original_processor, "_attention_backend", None),
            parallel_config=getattr(self.original_processor, "_parallel_config", None),
        )
        hidden_states = hidden_states.flatten(2, 3).to(dtype)
        hidden_states = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class CapturePRXAttnProcessor2_0(BaseCaptureProcessor):
    """Capture PRX image queries against text+image keys."""

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
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_prx import apply_rope

        if encoder_hidden_states is None:
            raise ValueError("PRX attention requires encoder_hidden_states containing text tokens.")
        image_qkv = attn.img_qkv_proj(hidden_states)
        batch_size, image_tokens, _ = image_qkv.shape
        image_qkv = image_qkv.reshape(batch_size, image_tokens, 3, attn.heads, attn.head_dim)
        image_qkv = image_qkv.permute(2, 0, 3, 1, 4)
        image_query, image_key, image_value = image_qkv[0], image_qkv[1], image_qkv[2]
        image_query = attn.norm_q(image_query)
        image_key = attn.norm_k(image_key)

        text_kv = attn.txt_kv_proj(encoder_hidden_states)
        text_tokens = int(text_kv.shape[1])
        text_kv = text_kv.reshape(batch_size, text_tokens, 2, attn.heads, attn.head_dim)
        text_kv = text_kv.permute(2, 0, 3, 1, 4)
        text_key, text_value = text_kv[0], text_kv[1]
        text_key = attn.norm_added_k(text_key)
        if image_rotary_emb is not None:
            image_query = apply_rope(image_query, image_rotary_emb)
            image_key = apply_rope(image_key, image_rotary_emb)

        key = torch.cat((text_key, image_key), dim=2)
        value = torch.cat((text_value, image_value), dim=2)
        attention_mask_tensor = None
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[-1] != text_tokens:
                raise ValueError(f"Unsupported PRX attention_mask shape: {attention_mask.shape}")
            image_mask = torch.ones(
                (batch_size, image_tokens), dtype=torch.bool, device=image_query.device
            )
            joint_mask = torch.cat(
                [attention_mask.to(device=image_query.device, dtype=torch.bool), image_mask], dim=-1
            )
            attention_mask_tensor = joint_mask[:, None, None, :].expand(
                -1, attn.heads, image_tokens, -1
            )

        context = current_context()
        self._capture(
            attention_type="joint",
            query=image_query,
            key=key,
            layout=_image_query_text_image_key_layout(
                image_tokens,
                text_tokens,
                context.image_grid or infer_2d_grid(image_tokens),
            ),
            attention_mask=attention_mask_tensor,
            timestep=context.timestep,
            metadata=context.metadata,
        )

        output = dispatch_attention_fn(
            image_query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask_tensor,
            backend=getattr(self.original_processor, "_attention_backend", None),
            parallel_config=getattr(self.original_processor, "_parallel_config", None),
        )
        output = output.reshape(batch_size, image_tokens, attn.heads * attn.head_dim)
        output = attn.to_out[0](output)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output


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

        self.collector.add_record(
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
    "Flux2AttnProcessor": CaptureFlux2AttnProcessor,
    "Flux2ParallelSelfAttnProcessor": CaptureFlux2ParallelSelfAttnProcessor,
    "ZSingleStreamAttnProcessor": CaptureZSingleStreamAttnProcessor,
    "PRXAttnProcessor2_0": CapturePRXAttnProcessor2_0,
    "SanaAttnProcessor2_0": CaptureSanaAttnProcessor2_0,
    "SanaLinearAttnProcessor2_0": CaptureSanaLinearAttnProcessor2_0,
}

from .video import VIDEO_PROCESSOR_NAMES

SUPPORTED_PROCESSOR_NAMES.update(VIDEO_PROCESSOR_NAMES)


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
    layer_id: Optional[str] = None,
):
    context = current_context()
    if _is_text_self_attention(layer_id, attention_type):
        text_range = ModalityRange("text", 0, query_tokens, None)
        return make_layout([text_range], [text_range])
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


def _is_text_self_attention(layer_id: Optional[str], attention_type: str) -> bool:
    """Identify denoiser-internal attention that refines text tokens only."""
    return (
        attention_type == "self"
        and layer_id is not None
        and ".context_embedder.token_refiner." in layer_id
    )


def _local_text_components(
    context,
    *,
    component_name: str,
    batch_size: int,
    token_count: int,
):
    """Translate combined-sequence metadata back to a local encoder sequence."""
    for component in context.text_components or []:
        if str(component.get("name")) != component_name:
            continue
        token_batches = component.get("token_indices_by_batch") or []
        if not token_batches:
            break
        local_batches = [
            [int(index) for index in batch if 0 <= int(index) < token_count]
            for batch in token_batches
        ]
        item = dict(component)
        item["indices_by_batch"] = local_batches
        item["token_indices_by_batch"] = [list(batch) for batch in local_batches]
        return [item]

    indices = list(range(token_count))
    return [
        {
            "name": component_name,
            "fusion": "sequence",
            "indices_by_batch": [list(indices) for _ in range(batch_size)],
            "token_indices_by_batch": [list(indices) for _ in range(batch_size)],
        }
    ]


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


def _image_query_text_image_key_layout(
    image_tokens: int,
    text_tokens: int,
    image_grid: Optional[TokenGrid],
):
    query_image = ModalityRange("image", 0, image_tokens, image_grid)
    key_text = ModalityRange("text", 0, text_tokens, None)
    key_image = ModalityRange("image", text_tokens, text_tokens + image_tokens, image_grid)
    return make_layout([query_image], [key_text, key_image])


def _apply_z_image_rotary(value, freqs_cis):
    torch = require_torch()
    with torch.amp.autocast("cuda", enabled=False):
        complex_value = torch.view_as_complex(value.float().reshape(*value.shape[:-1], -1, 2))
        rotated = torch.view_as_real(complex_value * freqs_cis.unsqueeze(2)).flatten(3)
        return rotated.type_as(value)
