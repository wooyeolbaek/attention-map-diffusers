from __future__ import annotations

from typing import Optional

from attention_map_diffusers.capture.compute import require_torch
from attention_map_diffusers.capture.context import current_context
from attention_map_diffusers.capture.records import ModalityRange, TokenGrid, make_layout

from .base import BaseCaptureProcessor


def _video_grid(token_count: int) -> Optional[TokenGrid]:
    grid = current_context().video_grid
    if grid is not None and grid.spatial_tokens == token_count:
        return grid
    return None


def _joint_text_video_layout(text_tokens: int, video_tokens: int, grid: Optional[TokenGrid]):
    text = ModalityRange("text", 0, text_tokens)
    video = ModalityRange("video", text_tokens, text_tokens + video_tokens, grid)
    ranges = [text, video] if text_tokens else [ModalityRange("video", 0, video_tokens, grid)]
    return make_layout(ranges, ranges)


def _joint_video_text_layout(video_tokens: int, text_tokens: int, grid: Optional[TokenGrid]):
    video = ModalityRange("video", 0, video_tokens, grid)
    text = ModalityRange("text", video_tokens, video_tokens + text_tokens)
    ranges = [video, text] if text_tokens else [video]
    return make_layout(ranges, ranges)


def _video_cross_text_layout(video_tokens: int, text_tokens: int, grid: Optional[TokenGrid]):
    return make_layout(
        [ModalityRange("video", 0, video_tokens, grid)],
        [ModalityRange("text", 0, text_tokens)],
    )


def _wan_qkv(attn, hidden_states, encoder_hidden_states):
    encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states
    if getattr(attn, "fused_projections", False):
        if not attn.is_cross_attention:
            return attn.to_qkv(hidden_states).chunk(3, dim=-1)
        query = attn.to_q(hidden_states)
        key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
        return query, key, value
    return (
        attn.to_q(hidden_states),
        attn.to_k(encoder_hidden_states),
        attn.to_v(encoder_hidden_states),
    )


def _wan_added_kv(attn, encoder_hidden_states):
    if getattr(attn, "fused_projections", False):
        return attn.to_added_kv(encoder_hidden_states).chunk(2, dim=-1)
    return attn.add_k_proj(encoder_hidden_states), attn.add_v_proj(encoder_hidden_states)


class CaptureCogVideoXAttnProcessor2_0(BaseCaptureProcessor):
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        import torch.nn.functional as functional

        text_tokens = int(encoder_hidden_states.shape[1])
        hidden_states = require_torch().cat([encoder_hidden_states, hidden_states], dim=1)
        batch_size, sequence_length, _ = hidden_states.shape
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if getattr(attn, "fused_projections", False):
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_tokens:] = apply_rotary_emb(query[:, :, text_tokens:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_tokens:] = apply_rotary_emb(key[:, :, text_tokens:], image_rotary_emb)

        video_tokens = int(query.shape[-2]) - text_tokens
        context = current_context()
        self._capture(
            attention_type="joint",
            query=query,
            key=key,
            layout=_joint_text_video_layout(text_tokens, video_tokens, _video_grid(video_tokens)),
            attention_mask=attention_mask,
            timestep=context.timestep,
            metadata=context.metadata,
        )

        hidden_states = functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_tokens, hidden_states.shape[1] - text_tokens], dim=1
        )
        return hidden_states, encoder_hidden_states


class CaptureHunyuanVideoAttnProcessor2_0(BaseCaptureProcessor):
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

        video_tokens = int(hidden_states.shape[1])
        text_tokens = int(encoder_hidden_states.shape[1]) if encoder_hidden_states is not None else 0
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :video_tokens], image_rotary_emb, sequence_dim=1),
                        query[:, video_tokens:],
                    ],
                    dim=1,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :video_tokens], image_rotary_emb, sequence_dim=1),
                        key[:, video_tokens:],
                    ],
                    dim=1,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1))
            encoder_key = attn.add_k_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1))
            encoder_value = attn.add_v_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1))
            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)
            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

        context = current_context()
        self._capture(
            attention_type="joint" if text_tokens else "self",
            query=query.transpose(1, 2),
            key=key.transpose(1, 2),
            layout=_joint_video_text_layout(video_tokens, text_tokens, _video_grid(video_tokens)),
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
        ).flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
                [video_tokens, text_tokens], dim=1
            )
            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states


class CaptureHunyuanVideo15AttnProcessor2_0(BaseCaptureProcessor):
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        torch = require_torch()
        import torch.nn.functional as functional
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        video_tokens = int(hidden_states.shape[1])
        text_tokens = int(encoder_hidden_states.shape[1]) if encoder_hidden_states is not None else 0
        query = attn.norm_q(attn.to_q(hidden_states).unflatten(2, (attn.heads, -1)))
        key = attn.norm_k(attn.to_k(hidden_states).unflatten(2, (attn.heads, -1)))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        if encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1))
            encoder_key = attn.add_k_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1))
            encoder_value = attn.add_v_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1))
            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)
            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

        batch_size, sequence_length = int(query.shape[0]), int(query.shape[1])
        attention_mask = functional.pad(
            attention_mask, (sequence_length - attention_mask.shape[1], 0), value=True
        ).bool()
        mask = attention_mask.view(batch_size, 1, 1, sequence_length).repeat(
            1, 1, sequence_length, 1
        )
        attention_mask = (mask & mask.transpose(2, 3)).bool()

        context = current_context()
        self._capture(
            attention_type="joint",
            query=query.transpose(1, 2),
            key=key.transpose(1, 2),
            layout=_joint_video_text_layout(video_tokens, text_tokens, _video_grid(video_tokens)),
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
        ).flatten(2, 3).to(query.dtype)
        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            [video_tokens, text_tokens], dim=1
        )
        if getattr(attn, "to_out", None) is not None:
            hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if getattr(attn, "to_add_out", None) is not None:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states


class CaptureWanAttnProcessor(BaseCaptureProcessor):
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
    ):
        torch = require_torch()
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _wan_qkv(attn, hidden_states, encoder_hidden_states)
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            def apply_rotary(value, freqs_cos, freqs_sin):
                first, second = value.unflatten(-1, (-1, 2)).unbind(-1)
                cosine = freqs_cos[..., 0::2]
                sine = freqs_sin[..., 1::2]
                output = torch.empty_like(value)
                output[..., 0::2] = first * cosine - second * sine
                output[..., 1::2] = first * sine + second * cosine
                return output.type_as(value)

            query = apply_rotary(query, *rotary_emb)
            key = apply_rotary(key, *rotary_emb)

        video_tokens = int(query.shape[1])
        key_tokens = int(key.shape[1])
        context = current_context()
        capture_mask = attention_mask
        if capture_mask is not None and capture_mask.ndim == 2:
            capture_mask = capture_mask[:, None, None, :]
        if encoder_hidden_states is None:
            video = ModalityRange("video", 0, video_tokens, _video_grid(video_tokens))
            layout = make_layout([video], [video])
            attention_type = "self"
        else:
            layout = _video_cross_text_layout(video_tokens, key_tokens, _video_grid(video_tokens))
            attention_type = "cross"
        self._capture(
            attention_type=attention_type,
            query=query.transpose(1, 2),
            key=key.transpose(1, 2),
            layout=layout,
            attention_mask=capture_mask,
            timestep=context.timestep,
            metadata=context.metadata,
        )

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _wan_added_kv(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img).unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))
            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=getattr(self.original_processor, "_attention_backend", None),
                parallel_config=None,
            ).flatten(2, 3).type_as(query)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=getattr(self.original_processor, "_attention_backend", None),
            parallel_config=(
                getattr(self.original_processor, "_parallel_config", None)
                if encoder_hidden_states is None
                else None
            ),
        ).flatten(2, 3).type_as(query)
        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img
        return attn.to_out[1](attn.to_out[0](hidden_states))


VIDEO_PROCESSOR_NAMES = {
    "CogVideoXAttnProcessor2_0": CaptureCogVideoXAttnProcessor2_0,
    "FusedCogVideoXAttnProcessor2_0": CaptureCogVideoXAttnProcessor2_0,
    "HunyuanVideoAttnProcessor2_0": CaptureHunyuanVideoAttnProcessor2_0,
    "HunyuanVideo15AttnProcessor2_0": CaptureHunyuanVideo15AttnProcessor2_0,
    "WanAttnProcessor": CaptureWanAttnProcessor,
    "WanAttnProcessor2_0": CaptureWanAttnProcessor,
}


__all__ = [
    "CaptureCogVideoXAttnProcessor2_0",
    "CaptureHunyuanVideo15AttnProcessor2_0",
    "CaptureHunyuanVideoAttnProcessor2_0",
    "CaptureWanAttnProcessor",
    "VIDEO_PROCESSOR_NAMES",
]
