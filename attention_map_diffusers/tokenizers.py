from __future__ import annotations


def text_tokenizers(pipeline):
    """Return tokenizer input matching the denoiser's actual text sequence.

    Multi-encoder pipelines return a component-name mapping. HunyuanVideo 1.5
    adapters reproduce its glyph extraction and chat-template crop so token
    labels stay aligned with the reordered attention sequence.
    """

    transformer_name = getattr(
        getattr(pipeline, "transformer", None), "__class__", type(None)
    ).__name__
    if transformer_name == "SD3Transformer2DModel":
        return {
            "clip-l+clip-g-fused": getattr(pipeline, "tokenizer", None),
            "t5": getattr(pipeline, "tokenizer_3", None),
        }
    if transformer_name == "HunyuanVideo15Transformer3DModel":
        return {
            "byt5": _HunyuanByT5Tokenizer(pipeline),
            "mllm": _HunyuanMLLMTokenizer(pipeline),
        }
    if transformer_name == "FluxTransformer2DModel":
        return getattr(pipeline, "tokenizer_2", None)
    if transformer_name in {
        "Flux2Transformer2DModel",
        "ZImageTransformer2DModel",
        "PRXTransformer2DModel",
        "SanaTransformer2DModel",
        "CogVideoXTransformer3DModel",
        "WanTransformer3DModel",
    }:
        return getattr(pipeline, "tokenizer", None)
    return getattr(pipeline, "tokenizer_2", None) or getattr(pipeline, "tokenizer", None)


class _HunyuanMLLMTokenizer:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.tokenizer = pipeline.tokenizer

    def __call__(self, prompts, *, max_length, **kwargs):
        from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5 import (
            format_text_input,
        )

        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        formatted = format_text_input(prompt_list, self.pipeline.system_message)
        crop = int(self.pipeline.prompt_template_encode_start_idx)
        encoded = self.tokenizer.apply_chat_template(
            formatted,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding="max_length",
            max_length=int(self.pipeline.tokenizer_max_length) + crop,
            truncation=True,
            return_tensors="pt",
        )
        return _cropped_encoding(encoded.input_ids, encoded.attention_mask, crop, max_length)

    def convert_ids_to_tokens(self, token_ids):
        return self.tokenizer.convert_ids_to_tokens(token_ids)


class _HunyuanByT5Tokenizer:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.tokenizer = pipeline.tokenizer_2

    def __call__(self, prompts, *, max_length, **kwargs):
        import torch
        from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5 import (
            extract_glyph_texts,
        )

        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        glyphs = [extract_glyph_texts(prompt) for prompt in prompt_list]
        encoded = self.tokenizer(
            [glyph or "" for glyph in glyphs],
            padding="max_length",
            max_length=int(self.pipeline.tokenizer_2_max_length),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        masks = encoded.attention_mask.clone()
        for index, glyph in enumerate(glyphs):
            if glyph is None:
                masks[index] = torch.zeros_like(masks[index])
        return _cropped_encoding(encoded.input_ids, masks, 0, max_length)

    def convert_ids_to_tokens(self, token_ids):
        return self.tokenizer.convert_ids_to_tokens(token_ids)


def _cropped_encoding(input_ids, attention_mask, start: int, length: int):
    import torch

    input_ids = input_ids[:, start : start + length]
    attention_mask = attention_mask[:, start : start + length]
    missing = int(length) - int(input_ids.shape[1])
    if missing > 0:
        input_ids = torch.nn.functional.pad(input_ids, (0, missing), value=0)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, missing), value=0)
    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
    }


__all__ = ["text_tokenizers"]
