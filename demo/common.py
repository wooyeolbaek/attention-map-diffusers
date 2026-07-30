from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


ALL_ATTENTION_RELATIONS = ("image->text", "text->image", "image->image", "text->text")


@dataclass(frozen=True)
class DemoModelConfig:
    key: str
    pipeline_class: str
    model_id: str
    torch_dtype: str
    prompt: str
    output_name: str
    default_steps: int = 15
    default_height: Optional[int] = None
    default_width: Optional[int] = None
    guidance_scale: Optional[float] = None
    extra_from_pretrained_kwargs: Dict[str, object] = field(default_factory=dict)
    extra_call_kwargs: Dict[str, object] = field(default_factory=dict)
    relations: tuple[str, ...] = ALL_ATTENTION_RELATIONS
    tokenizer_attribute: str = "tokenizer"


MODEL_CONFIGS: Dict[str, DemoModelConfig] = {
    "sd": DemoModelConfig(
        key="sd",
        pipeline_class="DiffusionPipeline",
        # model_id="stabilityai/stable-diffusion-2-1",
        model_id="sd2-community/stable-diffusion-2-1",
        torch_dtype="float16",
        prompt="A photo of a puppy wearing a hat.",
        output_name="sd2-1",
        default_steps=15,
        default_height=768,
        default_width=768,
        guidance_scale=None,
        relations=("image->text", "image->image"),
    ),
    "sdxl": DemoModelConfig(
        key="sdxl",
        pipeline_class="DiffusionPipeline",
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype="float16",
        prompt="A photo of a puppy wearing a hat.",
        output_name="sdxl",
        default_steps=15,
        default_height=1024,
        default_width=1024,
        guidance_scale=None,
        relations=("image->text", "image->image"),
    ),
    "sd3": DemoModelConfig(
        key="sd3",
        pipeline_class="StableDiffusion3Pipeline",
        model_id="stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype="bfloat16",
        prompt="A capybara holding a sign that reads Hello World.",
        output_name="sd3",
        default_steps=15,
        default_height=1024,
        default_width=1024,
        guidance_scale=4.5,
        tokenizer_attribute="tokenizer_3",
    ),
    "sd3-5": DemoModelConfig(
        key="sd3-5",
        pipeline_class="StableDiffusion3Pipeline",
        model_id="stabilityai/stable-diffusion-3.5-medium",
        torch_dtype="bfloat16",
        prompt="A capybara holding a sign that reads Hello World.",
        output_name="sd3-5",
        default_steps=15,
        default_height=1024,
        default_width=1024,
        guidance_scale=4.5,
        tokenizer_attribute="tokenizer_3",
    ),
    "flux-schnell": DemoModelConfig(
        key="flux-schnell",
        pipeline_class="FluxPipeline",
        model_id="black-forest-labs/FLUX.1-schnell",
        torch_dtype="bfloat16",
        prompt="A capybara holding a sign that reads Hello World.",
        output_name="flux-schnell",
        default_steps=4,
        default_height=1024,
        default_width=1024,
        guidance_scale=0.0,
        tokenizer_attribute="tokenizer_2",
    ),
    "flux-dev": DemoModelConfig(
        key="flux-dev",
        pipeline_class="FluxPipeline",
        model_id="black-forest-labs/FLUX.1-dev",
        torch_dtype="bfloat16",
        prompt="A capybara holding a sign that reads Hello World.",
        output_name="flux-dev",
        default_steps=15,
        default_height=1024,
        default_width=1024,
        guidance_scale=4.5,
        tokenizer_attribute="tokenizer_2",
    ),
    "sana": DemoModelConfig(
        key="sana",
        pipeline_class="SanaPipeline",
        model_id="Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        torch_dtype="float16",
        prompt="a cyberpunk cat with a neon sign that says Sana",
        output_name="sana",
        default_steps=20,
        default_height=1024,
        default_width=1024,
        guidance_scale=5.0,
        extra_from_pretrained_kwargs={"variant": "fp16"},
        relations=("image->text", "image->image"),
    ),
}


def model_keys(selection: str) -> List[str]:
    if selection == "all":
        return list(MODEL_CONFIGS)
    keys = [item.strip() for item in selection.split(",") if item.strip()]
    unknown = [key for key in keys if key not in MODEL_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown model key(s): {', '.join(unknown)}")
    return keys


def relation_list(config: DemoModelConfig, selection: str) -> List[str]:
    if selection == "all":
        return list(config.relations)
    selected = [item.strip() for item in selection.split(",") if item.strip()]
    return [relation for relation in selected if relation in config.relations]


def parse_indices(value: Optional[str]):
    if value is None or value == "all":
        return None
    value = value.strip()
    if value == "center":
        return value
    if value.startswith("sample:"):
        try:
            count = int(value.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError("index samples must use 'sample:N' with a positive integer") from exc
        if count <= 0:
            raise ValueError("index samples must use 'sample:N' with a positive integer")
        return value
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise ValueError("indices must be 'all', 'center', 'sample:N', or a comma-separated list")
    return indices


def parse_heads(value: str):
    """Parse the CLI head selector into the capture API's expected type."""
    value = value.strip().lower()
    if value in {"all", "mean"}:
        return value
    heads = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not heads or any(head < 0 for head in heads):
        raise ValueError("heads must be 'mean', 'all', or a comma-separated list of non-negative indices")
    return list(dict.fromkeys(heads))
