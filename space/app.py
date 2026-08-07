from __future__ import annotations

import gc
import logging
import os
import re
import secrets
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from attention_map_diffusers import AttentionCapture, VisualizationConfig, text_tokenizers


MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
MODEL_VARIANT = os.getenv("MODEL_VARIANT", "fp16").strip()
LOAD_VARIANT = None if MODEL_VARIANT.lower() in {"", "none"} else MODEL_VARIANT
DEVICE = "cuda"
RELATIONS = ("image->text", "image->image")
MAX_CAPTURE_BYTES = 4 * 1024**3
MAX_PROMPT_LENGTH = 500

EXAMPLES = [
    "A red fox wearing round glasses beside a glowing blue lantern in a snowy pine forest.",
    "A cinematic portrait of an astronaut holding a sunflower, soft rim light, detailed suit.",
    "A small corgi surfing a turquoise wave at golden hour, editorial wildlife photography.",
]

_pipeline = None
_pipeline_lock = threading.Lock()
_inference_lock = threading.Lock()
logger = logging.getLogger("sdxl-attention-space")


def get_pipeline():
    """Load SDXL once, on demand, so the Space UI can start before weights load."""

    global _pipeline
    if _pipeline is not None:
        return _pipeline
    if not torch.cuda.is_available():
        raise RuntimeError("This Space requires a CUDA GPU runtime.")

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        torch.backends.cuda.matmul.allow_tf32 = True
        load_kwargs = {"torch_dtype": torch.float16, "use_safetensors": True}
        if LOAD_VARIANT is not None:
            load_kwargs["variant"] = LOAD_VARIANT
        pipeline = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, **load_kwargs)
        if hasattr(pipeline.vae, "enable_slicing"):
            pipeline.vae.enable_slicing()
        if hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()
        pipeline.set_progress_bar_config(disable=True)
        _pipeline = pipeline.to(DEVICE)
    return _pipeline


def _normalize_seed(seed) -> int:
    value = -1 if seed is None else int(seed)
    if value < 0:
        return secrets.randbelow(2**31)
    return value % (2**31)


def _last_scheduler_timestep(pipeline, steps: int) -> list[float]:
    # SDXL configures the same scheduler immediately before denoising. Passing
    # the exact last value prevents CPU copies of Q/K on all earlier steps.
    pipeline.scheduler.set_timesteps(steps, device=DEVICE)
    last = pipeline.scheduler.timesteps[-1]
    return [float(last.detach().cpu() if hasattr(last, "detach") else last)]


def _natural_key(path: Path):
    return [
        int(item) if item.isdigit() else item.lower()
        for item in re.split(r"(\d+)", path.name)
    ]


def _caption(path: Path) -> str:
    stem = path.stem
    if stem.startswith("token-"):
        return stem.removeprefix("token-").replace("_", " ")
    if stem.startswith("query-patch-"):
        return "center query patch · " + stem.removeprefix("query-patch-")
    return stem.replace("_", " ")


def _gallery(root: Path, relation_slug: str) -> list[tuple[Image.Image, str]]:
    overlay_root = root / "visuals" / relation_slug / "aggregate" / "batch-000" / "overlays"
    if not overlay_root.is_dir():
        return []
    paths = sorted(overlay_root.rglob("*.png"), key=_natural_key)
    items = []
    for path in paths:
        with Image.open(path) as image:
            items.append((image.convert("RGB").copy(), _caption(path)))
    return items


def _relation_counts(summary: dict) -> str:
    counts = summary.get("relations", {})
    return " · ".join(f"`{name}` {count} maps" for name, count in counts.items())


@torch.inference_mode()
def generate_attention(
    prompt: str,
    negative_prompt: str,
    seed,
    steps,
    guidance_scale,
    resolution,
    progress=gr.Progress(track_tqdm=False),
):
    """Generate one SDXL image and its last-step relation-aware attention maps."""

    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a prompt before generating.")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise gr.Error(f"Prompt must be at most {MAX_PROMPT_LENGTH} characters.")

    seed = _normalize_seed(seed)
    steps = int(steps)
    resolution = int(resolution)
    guidance_scale = float(guidance_scale)
    negative_prompt = (negative_prompt or "").strip()

    with _inference_lock:
        started = time.perf_counter()
        try:
            progress(0.05, desc="Loading SDXL")
            pipeline = get_pipeline()
            torch.cuda.reset_peak_memory_stats()
            capture_timesteps = _last_scheduler_timestep(pipeline, steps)
            generator = torch.Generator(device=DEVICE).manual_seed(seed)

            call_kwargs = {
                "prompt": [prompt],
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "height": resolution,
                "width": resolution,
                "generator": generator,
            }
            if negative_prompt:
                call_kwargs["negative_prompt"] = [negative_prompt]

            progress(0.12, desc="Generating and capturing Q/K")
            with AttentionCapture(
                pipeline,
                relations=RELATIONS,
                timesteps=capture_timesteps,
                mode="fast",
                heads="mean",
                offload="cpu",
                dtype=torch.float16,
                relation_query_indices={"image->image": "center"},
                max_capture_bytes=MAX_CAPTURE_BYTES,
                metadata={
                    "application": "sdxl-attention-space",
                    "seed": seed,
                    "steps": steps,
                    "resolution": resolution,
                },
            ) as capture:
                image = pipeline(**call_kwargs).images[0]

            progress(0.76, desc="Computing attention probabilities")
            maps = capture.compute(compute_device=DEVICE)
            validation = maps.validate()
            if validation["status"] != "pass":
                raise RuntimeError(
                    "Attention validation failed: " + "; ".join(validation["issues"][:3])
                )

            progress(0.86, desc="Rendering overlays")
            with tempfile.TemporaryDirectory(prefix="sdxl-attention-") as directory:
                output_root = Path(directory) / "attention"
                manifest = maps.save(
                    str(output_root),
                    tokenizer=text_tokenizers(pipeline),
                    prompts=[prompt],
                    images=[image],
                    save_raw=False,
                    visualization_config=VisualizationConfig(
                        spatial_long_side=512,
                        max_items=24,
                        overview_items=12,
                        overlay_alpha=0.58,
                        save_layer_maps=False,
                        show_spatial_labels=False,
                    ),
                )
                token_gallery = _gallery(output_root, "image-to-text")
                patch_gallery = _gallery(output_root, "image-to-image")

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak_gib = torch.cuda.max_memory_allocated() / 1024**3
            summary = maps.summary()
            metadata = {
                "model": MODEL_ID,
                "model_variant": LOAD_VARIANT,
                "seed": seed,
                "resolution": f"{resolution}x{resolution}",
                "steps": steps,
                "guidance_scale": guidance_scale,
                "capture": "last timestep, head mean",
                "attention": summary,
                "peak_capture_mib": round(
                    maps.metadata.get("peak_capture_bytes", 0) / 1024**2, 1
                ),
                "peak_cuda_gib": round(peak_gib, 2),
                "elapsed_seconds": round(elapsed, 2),
                "warnings": manifest.get("warnings", []),
            }
            status = (
                f"Completed in **{elapsed:.1f}s** with seed `{seed}` · "
                f"{_relation_counts(summary)}"
            )
            progress(1.0, desc="Done")
            return image, token_gallery, patch_gallery, metadata, status
        except torch.cuda.OutOfMemoryError as exc:
            logger.exception("CUDA out of memory")
            raise gr.Error(
                "GPU memory was exhausted. Retry at 768px or wait for the "
                "current request to finish."
            ) from exc
        except gr.Error:
            raise
        except Exception as exc:
            logger.exception("Generation failed")
            raise gr.Error(f"Generation failed: {exc}") from exc
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def build_demo():
    with gr.Blocks(title="SDXL Attention Explorer") as demo:
        gr.Markdown(
            """
            # SDXL Attention Explorer

            Generate an image and inspect every attention relation exposed by the SDXL denoiser.
            Token maps show **image→text** cross-attention; the patch map shows
            **image→image** self-attention for the center query patch. Attention is averaged over
            heads and layers at the final denoising timestep.
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=6):
                prompt = gr.Textbox(
                    label="Prompt",
                    value=EXAMPLES[0],
                    lines=3,
                    max_lines=6,
                    placeholder="Describe the image to generate...",
                )
                with gr.Accordion("Generation controls", open=False):
                    negative_prompt = gr.Textbox(
                        label="Negative prompt",
                        value="blurry, low quality, distorted",
                        lines=2,
                    )
                    with gr.Row():
                        seed = gr.Number(label="Seed (-1 = random)", value=2026, precision=0)
                        resolution = gr.Radio(
                            choices=[768, 1024],
                            value=768,
                            label="Resolution",
                        )
                    with gr.Row():
                        steps = gr.Slider(10, 30, value=20, step=1, label="Steps")
                        guidance = gr.Slider(
                            1.0, 10.0, value=7.0, step=0.5, label="Guidance"
                        )
                generate = gr.Button("Generate attention maps", variant="primary")
                status = gr.Markdown("Ready. The model loads on the first request.")
            with gr.Column(scale=5):
                generated = gr.Image(
                    label="Generated image", type="pil", format="png", height=480
                )

        with gr.Tabs():
            with gr.Tab("image→text · tokens"):
                token_maps = gr.Gallery(
                    label="Token attention overlays",
                    columns=4,
                    height=620,
                    object_fit="contain",
                    format="png",
                    buttons=["download", "download_all", "fullscreen"],
                )
                gr.Markdown(
                    "Each tile is one prompt token. Labels are shown by the UI; "
                    "the exported image "
                    "contains only the generated image and its attention overlay."
                )
            with gr.Tab("image→image · center patch"):
                patch_maps = gr.Gallery(
                    label="Spatial self-attention overlay",
                    columns=2,
                    height=620,
                    object_fit="contain",
                    format="png",
                    buttons=["download", "fullscreen"],
                )
            with gr.Tab("Run metadata"):
                metadata = gr.JSON(label="Capture and performance metadata")

        gr.Examples(examples=[[item] for item in EXAMPLES], inputs=[prompt])
        gr.Markdown(
            "Built with [attention-map-diffusers]"
            "(https://github.com/wooyeolbaek/attention-map-diffusers) "
            "and Hugging Face Diffusers. Raw `.pt` tensors are not stored by this demo."
        )

        inputs = [prompt, negative_prompt, seed, steps, guidance, resolution]
        outputs = [generated, token_maps, patch_maps, metadata, status]
        generate.click(
            generate_attention,
            inputs=inputs,
            outputs=outputs,
            api_name="generate",
            api_description=(
                "Generate one SDXL image and its image→text/image→image attention overlays."
            ),
            concurrency_limit=1,
            show_progress="minimal",
            scroll_to_output=True,
        )
        prompt.submit(
            generate_attention,
            inputs=inputs,
            outputs=outputs,
            api_visibility="private",
            concurrency_limit=1,
            show_progress="minimal",
        )
    return demo.queue(max_size=8, default_concurrency_limit=1)


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
