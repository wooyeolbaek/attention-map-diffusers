from __future__ import annotations

import argparse
import gc
import json
import os
from typing import List, Optional

from PIL import Image

from common import (
    ALL_ATTENTION_RELATIONS,
    MODEL_CONFIGS,
    model_keys,
    parse_heads,
    parse_indices,
    relation_list,
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run attention-map capture demos for supported Diffusers models.")
    parser.add_argument("--model", default="sd", help="Model key, comma-list, or 'all'.")
    parser.add_argument("--relations", default="all", help="Relation list, e.g. video->text,text->video, or 'all'.")
    parser.add_argument("--mode", default="fast", choices=["fast", "exact"])
    parser.add_argument(
        "--capture-timesteps",
        default="last",
        choices=["last", "all"],
        help="Capture only the final denoising timestep (default) or every timestep.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override the model-specific quality default; low values are intended only for smoke tests.",
    )
    parser.add_argument("--height", type=int, default=None, help="Override the model's native height.")
    parser.add_argument("--width", type=int, default=None, help="Override the model's native width.")
    parser.add_argument("--num-frames", type=int, default=None, help="Override a video model's frame count.")
    parser.add_argument("--fps", type=int, default=None, help="GIF playback rate for video outputs.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--offload",
        default="same",
        help="Captured tensor device. 'same' follows --device; use 'cpu' when VRAM is constrained.",
    )
    parser.add_argument("--output-dir", default="demo-outputs")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--seed", type=int, default=0, help="Deterministic generation seed.")
    parser.add_argument("--query-indices", default="all", help="Global query selection; keep 'all' for token PNGs.")
    parser.add_argument(
        "--large-relation-query-indices",
        default=None,
        help=(
            "Query patches for image->image or video->video: 'center', 'sample:N', 'all', or indices. "
            "The model default is all image patches and the center video patch."
        ),
    )
    parser.add_argument(
        "--large-relation-key-indices",
        default="all",
        help="Key patches for image->image or video->video. Keep 'all' for complete spatial maps.",
    )
    parser.add_argument("--key-indices", default="all", help="Use a comma list to limit key tokens.")
    parser.add_argument(
        "--heads",
        default="mean",
        help="Head reduction: 'mean', 'all', or a comma-separated list such as '0,2'.",
    )
    parser.add_argument(
        "--max-visual-items",
        default=None,
        help="Maximum token/patch outputs per map, or 'all'. Video models default to 16.",
    )
    parser.add_argument(
        "--spatial-long-side",
        type=int,
        default=512,
        help="Long side in pixels for spatial maps and overlays.",
    )
    parser.add_argument(
        "--show-spatial-labels",
        action="store_true",
        help="Add a white title/token header to spatial maps and overlays; the default writes image-only PNGs.",
    )
    parser.add_argument(
        "--save-layer-maps",
        action="store_true",
        help="Also render every timestep/layer. Aggregate-only output is the faster default.",
    )
    parser.add_argument("--save-raw", action="store_true", help="Also save each captured attention tensor as .pt.")
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Run generation and in-memory tensor validation without writing attention PNGs.",
    )
    parser.add_argument("--max-map-elements", type=int, default=128_000_000)
    parser.add_argument(
        "--max-capture-gib",
        type=float,
        default=8.0,
        help="Maximum cumulative Q/K/map storage in GiB; use 0 to disable the guard.",
    )
    parser.add_argument("--allow-large-maps", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the execution matrix without importing ML deps.")
    args = parser.parse_args(argv)

    try:
        selected_heads = parse_heads(args.heads)
    except ValueError as exc:
        parser.error(str(exc))

    for name in ("steps", "height", "width", "num_frames", "fps", "spatial_long_side", "max_map_elements"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.max_capture_gib < 0:
        parser.error("--max-capture-gib must be non-negative.")

    try:
        query_indices = parse_indices(args.query_indices)
        key_indices = parse_indices(args.key_indices)
        large_query_override = (
            parse_indices(args.large_relation_query_indices)
            if args.large_relation_query_indices is not None
            else None
        )
        large_relation_key_indices = parse_indices(args.large_relation_key_indices)
    except ValueError as exc:
        parser.error(str(exc))

    if args.max_visual_items is None:
        max_visual_items_override = None
    elif str(args.max_visual_items).lower() == "all":
        max_visual_items_override = "all"
    else:
        try:
            max_visual_items_override = int(args.max_visual_items)
        except ValueError:
            parser.error("--max-visual-items must be a positive integer or 'all'.")
        if max_visual_items_override <= 0:
            parser.error("--max-visual-items must be a positive integer or 'all'.")

    try:
        selected_models = model_keys(args.model)
    except ValueError as exc:
        parser.error(str(exc))
    if args.relations != "all":
        requested_relations = [item.strip() for item in args.relations.split(",") if item.strip()]
        unknown_relations = [
            relation for relation in requested_relations if relation not in ALL_ATTENTION_RELATIONS
        ]
        if unknown_relations:
            parser.error("unknown attention relation(s): " + ", ".join(unknown_relations))
    empty_relation_models = [
        key for key in selected_models if not relation_list(MODEL_CONFIGS[key], args.relations)
    ]
    if empty_relation_models:
        parser.error(
            "the requested relation selection is unsupported for: "
            + ", ".join(empty_relation_models)
        )
    if args.dry_run:
        for key in selected_models:
            config = MODEL_CONFIGS[key]
            print(f"{key}: {config.model_id}")
            print(f"  relations: {', '.join(relation_list(config, args.relations))}")
        return 0

    import torch
    import diffusers
    from attention_map_diffusers import AttentionCapture, VisualizationConfig, text_tokenizers

    os.makedirs(args.output_dir, exist_ok=True)
    failures = []
    for key in selected_models:
        config = MODEL_CONFIGS[key]
        relations = relation_list(config, args.relations)
        large_query_spec = (
            args.large_relation_query_indices
            if args.large_relation_query_indices is not None
            else config.default_large_relation_query_indices
        )
        large_relation_query_indices = (
            large_query_override
            if args.large_relation_query_indices is not None
            else parse_indices(large_query_spec)
        )
        max_visual_items = (
            config.default_max_visual_items
            if max_visual_items_override is None
            else None if max_visual_items_override == "all" else max_visual_items_override
        )
        pipe = None
        result = None
        maps = None
        capture = None
        tokenizer = None
        generated_images = None
        generated_videos = None
        try:
            pipeline_cls = getattr(diffusers, config.pipeline_class)
            dtype = getattr(torch, config.torch_dtype)
            print(f"[{key}] loading {config.model_id}")
            pipe = pipeline_cls.from_pretrained(
                config.model_id,
                torch_dtype=dtype,
                **config.extra_from_pretrained_kwargs,
            )
            pipe = pipe.to(args.device)

            prompt = args.prompt or config.prompt
            call_kwargs = dict(config.extra_call_kwargs)
            actual_steps = args.steps if args.steps is not None else config.default_steps
            call_kwargs["num_inference_steps"] = actual_steps
            call_kwargs["generator"] = torch.Generator(device=args.device).manual_seed(args.seed)
            if actual_steps < config.default_steps:
                print(
                    f"[{key}] warning: {actual_steps} steps is below the recommended "
                    f"{config.default_steps}; generated output may look incomplete or broken"
                )
            if config.guidance_scale is not None:
                call_kwargs["guidance_scale"] = config.guidance_scale
            height = args.height if args.height is not None else config.default_height
            width = args.width if args.width is not None else config.default_width
            if (
                args.height is not None
                and config.default_height is not None
                and args.height < config.default_height
            ) or (
                args.width is not None
                and config.default_width is not None
                and args.width < config.default_width
            ):
                print(
                    f"[{key}] warning: requested {height}x{width} is below the recommended "
                    f"{config.default_height}x{config.default_width}; composition may repeat or degrade"
                )
            if height is not None:
                call_kwargs["height"] = height
            if width is not None:
                call_kwargs["width"] = width
            num_frames = args.num_frames if args.num_frames is not None else config.default_num_frames
            if num_frames is not None:
                call_kwargs["num_frames"] = num_frames
            if args.skip_visualization:
                call_kwargs["output_type"] = "latent"

            print(f"[{key}] capture relations: {', '.join(relations)}")
            relation_query_indices = {
                relation: large_relation_query_indices
                for relation in relations
                if relation in {"image->image", "video->video"} and large_relation_query_indices is not None
            }
            relation_key_indices = {
                relation: large_relation_key_indices
                for relation in relations
                if relation in {"image->image", "video->video"} and large_relation_key_indices is not None
            }
            capture_timesteps = args.capture_timesteps
            if (
                args.capture_timesteps == "last"
                and config.pipeline_class in {"CogVideoXPipeline", "WanPipeline"}
            ):
                # These pipelines call scheduler.set_timesteps(steps, device)
                # unchanged. Filtering by that exact final value avoids copying
                # native-resolution video Q/K to CPU on every earlier step.
                pipe.scheduler.set_timesteps(actual_steps, device=args.device)
                capture_timesteps = [float(pipe.scheduler.timesteps[-1])]
            with AttentionCapture(
                pipe,
                relations=relations,
                timesteps=capture_timesteps,
                mode=args.mode,
                heads=selected_heads,
                offload=args.device if args.offload == "same" else args.offload,
                query_indices=query_indices,
                key_indices=key_indices,
                relation_query_indices=relation_query_indices,
                relation_key_indices=relation_key_indices,
                max_map_elements=args.max_map_elements,
                max_capture_bytes=(
                    None if args.max_capture_gib == 0 else int(args.max_capture_gib * 1024**3)
                ),
                allow_large_maps=args.allow_large_maps,
                metadata={
                    "model_key": key,
                    "model_id": config.model_id,
                    "seed": args.seed,
                    "steps": actual_steps,
                    "height": height,
                    "width": width,
                    "num_frames": num_frames,
                },
            ) as capture:
                result = pipe(prompt=[prompt], **call_kwargs)
                compute_device = None
                if args.offload == "cpu" and str(args.device).startswith("cuda"):
                    # Native video Q/K can coexist with generation safely on
                    # CPU. Release model VRAM after decoding, then recompute
                    # each relation on CUDA without retaining duplicate Q/K.
                    for component in pipe.components.values():
                        if isinstance(component, torch.nn.Module):
                            component.to("cpu")
                    torch.cuda.empty_cache()
                    compute_device = args.device
                maps = capture.compute(compute_device=compute_device)

            model_dir = os.path.join(args.output_dir, key)
            os.makedirs(model_dir, exist_ok=True)
            tensor_validation = maps.validate()
            with open(os.path.join(model_dir, "tensor-validation.json"), "w", encoding="utf-8") as handle:
                json.dump(tensor_validation, handle, ensure_ascii=False, indent=2)
            if tensor_validation["status"] != "pass":
                raise RuntimeError(
                    "captured attention tensor validation failed: "
                    + "; ".join(tensor_validation["issues"][:3])
                )
            generated_images = (
                list(result.images)
                if config.modality == "image" and not args.skip_visualization and hasattr(result, "images")
                else []
            )
            generated_videos = (
                _video_batches(result.frames)
                if config.modality == "video" and not args.skip_visualization and hasattr(result, "frames")
                else []
            )
            generated_dir = os.path.join(model_dir, "generated")
            os.makedirs(generated_dir, exist_ok=True)
            if generated_images:
                for index, image in enumerate(generated_images):
                    image_path = os.path.join(generated_dir, f"{index:03d}-{config.output_name}.png")
                    image.save(image_path)
                    with Image.open(image_path) as saved_image:
                        saved_image.verify()
            for index, video in enumerate(generated_videos):
                video_dir = os.path.join(generated_dir, f"{index:03d}-{config.output_name}")
                os.makedirs(video_dir, exist_ok=True)
                for frame_index, frame in enumerate(video):
                    frame.save(os.path.join(video_dir, f"frame-{frame_index:04d}.png"))
                _save_video_gif(
                    video,
                    os.path.join(generated_dir, f"{index:03d}-{config.output_name}.gif"),
                    args.fps or config.fps,
                )

            with open(os.path.join(model_dir, "run-config.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "model": key,
                        "model_id": config.model_id,
                        "diffusers_version": diffusers.__version__,
                        "torch_version": torch.__version__,
                        "prompt": prompt,
                        "seed": args.seed,
                        "steps": actual_steps,
                        "recommended_steps": config.default_steps,
                        "height": height,
                        "width": width,
                        "num_frames": num_frames,
                        "fps": args.fps or config.fps,
                        "generated_sizes": [list(image.size) for image in generated_images],
                        "generated_video_sizes": [
                            [len(video), *video[0].size] for video in generated_videos if video
                        ],
                        "attention_summary": maps.summary(),
                        "guidance_roles": {
                            str(role): sum(
                                record.metadata.get("guidance_role") == role
                                for record in maps.records
                            )
                            for role in sorted(
                                {
                                    record.metadata.get("guidance_role")
                                    for record in maps.records
                                    if record.metadata.get("guidance_role") is not None
                                }
                            )
                        },
                        "attention_coverage": capture.coverage,
                        "text_conditioning": capture.text_conditioning,
                        "relations": relations,
                        "mode": args.mode,
                        "capture_timesteps": args.capture_timesteps,
                        "offload": args.device if args.offload == "same" else args.offload,
                        "query_indices": args.query_indices,
                        "large_relation_query_indices": large_query_spec,
                        "large_relation_key_indices": args.large_relation_key_indices,
                        "key_indices": args.key_indices,
                        "heads": selected_heads,
                        "max_visual_items": max_visual_items,
                        "spatial_long_side": args.spatial_long_side,
                        "show_spatial_labels": args.show_spatial_labels,
                        "save_layer_maps": args.save_layer_maps,
                        "raw_tensors": args.save_raw,
                        "visualizations": not args.skip_visualization,
                        "max_map_elements": args.max_map_elements,
                        "max_capture_gib": args.max_capture_gib,
                        "allow_large_maps": args.allow_large_maps,
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )

            if not args.skip_visualization or args.save_raw:
                tokenizer = (
                    text_tokenizers(pipe)
                    if not args.skip_visualization
                    else None
                )
                maps.save(
                    os.path.join(model_dir, "attention"),
                    tokenizer=tokenizer,
                    prompts=[prompt],
                    images=generated_images,
                    videos=generated_videos,
                    unconditional=None,
                    save_raw=args.save_raw,
                    save_visualizations=not args.skip_visualization,
                    visualization_config=VisualizationConfig(
                        spatial_long_side=args.spatial_long_side,
                        max_items=max_visual_items,
                        save_layer_maps=args.save_layer_maps,
                        show_spatial_labels=args.show_spatial_labels,
                        video_fps=args.fps or config.fps,
                    ),
                )
            print(f"[{key}] ok: {maps.summary()}")
        except Exception as exc:
            failures.append((key, exc))
            print(f"[{key}] failed: {type(exc).__name__}: {exc}")
        finally:
            del pipe, result, maps, capture, tokenizer, generated_images, generated_videos
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    if failures:
        print("Failures:")
        for key, exc in failures:
            print(f"  {key}: {type(exc).__name__}: {exc}")
        return 1
    return 0


def _video_batches(value):
    from attention_map_diffusers.visualization import _as_pil_videos

    return _as_pil_videos(value)


def _save_video_gif(frames, path: str, fps: int) -> None:
    if not frames:
        return
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, round(1000 / fps)),
        loop=0,
        optimize=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
