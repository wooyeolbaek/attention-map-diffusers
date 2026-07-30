from __future__ import annotations

import argparse
import gc
import json
import os
from typing import List

from PIL import Image

from common import (
    ALL_ATTENTION_RELATIONS,
    MODEL_CONFIGS,
    model_keys,
    parse_heads,
    parse_indices,
    relation_list,
)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run attention-map capture demos for supported Diffusers models.")
    parser.add_argument("--model", default="sd", help="Model key, comma-list, or 'all'.")
    parser.add_argument("--relations", default="all", help="Relation list, e.g. image->text,text->image, or 'all'.")
    parser.add_argument("--mode", default="fast", choices=["fast", "exact", "stream"])
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
        default="all",
        help="Query patches for image->image: 'center', 'sample:N', 'all', or comma-separated indices.",
    )
    parser.add_argument(
        "--large-relation-key-indices",
        default="all",
        help="Key patches for image->image. Keep 'all' to render each query patch as a spatial map.",
    )
    parser.add_argument("--key-indices", default="all", help="Use a comma list to limit key tokens.")
    parser.add_argument(
        "--heads",
        default="mean",
        help="Head reduction: 'mean', 'all', or a comma-separated list such as '0,2'.",
    )
    parser.add_argument(
        "--max-visual-items",
        default="all",
        help="Maximum token/patch PNGs per map, or 'all' (default).",
    )
    parser.add_argument(
        "--spatial-long-side",
        type=int,
        default=512,
        help="Long side in pixels for spatial maps and overlays.",
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
    parser.add_argument("--allow-large-maps", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the execution matrix without importing ML deps.")
    args = parser.parse_args(argv)

    try:
        selected_heads = parse_heads(args.heads)
    except ValueError as exc:
        parser.error(str(exc))

    for name in ("steps", "height", "width", "spatial_long_side", "max_map_elements"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")

    try:
        query_indices = parse_indices(args.query_indices)
        key_indices = parse_indices(args.key_indices)
        large_relation_query_indices = parse_indices(args.large_relation_query_indices)
        large_relation_key_indices = parse_indices(args.large_relation_key_indices)
    except ValueError as exc:
        parser.error(str(exc))

    if str(args.max_visual_items).lower() == "all":
        max_visual_items = None
    else:
        try:
            max_visual_items = int(args.max_visual_items)
        except ValueError:
            parser.error("--max-visual-items must be a positive integer or 'all'.")
        if max_visual_items <= 0:
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
    from attention_map_diffusers import AttentionCapture, VisualizationConfig

    os.makedirs(args.output_dir, exist_ok=True)
    failures = []
    for key in selected_models:
        config = MODEL_CONFIGS[key]
        relations = relation_list(config, args.relations)
        pipe = None
        result = None
        maps = None
        capture = None
        tokenizer = None
        generated_images = None
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
                    f"{config.default_steps}; generated images may look incomplete or broken"
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
            if args.skip_visualization:
                call_kwargs["output_type"] = "latent"

            print(f"[{key}] capture relations: {', '.join(relations)}")
            relation_query_indices = {
                relation: large_relation_query_indices
                for relation in relations
                if relation == "image->image" and large_relation_query_indices is not None
            }
            relation_key_indices = {
                relation: large_relation_key_indices
                for relation in relations
                if relation == "image->image" and large_relation_key_indices is not None
            }
            with AttentionCapture(
                pipe,
                relations=relations,
                timesteps=args.capture_timesteps,
                mode=args.mode,
                heads=selected_heads,
                offload=args.device if args.offload == "same" else args.offload,
                query_indices=query_indices,
                key_indices=key_indices,
                relation_query_indices=relation_query_indices,
                relation_key_indices=relation_key_indices,
                max_map_elements=args.max_map_elements,
                allow_large_maps=args.allow_large_maps,
            ) as capture:
                result = pipe([prompt], **call_kwargs)
                maps = capture.compute()

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
                if not args.skip_visualization and hasattr(result, "images")
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

            with open(os.path.join(model_dir, "run-config.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "model": key,
                        "model_id": config.model_id,
                        "prompt": prompt,
                        "seed": args.seed,
                        "steps": actual_steps,
                        "recommended_steps": config.default_steps,
                        "height": height,
                        "width": width,
                        "generated_sizes": [list(image.size) for image in generated_images],
                        "relations": relations,
                        "mode": args.mode,
                        "capture_timesteps": args.capture_timesteps,
                        "offload": args.device if args.offload == "same" else args.offload,
                        "query_indices": args.query_indices,
                        "large_relation_query_indices": args.large_relation_query_indices,
                        "large_relation_key_indices": args.large_relation_key_indices,
                        "key_indices": args.key_indices,
                        "heads": selected_heads,
                        "max_visual_items": max_visual_items,
                        "spatial_long_side": args.spatial_long_side,
                        "save_layer_maps": args.save_layer_maps,
                        "raw_tensors": args.save_raw,
                        "visualizations": not args.skip_visualization,
                        "max_map_elements": args.max_map_elements,
                        "allow_large_maps": args.allow_large_maps,
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )

            if not args.skip_visualization or args.save_raw:
                tokenizer = (
                    getattr(pipe, config.tokenizer_attribute, None)
                    if not args.skip_visualization
                    else None
                )
                maps.save(
                    os.path.join(model_dir, "attention"),
                    tokenizer=tokenizer,
                    prompts=[prompt],
                    images=generated_images,
                    unconditional=None,
                    save_raw=args.save_raw,
                    save_visualizations=not args.skip_visualization,
                    visualization_config=VisualizationConfig(
                        spatial_long_side=args.spatial_long_side,
                        max_items=max_visual_items,
                        save_layer_maps=args.save_layer_maps,
                    ),
                )
            print(f"[{key}] ok: {maps.summary()}")
        except Exception as exc:
            failures.append((key, exc))
            print(f"[{key}] failed: {type(exc).__name__}: {exc}")
        finally:
            del pipe, result, maps, capture, tokenizer, generated_images
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


if __name__ == "__main__":
    raise SystemExit(main())
