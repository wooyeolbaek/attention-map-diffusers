from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from common import MODEL_CONFIGS, model_keys


def _generator(torch, device: str, seed: int):
    return torch.Generator(device=device).manual_seed(seed)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare real Diffusers latent output with attention capture disabled and enabled."
    )
    parser.add_argument("--model", default="all", help="Model key, comma-list, or 'all'.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("generation-parity.json"))
    args = parser.parse_args()

    import diffusers
    import torch

    from attention_map_diffusers import AttentionCapture

    report = {"status": "pass", "models": {}}
    for key in model_keys(args.model):
        config = MODEL_CONFIGS[key]
        pipe = baseline = captured = capture = None
        try:
            pipeline_cls = getattr(diffusers, config.pipeline_class)
            pipe = pipeline_cls.from_pretrained(
                config.model_id,
                torch_dtype=getattr(torch, config.torch_dtype),
                **config.extra_from_pretrained_kwargs,
            ).to(args.device)
            pipe.set_progress_bar_config(disable=True)

            call_kwargs = dict(config.extra_call_kwargs)
            call_kwargs.update(
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width,
                output_type="latent",
            )
            if config.guidance_scale is not None:
                call_kwargs["guidance_scale"] = config.guidance_scale

            prompt = config.prompt
            baseline = pipe(
                [prompt], generator=_generator(torch, args.device, args.seed), **call_kwargs
            ).images.detach().float()

            with AttentionCapture(
                pipe,
                relations=["image->text"],
                mode="exact",
                heads="mean",
                offload="cpu",
                relation_query_indices={"image->text": "center"},
                relation_key_indices={"image->text": (0,)},
            ) as capture:
                captured = pipe(
                    [prompt], generator=_generator(torch, args.device, args.seed), **call_kwargs
                ).images.detach().float()

            difference = (baseline - captured).abs()
            max_abs = float(difference.max())
            mean_abs = float(difference.mean())
            finite = bool(torch.isfinite(baseline).all() and torch.isfinite(captured).all())
            passed = finite and torch.equal(baseline, captured)
            report["models"][key] = {
                "status": "pass" if passed else "fail",
                "shape": list(baseline.shape),
                "dtype": str(baseline.dtype),
                "finite": finite,
                "max_abs_difference": max_abs,
                "mean_abs_difference": mean_abs,
                "captured_records": len(capture.records),
            }
            if not passed:
                report["status"] = "fail"
            print(f"[{report['models'][key]['status'].upper()}] {key}: max_abs={max_abs:.8g}")
        except Exception as exc:
            report["status"] = "fail"
            report["models"][key] = {
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[FAIL] {key}: {type(exc).__name__}: {exc}")
        finally:
            del pipe, baseline, captured, capture
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"report: {args.output}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
