from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

from common import MODEL_CONFIGS, model_keys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit every denoiser attention processor without allocating model weights."
    )
    parser.add_argument("--model", default="all", help="Model key, comma-list, or 'all'.")
    parser.add_argument("--output", type=Path, default=Path("attention-coverage.json"))
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    import diffusers
    from accelerate import init_empty_weights

    from attention_map_diffusers import AttentionCapture

    report = {"status": "pass", "models": {}}
    for key in model_keys(args.model):
        model_config = MODEL_CONFIGS[key]
        denoiser = capture = None
        try:
            pipeline_index = diffusers.DiffusionPipeline.load_config(
                model_config.model_id,
                local_files_only=args.local_files_only,
            )
            component_name = "transformer" if "transformer" in pipeline_index else "unet"
            component_spec = pipeline_index[component_name]
            denoiser_class = getattr(diffusers, component_spec[1])
            denoiser_config = denoiser_class.load_config(
                model_config.model_id,
                subfolder=component_name,
                local_files_only=args.local_files_only,
            )
            with init_empty_weights():
                denoiser = denoiser_class.from_config(denoiser_config)
            pipeline_parts = {component_name: denoiser}
            for tokenizer_name in ("tokenizer", "tokenizer_2", "tokenizer_3"):
                tokenizer_spec = pipeline_index.get(tokenizer_name)
                if tokenizer_spec and any(tokenizer_spec):
                    pipeline_parts[tokenizer_name] = SimpleNamespace(model_max_length=77)
            pipeline = SimpleNamespace(**pipeline_parts)
            capture = AttentionCapture(pipeline, relations=model_config.relations)
            capture.install()
            coverage = dict(capture.coverage)
            coverage.pop("captured_layers", None)
            report["models"][key] = {
                "status": "pass",
                "denoiser": denoiser_class.__name__,
                "relations": list(model_config.relations),
                "coverage": coverage,
                "text_conditioning": capture.text_conditioning,
            }
            print(
                f"[PASS] {key}: {coverage['captured_modules']}/"
                f"{coverage['attention_modules']} attention modules"
            )
        except Exception as exc:
            report["status"] = "fail"
            report["models"][key] = {
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[FAIL] {key}: {type(exc).__name__}: {exc}")
        finally:
            if capture is not None:
                capture.restore()
            del denoiser, capture
            gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"report: {args.output}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
