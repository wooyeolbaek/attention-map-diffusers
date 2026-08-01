from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import MODEL_CONFIGS, model_keys


SUCCESS_PROFILES = {
    "fast-last": {
        "mode": "fast",
        "capture_timesteps": "last",
        "visualizations": True,
        "raw_tensors": False,
        "save_layer_maps": False,
        "relations": "all",
        "options": {
            "steps": 2, "height": 256, "width": 256, "seed": 123,
            "heads": "mean", "query_indices": "all", "key_indices": "all",
            "large_relation_query_indices": "sample:3", "large_relation_key_indices": "all",
            "max_visual_items": 2, "spatial_long_side": 128,
            "max_map_elements": 128_000_000, "allow_large_maps": False,
        },
        "offload": "cuda",
    },
    "exact-all-layer-raw": {
        "mode": "exact",
        "capture_timesteps": "all",
        "visualizations": True,
        "raw_tensors": True,
        "save_layer_maps": True,
        "relations": "all",
        "options": {
            "steps": 2, "height": 256, "width": 256, "seed": 123,
            "heads": "all", "query_indices": "all", "key_indices": "all",
            "large_relation_query_indices": "center", "large_relation_key_indices": "all",
            "max_visual_items": 1, "spatial_long_side": 96,
            "max_map_elements": 1, "allow_large_maps": True,
        },
        "offload": "cpu",
    },
    "keypatch-visual": {
        "mode": "fast",
        "capture_timesteps": "last",
        "visualizations": True,
        "raw_tensors": False,
        "save_layer_maps": False,
        "relations": ["image->image"],
        "options": {
            "steps": 1, "height": 256, "width": 256, "seed": 321,
            "heads": "mean", "query_indices": "all", "key_indices": "all",
            "large_relation_query_indices": "all", "large_relation_key_indices": "sample:3",
            "max_visual_items": 2, "spatial_long_side": 112,
            "max_map_elements": 128_000_000, "allow_large_maps": False,
        },
        "offload": "cuda",
    },
    "selected-skip": {
        "mode": "fast",
        "capture_timesteps": "last",
        "visualizations": False,
        "raw_tensors": True,
        "save_layer_maps": False,
        "relations": "all",
        "options": {
            "steps": 1, "height": 256, "width": 256, "seed": 777,
            "heads": [0, 2], "query_indices": "center", "key_indices": "sample:3",
            "large_relation_query_indices": "center", "large_relation_key_indices": "sample:3",
            "max_visual_items": None, "spatial_long_side": 512,
            "max_map_elements": 128_000_000, "allow_large_maps": False,
        },
        "offload": "cuda",
    },
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_image(path: Path, issues: list[str]) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        issues.append(f"invalid image {path}: {type(exc).__name__}: {exc}")


def _expected_relations(key: str, profile: dict) -> set[str]:
    relations = profile["relations"]
    if relations == ["image->image"] and MODEL_CONFIGS[key].modality == "video":
        return {"video->video"}
    return set(MODEL_CONFIGS[key].relations if relations == "all" else relations)


def validate_success_profile(root: Path, profile_name: str, key: str) -> dict:
    expected = SUCCESS_PROFILES[profile_name]
    model_root = root / profile_name / key
    issues: list[str] = []
    run_path = model_root / "run-config.json"
    tensor_path = model_root / "tensor-validation.json"
    if not run_path.is_file() or not tensor_path.is_file():
        return {"status": "fail", "issues": ["missing run-config.json or tensor-validation.json"]}

    run = _json(run_path)
    tensor_validation = _json(tensor_path)
    if tensor_validation.get("status") != "pass":
        issues.append(f"tensor validation failed: {tensor_validation.get('issues', [])[:3]}")
    for field in ("mode", "capture_timesteps", "visualizations", "raw_tensors", "save_layer_maps"):
        if run.get(field) != expected[field]:
            issues.append(f"run config {field}={run.get(field)!r}, expected={expected[field]!r}")
    for field, value in expected["options"].items():
        if run.get(field) != value:
            issues.append(f"run config {field}={run.get(field)!r}, expected={value!r}")
    if expected["offload"] == "cuda":
        recorded_offload = str(run.get("offload", ""))
        if recorded_offload != "cuda" and not recorded_offload.startswith("cuda:"):
            issues.append(f"run config offload={run.get('offload')!r}, expected CUDA device")
    elif run.get("offload") != expected["offload"]:
        issues.append(f"run config offload={run.get('offload')!r}, expected={expected['offload']!r}")
    expected_relations = _expected_relations(key, expected)
    if set(run.get("relations", [])) != expected_relations:
        issues.append(f"run relation mismatch: {run.get('relations')}")

    attention_root = model_root / "attention"
    if not expected["visualizations"]:
        if list((model_root / "generated").rglob("*.png")) or list(
            (model_root / "generated").glob("*.gif")
        ):
            issues.append("--skip-visualization unexpectedly decoded generated media")
        metadata_path = attention_root / "metadata.json"
        raw_files = sorted((attention_root / "raw").rglob("*.pt")) if attention_root.exists() else []
        component_raw_files = (
            sorted((attention_root / "raw-components").rglob("*.pt"))
            if attention_root.exists()
            else []
        )
        if expected["raw_tensors"]:
            if not metadata_path.is_file():
                issues.append("--save-raw with skipped PNGs did not write metadata")
            else:
                metadata = _json(metadata_path)
                if len(raw_files) != len(metadata["records"]):
                    issues.append(f"raw tensor count={len(raw_files)}, records={len(metadata['records'])}")
                if set(metadata["summary"]["relations"]) != expected_relations:
                    issues.append("skipped-visualization metadata relation mismatch")
                if metadata["summary"].get("text_components") and not component_raw_files:
                    issues.append("token-concatenated text components were not saved separately")
                if any(
                    len(record["metadata"].get("query_indices", [])) != 1
                    or not (1 <= len(record["metadata"].get("key_indices", [])) <= 3)
                    for record in metadata["records"]
                ):
                    issues.append("global query/key selections were not applied to every raw record")
            if not raw_files:
                issues.append("--save-raw with skipped PNGs did not write .pt files")
            elif not bool(torch.isfinite(torch.load(raw_files[0], map_location="cpu", weights_only=True)).all()):
                issues.append("non-finite raw tensor in skipped-visualization profile")
        elif attention_root.exists():
            issues.append("--skip-visualization unexpectedly created an attention directory")
        if (attention_root / "visuals").exists():
            issues.append("--skip-visualization unexpectedly wrote attention PNGs")
        if any(
            shape[1] != 2
            for stats in tensor_validation.get("relations", {}).values()
            for shape in stats.get("shape_variants", [])
        ):
            issues.append("--heads 0,2 did not retain exactly two heads")
        return {
            "status": "pass" if not issues else "fail",
            "issues": issues,
            "records": sum(item.get("records", 0) for item in tensor_validation.get("relations", {}).values()),
            "raw_files": len(raw_files),
            "component_raw_files": len(component_raw_files),
        }

    metadata_path = attention_root / "metadata.json"
    manifest_path = attention_root / "visuals" / "visualization.json"
    if not metadata_path.is_file() or not manifest_path.is_file():
        issues.append("missing attention metadata or visualization manifest")
        return {"status": "fail", "issues": issues}
    metadata = _json(metadata_path)
    manifest = _json(manifest_path)
    if set(metadata["summary"]["relations"]) != expected_relations:
        issues.append("attention metadata relation mismatch")
    spatial_relation = "video->video" if MODEL_CONFIGS[key].modality == "video" else "image->image"
    spatial_records = [
        record for record in metadata["records"] if record["relation"] == spatial_relation
    ]
    if spatial_records:
        expected_query_count = {"fast-last": 3, "exact-all-layer-raw": 1}.get(profile_name)
        if expected_query_count is not None and any(
            len(record["metadata"].get("query_indices", [])) != expected_query_count
            for record in spatial_records
        ):
            issues.append(
                f"{spatial_relation} query selection did not retain {expected_query_count} patches"
            )
        if profile_name == "keypatch-visual" and any(
            len(record["metadata"].get("key_indices", [])) != 3 for record in spatial_records
        ):
            issues.append(f"{spatial_relation} key selection did not retain 3 patches")
    if manifest.get("warnings"):
        issues.extend(f"visualizer warning: {warning}" for warning in manifest["warnings"])

    timestep_count = len(metadata["summary"].get("timesteps", []))
    if profile_name == "exact-all-layer-raw":
        if not 1 <= timestep_count <= expected["options"]["steps"]:
            issues.append(
                f"all-timestep profile retained {timestep_count} timesteps, expected between "
                f"1 and {expected['options']['steps']}"
            )
    elif timestep_count != 1:
        issues.append(f"last-timestep profile retained {timestep_count} timesteps, expected 1")

    raw_files = sorted((attention_root / "raw").rglob("*.pt"))
    component_raw_files = sorted((attention_root / "raw-components").rglob("*.pt"))
    if expected["raw_tensors"]:
        if len(raw_files) != len(metadata["records"]):
            issues.append(f"raw tensor count={len(raw_files)}, records={len(metadata['records'])}")
        if metadata["summary"].get("text_components") and not component_raw_files:
            issues.append("token-concatenated text components were not saved separately")
        sampled_relations = set()
        for path in raw_files:
            relation = path.parent.name
            if relation in sampled_relations:
                continue
            sampled_relations.add(relation)
            try:
                value = torch.load(path, map_location="cpu", weights_only=True)
                if value.ndim != 4 or not bool(torch.isfinite(value).all()):
                    issues.append(f"invalid raw tensor {path}")
                if profile_name == "exact-all-layer-raw" and value.shape[1] <= 1:
                    issues.append(f"--heads all unexpectedly retained one head in {path}")
            except Exception as exc:
                issues.append(f"cannot load raw tensor {path}: {type(exc).__name__}: {exc}")
    elif raw_files:
        issues.append(f"raw tensors are opt-in but {len(raw_files)} .pt files were written")

    visual_files = []
    for relation in expected_relations:
        relation_manifest = manifest.get("relations", {}).get(relation)
        if not relation_manifest or not relation_manifest.get("aggregate"):
            issues.append(f"{relation}: missing aggregate visualization")
            continue
        aggregate_files = [Path(item) for item in relation_manifest["aggregate"].get("files", [])]
        if not aggregate_files:
            issues.append(f"{relation}: empty aggregate visualization")
        visual_files.extend(aggregate_files)
        if expected["save_layer_maps"]:
            expected_records = metadata["summary"]["relations"][relation]
            layer_records = relation_manifest.get("records", [])
            if len(layer_records) != expected_records:
                issues.append(f"{relation}: layer visuals={len(layer_records)}, records={expected_records}")
            for item in layer_records:
                files = [Path(path) for path in item.get("files", [])]
                if not files:
                    issues.append(f"{relation}: empty timestep/layer visualization")
                visual_files.extend(files)

    for path in dict.fromkeys(visual_files):
        if not path.is_file():
            issues.append(f"missing visualization file {path}")
        elif path.suffix.lower() in {".png", ".gif"}:
            _verify_image(path, issues)
        if "maps" in path.parts and "overlays" in path.parts:
            issues.append(f"overlay leaked into maps directory: {path}")

    config = MODEL_CONFIGS[key]
    if config.modality == "video":
        generated = sorted((model_root / "generated").glob("*.gif"))
        frame_dirs = [path for path in (model_root / "generated").iterdir() if path.is_dir()]
        if len(generated) != 1 or len(frame_dirs) != 1 or not list(frame_dirs[0].glob("frame-*.png")):
            issues.append("expected one generated GIF and one non-empty frame directory")
        else:
            _verify_image(generated[0], issues)
            for path in frame_dirs[0].glob("frame-*.png"):
                _verify_image(path, issues)
    else:
        generated = sorted((model_root / "generated").glob("*.png"))
        if len(generated) != 1:
            issues.append(f"expected one generated PNG, found {len(generated)}")
        else:
            _verify_image(generated[0], issues)

    if profile_name == "keypatch-visual":
        slug = spatial_relation.replace("->", "-to-")
        batch = attention_root / "visuals" / slug / "aggregate" / "batch-000"
        extension = ".gif" if config.modality == "video" else ".png"
        maps = sorted((batch / "maps").glob(f"key-patch-*{extension}"))
        overlays = sorted((batch / "overlays").glob(f"key-patch-*{extension}"))
        if len(maps) != 2 or len(overlays) != 2:
            issues.append(
                f"selected key-patch maps/overlays={len(maps)}/{len(overlays)}, expected=2/2"
            )

    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "records": len(metadata["records"]),
        "timesteps": timestep_count,
        "raw_files": len(raw_files),
        "component_raw_files": len(component_raw_files),
        "visual_files": len(set(visual_files)),
    }


def validate_parity(root: Path, key: str) -> dict:
    arrays = {}
    issues = []
    for profile in ("fast-last", "exact-all-layer-raw"):
        generated_root = root / profile / key / "generated"
        if MODEL_CONFIGS[key].modality == "video":
            frame_dirs = [path for path in generated_root.iterdir() if path.is_dir()]
            files = sorted(frame_dirs[0].glob("frame-*.png")) if len(frame_dirs) == 1 else []
            if not files:
                issues.append(f"{profile}: generated video frames missing")
                continue
            arrays[profile] = np.stack(
                [np.asarray(Image.open(path).convert("RGB")) for path in files]
            )
        else:
            files = sorted(generated_root.glob("*.png"))
            if len(files) != 1:
                issues.append(f"{profile}: generated PNG missing")
                continue
            with Image.open(files[0]) as image:
                arrays[profile] = np.asarray(image.convert("RGB"))
    maximum = 0
    if len(arrays) == 2:
        baseline = arrays["fast-last"].astype(np.int16)
        for profile in ("exact-all-layer-raw",):
            if arrays[profile].shape != baseline.shape:
                issues.append(f"{profile}: generated shape differs from fast mode")
                continue
            maximum = max(maximum, int(np.abs(arrays[profile].astype(np.int16) - baseline).max()))
        if maximum != 0:
            issues.append(f"capture mode changed generated pixels (max_abs={maximum})")
    return {"status": "pass" if not issues else "fail", "issues": issues, "max_abs": maximum}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the per-model CLI option coverage matrix.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--model", default="all", help="Model key, comma-list, or 'all'.")
    args = parser.parse_args()
    selected_models = model_keys(args.model)

    report = {"status": "pass", "profiles": {}, "parity": {}, "guards": {}}
    for profile in SUCCESS_PROFILES:
        report["profiles"][profile] = {}
        for key in selected_models:
            result = validate_success_profile(args.output_dir, profile, key)
            report["profiles"][profile][key] = result
            if result["status"] != "pass":
                report["status"] = "fail"
            print(f"[{result['status'].upper()}] {profile}/{key}: records={result.get('records', 0)}")
            for issue in result["issues"]:
                print(f"  issue: {issue}")

    for key in selected_models:
        result = validate_parity(args.output_dir, key)
        report["parity"][key] = result
        if result["status"] != "pass":
            report["status"] = "fail"
        print(f"[{result['status'].upper()}] mode parity/{key}: max_abs={result['max_abs']}")

        log_path = args.output_dir / "guard" / f"{key}.log"
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        passed = "Refusing to materialize" in log and "failed: RuntimeError" in log
        report["guards"][key] = {
            "status": "pass" if passed else "fail",
            "limit_message": "Refusing to materialize" in log,
        }
        if not passed:
            report["status"] = "fail"
        print(f"[{'PASS' if passed else 'FAIL'}] map-size guard/{key}")

    report_path = args.report or args.output_dir / "option-matrix-validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"report: {report_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
