from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image

from common import MODEL_CONFIGS


PATCH_PATTERN = re.compile(r"query-patch-(\d+)-y(\d+)-x(\d+)\.png$")


def _verify_png(path: Path, issues: list[str]) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        issues.append(f"invalid PNG {path}: {type(exc).__name__}: {exc}")


def validate_model(model_root: Path, key: str) -> dict:
    config = MODEL_CONFIGS[key]
    issues: list[str] = []
    attention_root = model_root / "attention"
    metadata_path = attention_root / "metadata.json"
    manifest_path = attention_root / "visuals" / "visualization.json"
    run_config_path = model_root / "run-config.json"
    required = (metadata_path, manifest_path, run_config_path)
    if any(not path.is_file() for path in required):
        return {"status": "fail", "issues": ["missing metadata, manifest, or run config"]}

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    expected_relations = set(config.relations)
    actual_relations = set(metadata["summary"]["relations"])
    if actual_relations != expected_relations:
        issues.append(f"relation mismatch: expected={sorted(expected_relations)}, actual={sorted(actual_relations)}")
    if len(metadata["summary"].get("timesteps", [])) != 1:
        issues.append("capture did not retain exactly one final timestep")
    if manifest.get("warnings"):
        issues.extend(f"visualizer warning: {item}" for item in manifest["warnings"])

    if run_config.get("steps") != config.default_steps:
        issues.append(f"steps={run_config.get('steps')}, expected={config.default_steps}")
    if [run_config.get("height"), run_config.get("width")] != [
        config.default_height,
        config.default_width,
    ]:
        issues.append("non-native generation resolution")
    if run_config.get("capture_timesteps") != "last":
        issues.append("capture_timesteps is not 'last'")
    if run_config.get("raw_tensors"):
        issues.append("run unexpectedly enabled raw tensors")

    records_by_relation: dict[str, list[dict]] = {}
    for record in metadata["records"]:
        records_by_relation.setdefault(record["relation"], []).append(record)
        if not record.get("has_data") or record.get("has_qk"):
            issues.append(f"{record['layer_id']} did not finish as a computed attention record")
        if record.get("metadata", {}).get("query_indices") is not None:
            issues.append(f"{record['relation']} unexpectedly selected query patches")
        if record.get("metadata", {}).get("key_indices") is not None:
            issues.append(f"{record['relation']} unexpectedly selected key patches")

    stats: dict[str, dict] = {}
    for relation in config.relations:
        relation_slug = relation.replace("->", "-to-")
        batch_root = attention_root / "visuals" / relation_slug / "aggregate" / "batch-000"
        map_root = batch_root / "maps"
        overlay_root = batch_root / "overlays"
        maps = sorted(map_root.glob("*.png"))
        overlays = sorted(overlay_root.glob("*.png")) if overlay_root.is_dir() else []
        relation_manifest = manifest.get("relations", {}).get(relation, {})
        aggregate = relation_manifest.get("aggregate")
        if not aggregate or int(aggregate.get("source_records", 0)) <= 0:
            issues.append(f"{relation}: missing aggregate or source records")
        if not (attention_root / "visuals" / relation_slug / "aggregate" / "overview-batch-000.png").is_file():
            issues.append(f"{relation}: missing overview")
        if not maps:
            issues.append(f"{relation}: no pure maps")

        if relation == "text->text":
            if [path.name for path in maps] != ["attention-matrix.png"]:
                issues.append("text->text: expected exactly one token-labelled attention matrix")
            if overlays:
                issues.append("text->text: overlays should not be produced")
        else:
            if len(maps) != len(overlays):
                issues.append(f"{relation}: map/overlay mismatch {len(maps)} != {len(overlays)}")
            if {path.name for path in maps} != {path.name for path in overlays}:
                issues.append(f"{relation}: map/overlay filenames do not match")

        if relation == "image->image":
            parsed = []
            for path in maps:
                match = PATCH_PATTERN.match(path.name)
                if match is None:
                    issues.append(f"image->image: invalid patch filename {path.name}")
                    continue
                index, row, col = (int(value) for value in match.groups())
                parsed.append((index, row, col))
            indices = [item[0] for item in parsed]
            if indices != list(range(len(parsed))):
                issues.append("image->image: query patch indices are not complete and contiguous")
            candidate_grids = {
                (int(record["q_shape"]["height"]), int(record["q_shape"]["width"]))
                for record in records_by_relation.get(relation, [])
                if record.get("q_shape") and record["q_shape"].get("height") and record["q_shape"].get("width")
            }
            matching_grids = {(height, width) for height, width in candidate_grids if height * width == len(maps)}
            if not matching_grids:
                issues.append(
                    f"image->image: {len(maps)} maps does not match any captured query patch grid {candidate_grids}"
                )
            elif parsed:
                height, width = next(iter(matching_grids))
                if any(index != row * width + col for index, row, col in parsed):
                    issues.append("image->image: filename row/column labels do not match patch indices")

        for path in maps + overlays:
            _verify_png(path, issues)
        stats[relation] = {
            "records": len(records_by_relation.get(relation, [])),
            "maps": len(maps),
            "overlays": len(overlays),
            "source_records": aggregate.get("source_records") if aggregate else None,
        }

    generated = sorted((model_root / "generated").glob("*.png"))
    if len(generated) != 1:
        issues.append(f"expected one generated image, found {len(generated)}")
        generated_size = None
    else:
        _verify_png(generated[0], issues)
        with Image.open(generated[0]) as image:
            generated_size = list(image.size)
        expected_size = [config.default_width, config.default_height]
        if generated_size != expected_size:
            issues.append(f"generated size={generated_size}, expected={expected_size}")
    if list(attention_root.rglob("*.pt")):
        issues.append("raw .pt files were unexpectedly created")

    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "generated_size": generated_size,
        "relations": stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a full model/relation visualization matrix.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--tensor-validation-dir",
        type=Path,
        default=None,
        help="Optional numerical-only run containing per-model tensor-validation.json files.",
    )
    args = parser.parse_args()

    report = {"status": "pass", "models": {}}
    for key in MODEL_CONFIGS:
        result = validate_model(args.output_dir / key, key)
        if args.tensor_validation_dir is not None:
            tensor_path = args.tensor_validation_dir / key / "tensor-validation.json"
            if not tensor_path.is_file():
                result["issues"].append("missing in-memory tensor validation report")
                result["status"] = "fail"
            else:
                tensor_validation = json.loads(tensor_path.read_text(encoding="utf-8"))
                result["tensor_validation"] = tensor_validation
                if tensor_validation.get("status") != "pass":
                    result["issues"].append("in-memory tensor validation failed")
                    result["issues"].extend(tensor_validation.get("issues", []))
                    result["status"] = "fail"
        report["models"][key] = result
        if result["status"] != "pass":
            report["status"] = "fail"
        print(f"[{result['status'].upper()}] {key}: generated={result.get('generated_size')}")
        for relation, stats in result.get("relations", {}).items():
            print(
                f"  {relation}: records={stats['records']}, maps={stats['maps']}, "
                f"overlays={stats['overlays']}, aggregate_sources={stats['source_records']}"
            )
        for issue in result["issues"]:
            print(f"  issue: {issue}")

    report_path = args.report or args.output_dir / "full-matrix-validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"report: {report_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
