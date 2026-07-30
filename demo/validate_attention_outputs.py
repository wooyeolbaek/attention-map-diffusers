from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
from PIL import Image

from common import MODEL_CONFIGS


def validate_model(model_dir: Path, expected_relations: Iterable[str]) -> Dict[str, Any]:
    attention_dir = model_dir / "attention"
    metadata_path = attention_dir / "metadata.json"
    visual_path = attention_dir / "visuals" / "visualization.json"
    issues = []

    if not metadata_path.is_file() or not visual_path.is_file():
        return {"status": "fail", "issues": ["missing metadata or visualization manifest"]}

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    visual = json.loads(visual_path.read_text(encoding="utf-8"))
    expected_relations = tuple(expected_relations)
    actual_relations = set(metadata["summary"]["relations"])
    if actual_relations != set(expected_relations):
        issues.append(f"relation mismatch: expected={expected_relations}, actual={sorted(actual_relations)}")
    if visual.get("warnings"):
        issues.extend(f"visualizer: {warning}" for warning in visual["warnings"])

    relation_stats: Dict[str, Dict[str, Any]] = {
        relation: {
            "records": 0,
            "elements": 0,
            "min": float("inf"),
            "max": float("-inf"),
            "mean": 0.0,
            "std": 0.0,
            "constant_records": 0,
            "linear_inactive_rows": 0,
            "shape_variants": set(),
            "row_sum_max_error": 0.0,
        }
        for relation in expected_relations
    }
    sample_moments = defaultdict(lambda: [0, 0.0, 0.0])
    grouped_masses: Dict[Tuple[Any, ...], Dict[str, Any]] = defaultdict(dict)

    for index, record in enumerate(metadata["records"]):
        relation = record["relation"]
        if relation not in relation_stats:
            continue
        raw_dir = attention_dir / "raw" / relation.replace("->", "-to-")
        matches = list(raw_dir.glob(f"{index:04d}-*.pt"))
        if len(matches) != 1:
            issues.append(f"{relation} record {index}: expected one raw tensor, found {len(matches)}")
            continue

        tensor = torch.load(matches[0], map_location="cpu", weights_only=True)
        stats = relation_stats[relation]
        stats["records"] += 1
        stats["elements"] += tensor.numel()
        stats["shape_variants"].add(tuple(tensor.shape))

        if tensor.ndim != 4:
            issues.append(f"{relation} record {index}: rank {tensor.ndim}, expected 4")
            continue
        if tensor.shape[1] != 1:
            issues.append(f"{relation} record {index}: heads={tensor.shape[1]}, expected reduced head=1")

        query_indices = record["metadata"].get("query_indices")
        key_indices = record["metadata"].get("key_indices")
        expected_q = len(query_indices) if query_indices is not None else record["q_range"][1] - record["q_range"][0]
        expected_k = len(key_indices) if key_indices is not None else record["k_range"][1] - record["k_range"][0]
        if tuple(tensor.shape[-2:]) != (expected_q, expected_k):
            issues.append(
                f"{relation} record {index}: shape={tuple(tensor.shape[-2:])}, expected={(expected_q, expected_k)}"
            )

        finite = bool(torch.isfinite(tensor).all())
        if not finite:
            issues.append(f"{relation} record {index}: NaN or Inf")
            continue
        value_min = float(tensor.min())
        value_max = float(tensor.max())
        stats["min"] = min(stats["min"], value_min)
        stats["max"] = max(stats["max"], value_max)
        if value_min < -1e-5:
            issues.append(f"{relation} record {index}: negative attention value {value_min}")
        if value_max - value_min <= 1e-8:
            stats["constant_records"] += 1

        flat = tensor.float().flatten()
        if flat.numel() > 250_000:
            flat = flat[:: max(1, flat.numel() // 250_000)]
        moments = sample_moments[relation]
        moments[0] += flat.numel()
        moments[1] += float(flat.sum())
        moments[2] += float((flat * flat).sum())

        row_mass = tensor.float().sum(dim=-1)
        group_key = (record["layer_id"], record["timestep"], record["attention_type"])
        grouped_masses[group_key][relation] = {
            "mass": row_mass,
            "query_indices": query_indices,
        }

        if record["attention_type"] == "linear-self":
            scaled_mass = row_mass * int(record["heads"])
            quantization_error = float((scaled_mass - scaled_mass.round()).abs().max())
            if float(row_mass.max()) > 1.02 or quantization_error > 0.02:
                issues.append(
                    f"{relation} record {index}: invalid linear-attention head mass "
                    f"(max={float(row_mass.max()):.6f}, quant_error={quantization_error:.6f})"
                )
            stats["linear_inactive_rows"] += int((scaled_mass.round() < int(record["heads"])).sum())
        elif len(expected_relations) == 2:
            error = float((row_mass - 1.0).abs().max())
            stats["row_sum_max_error"] = max(stats["row_sum_max_error"], error)

    joint_errors = {"image_query": 0.0, "text_query": 0.0, "groups": 0}
    if len(expected_relations) == 4:
        required = {"image->text", "text->image", "image->image", "text->text"}
        for group in grouped_masses.values():
            if not required.issubset(group):
                continue
            image_index = group["image->image"]["query_indices"]
            if not image_index or len(image_index) != 1:
                issues.append("joint image->image record did not contain one selected query patch")
                continue
            image_text_mass = group["image->text"]["mass"][..., int(image_index[0])]
            image_mass = image_text_mass + group["image->image"]["mass"][..., 0]
            text_mass = group["text->image"]["mass"] + group["text->text"]["mass"]
            joint_errors["image_query"] = max(
                joint_errors["image_query"], float((image_mass - 1.0).abs().max())
            )
            joint_errors["text_query"] = max(
                joint_errors["text_query"], float((text_mass - 1.0).abs().max())
            )
            joint_errors["groups"] += 1
        if joint_errors["groups"] == 0:
            issues.append("no complete four-relation joint groups found")

    for relation, stats in relation_stats.items():
        count, total, squared = sample_moments[relation]
        if count:
            mean = total / count
            variance = max(0.0, squared / count - mean * mean)
            stats["mean"] = mean
            stats["std"] = variance**0.5
        stats["shape_variants"] = [list(shape) for shape in sorted(stats["shape_variants"])]
        if stats["records"] == 0:
            issues.append(f"{relation}: no raw records")
        if stats["constant_records"]:
            issues.append(f"{relation}: {stats['constant_records']} constant raw records")
        if len(expected_relations) == 2 and stats["row_sum_max_error"] > 0.02:
            issues.append(f"{relation}: row-sum error {stats['row_sum_max_error']:.6f}")

        visual_relation = visual.get("relations", {}).get(relation)
        if not visual_relation or not visual_relation.get("aggregate"):
            issues.append(f"{relation}: missing aggregate visual")
            continue
        aggregate = visual_relation["aggregate"]
        expected_records = metadata["summary"]["relations"][relation]
        if aggregate.get("source_records") != expected_records:
            issues.append(
                f"{relation}: aggregate uses {aggregate.get('source_records')}/{expected_records} records"
            )
        files = [Path(path) for path in aggregate.get("files", [])]
        if not files or any(not path.is_file() for path in files):
            issues.append(f"{relation}: missing aggregate PNG files")
            continue
        names = [path.name for path in files]
        expected_fragment = {
            "text->text": "attention-matrix.png",
            "image->image": "query-patch-",
            "image->text": "token-",
            "text->image": "token-",
        }[relation]
        if not any(expected_fragment in name for name in names):
            issues.append(f"{relation}: expected visual type {expected_fragment!r} not found")
        if relation != "text->text" and not any(path.parent.name == "overlays" for path in files):
            issues.append(f"{relation}: no generated-image overlay")
        if any("batch-001" in str(path) for path in files):
            issues.append(f"{relation}: unconditional CFG batch leaked into aggregate visuals")
        for path in files:
            if path.suffix.lower() != ".png":
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                issues.append(f"{relation}: invalid PNG {path.name}: {exc}")

    generated = sorted((model_dir / "generated").glob("*.png"))
    if not generated:
        issues.append("missing generated image")
        generated_size = None
    else:
        with Image.open(generated[0]) as image:
            generated_size = list(image.size)
            image.verify()

    if len(expected_relations) == 4:
        if joint_errors["image_query"] > 0.02 or joint_errors["text_query"] > 0.02:
            issues.append(f"joint relation mass error: {joint_errors}")

    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "summary": metadata["summary"],
        "generated_size": generated_size,
        "relations": relation_stats,
        "joint_mass": joint_errors if len(expected_relations) == 4 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate raw and rendered attention-map demo outputs.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    report = {"models": {}, "status": "pass"}
    for model, config in MODEL_CONFIGS.items():
        result = validate_model(args.output_dir / model, config.relations)
        report["models"][model] = result
        if result["status"] != "pass":
            report["status"] = "fail"
        print(f"[{result['status'].upper()}] {model}")
        for relation, stats in result.get("relations", {}).items():
            print(
                f"  {relation}: records={stats['records']}, shapes={stats['shape_variants']}, "
                f"range=({stats['min']:.6g}, {stats['max']:.6g}), std={stats['std']:.6g}"
            )
        for issue in result["issues"]:
            print(f"  issue: {issue}")

    report_path = args.report or args.output_dir / "validation-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"report: {report_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
