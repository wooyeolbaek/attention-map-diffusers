from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional

from .compute import compute_attention_probs, output_head_count, require_torch, tensor_to_device
from .config import CaptureConfig, normalize_indices
from .records import AttentionLayout, AttentionRecord, iter_supported_relations


class AttentionCollector:
    def __init__(self, config: CaptureConfig):
        self.config = config
        self.records: List[AttentionRecord] = []
        self._step_index: Optional[int] = None
        self._timestep: Optional[float] = None

    @property
    def step_index(self) -> Optional[int]:
        return self._step_index

    @property
    def timestep(self) -> Optional[float]:
        return self._timestep

    def clear(self) -> None:
        self.records.clear()

    def set_step(self, step_index: Optional[int] = None, timestep: Any = None) -> None:
        next_timestep = _to_float(timestep)
        if (
            self.config.timesteps == "last"
            and next_timestep is not None
            and self._timestep is not None
            and next_timestep != self._timestep
        ):
            # Keep generation unchanged while retaining only the most recent
            # denoising step's records. This bounds VRAM for full patch maps.
            self.records.clear()
        self._step_index = step_index
        self._timestep = next_timestep

    def enabled_for(self, layer_id: str, timestep: Any = None) -> bool:
        if self.config.layers != "all":
            patterns = self.config.layers if not isinstance(self.config.layers, str) else [self.config.layers]
            if not any(pattern in layer_id for pattern in patterns):
                return False
        if self.config.timesteps not in {"all", "last"}:
            current = _to_int(timestep if timestep is not None else self._timestep)
            if current not in {int(item) for item in self.config.timesteps}:
                return False
        return True

    def capture_qk(
        self,
        *,
        layer_id: str,
        attention_type: str,
        query,
        key,
        layout: AttentionLayout,
        attention_mask=None,
        scale: Optional[float] = None,
        timestep: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled_for(layer_id, timestep=timestep):
            return

        torch = require_torch()
        timestep_value = _to_float(timestep) if timestep is not None else self._timestep
        metadata = dict(metadata or {})

        for relation in iter_supported_relations(layout, self.config.relations):
            q_range, k_range = layout.relation_ranges(relation)
            q = query[:, :, q_range.start : q_range.end]

            query_selection = self.config.query_indices
            if self.config.relation_query_indices is not None and relation in self.config.relation_query_indices:
                query_selection = self.config.relation_query_indices[relation]
            query_indices = normalize_indices(query_selection, q.shape[-2], q_range.grid)

            key_selection = self.config.key_indices if self.config.key_indices is not None else self.config.tokens
            if self.config.relation_key_indices is not None and relation in self.config.relation_key_indices:
                key_selection = self.config.relation_key_indices[relation]
            key_indices = normalize_indices(key_selection, k_range.length, k_range.grid)

            selected_q_indices = None
            if query_indices is not None:
                selected_q_indices = torch.tensor(query_indices, device=q.device)
                q = q.index_select(-2, selected_q_indices)

            # Attention probabilities must be normalized over the same complete
            # key sequence used by the model. Relation/key selection is applied
            # only after softmax; slicing K first would incorrectly make every
            # relation sum to one on its own.
            output_key_indices = list(range(k_range.start, k_range.end))
            if key_indices is not None:
                output_key_indices = [k_range.start + index for index in key_indices]
            relation_mask = None
            if attention_mask is not None:
                relation_mask = _slice_query_attention_mask(
                    attention_mask,
                    q_range.start,
                    q_range.end,
                    selected_q_indices,
                )

            map_elements = (
                q.shape[0]
                * output_head_count(self.config.heads, q.shape[1])
                * q.shape[-2]
                * len(output_key_indices)
            )
            if (
                self.config.mode in {"exact", "stream"}
                and map_elements > self.config.max_map_elements
                and not self.config.allow_large_maps
            ):
                raise RuntimeError(
                    f"Refusing to materialize {map_elements:,} attention values for {layer_id} {relation}. "
                    "Use query_indices/key_indices, increase max_map_elements, or use mode='fast'."
                )

            data = None
            q_store = q
            k_store = key
            if self.config.mode in {"exact", "stream"}:
                data = compute_attention_probs(
                    q,
                    key,
                    attention_mask=relation_mask,
                    scale=scale,
                    chunk_size=self.config.chunk_size,
                    heads=self.config.heads,
                    key_indices=output_key_indices,
                )
                data = tensor_to_device(data, self.config.offload, self.config.dtype)
                q_store = None
                k_store = None
            else:
                q_store = tensor_to_device(q_store, self.config.offload, self.config.dtype)
                k_store = tensor_to_device(key, self.config.offload, self.config.dtype)
                # Preserve boolean SDPA masks even when captured Q/K tensors
                # are explicitly cast to another storage dtype.
                relation_mask = tensor_to_device(relation_mask, self.config.offload)

            record_metadata = dict(metadata)
            if query_indices is not None:
                record_metadata["query_indices"] = tuple(query_indices)
            if key_indices is not None:
                record_metadata["key_indices"] = tuple(key_indices)
            record_metadata["attention_key_indices"] = tuple(output_key_indices)

            self.records.append(
                AttentionRecord(
                    layer_id=layer_id,
                    relation=relation,
                    attention_type=attention_type,
                    q_range=(q_range.start, q_range.end),
                    k_range=(k_range.start, k_range.end),
                    q_shape=q_range.grid,
                    k_shape=k_range.grid,
                    heads=query.shape[1],
                    dtype=str(query.dtype).replace("torch.", ""),
                    device=str(query.device),
                    step_index=self._step_index,
                    timestep=timestep_value,
                    data=data,
                    q=q_store,
                    k=k_store,
                    attention_mask=relation_mask,
                    scale=scale,
                    metadata=record_metadata,
                )
            )

    def compute(self) -> "AttentionMaps":
        computed = []
        for record in self.records:
            if record.data is not None:
                computed.append(record)
                continue
            if record.q is None or record.k is None:
                computed.append(record)
                continue
            output_key_indices = record.metadata.get("attention_key_indices")
            output_key_count = len(output_key_indices) if output_key_indices is not None else record.k.shape[-2]
            map_elements = (
                record.q.shape[0]
                * output_head_count(self.config.heads, record.q.shape[1])
                * record.q.shape[-2]
                * output_key_count
            )
            if map_elements > self.config.max_map_elements and not self.config.allow_large_maps:
                raise RuntimeError(
                    f"Refusing to compute {map_elements:,} attention values for {record.layer_id} "
                    f"{record.relation}. Use query_indices/key_indices or increase max_map_elements."
                )
            data = compute_attention_probs(
                record.q,
                record.k,
                attention_mask=record.attention_mask,
                scale=record.scale,
                chunk_size=self.config.chunk_size,
                heads=self.config.heads,
                key_indices=record.metadata.get("attention_key_indices"),
            )
            data = tensor_to_device(data, self.config.offload, self.config.dtype)
            computed.append(replace(record, data=data, q=None, k=None, attention_mask=None))
        return AttentionMaps(computed)


class AttentionMaps:
    def __init__(self, records: Iterable[AttentionRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def by_relation(self) -> Dict[str, List[AttentionRecord]]:
        result: Dict[str, List[AttentionRecord]] = {}
        for record in self.records:
            result.setdefault(record.relation, []).append(record)
        return result

    def summary(self) -> Dict[str, Any]:
        by_relation = self.by_relation()
        return {
            "records": len(self.records),
            "relations": {relation: len(items) for relation, items in by_relation.items()},
            "layers": len({record.layer_id for record in self.records}),
            "timesteps": sorted({record.timestep for record in self.records if record.timestep is not None}),
        }

    def validate(self, tolerance: float = 0.02) -> Dict[str, Any]:
        """Validate captured tensors in memory without requiring raw ``.pt`` files."""
        torch = require_torch()
        issues: List[str] = []
        relation_stats: Dict[str, Dict[str, Any]] = {}
        joint_groups: Dict[Any, Dict[str, Any]] = {}

        for relation, records in self.by_relation().items():
            stats = {
                "records": 0,
                "finite": True,
                "minimum": float("inf"),
                "maximum": float("-inf"),
                "shape_variants": set(),
                "row_sum_max_error": 0.0,
            }
            relation_stats[relation] = stats
            for record in records:
                data = record.data
                if data is None or data.ndim != 4:
                    issues.append(f"{relation} {record.layer_id}: missing or non-rank-4 tensor")
                    continue
                stats["records"] += 1
                stats["shape_variants"].add(tuple(int(value) for value in data.shape))
                finite = bool(torch.isfinite(data).all())
                stats["finite"] = stats["finite"] and finite
                if not finite:
                    issues.append(f"{relation} {record.layer_id}: NaN or Inf")
                    continue
                minimum = float(data.amin())
                maximum = float(data.amax())
                stats["minimum"] = min(stats["minimum"], minimum)
                stats["maximum"] = max(stats["maximum"], maximum)
                if minimum < -tolerance or maximum > 1.0 + tolerance:
                    issues.append(
                        f"{relation} {record.layer_id}: probability range [{minimum}, {maximum}]"
                    )

                row_mass = data.sum(dim=-1, dtype=torch.float32)
                keys_complete = record.metadata.get("key_indices") is None
                if record.attention_type == "joint":
                    group_key = (record.layer_id, record.timestep, record.attention_type)
                    query_indices = record.metadata.get("query_indices")
                    if query_indices is None:
                        query_indices = tuple(range(int(data.shape[-2])))
                    joint_groups.setdefault(group_key, {})[relation] = {
                        "mass": row_mass,
                        "query_indices": tuple(int(index) for index in query_indices),
                        "keys_complete": keys_complete,
                    }
                elif record.attention_type == "linear-self":
                    if float(row_mass.max()) > 1.0 + tolerance:
                        issues.append(f"{relation} {record.layer_id}: linear row mass exceeds one")
                elif keys_complete:
                    error = float((row_mass - 1.0).abs().max())
                    stats["row_sum_max_error"] = max(stats["row_sum_max_error"], error)
                    if error > tolerance:
                        issues.append(f"{relation} {record.layer_id}: row-sum error {error}")
                elif float(row_mass.max()) > 1.0 + tolerance:
                    issues.append(f"{relation} {record.layer_id}: selected-key row mass exceeds one")

        joint_stats = {
            "groups": 0,
            "skipped_partial_key_groups": 0,
            "skipped_disjoint_query_groups": 0,
            "image_query_max_error": 0.0,
            "text_query_max_error": 0.0,
        }
        for group in joint_groups.values():
            evaluated = False
            for query_name, left_relation, right_relation in (
                ("image", "image->image", "image->text"),
                ("text", "text->image", "text->text"),
            ):
                if left_relation not in group or right_relation not in group:
                    continue
                if not group[left_relation]["keys_complete"] or not group[right_relation]["keys_complete"]:
                    joint_stats["skipped_partial_key_groups"] += 1
                    continue
                pair = _aligned_query_masses(group[left_relation], group[right_relation])
                if pair is None:
                    joint_stats["skipped_disjoint_query_groups"] += 1
                    continue
                error = float((pair[0] + pair[1] - 1.0).abs().max())
                stat_name = f"{query_name}_query_max_error"
                joint_stats[stat_name] = max(joint_stats[stat_name], error)
                evaluated = True
            if evaluated:
                joint_stats["groups"] += 1
        if joint_stats["image_query_max_error"] > tolerance:
            issues.append(f"joint image-query normalization error {joint_stats['image_query_max_error']}")
        if joint_stats["text_query_max_error"] > tolerance:
            issues.append(f"joint text-query normalization error {joint_stats['text_query_max_error']}")

        for stats in relation_stats.values():
            stats["shape_variants"] = [list(shape) for shape in sorted(stats["shape_variants"])]
        return {
            "status": "pass" if not issues else "fail",
            "tolerance": tolerance,
            "issues": issues,
            "relations": relation_stats,
            "joint_normalization": joint_stats,
        }

    def to_legacy(self, relation: str = "image->text") -> Dict[Any, Dict[str, Any]]:
        legacy: Dict[Any, Dict[str, Any]] = {}
        for record in self.records:
            if record.relation != relation or record.data is None:
                continue
            if record.q_shape is None or record.q_shape.height is None or record.q_shape.width is None:
                continue
            data = record.data
            if data.ndim != 4:
                continue
            batch, heads, query_tokens, key_tokens = data.shape
            if query_tokens != record.q_shape.height * record.q_shape.width:
                continue
            tensor = data.reshape(batch, heads, record.q_shape.height, record.q_shape.width, key_tokens)
            timestep = record.timestep if record.timestep is not None else "unknown"
            legacy.setdefault(timestep, {})[record.layer_id] = tensor
        return legacy

    def save(
        self,
        base_dir: str,
        *,
        tokenizer=None,
        prompts=None,
        images=None,
        relation: Optional[str] = None,
        unconditional: Optional[bool] = None,
        visualization_config=None,
        save_raw: bool = False,
        save_visualizations: bool = True,
    ) -> Dict[str, Any]:
        """Save metadata and relation-aware visualizations.

        When ``relation`` is omitted every captured relation is rendered below
        ``visuals/<query>-to-<key>``. Passing a relation keeps the legacy
        single-relation behavior while using the same renderer. When generated
        ``images`` are provided, a doubled classifier-free-guidance batch is
        detected automatically and only its conditional half is visualized.
        Raw ``.pt`` tensors are optional and are written only when
        ``save_raw=True``.
        """
        os.makedirs(base_dir, exist_ok=True)
        selected_records = [
            record for record in self.records if relation is None or record.relation == relation
        ]
        with open(os.path.join(base_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "summary": self.summary(),
                    "selected_relation": relation,
                    "records": [record.as_metadata() for record in selected_records],
                },
                handle,
                indent=2,
                sort_keys=True,
            )

        if save_raw:
            torch = require_torch()
            for index, record in enumerate(selected_records):
                if record.data is None:
                    continue
                relation_dir = os.path.join(base_dir, "raw", record.relation.replace("->", "-to-"))
                os.makedirs(relation_dir, exist_ok=True)
                safe_layer = record.layer_id.replace("/", "_").replace(".", "_")
                timestep = "none" if record.timestep is None else str(record.timestep).replace(".", "_")
                filename = f"{index:04d}-{timestep}-{safe_layer}.pt"
                torch.save(record.data, os.path.join(relation_dir, filename))

        if not save_visualizations:
            return {"relations": {}, "warnings": []}

        from attention_map_diffusers.visualization import AttentionVisualizer

        visualizer = AttentionVisualizer(visualization_config)
        relations = None if relation is None else [relation]
        return visualizer.save(
            selected_records,
            os.path.join(base_dir, "visuals"),
            tokenizer=tokenizer,
            prompts=prompts,
            images=images,
            relations=relations,
            unconditional=unconditional,
        )


def _aligned_query_masses(left: Dict[str, Any], right: Dict[str, Any]):
    """Align relation masses by their modality-relative query-token indices."""
    torch = require_torch()
    left_positions = {index: position for position, index in enumerate(left["query_indices"])}
    right_positions = {index: position for position, index in enumerate(right["query_indices"])}
    shared = sorted(set(left_positions).intersection(right_positions))
    if not shared:
        return None
    left_index = torch.tensor(
        [left_positions[index] for index in shared],
        device=left["mass"].device,
        dtype=torch.long,
    )
    right_index = torch.tensor(
        [right_positions[index] for index in shared],
        device=right["mass"].device,
        dtype=torch.long,
    )
    return (
        left["mass"].index_select(-1, left_index),
        right["mass"].index_select(-1, right_index),
    )


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "flatten"):
        value = value.flatten()[0]
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    value = _to_float(value)
    return None if value is None else int(value)


def _slice_query_attention_mask(
    attention_mask,
    q_start: int,
    q_end: int,
    selected_q_indices=None,
):
    if attention_mask is None:
        return None
    mask = attention_mask
    if mask.ndim < 4:
        return None
    if mask.shape[-2] != 1:
        mask = mask[..., q_start:q_end, :]
    if selected_q_indices is not None and mask.shape[-2] != 1:
        mask = mask.index_select(-2, selected_q_indices)
    return mask
