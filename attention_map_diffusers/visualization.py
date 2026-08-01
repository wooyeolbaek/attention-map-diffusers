from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .capture.compute import require_torch
from .capture.records import AttentionRecord, TokenGrid, parse_relation


@dataclass(frozen=True)
class VisualizationConfig:
    """Controls relation-aware attention-map rendering.

    ``max_items`` only limits the number of PNGs written for a single record;
    it never changes the captured tensor or the aggregate. Text padding tokens
    are hidden automatically when a tokenizer is available. Spatial maps and
    overlays contain only the image by default; set ``show_spatial_labels`` to
    add the token/patch title and patch-grid annotation.
    """

    spatial_long_side: int = 512
    max_items: Optional[int] = None
    overview_columns: int = 4
    overview_items: int = 16
    overlay_alpha: float = 0.55
    colormap: str = "magma"
    save_layer_maps: bool = False
    save_aggregate: bool = True
    save_overlays: bool = True
    show_spatial_labels: bool = False
    max_matrix_tokens: Optional[int] = 128
    video_fps: int = 8

    def __post_init__(self) -> None:
        if self.spatial_long_side <= 0:
            raise ValueError("spatial_long_side must be positive.")
        if self.max_items is not None and self.max_items <= 0:
            raise ValueError("max_items must be positive or None.")
        if self.overview_columns <= 0:
            raise ValueError("overview_columns must be positive.")
        if self.overview_items <= 0:
            raise ValueError("overview_items must be positive.")
        if not 0.0 <= self.overlay_alpha <= 1.0:
            raise ValueError("overlay_alpha must be between 0 and 1.")
        if self.colormap not in _COLORMAPS:
            raise ValueError(f"Unknown colormap {self.colormap!r}; choose one of {sorted(_COLORMAPS)}.")
        if self.video_fps <= 0:
            raise ValueError("video_fps must be positive.")


class AttentionVisualizer:
    """Render all supported modality relations through one consistent API."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()

    def save(
        self,
        records: Iterable[AttentionRecord],
        base_dir: str,
        *,
        tokenizer=None,
        prompts=None,
        images=None,
        videos=None,
        relations: Optional[Sequence[str]] = None,
        unconditional: Optional[bool] = None,
    ) -> Dict[str, Any]:
        records = [record for record in records if record.data is not None]
        if relations is not None:
            relation_set = set(relations)
            records = [record for record in records if record.relation in relation_set]
        records = _select_guidance_records(records, unconditional)

        output_root = Path(base_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        resolver = _TokenResolver(tokenizer, prompts, records)
        source_images = _as_pil_images(images)
        source_videos = _as_pil_videos(videos)
        conditional_batch_count = len(source_images) or len(source_videos) or None
        manifest: Dict[str, Any] = {"relations": {}, "warnings": []}

        by_relation: Dict[str, List[AttentionRecord]] = {}
        for record in records:
            by_relation.setdefault(record.relation, []).append(record)

        for relation, relation_records in by_relation.items():
            relation_root = output_root / _relation_slug(relation)
            relation_manifest = {"records": [], "aggregate": None}
            manifest["relations"][relation] = relation_manifest

            if self.config.save_layer_maps:
                for record_index, record in enumerate(relation_records):
                    data = _prepare_data(record.data, unconditional, conditional_batch_count)
                    if data is None:
                        continue
                    record_dir = relation_root / "steps" / _step_slug(record) / _safe_component(record.layer_id)
                    result = self._render_record(
                        record,
                        data,
                        record_dir,
                        resolver=resolver,
                        source_images=source_images,
                        source_videos=source_videos,
                    )
                    result.update({"index": record_index, "layer": record.layer_id, "step": record.step_index,
                                   "timestep": record.timestep})
                    relation_manifest["records"].append(result)
                    manifest["warnings"].extend(result.pop("warnings", []))

            if self.config.save_aggregate:
                aggregate = _aggregate_compatible(
                    relation_records, unconditional, conditional_batch_count
                )
                if aggregate is not None:
                    aggregate_record, aggregate_data, count = aggregate
                    aggregate_dir = relation_root / "aggregate"
                    result = self._render_record(
                        aggregate_record,
                        aggregate_data,
                        aggregate_dir,
                        resolver=resolver,
                        source_images=source_images,
                        source_videos=source_videos,
                        title_prefix=f"aggregate ({count} maps)",
                    )
                    result["source_records"] = count
                    relation_manifest["aggregate"] = result
                    manifest["warnings"].extend(result.pop("warnings", []))

        with (output_root / "visualization.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        return manifest

    def _render_record(
        self,
        record: AttentionRecord,
        data,
        output_dir: Path,
        *,
        resolver: "_TokenResolver",
        source_images: Sequence[Any],
        source_videos: Sequence[Sequence[Any]],
        title_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        relation = record.relation
        warnings: List[str] = []
        files: List[str] = []
        batch_overviews: Dict[int, List[Tuple[str, Any]]] = {}

        for batch in range(data.shape[0]):
            batch_dir = output_dir / f"batch-{batch:03d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            map_dir = batch_dir / "maps"
            map_dir.mkdir(parents=True, exist_ok=True)
            overlay_dir = batch_dir / "overlays"
            source = source_images[batch] if batch < len(source_images) else None
            source_video = source_videos[batch] if batch < len(source_videos) else []
            prefix = _record_title(record, relation, title_prefix)

            if relation == "text->text":
                for q_component, positions_q, labels_q in resolver.axes(
                    record, "query", batch, data.shape[1]
                ):
                    for k_component, positions_k, labels_k in resolver.axes(
                        record, "key", batch, data.shape[2]
                    ):
                        limited_q = _limit_axis(
                            positions_q, labels_q, self.config.max_matrix_tokens
                        )
                        limited_k = _limit_axis(
                            positions_k, labels_k, self.config.max_matrix_tokens
                        )
                        matrix = _select_matrix(data[batch], limited_q[0], limited_k[0])
                        component_label = _component_pair(q_component, k_component)
                        title = prefix if component_label is None else f"{prefix}\n{component_label}"
                        image = _render_text_matrix(
                            matrix, limited_q[1], limited_k[1], title, self.config.colormap
                        )
                        destination = map_dir
                        if component_label is not None:
                            destination = map_dir / _safe_component(component_label)
                            destination.mkdir(parents=True, exist_ok=True)
                        path = destination / "attention-matrix.png"
                        image.save(path)
                        files.append(str(path))
                        self._append_overview(
                            batch_overviews, batch, component_label or "token × token", image
                        )

            elif relation == "image->text":
                if not _is_full_grid(record.q_shape, data.shape[1], record.metadata.get("query_indices")):
                    warnings.append(_unsupported_warning(record, "image query axis cannot be reshaped to its grid"))
                    continue
                for component, positions, labels in resolver.axes(record, "key", batch, data.shape[2]):
                    component_maps, component_overlays = _component_dirs(
                        map_dir, overlay_dir, component
                    )
                    for data_position, label in _limited_items(positions, labels, self.config.max_items):
                        spatial = data[batch, :, data_position].reshape(
                            int(record.q_shape.height), int(record.q_shape.width)
                        )
                        stem = f"token-{_safe_component(label, max_length=80)}"
                        title = f"{prefix}\nkey token: {label}"
                        rendered = self._save_spatial(
                            spatial, component_maps, component_overlays, stem, title, source
                        )
                        files.extend(rendered[0])
                        self._append_overview(batch_overviews, batch, label, rendered[1])

            elif relation == "text->image":
                if not _is_full_grid(record.k_shape, data.shape[2], record.metadata.get("key_indices")):
                    warnings.append(_unsupported_warning(record, "image key axis cannot be reshaped to its grid"))
                    continue
                for component, positions, labels in resolver.axes(record, "query", batch, data.shape[1]):
                    component_maps, component_overlays = _component_dirs(
                        map_dir, overlay_dir, component
                    )
                    for data_position, label in _limited_items(positions, labels, self.config.max_items):
                        spatial = data[batch, data_position, :].reshape(
                            int(record.k_shape.height), int(record.k_shape.width)
                        )
                        stem = f"token-{_safe_component(label, max_length=80)}"
                        title = f"{prefix}\nquery token: {label}"
                        rendered = self._save_spatial(
                            spatial, component_maps, component_overlays, stem, title, source
                        )
                        files.extend(rendered[0])
                        self._append_overview(batch_overviews, batch, label, rendered[1])

            elif relation == "image->image":
                query_full = _is_full_grid(record.q_shape, data.shape[1], record.metadata.get("query_indices"))
                key_full = _is_full_grid(record.k_shape, data.shape[2], record.metadata.get("key_indices"))
                if key_full:
                    query_positions, query_labels = _patch_axis(record, "query", data.shape[1])
                    for data_position, label in _limited_items(
                        query_positions, query_labels, self.config.max_items
                    ):
                        spatial = data[batch, data_position, :].reshape(
                            int(record.k_shape.height), int(record.k_shape.width)
                        )
                        stem = f"query-patch-{_safe_component(label)}"
                        title = f"{prefix}\nquery patch: {label} → key patch grid"
                        rendered = self._save_spatial(spatial, map_dir, overlay_dir, stem, title, source)
                        files.extend(rendered[0])
                        self._append_overview(
                            batch_overviews, batch, f"query {label}", rendered[1]
                        )
                elif query_full:
                    key_positions, key_labels = _patch_axis(record, "key", data.shape[2])
                    for data_position, label in _limited_items(key_positions, key_labels, self.config.max_items):
                        spatial = data[batch, :, data_position].reshape(
                            int(record.q_shape.height), int(record.q_shape.width)
                        )
                        stem = f"key-patch-{_safe_component(label)}"
                        title = f"{prefix}\nattention received by key patch: {label}"
                        rendered = self._save_spatial(spatial, map_dir, overlay_dir, stem, title, source)
                        files.extend(rendered[0])
                        self._append_overview(
                            batch_overviews, batch, f"key {label}", rendered[1]
                        )
                else:
                    warnings.append(_unsupported_warning(record, "neither image axis contains a complete patch grid"))

            elif relation == "video->text":
                if not _is_full_grid(record.q_shape, data.shape[1], record.metadata.get("query_indices")):
                    warnings.append(_unsupported_warning(record, "video query axis cannot be reshaped to T×H×W"))
                    continue
                for component, positions, labels in resolver.axes(record, "key", batch, data.shape[2]):
                    component_maps, component_overlays = _component_dirs(
                        map_dir, overlay_dir, component
                    )
                    for data_position, label in _limited_items(positions, labels, self.config.max_items):
                        volume = data[batch, :, data_position].reshape(
                            int(record.q_shape.frames or 1),
                            int(record.q_shape.height),
                            int(record.q_shape.width),
                        )
                        stem = f"token-{_safe_component(label, max_length=80)}"
                        title = f"{prefix}\nkey token: {label}"
                        rendered = self._save_video_spatial(
                            volume, component_maps, component_overlays, stem, title, source_video
                        )
                        files.extend(rendered[0])
                        self._append_overview(batch_overviews, batch, label, rendered[1])

            elif relation == "text->video":
                if not _is_full_grid(record.k_shape, data.shape[2], record.metadata.get("key_indices")):
                    warnings.append(_unsupported_warning(record, "video key axis cannot be reshaped to T×H×W"))
                    continue
                for component, positions, labels in resolver.axes(record, "query", batch, data.shape[1]):
                    component_maps, component_overlays = _component_dirs(
                        map_dir, overlay_dir, component
                    )
                    for data_position, label in _limited_items(positions, labels, self.config.max_items):
                        volume = data[batch, data_position, :].reshape(
                            int(record.k_shape.frames or 1),
                            int(record.k_shape.height),
                            int(record.k_shape.width),
                        )
                        stem = f"token-{_safe_component(label, max_length=80)}"
                        title = f"{prefix}\nquery token: {label}"
                        rendered = self._save_video_spatial(
                            volume, component_maps, component_overlays, stem, title, source_video
                        )
                        files.extend(rendered[0])
                        self._append_overview(batch_overviews, batch, label, rendered[1])

            elif relation == "video->video":
                query_full = _is_full_grid(record.q_shape, data.shape[1], record.metadata.get("query_indices"))
                key_full = _is_full_grid(record.k_shape, data.shape[2], record.metadata.get("key_indices"))
                if key_full:
                    positions, labels = _patch_axis(record, "query", data.shape[1])
                    for data_position, label in _limited_items(positions, labels, self.config.max_items):
                        volume = data[batch, data_position, :].reshape(
                            int(record.k_shape.frames or 1),
                            int(record.k_shape.height),
                            int(record.k_shape.width),
                        )
                        stem = f"query-patch-{_safe_component(label)}"
                        title = f"{prefix}\nquery patch: {label} → key video grid"
                        rendered = self._save_video_spatial(
                            volume, map_dir, overlay_dir, stem, title, source_video
                        )
                        files.extend(rendered[0])
                        self._append_overview(batch_overviews, batch, f"query {label}", rendered[1])
                elif query_full:
                    positions, labels = _patch_axis(record, "key", data.shape[2])
                    for data_position, label in _limited_items(positions, labels, self.config.max_items):
                        volume = data[batch, :, data_position].reshape(
                            int(record.q_shape.frames or 1),
                            int(record.q_shape.height),
                            int(record.q_shape.width),
                        )
                        stem = f"key-patch-{_safe_component(label)}"
                        title = f"{prefix}\nattention received by key patch: {label}"
                        rendered = self._save_video_spatial(
                            volume, map_dir, overlay_dir, stem, title, source_video
                        )
                        files.extend(rendered[0])
                        self._append_overview(batch_overviews, batch, f"key {label}", rendered[1])
                else:
                    warnings.append(_unsupported_warning(record, "neither video axis contains a complete T×H×W grid"))
            else:
                warnings.append(_unsupported_warning(record, f"relation {relation!r} has no renderer"))

        for batch, items in batch_overviews.items():
            if not items:
                continue
            overview = _make_overview(items, self.config.overview_columns)
            overview_path = output_dir / f"overview-batch-{batch:03d}.png"
            overview.save(overview_path)
            files.append(str(overview_path))

        return {"files": files, "warnings": warnings}

    def _append_overview(self, overviews, batch: int, label: str, image) -> None:
        items = overviews.setdefault(batch, [])
        if len(items) < self.config.overview_items:
            items.append((label, image))

    def _save_spatial(
        self,
        spatial,
        map_dir: Path,
        overlay_dir: Path,
        stem: str,
        title: str,
        source,
    ):
        heatmap, raw_heatmap = _render_spatial_map(
            spatial,
            title,
            colormap=self.config.colormap,
            long_side=self.config.spatial_long_side,
            show_labels=self.config.show_spatial_labels,
        )
        heatmap_path = map_dir / f"{stem}.png"
        heatmap.save(heatmap_path)
        files = [str(heatmap_path)]
        overview_image = heatmap

        if source is not None and self.config.save_overlays:
            overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay = _render_overlay(
                source,
                raw_heatmap,
                title,
                self.config.overlay_alpha,
                self.config.spatial_long_side,
                self.config.show_spatial_labels,
            )
            overlay_path = overlay_dir / f"{stem}.png"
            overlay.save(overlay_path)
            files.append(str(overlay_path))
            overview_image = overlay
        return files, overview_image

    def _save_video_spatial(
        self,
        volume,
        map_dir: Path,
        overlay_dir: Path,
        stem: str,
        title: str,
        source_frames: Sequence[Any],
    ):
        frame_root = map_dir / stem / "frames"
        frame_root.mkdir(parents=True, exist_ok=True)
        rendered_frames = []
        files = []
        minimum, maximum = volume.detach().float().amin(), volume.detach().float().amax()
        for frame_index, spatial in enumerate(volume):
            heatmap, _ = _render_spatial_map(
                spatial,
                f"{title}\nframe: {frame_index}",
                colormap=self.config.colormap,
                long_side=self.config.spatial_long_side,
                show_labels=self.config.show_spatial_labels,
                value_range=(minimum, maximum),
            )
            path = frame_root / f"frame-{frame_index:04d}.png"
            heatmap.save(path)
            files.append(str(path))
            rendered_frames.append(heatmap)
        animation_path = map_dir / f"{stem}.gif"
        _save_gif(rendered_frames, animation_path, self.config.video_fps)
        files.append(str(animation_path))
        overview = rendered_frames[len(rendered_frames) // 2]

        if source_frames and self.config.save_overlays:
            overlay_root = overlay_dir / stem / "frames"
            overlay_root.mkdir(parents=True, exist_ok=True)
            aligned = _resize_video_volume(volume, len(source_frames))
            overlay_frames = []
            for frame_index, (spatial, source) in enumerate(zip(aligned, source_frames)):
                _, raw = _render_spatial_map(
                    spatial,
                    title,
                    colormap=self.config.colormap,
                    long_side=self.config.spatial_long_side,
                    show_labels=False,
                    value_range=(minimum, maximum),
                )
                overlay = _render_overlay(
                    source,
                    raw,
                    f"{title}\nframe: {frame_index}",
                    self.config.overlay_alpha,
                    self.config.spatial_long_side,
                    self.config.show_spatial_labels,
                )
                path = overlay_root / f"frame-{frame_index:04d}.png"
                overlay.save(path)
                files.append(str(path))
                overlay_frames.append(overlay)
            overlay_animation = overlay_dir / f"{stem}.gif"
            _save_gif(overlay_frames, overlay_animation, self.config.video_fps)
            files.append(str(overlay_animation))
            overview = overlay_frames[len(overlay_frames) // 2]
        return files, overview


class _TokenResolver:
    def __init__(self, tokenizer, prompts, records: Sequence[AttentionRecord]):
        self.groups: List[Dict[str, Any]] = []
        self.component_groups: Dict[str, List[Dict[str, Any]]] = {}
        if tokenizer is None or prompts is None:
            return
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        max_length = max((_text_length(record) for record in records), default=0)
        if max_length <= 0:
            return

        if isinstance(tokenizer, Mapping):
            for component in _text_component_specs(records):
                name = str(component.get("name", "text"))
                component_tokenizer = tokenizer.get(name)
                if component_tokenizer is None:
                    continue
                component_length = _component_token_length(component)
                self.component_groups[name] = self._encode_groups(
                    component_tokenizer,
                    prompt_list,
                    max(1, component_length),
                )
            fallback = tokenizer.get("text") or next(iter(tokenizer.values()), None)
            if fallback is not None:
                self.groups = self._encode_groups(fallback, prompt_list, max_length)
            return

        self.groups = self._encode_groups(tokenizer, prompt_list, max_length)

    @staticmethod
    def _encode_groups(tokenizer, prompt_list, max_length):
        groups = []

        try:
            encoded = tokenizer(
                prompt_list,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
            )
            ids = _nested_lists(encoded["input_ids"])
            masks = _nested_lists(encoded.get("attention_mask"))
            for batch, token_ids in enumerate(ids):
                raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
                labels = [_display_token(token, index) for index, token in enumerate(raw_tokens)]
                mask = masks[batch] if batch < len(masks) else [1] * len(labels)
                active = {index for index, value in enumerate(mask) if int(value)}
                groups.append({"labels": labels, "active": active})
        except Exception:
            return []
        return groups

    def axis(self, record: AttentionRecord, axis: str, batch: int, length: int):
        selected = record.metadata.get(f"{axis}_indices")
        original_indices = list(range(length)) if selected is None else [int(item) for item in selected][:length]
        data_positions = list(range(len(original_indices)))
        if not self.groups:
            return data_positions, [f"{index}:token" for index in original_indices]

        group = self.groups[min(batch, len(self.groups) - 1)]
        labels = group["labels"]
        active = group["active"]
        filtered = [
            (position, labels[index] if 0 <= index < len(labels) else f"{index}:token")
            for position, index in zip(data_positions, original_indices)
            if index in active
        ]
        if not filtered:
            return data_positions, [
                labels[index] if 0 <= index < len(labels) else f"{index}:token" for index in original_indices
            ]
        return [item[0] for item in filtered], [item[1] for item in filtered]

    def axes(self, record: AttentionRecord, axis: str, batch: int, length: int):
        query_name, key_name = parse_relation(record.relation)
        modality = query_name if axis == "query" else key_name
        components = record.metadata.get("text_components")
        if modality != "text" or not components:
            positions, labels = self.axis(record, axis, batch, length)
            return [(None, positions, labels)]

        selected = record.metadata.get(f"{axis}_indices")
        original_indices = list(range(length)) if selected is None else [int(item) for item in selected][:length]
        position_by_original = {original: position for position, original in enumerate(original_indices)}
        axes = []
        for component in components:
            indices_by_batch = component.get("indices_by_batch") or []
            token_indices_by_batch = component.get("token_indices_by_batch") or []
            if not indices_by_batch:
                continue
            component_indices = indices_by_batch[min(batch, len(indices_by_batch) - 1)]
            token_indices = (
                token_indices_by_batch[min(batch, len(token_indices_by_batch) - 1)]
                if token_indices_by_batch
                else list(range(len(component_indices)))
            )
            name = str(component.get("name", "text"))
            group_list = self.component_groups.get(name, [])
            group = group_list[min(batch, len(group_list) - 1)] if group_list else None
            positions = []
            labels = []
            for original, token_index in zip(component_indices, token_indices):
                if int(original) not in position_by_original:
                    continue
                if group is not None and int(token_index) not in group["active"]:
                    continue
                positions.append(position_by_original[int(original)])
                if group is not None and 0 <= int(token_index) < len(group["labels"]):
                    token_label = group["labels"][int(token_index)]
                else:
                    token_label = f"{int(token_index)}:token"
                labels.append(f"{name}/{token_label}")
            if positions:
                axes.append((name, positions, labels))
        if axes:
            return axes
        positions, labels = self.axis(record, axis, batch, length)
        return [(None, positions, labels)]


def _text_component_specs(records: Sequence[AttentionRecord]):
    seen = set()
    result = []
    for record in records:
        for component in record.metadata.get("text_components") or []:
            name = str(component.get("name", "text"))
            if name in seen:
                continue
            seen.add(name)
            result.append(component)
    return result


def _component_token_length(component: Dict[str, Any]) -> int:
    values = [
        int(index)
        for batch_indices in component.get("token_indices_by_batch") or []
        for index in batch_indices
    ]
    if values:
        return max(values) + 1
    length = component.get("length")
    return int(length) if length is not None else 1


def _component_dirs(map_dir: Path, overlay_dir: Path, component: Optional[str]):
    if component is None:
        return map_dir, overlay_dir
    slug = _safe_component(component)
    component_map_dir = map_dir / slug
    component_map_dir.mkdir(parents=True, exist_ok=True)
    return component_map_dir, overlay_dir / slug


def _component_pair(query_component: Optional[str], key_component: Optional[str]):
    if query_component is None and key_component is None:
        return None
    return f"{query_component or 'text'}-to-{key_component or 'text'}"


def _record_title(
    record: AttentionRecord,
    relation: str,
    title_prefix: Optional[str],
) -> str:
    # A text matrix already communicates that it is the aggregate view through
    # its output path. Keep the rendered figure itself publication-clean.
    if relation == "text->text" and record.layer_id == "aggregate":
        return relation
    return title_prefix or f"{relation} | {record.layer_id}"


def _prepare_data(data, unconditional: Optional[bool], conditional_batch_count: Optional[int] = None):
    if data is None or data.ndim != 4:
        return None
    data = data.detach().float().mean(dim=1)
    select_conditional = unconditional is True
    if unconditional is None and conditional_batch_count is not None:
        select_conditional = data.shape[0] == conditional_batch_count * 2
    if select_conditional and data.shape[0] % 2 == 0:
        data = data.chunk(2, dim=0)[1]
    return data


def _select_guidance_records(
    records: Sequence[AttentionRecord], unconditional: Optional[bool]
) -> List[AttentionRecord]:
    """Keep conditional records when a pipeline evaluates CFG in separate calls.

    ``unconditional=False`` retains the legacy opt-out behavior and renders all
    calls. Raw tensors and metadata are unaffected because this selection is
    visualization-only.
    """
    records = list(records)
    if unconditional is False:
        return records
    roles = {record.metadata.get("guidance_role") for record in records}
    if "conditional" not in roles or not any(role != "conditional" for role in roles):
        return records
    return [record for record in records if record.metadata.get("guidance_role") == "conditional"]


def _aggregate_compatible(
    records: Sequence[AttentionRecord],
    unconditional: Optional[bool],
    conditional_batch_count: Optional[int] = None,
):
    """Average records without moving GPU tensors to CPU.

    Spatial relations are resized to their largest captured patch grid first,
    allowing UNet layers at different resolutions to contribute to one useful
    aggregate. Matrix-only relations fall back to exact-shape grouping.
    """
    prepared = []
    for record in records:
        data = _prepare_data(record.data, unconditional, conditional_batch_count)
        if data is not None:
            prepared.append((record, data))

    spatial = _aggregate_spatial(prepared)
    if spatial is not None:
        return spatial

    groups: Dict[Tuple[Any, ...], List[Tuple[AttentionRecord, Any]]] = {}
    for record, data in prepared:
        signature = (
            record.relation,
            tuple(data.shape),
            _grid_tuple(record.q_shape),
            _grid_tuple(record.k_shape),
            tuple(record.metadata.get("query_indices", ())),
            tuple(record.metadata.get("key_indices", ())),
        )
        groups.setdefault(signature, []).append((record, data))
    if not groups:
        return None
    compatible = max(groups.values(), key=len)
    total = compatible[0][1].clone()
    for _, data in compatible[1:]:
        total.add_(data.to(total.device))
    total.div_(len(compatible))
    aggregate_record = replace(
        compatible[0][0],
        layer_id="aggregate",
        step_index=None,
        timestep=None,
        data=None,
        q=None,
        k=None,
        attention_mask=None,
    )
    return aggregate_record, total, len(compatible)


def _aggregate_spatial(prepared: Sequence[Tuple[AttentionRecord, Any]]):
    if not prepared:
        return None
    relation = prepared[0][0].relation
    spatial_axis = None
    if relation == "image->text":
        spatial_axis = "query"
    elif relation == "video->text":
        spatial_axis = "query"
    elif relation == "text->image":
        spatial_axis = "key"
    elif relation == "text->video":
        spatial_axis = "key"
    elif relation in {"image->image", "video->video"}:
        # The normal per-query-patch view requires a full key grid. Fall back
        # to the inverse view only when capture selected key patches instead.
        if any(_record_has_full_grid(record, data, "key") for record, data in prepared):
            spatial_axis = "key"
        elif any(_record_has_full_grid(record, data, "query") for record, data in prepared):
            spatial_axis = "query"
    if spatial_axis is None:
        return None

    compatible_groups: Dict[Tuple[int, ...], List[Tuple[AttentionRecord, Any]]] = {}
    for record, data in prepared:
        if not _record_has_full_grid(record, data, spatial_axis):
            continue
        non_spatial_shape = (
            int(data.shape[0]),
            int(data.shape[2] if spatial_axis == "query" else data.shape[1]),
        )
        compatible_groups.setdefault(non_spatial_shape, []).append((record, data))
    if not compatible_groups:
        return None

    compatible = max(compatible_groups.values(), key=len)
    target_record, _ = max(
        compatible,
        key=lambda item: _grid_for_axis(item[0], spatial_axis).spatial_tokens or 0,
    )
    target_grid = _grid_for_axis(target_record, spatial_axis)
    resized = [
        _resize_spatial_axis(data, _grid_for_axis(record, spatial_axis), target_grid, spatial_axis)
        for record, data in compatible
    ]
    total = resized[0].clone()
    for data in resized[1:]:
        total.add_(data.to(total.device))
    total.div_(len(resized))

    metadata = dict(target_record.metadata)
    q_shape, k_shape = target_record.q_shape, target_record.k_shape
    q_range, k_range = target_record.q_range, target_record.k_range
    target_tokens = int(target_grid.spatial_tokens)
    if spatial_axis == "query":
        q_shape = target_grid
        q_range = (q_range[0], q_range[0] + target_tokens)
    else:
        k_shape = target_grid
        k_range = (k_range[0], k_range[0] + target_tokens)
        if relation in {"image->image", "video->video"} and metadata.get("query_indices") is not None:
            # Each source record selected the center of its own grid. Label the
            # aggregate using the corresponding center on the target grid.
            if len(metadata["query_indices"]) == 1:
                center = ((int(target_grid.frames or 1) // 2) * int(target_grid.height))
                center += int(target_grid.height) // 2
                center *= int(target_grid.width)
                center += int(target_grid.width) // 2
                metadata["query_indices"] = (center,)
                q_shape = target_grid

    aggregate_record = replace(
        target_record,
        layer_id="aggregate",
        step_index=None,
        timestep=None,
        q_range=q_range,
        k_range=k_range,
        q_shape=q_shape,
        k_shape=k_shape,
        data=None,
        q=None,
        k=None,
        attention_mask=None,
        metadata=metadata,
    )
    return aggregate_record, total, len(compatible)


def _record_has_full_grid(record: AttentionRecord, data, axis: str) -> bool:
    grid = _grid_for_axis(record, axis)
    length = int(data.shape[1] if axis == "query" else data.shape[2])
    return _is_full_grid(grid, length, record.metadata.get(f"{axis}_indices"))


def _grid_for_axis(record: AttentionRecord, axis: str):
    return record.q_shape if axis == "query" else record.k_shape


def _resize_spatial_axis(data, source_grid: TokenGrid, target_grid: TokenGrid, axis: str):
    if (
        source_grid.frames == target_grid.frames
        and source_grid.height == target_grid.height
        and source_grid.width == target_grid.width
    ):
        return data
    import torch.nn.functional as functional

    batch = int(data.shape[0])
    source_frames = int(source_grid.frames or 1)
    target_frames = int(target_grid.frames or 1)
    if axis == "query":
        other = int(data.shape[2])
        maps = data.transpose(1, 2).reshape(
            batch, other, source_frames, int(source_grid.height), int(source_grid.width)
        )
        maps = functional.interpolate(
            maps,
            size=(target_frames, int(target_grid.height), int(target_grid.width)),
            mode="trilinear",
            align_corners=False,
        )
        return maps.flatten(2).transpose(1, 2)
    other = int(data.shape[1])
    maps = data.reshape(
        batch, other, source_frames, int(source_grid.height), int(source_grid.width)
    )
    maps = functional.interpolate(
        maps,
        size=(target_frames, int(target_grid.height), int(target_grid.width)),
        mode="trilinear",
        align_corners=False,
    )
    return maps.flatten(2)


def _render_text_matrix(matrix, query_labels, key_labels, title: str, colormap: str):
    from PIL import Image, ImageDraw

    rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
    cell = max(8, min(24, 1000 // max(1, rows, cols)))
    map_image = _colorize(matrix, colormap).resize((max(1, cols * cell), max(1, rows * cell)), Image.Resampling.NEAREST)
    font = _font(13)
    small_font = _font(11)
    title_font = _font(16)
    left = min(320, max(120, max((_text_width(label, small_font) for label in query_labels), default=80) + 18))
    top = 56
    bottom = min(260, max(130, max((_text_width(label, small_font) for label in key_labels), default=80) + 35))
    canvas = Image.new("RGB", (left + map_image.width + 24, top + map_image.height + bottom), "white")
    canvas.paste(map_image, (left, top))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill="#111827", font=title_font)
    draw.rectangle((left, top, left + map_image.width, top + map_image.height), outline="#334155", width=1)

    for row, label in enumerate(query_labels):
        y = top + row * cell + cell / 2
        draw.text((left - 8, y), label, fill="#334155", font=small_font, anchor="rm")
    for col, label in enumerate(key_labels):
        label_width = max(12, _text_width(label, small_font) + 4)
        label_image = Image.new("RGBA", (label_width, 18), (255, 255, 255, 0))
        ImageDraw.Draw(label_image).text((0, 0), label, fill="#334155", font=small_font)
        rotated = label_image.rotate(90, expand=True)
        x = int(left + col * cell + cell / 2 - rotated.width / 2)
        canvas.paste(rotated, (x, top + map_image.height + 6), rotated)
    draw.text((left + map_image.width / 2, canvas.height - 12), "Key tokens (x)", fill="#111827", font=font, anchor="ms")
    draw.text((8, top + map_image.height / 2), "Query tokens (y)", fill="#111827", font=font, anchor="lm")
    return canvas


def _render_spatial_map(
    spatial,
    title: str,
    *,
    colormap: str,
    long_side: int,
    show_labels: bool,
    value_range=None,
):
    from PIL import Image, ImageDraw

    grid_h, grid_w = int(spatial.shape[-2]), int(spatial.shape[-1])
    if grid_h >= grid_w:
        height = long_side
        width = max(1, round(long_side * grid_w / grid_h))
    else:
        width = long_side
        height = max(1, round(long_side * grid_h / grid_w))
    raw = _colorize(spatial, colormap, value_range=value_range).resize(
        (width, height), Image.Resampling.NEAREST
    )
    if not show_labels:
        return raw, raw

    title_lines = title.splitlines()
    header = 20 + 19 * len(title_lines)
    canvas = Image.new("RGB", (width, height + header + 25), "white")
    canvas.paste(raw, (0, header))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(title_lines):
        draw.text((10, 8 + index * 19), line, fill="#111827", font=_font(14 if index else 16))
    draw.text((10, header + height + 7), f"patch grid: {grid_h}×{grid_w}", fill="#475569", font=_font(12))
    return canvas, raw


def _render_overlay(
    source,
    heatmap,
    title: str,
    alpha: float,
    long_side: int,
    show_labels: bool,
):
    from PIL import Image, ImageDraw

    source = source.convert("RGB")
    if max(source.size) != long_side:
        scale = long_side / max(source.size)
        source = source.resize(
            (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
            Image.Resampling.LANCZOS,
        )
    overlay = heatmap.resize(source.size, Image.Resampling.BILINEAR)
    blended = Image.blend(source, overlay, alpha)
    if not show_labels:
        return blended

    header = 58 if "\n" in title else 38
    canvas = Image.new("RGB", (source.width, source.height + header), "white")
    canvas.paste(blended, (0, header))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(title.splitlines()):
        draw.text((10, 7 + index * 19), line, fill="#111827", font=_font(14 if index else 16))
    return canvas


def _make_overview(items: Sequence[Tuple[str, Any]], columns: int):
    from PIL import Image, ImageDraw

    items = list(items)
    columns = min(columns, len(items))
    thumb_w, thumb_h = 280, 310
    rows = math.ceil(len(items) / columns)
    canvas = Image.new("RGB", (columns * thumb_w, rows * thumb_h), "#e2e8f0")
    draw = ImageDraw.Draw(canvas)
    for index, (label, item) in enumerate(items):
        row, col = divmod(index, columns)
        image = item.convert("RGB").copy()
        image.thumbnail((thumb_w - 12, thumb_h - 42))
        x = col * thumb_w + (thumb_w - image.width) // 2
        y = row * thumb_h + 31
        canvas.paste(image, (x, y))
        draw.text((col * thumb_w + 8, row * thumb_h + 7), label, fill="#0f172a", font=_font(13))
    return canvas


def _colorize(tensor, colormap: str, value_range=None):
    from PIL import Image

    torch = require_torch()
    values = torch.nan_to_num(tensor.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    minimum, maximum = value_range if value_range is not None else (values.amin(), values.amax())
    values = (values - minimum) / (maximum - minimum).clamp_min(1e-8)
    anchors = torch.tensor(_COLORMAPS[colormap], dtype=torch.float32, device=values.device)
    scaled = values * (anchors.shape[0] - 1)
    low = scaled.floor().long().clamp_max(anchors.shape[0] - 1)
    high = (low + 1).clamp_max(anchors.shape[0] - 1)
    fraction = (scaled - low).unsqueeze(-1)
    rgb = anchors[low] * (1.0 - fraction) + anchors[high] * fraction
    array = rgb.mul(255).round().byte().cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def _patch_axis(record: AttentionRecord, axis: str, length: int):
    selection = record.metadata.get(f"{axis}_indices")
    indices = list(range(length)) if selection is None else [int(value) for value in selection][:length]
    grid = record.q_shape if axis == "query" else record.k_shape
    labels = []
    for index in indices:
        if grid is not None and grid.width is not None:
            frame_size = int(grid.height or 1) * int(grid.width)
            frame, spatial_index = divmod(index, frame_size)
            row, col = divmod(spatial_index, int(grid.width))
            if (grid.frames or 1) > 1:
                labels.append(f"{index:05d}-t{frame:03d}-y{row:03d}-x{col:03d}")
            else:
                labels.append(f"{index:05d}-y{row:03d}-x{col:03d}")
        else:
            labels.append(f"{index:05d}")
    return list(range(len(indices))), labels


def _is_full_grid(grid: Optional[TokenGrid], data_length: int, selection) -> bool:
    return (
        grid is not None
        and grid.height is not None
        and grid.width is not None
        and selection is None
        and data_length == int(grid.spatial_tokens)
    )


def _select_matrix(matrix, query_positions: Sequence[int], key_positions: Sequence[int]):
    torch = require_torch()
    query_index = torch.tensor(query_positions, device=matrix.device, dtype=torch.long)
    key_index = torch.tensor(key_positions, device=matrix.device, dtype=torch.long)
    return matrix.index_select(0, query_index).index_select(1, key_index)


def _limit_axis(positions, labels, maximum):
    if maximum is None or len(positions) <= maximum:
        return positions, labels
    return positions[:maximum], labels[:maximum]


def _limited_items(positions, labels, maximum):
    items = list(zip(positions, labels))
    return items if maximum is None else items[:maximum]


def _text_length(record: AttentionRecord) -> int:
    query_name, key_name = parse_relation(record.relation)
    lengths = []
    if query_name == "text":
        lengths.append(record.q_range[1] - record.q_range[0])
    if key_name == "text":
        lengths.append(record.k_range[1] - record.k_range[0])
    return max(lengths, default=0)


def _nested_lists(value) -> List[List[int]]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not value:
        return []
    return value if isinstance(value[0], list) else [value]


def _display_token(token: Any, index: int) -> str:
    value = str(token)
    value = value.replace("</w>", "").replace("Ġ", "▁")
    value = value.replace("\n", "\\n").replace("\t", "\\t")
    value = value if value else "∅"
    return f"{index}:{value}"


def _as_pil_images(images) -> List[Any]:
    if images is None:
        return []
    if hasattr(images, "convert"):
        return [images]
    if hasattr(images, "images"):
        images = images.images
    try:
        result = []
        for image in images:
            result.append(_to_pil_frame(image))
        return result
    except Exception:
        return []


def _as_pil_videos(videos) -> List[List[Any]]:
    if videos is None:
        return []
    if hasattr(videos, "frames"):
        videos = videos.frames
    try:
        import numpy as np

        if hasattr(videos, "detach"):
            videos = videos.detach().cpu().numpy()
        if isinstance(videos, np.ndarray):
            if videos.ndim == 4:
                videos = [videos]
            elif videos.ndim != 5:
                return []
        else:
            videos = list(videos)
            if not videos:
                return []
            first = videos[0]
            first_array = (
                first.detach().cpu().numpy() if hasattr(first, "detach") else np.asarray(first)
            )
            if hasattr(first, "convert") or first_array.ndim == 3:
                videos = [videos]

        result = []
        for video in videos:
            frames = list(video) if not hasattr(video, "detach") else list(video.detach().cpu())
            converted = [_to_pil_frame(frame) for frame in frames]
            if converted:
                result.append(converted)
        return result
    except Exception:
        return []


def _to_pil_frame(frame):
    if hasattr(frame, "convert"):
        return frame.convert("RGB")
    import numpy as np
    from PIL import Image

    array = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
    if array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        if array.max(initial=0.0) <= 1.0:
            array = array * 255.0
    array = np.clip(array, 0, 255).round().astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array).convert("RGB")


def _resize_video_volume(volume, frames: int):
    if int(volume.shape[0]) == frames:
        return volume
    import torch.nn.functional as functional

    resized = functional.interpolate(
        volume.detach().float()[None, None],
        size=(frames, int(volume.shape[-2]), int(volume.shape[-1])),
        mode="trilinear",
        align_corners=False,
    )
    return resized[0, 0].to(volume.dtype)


def _save_gif(frames: Sequence[Any], path: Path, fps: int) -> None:
    if not frames:
        return
    duration = max(1, round(1000 / fps))
    frames[0].save(
        path,
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration,
        loop=0,
        optimize=False,
    )


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _text_width(text: str, font) -> int:
    try:
        return int(font.getlength(text))
    except AttributeError:
        return int(font.getbbox(text)[2])


def _step_slug(record: AttentionRecord) -> str:
    if record.step_index is not None:
        return f"step-{int(record.step_index):04d}"
    if record.timestep is not None:
        value = str(record.timestep).replace(".", "_").replace("-", "m")
        return f"timestep-{value}"
    return "step-unknown"


def _relation_slug(relation: str) -> str:
    return relation.replace("->", "-to-")


def _safe_component(value: Any, max_length: int = 140) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("._") or "item"
    return value[:max_length]


def _grid_tuple(grid: Optional[TokenGrid]):
    return None if grid is None else (
        grid.frames,
        grid.height,
        grid.width,
        grid.patch_size,
        grid.temporal_patch_size,
    )


def _unsupported_warning(record: AttentionRecord, reason: str) -> str:
    return f"Skipped {record.relation} at {record.layer_id}: {reason}."


_COLORMAPS = {
    # Compact perceptual approximations; interpolation happens on the tensor's
    # current device, so CUDA records are colorized without an early CPU copy.
    "magma": (
        (0.001, 0.000, 0.014),
        (0.126, 0.016, 0.302),
        (0.445, 0.122, 0.507),
        (0.741, 0.220, 0.365),
        (0.944, 0.378, 0.258),
        (0.987, 0.733, 0.450),
        (0.987, 0.991, 0.750),
    ),
    "viridis": (
        (0.267, 0.005, 0.329),
        (0.283, 0.141, 0.458),
        (0.254, 0.265, 0.530),
        (0.207, 0.372, 0.553),
        (0.164, 0.471, 0.558),
        (0.128, 0.567, 0.551),
        (0.135, 0.659, 0.518),
        (0.267, 0.749, 0.441),
        (0.478, 0.821, 0.318),
        (0.741, 0.873, 0.150),
        (0.993, 0.906, 0.144),
    ),
}


__all__ = ["AttentionVisualizer", "VisualizationConfig"]
