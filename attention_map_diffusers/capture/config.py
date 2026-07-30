from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union


ALL_RELATIONS: Tuple[str, ...] = (
    "image->text",
    "text->image",
    "image->image",
    "text->text",
    "video->text",
    "text->video",
    "video->video",
)

DEFAULT_RELATIONS: Tuple[str, ...] = ("image->text",)

IndexSelection = Union[str, int, Sequence[int], None]


@dataclass(frozen=True)
class CaptureConfig:
    relations: Tuple[str, ...] = DEFAULT_RELATIONS
    layers: Union[str, Sequence[str]] = "all"
    timesteps: Union[str, Sequence[int]] = "all"
    heads: Union[str, Sequence[int]] = "mean"
    mode: str = "fast"
    offload: Optional[str] = "cpu"
    dtype: Optional[str] = None
    query_indices: IndexSelection = None
    key_indices: IndexSelection = None
    relation_query_indices: Optional[Mapping[str, IndexSelection]] = None
    relation_key_indices: Optional[Mapping[str, IndexSelection]] = None
    tokens: IndexSelection = None
    chunk_size: int = 1024
    max_map_elements: int = 128_000_000
    allow_large_maps: bool = False
    capture_unconditional: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "relations", normalize_relations(self.relations))
        if isinstance(self.timesteps, str) and self.timesteps not in {"all", "last"}:
            raise ValueError("timesteps must be 'all', 'last', or a sequence of timestep integers.")
        if self.mode not in {"fast", "exact", "stream"}:
            raise ValueError("mode must be one of: 'fast', 'exact', 'stream'.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.max_map_elements <= 0:
            raise ValueError("max_map_elements must be positive.")


def normalize_relations(relations: Union[str, Iterable[str], None]) -> Tuple[str, ...]:
    if relations is None:
        return DEFAULT_RELATIONS
    if isinstance(relations, str):
        if relations == "all":
            return ALL_RELATIONS
        relations = [item.strip() for item in relations.split(",") if item.strip()]
    normalized = []
    for relation in relations:
        relation = relation.strip()
        if relation == "all":
            normalized.extend(ALL_RELATIONS)
        elif relation and relation not in normalized:
            normalized.append(relation)
    return tuple(normalized or DEFAULT_RELATIONS)


def normalize_indices(selection: IndexSelection, length: int, grid=None) -> Optional[Tuple[int, ...]]:
    if selection is None or selection == "all":
        return None
    if isinstance(selection, int):
        selection = [selection]
    if isinstance(selection, str):
        if not selection:
            return None
        if selection == "center":
            if (
                grid is not None
                and getattr(grid, "height", None) is not None
                and getattr(grid, "width", None) is not None
                and getattr(grid, "spatial_tokens", None) == length
            ):
                frame = (getattr(grid, "frames", None) or 1) // 2
                index = frame * int(grid.height) * int(grid.width)
                index += (int(grid.height) // 2) * int(grid.width) + int(grid.width) // 2
                return (index,)
            return (length // 2,)
        if selection.startswith("sample:"):
            count = max(1, int(selection.split(":", 1)[1]))
            if count == 1:
                return (length // 2,)
            step = (length - 1) / (count - 1)
            return tuple(dict.fromkeys(round(index * step) for index in range(count)))
        selection = [int(item.strip()) for item in selection.split(",") if item.strip()]
    result = []
    for index in selection:
        index = int(index)
        if index < 0:
            index += length
        if 0 <= index < length:
            result.append(index)
    return tuple(dict.fromkeys(result))
