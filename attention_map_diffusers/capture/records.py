from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple


@dataclass(frozen=True)
class TokenGrid:
    frames: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    patch_size: Optional[int] = None

    @property
    def spatial_tokens(self) -> Optional[int]:
        if self.height is None or self.width is None:
            return None
        frames = self.frames or 1
        return frames * self.height * self.width

    def as_dict(self) -> Dict[str, Optional[int]]:
        return {
            "frames": self.frames,
            "height": self.height,
            "width": self.width,
            "patch_size": self.patch_size,
        }


@dataclass(frozen=True)
class ModalityRange:
    name: str
    start: int
    end: int
    grid: Optional[TokenGrid] = None

    @property
    def length(self) -> int:
        return self.end - self.start

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "grid": self.grid.as_dict() if self.grid is not None else None,
        }


@dataclass(frozen=True)
class ModalityLayout:
    ranges: Mapping[str, ModalityRange] = field(default_factory=dict)

    def __contains__(self, name: str) -> bool:
        return name in self.ranges

    def __getitem__(self, name: str) -> ModalityRange:
        return self.ranges[name]

    def get(self, name: str) -> Optional[ModalityRange]:
        return self.ranges.get(name)

    def names(self) -> Tuple[str, ...]:
        return tuple(self.ranges.keys())

    def as_dict(self) -> Dict[str, Any]:
        return {name: value.as_dict() for name, value in self.ranges.items()}


@dataclass(frozen=True)
class AttentionLayout:
    query: ModalityLayout
    key: ModalityLayout

    def has_relation(self, relation: str) -> bool:
        query_name, key_name = parse_relation(relation)
        return query_name in self.query and key_name in self.key

    def relation_ranges(self, relation: str) -> Tuple[ModalityRange, ModalityRange]:
        query_name, key_name = parse_relation(relation)
        return self.query[query_name], self.key[key_name]

    def as_dict(self) -> Dict[str, Any]:
        return {"query": self.query.as_dict(), "key": self.key.as_dict()}


@dataclass
class AttentionRecord:
    layer_id: str
    relation: str
    attention_type: str
    q_range: Tuple[int, int]
    k_range: Tuple[int, int]
    q_shape: Optional[TokenGrid]
    k_shape: Optional[TokenGrid]
    heads: int
    dtype: str
    device: str
    step_index: Optional[int] = None
    timestep: Optional[float] = None
    data: Any = None
    q: Any = None
    k: Any = None
    attention_mask: Any = None
    scale: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def without_qk(self) -> "AttentionRecord":
        return replace(self, q=None, k=None, attention_mask=None)

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "relation": self.relation,
            "attention_type": self.attention_type,
            "q_range": self.q_range,
            "k_range": self.k_range,
            "q_shape": self.q_shape.as_dict() if self.q_shape is not None else None,
            "k_shape": self.k_shape.as_dict() if self.k_shape is not None else None,
            "heads": self.heads,
            "dtype": self.dtype,
            "device": self.device,
            "step_index": self.step_index,
            "timestep": self.timestep,
            "has_data": self.data is not None,
            "has_qk": self.q is not None and self.k is not None,
            "metadata": dict(self.metadata),
        }


def parse_relation(relation: str) -> Tuple[str, str]:
    if "->" not in relation:
        raise ValueError(f"Attention relation must look like 'query->key', got {relation!r}.")
    query_name, key_name = [part.strip() for part in relation.split("->", 1)]
    if not query_name or not key_name:
        raise ValueError(f"Attention relation must look like 'query->key', got {relation!r}.")
    return query_name, key_name


def make_layout(
    query_ranges: Iterable[ModalityRange],
    key_ranges: Iterable[ModalityRange],
) -> AttentionLayout:
    return AttentionLayout(
        query=ModalityLayout({item.name: item for item in query_ranges}),
        key=ModalityLayout({item.name: item for item in key_ranges}),
    )


def infer_2d_grid(token_count: int) -> Optional[TokenGrid]:
    root = int(token_count**0.5)
    if root * root == token_count:
        return TokenGrid(height=root, width=root)
    return None


def iter_supported_relations(layout: AttentionLayout, relations: Iterable[str]) -> Iterator[str]:
    for relation in relations:
        if layout.has_relation(relation):
            yield relation
