from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

from .records import TokenGrid, infer_2d_grid


@dataclass
class CaptureRuntimeContext:
    timestep: Optional[float] = None
    step_index: Optional[int] = None
    image_grid: Optional[TokenGrid] = None
    text_tokens: Optional[int] = None
    image_tokens: Optional[int] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


_context_stack: ContextVar[tuple[CaptureRuntimeContext, ...]] = ContextVar(
    "attention_map_capture_context_stack", default=()
)


def current_context() -> CaptureRuntimeContext:
    stack = _context_stack.get()
    if not stack:
        return CaptureRuntimeContext()
    merged = CaptureRuntimeContext()
    for item in stack:
        if item.timestep is not None:
            merged.timestep = item.timestep
        if item.step_index is not None:
            merged.step_index = item.step_index
        if item.image_grid is not None:
            merged.image_grid = item.image_grid
        if item.text_tokens is not None:
            merged.text_tokens = item.text_tokens
        if item.image_tokens is not None:
            merged.image_tokens = item.image_tokens
        merged.metadata.update(item.metadata)
    return merged


@contextmanager
def pushed_context(context: CaptureRuntimeContext) -> Iterator[None]:
    token = push_context(context)
    try:
        yield
    finally:
        pop_context(token)


def push_context(context: CaptureRuntimeContext):
    stack = _context_stack.get()
    return _context_stack.set(stack + (context,))


def pop_context(token) -> None:
    _context_stack.reset(token)


def infer_grid_from_hidden_states(hidden_states, patch_size: Optional[int] = None) -> Optional[TokenGrid]:
    if hidden_states is None or not hasattr(hidden_states, "shape"):
        return None
    shape = tuple(hidden_states.shape)
    if len(shape) == 4:
        height, width = int(shape[-2]), int(shape[-1])
        if patch_size:
            height //= patch_size
            width //= patch_size
        return TokenGrid(height=height, width=width, patch_size=patch_size)
    if len(shape) == 3:
        return infer_2d_grid(int(shape[1]))
    return None


def infer_grid_from_img_ids(img_ids) -> Optional[TokenGrid]:
    if img_ids is None or not hasattr(img_ids, "shape"):
        return None
    try:
        ids = img_ids[0] if img_ids.ndim == 3 else img_ids
        if ids.shape[-1] < 3:
            return infer_2d_grid(int(ids.shape[0]))
        height = int(ids[:, 1].max().item()) + 1
        width = int(ids[:, 2].max().item()) + 1
        return TokenGrid(height=height, width=width)
    except Exception:
        return infer_2d_grid(int(img_ids.shape[-2]))
