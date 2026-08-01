from __future__ import annotations


class BaseCaptureProcessor:
    def __init__(self, collector, layer_id: str, original_processor=None):
        self.collector = collector
        self.layer_id = layer_id
        self.original_processor = original_processor

    def _capture(self, **kwargs) -> None:
        self.collector.capture_qk(layer_id=self.layer_id, **kwargs)


__all__ = ["BaseCaptureProcessor"]
