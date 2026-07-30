import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from attention_map_diffusers.capture.collector import AttentionCollector
from attention_map_diffusers.capture.config import CaptureConfig, normalize_indices
from attention_map_diffusers.capture.compute import compute_attention_probs, reduce_heads
from attention_map_diffusers.capture.records import AttentionRecord, ModalityRange, TokenGrid, make_layout
from attention_map_diffusers.visualization import AttentionVisualizer, VisualizationConfig


class FakeTokenizer:
    def __call__(self, prompts, *, padding, truncation, max_length, return_attention_mask):
        ids = []
        masks = []
        for _ in prompts:
            active = min(4, max_length)
            ids.append(list(range(active)) + [99] * (max_length - active))
            masks.append([1] * active + [0] * (max_length - active))
        return {"input_ids": ids, "attention_mask": masks}

    def convert_ids_to_tokens(self, token_ids):
        names = ["<s>", "▁red", "▁cat", "</s>"]
        return [names[index] if index < len(names) else "<pad>" for index in token_ids]


def make_record(relation, data, *, q_grid=None, k_grid=None, metadata=None):
    q_length, k_length = data.shape[-2:]
    original_q = q_grid.spatial_tokens if q_grid and metadata and metadata.get("query_indices") else q_length
    original_k = k_grid.spatial_tokens if k_grid and metadata and metadata.get("key_indices") else k_length
    return AttentionRecord(
        layer_id="transformer.blocks.0.attn",
        relation=relation,
        attention_type="joint",
        q_range=(0, original_q),
        k_range=(0, original_k),
        q_shape=q_grid,
        k_shape=k_grid,
        heads=data.shape[1],
        dtype=str(data.dtype),
        device=str(data.device),
        timestep=1.0,
        data=data,
        metadata=metadata or {},
    )


class VisualizationTests(unittest.TestCase):
    def test_full_patch_aggregate_is_the_safe_default(self):
        config = VisualizationConfig()
        self.assertIsNone(config.max_items)
        self.assertFalse(config.save_layer_maps)
        self.assertEqual(CaptureConfig().max_map_elements, 128_000_000)

    def test_last_timestep_mode_discards_earlier_records(self):
        query = torch.randn(1, 1, 4, 8)
        grid = TokenGrid(height=2, width=2)
        image = ModalityRange("image", 0, 4, grid)
        collector = AttentionCollector(
            CaptureConfig(relations=["image->image"], timesteps="last", mode="exact", offload=None)
        )
        for timestep in (10, 5):
            collector.set_step(timestep=timestep)
            collector.capture_qk(
                layer_id=f"step-{timestep}",
                attention_type="self",
                query=query,
                key=query,
                layout=make_layout([image], [image]),
            )
        self.assertEqual(len(collector.records), 1)
        self.assertEqual(collector.records[0].timestep, 5.0)

    def test_overlay_respects_spatial_long_side(self):
        grid = TokenGrid(height=2, width=2)
        record = make_record("image->image", torch.rand(1, 1, 4, 4), q_grid=grid, k_grid=grid)
        config = VisualizationConfig(save_layer_maps=False, spatial_long_side=48, max_items=1)
        with tempfile.TemporaryDirectory() as directory:
            AttentionVisualizer(config).save(
                [record], directory, images=[Image.new("RGB", (160, 80), "white")]
            )
            overlay = next(
                (Path(directory) / "image-to-image/aggregate/batch-000/overlays").glob("*.png")
            )
            with Image.open(overlay) as image:
                self.assertEqual(image.width, 48)

    def test_all_four_relations_are_rendered(self):
        grid = TokenGrid(height=2, width=3)
        records = [
            make_record("text->text", torch.rand(1, 2, 6, 6)),
            make_record("image->text", torch.rand(1, 2, 6, 6), q_grid=grid),
            make_record("text->image", torch.rand(1, 2, 6, 6), k_grid=grid),
            make_record(
                "image->image",
                torch.rand(1, 2, 1, 6),
                q_grid=grid,
                k_grid=grid,
                metadata={"query_indices": (3,)},
            ),
        ]
        config = VisualizationConfig(spatial_long_side=96, max_items=8, overview_columns=2)

        with tempfile.TemporaryDirectory() as directory:
            manifest = AttentionVisualizer(config).save(
                records,
                directory,
                tokenizer=FakeTokenizer(),
                prompts=["red cat"],
                images=[Image.new("RGB", (128, 96), "white")],
            )
            root = Path(directory)
            self.assertEqual(set(manifest["relations"]), {
                "text->text", "image->text", "text->image", "image->image"
            })
            self.assertTrue((root / "text-to-text/aggregate/batch-000/maps/attention-matrix.png").is_file())
            self.assertTrue(list((root / "image-to-text/aggregate/batch-000/maps").glob("token-*.png")))
            self.assertTrue(list((root / "text-to-image/aggregate/batch-000/maps").glob("token-*.png")))
            self.assertTrue(list((root / "image-to-image/aggregate/batch-000/maps").glob("query-patch-*.png")))
            self.assertTrue(list((root / "image-to-image/aggregate/batch-000/overlays").glob("query-patch-*.png")))
            self.assertTrue((root / "visualization.json").is_file())
            parsed = json.loads((root / "visualization.json").read_text())
            self.assertFalse(parsed["warnings"])

    def test_text_matrix_contains_axis_margins(self):
        record = make_record("text->text", torch.rand(1, 1, 6, 6))
        config = VisualizationConfig(save_layer_maps=False, spatial_long_side=64)
        with tempfile.TemporaryDirectory() as directory:
            AttentionVisualizer(config).save(
                [record], directory, tokenizer=FakeTokenizer(), prompts=["red cat"]
            )
            with Image.open(Path(directory) / "text-to-text/aggregate/batch-000/maps/attention-matrix.png") as image:
                # Four active tokens at <=24px cells would be <=96px without
                # the explicit token-labelled x/y margins.
                self.assertGreater(image.width, 180)
                self.assertGreater(image.height, 180)

    def test_cfg_batch_is_reduced_to_conditional_half_automatically(self):
        record = make_record("text->text", torch.rand(2, 1, 6, 6))
        config = VisualizationConfig(save_layer_maps=False, spatial_long_side=64)
        with tempfile.TemporaryDirectory() as directory:
            AttentionVisualizer(config).save(
                [record],
                directory,
                tokenizer=FakeTokenizer(),
                prompts=["red cat"],
                images=[Image.new("RGB", (64, 64), "white")],
            )
            aggregate = Path(directory) / "text-to-text/aggregate"
            self.assertTrue((aggregate / "batch-000/maps/attention-matrix.png").is_file())
            self.assertFalse((aggregate / "batch-001").exists())

    def test_image_to_image_writes_one_map_and_overlay_per_query_patch(self):
        grid = TokenGrid(height=3, width=4)
        record = make_record("image->image", torch.rand(1, 1, 12, 12), q_grid=grid, k_grid=grid)
        config = VisualizationConfig(
            save_layer_maps=False,
            spatial_long_side=64,
            max_items=None,
            overview_items=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = AttentionVisualizer(config).save(
                [record], directory, images=[Image.new("RGB", (64, 48), "white")]
            )
            batch = Path(directory) / "image-to-image/aggregate/batch-000"
            maps = sorted((batch / "maps").glob("query-patch-*.png"))
            overlays = sorted((batch / "overlays").glob("query-patch-*.png"))
            self.assertEqual(len(maps), 12)
            self.assertEqual(len(overlays), 12)
            self.assertEqual(len(manifest["relations"]["image->image"]["aggregate"]["files"]), 25)
            self.assertTrue((Path(directory) / "image-to-image/aggregate/overview-batch-000.png").is_file())

    def test_raw_tensors_are_opt_in(self):
        record = make_record("text->text", torch.rand(1, 1, 4, 4))
        collector = AttentionCollector(CaptureConfig(relations=["text->text"]))
        collector.records.append(record)
        with tempfile.TemporaryDirectory() as directory:
            collector.compute().save(
                directory,
                visualization_config=VisualizationConfig(save_layer_maps=False),
            )
            self.assertFalse((Path(directory) / "raw").exists())

    def test_raw_tensors_can_be_saved_without_visualizations(self):
        record = make_record("text->text", torch.rand(1, 1, 4, 4))
        collector = AttentionCollector(CaptureConfig(relations=["text->text"]))
        collector.records.append(record)
        with tempfile.TemporaryDirectory() as directory:
            result = collector.compute().save(
                directory,
                save_raw=True,
                save_visualizations=False,
            )
            self.assertEqual(len(list((Path(directory) / "raw").rglob("*.pt"))), 1)
            self.assertTrue((Path(directory) / "metadata.json").is_file())
            self.assertFalse((Path(directory) / "visuals").exists())
            self.assertEqual(result, {"relations": {}, "warnings": []})

    def test_multiresolution_spatial_records_share_one_aggregate(self):
        records = [
            make_record("image->text", torch.rand(1, 1, 4, 6), q_grid=TokenGrid(height=2, width=2)),
            make_record("image->text", torch.rand(1, 1, 16, 6), q_grid=TokenGrid(height=4, width=4)),
        ]
        config = VisualizationConfig(save_layer_maps=False, spatial_long_side=64)
        with tempfile.TemporaryDirectory() as directory:
            manifest = AttentionVisualizer(config).save(
                records, directory, tokenizer=FakeTokenizer(), prompts=["red cat"]
            )
            aggregate = manifest["relations"]["image->text"]["aggregate"]
            self.assertEqual(aggregate["source_records"], 2)
            self.assertFalse(manifest["warnings"])


class CaptureComputeTests(unittest.TestCase):
    def test_head_mean_is_reduced_during_chunked_compute(self):
        query = torch.randn(2, 4, 7, 8)
        key = torch.randn(2, 4, 5, 8)
        actual = compute_attention_probs(query, key, chunk_size=3, heads="mean")
        expected = torch.softmax(query.float() @ key.float().transpose(-2, -1) / (8**0.5), dim=-1).mean(
            dim=1, keepdim=True
        )
        self.assertEqual(tuple(actual.shape), (2, 1, 7, 5))
        torch.testing.assert_close(actual, expected)

    def test_dynamic_patch_selection(self):
        self.assertEqual(normalize_indices("center", 9), (4,))
        self.assertEqual(normalize_indices("sample:3", 9), (0, 4, 8))

        query = torch.randn(1, 1, 16, 4)
        grid = TokenGrid(height=4, width=4)
        image = ModalityRange("image", 0, 16, grid)
        collector = AttentionCollector(
            CaptureConfig(relations=["image->image"], mode="exact", relation_query_indices={"image->image": "center"})
        )
        collector.capture_qk(
            layer_id="patches",
            attention_type="self",
            query=query,
            key=query,
            layout=make_layout([image], [image]),
        )
        self.assertEqual(collector.records[0].metadata["query_indices"], (10,))

    def test_map_limit_counts_the_heads_retained_after_reduction(self):
        query = torch.randn(1, 4, 4, 8)
        grid = TokenGrid(height=2, width=2)
        image = ModalityRange("image", 0, 4, grid)
        collector = AttentionCollector(
            CaptureConfig(
                relations=["image->image"],
                mode="exact",
                heads="mean",
                max_map_elements=16,
                offload=None,
            )
        )
        collector.capture_qk(
            layer_id="full-patches",
            attention_type="self",
            query=query,
            key=query,
            layout=make_layout([image], [image]),
        )
        self.assertEqual(tuple(collector.records[0].data.shape), (1, 1, 4, 4))

    def test_joint_relations_keep_full_softmax_normalization(self):
        query = torch.randn(1, 2, 4, 8)
        key = torch.randn(1, 2, 4, 8)
        image = ModalityRange("image", 0, 2, TokenGrid(height=1, width=2))
        text = ModalityRange("text", 2, 4)
        collector = AttentionCollector(CaptureConfig(relations="all", mode="exact", offload=None, heads="mean"))
        collector.capture_qk(
            layer_id="joint",
            attention_type="joint",
            query=query,
            key=key,
            layout=make_layout([image, text], [image, text]),
        )
        records = {record.relation: record.data for record in collector.records}
        image_row_sum = records["image->image"].sum(-1) + records["image->text"].sum(-1)
        text_row_sum = records["text->image"].sum(-1) + records["text->text"].sum(-1)
        torch.testing.assert_close(image_row_sum, torch.ones_like(image_row_sum))
        torch.testing.assert_close(text_row_sum, torch.ones_like(text_row_sum))
        self.assertTrue(torch.all(records["image->text"].sum(-1) < 1.0))
        validation = collector.compute().validate()
        self.assertEqual(validation["status"], "pass")
        self.assertEqual(validation["joint_normalization"]["groups"], 1)

    def test_selected_keys_are_valid_partial_probability_mass(self):
        query = torch.randn(1, 2, 4, 8)
        key = torch.randn(1, 2, 6, 8)
        image = ModalityRange("image", 0, 4, TokenGrid(height=2, width=2))
        text = ModalityRange("text", 0, 6)
        collector = AttentionCollector(
            CaptureConfig(
                relations=["image->text"],
                mode="exact",
                offload=None,
                heads="mean",
                key_indices="sample:2",
            )
        )
        collector.capture_qk(
            layer_id="partial-keys",
            attention_type="cross",
            query=query,
            key=key,
            layout=make_layout([image], [text]),
        )
        validation = collector.compute().validate()
        self.assertEqual(validation["status"], "pass")
        self.assertTrue(torch.all(collector.records[0].data.sum(-1) <= 1.0 + 1e-6))

    def test_joint_validation_aligns_different_query_selections(self):
        query = torch.randn(1, 2, 8, 8)
        key = torch.randn(1, 2, 8, 8)
        image = ModalityRange("image", 0, 4, TokenGrid(height=2, width=2))
        text = ModalityRange("text", 4, 8)
        collector = AttentionCollector(
            CaptureConfig(
                relations="all",
                mode="exact",
                offload=None,
                heads="mean",
                relation_query_indices={"image->image": "center"},
            )
        )
        collector.capture_qk(
            layer_id="joint-selected-query",
            attention_type="joint",
            query=query,
            key=key,
            layout=make_layout([image, text], [image, text]),
        )
        validation = collector.compute().validate()
        self.assertEqual(validation["status"], "pass")
        self.assertEqual(validation["joint_normalization"]["groups"], 1)

    def test_explicit_head_selection(self):
        tensor = torch.rand(1, 4, 3, 2)
        selected = reduce_heads(tensor, [0, 2])
        torch.testing.assert_close(selected, tensor[:, [0, 2]])


if __name__ == "__main__":
    unittest.main()
