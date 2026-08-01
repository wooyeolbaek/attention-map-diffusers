import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from attention_map_diffusers.capture.collector import AttentionCollector
from attention_map_diffusers.capture.config import ALL_RELATIONS, CaptureConfig, normalize_indices
from attention_map_diffusers.capture.compute import compute_attention_probs, reduce_heads
from attention_map_diffusers.capture.records import AttentionRecord, ModalityRange, TokenGrid, make_layout
from attention_map_diffusers.visualization import AttentionVisualizer, VisualizationConfig, _record_title


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
    def test_text_matrix_aggregate_title_is_clean(self):
        record = make_record("text->text", torch.rand(1, 1, 4, 4))
        record.layer_id = "aggregate"
        self.assertEqual(_record_title(record, "text->text", "aggregate (25 maps)"), "text->text")

    def test_full_patch_aggregate_is_the_safe_default(self):
        config = VisualizationConfig()
        self.assertIsNone(config.max_items)
        self.assertFalse(config.save_layer_maps)
        self.assertFalse(config.show_spatial_labels)
        self.assertEqual(CaptureConfig().max_map_elements, 128_000_000)
        self.assertEqual(CaptureConfig().timesteps, "last")

    def test_only_implemented_relations_and_modes_are_accepted(self):
        self.assertEqual(
            ALL_RELATIONS,
            (
                "image->text",
                "text->image",
                "image->image",
                "text->text",
                "video->text",
                "text->video",
                "video->video",
            ),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported attention relation"):
            CaptureConfig(relations=["audio->text"])
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            CaptureConfig(mode="stream")

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
        self.assertEqual(collector.records[0].step_index, 1)

    def test_fast_capture_can_recompute_on_a_transient_device(self):
        query = torch.randn(1, 2, 4, 8)
        grid = TokenGrid(height=2, width=2)
        image = ModalityRange("image", 0, 4, grid)
        collector = AttentionCollector(
            CaptureConfig(
                relations=["image->image"],
                mode="fast",
                offload="cpu",
                heads="mean",
            )
        )
        collector.capture_qk(
            layer_id="test",
            attention_type="self",
            query=query,
            key=query,
            layout=make_layout([image], [image]),
        )
        maps = collector.compute(compute_device="cpu")
        self.assertEqual(maps.records[0].data.device.type, "cpu")
        self.assertEqual(maps.validate()["status"], "pass")

    def test_explicit_timestep_sequence_captures_only_selected_steps(self):
        query = torch.randn(1, 1, 4, 8)
        grid = TokenGrid(height=2, width=2)
        image = ModalityRange("image", 0, 4, grid)
        collector = AttentionCollector(
            CaptureConfig(relations=["image->image"], timesteps=[10, 5], mode="exact")
        )
        for timestep in (10, 7, 5):
            collector.set_step(timestep=timestep)
            collector.capture_qk(
                layer_id=f"step-{timestep}",
                attention_type="self",
                query=query,
                key=query,
                layout=make_layout([image], [image]),
            )
        self.assertEqual([record.timestep for record in collector.records], [10.0, 5.0])

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
                self.assertEqual(image.size, (48, 24))

    def test_spatial_labels_are_opt_in(self):
        grid = TokenGrid(height=2, width=2)
        record = make_record("image->image", torch.rand(1, 1, 4, 4), q_grid=grid, k_grid=grid)
        source = Image.new("RGB", (80, 40), "white")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            AttentionVisualizer(VisualizationConfig(spatial_long_side=48, max_items=1)).save(
                [record], root / "plain", images=[source]
            )
            AttentionVisualizer(
                VisualizationConfig(
                    spatial_long_side=48,
                    max_items=1,
                    show_spatial_labels=True,
                )
            ).save([record], root / "labelled", images=[source])

            plain_map = next((root / "plain/image-to-image/aggregate/batch-000/maps").glob("*.png"))
            plain_overlay = next(
                (root / "plain/image-to-image/aggregate/batch-000/overlays").glob("*.png")
            )
            labelled_map = next(
                (root / "labelled/image-to-image/aggregate/batch-000/maps").glob("*.png")
            )
            labelled_overlay = next(
                (root / "labelled/image-to-image/aggregate/batch-000/overlays").glob("*.png")
            )
            with Image.open(plain_map) as image:
                self.assertEqual(image.size, (48, 48))
            with Image.open(plain_overlay) as image:
                self.assertEqual(image.size, (48, 24))
            with Image.open(labelled_map) as image:
                self.assertGreater(image.height, 48)
            with Image.open(labelled_overlay) as image:
                self.assertGreater(image.height, 24)

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

    def test_token_concatenated_text_encoders_are_saved_by_component(self):
        components = [
            {
                "name": "clip-fused",
                "indices_by_batch": [[0, 1]],
                "token_indices_by_batch": [[0, 1]],
            },
            {
                "name": "t5",
                "indices_by_batch": [[2, 3]],
                "token_indices_by_batch": [[0, 1]],
            },
        ]
        grid = TokenGrid(height=2, width=2)
        records = [
            make_record(
                "image->text",
                torch.rand(1, 1, 4, 4),
                q_grid=grid,
                metadata={"text_components": components},
            ),
            make_record(
                "text->text",
                torch.rand(1, 1, 4, 4),
                metadata={"text_components": components},
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = AttentionVisualizer(VisualizationConfig(max_items=None)).save(
                records,
                root / "visuals",
                tokenizer={"clip-fused": FakeTokenizer(), "t5": FakeTokenizer()},
                prompts=["red cat"],
            )
            image_maps = root / "visuals/image-to-text/aggregate/batch-000/maps"
            self.assertTrue(list((image_maps / "clip-fused").glob("token-*.png")))
            self.assertTrue(list((image_maps / "t5").glob("token-*.png")))
            matrix_maps = root / "visuals/text-to-text/aggregate/batch-000/maps"
            self.assertTrue((matrix_maps / "clip-fused-to-t5/attention-matrix.png").is_file())
            self.assertTrue((matrix_maps / "t5-to-clip-fused/attention-matrix.png").is_file())
            self.assertFalse(manifest["warnings"])

            collector = AttentionCollector(CaptureConfig(relations="all"))
            for record in records:
                collector.add_record(record)
            collector.compute().save(
                root / "saved",
                save_raw=True,
                save_visualizations=False,
            )
            self.assertEqual(len(list((root / "saved/raw").rglob("*.pt"))), 2)
            self.assertEqual(len(list((root / "saved/raw-components").rglob("*.pt"))), 6)

    def test_video_relations_render_frame_maps_gifs_and_overlays(self):
        grid = TokenGrid(frames=2, height=2, width=3, patch_size=2, temporal_patch_size=1)
        records = [
            make_record("video->text", torch.rand(1, 1, 12, 4), q_grid=grid),
            make_record("text->video", torch.rand(1, 1, 4, 12), k_grid=grid),
            make_record(
                "video->video",
                torch.rand(1, 1, 1, 12),
                q_grid=grid,
                k_grid=grid,
                metadata={"query_indices": (7,)},
            ),
        ]
        source_video = [[Image.new("RGB", (96, 64), color) for color in ("red", "green", "blue", "white")]]
        config = VisualizationConfig(
            spatial_long_side=64,
            max_items=1,
            overview_items=1,
            video_fps=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = AttentionVisualizer(config).save(
                records,
                directory,
                tokenizer=FakeTokenizer(),
                prompts=["red cat"],
                videos=source_video,
            )
            root = Path(directory)
            self.assertEqual(
                set(manifest["relations"]),
                {"video->text", "text->video", "video->video"},
            )
            self.assertFalse(manifest["warnings"])
            for slug in ("video-to-text", "text-to-video", "video-to-video"):
                aggregate = root / slug / "aggregate" / "batch-000"
                self.assertTrue(list((aggregate / "maps").glob("*.gif")))
                self.assertTrue(list((aggregate / "overlays").glob("*.gif")))
                self.assertEqual(len(list((aggregate / "maps").rglob("frame-*.png"))), 2)
                self.assertEqual(len(list((aggregate / "overlays").rglob("frame-*.png"))), 4)

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

    def test_separate_cfg_calls_use_conditional_record_by_default(self):
        conditional = make_record(
            "text->text",
            torch.eye(6).reshape(1, 1, 6, 6),
            metadata={"guidance_role": "conditional"},
        )
        unconditional = make_record(
            "text->text",
            torch.ones(1, 1, 6, 6) / 6,
            metadata={"guidance_role": "unconditional"},
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = AttentionVisualizer(VisualizationConfig()).save(
                [conditional, unconditional],
                directory,
                tokenizer=FakeTokenizer(),
                prompts=["red cat"],
            )
            self.assertEqual(
                manifest["relations"]["text->text"]["aggregate"]["source_records"],
                1,
            )

        with tempfile.TemporaryDirectory() as directory:
            manifest = AttentionVisualizer(VisualizationConfig()).save(
                [conditional, unconditional],
                directory,
                tokenizer=FakeTokenizer(),
                prompts=["red cat"],
                unconditional=False,
            )
            self.assertEqual(
                manifest["relations"]["text->text"]["aggregate"]["source_records"],
                2,
            )

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

    def test_saved_metadata_contains_capture_provenance_and_prompts(self):
        record = make_record("text->text", torch.rand(1, 1, 4, 4))
        collector = AttentionCollector(
            CaptureConfig(relations=["text->text"]),
            metadata={"model_id": "example/video-model", "seed": 2026},
        )
        collector.add_record(record)
        with tempfile.TemporaryDirectory() as directory:
            collector.compute().save(
                directory,
                prompts=["a fox running through snow"],
                save_visualizations=False,
            )
            metadata = json.loads((Path(directory) / "metadata.json").read_text())
            self.assertEqual(metadata["provenance"]["model_id"], "example/video-model")
            self.assertEqual(metadata["provenance"]["seed"], 2026)
            self.assertEqual(metadata["provenance"]["prompts"], ["a fox running through snow"])
            self.assertIn("peak_capture_bytes", metadata["provenance"])

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
    def test_capture_budget_also_covers_maps_materialized_after_fast_capture(self):
        query = torch.randn(1, 1, 4, 8)
        image = ModalityRange("image", 0, 2, TokenGrid(height=1, width=2))
        text = ModalityRange("text", 2, 4)
        collector = AttentionCollector(
            CaptureConfig(
                relations="all",
                mode="fast",
                offload=None,
                max_capture_bytes=260,
            )
        )
        collector.capture_qk(
            layer_id="post-compute-budget",
            attention_type="joint",
            query=query,
            key=query.clone(),
            layout=make_layout([image, text], [image, text]),
        )
        with self.assertRaisesRegex(RuntimeError, "cumulative capture budget"):
            collector.compute()

    def test_total_capture_budget_stops_cumulative_qk_storage(self):
        query = torch.randn(1, 2, 4, 8)
        image = ModalityRange("image", 0, 4, TokenGrid(height=2, width=2))
        collector = AttentionCollector(
            CaptureConfig(
                relations=["image->image"],
                mode="fast",
                offload=None,
                max_capture_bytes=500,
            )
        )
        with self.assertRaisesRegex(RuntimeError, "Capture storage would exceed"):
            collector.capture_qk(
                layer_id="over-budget",
                attention_type="self",
                query=query,
                key=query.clone(),
                layout=make_layout([image], [image]),
            )
        self.assertFalse(collector.records)

    def test_fast_capture_shares_offloaded_qk_and_releases_them_after_compute(self):
        query = torch.randn(1, 2, 4, 8)
        key = torch.randn(1, 2, 4, 8)
        image = ModalityRange("image", 0, 2, TokenGrid(height=1, width=2))
        text = ModalityRange("text", 2, 4)
        collector = AttentionCollector(
            CaptureConfig(relations="all", mode="fast", offload="cpu", dtype="float64")
        )
        collector.capture_qk(
            layer_id="shared-storage",
            attention_type="joint",
            query=query,
            key=key,
            layout=make_layout([image, text], [image, text]),
        )

        self.assertEqual(len(collector.records), 4)
        self.assertEqual(
            len({record.q.untyped_storage().data_ptr() for record in collector.records}),
            1,
        )
        self.assertEqual(
            len({record.k.untyped_storage().data_ptr() for record in collector.records}),
            1,
        )

        maps = collector.compute()
        self.assertTrue(all(record.data is not None for record in maps))
        self.assertTrue(all(record.q is None and record.k is None for record in maps))
        self.assertTrue(
            all(record.q is None and record.k is None for record in collector.records)
        )

    def test_head_mean_is_reduced_during_chunked_compute(self):
        query = torch.randn(2, 4, 7, 8)
        key = torch.randn(2, 4, 5, 8)
        actual = compute_attention_probs(query, key, chunk_size=3, heads="mean")
        expected = torch.softmax(query.float() @ key.float().transpose(-2, -1) / (8**0.5), dim=-1).mean(
            dim=1, keepdim=True
        )
        self.assertEqual(tuple(actual.shape), (2, 1, 7, 5))
        torch.testing.assert_close(actual, expected)

    def test_boolean_attention_mask_uses_sdpa_semantics(self):
        torch.manual_seed(41)
        query = torch.randn(1, 2, 4, 8)
        key = torch.randn(1, 2, 5, 8)
        mask = torch.tensor([[[[True, True, False, True, False]]]])
        actual = compute_attention_probs(query, key, attention_mask=mask, chunk_size=2, heads="all")
        logits = torch.matmul(query.float(), key.float().transpose(-2, -1)) * (8**-0.5)
        expected = torch.softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=-1)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.equal(actual[..., 2], torch.zeros_like(actual[..., 2])))
        self.assertTrue(torch.equal(actual[..., 4], torch.zeros_like(actual[..., 4])))

    def test_fast_capture_preserves_boolean_mask_when_qk_are_cast(self):
        query = torch.randn(1, 1, 4, 8)
        image = ModalityRange("image", 0, 4, TokenGrid(height=2, width=2))
        mask = torch.tensor([[[[True, True, False, True]]]])
        collector = AttentionCollector(
            CaptureConfig(
                relations=["image->image"],
                mode="fast",
                offload="cpu",
                dtype="float16",
            )
        )
        collector.capture_qk(
            layer_id="masked",
            attention_type="self",
            query=query,
            key=query,
            layout=make_layout([image], [image]),
            attention_mask=mask,
        )
        self.assertEqual(collector.records[0].attention_mask.dtype, torch.bool)
        data = collector.compute().records[0].data
        self.assertTrue(torch.equal(data[..., 2], torch.zeros_like(data[..., 2])))

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
