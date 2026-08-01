import unittest

import torch
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor2_0,
    CogVideoXAttnProcessor2_0,
    FusedCogVideoXAttnProcessor2_0,
    JointAttnProcessor2_0,
    SanaLinearAttnProcessor2_0,
)
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoAttnProcessor2_0
from diffusers.models.transformers.transformer_hunyuan_video15 import HunyuanVideo15AttnProcessor2_0
from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2AttnProcessor,
    Flux2ParallelSelfAttention,
    Flux2ParallelSelfAttnProcessor,
)
from diffusers.models.transformers.transformer_prx import PRXAttention, PRXAttnProcessor2_0
from diffusers.models.transformers.transformer_z_image import ZSingleStreamAttnProcessor

from attention_map_diffusers.capture import AttentionCollector, CaptureConfig
from attention_map_diffusers.capture.context import CaptureRuntimeContext, pushed_context
from attention_map_diffusers.capture.records import TokenGrid
from attention_map_diffusers.processors.capture import make_capture_processor


class ProcessorParityTests(unittest.TestCase):
    def _compare(self, attention, processor, hidden_states, encoder_hidden_states, relations):
        attention.set_processor(processor)
        with torch.no_grad():
            expected = attention(hidden_states, encoder_hidden_states=encoder_hidden_states)

        collector = AttentionCollector(
            CaptureConfig(relations=relations, mode="exact", offload=None, heads="mean", allow_large_maps=True)
        )
        capture_processor = make_capture_processor(processor, collector, "test.attn")
        self.assertIsNotNone(capture_processor)
        attention.set_processor(capture_processor)
        with torch.no_grad():
            actual = attention(hidden_states, encoder_hidden_states=encoder_hidden_states)

        expected_values = expected if isinstance(expected, tuple) else (expected,)
        actual_values = actual if isinstance(actual, tuple) else (actual,)
        self.assertEqual(len(actual_values), len(expected_values))
        for actual_value, expected_value in zip(actual_values, expected_values):
            torch.testing.assert_close(actual_value, expected_value)
        self.assertEqual({record.relation for record in collector.records}, set(relations))
        return collector

    def test_sdpa_cross_attention_parity(self):
        torch.manual_seed(0)
        attention = Attention(
            query_dim=32,
            cross_attention_dim=32,
            heads=4,
            dim_head=8,
            qk_norm="layer_norm",
        )
        self._compare(
            attention,
            AttnProcessor2_0(),
            torch.randn(2, 6, 32),
            torch.randn(2, 5, 32),
            ["image->text"],
        )

    def test_sd3_joint_attention_parity(self):
        torch.manual_seed(1)
        attention = _joint_attention()
        self._compare(
            attention,
            JointAttnProcessor2_0(),
            torch.randn(2, 6, 32),
            torch.randn(2, 5, 32),
            ["image->text", "text->image", "image->image", "text->text"],
        )

    def test_flux_joint_attention_parity(self):
        torch.manual_seed(2)
        attention = _joint_attention()
        self._compare(
            attention,
            FluxAttnProcessor(),
            torch.randn(2, 6, 32),
            torch.randn(2, 5, 32),
            ["image->text", "text->image", "image->image", "text->text"],
        )

    def test_flux2_double_stream_parity(self):
        torch.manual_seed(21)
        attention = Flux2Attention(
            query_dim=32,
            added_kv_proj_dim=32,
            heads=4,
            dim_head=8,
            out_dim=32,
        )
        self._compare(
            attention,
            Flux2AttnProcessor(),
            torch.randn(2, 6, 32),
            torch.randn(2, 5, 32),
            ["image->text", "text->image", "image->image", "text->text"],
        )

    def test_flux2_parallel_single_stream_parity(self):
        torch.manual_seed(22)
        attention = Flux2ParallelSelfAttention(query_dim=32, heads=4, dim_head=8, out_dim=32)
        processor = Flux2ParallelSelfAttnProcessor()
        hidden_states = torch.randn(2, 11, 32)
        attention.set_processor(processor)
        with torch.no_grad():
            expected = attention(hidden_states)
        collector = AttentionCollector(
            CaptureConfig(
                relations=["image->text", "text->image", "image->image", "text->text"],
                mode="exact",
                offload=None,
                allow_large_maps=True,
            )
        )
        attention.set_processor(make_capture_processor(processor, collector, "transformer.single.attn"))
        context = CaptureRuntimeContext(
            image_grid=TokenGrid(height=2, width=3), image_tokens=6, text_tokens=5
        )
        with torch.no_grad(), pushed_context(context):
            actual = attention(hidden_states)
        torch.testing.assert_close(actual, expected)
        self.assertEqual({record.relation for record in collector.records}, set(collector.config.relations))

    def test_z_image_unified_stream_parity_with_boolean_mask(self):
        torch.manual_seed(23)
        attention = Attention(
            query_dim=32,
            heads=4,
            dim_head=8,
            qk_norm="rms_norm",
            bias=False,
            out_bias=False,
        )
        processor = ZSingleStreamAttnProcessor()
        hidden_states = torch.randn(2, 9, 32)
        attention_mask = torch.tensor(
            [[True] * 8 + [False], [True] * 7 + [False, False]], dtype=torch.bool
        )
        attention.set_processor(processor)
        with torch.no_grad():
            expected = attention(hidden_states, attention_mask=attention_mask)
        collector = AttentionCollector(
            CaptureConfig(
                relations=["image->text", "text->image", "image->image", "text->text"],
                mode="exact",
                offload=None,
                allow_large_maps=True,
            )
        )
        attention.set_processor(make_capture_processor(processor, collector, "transformer.layers.0.attention"))
        context = CaptureRuntimeContext(
            image_grid=TokenGrid(height=2, width=3), image_tokens=6, text_tokens=3
        )
        with torch.no_grad(), pushed_context(context):
            actual = attention(hidden_states, attention_mask=attention_mask)
        torch.testing.assert_close(actual, expected)
        self.assertEqual({record.relation for record in collector.records}, set(collector.config.relations))
        validation = collector.compute().validate()
        self.assertEqual(validation["status"], "pass")
        self.assertEqual(validation["joint_normalization"]["groups"], 1)

    def test_prx_image_query_parity(self):
        torch.manual_seed(24)
        attention = PRXAttention(query_dim=32, heads=4, dim_head=8)
        processor = PRXAttnProcessor2_0()
        hidden_states = torch.randn(2, 6, 32)
        encoder_hidden_states = torch.randn(2, 5, 32)
        attention_mask = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, False]], dtype=torch.bool
        )
        attention.set_processor(processor)
        with torch.no_grad():
            expected = attention(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )
        collector = AttentionCollector(
            CaptureConfig(
                relations=["image->text", "image->image"],
                mode="exact",
                offload=None,
                allow_large_maps=True,
            )
        )
        attention.set_processor(make_capture_processor(processor, collector, "transformer.blocks.0.attn"))
        context = CaptureRuntimeContext(
            image_grid=TokenGrid(height=2, width=3), image_tokens=6, text_tokens=5
        )
        with torch.no_grad(), pushed_context(context):
            actual = attention(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )
        torch.testing.assert_close(actual, expected)
        self.assertEqual({record.relation for record in collector.records}, {"image->text", "image->image"})
        validation = collector.compute().validate()
        self.assertEqual(validation["status"], "pass")
        self.assertEqual(validation["joint_normalization"]["groups"], 1)

    def test_sana_linear_attention_parity_and_normalization(self):
        torch.manual_seed(3)
        attention = Attention(query_dim=32, heads=4, dim_head=8)
        collector = self._compare(
            attention,
            SanaLinearAttnProcessor2_0(),
            torch.randn(2, 16, 32),
            None,
            ["image->image"],
        )
        row_mass = collector.records[0].data.float().sum(dim=-1)
        self.assertTrue(torch.all(row_mass <= 1.0 + 1e-5))
        # ReLU linear attention can have an all-zero q vector for a head. The
        # mean map mass is therefore the number of active heads / total heads.
        active_heads = row_mass * attention.heads
        torch.testing.assert_close(active_heads, active_heads.round(), atol=1e-5, rtol=1e-5)

    def test_cogvideox_joint_video_attention_parity(self):
        torch.manual_seed(31)
        attention = Attention(query_dim=32, heads=4, dim_head=8, qk_norm="layer_norm")
        processor = CogVideoXAttnProcessor2_0()
        hidden_states = torch.randn(2, 6, 32)
        encoder_hidden_states = torch.randn(2, 5, 32)
        attention.set_processor(processor)
        with torch.no_grad():
            expected = attention(hidden_states, encoder_hidden_states=encoder_hidden_states)

        collector = AttentionCollector(
            CaptureConfig(
                relations=["video->text", "text->video", "video->video", "text->text"],
                mode="exact",
                offload=None,
                allow_large_maps=True,
            )
        )
        attention.set_processor(make_capture_processor(processor, collector, "transformer.blocks.0.attn"))
        context = CaptureRuntimeContext(
            video_grid=TokenGrid(frames=2, height=1, width=3), video_tokens=6, text_tokens=5
        )
        with torch.no_grad(), pushed_context(context):
            actual = attention(hidden_states, encoder_hidden_states=encoder_hidden_states)
        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(actual_value, expected_value)
        self.assertEqual({record.relation for record in collector.records}, set(collector.config.relations))
        self.assertEqual(collector.compute().validate()["status"], "pass")

    def test_fused_cogvideox_joint_video_attention_parity(self):
        torch.manual_seed(35)
        attention = Attention(query_dim=32, heads=4, dim_head=8, qk_norm="layer_norm")
        attention.fuse_projections(fuse=True)
        processor = FusedCogVideoXAttnProcessor2_0()
        hidden_states = torch.randn(2, 6, 32)
        encoder_hidden_states = torch.randn(2, 5, 32)
        attention.set_processor(processor)
        with torch.no_grad():
            expected = attention(hidden_states, encoder_hidden_states=encoder_hidden_states)
        collector = AttentionCollector(
            CaptureConfig(relations="all", mode="exact", offload=None, allow_large_maps=True)
        )
        attention.set_processor(make_capture_processor(processor, collector, "transformer.blocks.0.attn"))
        context = CaptureRuntimeContext(
            video_grid=TokenGrid(frames=2, height=1, width=3), video_tokens=6, text_tokens=5
        )
        with torch.no_grad(), pushed_context(context):
            actual = attention(hidden_states, encoder_hidden_states=encoder_hidden_states)
        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(actual_value, expected_value)
        self.assertEqual(collector.compute().validate()["status"], "pass")

    def test_hunyuan_video_joint_attention_parity(self):
        torch.manual_seed(32)
        attention = _joint_attention()
        processor = HunyuanVideoAttnProcessor2_0()
        hidden_states = torch.randn(2, 6, 32)
        encoder_hidden_states = torch.randn(2, 5, 32)
        attention.set_processor(processor)
        with torch.no_grad():
            expected = attention(hidden_states, encoder_hidden_states=encoder_hidden_states)

        collector = AttentionCollector(
            CaptureConfig(relations="all", mode="exact", offload=None, allow_large_maps=True)
        )
        attention.set_processor(make_capture_processor(processor, collector, "transformer.blocks.0.attn"))
        context = CaptureRuntimeContext(
            video_grid=TokenGrid(frames=2, height=1, width=3), video_tokens=6, text_tokens=5
        )
        with torch.no_grad(), pushed_context(context):
            actual = attention(hidden_states, encoder_hidden_states=encoder_hidden_states)
        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(actual_value, expected_value)
        self.assertEqual(
            {record.relation for record in collector.records},
            {"video->text", "text->video", "video->video", "text->text"},
        )
        self.assertEqual(collector.compute().validate()["status"], "pass")

    def test_hunyuan_video15_joint_attention_parity_with_mask(self):
        torch.manual_seed(33)
        attention = _joint_attention()
        processor = HunyuanVideo15AttnProcessor2_0()
        hidden_states = torch.randn(2, 6, 32)
        encoder_hidden_states = torch.randn(2, 5, 32)
        attention_mask = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, False]]
        )
        attention.set_processor(processor)
        with torch.no_grad():
            expected = attention(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )

        collector = AttentionCollector(
            CaptureConfig(relations="all", mode="exact", offload=None, allow_large_maps=True)
        )
        attention.set_processor(make_capture_processor(processor, collector, "transformer.blocks.0.attn"))
        context = CaptureRuntimeContext(
            video_grid=TokenGrid(frames=2, height=1, width=3), video_tokens=6, text_tokens=5
        )
        with torch.no_grad(), pushed_context(context):
            actual = attention(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )
        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(actual_value, expected_value)
        self.assertEqual(collector.compute().validate()["status"], "pass")

    def test_hunyuan_video15_mllm_refiner_is_captured_as_text_attention(self):
        torch.manual_seed(36)
        attention = Attention(query_dim=32, heads=4, dim_head=8)
        processor = AttnProcessor2_0()
        hidden_states = torch.randn(2, 5, 32)
        active = [[0, 1, 3], [0, 2, 3, 4]]
        attention_mask = torch.zeros(2, 1, 5, 5, dtype=torch.bool)
        for batch_index, indices in enumerate(active):
            attention_mask[batch_index, 0][indices] = True
            attention_mask[batch_index, 0, :, [index for index in range(5) if index not in indices]] = False

        attention.set_processor(processor)
        with torch.no_grad():
            expected = attention(hidden_states, attention_mask=attention_mask)

        collector = AttentionCollector(
            CaptureConfig(relations="all", mode="exact", offload=None, allow_large_maps=True)
        )
        layer_id = "transformer.context_embedder.token_refiner.refiner_blocks.0.attn"
        attention.set_processor(make_capture_processor(processor, collector, layer_id))
        context = CaptureRuntimeContext(
            text_components=[
                {
                    "name": "byt5",
                    "indices_by_batch": [[0], [0]],
                    "token_indices_by_batch": [[0], [0]],
                },
                {
                    "name": "mllm",
                    "indices_by_batch": [[4, 5, 6], [4, 5, 6, 7]],
                    "token_indices_by_batch": active,
                },
            ],
            metadata={"text_components": []},
        )
        with torch.no_grad(), pushed_context(context):
            actual = attention(hidden_states, attention_mask=attention_mask)

        torch.testing.assert_close(actual, expected)
        self.assertEqual([record.relation for record in collector.records], ["text->text"])
        record = collector.records[0]
        self.assertEqual(record.metadata["text_attention_stage"], "token-refiner")
        self.assertEqual(
            record.metadata["text_components"][0]["indices_by_batch"],
            active,
        )
        maps = collector.compute()
        self.assertEqual(maps.validate()["status"], "pass")
        for batch_index, indices in enumerate(active):
            inactive = [index for index in range(5) if index not in indices]
            if inactive:
                self.assertTrue(
                    torch.equal(
                        maps.records[0].data[batch_index, :, inactive],
                        torch.zeros_like(maps.records[0].data[batch_index, :, inactive]),
                    )
                )

    def test_wan_video_self_and_text_cross_attention_parity(self):
        torch.manual_seed(34)
        context = CaptureRuntimeContext(
            video_grid=TokenGrid(frames=2, height=1, width=3), video_tokens=6, text_tokens=5
        )
        for is_cross, relation in ((False, "video->video"), (True, "video->text")):
            attention = WanAttention(
                dim=32,
                heads=4,
                dim_head=8,
                cross_attention_dim_head=8 if is_cross else None,
                is_cross_attention=is_cross,
                processor=WanAttnProcessor(),
            )
            hidden_states = torch.randn(2, 6, 32)
            encoder_hidden_states = torch.randn(2, 5, 32) if is_cross else None
            with torch.no_grad():
                expected = attention(hidden_states, encoder_hidden_states=encoder_hidden_states)
            collector = AttentionCollector(
                CaptureConfig(relations=[relation], mode="exact", offload=None, allow_large_maps=True)
            )
            attention.set_processor(
                make_capture_processor(attention.processor, collector, f"transformer.{relation}.attn")
            )
            with torch.no_grad(), pushed_context(context):
                actual = attention(hidden_states, encoder_hidden_states=encoder_hidden_states)
            torch.testing.assert_close(actual, expected)
            self.assertEqual([record.relation for record in collector.records], [relation])
            self.assertEqual(collector.compute().validate()["status"], "pass")


def _joint_attention():
    return Attention(
        query_dim=32,
        added_kv_proj_dim=32,
        heads=4,
        dim_head=8,
        out_dim=32,
        context_pre_only=False,
        qk_norm="rms_norm",
    )


if __name__ == "__main__":
    unittest.main()
