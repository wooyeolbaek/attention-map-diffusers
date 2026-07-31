import unittest

import torch
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor2_0,
    JointAttnProcessor2_0,
    SanaLinearAttnProcessor2_0,
)
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
