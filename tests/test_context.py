import unittest
from types import SimpleNamespace

import torch
from torch import nn
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from attention_map_diffusers import AttentionCapture
from attention_map_diffusers.attention_capture import _context_from_forward, _guidance_role


class CaptureContextTests(unittest.TestCase):
    def test_separate_guidance_call_roles_are_stable(self):
        self.assertEqual(_guidance_role(0, enabled=True, conditions=2), "conditional")
        self.assertEqual(_guidance_role(1, enabled=True, conditions=2), "unconditional")
        self.assertEqual(_guidance_role(2, enabled=True, conditions=3), "guidance-2")
        self.assertEqual(_guidance_role(4, enabled=False, conditions=2), "conditional")

    def test_sd3_token_concatenation_preserves_clip_and_t5_boundaries(self):
        module = type("SD3Transformer2DModel", (), {})()
        module.config = SimpleNamespace(patch_size=2)
        schema = {
            "components": [
                {"name": "clip-l+clip-g-fused", "tokenizer_attribute": "tokenizer", "length": 3},
                {"name": "t5", "tokenizer_attribute": "tokenizer_3"},
            ]
        }
        context = _context_from_forward(
            module,
            (),
            {
                "hidden_states": torch.randn(2, 4, 8),
                "encoder_hidden_states": torch.randn(2, 8, 16),
            },
            "transformer",
            text_conditioning=schema,
        )
        components = context.text_components
        self.assertEqual([item["name"] for item in components], ["clip-l+clip-g-fused", "t5"])
        self.assertEqual(components[0]["indices_by_batch"][0], [0, 1, 2])
        self.assertEqual(components[1]["indices_by_batch"][0], [3, 4, 5, 6, 7])

    def test_hunyuan15_reordered_text_components_follow_active_token_positions(self):
        module = type("HunyuanVideo15Transformer3DModel", (), {})()
        module.config = SimpleNamespace(patch_size=2, patch_size_t=1)
        schema = {
            "components": [
                {"name": "byt5", "tokenizer_attribute": "tokenizer_2"},
                {"name": "mllm", "tokenizer_attribute": "tokenizer"},
            ]
        }
        context = _context_from_forward(
            module,
            (),
            {
                "hidden_states": torch.randn(1, 8, 3, 8, 8),
                "encoder_hidden_states": torch.randn(1, 3, 16),
                "encoder_attention_mask": torch.tensor([[1, 1, 0]]),
                "encoder_hidden_states_2": torch.randn(1, 4, 16),
                "encoder_attention_mask_2": torch.tensor([[1, 0, 1, 0]]),
                "image_embeds": torch.zeros(1, 2, 16),
            },
            "transformer",
            text_conditioning=schema,
        )
        components = {item["name"]: item for item in context.text_components}
        self.assertEqual(components["byt5"]["indices_by_batch"][0], [0, 1])
        self.assertEqual(components["byt5"]["token_indices_by_batch"][0], [0, 2])
        self.assertEqual(components["mllm"]["indices_by_batch"][0], [2, 3])
        self.assertEqual(components["mllm"]["token_indices_by_batch"][0], [0, 1])

    def test_video_contexts_preserve_temporal_and_spatial_patch_grid(self):
        cases = (
            ("CogVideoXTransformer3DModel", (1, 5, 8, 8, 12), 2, None, (5, 4, 6)),
            ("HunyuanVideo15Transformer3DModel", (1, 8, 5, 8, 12), 2, 1, (5, 4, 6)),
            ("WanTransformer3DModel", (1, 8, 5, 8, 12), (1, 2, 2), None, (5, 4, 6)),
        )
        for class_name, shape, patch_size, patch_size_t, expected in cases:
            module = type(class_name, (), {})()
            module.config = SimpleNamespace(patch_size=patch_size, patch_size_t=patch_size_t)
            hidden_states = torch.randn(shape)
            encoder_hidden_states = torch.randn(1, 7, 32)
            context = _context_from_forward(
                module,
                (),
                {
                    "hidden_states": hidden_states,
                    "encoder_hidden_states": encoder_hidden_states,
                    "timestep": torch.tensor([0.75]),
                },
                "transformer",
            )
            self.assertEqual(
                (context.video_grid.frames, context.video_grid.height, context.video_grid.width),
                expected,
            )
            self.assertEqual(context.video_tokens, expected[0] * expected[1] * expected[2])
            self.assertEqual(context.text_tokens, 7)

    def test_flux2_context_uses_position_id_grid(self):
        module = type("Flux2Transformer2DModel", (), {})()
        img_ids = torch.tensor(
            [[[0, row, col, 0] for row in range(2) for col in range(3)]], dtype=torch.long
        )
        context = _context_from_forward(
            module,
            (),
            {
                "hidden_states": torch.randn(1, 6, 32),
                "encoder_hidden_states": torch.randn(1, 5, 64),
                "img_ids": img_ids,
                "timestep": torch.tensor([0.5]),
            },
            "transformer",
        )
        self.assertEqual((context.image_grid.height, context.image_grid.width), (2, 3))
        self.assertEqual((context.image_tokens, context.text_tokens), (6, 5))

    def test_z_image_context_accounts_for_patch_and_sequence_padding(self):
        module = type("ZImageTransformer2DModel", (), {})()
        context = _context_from_forward(
            module,
            (),
            {
                "x": [torch.randn(16, 1, 32, 48)],
                "t": torch.tensor([0.25]),
                "cap_feats": [torch.randn(17, 2560)],
                "patch_size": 2,
                "f_patch_size": 1,
            },
            "transformer",
        )
        self.assertEqual((context.image_grid.height, context.image_grid.width), (16, 24))
        self.assertEqual(context.image_tokens, 384)
        self.assertEqual(context.text_tokens, 32)

    def test_prx_context_uses_patch_grid_instead_of_channel_count(self):
        module = type("PRXTransformer2DModel", (), {})()
        module.config = SimpleNamespace(patch_size=16)
        context = _context_from_forward(
            module,
            (),
            {
                "hidden_states": torch.randn(1, 3, 64, 96),
                "encoder_hidden_states": torch.randn(1, 12, 2048),
                "timestep": torch.tensor([0.75]),
            },
            "transformer",
        )
        self.assertEqual((context.image_grid.height, context.image_grid.width), (4, 6))
        self.assertEqual((context.image_tokens, context.text_tokens), (24, 12))


class CaptureIsolationTests(unittest.TestCase):
    def test_capture_reports_complete_processor_coverage(self):
        pipeline = _FakePipeline()
        with AttentionCapture(pipeline, relations=["image->text"]) as capture:
            self.assertEqual(capture.coverage["attention_modules"], 1)
            self.assertEqual(capture.coverage["captured_modules"], 1)
            self.assertFalse(capture.coverage["unsupported"])

    def test_unknown_denoiser_processor_fails_instead_of_being_silently_skipped(self):
        pipeline = _FakePipeline()
        pipeline.transformer.attn.set_processor(_UnknownProcessor())
        with self.assertRaisesRegex(RuntimeError, "coverage is incomplete"):
            with AttentionCapture(pipeline, relations=["image->text"]):
                pass

    def test_two_pipeline_instances_capture_and_restore_independently(self):
        first = _FakePipeline()
        second = _FakePipeline()
        first_original = first.transformer.attn.processor
        second_original = second.transformer.attn.processor
        with AttentionCapture(first, relations=["image->text"]) as first_capture:
            self.assertIsNot(first.transformer.attn.processor, first_original)
            with AttentionCapture(second, relations=["image->text"]) as second_capture:
                self.assertIsNot(second.transformer.attn.processor, second_original)
                self.assertIsNot(first_capture.collector, second_capture.collector)
            self.assertIs(second.transformer.attn.processor, second_original)
            self.assertIsNot(first.transformer.attn.processor, first_original)
        self.assertIs(first.transformer.attn.processor, first_original)

    def test_same_pipeline_nested_capture_is_rejected_without_corrupting_outer_capture(self):
        pipeline = _FakePipeline()
        original = pipeline.transformer.attn.processor
        with AttentionCapture(pipeline, relations=["image->text"]):
            outer_processor = pipeline.transformer.attn.processor
            with self.assertRaisesRegex(RuntimeError, "already has an active AttentionCapture"):
                with AttentionCapture(pipeline, relations=["image->text"]):
                    pass
            self.assertIs(pipeline.transformer.attn.processor, outer_processor)
        self.assertIs(pipeline.transformer.attn.processor, original)

    def test_exception_restores_processors_and_hooks(self):
        pipeline = _FakePipeline()
        original = pipeline.transformer.attn.processor
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            with AttentionCapture(pipeline, relations=["image->text"]):
                raise RuntimeError("generation failed")
        self.assertIs(pipeline.transformer.attn.processor, original)
        self.assertFalse(hasattr(pipeline.transformer.attn, "_attention_map_capture_owner"))


class _FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention(
            query_dim=16,
            cross_attention_dim=16,
            heads=2,
            dim_head=8,
            processor=AttnProcessor2_0(),
        )

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None):
        return self.attn(hidden_states, encoder_hidden_states=encoder_hidden_states)


class _FakePipeline:
    def __init__(self):
        self.transformer = _FakeTransformer()


class _UnknownProcessor:
    pass


if __name__ == "__main__":
    unittest.main()
