import unittest
from types import SimpleNamespace

import torch

from attention_map_diffusers.attention_capture import _context_from_forward


class CaptureContextTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
