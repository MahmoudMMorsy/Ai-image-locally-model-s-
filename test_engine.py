"""
Comprehensive Unit Tests for Pixel Art Engine
Tests Text-to-Sprite, Image-to-Sprite conversion, Palette Quantization, and Animation generation.
"""
import unittest
import numpy as np
from PIL import Image, ImageDraw
import torch

from pixel_art_engine.palette import quantize_to_pixel_art, SIGNATURE_PALETTE
from pixel_art_engine.engine import PixelSpriteEngine
from pixel_art_engine.converter import ImageToSpriteConverter
from pixel_art_engine.animation import SpriteAnimationGenerator


class TestPixelArtEngine(unittest.TestCase):

    def setUp(self):
        self.engine = PixelSpriteEngine(device="cpu")
        self.converter = ImageToSpriteConverter(self.engine, device="cpu")
        self.animator = SpriteAnimationGenerator(self.engine, device="cpu")

    def test_palette_quantization(self):
        """Test crisp 64x64 quantization to signature palette."""
        dummy_img = Image.new("RGBA", (100, 100), (200, 100, 50, 255))
        quantized = quantize_to_pixel_art(dummy_img, size=(64, 64), palette=SIGNATURE_PALETTE)

        self.assertEqual(quantized.size, (64, 64))
        self.assertEqual(quantized.mode, "RGBA")

        # Check all colors in quantized image belong strictly to SIGNATURE_PALETTE
        arr = np.array(quantized)
        unique_colors = np.unique(arr.reshape(-1, 4), axis=0)

        for c in unique_colors:
            matches = np.all(SIGNATURE_PALETTE == c, axis=1)
            self.assertTrue(np.any(matches), f"Color {c} not in signature palette")

    def test_text_to_sprite_generation(self):
        """Test deterministic generation of 64x64 pixel sprites from text prompts."""
        img1 = self.engine.generate_sprite(prompt="pixel knight warrior", seed=42)
        img2 = self.engine.generate_sprite(prompt="pixel knight warrior", seed=42)

        self.assertEqual(img1.size, (64, 64))
        self.assertEqual(img1.mode, "RGBA")

        # Same seed should yield identical pixel array
        np.testing.assert_array_equal(np.array(img1), np.array(img2))

    def test_image_to_sprite_conversion(self):
        """Test img2img conversion of external images into signature pixel art characters."""
        input_img = Image.new("RGBA", (250, 250), (255, 255, 255, 0))
        draw = ImageDraw.Draw(input_img)
        draw.rectangle([50, 50, 200, 200], fill=(50, 120, 220, 255))

        sprite = self.converter.convert_image_to_sprite(input_img, strength=0.4)
        self.assertEqual(sprite.size, (64, 64))
        self.assertEqual(sprite.mode, "RGBA")

    def test_animation_generation(self):
        """Test sequential frame animation generation, sprite sheet layout, and GIF encoding."""
        base_char = self.engine.generate_sprite(prompt="pixel hero warrior", seed=100)
        frames, sheet, gif_bytes = self.animator.generate_animation(base_char, action="run", num_frames=4)

        self.assertEqual(len(frames), 4)
        for f in frames:
            self.assertEqual(f.size, (64, 64))

        self.assertEqual(sheet.size, (256, 64))
        self.assertGreater(len(gif_bytes), 0)


if __name__ == "__main__":
    unittest.main()
