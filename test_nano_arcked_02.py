import unittest
from PIL import Image
from models.nano_arcked_02.inference.nanopixel_engine import NanoArcKed02Engine

class TestNanoArcKed02Engine(unittest.TestCase):
    def setUp(self):
        self.engine = NanoArcKed02Engine(device="cpu")

    def test_text_to_image(self):
        img_sig = self.engine.generate_sprite("pixel paladin knight", palette_mode="signature", seed=42)
        img_gb = self.engine.generate_sprite("pixel paladin knight", palette_mode="gameboy", seed=42)
        img_nes = self.engine.generate_sprite("pixel paladin knight", palette_mode="nes", seed=42)

        self.assertEqual(img_sig.size, (64, 64))
        self.assertEqual(img_gb.size, (64, 64))
        self.assertEqual(img_nes.size, (64, 64))

    def test_image_to_image(self):
        input_img = Image.new("RGBA", (64, 64), (100, 150, 200, 255))
        converted = self.engine.convert_image_to_sprite(input_img, strength=0.5, palette_mode="gameboy")
        self.assertEqual(converted.size, (64, 64))

    def test_animation_and_anim2anim(self):
        base_char = Image.new("RGBA", (64, 64), (200, 100, 50, 255))
        frames, sheet, gif_bytes = self.engine.generate_animation(base_char, action="run", num_frames=4, palette_mode="nes")

        self.assertEqual(len(frames), 4)
        self.assertEqual(sheet.size, (256, 64))
        self.assertGreater(len(gif_bytes), 0)

        a2a_frames, a2a_sheet, a2a_gif_bytes = self.engine.anim2anim_synthesis(frames, target_action="attack", palette_mode="signature")
        self.assertEqual(len(a2a_frames), 4)
        self.assertEqual(a2a_sheet.size, (256, 64))
        self.assertGreater(len(a2a_gif_bytes), 0)

if __name__ == "__main__":
    unittest.main()
