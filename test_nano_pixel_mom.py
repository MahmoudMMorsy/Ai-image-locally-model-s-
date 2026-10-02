import unittest
import os
import torch
from models.nano_pixel_mom.model import NanoPixelMomUNet, NanoPixelMomDecoder
from models.nano_pixel_mom.dataset_processor import PixelMomDataset
from models.nano_pixel_mom.inference_engine import NanoPixelMomEngine

class TestNanoPixelMom(unittest.TestCase):
    def test_unet_architecture(self):
        unet = NanoPixelMomUNet()
        x = torch.randn(2, 4, 64, 64)
        t = torch.tensor([5, 10]).long()
        c = torch.randn(2, 64)
        out = unet(x, t, c)
        self.assertEqual(out.shape, (2, 4, 64, 64))

    def test_decoder_architecture(self):
        decoder = NanoPixelMomDecoder()
        z = torch.randn(2, 4, 64, 64)
        out = decoder(z)
        self.assertEqual(out.shape, (2, 4, 64, 64))

    def test_dataset_loader(self):
        dataset = PixelMomDataset()
        self.assertGreater(len(dataset), 0)
        img, cond = dataset[0]
        self.assertEqual(img.shape, (4, 64, 64))
        self.assertEqual(cond.shape, (64,))

    def test_engine_inference(self):
        engine = NanoPixelMomEngine()
        sprite = engine.generate_sprite(prompt="test hero", seed=123)
        self.assertEqual(sprite.size, (64, 64))
        self.assertEqual(sprite.mode, "RGBA")

    def test_weights_and_onnx_exist(self):
        weights_dir = "models/nano_pixel_mom/weights"
        self.assertTrue(os.path.exists(os.path.join(weights_dir, "nano_pixel_mom_unet.pt")))
        self.assertTrue(os.path.exists(os.path.join(weights_dir, "nano_pixel_mom_decoder.pt")))
        self.assertTrue(os.path.exists(os.path.join(weights_dir, "nano_pixel_mom_unet.onnx")))
        self.assertTrue(os.path.exists(os.path.join(weights_dir, "nano_pixel_mom_decoder.onnx")))

if __name__ == "__main__":
    unittest.main()
