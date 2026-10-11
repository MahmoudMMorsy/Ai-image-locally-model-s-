"""Unit tests for arcade_t2i neural text-to-image module."""
import os
import unittest
import torch
import numpy as np

from text_lang import TextProcessor
from model_t2i import TextCVAE
from generate_t2i import infer_class_id

class TestArcadeT2I(unittest.TestCase):
    def setUp(self):
        self.tp = TextProcessor()

    def test_text_normalization_and_tokenization(self):
        tokens_ar = self.tp.tokenize("فارس")
        tokens_en = self.tp.tokenize("knight")
        self.assertEqual(tokens_ar, tokens_en)
        self.assertEqual(tokens_ar[0], 2)

    def test_infer_class_id(self):
        self.assertEqual(infer_class_id("knight"), 0)
        self.assertEqual(infer_class_id("فارس"), 0)
        self.assertEqual(infer_class_id("mage"), 1)
        self.assertEqual(infer_class_id("ساحر"), 1)

    def test_model_forward_and_generate(self):
        model = TextCVAE(n_colors=48, latent=64, base=24, cond_dim=128, n_tokens=100)
        tokens = torch.tensor([[2, 0, 0, 0]], dtype=torch.long)
        class_id = torch.tensor([0], dtype=torch.long)
        dummy_idx = torch.randint(0, 48, (1, 64, 64), dtype=torch.long)

        logits, mu, lv = model(dummy_idx, tokens, class_id)
        self.assertEqual(logits.shape, torch.Size([1, 48, 64, 64]))

        gen = model.generate(tokens, class_id, n=1)
        self.assertEqual(gen.shape, torch.Size([1, 64, 64]))

if __name__ == "__main__":
    unittest.main()
