"""Unit tests for the flux_student character text-to-image pipeline."""
import unittest
from pathlib import Path
import torch
from student.model import StudentT2I, count_params
from student.filter import is_safe_prompt
from scripts.inference import tokenize, sample_character


class TestFluxStudentPipeline(unittest.TestCase):
    def test_parameter_counts(self):
        m48 = StudentT2I(base=48)
        p48 = count_params(m48)
        self.assertAlmostEqual(p48 / 1e6, 10.0, delta=0.5)

        m96 = StudentT2I(base=96)
        p96 = count_params(m96)
        self.assertAlmostEqual(p96 / 1e6, 26.6, delta=1.0)

    def test_forward_pass(self):
        model = StudentT2I(base=48)
        x = torch.randn(2, 3, 64, 64)
        t = torch.randint(0, 1000, (2,))
        ids = torch.randint(0, 1000, (2, 20))
        out = model(x, t, ids)
        self.assertEqual(out.shape, (2, 3, 64, 64))

    def test_tokenization(self):
        toks = tokenize("a warrior with sword and shield")
        self.assertEqual(toks.shape, (1, 20))
        self.assertEqual(toks[0, 0].item(), 1)  # BOS token

    def test_content_filter(self):
        self.assertTrue(is_safe_prompt("a cyber soldier in armor"))
        self.assertTrue(is_safe_prompt("فارس بدرع لامع"))
        self.assertFalse(is_safe_prompt("the prophet Muhammad in a desert"))
        self.assertFalse(is_safe_prompt("تمثال نبي الله موسى"))

    def test_sampling_and_resize(self):
        tmp_out = Path("/tmp/test_sample_32.png")
        sample_character(
            prompt="a robot android character, full body",
            checkpoint_path=Path("/tmp/nonexistent.pt"),
            out_path=tmp_out,
            target_size=32,
            steps=2,
            base=48,
            device="cpu"
        )
        self.assertTrue(tmp_out.exists())
        from PIL import Image
        with Image.open(tmp_out) as im:
            self.assertEqual(im.size, (32, 32))
        tmp_out.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
