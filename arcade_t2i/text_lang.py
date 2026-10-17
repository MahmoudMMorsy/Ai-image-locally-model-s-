"""Text processing and Arabic normalization for bilingual arcade T2I."""
import re
import json
import os
import torch

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC_RESHAPER = True
except ImportError:
    HAS_ARABIC_RESHAPER = False

def normalize_arabic(text: str) -> str:
    """Normalize Arabic characters (Alef, Yaa, Ta Marbouta, remove diacritics)."""
    if not text:
        return ""
    text = re.sub(r'[\u064B-\u0652]', '', text)
    text = re.sub(r'[\u0622\u0623\u0625]', '\u0627', text)
    text = re.sub(r'\u0649', '\u064A', text)
    text = re.sub(r'\u0629', '\u0647', text)
    return text.strip().lower()

class TextProcessor:
    def __init__(self, vocab_path: str = "arcade_t2i/bilingual_vocab.json"):
        if not os.path.exists(vocab_path):
            vocab_path = os.path.join(os.path.dirname(__file__), "bilingual_vocab.json")

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.unk_id = self.vocab.get("<unk>", 1)
        self.pad_id = self.vocab.get("<pad>", 0)

    def tokenize(self, text: str) -> list[int]:
        text_norm = normalize_arabic(text)
        words = re.findall(r'[\w]+', text_norm, re.UNICODE)

        token_ids = []
        for w in words:
            if w in self.vocab:
                token_ids.append(self.vocab[w])
            else:
                matched = False
                for k, v in self.vocab.items():
                    if k in w or w in k:
                        token_ids.append(v)
                        matched = True
                        break
                if not matched:
                    token_ids.append(self.unk_id)

        if not token_ids:
            token_ids = [self.unk_id]

        return token_ids

    def encode(self, text: str, max_len: int = 16) -> torch.Tensor:
        ids = self.tokenize(text)
        if len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return torch.tensor(ids, dtype=torch.long)

    def reshape_arabic_for_display(self, text: str) -> str:
        if HAS_ARABIC_RESHAPER:
            reshaped = arabic_reshaper.reshape(text)
            return get_display(reshaped)
        return text

if __name__ == "__main__":
    tp = TextProcessor()
    samples = ["فارس", "knight", "ساحر", "mage", "روبوت", "robot", "قطة", "cat", "تنين", "dragon"]
    print("Testing tokenization:")
    for s in samples:
        toks = tp.tokenize(s)
        enc = tp.encode(s)
        print(f"  Prompt: '{s}' -> Tokens: {toks} -> Tensor shape: {enc.shape}")
