"""Content policy and religious figure exclusion filter."""
from __future__ import annotations
import re

PROHIBITED_TERMS = [
    # Divine and religious sacred representations
    r"\ballah\b", r"\bgod\b", r"\bdeity\b", r"\bdivine\b", r"\blord\b",
    r"الله", r"رب", r"إله", r"الإله",
    # Islamic Prophets and Messengers
    r"\bmuhammad\b", r"\bmohammed\b", r"\bjesus\b", r"\bisa\b", r"\bmoses\b",
    r"\bmusa\b", r"\babraham\b", r"\bibrahim\b", r"\bnoah\b", r"\bnuh\b",
    r"\badam\b", r"\byusuf\b", r"\bjoseph\b", r"\bdavid\b", r"\bdawood\b",
    r"\bsolomon\b", r"\bsulaiman\b", r"\bprophet\b", r"\bmessenger\b",
    r"محمد", r"عيسى", r"موسى", r"إبراهيم", r"نوح", r"آدم", r"يوسف",
    r"داود", r"سليمان", r"نبي", r"رسول",
    # Sahaba (Companions)
    r"\babu bakr\b", r"\bumar\b", r"\bothman\b", r"\buthman\b", r"\bali\b",
    r"أبو بكر", r"عمر", r"عثمان", r"علي", r"الصحابة", r"صحابي",
    # Embodying sacred Islamic symbols
    r"\bkaaba\b", r"\bquran\b", r"\bkoran\b", r"\bmosque\b", r"\bmecca\b", r"\bmedina\b",
    r"الكعبة", r"كعبة", r"القرآن", r"مصحف", r"مسجد", r"مكة", r"المدينة المنورة"
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PROHIBITED_TERMS]

def is_safe_prompt(prompt: str) -> bool:
    """Checks whether prompt complies with strict Islamic religious exclusion policy."""
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(prompt):
            return False
    return True
