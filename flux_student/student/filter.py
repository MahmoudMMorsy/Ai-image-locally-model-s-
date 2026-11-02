"""Content policy and religious figure exclusion filter."""
from __future__ import annotations
import re

PROHIBITED_TERMS = [
    # Divine and religious sacred representations
    r"\ballah\b", r"\bgod\b", r"\bdeity\b", r"\bdivine\b",
    r"(?<![\w\u0600-\u06FF])الله(?![\w\u0600-\u06FF])",
    r"(?<![\w\u0600-\u06FF])رب(?![\w\u0600-\u06FF])",
    r"(?<![\w\u0600-\u06FF])إله(?![\w\u0600-\u06FF])",
    r"(?<![\w\u0600-\u06FF])الإله(?![\w\u0600-\u06FF])",
    # Islamic Prophets and Messengers
    r"\bmuhammad\b", r"\bmohammed\b", r"\bjesus\b", r"\bisa\b", r"\bmoses\b",
    r"\bmusa\b", r"\babraham\b", r"\bibrahim\b", r"\bnoah\b", r"\bnuh\b",
    r"\badam\b", r"\byusuf\b", r"\bjoseph\b", r"\bdavid\b", r"\bdawood\b",
    r"\bsolomon\b", r"\bsulaiman\b", r"\bprophet\b", r"\bmessenger\b",
    r"(?<![\w\u0600-\u06FF])محمد(?![\w\u0600-\u06FF])",
    r"(?<![\w\u0600-\u06FF])عيسى(?![\w\u0600-\u06FF])",
    r"(?<![\w\u0600-\u06FF])موسى(?![\w\u0600-\u06FF])",
    r"(?<![\w\u0600-\u06FF])إبراهيم(?![\w\u0600-\u06FF])",
    r"(?<![\w\u0600-\u06FF])نوح(?![\w\u0600-\u06FF])",
    r"(?<![\w\u0600-\u06FF])آدم(?![\w\u0600-\u06FF])",
    r"(?<![\w\u0600-\u06FF])يوسف(?![\w\u0600-\u06FF])",
    r"(?<![\w\u0600-\u06FF])داود(?![\w\u0600-\u06FF])",
    r"(?<![\w\u0600-\u06FF])سليمان(?![\w\u0600-\u06FF])",
    r"(?<![\w\u0600-\u06FF])نبي(?![\w\u0600-\u06FF])",
    r"(?<![\w\u0600-\u06FF])رسول(?![\w\u0600-\u06FF])",
    # Sahaba (Companions)
    r"\babu bakr\b", r"\bumar\b", r"\bothman\b", r"\buthman\b", r"\bali\b",
    r"أبو بكر",
    r"(?<![\w\u0600-\u06FF])عمر(?![\w\u0600-\u06FF])",
    r"(?<![\w\u0600-\u06FF])عثمان(?![\w\u0600-\u06FF])",
    r"(?<![\w\u0600-\u06FF])علي(?![\w\u0600-\u06FF])",
    r"الصحابة", r"صحابي",
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
