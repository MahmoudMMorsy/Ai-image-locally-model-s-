"""Content policy and religious figure exclusion filter."""
from __future__ import annotations
import re

# Unicode-aware word boundary for Arabic and alphanumeric characters
# Matches only whole words in Arabic/English to prevent false positives (e.g. محارب, بربرية, تربية)
_WB_LEFT = r"(?<![\w\u0600-\u06FF])"
_WB_RIGHT = r"(?![\w\u0600-\u06FF])"

def _wb(term: str) -> str:
    return rf"{_WB_LEFT}{re.escape(term)}{_WB_RIGHT}"

PROHIBITED_TERMS = [
    # Divine and religious sacred representations
    r"\ballah\b", r"\bgod\b", r"\bdeity\b", r"\bdivine\b", r"\blord\b",
    _wb("الله"), _wb("رب"), _wb("إله"), _wb("الإله"),
    # Islamic Prophets and Messengers
    r"\bmuhammad\b", r"\bmohammed\b", r"\bjesus\b", r"\bisa\b", r"\bmoses\b",
    r"\bmusa\b", r"\babraham\b", r"\bibrahim\b", r"\bnoah\b", r"\bnuh\b",
    r"\badam\b", r"\byusuf\b", r"\bjoseph\b", r"\bdavid\b", r"\bdawood\b",
    r"\bsolomon\b", r"\bsulaiman\b", r"\bprophet\b", r"\bmessenger\b",
    _wb("محمد"), _wb("عيسى"), _wb("موسى"), _wb("إبراهيم"), _wb("نوح"), _wb("آدم"), _wb("يوسف"),
    _wb("داود"), _wb("سليمان"), _wb("نبي"), _wb("رسول"),
    # Sahaba (Companions)
    r"\babu bakr\b", r"\bumar\b", r"\bothman\b", r"\buthman\b", r"\bali\b",
    _wb("أبو بكر"), _wb("عمر"), _wb("عثمان"), _wb("علي"), _wb("الصحابة"), _wb("صحابي"),
    # Embodying sacred Islamic symbols
    r"\bkaaba\b", r"\bquran\b", r"\bkoran\b", r"\bmosque\b", r"\bmecca\b", r"\bmedina\b",
    _wb("الكعبة"), _wb("كعبة"), _wb("القرآن"), _wb("قرآن"), _wb("مصحف"), _wb("المصحف"), _wb("مسجد"), _wb("المسجد"), _wb("مكة"), _wb("المدينة المنورة")
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PROHIBITED_TERMS]

def is_safe_prompt(prompt: str) -> bool:
    """Checks whether prompt complies with strict Islamic religious exclusion policy."""
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(prompt):
            return False
    return True
