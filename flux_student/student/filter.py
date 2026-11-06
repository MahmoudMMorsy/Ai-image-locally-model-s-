"""Content policy and religious figure exclusion filter."""
from __future__ import annotations
import re

# Unicode word boundary for multilingual (English + Arabic)
_WB_LEFT = r"(?<![\w\u0600-\u06FF])"
_WB_RIGHT = r"(?![\w\u0600-\u06FF])"

RAW_PROHIBITED_WORDS = [
    # Divine and religious sacred representations
    "allah", "god", "deity", "divine", "lord",
    "الله", "رب", "إله", "الإله", "الرب",
    # Islamic Prophets and Messengers
    "muhammad", "mohammed", "jesus", "isa", "moses",
    "musa", "abraham", "ibrahim", "noah", "nuh",
    "adam", "yusuf", "joseph", "david", "dawood",
    "solomon", "sulaiman", "prophet", "messenger",
    "محمد", "عيسى", "موسى", "إبراهيم", "نوح", "آدم", "يوسف",
    "داود", "سليمان", "نبي", "رسول",
    # Sahaba (Companions)
    "abu bakr", "umar", "othman", "uthman", "ali",
    "أبو بكر", "عمر", "عثمان", "علي", "الصحابة", "صحابي",
    # Embodying sacred Islamic symbols
    "kaaba", "quran", "koran", "mosque", "mecca", "medina",
    "الكعبة", "كعبة", "القرآن", "مصحف", "مسجد", "مكة", "المدينة المنورة"
]

_COMPILED_PATTERNS = [
    re.compile(f"{_WB_LEFT}{re.escape(w)}{_WB_RIGHT}", re.IGNORECASE)
    for w in RAW_PROHIBITED_WORDS
]

def is_safe_prompt(prompt: str) -> bool:
    """Checks whether prompt complies with strict Islamic religious exclusion policy."""
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(prompt):
            return False
    return True
