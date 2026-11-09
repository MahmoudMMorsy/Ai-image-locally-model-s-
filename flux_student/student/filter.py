"""Content policy and religious figure exclusion filter."""
from __future__ import annotations
import re

# Use Unicode boundary checks (?<![\w\u0600-\u06FF]) and (?![\w\u0600-\u06FF]) for Arabic and English terms
# to prevent false positives (e.g. 'محارب' containing 'رب', 'مدرب', etc.)
_B_LEFT = r"(?<![\w\u0600-\u06FF])"
_B_RIGHT = r"(?![\w\u0600-\u06FF])"

PROHIBITED_TERMS = [
    # Divine and religious sacred representations
    r"allah", r"god", r"deity", r"divine", r"lord",
    r"الله", r"رب", r"إله", r"الإله",
    # Islamic Prophets and Messengers
    r"muhammad", r"mohammed", r"jesus", r"isa", r"moses",
    r"musa", r"abraham", r"ibrahim", r"noah", r"nuh",
    r"adam", r"yusuf", r"joseph", r"david", r"dawood",
    r"solomon", r"sulaiman", r"prophet", r"messenger",
    r"محمد", r"عيسى", r"موسى", r"إبراهيم", r"نوح", r"آدم", r"يوسف",
    r"داود", r"سليمان", r"نبي", r"رسول",
    # Sahaba (Companions)
    r"abu bakr", r"umar", r"othman", r"uthman", r"ali",
    r"أبو بكر", r"عمر", r"عثمان", r"علي", r"الصحابة", r"صحابي",
    # Embodying sacred Islamic symbols
    r"kaaba", r"quran", r"koran", r"mosque", r"mecca", r"medina",
    r"الكعبة", r"كعبة", r"القرآن", r"مصحف", r"مسجد", r"مكة", r"المدينة المنورة"
]

_COMPILED_PATTERNS = [
    re.compile(_B_LEFT + re.escape(p) + _B_RIGHT, re.IGNORECASE)
    for p in PROHIBITED_TERMS
]

def is_safe_prompt(prompt: str) -> bool:
    """Checks whether prompt complies with strict Islamic religious exclusion policy."""
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(prompt):
            return False
    return True
