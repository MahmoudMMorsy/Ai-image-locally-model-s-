"""
فحص الحظر والتحقق من التصنيفات قبل قبول صورة تدريب أو برومبت توليد.
يحظر تصوير الذات الإلهية والأنبياء والمرسلين والصحابة في الإسلام.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

_DIR = Path(__file__).resolve().parent


def _load_json(name: str) -> dict:
    with open(_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # strip Arabic diacritics
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_blocked_lexicon(blocked: dict | None = None) -> list[str]:
    blocked = blocked or _load_json("blocked_entities.json")
    terms: list[str] = []
    for _cat, payload in blocked.get("entities", {}).items():
        for lang in ("ar", "en"):
            for t in payload.get(lang, []):
                nt = normalize_text(t)
                if nt and nt not in terms:
                    terms.append(nt)
    # longest first for matching
    terms.sort(key=len, reverse=True)
    return terms


def scan_text_for_blocked(text: str, lexicon: list[str] | None = None) -> list[str]:
    """Return list of matched blocked terms found in text."""
    lexicon = lexicon if lexicon is not None else build_blocked_lexicon()
    norm = normalize_text(text)
    if not norm:
        return []
    hits = []
    for term in lexicon:
        if term in norm:
            hits.append(term)
    return hits


def scan_record_fields(record: dict, lexicon: list[str] | None = None) -> list[str]:
    lexicon = lexicon if lexicon is not None else build_blocked_lexicon()
    chunks = []
    for key in (
        "name",
        "franchise_name",
        "caption_ar",
        "caption_en",
        "file_path",
        "image_id",
    ):
        v = record.get(key)
        if isinstance(v, str):
            chunks.append(v)
    for key in ("name_aliases", "tags_freeform"):
        v = record.get(key)
        if isinstance(v, list):
            chunks.extend(str(x) for x in v)
    return scan_text_for_blocked(" | ".join(chunks), lexicon)


def apply_safety_check(record: dict, lexicon: list[str] | None = None) -> dict:
    """Mutates and returns record with safety_check filled."""
    hits = scan_record_fields(record, lexicon)
    if hits:
        record["safety_check"] = {
            "blocked_scan": "failed",
            "status": "rejected",
            "matched_blocked_terms": hits,
            "reviewer_note": "مطابقة لكيان محظور (مقدسات إسلامية) — مرفوض",
        }
    else:
        record["safety_check"] = {
            "blocked_scan": "passed",
            "status": "allowed",
            "matched_blocked_terms": [],
        }
    return record


def validate_taxonomy_refs(record: dict, taxonomy: dict | None = None) -> list[str]:
    """Return list of validation errors for taxonomy axis IDs."""
    taxonomy = taxonomy or _load_json("character_taxonomy.json")
    errors: list[str] = []
    skip = {
        "version",
        "name",
        "description",
        "locale",
    }
    for axis_name, axis in taxonomy.items():
        if axis_name in skip or not isinstance(axis, dict):
            continue
        if "values" not in axis:
            continue
        allowed = {v["id"] for v in axis["values"]}
        val = record.get(axis_name)
        if val is None:
            continue
        multi = axis.get("multi", False)
        if multi:
            if not isinstance(val, list):
                errors.append(f"{axis_name}: expected list")
                continue
            for item in val:
                if item not in allowed:
                    errors.append(f"{axis_name}: invalid id '{item}'")
        else:
            if val not in allowed:
                errors.append(f"{axis_name}: invalid id '{val}'")
    return errors


def is_prompt_allowed(prompt: str, lexicon: list[str] | None = None) -> tuple[bool, list[str]]:
    hits = scan_text_for_blocked(prompt, lexicon)
    return (len(hits) == 0, hits)


def empty_label_template() -> dict[str, Any]:
    return {
        "image_id": "",
        "file_path": "",
        "media_type": "arcade_sprite",
        "species_or_form": "human",
        "gender_presentation": "ambiguous",
        "age_band": "adult",
        "body_type": "athletic",
        "role_archetype": ["npc_generic"],
        "setting_era": ["unspecified"] if False else ["modern"],
        "culture_region": ["unspecified"],
        "source_domain": ["original_oc"],
        "mythology_family": ["none"],
        "palette_mood": ["retro_limited"],
        "pose_view": "full_body",
        "equipment_tags": ["none"],
        "franchise_hint": "none",
        "name": "",
        "name_aliases": [],
        "franchise_name": "",
        "caption_ar": "",
        "caption_en": "",
        "tags_freeform": [],
        "safety_check": {
            "blocked_scan": "pending",
            "status": "needs_review",
            "matched_blocked_terms": [],
        },
        "split": "train",
        "quality_score": 0.5,
        "is_full_body": True,
        "has_transparency": True,
        "annotator": "unassigned",
    }


if __name__ == "__main__":
    lex = build_blocked_lexicon()
    print(f"blocked terms loaded: {len(lex)}")
    for test in [
        "pixel warrior with sword",
        "صورة النبي محمد",
        "Greek Zeus thunder",
        "علي بن أبي طالب فارس",
        "original elf mage OC",
    ]:
        ok, hits = is_prompt_allowed(test, lex)
        print(f"  [{('OK' if ok else 'BLOCK')}] {test!r} -> {hits}")
