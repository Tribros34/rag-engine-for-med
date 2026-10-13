"""Niyet tespiti (Bölüm 7.5). İmza sabittir: detect_intent(question) -> list[str]."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from anatomy_ks.config import get_settings
from anatomy_ks.models import CONTENT_TYPES

_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})


def _fold(text: str) -> str:
    return f" {text.translate(_TR_LOWER).casefold().strip()} "


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    kw = {k: [str(x).casefold() for x in v] for k, v in data.get("keywords", {}).items()}
    amb = {str(k).casefold(): list(v) for k, v in data.get("ambiguous", {}).items()}
    cards = [str(x).casefold() for x in data.get("card_patterns", [])]
    for k in [*kw, *(i for v in amb.values() for i in v)]:
        if k not in CONTENT_TYPES:
            raise ValueError(f"intents.yaml: sözlük dışı niyet {k}")
    return {"keywords": kw, "ambiguous": amb, "card_patterns": cards}


def _cfg() -> dict[str, Any]:
    return _load(str(get_settings().config_dir / "intents.yaml"))


def detect_intent(question: str) -> list[str]:
    q = _fold(question)
    cfg = _cfg()
    found = {i for i, kws in cfg["keywords"].items() if any(w in q for w in kws)}
    for kw, intents in cfg["ambiguous"].items():
        if kw in q:
            found |= set(intents)
    return [c for c in CONTENT_TYPES if c in found]


def is_card_question(question: str | None) -> bool:
    """Soru boşsa veya "bu nedir / anlat" gibi genel bir tanım sorusuysa True."""
    if not question or not question.strip():
        return True
    q = re.sub(r"[^\w\s]", " ", _fold(question))
    q = re.sub(r"\s+", " ", q).strip()
    if len(q.split()) > 4:
        return False
    return any(
        q == p or q.endswith(f" {p}") or q.startswith(f"{p} ") for p in _cfg()["card_patterns"]
    )
