"""Başlık-sınırlı chunking (Bölüm 7.1).

- Bir chunk iki başlığı karıştırmaz.
- Hedef 200–400 kelime; cümle ortadan kesilmez; ardışık chunk'lar 1–2 cümle örtüşür.
- Her chunk metni "<başlık yolu> : " ön ekiyle başlar (embedding bu haliyle alınır).
- Altyazılar ayrı chunk (is_caption=True).
- Tablolar satır grubuyla bölünür, başlık satırı tekrarlanır.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Literal

TARGET_MIN_WORDS = 200
TARGET_MAX_WORDS = 400
OVERLAP_SENTENCES = 2
OVERLAP_MAX_WORDS = 60

_SENT_RE = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[A-Z0-9\"'(\[])")
_ABBREV_END = re.compile(r"\b(?:Fig|Figs|e\.g|i\.e|cf|vs|approx|No|St|Dr|Mr|al)\.$")


@dataclass
class Block:
    kind: Literal["para", "caption", "table"]
    text: str = ""
    header: str = ""
    rows: list[str] = field(default_factory=list)


@dataclass
class Section:
    """Tek bir başlık altındaki içerik."""

    heading_path: list[str]
    blocks: list[Block]
    source: str
    source_section: str
    source_url: str
    source_tier: Literal[1, 2]
    license: str
    chunk_prefix: str  # ör. "openstax-ch08-s04"


@dataclass
class RawChunk:
    chunk_id: str
    text: str
    body: str
    heading_path: list[str]
    is_caption: bool
    source: str
    source_section: str
    source_url: str
    source_tier: Literal[1, 2]
    license: str
    word_count: int
    text_hash: str


def split_sentences(text: str) -> list[str]:
    parts = _SENT_RE.split(text.strip())
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # "Fig." gibi kısaltmalardan sonra bölünmüşse geri birleştir
        if out and _ABBREV_END.search(out[-1]):
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


def words(text: str) -> int:
    return len(text.split())


def heading_prefix(path: list[str]) -> str:
    return " > ".join(h for h in path if h)


def _hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pack_sentences(sentences: list[str]) -> list[str]:
    """Cümleleri 200–400 kelimelik parçalara, 1–2 cümle örtüşmeyle paketler."""
    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    i = 0
    while i < len(sentences):
        s = sentences[i]
        w = words(s)
        if cur and cur_words + w > TARGET_MAX_WORDS:
            chunks.append(" ".join(cur))
            # örtüşme: son 1–2 cümle (çok uzun değilse)
            overlap: list[str] = []
            ow = 0
            for prev in reversed(cur[-OVERLAP_SENTENCES:]):
                if ow + words(prev) > OVERLAP_MAX_WORDS:
                    break
                overlap.insert(0, prev)
                ow += words(prev)
            cur = overlap
            cur_words = ow
            continue
        cur.append(s)
        cur_words += w
        i += 1
    if cur:
        tail = " ".join(cur)
        # Kısa kuyruk (yalnız örtüşme + az yeni metin) bir öncekine eklenebiliyorsa ekle
        if chunks and words(tail) < TARGET_MIN_WORDS // 2:
            prev = chunks[-1]
            prev_sents = split_sentences(prev)
            new_part = [s for s in split_sentences(tail) if s not in prev_sents]
            if words(prev) + words(" ".join(new_part)) <= TARGET_MAX_WORDS + TARGET_MIN_WORDS // 2:
                chunks[-1] = " ".join([prev, *new_part]).strip()
            else:
                chunks.append(tail)
        else:
            chunks.append(tail)
    return chunks


def _table_chunks(block: Block) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    cur_words = words(block.header)
    for row in block.rows:
        if cur and cur_words + words(row) > TARGET_MAX_WORDS:
            out.append("\n".join([block.header, *cur]).strip())
            cur, cur_words = [], words(block.header)
        cur.append(row)
        cur_words += words(row)
    if cur:
        out.append("\n".join([block.header, *cur]).strip())
    return out


def chunk_section(sec: Section, start_index: int = 0) -> list[RawChunk]:
    """Bir başlık bölümünü chunk'lara böler. Chunk ID'leri deterministiktir (R9)."""
    prefix = heading_prefix(sec.heading_path)
    bodies: list[tuple[str, bool]] = []

    sentences: list[str] = []
    for b in sec.blocks:
        if b.kind == "para":
            sentences.extend(split_sentences(b.text))
        elif b.kind == "caption":
            if words(b.text) >= 4:
                bodies.append((b.text, True))
        elif b.kind == "table":
            if sentences:
                bodies.extend((t, False) for t in _pack_sentences(sentences))
                sentences = []
            bodies.extend((t, False) for t in _table_chunks(b))
    if sentences:
        bodies.extend((t, False) for t in _pack_sentences(sentences))

    # Metin chunk'ları önce, altyazılar sonra — sıra deterministik
    ordered = [b for b in bodies if not b[1]] + [b for b in bodies if b[1]]
    out: list[RawChunk] = []
    for i, (body, is_caption) in enumerate(ordered):
        text = f"{prefix} : {body}" if prefix else body
        out.append(
            RawChunk(
                chunk_id=f"{sec.chunk_prefix}-{start_index + i:04d}",
                text=text,
                body=body,
                heading_path=list(sec.heading_path),
                is_caption=is_caption,
                source=sec.source,
                source_section=sec.source_section,
                source_url=sec.source_url,
                source_tier=sec.source_tier,
                license=sec.license,
                word_count=words(text),
                text_hash=_hash(text),
            )
        )
    return out
