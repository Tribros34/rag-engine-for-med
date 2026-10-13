"""Gray's Anatomy (1918, 20. baskı) Internet Archive OCR metni → başlık bölümleri.

OCR temizliği: sayfa numaraları, sayfa üst başlıkları (running heads), figür etiket artıkları,
dipnotlar atılır; satır sonu tirelemeleri birleştirilir; sayfa sonunda bölünen paragraflar
yeniden birleştirilir. Figür altyazıları ("Fig. 243.— ...") is_caption olarak tutulur.
Başlıklar: bağımsız başlık satırları ("The Femur (Thigh Bone).") ve paragraf içi
"run-in" başlıklar ("The Head (caput femoris).— The head ...").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from anatomy_ks.ingestion.chunker import Block, Section

SOURCE = "Gray's Anatomy of the Human Body (1918)"
LICENSE = "Public domain"
SOURCE_URL = "https://archive.org/details/anatomyofhumanbo00grayrich"

_WS = re.compile(r"[ \t]+")
_PAGE_RE = re.compile(r"^\(?\d{1,4}\)?(\s*\(\d{1,4}\))?$")
_FIG_RE = re.compile(
    r"^(?:Fig|Fio|Pig|FiG|Fia)[.,]?\s*\d+[a-z]?\s*[.,]?\s*[—–-]{1,2}\s*(?P<cap>.+)$"
)
_FIG_LABEL_RE = re.compile(r"^(?:Fig|Fio|Pig|FiG|Fia)[.,]?\s*\d+[a-z]?\.?$")
_RUNIN_RE = re.compile(
    r"^(?P<h>[A-Z][^—–]{2,90}?)\s*[.:]?\s*[\.\-]?\s*(?:—|–|\.\s?-\s|\.—|\.\s—|\.-)\s*(?P<body>\S.*)$"
)
_SMALL_WORDS = {
    "of",
    "the",
    "and",
    "or",
    "in",
    "to",
    "a",
    "an",
    "at",
    "by",
    "for",
    "with",
    "on",
    "from",
    "its",
    "their",
    "into",
    "upon",
    "between",
}
_TERMINAL = (".", ":", ";", "?", "!", '"', "'", ")")


@dataclass
class GraysPart:
    key: str  # OCR'daki üst başlık, ör. "OSTEOLOGY"
    label: str  # ör. "II. Osteology"
    slug: str


def load_parts(cfg_path: Path) -> tuple[list[GraysPart], dict[str, Any]]:
    cfg: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    parts = [
        GraysPart(p["key"], p["label"], p["slug"]) for p in cfg["parts"] if p.get("include", True)
    ]
    return parts, cfg


def _norm(line: str) -> str:
    return _WS.sub(" ", line.replace("­", "")).strip()


def _is_running_head(line: str, part_keys: set[str]) -> bool:
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return True
    if line.rstrip(".").strip() in part_keys and not line.endswith("."):
        return True
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) > 0.9 and len(line.split()) <= 8 and not line.endswith(".")


def _titlecase_ratio(text: str) -> float:
    ws = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", text) if w.lower() not in _SMALL_WORDS]
    if not ws:
        return 0.0
    return sum(1 for w in ws if w[0].isupper()) / len(ws)


def _is_heading(par: str) -> bool:
    if _FIG_RE.match(par) or _FIG_LABEL_RE.match(par):
        return False
    n = len(par.split())
    # Gray's bağımsız başlıkları noktayla biter ("The Femur (Thigh Bone).");
    # figür etiketleri bitmez.
    if n == 0 or n > 12 or not par.endswith("."):
        return False
    if not par[0].isupper() or re.search(r"\d|\b(?:Fig|Fio|Proc|Jour|Anat|Arch|Bd|vol)\b", par):
        return False
    stripped = re.sub(r"\([^)]*\)", "", par)
    return _titlecase_ratio(stripped) >= 0.75 and len(re.findall(r"[a-z]", par)) >= 3


def _join_lines(lines: list[str]) -> str:
    out = ""
    for ln in lines:
        if not out:
            out = ln
        elif out.endswith("-") and ln[:1].islower():
            out = out[:-1] + ln
        elif out.endswith("- ") and ln[:1].islower():
            out = out[:-2] + ln
        else:
            out = f"{out} {ln}"
    return out


def _clean_text(text: str) -> str:
    text = text.replace("■", "").replace("{", "(").replace("}", ")")
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"[\^|]", "", text)
    text = re.sub(r"(?<=\w)'\s*(\d)?(?=\s)", "", text)  # dipnot işaretleri (')
    return _WS.sub(" ", text).strip()


def _part_ranges(
    lines: list[str], parts: list[GraysPart], end_key: str
) -> list[tuple[GraysPart, int, int]]:
    norm = [_norm(x) for x in lines]
    starts: dict[str, int] = {}
    for p in parts:
        titled = [i for i, x in enumerate(norm) if x == f"{p.key}."]
        heads = [i for i, x in enumerate(norm) if x == p.key]
        if titled and (len(titled) > 1 or not heads or titled[-1] > heads[0] - 200):
            starts[p.key] = titled[-1]
        elif heads:
            starts[p.key] = heads[0]
    end_titled = [i for i, x in enumerate(norm) if x == f"{end_key}."]
    end = end_titled[-1] if end_titled else len(lines)
    ordered = sorted((starts[p.key], p) for p in parts if p.key in starts)
    out: list[tuple[GraysPart, int, int]] = []
    for idx, (s, p) in enumerate(ordered):
        e = ordered[idx + 1][0] if idx + 1 < len(ordered) else end
        out.append((p, s + 1, e))
    return out


def _paragraphs(lines: list[str], part_keys: set[str]) -> list[str]:
    paras: list[str] = []
    cur: list[str] = []
    for raw in lines:
        ln = _norm(raw)
        if not ln:
            if cur:
                paras.append(_join_lines(cur))
                cur = []
            continue
        if _PAGE_RE.match(ln) or _is_running_head(ln, part_keys) or len(ln) <= 2:
            continue
        cur.append(ln)
    if cur:
        paras.append(_join_lines(cur))
    # sayfa sonunda bölünmüş paragrafları birleştir
    merged: list[str] = []
    for p in paras:
        if (
            merged
            and not merged[-1].endswith(_TERMINAL)
            and p[:1].islower()
            and not _FIG_RE.match(merged[-1])
        ):
            merged[-1] = _join_lines([merged[-1], p])
        else:
            merged.append(p)
    return merged


def extract_part(lines: list[str], part: GraysPart, part_keys: set[str]) -> list[Section]:
    sections: list[Section] = []
    h1 = ""
    h2 = ""
    blocks: list[Block] = []
    idx = 0

    def flush() -> None:
        nonlocal blocks, idx
        if blocks:
            path = [part.label] + [h for h in (h1, h2) if h]
            sections.append(
                Section(
                    heading_path=path,
                    blocks=blocks,
                    source=SOURCE,
                    source_section=" > ".join([part.label] + ([h1] if h1 else [])),
                    source_url=SOURCE_URL,
                    source_tier=2,
                    license=LICENSE,
                    chunk_prefix=f"grays-{part.slug}-{idx:04d}",
                )
            )
            idx += 1
        blocks = []

    for par in _paragraphs(lines, part_keys):
        par = _clean_text(par)
        if not par:
            continue
        fig = _FIG_RE.match(par)
        if fig:
            cap = fig.group("cap").strip()
            if len(cap.split()) >= 4:
                blocks.append(Block("caption", text=cap))
            continue
        if _FIG_LABEL_RE.match(par):
            continue
        if par[0] in "'\"*†‡" and len(par.split()) < 40:
            continue  # dipnot
        if _is_heading(par):
            flush()
            h1 = par.rstrip(".").strip()
            h2 = ""
            continue
        m = _RUNIN_RE.match(par)
        if (
            m
            and len(m.group("h").split()) <= 10
            and not re.search(r"\d", m.group("h"))
            and _titlecase_ratio(re.sub(r"\([^)]*\)|\{[^}]*\}", "", m.group("h"))) >= 0.6
        ):
            flush()
            h2 = m.group("h").strip().rstrip(".").strip()
            par = m.group("body").strip()
        n = len(par.split())
        if n < 8 and not par.endswith(_TERMINAL):
            continue  # figür etiketi artığı
        if n >= 4:
            blocks.append(Block("para", text=par))
    flush()
    return sections


def extract_grays(txt_path: Path, cfg_path: Path) -> list[Section]:
    parts, cfg = load_parts(cfg_path)
    all_keys = {p["key"] for p in cfg["parts"]}
    lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[Section] = []
    all_parts = [GraysPart(p["key"], p["label"], p["slug"]) for p in cfg["parts"]]
    included = {p.key for p in parts}
    for part, s, e in _part_ranges(lines, all_parts, str(cfg.get("end_key", "INDEX"))):
        if part.key not in included:
            continue
        out.extend(extract_part(lines[s:e], part, all_keys))
    return out
