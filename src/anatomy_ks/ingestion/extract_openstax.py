"""OpenStax Anatomy & Physiology (1e) Markdown → başlık bölümleri.

Atılanlar: görseller (yalnızca altyazı tutulur, R4), öğrenme hedefleri, alıştırmalar,
sözlük, bölüm sonu sorular, "interactive link" ve "career connection" kutuları, link işaretleri.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from bs4 import BeautifulSoup, Tag

from anatomy_ks.ingestion.chunker import Block, Section

SOURCE = "OpenStax Anatomy and Physiology (1e)"
LICENSE = "CC BY"  # LICENSE.txt ile doğrulanır; sürüm config'den okunur
BOOK_URL = "https://openstax.org/details/books/anatomy-and-physiology"

_SUMMARY_RE = re.compile(r"^(?P<indent>\s*)\d+\.\s+(?:\{:\s*\.(?P<kind>\w+)\}\s*)?(?P<rest>.*)$")
_LINK_RE = re.compile(r"\[(?P<title>[^\]]+)\]\((?P<href>contents/[^)]+\.md)\)")
_IMG_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\((?P<src><[^>]*>|[^\"()]*?)\s*(?:\"(?P<title>.*?)\")?\)(\{:[^}]*\})?"
)
_EMPTY_PAREN_RE = re.compile(r"\((?:\s|see|also|and|or|[,;]|\b[a-h]\b|\[link\])*\)", re.I)
_ATTR_RE = re.compile(r"\{:[^}]*\}")
_REFLINK_RE = re.compile(r"\(\s*\[\\\[link\\\]\]\([^)]*\)\s*\)|\[\\\[link\\\]\]\([^)]*\)")
_SEE_LINK_RE = re.compile(
    r"\((?:[^()\[]{0,40}?)\[\\\[link\\\]\]\([^)]*\)(?:[^()]{0,40}?)\)|\[\\\[link\\\]\]\([^)]*\)",
    re.I,
)
_MDLINK_RE = re.compile(r"\[([^\]]+)\]\[\d+\]|\[([^\]]+)\]\([^)]*\)")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__")
_ITAL_RE = re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])|(?<![\w_])_([^_\n]+)_(?![\w_])")
_REFDEF_RE = re.compile(r"^\[\d+\]:\s+\S+.*$", re.M)
_HEADING_RE = re.compile(r"^(#{2,6})\s+(.*)$")
_WS_RE = re.compile(r"[ \t]+")
_STOP_HEADINGS = (
    "chapter review",
    "review questions",
    "critical thinking questions",
    "glossary",
    "interactive link questions",
    "references",
)
_DROP_NOTE_CLASSES = ("interactive", "career", "um")


@dataclass
class TocSection:
    unit: str
    chapter: str
    chapter_no: int
    section: str
    section_no: int
    path: Path


def parse_summary(book_dir: Path) -> list[TocSection]:
    out: list[TocSection] = []
    unit = chapter = ""
    ch_no = 0
    sec_no = 0
    for line in (book_dir / "SUMMARY.md").read_text(encoding="utf-8").splitlines():
        m = _SUMMARY_RE.match(line)
        if not m:
            continue
        kind = m.group("kind")
        rest = m.group("rest")
        link = _LINK_RE.search(rest)
        title = link.group("title") if link else rest.strip()
        if kind == "part":
            unit = title
        elif kind == "chapter":
            ch_no += 1
            chapter = title
            sec_no = 0
        elif kind is None and link and chapter:
            sec_no += 1
            out.append(
                TocSection(unit, chapter, ch_no, title, sec_no, book_dir / link.group("href"))
            )
    return out


def load_selection(cfg_path: Path) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return data


def select_sections(toc: list[TocSection], cfg: dict[str, Any]) -> list[TocSection]:
    chapters: dict[str, Any] = cfg.get("chapters", {})
    out: list[TocSection] = []
    for s in toc:
        if s.chapter not in chapters:
            continue
        rule = chapters[s.chapter]
        exclude = set((rule or {}).get("exclude_sections", []) or [])
        if s.section in exclude:
            continue
        out.append(s)
    return out


def _clean_inline(text: str) -> str:
    text = _SEE_LINK_RE.sub("", text)
    text = _REFLINK_RE.sub("", text)
    text = _IMG_RE.sub("", text)
    text = _ATTR_RE.sub("", text)
    text = _MDLINK_RE.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _BOLD_RE.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _ITAL_RE.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = text.replace("\\", "").replace("[link]", "")
    text = _EMPTY_PAREN_RE.sub("", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return _WS_RE.sub(" ", text).strip()


def _table_block(tbl: Tag) -> Block:
    rows = tbl.find_all("tr")
    header_cells: list[str] = []
    title = ""
    body_rows: list[str] = []
    for tr in rows:
        ths = [c.get_text(" ", strip=True) for c in tr.find_all("th")]
        tds = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
        if ths and not tds:
            if len(ths) == 1:
                title = ths[0]
            else:
                header_cells = ths
            continue
        cells = tds or ths
        if header_cells and len(header_cells) == len(cells):
            body_rows.append(
                " | ".join(f"{h}: {c}" for h, c in zip(header_cells, cells, strict=True))
            )
        else:
            body_rows.append(" | ".join(cells))
    header = " ".join(x for x in [title, " | ".join(header_cells)] if x)
    return Block("table", header=_clean_inline(header), rows=[_clean_inline(r) for r in body_rows])


def _preprocess(md: str) -> tuple[str, list[Block]]:
    """HTML kısımlarını temizle; tabloları ayrı blok olarak çıkar (yer tutucu bırakarak)."""
    md = re.sub(r"^---\n.*?\n---\n", "", md, flags=re.S)
    md = _REFDEF_RE.sub("", md)
    soup = BeautifulSoup(md, "html.parser")
    for div in soup.find_all("div"):
        if not isinstance(div, Tag) or div.decomposed:
            continue
        dtype = div.get("data-type")
        classes: list[str] = list(div.get("class") or [])
        if (
            dtype in ("exercise", "glossary", "abstract", "problem", "solution")
            or dtype == "note"
            and any(c in _DROP_NOTE_CLASSES for c in classes)
        ):
            div.decompose()
    tables: list[Block] = []
    for i, tbl in enumerate(soup.find_all("table")):
        if not isinstance(tbl, Tag):
            continue
        tables.append(_table_block(tbl))
        tbl.replace_with(f"\n\n@@TABLE{i}@@\n\n")
    # not başlıkları: <div data-type="title">X</div> → "X:" satırı
    for t in soup.find_all(["div", "span"], attrs={"data-type": "title"}):
        if isinstance(t, Tag):
            t.replace_with(f"{t.get_text(' ', strip=True)}: ")
    for media in soup.find_all("span", attrs={"data-type": "media"}):
        if isinstance(media, Tag):
            media.decompose()
    for tag in soup.find_all(["div", "span", "sup", "sub", "em", "strong", "a", "figure"]):
        if isinstance(tag, Tag):
            tag.unwrap()
    for tag in soup.find_all(True):
        if isinstance(tag, Tag):
            tag.unwrap()
    # formatter=None: metin kaçışlanmasın (& → &amp; olmasın)
    return soup.decode(formatter=None), tables  # type: ignore[arg-type]


def extract_section(ts: TocSection, license_str: str) -> list[Section]:
    md = ts.path.read_text(encoding="utf-8")
    text, tables = _preprocess(md)
    source_section = f"{ts.chapter_no}.{ts.section_no} {ts.section}"
    prefix = f"openstax-ch{ts.chapter_no:02d}-s{ts.section_no:02d}"
    sections: list[Section] = []
    base = [ts.chapter, ts.section]
    sub: list[str] = []
    blocks: list[Block] = []
    sub_idx = 0

    def flush() -> None:
        nonlocal blocks, sub_idx
        if blocks:
            sections.append(
                Section(
                    heading_path=base + sub,
                    blocks=blocks,
                    source=SOURCE,
                    source_section=source_section,
                    source_url=BOOK_URL,
                    source_tier=1,
                    license=license_str,
                    chunk_prefix=f"{prefix}-h{sub_idx:02d}",
                )
            )
        blocks = []
        sub_idx += 1

    stop = False
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para or stop:
            continue
        lines = para.splitlines()
        hm = _HEADING_RE.match(lines[0])
        if hm:
            title = _clean_inline(hm.group(2))
            if title.casefold() in _STOP_HEADINGS:
                stop = True
                continue
            flush()
            level = len(hm.group(1)) - 3  # ### → 0, #### → 1
            sub = sub[: max(level, 0)] + [title]
            para = "\n".join(lines[1:]).strip()
            if not para:
                continue
        tm = re.fullmatch(r"@@TABLE(\d+)@@", para)
        if tm:
            tb = tables[int(tm.group(1))]
            if tb.rows:
                blocks.append(tb)
            continue
        for img in _IMG_RE.finditer(para):
            cap = (img.group("title") or "").strip()
            if cap:
                blocks.append(Block("caption", text=_clean_inline(cap)))
        if para.lstrip().startswith(("* ", "- ")):
            items = [_clean_inline(x.lstrip("*- ").strip()) for x in para.splitlines()]
            body = "; ".join(i for i in items if i)
            if body:
                blocks.append(Block("para", text=body.rstrip(";") + "."))
            continue
        body = _clean_inline(para.replace("\n", " "))
        if len(body.split()) >= 3:
            blocks.append(Block("para", text=body))
    flush()
    return sections


def extract_openstax(book_dir: Path, cfg_path: Path) -> list[Section]:
    cfg = load_selection(cfg_path)
    license_str = str(cfg.get("license", LICENSE))
    toc = select_sections(parse_summary(book_dir), cfg)
    out: list[Section] = []
    for ts in toc:
        out.extend(extract_section(ts, license_str))
    return out
