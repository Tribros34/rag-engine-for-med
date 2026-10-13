"""Başlık / metin → FMA ID bağlayıcı (Bölüm 7.2).

1. Başlık yolunun en derin anlamlı başlığı normalize edilip registry adlarında birebir aranır.
2. Yoksa rapidfuzz token_sort_ratio ≥ 90 → kabul; 85–90 → link_review.csv.
3. Yoksa bir üst başlığa çıkılır.
4. Metindeki yapı adları (kelime sınırına duyarlı, en uzun eşleşme) → related_ids.
5. Taraflı yapılar tarafsız (generic) ID'ye bağlanır.
Ek (raporda belirtilir): Başlıklar bağlanamazsa, gövde metninde en az 2 kez geçen baskın yapı
fma_id olarak kullanılır; bölüm (chapter/part) düzeyi başlık en son denenir.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

from anatomy_ks.registry import Registry, normalize_name

FUZZY_ACCEPT = 90.0
FUZZY_REVIEW = 85.0
MAX_NGRAM = 7
MAX_RELATED = 25
MIN_DOMINANT_MENTIONS = 2

_ARTICLES = {"the", "a", "an"}
# Tek başına geçtiğinde yapı adı sayılmayacak kadar genel sözcükler (mention tespiti için).
GENERIC_WORDS = {
    "head",
    "neck",
    "body",
    "trunk",
    "back",
    "face",
    "arm",
    "leg",
    "hand",
    "foot",
    "organ",
    "tissue",
    "bone",
    "muscle",
    "nerve",
    "artery",
    "vein",
    "joint",
    "skin",
    "hair",
    "cell",
    "duct",
    "gland",
    "vessel",
    "membrane",
    "cavity",
    "wall",
    "surface",
    "process",
    "ligament",
    "tendon",
    "fascia",
    "space",
    "region",
    "system",
    "part",
    "border",
    "angle",
    "base",
    "apex",
    "root",
    "crest",
    "line",
    "margin",
    "fossa",
    "canal",
    "septum",
    "lobe",
    "horn",
    "column",
    "plate",
    "ring",
    "floor",
    "roof",
    "capsule",
    "sheath",
    "valve",
    "sinus",
    "cortex",
    "lumen",
    "neuron",
    "fiber",
    "fibre",
    "layer",
    "node",
    "segment",
    "branch",
    "limb",
    "digit",
    "lip",
    "tubercle",
    "tuberosity",
    "spine",
    "groove",
    "notch",
    "facet",
    "condyle",
    "foramen",
    "shaft",
    "mass",
    "arch",
    "pedicle",
    "lamina",
    "ramus",
    "tract",
    "nucleus",
    "bulb",
    "cord",
    "plexus",
    "ganglion",
    "lobule",
    "papilla",
    "follicle",
    "epithelium",
    "mucosa",
    "serosa",
    "skeleton",
    "cartilage",
    "blood",
    "lymph",
    "fat",
    "matrix",
    "orifice",
    "opening",
    "aperture",
    "chamber",
    "tube",
    "flap",
    "fold",
    "pouch",
    "recess",
    "tunic",
    "stroma",
    "parenchyma",
    "capillary",
    "bone element",
    "organism",
    "anatomical structure",
    "multicellular organism",
    "organ part",
    "muscle organ",
    "structure",
    "portion",
    "aspect",
    "side",
    "end",
    "extremity",
    "upper extremity",
    "lower extremity",
    "trochanter",
    "intermedium",
    "atlas",
    "axis",
    "pes",
    "manus",
    "pons",
    "iris",
    "ileum",
    "vagina",
    "cervix",
}
# "head of X", "neck of X" gibi alt yapı başlıkları için üst başlıkla birleştirilen sözcükler
COMPOSABLE = {
    "head",
    "neck",
    "body",
    "shaft",
    "base",
    "apex",
    "border",
    "surface",
    "angle",
    "extremity",
    "upper extremity",
    "lower extremity",
    "tubercle",
    "tuberosity",
    "condyle",
    "spine",
    "process",
    "articulation",
    "articulations",
    "attachment",
    "attachments",
    "structure",
    "relations",
    "nerves",
    "vessels",
    "blood supply",
    "arteries",
    "veins",
    "lymphatics",
    "insertion",
    "origin",
}


def _singular(tok: str) -> str:
    if len(tok) <= 3:
        return tok
    if tok.endswith("ies"):
        return tok[:-3] + "y"
    if tok.endswith(("ches", "shes", "sses", "xes")):
        return tok[:-2]
    if tok.endswith("ae") and len(tok) > 4:
        return tok[:-1]
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is", "ys", "as", "os")):
        return tok[:-1]
    return tok


_TOKEN_MAP = {"extremity": "limb", "fibre": "fiber", "oesophagus": "esophagus"}


def link_tokens(text: str) -> list[str]:
    toks = [_singular(t) for t in normalize_name(text).split() if t not in _ARTICLES]
    return [_TOKEN_MAP.get(t, t) for t in toks]


def link_key(text: str) -> str:
    return " ".join(link_tokens(text))


_PAREN_RE = re.compile(r"\(([^)]*)\)")
_LEAD_RE = re.compile(r"^(?:the|a|an)\s+", re.I)


def heading_variants(h: str, parent: str | None) -> list[str]:
    base = _PAREN_RE.sub(" ", h)
    base = _LEAD_RE.sub("", base).strip(" .,:;—-")
    out: list[str] = []
    if base:
        out.append(base)
    for inner in _PAREN_RE.findall(h):
        for piece in re.split(r"[;,]| or ", inner):
            piece = piece.strip(" .")
            if len(piece) > 3 and not re.search(r"\bfig", piece, re.I):
                out.append(piece)
    # "Interior of the Cerebral Hemispheres" → "Cerebral Hemispheres"
    m = re.match(r"^(?P<x>[\w\s]+?)\s+of\s+(?:the\s+)?(?P<y>.+)$", base, re.I)
    if m and len(m.group("x").split()) <= 4:
        out.append(m.group("y"))
    if parent and link_key(base) in {link_key(c) for c in COMPOSABLE}:
        pcore = _LEAD_RE.sub("", _PAREN_RE.sub(" ", parent)).strip(" .,:;")
        out.insert(0, f"{base} of {pcore}")
    return out


@dataclass
class LinkResult:
    fma_id: str | None
    related_ids: list[str] = field(default_factory=list)
    method: str = "unlinked"
    matched: str = ""
    score: float = 0.0
    review: list[tuple[str, str, float]] = field(default_factory=list)  # (aday metin, id, skor)


class Linker:
    def __init__(self, registry: Registry) -> None:
        self.reg = registry
        best: dict[str, tuple[int, bool, int, str]] = {}
        for s in registry.all():
            if s.deprecated:
                continue
            target = s.generic_id or s.id
            sided = s.generic_id is not None
            prios = [(s.name_en, 0), (s.name_la, 1), (s.name_tr, 1)] + [(x, 2) for x in s.synonyms]
            for name, prio in prios:
                k = link_key(name)
                if not k:
                    continue
                cand = (prio, sided, -registry.descendant_count(s.id), target)
                if k not in best or cand < best[k]:
                    best[k] = cand
        self._index: dict[str, str] = {k: v[3] for k, v in best.items()}
        self._keys = list(self._index)
        self._mention_index = {
            k: v
            for k, v in self._index.items()
            if k not in {link_key(w) for w in GENERIC_WORDS} and len(k) >= 4
        }
        self._max_n = min(MAX_NGRAM, max((len(k.split()) for k in self._mention_index), default=1))
        self._fuzzy_cache: dict[str, tuple[str, float] | None] = {}
        self._generic_keys = {link_key(w) for w in GENERIC_WORDS | COMPOSABLE} | {
            link_key(w) + "s" for w in GENERIC_WORDS | COMPOSABLE
        }

    # ---------- başlık ----------
    def _exact(self, text: str) -> str | None:
        return self._index.get(link_key(text))

    def _fuzzy(self, text: str) -> tuple[str, float] | None:
        k = link_key(text)
        if not k or len(k) < 4:
            return None
        if k in self._fuzzy_cache:
            return self._fuzzy_cache[k]
        res = process.extractOne(
            k, self._keys, scorer=fuzz.token_sort_ratio, score_cutoff=FUZZY_REVIEW
        )
        out = None
        if res:
            cand = str(res[0])
            # OCR gürültüsünde farklı uzunluktaki adlara kaymayı önle
            same_shape = abs(len(cand.split()) - len(k.split())) <= 1 and (
                min(len(cand), len(k)) / max(len(cand), len(k)) >= 0.8
            )
            if same_shape:
                out = (self._index[cand], float(res[1]))
        self._fuzzy_cache[k] = out
        return out

    def link_heading_level(
        self, h: str, parent: str | None, review: list[tuple[str, str, float]]
    ) -> tuple[str, str, float] | None:
        variants = [v for v in heading_variants(h, parent) if link_key(v) not in self._generic_keys]
        for v in variants:
            sid = self._exact(v)
            if sid:
                return sid, v, 100.0
        for v in variants:
            fz = self._fuzzy(v)
            if fz:
                sid, score = fz
                if score >= FUZZY_ACCEPT:
                    return sid, v, score
                review.append((v, sid, score))
        return None

    # ---------- metin ----------
    def mentions(self, text: str) -> Counter[str]:
        toks = link_tokens(text)
        found: Counter[str] = Counter()
        i = 0
        while i < len(toks):
            matched = False
            for n in range(min(self._max_n, len(toks) - i), 0, -1):
                key = " ".join(toks[i : i + n])
                sid = self._mention_index.get(key)
                if sid:
                    found[sid] += 1
                    i += n
                    matched = True
                    break
            if not matched:
                i += 1
        return found

    # ---------- chunk ----------
    def link(self, heading_path: list[str], body: str) -> LinkResult:
        review: list[tuple[str, str, float]] = []
        ment = self.mentions(body)
        res = LinkResult(fma_id=None, review=review)
        # 1–3: en derinden ikinci seviyeye kadar başlıklar
        for lvl in range(len(heading_path) - 1, 0, -1):
            parent = heading_path[lvl - 1] if lvl - 1 >= 1 else None
            hit = self.link_heading_level(heading_path[lvl], parent, review)
            if hit:
                res.fma_id, res.matched, res.score = hit
                res.method = "heading_exact" if hit[2] >= 100 else "heading_fuzzy"
                break
        # ek: baskın mention
        if res.fma_id is None and ment:
            sid, cnt = ment.most_common(1)[0]
            if cnt >= MIN_DOMINANT_MENTIONS:
                res.fma_id, res.method, res.score = sid, "mention", float(cnt)
        # en son: bölüm düzeyi başlık
        if res.fma_id is None and heading_path:
            hit = self.link_heading_level(heading_path[0], None, review)
            if hit:
                res.fma_id, res.matched, res.score = hit
                res.method = "chapter_heading"
        if res.fma_id is None:
            return res
        related = [res.fma_id] + [sid for sid, _ in ment.most_common() if sid != res.fma_id]
        res.related_ids = related[:MAX_RELATED]
        return res
