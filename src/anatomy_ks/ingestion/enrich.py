"""Zenginleştirme (Bölüm 7.3): ancestors/system/region/laterality + content_type.

content_type her chunk için bir kez sınıflandırılır; önbellek anahtarı chunk_id + text_hash.
LLM yapılandırılmışsa LLM (JSON çıktısı, sözlük dışı değerler atılır); değilse anahtar kelime
tabanlı yedek sınıflandırıcı kullanılır ve önbellekte `method: "heuristic"` olarak işaretlenir.
LLM sonradan yapılandırılınca yalnızca heuristic kayıtlar yeniden sınıflandırılır.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from anatomy_ks.ingestion.chunker import RawChunk
from anatomy_ks.ingestion.linker import LinkResult
from anatomy_ks.llm.base import LLMClient, LLMUnavailable
from anatomy_ks.models import CONTENT_TYPES, SYSTEMS, ChunkPayload
from anatomy_ks.registry import Registry

log = logging.getLogger(__name__)

HEURISTIC_RULES: dict[str, list[str]] = {
    "konum_komsuluk": [
        r"\blies?\b",
        r"\blocated\b",
        r"\bsituated\b",
        r"\brelations?\b",
        r"\banterior to\b",
        r"\bposterior to\b",
        r"\bmedial to\b",
        r"\blateral to\b",
        r"\bsuperior to\b",
        r"\binferior to\b",
        r"\badjacent\b",
        r"\bbounded\b",
        r"\bin front of\b",
        r"\bbehind\b",
        r"\bdeep to\b",
        r"\bsuperficial to\b",
        r"\bborders?\b",
    ],
    "yapisma": [
        r"\borigin\b",
        r"\binsertion\b",
        r"\binserted\b",
        r"\barises?\b",
        r"\boriginates?\b",
        r"\battach(?:ed|es|ment)\b",
        r"\bgives? attachment\b",
    ],
    "eklem": [r"\barticulat\w*", r"\bjoints?\b", r"\bligaments?\b", r"\bsynovial\b"],
    "kanlanma": [
        r"\barter(?:y|ies|ial)\b",
        r"\bveins?\b",
        r"\bvenous\b",
        r"\bblood supply\b",
        r"\bsupplied with blood\b",
        r"\bdrain(?:s|ed|age)?\b",
        r"\bcapillar\w*",
        r"\bvascular\b",
    ],
    "innervasyon": [
        r"\bnerves?\b",
        r"\binnervat\w*",
        r"\bnerve supply\b",
        r"\bplexus\b",
        r"\bganglion\b",
    ],
    "fonksiyon": [
        r"\bfunctions?\b",
        r"\bactions?\b",
        r"\bflex(?:es|ion)\b",
        r"\bextend(?:s)?\b",
        r"\bextension\b",
        r"\babduct\w*",
        r"\badduct\w*",
        r"\brotat\w*",
        r"\bcontract\w*",
        r"\bmovements?\b",
        r"\bresponsible for\b",
        r"\bserves? to\b",
        r"\bsecret\w*",
    ],
    "klinik": [
        r"\bfractures?\b",
        r"\binjur\w*",
        r"\bdiseases?\b",
        r"\bdisorders?\b",
        r"\bclinical\b",
        r"\bsyndrome\b",
        r"\bpain\b",
        r"\bdamage\w*",
        r"\bsurg\w*",
        r"\binflammation\b",
        r"\bcancer\b",
        r"\b\w+itis\b",
        r"\bdislocat\w*",
    ],
    "gelisim": [
        r"\bdevelop\w*",
        r"\bembryo\w*",
        r"\bossif\w*",
        r"\bfetal\b",
        r"\bfetus\b",
        r"\bprimordi\w*",
        r"\bweek of\b",
        r"\bcentres? of ossification\b",
    ],
}
_COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in HEURISTIC_RULES.items()}

CLASSIFY_SYSTEM = (
    "You label anatomy textbook passages. Return ONLY a JSON object of the form "
    '{"content_type": [..]} using 1-3 labels from this exact list: '
    + ", ".join(CONTENT_TYPES)
    + ". Meanings: tanim=definition/general description; konum_komsuluk=location and "
    "relations; yapisma=muscle origin/insertion/attachments; eklem=articulations/joints/"
    "ligaments; kanlanma=blood supply/drainage; innervasyon=nerve supply; fonksiyon=function/"
    "action/movement; klinik=clinical/injury/disease; gelisim=development/ossification/embryology."
)


def heuristic_content_type(text: str) -> list[str]:
    scores: dict[str, int] = {}
    for label, pats in _COMPILED.items():
        n = sum(len(p.findall(text)) for p in pats)
        if n:
            scores[label] = n
    picked = [k for k, v in sorted(scores.items(), key=lambda kv: -kv[1]) if v >= 2][:3]
    if not picked:
        picked = ["tanim"]
    elif len(text.split()) > 120 and "tanim" not in picked and len(picked) < 3:
        # uzun tarif paragrafları genelde tanım da içerir
        picked.append("tanim")
    return [c for c in CONTENT_TYPES if c in picked]


def parse_content_type(raw: str) -> list[str]:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no json")
    data = json.loads(m.group(0))
    vals = data.get("content_type", [])
    if not isinstance(vals, list):
        raise ValueError("bad type")
    return [c for c in CONTENT_TYPES if c in {str(v).strip() for v in vals}]


class ContentTypeClassifier:
    def __init__(self, cache_path: Path, llm: LLMClient | None) -> None:
        self.cache_path = cache_path
        self.llm = llm
        self.cache: dict[str, dict[str, Any]] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec
        self.stats = {"cache": 0, "llm": 0, "heuristic": 0, "llm_failed": 0}

    @staticmethod
    def key(chunk_id: str, text_hash: str) -> str:
        return f"{chunk_id}|{text_hash}"

    def classify(self, chunk_id: str, text_hash: str, text: str) -> list[str]:
        k = self.key(chunk_id, text_hash)
        rec = self.cache.get(k)
        if rec and (rec["method"] == "llm" or self.llm is None):
            self.stats["cache"] += 1
            return list(rec["content_type"])
        labels: list[str] | None = None
        method = "heuristic"
        if self.llm is not None:
            for _ in range(2):  # bozuk JSON → bir kez yeniden dene
                try:
                    labels = parse_content_type(self.llm.complete(CLASSIFY_SYSTEM, text[:6000], 60))
                    method = "llm"
                    break
                except LLMUnavailable:
                    self.llm = None  # erişilemiyorsa bu çalıştırmada heuristic'e düş
                    break
                except (ValueError, json.JSONDecodeError):
                    continue
            if method != "llm":
                self.stats["llm_failed"] += 1
        if labels is None:
            labels = heuristic_content_type(text)
        if not labels:
            labels = ["tanim"]
        self.stats[method] += 1
        self.cache[k] = {"key": k, "content_type": labels, "method": method}
        return labels

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as f:
            for k in sorted(self.cache):
                f.write(json.dumps(self.cache[k], ensure_ascii=False) + "\n")


def enrich(
    chunk: RawChunk, link: LinkResult, reg: Registry, classifier: ContentTypeClassifier
) -> ChunkPayload:
    assert link.fma_id is not None
    s = reg.resolve(link.fma_id)
    related = [link.fma_id] + [r for r in link.related_ids if r != link.fma_id]
    systems = set(s.system)
    for rid in related:
        rs = reg.get(rid)
        if rs:
            systems.update(rs.system)
    return ChunkPayload(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        heading_path=chunk.heading_path,
        fma_id=link.fma_id,
        related_ids=related,
        ancestors=list(s.ancestors),
        system=[x for x in SYSTEMS if x in systems],
        region=list(s.region),
        laterality=s.laterality,
        content_type=classifier.classify(chunk.chunk_id, chunk.text_hash, chunk.text),
        is_caption=chunk.is_caption,
        source=chunk.source,
        source_section=chunk.source_section,
        source_url=chunk.source_url,
        source_tier=chunk.source_tier,
        license=chunk.license,
        lang="en",
        token_count=round(chunk.word_count * 1.35),
        text_hash=chunk.text_hash,
    )
