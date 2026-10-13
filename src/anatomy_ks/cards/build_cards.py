"""Yapı kartları (Bölüm 7.7): depolama + üretim.

Üretim LLM ister. LLM yokken yalnızca ontolojiden (Katman 0) türetilen, atıfsız bir
"iskelet kart" yazılır (bölümler null, özet = ad + sistem + bölge). LLM gelince
`scripts/build_cards.py --force` ile gerçek kartlar üretilir.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from anatomy_ks.llm.base import LLMClient, LLMUnavailable
from anatomy_ks.models import CONTENT_TYPES, FAQItem, Structure, StructureCard
from anatomy_ks.registry import Registry
from anatomy_ks.retrieval.prompt import SYSTEM_PROMPT, format_sources, validate_citations
from anatomy_ks.retrieval.search import Hit, SearchResult

log = logging.getLogger(__name__)

CARD_SECTIONS = tuple(c for c in CONTENT_TYPES if c != "gelisim")
SECTION_QUESTIONS = {
    "tanim": "Bu yapı nedir? Genel tanımını ve kısımlarını yaz.",
    "konum_komsuluk": "Bu yapı nerede bulunur ve hangi yapılarla komşudur?",
    "eklem": "Bu yapı hangi yapılarla eklem yapar?",
    "yapisma": "Bu yapıya hangi kaslar yapışır veya bu kasın origo ve insersiyosu nedir?",
    "kanlanma": "Bu yapının kanlanması (arterleri ve venleri) nasıldır?",
    "innervasyon": "Bu yapıyı hangi sinirler innerve eder?",
    "fonksiyon": "Bu yapının görevi / fonksiyonu nedir?",
    "klinik": "Bu yapıyla ilgili kaynaklarda geçen klinik bilgiler nelerdir?",
}
SYSTEM_TR = {
    "skeletal": "iskelet",
    "articular": "eklem",
    "muscular": "kas",
    "nervous": "sinir",
    "cardiovascular": "kalp-damar",
    "lymphatic": "lenfatik",
    "respiratory": "solunum",
    "digestive": "sindirim",
    "urinary": "üriner",
    "reproductive": "üreme",
    "endocrine": "endokrin",
    "integumentary": "örtü",
    "special_senses": "duyu",
}
REGION_TR = {
    "head_neck": "baş-boyun",
    "back": "sırt",
    "thorax": "göğüs",
    "abdomen": "karın",
    "pelvis_perineum": "pelvis-perine",
    "upper_limb": "üst ekstremite",
    "lower_limb": "alt ekstremite",
}
LATERALITY_TR = {
    "left": "sol",
    "right": "sağ",
    "midline": "orta hat",
    "bilateral": "çift (sağ ve sol)",
}


def card_path(cards_dir: Path, structure_id: str) -> Path:
    return cards_dir / f"{structure_id.replace(':', '_')}.json"


class CardStore:
    def __init__(self, cards_dir: Path, registry: Registry) -> None:
        self.dir = cards_dir
        self.reg = registry
        self._cache: dict[str, StructureCard | None] = {}

    def get(self, s: Structure) -> StructureCard | None:
        """Taraflı yapılar generic_id'nin kartını kullanır."""
        sid = s.generic_id or s.id
        if sid not in self._cache:
            p = card_path(self.dir, sid)
            self._cache[sid] = (
                StructureCard.model_validate_json(p.read_text(encoding="utf-8"))
                if p.exists()
                else None
            )
        return self._cache[sid]

    def get_or_skeleton(self, s: Structure) -> StructureCard:
        return self.get(s) or skeleton_card(self.reg, s)

    def invalidate(self) -> None:
        self._cache.clear()


def skeleton_card(reg: Registry, s: Structure) -> StructureCard:
    """LLM'siz, yalnızca ontoloji (Katman 0) bilgisinden kart. Atıf numarası içermez."""
    base = reg.get(s.generic_id) if s.generic_id else s
    base = base or s
    parts = [f"{reg.display_name(base)}"]
    sys_txt = ", ".join(SYSTEM_TR[x] for x in base.system if x in SYSTEM_TR)
    reg_txt = ", ".join(REGION_TR[x] for x in base.region if x in REGION_TR)
    desc = []
    if sys_txt:
        desc.append(f"{sys_txt} sistemine ait")
    if reg_txt:
        desc.append(f"{reg_txt} bölgesinde yer alan")
    summary = parts[0] + (", " + " ve ".join(desc) if desc else "") + " bir yapıdır."
    if s.laterality in LATERALITY_TR:
        summary += f" Taraf: {LATERALITY_TR[s.laterality]}."
    parents = [reg.get(p) for p in base.parent_ids[:3]]
    pnames = [reg.display_name(p) for p in parents if p]
    if pnames:
        summary += " Üst yapı(lar): " + "; ".join(pnames) + "."
    summary += " (Ontoloji bilgisi; ayrıntılı kart henüz üretilmedi.)"
    return StructureCard(
        structure_id=base.id,
        summary=summary,
        sections={c: None for c in CARD_SECTIONS},
        faq=[],
        source_chunks=[],
        generated_at=datetime.now(UTC).isoformat(),
        reviewed=False,
    )


SUMMARY_INSTR = (
    "Aşağıdaki kaynaklara dayanarak bu yapı için 2-4 cümlelik Türkçe bir özet yaz. "
    "Kurallar aynıdır: yalnızca kaynaklar, her cümlede [n] atıfı."
)
FAQ_INSTR = (
    "Aşağıdaki kaynaklara dayanarak bu yapı hakkında öğrencilerin sorabileceği 5-8 soru ve "
    'kısa Türkçe cevap üret. Yalnızca JSON döndür: {"faq": [{"q": "...", "a": "... [1]"}]}. '
    "Cevaplar yalnızca kaynaklara dayanmalı ve [n] atıfı içermeli."
)


def _section_prompt(s: Structure, hits: list[Hit], instruction: str) -> str:
    return (
        f"<secili_yapi>\nTürkçe: {s.name_tr} | Latince: {s.name_la} | İngilizce: {s.name_en} | "
        f"Taraf: {s.laterality}\n</secili_yapi>\n\n"
        f"<kaynaklar>\n{format_sources(hits)}\n</kaynaklar>"
        f"\n\n<gorev>\n{instruction}\n</gorev>"
    )


def _renumber(text: str, mapping: dict[int, int]) -> str:
    def repl(m: re.Match[str]) -> str:
        return f"[{mapping.get(int(m.group(1)), 0)}]"

    return re.sub(r"\[(\d+)\]", repl, text)


def build_card(
    s: Structure,
    llm: LLMClient,
    search_fn: Callable[[str, list[str], int], SearchResult],
) -> StructureCard:
    """Her içerik türü ayrı aranır, bulunan chunk'lar 2-4 cümlelik Türkçe özete çevrilir."""
    sections: dict[str, str | None] = {}
    used: list[str] = []
    all_hits: dict[str, Hit] = {}
    for ct in CARD_SECTIONS:
        res = search_fn(SECTION_QUESTIONS[ct], [ct], 4)
        hits = [h for h in res.hits if ct in (h.payload.get("content_type") or [])]
        if res.stage != 1 or not hits:
            sections[ct] = None
            continue
        text = llm.complete(
            SYSTEM_PROMPT, _section_prompt(s, hits, SECTION_QUESTIONS[ct] + " 2-4 cümle."), 400
        )
        text, _ = validate_citations(text.strip(), len(hits))
        if "bulunmuyor" in text and len(text) < 120:
            sections[ct] = None
            continue
        # bölüm içi [n] → kartın birleşik kaynak listesine yeniden numaralandır
        mapping: dict[int, int] = {}
        for i, h in enumerate(hits, start=1):
            if h.chunk_id not in all_hits:
                all_hits[h.chunk_id] = h
                used.append(h.chunk_id)
            mapping[i] = used.index(h.chunk_id) + 1
        sections[ct] = _renumber(text, mapping)
    hits_all = [all_hits[c] for c in used]
    summary = ""
    if hits_all:
        summary, _ = validate_citations(
            llm.complete(
                SYSTEM_PROMPT, _section_prompt(s, hits_all[:8], SUMMARY_INSTR), 300
            ).strip(),
            min(len(hits_all), 8),
        )
    faq: list[FAQItem] = []
    if hits_all:
        for _ in range(2):
            raw = llm.complete(SYSTEM_PROMPT, _section_prompt(s, hits_all[:8], FAQ_INSTR), 1200)
            m = re.search(r"\{.*\}", raw, re.S)
            try:
                data = json.loads(m.group(0)) if m else {}
                for item in data.get("faq", [])[:10]:
                    a, _ = validate_citations(str(item["a"]), min(len(hits_all), 8))
                    cites = {int(x) for x in re.findall(r"\[(\d+)\]", a)}
                    faq.append(
                        FAQItem(
                            q=str(item["q"]),
                            a=a,
                            source_chunks=[used[i - 1] for i in sorted(cites)],
                        )
                    )
                break
            except (ValueError, KeyError, TypeError):
                faq = []
                continue
    return StructureCard(
        structure_id=s.id,
        summary=summary or "Bu yapı için kaynaklarda özet bilgi bulunamadı.",
        sections=sections,
        faq=faq,
        source_chunks=used,
        generated_at=datetime.now(UTC).isoformat(),
        reviewed=False,
    )


def save_card(card: StructureCard, cards_dir: Path) -> Path:
    cards_dir.mkdir(parents=True, exist_ok=True)
    p = card_path(cards_dir, card.structure_id)
    p.write_text(card.model_dump_json(indent=2), encoding="utf-8")
    return p


__all__ = [
    "CardStore",
    "LLMUnavailable",
    "build_card",
    "card_path",
    "save_card",
    "skeleton_card",
]
