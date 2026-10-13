"""Soru-cevap orkestrasyonu (Bölüm 7.4 akışı). API ve script'ler bu katmanı kullanır."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient

from anatomy_ks.cards.build_cards import CARD_SECTIONS, CardStore
from anatomy_ks.config import Settings
from anatomy_ks.llm.base import LLMClient, LLMUnavailable
from anatomy_ks.logging import log_question
from anatomy_ks.models import AskResponse, SourceRef, Structure, StructureCard
from anatomy_ks.registry import Registry
from anatomy_ks.retrieval.intent import detect_intent, is_card_question
from anatomy_ks.retrieval.prompt import (
    NOT_FOUND_TEXT,
    SYSTEM_PROMPT,
    CitationFilter,
    build_user_message,
    validate_citations,
)
from anatomy_ks.retrieval.search import Hit, SearchResult, search
from anatomy_ks.vectordb import point_id

log = logging.getLogger(__name__)

SECTION_TITLES_TR = {
    "tanim": "Tanım",
    "konum_komsuluk": "Konum ve komşuluk",
    "eklem": "Eklemler",
    "yapisma": "Yapışan kaslar",
    "kanlanma": "Kanlanma",
    "innervasyon": "İnnervasyon",
    "fonksiyon": "Fonksiyon",
    "klinik": "Klinik",
}


@dataclass
class Deps:
    settings: Settings
    registry: Registry
    cards: CardStore
    client: QdrantClient
    embed: Callable[[str], list[float]]
    llm: LLMClient


def to_sources(hits: list[Hit]) -> list[SourceRef]:
    out = []
    for i, h in enumerate(hits, start=1):
        p = h.payload
        out.append(
            SourceRef(
                n=i,
                chunk_id=h.chunk_id,
                text=str(p.get("text", "")),
                source=str(p.get("source", "")),
                section=str(p.get("source_section", "")),
                source_tier=int(p.get("source_tier", 0) or 0),
                license=str(p.get("license", "")),
                score=round(h.final, 4),
                source_url=p.get("source_url"),
            )
        )
    return out


def card_answer_text(card: StructureCard) -> str:
    lines = [card.summary.strip()]
    for key in CARD_SECTIONS:
        val = card.sections.get(key)
        if val:
            lines.append(f"**{SECTION_TITLES_TR.get(key, key)}:** {val.strip()}")
    return "\n\n".join(lines)


def card_sources(deps: Deps, card: StructureCard) -> list[SourceRef]:
    if not card.source_chunks:
        return []
    try:
        pts = deps.client.retrieve(
            deps.settings.qdrant_collection,
            ids=[point_id(c) for c in card.source_chunks],
            with_payload=True,
        )
    except Exception:  # vektör DB yoksa kart yine döner
        return []
    by_id = {str((p.payload or {}).get("chunk_id")): p for p in pts}
    hits = [
        Hit(c, 1.0, 1.0, dict(by_id[c].payload or {})) for c in card.source_chunks if c in by_id
    ]
    return to_sources(hits)


def card_response(deps: Deps, s: Structure, t0: float, intents: list[str]) -> AskResponse:
    card = deps.cards.get_or_skeleton(s)
    return AskResponse(
        structure_id=s.id,
        structure_name=deps.registry.display_name(s),
        answer=card_answer_text(card),
        mode="card",
        intents=intents,
        sources=card_sources(deps, card),
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


def _debug(res: SearchResult) -> dict[str, Any]:
    return {
        "filter_field": res.field,
        "filter_ids": res.ids,
        "stage": res.stage,
        "candidates": res.candidates,
        "scores": [
            {"chunk_id": h.chunk_id, "score": round(h.score, 4), "final": round(h.final, 4)}
            for h in res.hits
        ],
    }


def retrieve(deps: Deps, s: Structure, question: str, k: int) -> SearchResult:
    intents = detect_intent(question)
    return search(
        question,
        s,
        intents,
        k,
        registry=deps.registry,
        client=deps.client,
        embed=deps.embed,
        settings=deps.settings,
    )


def ask(deps: Deps, structure_id: str, question: str | None, k: int) -> AskResponse:
    """JSON cevap. LLMUnavailable / VectorDBUnavailable / StructureNotFound fırlatabilir."""
    t0 = time.perf_counter()
    s = deps.registry.resolve(structure_id)
    log_question(log, question)
    if is_card_question(question):
        return card_response(deps, s, t0, [])
    assert question is not None
    res = retrieve(deps, s, question, k)
    name = deps.registry.display_name(s)
    if not res.hits:
        return AskResponse(
            structure_id=s.id,
            structure_name=name,
            answer=NOT_FOUND_TEXT,
            mode="no_context",
            intents=res.intents,
            sources=[],
            latency_ms=int((time.perf_counter() - t0) * 1000),
            debug=_debug(res),
        )
    card = deps.cards.get(s)
    user = build_user_message(s, card.summary if card else None, res.hits, question)
    answer = deps.llm.complete(SYSTEM_PROMPT, user, deps.settings.llm_max_tokens)
    answer, _ = validate_citations(answer.strip(), len(res.hits))
    return AskResponse(
        structure_id=s.id,
        structure_name=name,
        answer=answer,
        mode="rag",
        intents=res.intents,
        sources=to_sources(res.hits),
        latency_ms=int((time.perf_counter() - t0) * 1000),
        debug=_debug(res),
    )


def ask_stream(
    deps: Deps, structure_id: str, question: str | None, k: int
) -> Iterator[tuple[str, dict[str, Any]]]:
    """SSE olay üreteci: meta → token* → sources → done. Hatalar çağırana fırlatılır."""
    t0 = time.perf_counter()
    s = deps.registry.resolve(structure_id)
    log_question(log, question)
    name = deps.registry.display_name(s)
    if is_card_question(question):
        resp = card_response(deps, s, t0, [])
        yield "meta", {"structure_id": s.id, "structure_name": name, "intents": [], "mode": "card"}
        yield "token", {"t": resp.answer}
        yield "sources", {"sources": [x.model_dump() for x in resp.sources]}
        yield "done", {"latency_ms": int((time.perf_counter() - t0) * 1000)}
        return
    assert question is not None
    res = retrieve(deps, s, question, k)
    mode = "rag" if res.hits else "no_context"
    yield (
        "meta",
        {
            "structure_id": s.id,
            "structure_name": name,
            "intents": res.intents,
            "mode": mode,
            "debug": _debug(res),
        },
    )
    if not res.hits:
        yield "token", {"t": NOT_FOUND_TEXT}
        yield "sources", {"sources": []}
        yield "done", {"latency_ms": int((time.perf_counter() - t0) * 1000)}
        return
    card = deps.cards.get(s)
    user = build_user_message(s, card.summary if card else None, res.hits, question)
    filt = CitationFilter(len(res.hits))
    first_token_ms: int | None = None
    for tok in deps.llm.stream(SYSTEM_PROMPT, user, deps.settings.llm_max_tokens):
        out = filt.feed(tok)
        if out:
            if first_token_ms is None:
                first_token_ms = int((time.perf_counter() - t0) * 1000)
            yield "token", {"t": out}
    rest = filt.flush()
    if rest:
        yield "token", {"t": rest}
    yield "sources", {"sources": [x.model_dump() for x in to_sources(res.hits)]}
    yield (
        "done",
        {"latency_ms": int((time.perf_counter() - t0) * 1000), "first_token_ms": first_token_ms},
    )


__all__ = ["Deps", "LLMUnavailable", "ask", "ask_stream", "card_response"]
