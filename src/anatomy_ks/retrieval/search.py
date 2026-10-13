"""Filtreli iki aşamalı arama (Bölüm 7.4). Filtresiz arama YAPILMAZ."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from anatomy_ks.config import Settings
from anatomy_ks.models import Structure
from anatomy_ks.registry import Registry


@dataclass
class Hit:
    chunk_id: str
    score: float
    final: float
    payload: dict[str, Any]


@dataclass
class SearchResult:
    hits: list[Hit]
    ids: list[str]
    field: str
    stage: int
    intents: list[str]
    candidates: int = 0
    debug: dict[str, Any] = dc_field(default_factory=dict)


class VectorDBUnavailable(RuntimeError):
    pass


def _query(
    client: QdrantClient, coll: str, vec: list[float], flt: qm.Filter, limit: int
) -> list[qm.ScoredPoint]:
    try:
        return client.query_points(
            collection_name=coll, query=vec, query_filter=flt, limit=limit, with_payload=True
        ).points
    except Exception as e:  # ağ / sunucu hataları
        raise VectorDBUnavailable(str(e)) from e


def rank(
    points: list[qm.ScoredPoint], tier1_boost: float, k: int, max_per_section: int
) -> list[Hit]:
    hits = []
    for p in points:
        pl = dict(p.payload or {})
        final = float(p.score) + (tier1_boost if pl.get("source_tier") == 1 else 0.0)
        hits.append(Hit(str(pl.get("chunk_id", p.id)), float(p.score), final, pl))
    hits.sort(key=lambda h: (-h.final, h.chunk_id))
    out: list[Hit] = []
    per_sec: dict[str, int] = {}
    for h in hits:
        sec = f"{h.payload.get('source')}|{h.payload.get('source_section')}"
        if per_sec.get(sec, 0) >= max_per_section:
            continue
        per_sec[sec] = per_sec.get(sec, 0) + 1
        out.append(h)
        if len(out) >= k:
            break
    return out


def search(
    question: str,
    structure: Structure,
    intents: list[str],
    k: int,
    *,
    registry: Registry,
    client: QdrantClient,
    embed: Callable[[str], list[float]],
    settings: Settings,
) -> SearchResult:
    ids = registry.search_ids(structure.id, settings.max_neighbors, settings.neighbor_depth)
    fld = registry.filter_field(structure)
    if fld == "ancestors":
        # bölge/sistem seçimi: atalarında bu düğüm bulunan tüm chunk'lar
        ids = [structure.id] + ([structure.generic_id] if structure.generic_id else [])
    base = qm.FieldCondition(key=fld, match=qm.MatchAny(any=ids))
    vec = embed(question)
    limit = 2 * k
    stage = 1
    points: list[qm.ScoredPoint] = []
    if intents:
        f1 = qm.Filter(
            must=[base, qm.FieldCondition(key="content_type", match=qm.MatchAny(any=intents))]
        )
        points = _query(client, settings.qdrant_collection, vec, f1, limit)
    if len(points) < settings.min_hits:
        stage = 2
        points = _query(client, settings.qdrant_collection, vec, qm.Filter(must=[base]), limit)
    hits = rank(points, settings.tier1_boost, k, settings.max_per_section)
    return SearchResult(hits, ids, fld, stage, intents, len(points))
