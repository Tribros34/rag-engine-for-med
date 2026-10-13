"""/v1 endpoint'leri (Bölüm 6.6). Yollar ve alan adları sözleşmedir (R8)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from anatomy_ks.api.errors import APIError, error_body
from anatomy_ks.api.sse import sse
from anatomy_ks.llm.base import LLMUnavailable
from anatomy_ks.models import (
    REGIONS,
    SYSTEMS,
    AskRequest,
    AskResponse,
    HealthResponse,
    Structure,
    StructureCard,
)
from anatomy_ks.registry import StructureNotFound
from anatomy_ks.retrieval.search import VectorDBUnavailable
from anatomy_ks.service import Deps, ask, ask_stream, card_response

log = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")
MAX_QUESTION = 500


def deps(request: Request) -> Deps:
    d: Deps = request.app.state.deps
    return d


def _resolve(d: Deps, sid: str) -> Structure:
    try:
        return d.registry.resolve(sid)
    except StructureNotFound as e:
        raise APIError(404, "STRUCTURE_NOT_FOUND", f"Yapı bulunamadı: {e.args[0]}") from e


def _check_question(q: str | None) -> None:
    if q is not None and len(q) > MAX_QUESTION:
        raise APIError(400, "QUESTION_TOO_LONG", f"Soru en fazla {MAX_QUESTION} karakter olabilir.")


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    d = deps(request)
    try:
        d.client.get_collections()
        q = "ok"
        if not d.client.collection_exists(d.settings.qdrant_collection):
            q = "ok (koleksiyon yok)"
    except Exception as e:  # noqa: BLE001
        q = f"error: {type(e).__name__}"
    llm = "ok" if d.settings.llm_configured else "not_configured"
    manifest = request.app.state.manifest
    return HealthResponse(
        status="ok" if q.startswith("ok") else "degraded",
        qdrant=q,
        llm=llm,
        index_manifest=manifest.model_dump() if manifest else None,
    )


@router.get("/structures", response_model=list[Structure])
def structures(
    request: Request,
    q: str = "",
    system: str | None = Query(default=None),
    region: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=6000),
) -> list[Structure]:
    if system and system not in SYSTEMS:
        raise APIError(
            422, "VALIDATION_ERROR", f"system şunlardan biri olmalı: {', '.join(SYSTEMS)}"
        )
    if region and region not in REGIONS:
        raise APIError(
            422, "VALIDATION_ERROR", f"region şunlardan biri olmalı: {', '.join(REGIONS)}"
        )
    return deps(request).registry.search(q, system, region, limit)


@router.get("/structures/{structure_id}", response_model=Structure)
def structure(request: Request, structure_id: str) -> Structure:
    return _resolve(deps(request), structure_id)


@router.get("/structures/{structure_id}/card", response_model=StructureCard)
def card(request: Request, structure_id: str) -> StructureCard:
    d = deps(request)
    return d.cards.get_or_skeleton(_resolve(d, structure_id))


@router.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
async def ask_json(request: Request, req: AskRequest) -> Any:
    d = deps(request)
    _check_question(req.question)
    s = _resolve(d, req.structure_id)
    try:
        return await run_in_threadpool(ask, d, s.id, req.question, req.max_sources)
    except LLMUnavailable as e:
        fallback = card_response(d, s, 0.0, [])
        fallback.latency_ms = 0
        return JSONResponse(
            error_body(
                "LLM_UNAVAILABLE",
                f"LLM'e ulaşılamıyor ({e}); yapı kartı döndü.",
                fallback=fallback.model_dump(exclude_none=True),
            ),
            status_code=503,
        )
    except VectorDBUnavailable as e:
        raise APIError(503, "VECTOR_DB_UNAVAILABLE", "Vektör veritabanına ulaşılamıyor.") from e


@router.post("/ask/stream")
def ask_sse(request: Request, req: AskRequest) -> StreamingResponse:
    d = deps(request)
    _check_question(req.question)
    s = _resolve(d, req.structure_id)

    def gen() -> Iterator[str]:
        try:
            for event, data in ask_stream(d, s.id, req.question, req.max_sources):
                yield sse(event, data)
        except LLMUnavailable as e:
            fb = card_response(d, s, 0.0, [])
            yield sse(
                "error",
                {
                    "code": "LLM_UNAVAILABLE",
                    "message": f"LLM'e ulaşılamıyor ({e}).",
                    "fallback": fb.model_dump(exclude_none=True),
                },
            )
        except VectorDBUnavailable:
            yield sse(
                "error",
                {"code": "VECTOR_DB_UNAVAILABLE", "message": "Vektör veritabanına ulaşılamıyor."},
            )
        except Exception:  # noqa: BLE001
            log.exception("stream failed")
            yield sse("error", {"code": "INTERNAL_ERROR", "message": "Beklenmeyen hata."})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
