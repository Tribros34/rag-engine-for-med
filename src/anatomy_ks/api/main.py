"""FastAPI uygulaması.

Açılışta registry + ontoloji belleğe yüklenir; index manifest kontrol edilir (R5).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from anatomy_ks.api import errors
from anatomy_ks.api.routes import router
from anatomy_ks.cards.build_cards import CardStore
from anatomy_ks.config import Settings, get_settings
from anatomy_ks.llm.base import get_llm
from anatomy_ks.logging import setup_logging
from anatomy_ks.models import IndexManifest
from anatomy_ks.registry import Registry
from anatomy_ks.service import Deps
from anatomy_ks.vectordb import get_client, get_embedder, read_manifest

log = logging.getLogger(__name__)


class IndexModelMismatch(RuntimeError):
    """INDEX_MODEL_MISMATCH: indeks farklı bir embedding modeliyle oluşturulmuş."""


def check_manifest(settings: Settings) -> IndexManifest | None:
    m = read_manifest(settings.manifest_path)
    if m and m.embedding_model != settings.embedding_model:
        raise IndexModelMismatch(
            f"INDEX_MODEL_MISMATCH: indeks '{m.embedding_model}' ile oluşturulmuş, "
            f"EMBEDDING_MODEL='{settings.embedding_model}'. "
            "Yeniden indeksleyin veya ayarı düzeltin."
        )
    return m


def build_deps(settings: Settings) -> Deps:
    reg = Registry.load(settings.registry_path, settings.group_min_descendants)
    emb = get_embedder(
        settings.embedding_model, settings.embedding_device, settings.embedding_batch_size
    )
    return Deps(
        settings=settings,
        registry=reg,
        cards=CardStore(settings.cards_dir, reg),
        client=get_client(settings),
        embed=emb.encode_one,
        llm=get_llm(settings),
    )


def create_app(settings: Settings | None = None, deps: Deps | None = None) -> FastAPI:
    st = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging(st.log_level)
        app.state.manifest = check_manifest(st)  # uyuşmazlıkta başlamayı reddeder
        app.state.deps = deps or build_deps(st)
        log.info("ready", extra={"extra_data": {"structures": len(app.state.deps.registry)}})
        yield

    app = FastAPI(
        title="Anatomy Knowledge Service",
        version="1.0",
        description="3D/AR anatomi atlası için yapı-bağlamlı, kaynaklı soru-cevap servisi. "
        "Eğitim amaçlıdır; klinik karar için kullanılmaz.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=st.cors_list, allow_methods=["*"], allow_headers=["*"]
    )
    errors.install(app)
    app.include_router(router)
    return app


app = create_app()
