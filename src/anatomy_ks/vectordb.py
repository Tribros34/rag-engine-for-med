"""Qdrant istemcisi ve embedding modeli (indeksleme ve sorgu aynı modeli kullanır — R5)."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from anatomy_ks.config import Settings
from anatomy_ks.models import IndexManifest

KEYWORD_FIELDS = (
    "fma_id",
    "related_ids",
    "ancestors",
    "system",
    "region",
    "content_type",
    "laterality",
    "source",
)
INTEGER_FIELDS = ("source_tier",)

_client_lock = threading.Lock()
_clients: dict[str, QdrantClient] = {}


def get_client(settings: Settings) -> QdrantClient:
    """`local:<yol>` → gömülü yerel Qdrant (Docker'sız geliştirme); aksi halde sunucu URL'i."""
    url = settings.qdrant_url
    with _client_lock:
        if url not in _clients:
            if url.startswith("local:"):
                _clients[url] = QdrantClient(path=url.removeprefix("local:"))
            elif url == ":memory:":
                _clients[url] = QdrantClient(location=":memory:")
            else:
                _clients[url] = QdrantClient(url=url, timeout=10)
        return _clients[url]


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    if not client.collection_exists(name):
        client.create_collection(
            name, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE)
        )
    for f in KEYWORD_FIELDS:
        client.create_payload_index(name, f, field_schema=qm.PayloadSchemaType.KEYWORD)
    for f in INTEGER_FIELDS:
        client.create_payload_index(name, f, field_schema=qm.PayloadSchemaType.INTEGER)


class Embedder:
    """sentence-transformers sarmalayıcısı; normalize_embeddings=True."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        # chunk'lar ≤ ~700 token; dizi uzunluğunu sınırlamak bellek kullanımını sabitler
        self.model.max_seq_length = min(self.model.max_seq_length or 1024, 1024)
        dim = self.model.get_embedding_dimension()
        self.dim = int(dim) if dim else len(self.encode(["x"])[0])

    def encode(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return [list(map(float, v)) for v in vecs]

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


@lru_cache(maxsize=2)
def get_embedder(model_name: str, device: str, batch_size: int) -> Embedder:
    return Embedder(model_name, device, batch_size)


def read_manifest(path: Path) -> IndexManifest | None:
    if not path.exists():
        return None
    return IndexManifest.model_validate_json(path.read_text(encoding="utf-8"))


def write_manifest(
    path: Path, settings: Settings, dim: int, points: int, sources: list[str]
) -> IndexManifest:
    prev = read_manifest(path)
    m = IndexManifest(
        embedding_model=settings.embedding_model,
        dim=dim,
        collection=settings.qdrant_collection,
        points=points,
        created_at=prev.created_at
        if prev and prev.points == points
        else datetime.now(UTC).isoformat(),
        sources=sorted(sources),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(m.model_dump_json(indent=2), encoding="utf-8")
    return m


def payload_dict(obj: Any) -> dict[str, Any]:
    return json.loads(json.dumps(obj, ensure_ascii=False))  # type: ignore[no-any-return]
