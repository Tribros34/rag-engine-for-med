"""Embedding + Qdrant upsert (Faz 3). Idempotent: point ID = uuid5(NAMESPACE_URL, chunk_id) (R9).

Metni değişmemiş (aynı text_hash) point'ler yeniden embed edilmez; yalnızca payload güncellenir.
Artık var olmayan chunk'ların point'leri silinir, böylece ikinci çalıştırmada sayı değişmez.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from anatomy_ks.config import Settings
from anatomy_ks.models import ChunkPayload, IndexManifest
from anatomy_ks.vectordb import (
    Embedder,
    ensure_collection,
    point_id,
    write_manifest,
)

log = logging.getLogger(__name__)
BATCH = 64


def _existing_hashes(client: QdrantClient, collection: str) -> dict[str, str]:
    out: dict[str, str] = {}
    offset = None
    while True:
        pts, offset = client.scroll(
            collection, limit=512, offset=offset, with_payload=["text_hash"], with_vectors=False
        )
        for p in pts:
            out[str(p.id)] = str((p.payload or {}).get("text_hash", ""))
        if offset is None:
            break
    return out


def index_chunks(
    payloads: Sequence[ChunkPayload],
    settings: Settings,
    client: QdrantClient,
    embedder: Embedder,
) -> tuple[IndexManifest, dict[str, int]]:
    coll = settings.qdrant_collection
    ensure_collection(client, coll, embedder.dim)
    existing = _existing_hashes(client, coll)
    wanted = {point_id(p.chunk_id): p for p in payloads}
    stats = {"embedded": 0, "payload_only": 0, "deleted": 0}

    # silinmesi gerekenler
    stale = [pid for pid in existing if pid not in wanted]
    for i in range(0, len(stale), 512):
        client.delete(coll, points_selector=qm.PointIdsList(points=list(stale[i : i + 512])))
    stats["deleted"] = len(stale)

    to_embed = [p for pid, p in wanted.items() if existing.get(pid) != p.text_hash]
    to_update = [p for pid, p in wanted.items() if existing.get(pid) == p.text_hash]

    for i in range(0, len(to_embed), BATCH):
        batch = to_embed[i : i + BATCH]
        vecs = embedder.encode([p.text for p in batch])
        client.upsert(
            coll,
            points=[
                qm.PointStruct(id=point_id(p.chunk_id), vector=v, payload=p.model_dump())
                for p, v in zip(batch, vecs, strict=True)
            ],
        )
        stats["embedded"] += len(batch)
        log.info(
            "embedded", extra={"extra_data": {"done": stats["embedded"], "total": len(to_embed)}}
        )

    for p in to_update:
        client.overwrite_payload(coll, payload=p.model_dump(), points=[point_id(p.chunk_id)])
    stats["payload_only"] = len(to_update)

    count = client.count(coll, exact=True).count
    manifest = write_manifest(
        settings.manifest_path,
        settings,
        embedder.dim,
        count,
        sorted({p.source for p in payloads}),
    )
    return manifest, stats
