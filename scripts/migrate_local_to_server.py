"""Yerel (gömülü) Qdrant indeksini bir Qdrant sunucusuna kopyalar — yeniden embed etmeden.

QDRANT_URL=local:data/qdrant_local python scripts/migrate_local_to_server.py http://localhost:6333
"""

from __future__ import annotations

import sys

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from anatomy_ks.config import get_settings
from anatomy_ks.vectordb import ensure_collection, get_client


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    st = get_settings()
    src = get_client(st)
    dst = QdrantClient(url=sys.argv[1], timeout=30)
    coll = st.qdrant_collection
    info = src.get_collection(coll)
    vp = info.config.params.vectors
    dim = vp.size if isinstance(vp, qm.VectorParams) else 1024
    ensure_collection(dst, coll, dim)
    offset = None
    n = 0
    while True:
        pts, offset = src.scroll(
            coll, limit=256, offset=offset, with_payload=True, with_vectors=True
        )
        if pts:
            dst.upsert(
                coll,
                points=[
                    qm.PointStruct(id=p.id, vector=p.vector, payload=p.payload)  # type: ignore[arg-type]
                    for p in pts
                ],
            )
            n += len(pts)
        if offset is None:
            break
    print(f"kopyalandı: {n} point → {sys.argv[1]} ({dst.count(coll, exact=True).count} toplam)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
