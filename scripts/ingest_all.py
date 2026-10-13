"""Offline pipeline: indir → registry → çıkar → böl → bağla → zenginleştir → embed/indeksle.

Kullanım:
    python scripts/ingest_all.py                 # tüm adımlar
    python scripts/ingest_all.py --skip-index    # Qdrant'a yazmadan (Faz 2 çıktıları)
    python scripts/ingest_all.py --skip-registry # mevcut registry'yi kullan

Her adım idempotenttir (R9): aynı girdiyle ikinci çalıştırma aynı çıktıyı üretir.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict

from anatomy_ks.config import get_settings
from anatomy_ks.ingestion.chunker import RawChunk, chunk_section
from anatomy_ks.ingestion.download import ensure_grays, ensure_openstax, ensure_uberon
from anatomy_ks.ingestion.enrich import ContentTypeClassifier, enrich
from anatomy_ks.ingestion.extract_grays import extract_grays
from anatomy_ks.ingestion.extract_openstax import extract_openstax
from anatomy_ks.ingestion.linker import Linker, LinkResult
from anatomy_ks.llm.base import get_llm
from anatomy_ks.logging import setup_logging
from anatomy_ks.ontology import build_registry
from anatomy_ks.registry import Registry


def _dist(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    qs = statistics.quantiles(values, n=10) if len(values) > 1 else [values[0]] * 9
    return {
        "n": len(values),
        "min": min(values),
        "p10": qs[0],
        "median": statistics.median(values),
        "p90": qs[-1],
        "max": max(values),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-registry", action="store_true")
    ap.add_argument("--skip-index", action="store_true")
    args = ap.parse_args()

    st = get_settings()
    setup_logging(st.log_level)
    proc = st.processed_dir
    proc.mkdir(parents=True, exist_ok=True)

    # 1–2: ontoloji + registry
    obo = ensure_uberon(st.raw_dir)
    if not args.skip_registry or not st.registry_path.exists():
        structures, reg_report = build_registry(
            obo, st.config_dir / "region_roots.yaml", st.config_dir / "name_tr.yaml"
        )
        Registry.save(structures, st.registry_path)
        print("registry:", json.dumps(reg_report, ensure_ascii=False))
    reg = Registry.load(st.registry_path, st.group_min_descendants)

    # 3–5: kaynak + çıkarma + chunking
    book = ensure_openstax(st.raw_dir)
    grays_txt = ensure_grays(st.raw_dir)
    sections = extract_openstax(book, st.config_dir / "openstax_chapters.yaml")
    sections += extract_grays(grays_txt, st.config_dir / "grays_parts.yaml")
    chunks: list[RawChunk] = [c for s in sections for c in chunk_section(s)]

    # 6: linker
    linker = Linker(reg)
    linked: list[tuple[RawChunk, LinkResult]] = []
    review_rows: list[dict[str, str]] = []
    unlinked_headings: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = defaultdict(Counter)
    with (proc / "chunks_raw.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            lr = linker.link(c.heading_path, c.body)
            per_source[c.source][lr.method] += 1
            row = {
                **c.__dict__,
                "fma_id": lr.fma_id,
                "related_ids": lr.related_ids,
                "link_method": lr.method,
                "link_matched": lr.matched,
                "link_score": lr.score,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if lr.fma_id is None:
                unlinked_headings[" > ".join(c.heading_path)] += 1
                cand = lr.review[0] if lr.review else ("", "", 0.0)
                review_rows.append(
                    {
                        "chunk_id": c.chunk_id,
                        "heading_path": " > ".join(c.heading_path),
                        "candidate_text": cand[0],
                        "candidate_id": cand[1],
                        "score": f"{cand[2]:.1f}",
                        "reason": "fuzzy_85_90" if lr.review else "no_match",
                    }
                )
            else:
                linked.append((c, lr))
    with (proc / "link_review.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "chunk_id",
                "heading_path",
                "candidate_text",
                "candidate_id",
                "score",
                "reason",
            ],
        )
        w.writeheader()
        w.writerows(review_rows)

    by_src_words: dict[str, list[int]] = defaultdict(list)
    for c in chunks:
        by_src_words[c.source].append(c.word_count)
    link_report = {
        "sources": {
            src: {
                "chunks": sum(cnt.values()),
                "linked_pct": round(100 * (1 - cnt["unlinked"] / max(sum(cnt.values()), 1)), 1),
                "methods": dict(cnt),
                "captions": sum(1 for c in chunks if c.source == src and c.is_caption),
                "words": _dist(by_src_words[src]),
            }
            for src, cnt in per_source.items()
        },
        "top_unlinked_headings": unlinked_headings.most_common(50),
    }
    (proc / "link_report.json").write_text(
        json.dumps(link_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("link:", json.dumps(link_report["sources"], ensure_ascii=False))

    # 7: zenginleştirme
    llm = get_llm(st) if st.llm_configured else None
    clf = ContentTypeClassifier(proc / "content_type_cache.jsonl", llm)
    payloads = [enrich(c, lr, reg, clf) for c, lr in linked]
    clf.save()
    with (proc / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for p in payloads:
            f.write(p.model_dump_json() + "\n")
    ct = Counter(t for p in payloads for t in p.content_type)
    print(
        "enrich:",
        json.dumps(
            {"payloads": len(payloads), "classifier": clf.stats, "content_type": dict(ct)},
            ensure_ascii=False,
        ),
    )

    # 8: embedding + Qdrant
    if args.skip_index:
        return 0
    from anatomy_ks.ingestion.index import index_chunks
    from anatomy_ks.vectordb import get_client, get_embedder

    emb = get_embedder(st.embedding_model, st.embedding_device, st.embedding_batch_size)
    manifest, stats = index_chunks(payloads, st, get_client(st), emb)
    print("index:", json.dumps({"manifest": manifest.model_dump(), **stats}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
