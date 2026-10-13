"""Değerlendirme (Bölüm 12).

    python scripts/run_eval.py --init     # golden_set.jsonl tohumunu (10 soru) üret
    python scripts/run_eval.py            # metrikleri hesapla → data/eval/report_<tarih>.{json,md}

Tohum sorularda structure_id registry'den çözülür (R2); expected_chunk_ids, linker'ın o yapıya
bağladığı ve beklenen içerik türünü taşıyan chunk'lardır ("gümüş" etiket). Hepsi
"verified": false — uzman onayı gerekir.

LLM yoksa yalnızca retrieval/niyet/filtre/kart gecikmesi metrikleri üretilir; cevap kalitesi
metrikleri (bilgi kapsamı, uydurma, atıf geçerliliği, RAG gecikmesi) "n/a" olur.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anatomy_ks.api.main import build_deps
from anatomy_ks.config import get_settings
from anatomy_ks.llm.base import LLMUnavailable
from anatomy_ks.retrieval.intent import detect_intent
from anatomy_ks.retrieval.prompt import SYSTEM_PROMPT, build_user_message
from anatomy_ks.retrieval.search import search
from anatomy_ks.service import Deps, card_response

SEED: list[dict[str, Any]] = [
    {
        "name": "femur",
        "q": "Femur hangi yapılarla eklem yapar?",
        "intents": ["eklem"],
        "facts": ["acetabulum", "tibia", "patella"],
    },
    {
        "name": "femur",
        "q": "Femurun proksimal ucunda hangi kısımlar bulunur?",
        "intents": ["tanim"],
        "facts": ["trochanter", "head", "neck"],
    },
    {
        "name": "gluteus maximus",
        "q": "Gluteus maximus nereye yapışır?",
        "intents": ["yapisma"],
        "facts": ["gluteal tuberosity", "iliotibial"],
    },
    {
        "name": "biceps brachii",
        "q": "Biceps brachii'nin görevi nedir?",
        "intents": ["fonksiyon"],
        "facts": ["flex", "supinat"],
    },
    {
        "name": "heart",
        "q": "Kalp nerede bulunur, komşulukları nelerdir?",
        "intents": ["konum_komsuluk"],
        "facts": ["mediastinum", "diaphragm"],
    },
    {
        "name": "heart",
        "q": "Kalbin kanlanması nasıldır?",
        "intents": ["kanlanma"],
        "facts": ["coronary"],
    },
    {
        "name": "sciatic nerve",
        "q": "Siyatik sinir hangi kasları innerve eder?",
        "intents": ["innervasyon"],
        "facts": ["hamstring", "biceps femoris"],
    },
    {
        "name": "knee joint",
        "q": "Diz ekleminin bağları nelerdir?",
        "intents": ["eklem"],
        "facts": ["cruciate", "collateral", "menisc"],
    },
    {
        "name": "liver",
        "q": "Karaciğerin kan akımı nasıl gerçekleşir?",
        "intents": ["kanlanma"],
        "facts": ["portal", "hepatic artery"],
    },
    {
        "name": "lung",
        "q": "Akciğerin lobları ve konumu nedir?",
        "intents": ["konum_komsuluk", "tanim"],
        "facts": ["lobe", "pleura"],
    },
]

JUDGE_SYSTEM = (
    "You are a strict fact-checker. Given SOURCES and an ANSWER (Turkish), decide whether the "
    'answer contains any claim NOT supported by the sources. Return ONLY JSON: {"unsupported": '
    'true|false, "claims": ["..."]}'
)


def init_golden(d: Deps, out: Path) -> None:
    rows = []
    chunks = [
        json.loads(x)
        for x in (d.settings.processed_dir / "chunks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]
    by_name = {s.name_en.casefold(): s for s in d.registry.all()}
    for i, item in enumerate(SEED, start=1):
        s = by_name[item["name"]]
        exp = [
            c["chunk_id"]
            for c in chunks
            if c["fma_id"] == s.id
            and set(c["content_type"]) & set(item["intents"])
            and not c["is_caption"]
        ][:20]
        rows.append(
            {
                "id": f"q{i:03d}",
                "structure_id": s.id,
                "question": item["q"],
                "expected_intents": item["intents"],
                "expected_facts": item["facts"],
                "expected_chunk_ids": exp,
                "verified": False,
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print(f"yazıldı: {out} ({len(rows)} soru, verified=false)")


def pct(xs: list[float]) -> float | None:
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def p50(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 1) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    st = get_settings()
    d = build_deps(st)
    gpath = Path("data/eval/golden_set.jsonl")
    if args.init:
        init_golden(d, gpath)
        return 0
    rows = [json.loads(x) for x in gpath.read_text(encoding="utf-8").splitlines() if x.strip()]
    use_llm = st.llm_configured and not args.no_llm

    recall, filt, intent_ok, facts_cov, halluc, cite_valid = [], [], [], [], [], []
    card_ms, first_ms, full_ms, retr_ms = [], [], [], []
    per_q = []
    # ısınma (model yükleme gecikmesini ölçüme katma)
    d.embed("ısınma")
    for r in rows:
        s = d.registry.resolve(r["structure_id"])
        t0 = time.perf_counter()
        card_response(d, s, t0, [])
        card_ms.append((time.perf_counter() - t0) * 1000)
        intents = detect_intent(r["question"])
        intent_ok.append(float(bool(set(intents) & set(r["expected_intents"]))))
        t0 = time.perf_counter()
        res = search(
            r["question"],
            s,
            intents,
            6,
            registry=d.registry,
            client=d.client,
            embed=d.embed,
            settings=st,
        )
        retr_ms.append((time.perf_counter() - t0) * 1000)
        got = [h.chunk_id for h in res.hits]
        if r["expected_chunk_ids"]:
            recall.append(float(bool(set(got) & set(r["expected_chunk_ids"]))))
        allowed = set(res.ids)
        for h in res.hits:
            filt.append(float(bool(set(h.payload.get(res.field) or []) & allowed)))
        q_out: dict[str, Any] = {"id": r["id"], "intents": intents, "top": got, "stage": res.stage}
        if use_llm and res.hits:
            try:
                user = build_user_message(s, None, res.hits, r["question"])
                t0 = time.perf_counter()
                first = None
                parts = []
                for tok in d.llm.stream(SYSTEM_PROMPT, user, st.llm_max_tokens):
                    if first is None:
                        first = (time.perf_counter() - t0) * 1000
                    parts.append(tok)
                full_ms.append((time.perf_counter() - t0) * 1000)
                if first is not None:
                    first_ms.append(first)
                ans = "".join(parts)
                cites = [int(x) for x in re.findall(r"\[(\d+)\]", ans)]
                if cites:
                    cite_valid.append(sum(1 for c in cites if 1 <= c <= len(res.hits)) / len(cites))
                low = ans.casefold()
                facts_cov.append(
                    sum(1 for f in r["expected_facts"] if f.casefold() in low)
                    / max(len(r["expected_facts"]), 1)
                )
                src = "\n\n".join(f"[{i}] {h.payload['text']}" for i, h in enumerate(res.hits, 1))
                jr = d.llm.complete(JUDGE_SYSTEM, f"SOURCES:\n{src}\n\nANSWER:\n{ans}", 300)
                m = re.search(r"\{.*\}", jr, re.S)
                halluc.append(float(bool(json.loads(m.group(0)).get("unsupported")) if m else 0.0))
                q_out["answer"] = ans
            except (LLMUnavailable, ValueError) as e:
                q_out["llm_error"] = str(e)
        per_q.append(q_out)

    metrics = {
        "retrieval_recall_at_6_pct": pct(recall),
        "filter_accuracy_pct": pct(filt),
        "intent_accuracy_pct": pct(intent_ok),
        "fact_coverage_pct": pct(facts_cov),
        "hallucination_rate_pct": pct(halluc),
        "citation_validity_pct": pct(cite_valid),
        "card_latency_p50_ms": p50(card_ms),
        "retrieval_latency_p50_ms": p50(retr_ms),
        "rag_first_token_p50_ms": p50(first_ms),
        "rag_full_p50_ms": p50(full_ms),
    }
    targets = {
        "retrieval_recall_at_6_pct": ">= 85",
        "filter_accuracy_pct": "= 100",
        "intent_accuracy_pct": ">= 80",
        "fact_coverage_pct": ">= 70",
        "hallucination_rate_pct": "<= 5",
        "citation_validity_pct": "= 100",
        "card_latency_p50_ms": "< 100",
        "rag_first_token_p50_ms": "< 1500",
        "rag_full_p50_ms": "< 6000",
    }
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = {
        "generated_at": stamp,
        "n_questions": len(rows),
        "llm": use_llm,
        "verified_questions": sum(1 for r in rows if r.get("verified")),
        "metrics": metrics,
        "targets": targets,
        "per_question": per_q,
    }
    base = Path(f"data/eval/report_{stamp}")
    base.with_suffix(".json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = [
        f"# Değerlendirme raporu — {stamp}",
        "",
        f"Soru: {len(rows)} (uzman onaylı: {out['verified_questions']}) · LLM: {use_llm}",
        "",
        "| Metrik | Değer | Hedef |",
        "|---|---|---|",
    ]
    for k, v in metrics.items():
        md.append(f"| {k} | {'n/a' if v is None else v} | {targets.get(k, '—')} |")
    base.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
