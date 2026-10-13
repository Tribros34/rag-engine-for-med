"""Faz 3 kabul: demo yapıları için filtreli arama yalnızca yapı/komşu chunk'larını döndürür.

python scripts/check_filter.py [--names femur heart]
"""

from __future__ import annotations

import argparse
import sys

import yaml

from anatomy_ks.api.main import build_deps
from anatomy_ks.config import get_settings
from anatomy_ks.retrieval.search import search

QUESTIONS = ["Hangi kaslar yapışır?", "Kanlanması nasıldır?", "Nerede bulunur?"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*")
    args = ap.parse_args()
    st = get_settings()
    d = build_deps(st)
    names = (
        args.names
        or yaml.safe_load((st.config_dir / "demo_structures.yaml").read_text(encoding="utf-8"))[
            "structures"
        ]
    )
    by_name = {s.name_en.casefold(): s for s in d.registry.all()}
    total = ok = 0
    for name in names:
        s = by_name.get(str(name).casefold())
        if not s:
            continue
        for q in QUESTIONS:
            from anatomy_ks.retrieval.intent import detect_intent

            res = search(
                q,
                s,
                detect_intent(q),
                6,
                registry=d.registry,
                client=d.client,
                embed=d.embed,
                settings=st,
            )
            allowed = set(res.ids)
            for h in res.hits:
                total += 1
                vals = set(h.payload.get(res.field) or [])
                good = bool(vals & allowed)
                ok += good
                if not good:
                    print(f"  ✗ {name}: {h.chunk_id}")
            top = res.hits[0] if res.hits else None
            print(
                f"{name:20s} | {q:32s} | stage {res.stage} | {len(res.hits)} hit | "
                f"{(top.chunk_id if top else '-'):32s} | "
                f"{(top.payload.get('heading_path', [''])[-1][:40] if top else '')}"
            )
    pct = 100.0 * ok / total if total else 0.0
    print(f"\nFiltre doğruluğu: {ok}/{total} = {pct:.1f}%")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
