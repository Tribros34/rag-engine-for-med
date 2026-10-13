"""Demo yapıları için kart üretimi (Bölüm 7.7). LLM gerektirir.

python scripts/build_cards.py            # eksik kartları üret
python scripts/build_cards.py --force    # hepsini yeniden üret
python scripts/build_cards.py --names femur tibia
"""

from __future__ import annotations

import argparse
import sys

import yaml

from anatomy_ks.api.main import build_deps
from anatomy_ks.cards.build_cards import build_card, card_path, save_card
from anatomy_ks.config import get_settings
from anatomy_ks.llm.base import LLMUnavailable
from anatomy_ks.logging import setup_logging
from anatomy_ks.retrieval.search import SearchResult, search


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--names", nargs="*")
    args = ap.parse_args()
    st = get_settings()
    setup_logging(st.log_level)
    if not st.llm_configured:
        print(
            "LLM yapılandırılmadı (.env: LLM_PROVIDER / LLM_MODEL / LLM_API_KEY). Kart üretilemez."
        )
        return 2
    d = build_deps(st)
    names = (
        args.names
        or yaml.safe_load((st.config_dir / "demo_structures.yaml").read_text(encoding="utf-8"))[
            "structures"
        ]
    )
    by_name = {s.name_en.casefold(): s for s in d.registry.all()}
    ok = 0
    for name in names:
        s = by_name.get(str(name).casefold())
        if s is None:
            print(f"! registry'de yok: {name}")
            continue
        s = d.registry.resolve(s.generic_id) if s.generic_id else s
        if card_path(st.cards_dir, s.id).exists() and not args.force:
            print(f"= {name}: mevcut")
            continue

        def search_fn(q: str, intents: list[str], k: int, s=s) -> SearchResult:  # type: ignore[no-untyped-def]
            return search(
                q, s, intents, k, registry=d.registry, client=d.client, embed=d.embed, settings=st
            )

        try:
            card = build_card(s, d.llm, search_fn)
        except LLMUnavailable as e:
            print(f"! LLM hatası: {e}")
            return 3
        save_card(card, st.cards_dir)
        filled = sum(1 for v in card.sections.values() if v)
        print(f"+ {name}: {filled} bölüm, {len(card.faq)} SSS, {len(card.source_chunks)} kaynak")
        ok += 1
    print(f"tamam: {ok} kart")
    return 0


if __name__ == "__main__":
    sys.exit(main())
