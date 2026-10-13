from __future__ import annotations

from anatomy_ks.retrieval.prompt import CitationFilter, build_user_message, validate_citations
from anatomy_ks.retrieval.search import search
from anatomy_ks.service import Deps
from tests.conftest import by_name


def _search(deps: Deps, name: str, q: str, intents: list[str], k: int = 4):  # type: ignore[no-untyped-def]
    s = by_name(deps.registry, name)
    return search(
        q,
        s,
        intents,
        k,
        registry=deps.registry,
        client=deps.client,
        embed=deps.embed,
        settings=deps.settings,
    )


def test_filter_never_returns_other_structures(deps: Deps) -> None:
    res = _search(deps, "tibia", "forearm medial bone ulna flexor", [])
    ids = set(res.ids)
    assert res.hits, "tibia komşusu femur chunk'ları gelmeli"
    for h in res.hits:
        assert ids & set(h.payload["related_ids"])
    assert all("Ulna" not in h.payload["heading_path"] for h in res.hits)


def test_neighbor_chunks_found(deps: Deps) -> None:
    res = _search(deps, "femur", "which muscles insert", ["yapisma"])
    assert any("Gluteus Maximus" in h.payload["heading_path"] for h in res.hits)


def test_stage2_fallback(deps: Deps) -> None:
    res = _search(deps, "femur", "development", ["gelisim"])
    assert res.stage == 2 and res.hits


def test_tier1_boost_and_section_cap(deps: Deps) -> None:
    res = _search(deps, "femur", "femur", [])
    finals = [h.final for h in res.hits]
    assert finals == sorted(finals, reverse=True)
    for h in res.hits:
        boost = 0.03 if h.payload["source_tier"] == 1 else 0.0
        assert abs(h.final - h.score - boost) < 1e-9


def test_left_structure_uses_generic(deps: Deps) -> None:
    res = _search(deps, "left femur", "hip joint", [])
    assert res.hits and by_name(deps.registry, "femur").id in res.ids


def test_citation_validation() -> None:
    text, bad = validate_citations("A [1]. B [7]. C [2][3].", 2)
    assert text == "A [1]. B . C [2]." and bad == [7, 3]


def test_citation_filter_streaming() -> None:
    f = CitationFilter(2)
    out = "".join(f.feed(t) for t in ["Bir [", "1] iki [", "9", "] üç [x] [2]"]) + f.flush()
    assert out == "Bir [1] iki  üç [x] [2]"


def test_prompt_contains_sections_and_no_numbered_card(deps: Deps) -> None:
    res = _search(deps, "femur", "hip", [])
    s = by_name(deps.registry, "femur")
    msg = build_user_message(s, "Özet [1] cümlesi", res.hits, "Soru?")
    for tag in ("<secili_yapi>", "<yapi_ozeti>", "<kaynaklar>", "<soru>"):
        assert tag in msg
    assert "Özet  cümlesi" in msg  # kart özeti numaralandırılmaz
    assert "[1] (Katman" in msg
