from __future__ import annotations

import re
import subprocess
from pathlib import Path

from anatomy_ks.models import Structure
from anatomy_ks.ontology import parse_synonym
from anatomy_ks.registry import Registry, normalize_id, normalize_name
from tests.conftest import by_name

ROOT = Path(__file__).parents[1]


def test_only_fma_terms(structures: list[Structure]) -> None:
    names = {s.name_en for s in structures}
    assert "no fma term" not in names
    assert all(s.id.startswith("FMA:") for s in structures)


def test_fields(registry: Registry) -> None:
    f = by_name(registry, "femur")
    assert f.name_tr == "Uyluk kemiği" and f.name_la == "Femur"
    assert f.id in f.ancestors  # ancestors kendisini içerir
    assert f.system == ["skeletal"] and f.region == ["lower_limb"]
    assert "os femoris" in f.synonyms and "thigh bone" in f.synonyms
    assert "mesomere of fin" not in f.synonyms  # homoloji eş anlamlıları atılır
    assert by_name(registry, "gluteus maximus").id in f.neighbors  # ters yönlü ilişki


def test_laterality_generic(registry: Registry) -> None:
    left = by_name(registry, "left femur")
    femur = by_name(registry, "femur")
    assert left.laterality == "left" and left.generic_id == femur.id
    assert femur.laterality == "bilateral"


def test_search_same_structure_first(registry: Registry) -> None:
    femur = by_name(registry, "femur")
    for q in ("femur", "uyluk", "os femoris", "Uyluk Kemiği", "thigh"):
        assert registry.search(q, limit=3)[0].id == femur.id, q


def test_search_filters(registry: Registry) -> None:
    res = registry.search("", system="muscular", limit=50)
    assert res and all("muscular" in s.system for s in res)


def test_resolve_formats(registry: Registry) -> None:
    femur = by_name(registry, "femur")
    num = femur.id.split(":")[1]
    assert registry.resolve(f"FMA_{num}").id == femur.id
    assert registry.resolve(f"FMA%3A{num}").id == femur.id
    assert normalize_id("fma:12") == "FMA:12"


def test_search_ids_includes_generic_and_neighbors(registry: Registry) -> None:
    left = by_name(registry, "left femur")
    femur = by_name(registry, "femur")
    ids = registry.search_ids(left.id)
    assert ids[:2] == [left.id, femur.id]
    assert by_name(registry, "tibia").id in ids
    assert len(registry.search_ids(left.id, max_neighbors=1)) <= 3


def test_normalize_name_turkish() -> None:
    assert normalize_name("UYLUK KEMİĞİ") == normalize_name("uyluk kemiği") == "uyluk kemigi"


def test_parse_synonym() -> None:
    assert parse_synonym('"os femoris" RELATED OMO:0003011 [X:1]') == (
        "os femoris",
        "RELATED",
        True,
        False,
    )


def test_no_hardcoded_ids_in_src() -> None:
    """R2: src/ içinde elle yazılmış FMA/Uberon ID'si yok."""
    hits = []
    for p in (ROOT / "src").rglob("*.py"):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\b(FMA|UBERON):\d{3,}", line):
                hits.append(f"{p}:{i}")
    for p in (ROOT / "config").glob("*.yaml"):
        if re.search(r"\b(FMA|UBERON):\d{3,}", p.read_text(encoding="utf-8")):
            hits.append(str(p))
    assert hits == []


def test_no_3d_concepts_in_src() -> None:
    out = subprocess.run(
        ["grep", "-riE", "mesh|unity|raycast", str(ROOT / "src")], capture_output=True, text=True
    )
    assert out.stdout == ""
