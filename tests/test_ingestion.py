"""Çıkarıcılar ve zenginleştirme: görsel/navigasyon artığı yok, idempotent, sözlük dışı tür yok."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from anatomy_ks.ingestion.chunker import chunk_section
from anatomy_ks.ingestion.enrich import (
    ContentTypeClassifier,
    heuristic_content_type,
    parse_content_type,
)
from anatomy_ks.ingestion.extract_grays import GraysPart, extract_part
from anatomy_ks.ingestion.extract_openstax import TocSection, extract_section
from anatomy_ks.models import CONTENT_TYPES

NOISE = re.compile(r"!\[|\.jpg|\.png|<img|\{:|\[link\]|<div|<span|data-type|Review Questions")

OPENSTAX_MD = """---
title: "Bones of the Lower Limb"
layout: page
---

<div data-type="abstract" markdown="1">
By the end of this section, you will be able to:
* Identify the bones
</div>

Intro text about the lower limb and its bones in the body.

### Femur

The femur, or **thigh bone**{: data-type="term"}, is the single bone of the thigh ([\\[link\\]](#fig-1)). It is long.

 ![Alt text here.](../resources/femur.jpg "The femur is the single bone of the thigh region."){: #fig-1}

<div data-type="note" class="anatomy interactive" markdown="1">
Watch this [video][1] to see surgery.
</div>

<div data-type="note" class="anatomy disorders" markdown="1">
<div data-type="title">
Disorders
</div>
A fracture of the femoral neck is common in the elderly patients.
</div>

<table><thead><tr><th>Bone</th><th>Feature</th></tr></thead><tbody>
<tr><td>Femur</td><td>Longest bone</td></tr></tbody></table>

### Review Questions

<div data-type="exercise"><div data-type="problem" markdown="1">Which bone?</div></div>

[1]: http://openstax.org/l/x
"""

GRAYS_TXT = """

The  Femur  (Thigh  Bone).

The  femur  (Figs.  244,  245),  the  longest  and  strongest  bone  in  the  skeleton,  is
almost  perfectly  cylindrical  in  the  greater  part  of  its  extent.  It  is  separated  from
its  fellow  by  a  con-
siderable  interval.

THE  FEMUR

243

The  Head  (caput  femoris).—  The  head  which  is  globular  and  forms  rather  more
than  a  hemisphere,  is  directed  upward,  medialward,  and  a  little  forward.

Lesser  trochanter

Fig.  243.  —  Upper  extremity  of  right  femur  viewed  from  behind  and  above.

'  Wood,  Heath's  Dictionary  of  Practical  Surgery,  i,  426.
"""


@pytest.fixture()
def openstax_sections(tmp_path: Path):  # type: ignore[no-untyped-def]
    p = tmp_path / "m1.md"
    p.write_text(OPENSTAX_MD, encoding="utf-8")
    return extract_section(
        TocSection("U", "The Appendicular Skeleton", 8, "Bones of the Lower Limb", 4, p), "CC BY"
    )


def test_openstax_no_noise(openstax_sections) -> None:  # type: ignore[no-untyped-def]
    chunks = [c for s in openstax_sections for c in chunk_section(s)]
    assert chunks
    for c in chunks:
        assert not NOISE.search(c.text), c.text
        assert "surgery" not in c.text  # interactive kutusu atıldı
        assert "Which bone" not in c.text
    texts = " ".join(c.text for c in chunks)
    assert "thigh bone" in texts and "fracture of the femoral neck" in texts
    assert "Femur: Longest bone" in texts or "Bone: Femur" in texts  # tablo satırı
    caps = [c for c in chunks if c.is_caption]
    assert caps and caps[0].text.endswith("The femur is the single bone of the thigh region.")
    femur = [c for c in chunks if c.heading_path[-1] == "Femur" and not c.is_caption][0]
    assert femur.chunk_id.startswith("openstax-ch08-s04-")
    assert femur.source_section == "8.4 Bones of the Lower Limb"


def test_grays_cleaning() -> None:
    secs = extract_part(
        GRAYS_TXT.splitlines(), GraysPart("OSTEOLOGY", "II. Osteology", "osteology"), {"OSTEOLOGY"}
    )
    chunks = [c for s in secs for c in chunk_section(s)]
    paths = [tuple(c.heading_path) for c in chunks]
    assert ("II. Osteology", "The Femur (Thigh Bone)") in paths
    assert ("II. Osteology", "The Femur (Thigh Bone)", "The Head (caput femoris)") in paths
    allt = " ".join(c.text for c in chunks)
    assert "considerable" in allt  # tireleme birleşti
    assert "243" not in allt.replace("Fig. 243", "")  # sayfa no atıldı
    assert "Heath's Dictionary" not in allt  # dipnot atıldı
    assert "Lesser trochanter" not in allt  # figür etiketi atıldı
    assert any(c.is_caption and "Upper extremity of right femur" in c.text for c in chunks)
    assert "  " not in allt


def test_heuristic_content_type_vocab() -> None:
    got = heuristic_content_type(
        "The muscle arises from the ilium and is inserted into the femur; origin and insertion."
    )
    assert "yapisma" in got and all(g in CONTENT_TYPES for g in got)
    assert heuristic_content_type("Plain words.") == ["tanim"]


def test_parse_content_type_drops_unknown() -> None:
    assert parse_content_type('ok {"content_type": ["yapisma", "bogus", "klinik"]}') == [
        "yapisma",
        "klinik",
    ]
    with pytest.raises(ValueError):
        parse_content_type("no json")


def test_classifier_cache_idempotent(tmp_path: Path) -> None:
    class LLM:
        n = 0

        def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
            LLM.n += 1
            return '{"content_type": ["eklem", "nonsense"]}'

        def stream(self, *a: object, **k: object):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    c1 = ContentTypeClassifier(tmp_path / "c.jsonl", LLM())
    assert c1.classify("a", "h", "text") == ["eklem"]
    c1.save()
    c2 = ContentTypeClassifier(tmp_path / "c.jsonl", LLM())
    assert c2.classify("a", "h", "text") == ["eklem"] and LLM.n == 1
