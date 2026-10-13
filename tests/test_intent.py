from __future__ import annotations

import pytest

from anatomy_ks.models import CONTENT_TYPES
from anatomy_ks.retrieval.intent import detect_intent, is_card_question


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        ("Buraya hangi kaslar yapışır?", "yapisma"),
        ("Hangi sinir innerve eder?", "innervasyon"),
        ("Neyle eklem yapar?", "eklem"),
        ("Kanlanması nasıl, hangi arterler?", "kanlanma"),
        ("Görevi nedir, ne işe yarar?", "fonksiyon"),
        ("Kırığı olursa ne olur?", "klinik"),
        ("Nerede bulunur, komşuları neler?", "konum_komsuluk"),
        ("Kemikleşme nasıl olur?", "gelisim"),
    ],
)
def test_detect(q: str, expected: str) -> None:
    assert expected in detect_intent(q)


def test_ambiguous_besle() -> None:
    got = detect_intent("Bu yapıyı ne besler?")
    assert {"kanlanma", "innervasyon"} <= set(got)


def test_returns_list_in_vocab() -> None:
    got = detect_intent("İstanbul'da hava nasıl?")
    assert isinstance(got, list) and all(g in CONTENT_TYPES for g in got)


def test_turkish_dotted_i() -> None:
    assert "innervasyon" in detect_intent("HANGİ SİNİR?")


@pytest.mark.parametrize("q", [None, "", "  ", "Bu nedir?", "anlat", "Kısaca anlat"])
def test_card_questions(q: str | None) -> None:
    assert is_card_question(q)


def test_not_card_question() -> None:
    assert not is_card_question("Femura hangi kaslar yapışır ve hangi sinirler innerve eder?")
