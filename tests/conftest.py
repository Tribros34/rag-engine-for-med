"""Ortak test fixture'ları. Ağ ve LLM çağrıları mock'lanır; testler internetsiz geçer."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from anatomy_ks.cards.build_cards import CardStore
from anatomy_ks.config import Settings
from anatomy_ks.ingestion.chunker import Block, Section, chunk_section
from anatomy_ks.ingestion.enrich import ContentTypeClassifier, enrich
from anatomy_ks.ingestion.linker import Linker
from anatomy_ks.llm.base import LLMUnavailable
from anatomy_ks.models import Structure
from anatomy_ks.ontology import build_registry
from anatomy_ks.registry import Registry
from anatomy_ks.service import Deps
from anatomy_ks.vectordb import ensure_collection, get_client, point_id

FIX = Path(__file__).parent / "fixtures"
DIM = 64


def fake_embed(text: str) -> list[float]:
    """Deterministik bag-of-words hash embedding (testler için)."""
    v = [0.0] * DIM
    for w in re.findall(r"\w+", text.casefold()):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class FakeLLM:
    def __init__(
        self, answer: str = "Uyluk kemiği (femur) kalça eklemine katılır [1]. Uydurma [9]."
    ) -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def stream(self, system: str, user: str, max_tokens: int = 800) -> Iterator[str]:
        self.calls.append((system, user))
        for i in range(0, len(self.answer), 7):
            yield self.answer[i : i + 7]

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        self.calls.append((system, user))
        return self.answer


class DownLLM:
    def stream(self, system: str, user: str, max_tokens: int = 800) -> Iterator[str]:
        raise LLMUnavailable("down")
        yield ""  # pragma: no cover

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        raise LLMUnavailable("down")


@pytest.fixture(scope="session")
def structures() -> list[Structure]:
    s, _ = build_registry(FIX / "mini.obo", FIX / "region_roots.yaml", FIX / "name_tr.yaml")
    return s


@pytest.fixture(scope="session")
def registry(structures: list[Structure]) -> Registry:
    return Registry(structures, group_min_descendants=5)


def by_name(reg: Registry, name: str) -> Structure:
    return next(s for s in reg.all() if s.name_en == name)


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        qdrant_url=":memory:",
        qdrant_collection=f"test_{tmp_path.name}",
        data_dir=tmp_path,
        config_dir=Path(__file__).parents[1] / "config",
        llm_model="",
        min_hits=2,
        top_k=4,
    )


def _sections() -> list[Section]:
    meta = dict(source="Test Book", source_url="https://example.invalid", license="CC BY")
    return [
        Section(
            ["Skeleton", "Femur"],
            [
                Block(
                    "para",
                    text=(
                        "The femur is the longest bone of the body. Its head articulates with the hip bone "
                        "at the hip joint. The femur articulates with the tibia at the knee joint."
                    ),
                )
            ],
            source_section="1.1 Femur",
            source_tier=1,
            chunk_prefix="t-ch01-s01",
            **meta,
        ),
        Section(
            ["Muscles", "Gluteus Maximus"],
            [
                Block(
                    "para",
                    text=(
                        "The gluteus maximus arises from the ilium and is inserted into the gluteal tuberosity "
                        "of the femur. Its action is extension of the thigh."
                    ),
                )
            ],
            source_section="2.1 Gluteal muscles",
            source_tier=2,
            chunk_prefix="t-ch02-s01",
            **meta,
        ),
        Section(
            ["Forearm", "Ulna"],
            [
                Block(
                    "para",
                    text=(
                        "The ulna is the medial bone of the forearm. The flexor carpi ulnaris muscle lies "
                        "along it."
                    ),
                )
            ],
            source_section="3.1 Ulna",
            source_tier=1,
            chunk_prefix="t-ch03-s01",
            **meta,
        ),
    ]


@pytest.fixture()
def deps(settings: Settings, registry: Registry, tmp_path: Path) -> Deps:
    client = get_client(settings)
    ensure_collection(client, settings.qdrant_collection, DIM)
    linker = Linker(registry)
    clf = ContentTypeClassifier(tmp_path / "ct.jsonl", None)
    from qdrant_client import models as qm

    pts = []
    for sec in _sections():
        for c in chunk_section(sec):
            lr = linker.link(c.heading_path, c.body)
            assert lr.fma_id, c.heading_path
            p = enrich(c, lr, registry, clf)
            pts.append(
                qm.PointStruct(
                    id=point_id(p.chunk_id), vector=fake_embed(p.text), payload=p.model_dump()
                )
            )
    client.upsert(settings.qdrant_collection, points=pts)
    return Deps(
        settings=settings,
        registry=registry,
        cards=CardStore(tmp_path / "cards", registry),
        client=client,
        embed=fake_embed,
        llm=FakeLLM(),
    )
