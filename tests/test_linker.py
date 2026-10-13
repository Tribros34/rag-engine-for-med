from __future__ import annotations

from anatomy_ks.ingestion.linker import Linker, link_key
from anatomy_ks.registry import Registry
from tests.conftest import by_name


def test_exact_heading(registry: Registry) -> None:
    lk = Linker(registry)
    r = lk.link(["Skeleton", "The Femur (Thigh Bone)"], "Text.")
    assert r.fma_id == by_name(registry, "femur").id and r.method == "heading_exact"
    assert r.related_ids[0] == r.fma_id


def test_composed_subheading(registry: Registry) -> None:
    lk = Linker(registry)
    r = lk.link(["Osteology", "The Femur", "The Head (caput femoris)"], "Globular.")
    assert r.fma_id == by_name(registry, "head of femur").id


def test_walk_up_and_generic_word_not_linked(registry: Registry) -> None:
    lk = Linker(registry)
    r = lk.link(["Osteology", "Tibia", "Structure"], "Plain text.")
    assert r.fma_id == by_name(registry, "tibia").id


def test_fuzzy_threshold(registry: Registry) -> None:
    lk = Linker(registry)
    r = lk.link(["X", "Gluteus Maximuss"], "No names here.")
    assert r.fma_id == by_name(registry, "gluteus maximus").id and r.method == "heading_fuzzy"
    r2 = lk.link(["X", "Completely Unrelated Heading"], "Nothing.")
    assert r2.fma_id is None


def test_word_boundary_ulna_vs_ulnaris(registry: Registry) -> None:
    lk = Linker(registry)
    m = lk.mentions("The flexor carpi ulnaris muscle is strong.")
    assert by_name(registry, "ulna").id not in m
    assert by_name(registry, "flexor carpi ulnaris muscle").id in m


def test_related_ids_contains_fma_and_mentions(registry: Registry) -> None:
    lk = Linker(registry)
    r = lk.link(
        ["Muscles", "Gluteus Maximus"], "It is inserted into the femur near the femur head."
    )
    assert r.related_ids[0] == r.fma_id
    assert by_name(registry, "femur").id in r.related_ids


def test_sided_maps_to_generic(registry: Registry) -> None:
    lk = Linker(registry)
    r = lk.link(["X", "Left Femur"], "text")
    assert r.fma_id == by_name(registry, "femur").id


def test_dominant_mention_fallback(registry: Registry) -> None:
    lk = Linker(registry)
    r = lk.link(["Chapter", "Some Topic"], "The tibia is long. The tibia bears weight.")
    assert r.method == "mention" and r.fma_id == by_name(registry, "tibia").id


def test_link_key_plural() -> None:
    assert link_key("The Bones") == "bone"
    assert link_key("Lower Extremity") == "lower limb"
