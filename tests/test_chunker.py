from __future__ import annotations

from anatomy_ks.ingestion.chunker import Block, Section, chunk_section, split_sentences


def _sec(blocks: list[Block], path: list[str] | None = None) -> Section:
    return Section(path or ["Ch", "Femur"], blocks, "S", "1.1", "u", 1, "CC BY", "x-ch01-s01")


def _long_para(n_sent: int) -> str:
    return " ".join(
        f"Sentence number {i} describes the femur in some detail here." for i in range(n_sent)
    )


def test_heading_prefix_and_ids_deterministic() -> None:
    sec = _sec([Block("para", text=_long_para(5))])
    a = chunk_section(sec)
    b = chunk_section(sec)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b] == ["x-ch01-s01-0000"]
    assert a[0].text.startswith("Ch > Femur : ")
    assert a[0].text_hash == b[0].text_hash


def test_size_bounds_and_no_mid_sentence_cut() -> None:
    sec = _sec([Block("para", text=_long_para(120))])
    chunks = chunk_section(sec)
    assert len(chunks) > 1
    for c in chunks:
        assert c.word_count <= 400 + 60
        assert c.body.endswith("here.")  # cümle sonu
        assert c.body.startswith("Sentence number")


def test_overlap_between_consecutive_chunks() -> None:
    chunks = chunk_section(_sec([Block("para", text=_long_para(120))]))
    last = split_sentences(chunks[0].body)[-1]
    assert last in chunks[1].body


def test_captions_separate() -> None:
    sec = _sec(
        [Block("para", text=_long_para(3)), Block("caption", text="The femur in anterior view.")]
    )
    chunks = chunk_section(sec)
    assert [c.is_caption for c in chunks] == [False, True]
    assert chunks[1].text.startswith("Ch > Femur : The femur")


def test_table_header_repeated() -> None:
    rows = [
        f"Muscle: m{i} | Origin: ilium area number {i} with many extra words to fill"
        for i in range(80)
    ]
    sec = _sec([Block("table", header="Muscle | Origin", rows=rows)])
    chunks = chunk_section(sec)
    assert len(chunks) > 1
    assert all(c.body.startswith("Muscle | Origin") for c in chunks)


def test_sentence_split_keeps_fig_abbrev() -> None:
    assert len(split_sentences("See Fig. 12 for detail. Next sentence.")) == 2
