"""Tüm veri sözleşmeleri (Bölüm 6). Alan adları sözleşmedir; değiştirilmez (R8)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CONTENT_TYPES: tuple[str, ...] = (
    "tanim",
    "konum_komsuluk",
    "yapisma",
    "eklem",
    "kanlanma",
    "innervasyon",
    "fonksiyon",
    "klinik",
    "gelisim",
)
SYSTEMS: tuple[str, ...] = (
    "skeletal",
    "articular",
    "muscular",
    "nervous",
    "cardiovascular",
    "lymphatic",
    "respiratory",
    "digestive",
    "urinary",
    "reproductive",
    "endocrine",
    "integumentary",
    "special_senses",
)
REGIONS: tuple[str, ...] = (
    "head_neck",
    "back",
    "thorax",
    "abdomen",
    "pelvis_perineum",
    "upper_limb",
    "lower_limb",
)

Laterality = Literal["left", "right", "midline", "bilateral", "none"]
Mode = Literal["card", "rag", "no_context"]


def _only(allowed: tuple[str, ...], values: list[str], field: str) -> list[str]:
    bad = [v for v in values if v not in allowed]
    if bad:
        raise ValueError(f"{field}: sözlük dışı değer(ler) {bad}")
    return values


class Structure(BaseModel):
    id: str
    uberon_id: str
    name_la: str
    name_en: str
    name_tr: str
    name_tr_reviewed: bool = False
    synonyms: list[str] = []
    system: list[str] = []
    region: list[str] = []
    laterality: Laterality = "none"
    generic_id: str | None = None
    parent_ids: list[str] = []
    ancestors: list[str] = []
    neighbors: list[str] = []
    deprecated: bool = False

    @field_validator("system")
    @classmethod
    def _sys(cls, v: list[str]) -> list[str]:
        return _only(SYSTEMS, v, "system")

    @field_validator("region")
    @classmethod
    def _reg(cls, v: list[str]) -> list[str]:
        return _only(REGIONS, v, "region")


class ChunkPayload(BaseModel):
    chunk_id: str
    text: str
    heading_path: list[str]
    fma_id: str
    related_ids: list[str]
    ancestors: list[str]
    system: list[str]
    region: list[str]
    laterality: Laterality = "none"
    content_type: list[str]
    is_caption: bool = False
    source: str
    source_section: str
    source_url: str
    source_tier: Literal[1, 2]
    license: str
    lang: str = "en"
    token_count: int
    text_hash: str

    @field_validator("content_type")
    @classmethod
    def _ct(cls, v: list[str]) -> list[str]:
        return _only(CONTENT_TYPES, v, "content_type")

    @field_validator("system")
    @classmethod
    def _sys(cls, v: list[str]) -> list[str]:
        return _only(SYSTEMS, v, "system")

    @field_validator("region")
    @classmethod
    def _reg(cls, v: list[str]) -> list[str]:
        return _only(REGIONS, v, "region")


class FAQItem(BaseModel):
    q: str
    a: str
    source_chunks: list[str] = []


class StructureCard(BaseModel):
    structure_id: str
    summary: str
    sections: dict[str, str | None]
    faq: list[FAQItem] = []
    source_chunks: list[str] = []
    generated_at: str
    reviewed: bool = False


# ---------- API ----------


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    structure_id: str
    # Uzunluk sınırı (500) route'ta kontrol edilir ki 422 yerine 400 QUESTION_TOO_LONG dönsün.
    question: str | None = None
    lang: str = "tr"
    max_sources: int = Field(default=6, ge=1, le=10)


class SourceRef(BaseModel):
    n: int
    chunk_id: str
    text: str
    source: str
    section: str
    source_tier: int
    license: str
    score: float
    # Opsiyonel ek alan (R8 uyarınca opsiyonel): atıf için kaynak bağlantısı.
    source_url: str | None = None


class AskResponse(BaseModel):
    structure_id: str
    structure_name: str
    answer: str
    mode: Mode
    intents: list[str]
    sources: list[SourceRef]
    latency_ms: int
    # Opsiyonel ek alanlar (geliştirici paneli için): filtreye giren ID'ler ve filtre alanı.
    debug: dict[str, object] | None = None


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    llm: str
    index_manifest: dict[str, object] | None


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class IndexManifest(BaseModel):
    embedding_model: str
    dim: int
    collection: str
    points: int
    created_at: str
    sources: list[str] = []
