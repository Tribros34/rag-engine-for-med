"""Uygulama ayarları. Tüm değerler ortam değişkenlerinden / .env'den gelir."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "anatomy_chunks"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16
    llm_provider: Literal["anthropic", "openai_compatible"] = "openai_compatible"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = Field(default="", repr=False)
    llm_timeout_s: float = 30.0
    llm_max_tokens: int = 800
    top_k: int = 6
    min_hits: int = 3
    neighbor_depth: int = 1
    max_neighbors: int = 40
    tier1_boost: float = 0.03
    max_per_section: int = 2
    group_min_descendants: int = 100
    answer_lang: str = "tr"
    cors_origins: str = "*"
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    config_dir: Path = Path("config")

    @field_validator("embedding_device", "llm_model", "llm_base_url", mode="before")
    @classmethod
    def _strip(cls, v: object) -> object:
        # .env satır içi yorumları ("cpu   # cpu | cuda") pydantic-settings tarafından
        # temizlenmez; burada güvenli şekilde kırpılır.
        if isinstance(v, str):
            return v.split("#", 1)[0].strip()
        return v

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _strip_provider(cls, v: object) -> object:
        if isinstance(v, str):
            return v.split("#", 1)[0].strip()
        return v

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # --- türetilmiş yollar ---
    @property
    def registry_path(self) -> Path:
        return self.data_dir / "registry" / "structures.json"

    @property
    def cards_dir(self) -> Path:
        return self.data_dir / "registry" / "cards"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "index_manifest.json"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def llm_configured(self) -> bool:
        if not self.llm_model:
            return False
        if self.llm_provider == "anthropic":
            return bool(self.llm_api_key)
        return bool(self.llm_base_url)


@lru_cache
def get_settings() -> Settings:
    return Settings()
