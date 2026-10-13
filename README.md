# anatomy-knowledge-service

3D/AR anatomi atlası için **yapı-bağlamlı, kaynaklı soru-cevap servisi** (RAG).
Kullanıcı atlasta bir yapı seçer (FMA ID), doğal dilde soru sorar. Servis yalnızca o yapıya ve
ontolojik komşularına bağlı kaynak metinleri bulur ve **Türkçe, kaynak numaralı** cevap üretir.

> ⚕️ Eğitim amaçlıdır; klinik tanı/tedavi önerisi vermez.

Ajan talimatları ve tüm kurallar: [`CLAUDE.md`](CLAUDE.md). Faz raporları: [`docs/PHASE_REPORTS.md`](docs/PHASE_REPORTS.md).

---

## Hızlı başlangıç

### 1) LLM anahtarını ekle (son adım)

`.env` dosyasında (git'e girmez) **bir** sağlayıcıyı doldurun:

```dotenv
# Anthropic
LLM_PROVIDER=anthropic
LLM_MODEL=<model-adı>
LLM_API_KEY=<anahtar>

# veya OpenAI uyumlu (Ollama / vLLM / LM Studio / OpenAI)
LLM_PROVIDER=openai_compatible
LLM_MODEL=<model-adı>
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=            # gerekmiyorsa boş
```

Anahtar girildikten sonra (bir kez):

```bash
python scripts/ingest_all.py --skip-registry   # içerik türlerini LLM ile yeniden sınıflandırır (önbellekli)
python scripts/build_cards.py                  # demo yapıları için kartlar
python scripts/translate_names.py --limit 1000 && python scripts/build_registry.py  # (ops.) Türkçe adlar
python scripts/run_eval.py                     # değerlendirme raporu
```

LLM olmadan da çalışır: yapı arama, kart (ontoloji iskelet kartı), filtreli arama, kaynak listesi.
Serbest soru → `503 LLM_UNAVAILABLE` + `fallback` alanında kart.

### 2a) Docker ile (hedef kurulum)

```bash
cp .env.example .env                                 # LLM ayarlarını doldurun
docker compose up -d qdrant
docker compose --profile ingest run --rm ingest      # ilk kurulum: indir + indeksle (~15-30 dk, CPU)
docker compose up                                    # qdrant + api (:8000) + demo (:8501)
```

Yerelde zaten indekslediyseniz yeniden embed etmeden taşıyın:
`QDRANT_URL=local:data/qdrant_local python scripts/migrate_local_to_server.py http://localhost:6333`

### 2b) Docker'sız (yerel geliştirme)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[embed,demo,dev]"
cp .env.example .env && sed -i '' 's#^QDRANT_URL=.*#QDRANT_URL=local:data/qdrant_local#' .env
python scripts/ingest_all.py                         # indir → registry → chunk → link → embed
uvicorn anatomy_ks.api.main:app --port 8000          # API (+ /docs)
streamlit run demo/app.py                            # demo :8501
```

`QDRANT_URL=local:<yol>` gömülü Qdrant kullanır (tek süreç erişebilir: indeksleme sırasında API'yi kapatın).

---

## Mimari

```
OFFLINE  Uberon OBO ─► ontology.py ─► data/registry/structures.json (6068 FMA yapısı)
         OpenStax 1e ─┐
         Gray's 1918 ─┴► extract ─► chunker ─► linker (ad→FMA) ─► enrich ─► bge-m3 ─► Qdrant
ONLINE   Demo / 3D atlas ─(structure_id + soru)─► FastAPI /v1
           registry doğrula → kart hızlı yolu → niyet → ID genişletme (yapı+genel+komşu)
           → aşama-1 (yapı ∧ niyet) / aşama-2 (yapı) filtreli arama + Katman-1 bonusu
           → prompt (kart özeti + numaralı kaynaklar) → LLM stream (SSE) → atıf doğrulama
```

| Klasör | İçerik |
|---|---|
| `src/anatomy_ks/` | `config`, `models` (tüm şemalar), `ontology` (OBO→registry), `registry` (arama/ID genişletme), `vectordb`, `service` (soru-cevap akışı) |
| `ingestion/` | `download`, `extract_openstax`, `extract_grays`, `chunker`, `linker`, `enrich`, `index` |
| `retrieval/` | `intent`, `search`, `prompt` (Bölüm 7.6 prompt'u birebir + atıf doğrulama) |
| `llm/` | `anthropic`, `openai_compatible` (httpx) — model adı yalnızca `LLM_MODEL`'den |
| `api/` | `/v1` uçları, SSE, hata gövdeleri |
| `config/` | bölüm seçimleri, sistem/bölge kökleri, niyet kelimeleri, TR/LA ad sözlüğü, demo yapıları |

## API (`/v1`, sözleşme — R8)

| Metot | Yol | Not |
|---|---|---|
| GET | `/v1/health` | `{status, qdrant, llm, index_manifest}` |
| GET | `/v1/structures?q=&system=&region=&limit=` | TR/EN/LA ad + eş anlamlı; birebir → önek → fuzzy |
| GET | `/v1/structures/{id}` | `FMA%3A123` ve `FMA_123` kabul edilir |
| GET | `/v1/structures/{id}/card` | Kart; taraflı yapı → genel yapının kartı |
| POST | `/v1/ask` | `{structure_id, question?, lang, max_sources}` → `AskResponse` |
| POST | `/v1/ask/stream` | SSE: `meta → token* → sources → done`; hata: `error` |

Hatalar: `{"error": {"code", "message"}}` — `400 QUESTION_TOO_LONG`, `404 STRUCTURE_NOT_FOUND`,
`422 VALIDATION_ERROR`, `503 LLM_UNAVAILABLE` (+ `fallback` kart), `503 VECTOR_DB_UNAVAILABLE`.
`INDEX_MODEL_MISMATCH`: manifest'teki model ≠ `EMBEDDING_MODEL` ise API **açılmaz**.

**Sözleşmeye eklenen opsiyonel alanlar (R8 notu):** `SourceRef.source_url`, `AskResponse.debug`
(filtre ID'leri, aşama, skorlar), SSE `meta.mode`/`meta.debug`, `done.first_token_ms`,
503 gövdesinde `fallback`.

### 3D entegrasyonu

3D ekibi mesh → FMA ID eşlemesini kendi tarafında tutar; servis mesh adı bilmez (R1).
Dokunma → `GET /v1/structures/{id}/card` (anında kart) → soru → `POST /v1/ask` veya `/v1/ask/stream`.
Bölge/sistem katmanı seçilirse o düğümün ID'si gönderilir; servis `ancestors` filtresine geçer.

## Kaynaklar ve lisans

| Katman | Kaynak | Lisans | Edinim |
|---|---|---|---|
| 0 | Uberon (`releases/2026-10-01`) | CC BY | `purl.obolibrary.org/obo/uberon.obo` |
| 1 | OpenStax *Anatomy and Physiology* 1e | **CC BY 3.0** (depo `LICENSE.txt`) | `github.com/philschatz/anatomy-book` |
| 2 | Gray's *Anatomy of the Human Body*, 20th ed. (1918) | Public domain | Internet Archive `anatomyofhumanbo00grayrich` (tek OCR dosyası) |

Atıf: "OpenStax, Anatomy and Physiology. Rice University. CC BY." ·
"Gray H. Anatomy of the Human Body, 20th ed. (1918). Public domain." · Görsel indekslenmez (R4).

## Kalite

```bash
ruff check src tests scripts demo && mypy src && pytest -q
python scripts/check_filter.py     # Faz 3: filtre doğruluğu (gerçek indeks)
python scripts/run_eval.py         # Bölüm 12 metrikleri → data/eval/report_*.md
```
