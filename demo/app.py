"""Streamlit demo — "dokunma" yerine yapı seçme listesi (Bölüm 11).

Arka planda 3D atlasın çağıracağı /v1 API'sinin aynısı kullanılır.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 60

SYSTEMS = {
    "": "Tümü",
    "skeletal": "İskelet",
    "articular": "Eklem",
    "muscular": "Kas",
    "nervous": "Sinir",
    "cardiovascular": "Kalp-damar",
    "lymphatic": "Lenfatik",
    "respiratory": "Solunum",
    "digestive": "Sindirim",
    "urinary": "Üriner",
    "reproductive": "Üreme",
    "endocrine": "Endokrin",
    "integumentary": "Örtü",
    "special_senses": "Duyu organları",
}
REGIONS = {
    "": "Tümü",
    "head_neck": "Baş-boyun",
    "back": "Sırt",
    "thorax": "Göğüs",
    "abdomen": "Karın",
    "pelvis_perineum": "Pelvis-perine",
    "upper_limb": "Üst ekstremite",
    "lower_limb": "Alt ekstremite",
}
EXAMPLES = [
    "Hangi kaslar yapışır?",
    "Hangi sinir innerve eder?",
    "Neyle eklem yapar?",
    "Kanlanması nasıldır?",
    "Nerede bulunur, komşuları neler?",
]
SECTION_TITLES = {
    "tanim": "Tanım",
    "konum_komsuluk": "Konum ve komşuluk",
    "eklem": "Eklemler",
    "yapisma": "Yapışan kaslar",
    "kanlanma": "Kanlanma",
    "innervasyon": "İnnervasyon",
    "fonksiyon": "Fonksiyon",
    "klinik": "Klinik",
}

st.set_page_config(page_title="Anatomi Atlası — AI Asistan", page_icon="🦴", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def get_json(path: str, params: tuple[tuple[str, str], ...] = ()) -> Any:
    r = requests.get(f"{API}{path}", params=dict(params), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def stream_answer(sid: str, question: str, k: int) -> None:
    box = st.empty()
    text = ""
    meta: dict[str, Any] = {}
    sources: list[dict[str, Any]] = []
    done: dict[str, Any] = {}
    err: dict[str, Any] | None = None
    with requests.post(
        f"{API}/v1/ask/stream",
        json={"structure_id": sid, "question": question, "max_sources": k},
        stream=True,
        timeout=TIMEOUT,
    ) as r:
        if r.status_code != 200:
            st.error(r.json().get("error", {}).get("message", r.text))
            return
        event = None
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:])
                if event == "meta":
                    meta = data
                elif event == "token":
                    text += data["t"]
                    box.markdown(text + " ▌")
                elif event == "sources":
                    sources = data["sources"]
                elif event == "done":
                    done = data
                elif event == "error":
                    err = data
    box.markdown(text or "")
    if err:
        st.warning(f"{err.get('code')}: {err.get('message')}")
        fb = err.get("fallback")
        if fb:
            st.info("LLM kullanılamadığı için yapı kartı gösteriliyor:")
            st.markdown(fb.get("answer", ""))
            sources = fb.get("sources", [])
    st.session_state["last"] = {"meta": meta, "sources": sources, "done": done}


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    st.subheader("Kaynaklar")
    for s in sources:
        badge = "🟢 Katman 1" if s.get("source_tier") == 1 else "🟡 Katman 2"
        with st.expander(f"[{s['n']}] {badge} · {s['source']} — {s['section']}"):
            st.caption(f"Lisans: {s.get('license')} · skor: {s.get('score')} · {s.get('chunk_id')}")
            st.write(s["text"])


# ---------------- arayüz ----------------
st.title("🦴 Anatomi Atlası — AI Asistan (Demo)")
st.caption(
    "Yapı seçimi 3D atlastaki dokunmayı simüle eder; servis yalnızca yapı ID'si ile çalışır."
)

try:
    health = get_json("/v1/health")
except requests.RequestException as e:
    st.error(f"API'ye ulaşılamıyor ({API}): {e}")
    st.stop()

with st.sidebar:
    st.header("Yapı seç")
    q = st.text_input("Ara (TR / EN / Latince)", "femur")
    system = st.selectbox("Sistem", list(SYSTEMS), format_func=lambda x: SYSTEMS[x])
    region = st.selectbox("Bölge", list(REGIONS), format_func=lambda x: REGIONS[x])
    params = tuple(
        (k, v) for k, v in {"q": q, "system": system, "region": region, "limit": "30"}.items() if v
    )
    items = get_json("/v1/structures", params)
    labels = {f"{s['name_tr']} — {s['name_la']}": s["id"] for s in items}
    choice = st.selectbox("Sonuçlar (dokunma simülasyonu)", list(labels)) if labels else None
    k = st.slider("Kaynak sayısı", 1, 10, 6)
    dev = st.toggle("Geliştirici paneli", value=False)
    st.divider()
    st.caption(f"Qdrant: {health['qdrant']} · LLM: {health['llm']}")

if not choice:
    st.info("Sol panelden bir yapı arayın ve seçin.")
    st.stop()

sid = labels[choice]
structure = get_json(f"/v1/structures/{sid}")
card = get_json(f"/v1/structures/{sid}/card")

col_card, col_qa = st.columns([2, 3], gap="large")
with col_card:
    st.subheader(f"{structure['name_tr']} ({structure['name_la']})")
    st.caption(f"{structure['id']} · {structure['name_en']} · taraf: {structure['laterality']}")
    st.info(card["summary"])
    for key, title in SECTION_TITLES.items():
        val = card["sections"].get(key)
        if val:
            with st.expander(title):
                st.markdown(val)
    if card.get("faq"):
        with st.expander("Sık sorulanlar"):
            for f in card["faq"]:
                st.markdown(f"**{f['q']}**  \n{f['a']}")
    if not card.get("reviewed"):
        st.caption("⚠️ Kart uzman onayından geçmedi.")

with col_qa:
    st.subheader("Soru sor")
    cols = st.columns(len(EXAMPLES))
    for i, ex in enumerate(EXAMPLES):
        if cols[i].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state["question"] = ex
    question = st.text_area("Sorunuz", key="question", max_chars=500, height=80)
    if st.button("Sor", type="primary") and question.strip():
        stream_answer(sid, question, k)
    last = st.session_state.get("last")
    if last:
        render_sources(last["sources"])
        if dev:
            st.subheader("Geliştirici paneli")
            meta = last["meta"]
            st.json(
                {
                    "mode": meta.get("mode"),
                    "intents": meta.get("intents"),
                    "latency_ms": last["done"].get("latency_ms"),
                    "first_token_ms": last["done"].get("first_token_ms"),
                    **(meta.get("debug") or {}),
                }
            )

st.divider()
st.caption(
    "⚕️ Eğitim amaçlıdır, klinik karar için kullanılmaz. Kaynaklar: OpenStax Anatomy and "
    "Physiology (CC BY) · Gray's Anatomy of the Human Body, 1918 (public domain) · Uberon (CC BY)."
)
