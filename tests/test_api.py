from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from anatomy_ks.api.main import IndexModelMismatch, create_app
from anatomy_ks.config import Settings
from anatomy_ks.service import Deps
from anatomy_ks.vectordb import write_manifest
from tests.conftest import DownLLM, by_name


@pytest.fixture()
def client(deps: Deps) -> Iterator[TestClient]:
    app = create_app(deps.settings, deps)
    with TestClient(app) as c:
        yield c


def _fid(deps: Deps, name: str = "femur") -> str:
    return by_name(deps.registry, name).id


def _events(text: str) -> list[tuple[str, dict]]:  # type: ignore[type-arg]
    out = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        ev = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        out.append((ev, data))
    return out


def test_health(client: TestClient) -> None:
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["qdrant"].startswith("ok") and body["llm"] == "not_configured"
    assert set(body) == {"status", "qdrant", "llm", "index_manifest"}


def test_structures_search(client: TestClient, deps: Deps) -> None:
    for q in ("femur", "uyluk", "os femoris"):
        r = client.get("/v1/structures", params={"q": q})
        assert r.status_code == 200 and r.json()[0]["id"] == _fid(deps), q


def test_structure_id_formats(client: TestClient, deps: Deps) -> None:
    fid = _fid(deps)
    assert client.get(f"/v1/structures/{fid.replace(':', '%3A')}").json()["id"] == fid
    assert client.get(f"/v1/structures/{fid.replace(':', '_')}").json()["id"] == fid


def test_404(client: TestClient) -> None:
    r = client.get("/v1/structures/FMA_1")
    assert r.status_code == 404
    assert r.json() == {
        "error": {"code": "STRUCTURE_NOT_FOUND", "message": "Yapı bulunamadı: FMA:1"}
    }
    r = client.post("/v1/ask", json={"structure_id": "FMA:1", "question": "x"})
    assert r.status_code == 404


def test_validation_errors(client: TestClient, deps: Deps) -> None:
    r = client.post("/v1/ask", json={"structure_id": _fid(deps), "max_sources": 11})
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"
    r = client.post("/v1/ask", json={"structure_id": _fid(deps), "question": "x" * 501})
    assert r.status_code == 400 and r.json()["error"]["code"] == "QUESTION_TOO_LONG"


def test_card_fast_path_no_llm(client: TestClient, deps: Deps) -> None:
    r = client.post("/v1/ask", json={"structure_id": _fid(deps)})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "card" and "Uyluk kemiği" in body["answer"]
    assert deps.llm.calls == []  # type: ignore[attr-defined]
    c = client.get(f"/v1/structures/{_fid(deps)}/card")
    assert c.status_code == 200 and c.json()["structure_id"] == _fid(deps)


def test_left_uses_generic_card(client: TestClient, deps: Deps) -> None:
    r = client.get(f"/v1/structures/{_fid(deps, 'left femur')}/card")
    assert r.json()["structure_id"] == _fid(deps)


def test_ask_rag_and_citation_cleanup(client: TestClient, deps: Deps) -> None:
    r = client.post(
        "/v1/ask", json={"structure_id": _fid(deps), "question": "Hangi kaslar yapışır?"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "rag" and body["intents"] == ["yapisma"]
    assert "[9]" not in body["answer"] and "[1]" in body["answer"]
    assert [s["n"] for s in body["sources"]] == list(range(1, len(body["sources"]) + 1))
    assert {"n", "chunk_id", "text", "source", "section", "source_tier", "license", "score"} <= set(
        body["sources"][0]
    )
    system, user = deps.llm.calls[-1]  # type: ignore[attr-defined]
    assert "KESİN KURALLAR" in system and "<kaynaklar>" in user


def test_sse_order(client: TestClient, deps: Deps) -> None:
    r = client.post(
        "/v1/ask/stream", json={"structure_id": _fid(deps), "question": "Hangi kaslar yapışır?"}
    )
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = _events(r.text)
    names = [e for e, _ in ev]
    assert names[0] == "meta" and names[-2:] == ["sources", "done"]
    assert set(names[1:-2]) == {"token"}
    text = "".join(d["t"] for e, d in ev if e == "token")
    assert "[9]" not in text and "[1]" in text


def test_sse_card_mode(client: TestClient, deps: Deps) -> None:
    r = client.post("/v1/ask/stream", json={"structure_id": _fid(deps), "question": "Bu nedir?"})
    ev = _events(r.text)
    assert [e for e, _ in ev] == ["meta", "token", "sources", "done"]
    assert ev[0][1]["mode"] == "card"


def test_llm_down_card_still_works(client: TestClient, deps: Deps) -> None:
    deps.llm = DownLLM()
    r = client.post("/v1/ask", json={"structure_id": _fid(deps), "question": "bu nedir"})
    assert r.status_code == 200 and r.json()["mode"] == "card"
    r = client.post(
        "/v1/ask", json={"structure_id": _fid(deps), "question": "Hangi kaslar yapışır?"}
    )
    assert r.status_code == 503
    body = r.json()
    assert body["error"]["code"] == "LLM_UNAVAILABLE" and body["fallback"]["mode"] == "card"
    r = client.post(
        "/v1/ask/stream", json={"structure_id": _fid(deps), "question": "Hangi kaslar yapışır?"}
    )
    ev = _events(r.text)
    assert ev[-1][0] == "error" and ev[-1][1]["code"] == "LLM_UNAVAILABLE"


def test_no_context(client: TestClient, deps: Deps) -> None:
    gm = by_name(deps.registry, "skeletal system").id
    r = client.post("/v1/ask", json={"structure_id": gm, "question": "Hangi sinir innerve eder?"})
    body = r.json()
    if body["mode"] == "no_context":
        assert "bulunmuyor" in body["answer"] and body["sources"] == []


def test_vector_db_down(client: TestClient, deps: Deps) -> None:
    class Broken:
        def query_points(self, *a: object, **k: object) -> None:
            raise ConnectionError("down")

    deps.client = Broken()  # type: ignore[assignment]
    r = client.post(
        "/v1/ask", json={"structure_id": _fid(deps), "question": "Hangi kaslar yapışır?"}
    )
    assert r.status_code == 503 and r.json()["error"]["code"] == "VECTOR_DB_UNAVAILABLE"


def test_index_model_mismatch_refuses_start(settings: Settings, deps: Deps) -> None:
    other = settings.model_copy(update={"embedding_model": "other/model"})
    write_manifest(settings.manifest_path, other, 64, 1, ["x"])
    app = create_app(settings, deps)
    with pytest.raises(IndexModelMismatch), TestClient(app):
        pass
