"""Tests for the FastAPI web UI: page/health routes, backend switching, error mapping."""

from __future__ import annotations

from ticketag.exceptions import ConfigurationError

from .conftest import StubClassifier


def test_index_serves_the_html_page(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "TickeTag" in response.text


def test_health_check_reports_ok(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classify_returns_result_via_rag_backend(client, monkeypatch):
    stub = StubClassifier(category="Hardware", justification="matched 3 neighbors", confidence=0.8)
    monkeypatch.setattr("ticketag.webui.app.rag_classifier_from_env", lambda: stub)

    response = client.post(
        "/api/classify", json={"text": "my monitor won't turn on", "backend": "rag"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "category": "Hardware",
        "justification": "matched 3 neighbors",
        "confidence": 0.8,
        "backend": "rag",
    }


def test_classify_returns_result_via_zeroshot_backend(client, monkeypatch):
    stub = StubClassifier(
        category="Access", justification="asked to reset a password", confidence=0.65
    )
    monkeypatch.setattr("ticketag.webui.app.zeroshot_classifier_from_env", lambda: stub)

    response = client.post(
        "/api/classify", json={"text": "I can't log into the VPN", "backend": "zeroshot"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "category": "Access",
        "justification": "asked to reset a password",
        "confidence": 0.65,
        "backend": "zeroshot",
    }


def test_classify_calls_the_backend_matching_the_request(client, monkeypatch):
    rag_stub = StubClassifier(category="Hardware", justification="rag reason", confidence=0.7)
    zeroshot_stub = StubClassifier(category="Access", justification="zs reason", confidence=0.6)
    monkeypatch.setattr("ticketag.webui.app.rag_classifier_from_env", lambda: rag_stub)
    monkeypatch.setattr("ticketag.webui.app.zeroshot_classifier_from_env", lambda: zeroshot_stub)

    response = client.post("/api/classify", json={"text": "printer is jammed", "backend": "rag"})

    assert response.status_code == 200
    assert response.json()["backend"] == "rag"
    assert rag_stub.received == ["printer is jammed"]
    assert zeroshot_stub.received == []


def test_classify_rejects_blank_ticket_text(client, monkeypatch):
    stub = StubClassifier()
    monkeypatch.setattr("ticketag.webui.app.rag_classifier_from_env", lambda: stub)

    response = client.post("/api/classify", json={"text": "   ", "backend": "rag"})

    assert response.status_code == 400
    assert "no usable text" in response.json()["detail"]
    assert stub.received == []


def test_classify_maps_configuration_error_to_a_clean_response(client, monkeypatch):
    def _raise_missing_key():
        raise ConfigurationError(
            "Missing API key: set CHROMA_API_KEY in the environment or in .env"
        )

    monkeypatch.setattr("ticketag.webui.app.rag_classifier_from_env", _raise_missing_key)

    response = client.post("/api/classify", json={"text": "ticket text", "backend": "rag"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Missing API key: set CHROMA_API_KEY in the environment or in .env"
    }


def test_classify_keeps_the_other_backend_usable_after_one_fails(client, monkeypatch):
    def _raise_missing_key():
        raise ConfigurationError("Missing API key")

    zeroshot_stub = StubClassifier(category="Access", justification="zs reason", confidence=0.6)
    monkeypatch.setattr("ticketag.webui.app.rag_classifier_from_env", _raise_missing_key)
    monkeypatch.setattr("ticketag.webui.app.zeroshot_classifier_from_env", lambda: zeroshot_stub)

    failed = client.post("/api/classify", json={"text": "ticket text", "backend": "rag"})
    succeeded = client.post("/api/classify", json={"text": "ticket text", "backend": "zeroshot"})

    assert failed.status_code == 503
    assert succeeded.status_code == 200
    assert succeeded.json()["backend"] == "zeroshot"


def test_classify_reuses_the_cached_pipeline_across_requests(client, monkeypatch):
    build_calls: list[int] = []

    def _build():
        build_calls.append(1)
        return StubClassifier()

    monkeypatch.setattr("ticketag.webui.app.rag_classifier_from_env", _build)

    client.post("/api/classify", json={"text": "first ticket", "backend": "rag"})
    client.post("/api/classify", json={"text": "second ticket", "backend": "rag"})

    assert len(build_calls) == 1


def test_classify_rejects_an_unknown_backend_name(client):
    response = client.post("/api/classify", json={"text": "ticket text", "backend": "gpt5"})

    assert response.status_code == 422
