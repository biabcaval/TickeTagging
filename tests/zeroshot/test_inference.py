from __future__ import annotations

import io
import json
import urllib.error

import pytest

from ticketag.zeroshot.inference import HttpTransport, ProviderError

PAYLOAD = {"model": "vendor/model:free", "messages": [{"role": "user", "content": "hi"}]}


def fake_urlopen(monkeypatch, handler):
    """Replace the network call, recording the request the transport built."""
    sent = {}

    def opener(request, timeout=None):
        sent["request"] = request
        sent["timeout"] = timeout
        return handler()

    monkeypatch.setattr("urllib.request.urlopen", opener)
    return sent


def http_error(status: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://openrouter.ai/api/v1/chat/completions", status, "boom", {}, io.BytesIO(body)
    )


def test_send_posts_json_with_the_bearer_token(settings, monkeypatch):
    sent = fake_urlopen(monkeypatch, lambda: io.BytesIO(b'{"choices": []}'))

    HttpTransport(settings).send(PAYLOAD)

    request = sent["request"]
    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.get_header("Authorization") == f"Bearer {settings.api_token}"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data) == PAYLOAD
    assert sent["timeout"] == settings.timeout


def test_send_returns_the_decoded_response_body(settings, monkeypatch):
    body = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    fake_urlopen(monkeypatch, lambda: io.BytesIO(json.dumps(body).encode()))

    assert HttpTransport(settings).send(PAYLOAD) == body


def test_send_marks_a_rate_limit_response_retryable(settings, monkeypatch):
    body = json.dumps(
        {"error": {"message": "Provider returned error", "metadata": {"raw": "saturated"}}}
    ).encode()

    def raise_429():
        raise http_error(429, body)

    fake_urlopen(monkeypatch, raise_429)

    with pytest.raises(ProviderError) as caught:
        HttpTransport(settings).send(PAYLOAD)

    assert caught.value.status == 429
    assert caught.value.retryable
    assert "saturated" in str(caught.value)


def test_send_marks_an_auth_failure_permanent(settings, monkeypatch):
    def raise_401():
        raise http_error(401, b'{"error": {"message": "No auth credentials found"}}')

    fake_urlopen(monkeypatch, raise_401)

    with pytest.raises(ProviderError) as caught:
        HttpTransport(settings).send(PAYLOAD)

    assert caught.value.status == 401
    assert not caught.value.retryable
    assert "No auth credentials found" in str(caught.value)


def test_send_falls_back_to_the_raw_body_when_the_error_is_not_json(settings, monkeypatch):
    def raise_502():
        raise http_error(502, b"<html>bad gateway</html>")

    fake_urlopen(monkeypatch, raise_502)

    with pytest.raises(ProviderError, match="bad gateway"):
        HttpTransport(settings).send(PAYLOAD)


def test_send_treats_an_unreachable_endpoint_as_retryable(settings, monkeypatch):
    def raise_dns():
        raise urllib.error.URLError("Name or service not known")

    fake_urlopen(monkeypatch, raise_dns)

    with pytest.raises(ProviderError) as caught:
        HttpTransport(settings).send(PAYLOAD)

    assert caught.value.retryable


def test_send_treats_malformed_json_as_retryable(settings, monkeypatch):
    fake_urlopen(monkeypatch, lambda: io.BytesIO(b"not json at all"))

    with pytest.raises(ProviderError) as caught:
        HttpTransport(settings).send(PAYLOAD)

    assert caught.value.retryable
