"""Tests for GenAITransport/_read_reply: mapping google-genai responses and failures."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from ticketag.zeroshot.inference import ProviderError, RateLimiter, _read_reply


def response(text: str | None = None, finish_reason: str | None = None, block_reason=None):
    """Build a minimal stand-in for google.genai.types.GenerateContentResponse."""
    candidates = None
    if finish_reason is not None:
        candidates = [SimpleNamespace(finish_reason=SimpleNamespace(name=finish_reason))]
    return SimpleNamespace(
        text=text,
        candidates=candidates,
        prompt_feedback=SimpleNamespace(block_reason=block_reason) if block_reason else None,
    )


def test_read_reply_returns_the_response_text():
    assert _read_reply(response(text="ok", finish_reason="STOP")) == "ok"


def test_read_reply_raises_a_permanent_error_when_the_prompt_is_blocked():
    with pytest.raises(ProviderError) as caught:
        _read_reply(response(block_reason="SAFETY"))

    assert not caught.value.retryable
    assert "SAFETY" in str(caught.value)


def test_read_reply_raises_a_permanent_error_when_the_token_budget_runs_out():
    with pytest.raises(ProviderError, match="TICKETAG_MAX_TOKENS") as caught:
        _read_reply(response(text=None, finish_reason="MAX_TOKENS"))

    assert not caught.value.retryable


def test_read_reply_raises_a_permanent_error_for_a_safety_finish_reason():
    with pytest.raises(ProviderError) as caught:
        _read_reply(response(text=None, finish_reason="SAFETY"))

    assert not caught.value.retryable


def test_read_reply_raises_a_retryable_error_for_an_empty_reply_that_finished_cleanly():
    with pytest.raises(ProviderError) as caught:
        _read_reply(response(text="", finish_reason="STOP"))

    assert caught.value.retryable


def test_read_reply_raises_a_retryable_error_when_there_are_no_candidates_at_all():
    with pytest.raises(ProviderError) as caught:
        _read_reply(response(text=None))

    assert caught.value.retryable


def test_provider_error_marks_known_transient_statuses_retryable():
    assert ProviderError("boom", 429).retryable
    assert ProviderError("boom", 503).retryable


def test_provider_error_marks_other_statuses_permanent():
    assert not ProviderError("boom", 401).retryable
    assert not ProviderError("boom", 404).retryable


def test_provider_error_respects_an_explicit_retryable_override():
    assert ProviderError("boom", 200, retryable=True).retryable
    assert not ProviderError("boom", 429, retryable=False).retryable


def test_rate_limiter_spaces_calls_by_the_configured_interval():
    limiter = RateLimiter(requests_per_minute=1200)  # one call every 50 ms

    start = time.monotonic()
    for _ in range(3):
        limiter.wait()
    elapsed = time.monotonic() - start

    assert elapsed >= 0.09


def test_rate_limiter_is_disabled_when_unlimited():
    limiter = RateLimiter(requests_per_minute=0)

    start = time.monotonic()
    for _ in range(50):
        limiter.wait()

    assert time.monotonic() - start < 0.05
