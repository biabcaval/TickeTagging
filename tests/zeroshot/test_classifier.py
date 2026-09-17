from __future__ import annotations

import json

import pytest

from ticketag.exceptions import ClassificationError, InferenceError
from ticketag.zeroshot import ZeroShotTicketClassifier
from ticketag.zeroshot.inference import ProviderError

from .conftest import FakeTransport

VALID_REPLY = json.dumps(
    {"category": "Access", "justification": "User cannot log into the VPN.", "confidence": 0.9}
)


def build(settings, categories, *replies):
    transport = FakeTransport(*replies)
    return ZeroShotTicketClassifier(settings, categories, transport=transport), transport


def test_classify_returns_category_and_justification(settings, categories):
    classifier, _ = build(settings, categories, VALID_REPLY)

    result = classifier.classify("I cannot log into the VPN.")

    assert result.category == "Access"
    assert result.justification == "User cannot log into the VPN."
    assert result.confidence == pytest.approx(0.9)


def test_classify_sends_every_category_in_the_user_content(settings, categories):
    classifier, transport = build(settings, categories, VALID_REPLY)

    classifier.classify("My monitor is broken.")

    _model, _system, user_content, config = transport.calls[0]
    assert all(category.name in user_content for category in categories)
    assert config["temperature"] == settings.temperature


def test_classify_sends_the_configured_model(settings, categories):
    classifier, transport = build(settings, categories, VALID_REPLY)

    classifier.classify("ticket")

    assert transport.calls[0][0] == settings.model


def test_classify_extracts_json_wrapped_in_prose_and_fences(settings, categories):
    reply = f"Sure! Here you go:\n```json\n{VALID_REPLY}\n```\nHope it helps."
    classifier, _ = build(settings, categories, reply)

    assert classifier.classify("ticket").category == "Access"


def test_classify_normalizes_category_casing(settings, categories):
    reply = json.dumps(
        {"category": "hr support", "justification": "Payroll question.", "confidence": 0.7}
    )
    classifier, _ = build(settings, categories, reply)

    assert classifier.classify("ticket").category == "HR Support"


def test_classify_recovers_from_near_miss_category_name(settings, categories):
    reply = json.dumps(
        {"category": "Administrative Rights", "justification": "Needs admin.", "confidence": 0.8}
    )
    classifier, _ = build(settings, categories, reply)

    assert classifier.classify("ticket").category == "Administrative rights"


def test_classify_rejects_category_outside_the_allowed_set(settings, categories):
    reply = json.dumps(
        {"category": "Networking", "justification": "Router down.", "confidence": 0.8}
    )
    classifier, _ = build(settings, categories, reply)

    with pytest.raises(ClassificationError, match="outside the allowed categories"):
        classifier.classify("ticket")


def test_classify_raises_when_reply_has_no_json(settings, categories):
    classifier, _ = build(settings, categories, "I think this is a Hardware ticket.")

    with pytest.raises(ClassificationError, match="no JSON object"):
        classifier.classify("ticket")


def test_classify_raises_when_justification_is_missing(settings, categories):
    reply = json.dumps({"category": "Access", "confidence": 0.9})
    classifier, _ = build(settings, categories, reply)

    with pytest.raises(ClassificationError, match="no justification"):
        classifier.classify("ticket")


def test_classify_defaults_confidence_when_model_omits_it(settings, categories):
    reply = json.dumps({"category": "Storage", "justification": "Disk full on share."})
    classifier, _ = build(settings, categories, reply)

    assert classifier.classify("ticket").confidence == pytest.approx(0.5)


def test_classify_rescales_percentage_confidence(settings, categories):
    reply = json.dumps({"category": "Storage", "justification": "Disk full.", "confidence": 95})
    classifier, _ = build(settings, categories, reply)

    assert classifier.classify("ticket").confidence == pytest.approx(0.95)


def test_classify_retries_transient_errors_then_succeeds(settings, categories):
    rate_limited = ProviderError("Rate limit exceeded", 429)
    unreachable = ProviderError("Could not reach endpoint", retryable=True)
    classifier, transport = build(settings, categories, rate_limited, unreachable, VALID_REPLY)

    assert classifier.classify("ticket").category == "Access"
    assert len(transport.calls) == 3


def test_classify_does_not_retry_permanent_errors(settings, categories):
    unauthorized = ProviderError("Invalid API key", 401)
    classifier, transport = build(settings, categories, unauthorized, VALID_REPLY)

    with pytest.raises(InferenceError):
        classifier.classify("ticket")
    assert len(transport.calls) == 1


def test_classify_gives_up_after_max_retries(settings, categories):
    rate_limited = ProviderError("Rate limit exceeded", 429)
    classifier, transport = build(settings, categories, *[rate_limited] * settings.max_retries)

    with pytest.raises(InferenceError, match="after 3 attempt"):
        classifier.classify("ticket")
    assert len(transport.calls) == settings.max_retries


def test_classifier_requires_at_least_one_category(settings):
    with pytest.raises(ValueError, match="at least one category"):
        ZeroShotTicketClassifier(settings, (), transport=FakeTransport())


def test_classify_reports_the_upstream_reason_when_the_provider_keeps_failing(settings, categories):
    saturated = ProviderError("Resource exhausted, please retry later", 429)
    classifier, _ = build(settings, categories, *[saturated] * settings.max_retries)

    with pytest.raises(InferenceError, match="Resource exhausted"):
        classifier.classify("ticket")


def test_classify_retries_an_empty_reply_that_finished_cleanly(settings, categories):
    classifier, transport = build(settings, categories, None, VALID_REPLY)

    assert classifier.classify("ticket").category == "Access"
    assert len(transport.calls) == 2


def test_classify_fails_fast_on_a_token_budget_error(settings, categories):
    budget_error = ProviderError(
        "Model hit the token budget before answering; raise TICKETAG_MAX_TOKENS", retryable=False
    )
    classifier, transport = build(settings, categories, budget_error, VALID_REPLY)

    with pytest.raises(InferenceError, match="TICKETAG_MAX_TOKENS"):
        classifier.classify("ticket")
    assert len(transport.calls) == 1
