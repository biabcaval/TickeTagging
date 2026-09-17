from __future__ import annotations

import json
import time
from dataclasses import replace

import pytest

from ticketag.exceptions import ClassificationError, InferenceError
from ticketag.zeroshot import ZeroShotTicketClassifier
from ticketag.zeroshot.inference import ProviderError, RateLimiter

from .conftest import FakeTransport, Truncated, http_error

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


def test_classify_sends_every_category_in_the_prompt(settings, categories):
    classifier, transport = build(settings, categories, VALID_REPLY)

    classifier.classify("My monitor is broken.")

    prompt = transport.payloads[0]["messages"][1]["content"]
    assert all(category.name in prompt for category in categories)
    assert transport.payloads[0]["temperature"] == settings.temperature


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
    unreachable = ProviderError("Could not reach endpoint", retryable=True)
    classifier, transport = build(settings, categories, http_error(503), unreachable, VALID_REPLY)

    assert classifier.classify("ticket").category == "Access"
    assert len(transport.payloads) == 3


def test_classify_does_not_retry_permanent_errors(settings, categories):
    classifier, transport = build(settings, categories, http_error(401), VALID_REPLY)

    with pytest.raises(InferenceError):
        classifier.classify("ticket")
    assert len(transport.payloads) == 1


def test_classify_gives_up_after_max_retries(settings, categories):
    classifier, transport = build(settings, categories, *[http_error(429)] * settings.max_retries)

    with pytest.raises(InferenceError, match="after 3 attempt"):
        classifier.classify("ticket")
    assert len(transport.payloads) == settings.max_retries


def test_classifier_requires_at_least_one_category(settings):
    with pytest.raises(ValueError, match="at least one category"):
        ZeroShotTicketClassifier(settings, (), transport=FakeTransport())


def test_classify_retries_provider_error_returned_with_http_200(settings, categories):
    # OpenRouter reports upstream saturation in the body of a successful response.
    saturated = {
        "message": "Provider returned error",
        "code": 429,
        "metadata": {"raw": "temporarily rate-limited upstream", "provider_name": "Google"},
    }
    classifier, transport = build(settings, categories, saturated, VALID_REPLY)

    assert classifier.classify("ticket").category == "Access"
    assert len(transport.payloads) == 2


def test_classify_reports_the_upstream_reason_when_the_provider_keeps_failing(settings, categories):
    saturated = {"message": "Provider error", "code": 429, "metadata": {"raw": "saturated"}}
    classifier, _ = build(settings, categories, *[saturated] * settings.max_retries)

    with pytest.raises(InferenceError, match="saturated"):
        classifier.classify("ticket")


def test_classify_does_not_retry_a_permanent_provider_error(settings, categories):
    invalid = {"message": "No endpoints found", "code": 404}
    classifier, transport = build(settings, categories, invalid, VALID_REPLY)

    with pytest.raises(InferenceError, match="after 1 attempt"):
        classifier.classify("ticket")
    assert len(transport.payloads) == 1


def test_classify_retries_an_empty_reply_that_finished_cleanly(settings, categories):
    classifier, transport = build(settings, categories, "", VALID_REPLY)

    assert classifier.classify("ticket").category == "Access"
    assert len(transport.payloads) == 2


def test_classify_fails_fast_when_the_token_budget_runs_out(settings, categories):
    classifier, transport = build(settings, categories, Truncated(), VALID_REPLY)

    with pytest.raises(InferenceError, match="TICKETAG_MAX_TOKENS"):
        classifier.classify("ticket")
    assert len(transport.payloads) == 1


def test_classify_raises_when_the_response_carries_neither_choices_nor_error(settings, categories):
    classifier, _ = build(settings, categories, *[{}] * settings.max_retries)

    with pytest.raises(InferenceError):
        classifier.classify("ticket")


def test_classify_offers_fallback_models_to_openrouter(settings, categories):
    routed = replace(settings, fallback_models=("vendor/backup:free",))
    classifier, transport = build(routed, categories, VALID_REPLY)

    classifier.classify("ticket")

    assert transport.payloads[0]["models"] == [routed.model, "vendor/backup:free"]


def test_classify_trims_the_routing_list_to_what_openrouter_accepts(settings, categories):
    routed = replace(settings, fallback_models=("a:free", "b:free", "c:free", "d:free"))
    classifier, transport = build(routed, categories, VALID_REPLY)

    classifier.classify("ticket")

    assert transport.payloads[0]["models"] == [routed.model, "a:free", "b:free"]


def test_classify_omits_routing_when_no_fallbacks_are_configured(settings, categories):
    classifier, transport = build(settings, categories, VALID_REPLY)

    classifier.classify("ticket")

    assert "models" not in transport.payloads[0]


def test_classify_denies_provider_data_collection(settings, categories):
    classifier, transport = build(settings, categories, VALID_REPLY)

    classifier.classify("ticket")

    assert transport.payloads[0]["provider"] == {"data_collection": "deny"}


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
