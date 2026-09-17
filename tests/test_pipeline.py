from __future__ import annotations

import pytest

from ticketag.exceptions import ClassificationError, EmptyTicketError
from ticketag.models import Classification, ClassifiedTicket, FailedTicket
from ticketag.pipeline import TicketClassificationPipeline

HARDWARE = Classification("Hardware", "Broken monitor needs replacement.", 0.95)


class StubClassifier:
    """A backend stand-in, so pipeline tests stay independent of any model."""

    def __init__(self, *results: Classification | Exception) -> None:
        self.results = list(results)
        self.seen: list[str] = []

    def classify(self, ticket: str) -> Classification:
        self.seen.append(ticket)
        result = self.results.pop(0) if self.results else HARDWARE
        if isinstance(result, Exception):
            raise result
        return result


def test_run_passes_ticket_text_to_the_backend_unaltered():
    classifier = StubClassifier()
    pipeline = TicketClassificationPipeline(classifier)
    raw = "From: ana@corp.com\n<p>My monitor is BROKEN!</p>\nSent from my iPhone"

    result = pipeline.run(raw, ticket_id="T-1")

    assert isinstance(result, ClassifiedTicket)
    assert result.text == raw
    assert classifier.seen == [raw]
    assert result.classification.category == "Hardware"


def test_run_truncates_text_beyond_the_size_guard():
    pipeline = TicketClassificationPipeline(StubClassifier(), max_chars=40)

    result = pipeline.run("word " * 200, ticket_id="T-5")

    assert result.text.endswith("[... truncated ...]")
    assert len(result.text) < 80


def test_run_raises_when_ticket_is_blank():
    pipeline = TicketClassificationPipeline(StubClassifier())

    with pytest.raises(EmptyTicketError):
        pipeline.run("   \n\t ", ticket_id="T-2")


def test_run_batch_preserves_order_and_isolates_failures():
    pipeline = TicketClassificationPipeline(StubClassifier())

    results = list(
        pipeline.run_batch(
            [("a", "Monitor broken"), ("b", "   "), ("c", "Mouse broken")], max_workers=1
        )
    )

    assert [r.ticket_id for r in results] == ["a", "b", "c"]
    assert isinstance(results[1], FailedTicket)
    assert [r.classification.category for r in results if isinstance(r, ClassifiedTicket)] == [
        "Hardware",
        "Hardware",
    ]


def test_run_batch_records_a_backend_failure_without_stopping():
    classifier = StubClassifier(ClassificationError("model went sideways"), HARDWARE)
    pipeline = TicketClassificationPipeline(classifier)

    results = list(pipeline.run_batch([("a", "Monitor broken"), ("b", "Mouse broken")], 1))

    assert isinstance(results[0], FailedTicket)
    assert "sideways" in results[0].error
    assert isinstance(results[1], ClassifiedTicket)


def test_classified_ticket_to_dict_is_flat():
    pipeline = TicketClassificationPipeline(StubClassifier())

    row = pipeline.run("My monitor is broken!", ticket_id="T-3").to_dict()

    assert row == {
        "ticket_id": "T-3",
        "category": "Hardware",
        "justification": "Broken monitor needs replacement.",
        "confidence": 0.95,
    }
