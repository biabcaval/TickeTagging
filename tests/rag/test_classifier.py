"""Tests for RAGTicketClassifier: voting, tie-breaking, and write-back behavior."""

from __future__ import annotations

import pytest

from ticketag.exceptions import ClassificationError, KnowledgeBaseError
from ticketag.rag.classifier import RAGTicketClassifier, classifier_from_env
from ticketag.rag.store import Neighbor

from .conftest import StubStore

HARDWARE_NEIGHBORS = [
    Neighbor("T-1", "Hardware", 0.95),
    Neighbor("T-2", "Hardware", 0.90),
    Neighbor("T-3", "Hardware", 0.85),
    Neighbor("T-4", "Access", 0.80),
    Neighbor("T-5", "Miscellaneous", 0.75),
]


def test_classify_returns_the_majority_category_and_vote_share_confidence():
    classifier = RAGTicketClassifier(StubStore(HARDWARE_NEIGHBORS), k=5)

    result = classifier.classify("printer will not turn on")

    assert result.category == "Hardware"
    assert result.confidence == pytest.approx(0.6)


def test_classify_justification_reports_vote_and_similarity_without_quoting_text():
    classifier = RAGTicketClassifier(StubStore(HARDWARE_NEIGHBORS), k=5)

    result = classifier.classify("printer will not turn on")

    assert result.justification == (
        "3/5 nearest tickets (avg similarity 0.90) were classified as 'Hardware'."
    )
    assert "T-1" not in result.justification
    assert "printer" not in result.justification


def test_classify_breaks_ties_toward_the_single_closest_neighbor():
    tied = [
        Neighbor("T-1", "Access", 0.92),
        Neighbor("T-2", "Hardware", 0.88),
        Neighbor("T-3", "Hardware", 0.80),
        Neighbor("T-4", "Access", 0.75),
    ]
    classifier = RAGTicketClassifier(StubStore(tied), k=4)

    result = classifier.classify("ticket")

    assert result.category == "Access"  # T-1 is the single closest neighbor overall


def test_classify_raises_when_the_knowledge_base_is_empty():
    classifier = RAGTicketClassifier(StubStore([]), k=5)

    with pytest.raises(ClassificationError, match="knowledge base is empty"):
        classifier.classify("ticket")


def test_classify_writes_back_high_confidence_predictions():
    unanimous = [Neighbor(f"T-{i}", "Hardware", 0.9) for i in range(4)]
    store = StubStore(unanimous)
    classifier = RAGTicketClassifier(store, k=4, auto_add_threshold=0.8)

    result = classifier.classify("printer broken")

    assert result.confidence == pytest.approx(1.0)
    assert len(store.added) == 1
    ticket_id, text, category = store.added[0]
    assert text == "printer broken"
    assert category == "Hardware"
    assert len(ticket_id) == 16


def test_classify_does_not_write_back_low_confidence_predictions():
    store = StubStore(HARDWARE_NEIGHBORS)
    classifier = RAGTicketClassifier(store, k=5, auto_add_threshold=0.8)

    classifier.classify("printer broken")  # 3/5 = 0.6, below the 0.8 threshold

    assert store.added == []


def test_classify_uses_a_stable_content_hash_id_for_repeated_write_backs():
    store = StubStore([Neighbor("T-1", "Hardware", 1.0)] * 5)
    classifier = RAGTicketClassifier(store, k=5, auto_add_threshold=0.8)

    classifier.classify("printer broken")
    first_id = store.added[0][0]
    store.added.clear()
    classifier.classify("printer broken")

    assert store.added[0][0] == first_id


def test_classify_logs_and_continues_when_write_back_fails(caplog):
    store = StubStore([Neighbor("T-1", "Hardware", 1.0)] * 5)
    store.raise_on_add = KnowledgeBaseError("quota exceeded")
    classifier = RAGTicketClassifier(store, k=5, auto_add_threshold=0.8)

    with caplog.at_level("WARNING"):
        result = classifier.classify("printer broken")

    assert result.category == "Hardware"
    assert "quota exceeded" in caplog.text


def test_classifier_from_env_applies_the_configured_k(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_API_KEY", "ck-test")
    monkeypatch.setenv("TICKETAG_RAG_K", "3")
    neighbors = [
        Neighbor("T-1", "Hardware", 0.95),
        Neighbor("T-2", "Hardware", 0.90),
        Neighbor("T-3", "Access", 0.85),
        Neighbor("T-4", "Access", 0.80),
        Neighbor("T-5", "Access", 0.75),
    ]
    stub_store = StubStore(neighbors)
    monkeypatch.setattr("ticketag.rag.classifier.store_from_env", lambda settings: stub_store)

    classifier = classifier_from_env(env_file=tmp_path / "absent.env")

    # k=3 keeps only the first 3 neighbors (2 Hardware, 1 Access) -> Hardware wins.
    # The default k=5 would have made Access win 3-2 instead.
    assert classifier.classify("ticket").category == "Hardware"
