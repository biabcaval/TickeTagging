"""Tests for ChromaTicketStore: mapping Chroma responses and wrapping failures."""

from __future__ import annotations

import pytest

from ticketag.exceptions import KnowledgeBaseError
from ticketag.rag.config import Settings
from ticketag.rag.store import ChromaTicketStore, store_from_env

from .conftest import FakeCollection


def test_query_maps_chroma_response_into_neighbors_nearest_first():
    collection = FakeCollection(
        query_result={
            "ids": [["T-1", "T-2"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{"category": "Hardware"}, {"category": "Access"}]],
        }
    )
    store = ChromaTicketStore(collection)

    neighbors = store.query("printer is broken", k=2)

    assert [n.ticket_id for n in neighbors] == ["T-1", "T-2"]
    assert [n.category for n in neighbors] == ["Hardware", "Access"]
    assert neighbors[0].similarity == pytest.approx(0.9)
    assert neighbors[1].similarity == pytest.approx(0.7)


def test_query_returns_no_neighbors_for_an_empty_knowledge_base():
    store = ChromaTicketStore(FakeCollection())

    assert store.query("ticket", k=5) == []


def test_query_wraps_a_backend_failure_in_knowledge_base_error():
    store = ChromaTicketStore(FakeCollection(raise_error=RuntimeError("connection refused")))

    with pytest.raises(KnowledgeBaseError, match="connection refused"):
        store.query("ticket", k=5)


def test_add_upserts_a_single_row_with_category_metadata():
    collection = FakeCollection()
    store = ChromaTicketStore(collection)

    store.add("T-1", "printer is broken", "Hardware")

    assert collection.upserted == [(["T-1"], ["printer is broken"], [{"category": "Hardware"}])]


def test_add_many_batches_rows_and_returns_the_total_written():
    collection = FakeCollection()
    store = ChromaTicketStore(collection)
    rows = [(f"T-{i}", f"ticket {i}", "Hardware") for i in range(5)]

    total = store.add_many(rows, batch_size=2)

    assert total == 5
    assert [len(ids) for ids, _, _ in collection.upserted] == [2, 2, 1]


def test_add_many_logs_progress_after_each_batch(caplog):
    store = ChromaTicketStore(FakeCollection())
    rows = [(f"T-{i}", f"ticket {i}", "Hardware") for i in range(3)]

    with caplog.at_level("INFO"):
        store.add_many(rows, batch_size=2)

    assert "Upserted 2 ticket(s) so far" in caplog.text
    assert "Upserted 3 ticket(s) so far" in caplog.text


def test_add_many_wraps_a_backend_failure_in_knowledge_base_error():
    store = ChromaTicketStore(FakeCollection(raise_error=RuntimeError("quota exceeded")))

    with pytest.raises(KnowledgeBaseError, match="quota exceeded"):
        store.add_many([("T-1", "text", "Hardware")])


def test_count_returns_the_backend_total():
    store = ChromaTicketStore(FakeCollection())
    store.add_many([("T-1", "text", "Hardware")])

    assert store.count() == 1


def test_store_from_env_creates_a_cosine_collection(monkeypatch):
    calls = {}

    class FakeCloudClient:
        def __init__(self, **kwargs):
            calls["client_kwargs"] = kwargs

        def get_or_create_collection(self, name, metadata):
            calls["name"] = name
            calls["metadata"] = metadata
            collection = FakeCollection()
            collection.metadata = {"hnsw:space": "cosine"}
            return collection

    monkeypatch.setattr("ticketag.rag.store.chromadb.CloudClient", FakeCloudClient)
    settings = Settings(
        chroma_api_key="ck-test",
        chroma_tenant="t1",
        chroma_database="d1",
        collection_name="tickets",
    )

    store = store_from_env(settings)

    assert isinstance(store, ChromaTicketStore)
    assert calls["client_kwargs"] == {"api_key": "ck-test", "tenant": "t1", "database": "d1"}
    assert calls["name"] == "tickets"
    assert calls["metadata"] == {"hnsw:space": "cosine"}


def test_store_from_env_rejects_a_pre_existing_collection_with_a_different_metric(monkeypatch):
    class FakeCloudClient:
        def __init__(self, **kwargs):
            pass

        def get_or_create_collection(self, name, metadata):
            collection = FakeCollection()
            collection.metadata = {"hnsw:space": "l2"}
            return collection

    monkeypatch.setattr("ticketag.rag.store.chromadb.CloudClient", FakeCloudClient)
    settings = Settings(chroma_api_key="ck-test", collection_name="tickets")

    with pytest.raises(KnowledgeBaseError, match="tickets"):
        store_from_env(settings)
