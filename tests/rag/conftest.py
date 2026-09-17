"""Shared fixtures and test doubles for the RAG backend test suite."""

from __future__ import annotations

import pytest

from ticketag.rag.config import Settings
from ticketag.rag.store import Neighbor


class FakeCollection:
    """Stands in for a chromadb Collection: scripts query results, records writes.

    Pass ``raise_error`` to make every method raise it instead, simulating a
    Chroma/network failure.
    """

    def __init__(
        self,
        query_result: dict[str, list[list[object]]] | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        self.query_result = query_result or {"ids": [[]], "distances": [[]], "metadatas": [[]]}
        self.raise_error = raise_error
        self.upserted: list[tuple[list[str], list[str], list[dict[str, str]]]] = []

    def query(self, query_texts, n_results, include):
        if self.raise_error:
            raise self.raise_error
        return self.query_result

    def upsert(self, ids, documents, metadatas):
        if self.raise_error:
            raise self.raise_error
        self.upserted.append((list(ids), list(documents), list(metadatas)))

    def count(self):
        if self.raise_error:
            raise self.raise_error
        return sum(len(ids) for ids, _, _ in self.upserted)


class StubStore:
    """A store double returning scripted neighbors and recording write-backs."""

    def __init__(self, neighbors: list[Neighbor]) -> None:
        self.neighbors = neighbors
        self.added: list[tuple[str, str, str]] = []
        self.raise_on_add: Exception | None = None

    def query(self, text: str, k: int) -> list[Neighbor]:
        return self.neighbors[:k]

    def add(self, ticket_id: str, text: str, category: str) -> None:
        if self.raise_on_add:
            raise self.raise_on_add
        self.added.append((ticket_id, text, category))


@pytest.fixture
def settings() -> Settings:
    return Settings(chroma_api_key="ck-test", collection_name="tickets-test")
