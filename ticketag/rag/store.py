"""Chroma-backed knowledge base: store labeled tickets, retrieve similar ones.

Everything specific to talking to Chroma lives here. The default local ONNX
``all-MiniLM-L6-v2`` embedding function is used throughout (no embedding
function is ever passed explicitly), so no API key or network call is needed
just to turn text into vectors.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import chromadb

from ..exceptions import KnowledgeBaseError
from .config import Settings

logger = logging.getLogger(__name__)

# Cosine is the metric MiniLM sentence embeddings were trained for; Chroma's
# collection default is L2, so it must be set explicitly.
COLLECTION_METADATA = {"hnsw:space": "cosine"}


@dataclass(frozen=True, slots=True)
class Neighbor:
    """One retrieved match: a past ticket and how it was labeled."""

    ticket_id: str
    category: str
    similarity: float


class CollectionClient(Protocol):
    """The subset of a Chroma collection's interface the store depends on."""

    def query(
        self, query_texts: list[str], n_results: int, include: list[str]
    ) -> dict[str, list[list[object]]]:
        """Return the n_results nearest neighbors for each text in query_texts."""
        ...

    def upsert(
        self, ids: list[str], documents: list[str], metadatas: list[dict[str, str]]
    ) -> None:
        """Insert or overwrite records by id."""
        ...

    def count(self) -> int:
        """Return the number of records in the collection."""
        ...


class ChromaTicketStore:
    """Wraps a Chroma collection with the read/write operations the RAG backend needs."""

    def __init__(self, collection: CollectionClient) -> None:
        self._collection = collection

    def query(self, text: str, k: int) -> list[Neighbor]:
        """Return the k most similar known tickets, nearest first."""
        try:
            result = self._collection.query(
                query_texts=[text], n_results=k, include=["metadatas", "distances"]
            )
        except Exception as error:
            raise KnowledgeBaseError(f"Chroma query failed: {error}") from error
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        return [
            Neighbor(str(ticket_id), str((metadata or {}).get("category", "")), 1.0 - distance)
            for ticket_id, distance, metadata in zip(ids, distances, metadatas, strict=True)
        ]

    def add(self, ticket_id: str, text: str, category: str) -> None:
        """Insert or overwrite a single labeled ticket."""
        self.add_many([(ticket_id, text, category)])

    def add_many(self, rows: Iterable[tuple[str, str, str]], batch_size: int = 200) -> int:
        """Upsert (ticket_id, text, category) rows in chunks, returning the count written."""
        total = 0
        batch: list[tuple[str, str, str]] = []
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                total += self._upsert_batch(batch)
                logger.info("Upserted %d ticket(s) so far", total)
                batch = []
        if batch:
            total += self._upsert_batch(batch)
            logger.info("Upserted %d ticket(s) so far", total)
        return total

    def count(self) -> int:
        """Return how many tickets are currently stored."""
        try:
            return self._collection.count()
        except Exception as error:
            raise KnowledgeBaseError(f"Chroma count failed: {error}") from error

    def _upsert_batch(self, batch: list[tuple[str, str, str]]) -> int:
        ids, documents, categories = (list(field) for field in zip(*batch, strict=True))
        metadatas = [{"category": category} for category in categories]
        try:
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        except Exception as error:
            raise KnowledgeBaseError(f"Chroma upsert failed: {error}") from error
        return len(batch)


def store_from_env(settings: Settings) -> ChromaTicketStore:
    """Connect to Chroma Cloud and wrap the tuned collection."""
    client = chromadb.CloudClient(
        api_key=settings.chroma_api_key,
        tenant=settings.chroma_tenant,
        database=settings.chroma_database,
    )
    collection = client.get_or_create_collection(
        name=settings.collection_name, metadata=COLLECTION_METADATA
    )
    return ChromaTicketStore(collection)
