"""RAG backend: vote on a category using the k most similar known tickets.

No LLM is involved. The knowledge base already holds labeled tickets; this
module only tallies neighbor votes, breaks ties, and (above a confidence
threshold) writes the new ticket back so the knowledge base keeps growing.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter

from ..exceptions import ClassificationError, KnowledgeBaseError
from ..models import Classification
from .config import Settings
from .store import ChromaTicketStore, Neighbor, store_from_env

logger = logging.getLogger(__name__)


class RAGTicketClassifier:
    """Classifies a ticket by majority vote among its nearest known neighbors."""

    def __init__(
        self,
        store: ChromaTicketStore,
        k: int = 5,
        auto_add_threshold: float = 0.8,
    ) -> None:
        self._store = store
        self._k = k
        self._auto_add_threshold = auto_add_threshold

    def classify(self, ticket: str) -> Classification:
        """Return the category and rationale for a ticket, and grow the knowledge base."""
        neighbors = self._store.query(ticket, self._k)
        if not neighbors:
            raise ClassificationError("knowledge base is empty; run ticketag-rag-ingest first")
        category = _vote(neighbors)
        winners = [neighbor for neighbor in neighbors if neighbor.category == category]
        confidence = len(winners) / len(neighbors)
        avg_similarity = sum(neighbor.similarity for neighbor in winners) / len(winners)
        justification = (
            f"{len(winners)}/{len(neighbors)} nearest tickets (avg similarity "
            f"{avg_similarity:.2f}) were classified as {category!r}."
        )
        classification = Classification(category, justification, confidence)
        if confidence >= self._auto_add_threshold:
            self._remember(ticket, category)
        return classification

    def _remember(self, ticket: str, category: str) -> None:
        """Upsert a high-confidence prediction back into the knowledge base."""
        ticket_id = hashlib.sha256(ticket.encode("utf-8")).hexdigest()[:16]
        try:
            self._store.add(ticket_id, ticket, category)
        except KnowledgeBaseError as error:
            logger.warning(
                "Could not write ticket %s back to the knowledge base: %s", ticket_id, error
            )


def _vote(neighbors: list[Neighbor]) -> str:
    """Pick the majority category, breaking ties toward the single closest neighbor."""
    counts = Counter(neighbor.category for neighbor in neighbors)
    top_count = max(counts.values())
    tied = {category for category, count in counts.items() if count == top_count}
    if len(tied) == 1:
        return next(iter(tied))
    for neighbor in neighbors:  # neighbors is sorted nearest-first
        if neighbor.category in tied:
            return neighbor.category
    raise AssertionError("unreachable: neighbors is non-empty so some category must be tied")


def classifier_from_env(
    store: ChromaTicketStore | None = None, **settings_overrides: object
) -> RAGTicketClassifier:
    """Build the classifier from environment variables, keeping Settings out of the pipeline."""
    settings = Settings.from_env(**settings_overrides)
    return RAGTicketClassifier(
        store or store_from_env(settings),
        k=settings.k,
        auto_add_threshold=settings.auto_add_threshold,
    )
