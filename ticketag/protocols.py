"""The contract every classification backend implements."""

from __future__ import annotations

from typing import Protocol

from .models import Classification


class TicketClassifier(Protocol):
    """Turns one ticket body into a category plus a rationale.

    The pipeline depends on this alone, so a backend can be a hosted LLM, a local
    embedding index, or anything else able to justify its answer.
    """

    def classify(self, ticket: str) -> Classification:
        """Return the category and rationale for a ticket."""
        ...
