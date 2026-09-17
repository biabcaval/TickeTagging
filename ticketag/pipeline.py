"""End-to-end flow: ticket text in, category and justification out.

Backend-agnostic on purpose. The pipeline bounds the text, drives whatever
classifier it was handed, and keeps a batch alive when individual tickets fail.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor

from .exceptions import EmptyTicketError, TickeTagError
from .models import ClassifiedTicket, FailedTicket
from .protocols import TicketClassifier
from .text import MAX_TICKET_CHARS, bound_ticket

logger = logging.getLogger(__name__)


class TicketClassificationPipeline:
    """Runs one classification backend over single tickets or whole batches."""

    def __init__(
        self,
        classifier: TicketClassifier,
        max_chars: int = MAX_TICKET_CHARS,
    ) -> None:
        self._classifier = classifier
        self._max_chars = max_chars

    def run(self, text: str, ticket_id: str = "") -> ClassifiedTicket:
        """Classify a single ticket."""
        bounded = bound_ticket(text, self._max_chars)
        if not bounded:
            raise EmptyTicketError(f"Ticket {ticket_id or '<unnamed>'} has no usable text")
        classification = self._classifier.classify(bounded)
        logger.debug(
            "Classified ticket=%s as %s (confidence=%.2f)",
            ticket_id or "<unnamed>",
            classification.category,
            classification.confidence,
        )
        return ClassifiedTicket(ticket_id, bounded, classification)

    def run_batch(
        self,
        tickets: Iterable[tuple[str, str]],
        max_workers: int = 4,
    ) -> Iterator[ClassifiedTicket | FailedTicket]:
        """Classify (ticket_id, text) pairs concurrently, preserving input order."""

        def worker(item: tuple[str, str]) -> ClassifiedTicket | FailedTicket:
            ticket_id, text = item
            try:
                return self.run(text, ticket_id)
            except TickeTagError as error:
                logger.error("Ticket %s failed: %s", ticket_id or "<unnamed>", error)
                return FailedTicket(ticket_id, str(error))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            yield from pool.map(worker, tickets)
