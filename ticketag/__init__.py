"""TickeTag: support ticket classification with a justification for every decision.

This package holds only what every backend shares: the taxonomy, the result
types, the size guard and the pipeline. Backends are imported explicitly, as in
``from ticketag.zeroshot import classifier_from_env``, so pulling in the core
never drags in a backend's dependencies.
"""

from .exceptions import (
    ClassificationError,
    ConfigurationError,
    EmptyTicketError,
    InferenceError,
    TickeTagError,
)
from .models import Classification, ClassifiedTicket, FailedTicket
from .pipeline import TicketClassificationPipeline
from .protocols import TicketClassifier
from .taxonomy import DEFAULT_CATEGORIES, Category, parse_categories
from .text import MAX_TICKET_CHARS, bound_ticket

__all__ = [
    "DEFAULT_CATEGORIES",
    "MAX_TICKET_CHARS",
    "Category",
    "Classification",
    "ClassificationError",
    "ClassifiedTicket",
    "ConfigurationError",
    "EmptyTicketError",
    "FailedTicket",
    "InferenceError",
    "TickeTagError",
    "TicketClassificationPipeline",
    "TicketClassifier",
    "bound_ticket",
    "parse_categories",
]
