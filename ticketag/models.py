"""Structured data exchanged across the classification flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Classification:
    """A category decision together with its short rationale."""

    category: str
    justification: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.category.strip():
            raise ValueError("category must not be empty")
        if not self.justification.strip():
            raise ValueError("justification must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")


@dataclass(frozen=True, slots=True)
class ClassifiedTicket:
    """The end-to-end result for a single ticket."""

    ticket_id: str
    text: str
    classification: Classification

    def to_dict(self) -> dict[str, object]:
        """Flatten the result for JSON or CSV serialization."""
        return {"ticket_id": self.ticket_id, **asdict(self.classification)}


@dataclass(frozen=True, slots=True)
class FailedTicket:
    """A ticket the flow could not classify, kept so batches never lose rows."""

    ticket_id: str
    error: str

    def to_dict(self) -> dict[str, object]:
        """Flatten the failure for JSON or CSV serialization."""
        return {"ticket_id": self.ticket_id, "category": None, "error": self.error}
