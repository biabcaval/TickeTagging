"""Zero-shot ticket classification: build the prompt, read back a category."""

from __future__ import annotations

import difflib
import json
import logging
from typing import Any

from ..exceptions import ClassificationError
from ..models import Classification
from ..taxonomy import DEFAULT_CATEGORIES, Category
from .config import Settings
from .inference import ChatBackend, Transport
from .prompts import build_messages

logger = logging.getLogger(__name__)

MATCH_CUTOFF = 0.75
MAX_JUSTIFICATION_CHARS = 240
DEFAULT_CONFIDENCE = 0.5


class ZeroShotTicketClassifier:
    """Classifies a ticket into one of the caller-supplied categories."""

    def __init__(
        self,
        settings: Settings,
        categories: tuple[Category, ...],
        transport: Transport | None = None,
    ) -> None:
        if not categories:
            raise ValueError("at least one category is required")
        self._categories = categories
        self._names = [category.name for category in categories]
        self._backend = ChatBackend(settings, transport)

    def classify(self, ticket: str) -> Classification:
        """Return the category and rationale for a ticket."""
        reply = self._backend.complete(build_messages(ticket, self._categories))
        payload = _extract_json(reply)
        justification = str(payload.get("justification", "")).strip()
        if not justification:
            raise ClassificationError(f"Model returned no justification: {reply[:200]!r}")
        return Classification(
            category=self._resolve_category(str(payload.get("category", ""))),
            justification=justification[:MAX_JUSTIFICATION_CHARS],
            confidence=_coerce_confidence(payload.get("confidence")),
        )

    def _resolve_category(self, predicted: str) -> str:
        """Map the model output onto the allowed label set, tolerating small drifts."""
        candidate = predicted.strip().strip(".\"'")
        if not candidate:
            raise ClassificationError("Model returned an empty category")
        if candidate in self._names:
            return candidate
        folded = {name.casefold(): name for name in self._names}
        if candidate.casefold() in folded:
            return folded[candidate.casefold()]
        close = difflib.get_close_matches(candidate.casefold(), folded, n=1, cutoff=MATCH_CUTOFF)
        if close:
            logger.info("Normalized model category %r to %r", candidate, folded[close[0]])
            return folded[close[0]]
        raise ClassificationError(
            f"Model returned {candidate!r}, which is outside the allowed categories {self._names}"
        )


def classifier_from_env(
    categories: tuple[Category, ...] = DEFAULT_CATEGORIES,
    **settings_overrides: object,
) -> ZeroShotTicketClassifier:
    """Build the classifier from environment variables, keeping Settings out of the pipeline."""
    return ZeroShotTicketClassifier(Settings.from_env(**settings_overrides), categories)


def _extract_json(reply: str) -> dict[str, Any]:
    """Pull the first JSON object out of a reply that may carry prose or fences."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(reply):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(reply[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ClassificationError(f"Model reply contains no JSON object: {reply[:200]!r}")


def _coerce_confidence(value: object) -> float:
    """Normalize the reported confidence, defaulting when the model omits it."""
    try:
        confidence = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE
    if confidence > 1.0:
        confidence /= 100.0
    return min(max(confidence, 0.0), 1.0)
