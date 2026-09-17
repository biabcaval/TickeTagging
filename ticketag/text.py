"""Size guard applied to ticket text before any backend sees it.

Ticket text reaches the classifier as written. The only transformation is a cap
on length; it never removes words, punctuation or casing.
"""

from __future__ import annotations

import re

MAX_TICKET_CHARS = 6000
TRAILING_SPACES = re.compile(r"[ \t]+$", re.MULTILINE)
INLINE_SPACES = re.compile(r"[ \t\f\v]{2,}")
EXTRA_NEWLINES = re.compile(r"\n{3,}")


def bound_ticket(text: str, max_chars: int = MAX_TICKET_CHARS) -> str:
    """Collapse redundant spacing and cap the length, keeping every word intact."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = TRAILING_SPACES.sub("", text)
    text = INLINE_SPACES.sub(" ", text)
    text = EXTRA_NEWLINES.sub("\n\n", text).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rsplit(' ', 1)[0]} [... truncated ...]"
