"""Prompt construction for zero-shot ticket classification."""

from __future__ import annotations

from ..taxonomy import Category

SYSTEM_PROMPT = (
    "You are a support ticket triage engine. You read one ticket and assign it to exactly "
    "one category from a closed list supplied by the caller.\n"
    "Rules:\n"
    "1. Never invent a category; copy the chosen name character by character from the list.\n"
    "2. Judge the requester's main need, not incidental words mentioned in passing.\n"
    "3. Justify the decision in at most 25 words, quoting the decisive evidence.\n"
    "4. Report confidence as a number between 0 and 1 reflecting how clear the ticket is.\n"
    "5. Answer with a single JSON object and nothing else: "
    '{"category": str, "justification": str, "confidence": float}'
)

USER_PROMPT = """Allowed categories:
{categories}

Ticket:
\"\"\"
{ticket}
\"\"\"

Return the JSON object now."""


def build_messages(ticket: str, categories: tuple[Category, ...]) -> list[dict[str, str]]:
    """Assemble the chat messages for one zero-shot classification call."""
    listing = "\n".join(category.as_prompt_line() for category in categories)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(categories=listing, ticket=ticket)},
    ]
