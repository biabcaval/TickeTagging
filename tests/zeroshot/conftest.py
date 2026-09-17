from __future__ import annotations

import pytest

from ticketag.zeroshot import Settings
from ticketag.zeroshot.inference import ProviderError


class FakeTransport:
    """Stands in for the Gemini transport, replaying scripted replies.

    A str is a normal reply, an Exception is raised as-is, and `None` simulates
    the model returning no usable text at all (mapped to a retryable ProviderError,
    matching what `GenAITransport` does for an empty response).
    """

    def __init__(self, *replies: str | None | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str, str, dict]] = []

    def send(self, model: str, system_instruction: str, user_content: str, config: dict) -> str:
        self.calls.append((model, system_instruction, user_content, config))
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, Exception):
            raise reply
        if not reply:
            raise ProviderError(
                "Model returned an empty reply (finish_reason=None)", retryable=True
            )
        return reply


def provider_error(
    status: int, message: str = "boom", *, retryable: bool | None = None
) -> ProviderError:
    """The error GenAITransport raises for a failed API call."""
    return ProviderError(message, status, retryable=retryable)


@pytest.fixture
def settings() -> Settings:
    # backoff_seconds=0.0 and no throttling keep the suite instant.
    return Settings(
        api_key="test-gemini-key", max_retries=3, backoff_seconds=0.0, requests_per_minute=0
    )
