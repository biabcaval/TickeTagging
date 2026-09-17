from __future__ import annotations

import pytest

from ticketag.zeroshot import Settings
from ticketag.zeroshot.inference import ProviderError


class Truncated:
    """A reply the model cut short once the token budget ran out."""


class FakeTransport:
    """Stands in for the HTTP transport, replaying scripted responses.

    A str is a normal reply, a dict is one of OpenRouter's HTTP 200 bodies that
    carry a provider error instead of choices, and an exception is raised as-is.
    """

    def __init__(self, *replies: str | dict | Truncated | Exception) -> None:
        self.replies = list(replies)
        self.payloads: list[dict] = []

    def send(self, payload: dict) -> dict:
        self.payloads.append(payload)
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, dict):
            return {"error": reply}
        if isinstance(reply, Truncated):
            return {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
        return {"choices": [{"message": {"content": reply}, "finish_reason": "stop"}]}


def http_error(status_code: int) -> ProviderError:
    """The error the real transport raises for a non-2xx response."""
    return ProviderError(f"HTTP {status_code}: boom", status_code)


@pytest.fixture
def settings() -> Settings:
    # backoff_seconds=0.0 and no throttling keep the suite instant.
    return Settings(
        api_token="sk-or-test", max_retries=3, backoff_seconds=0.0, requests_per_minute=0
    )
