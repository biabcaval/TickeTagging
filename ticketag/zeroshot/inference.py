"""Chat completion transport for Gemini, via the `google-genai` SDK.

Everything here is about reaching the model reliably: one `generate_content`
call, a shared request budget, and retries for the failures worth repeating.
Nothing here knows what a ticket or a category is.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Protocol

from ..exceptions import InferenceError
from .config import Settings

logger = logging.getLogger(__name__)

TRANSIENT_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

# Finish reasons that mean "the model produced no usable text and won't on retry".
PERMANENT_FINISH_REASONS = frozenset({"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST"})


class ProviderError(Exception):
    """A failure reported by the Gemini API or by this transport."""

    def __init__(
        self, message: str, status: int | None = None, *, retryable: bool | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = status in TRANSIENT_STATUS if retryable is None else retryable


class Transport(Protocol):
    """Sends one system+user prompt to the model and returns the reply text."""

    def send(
        self, model: str, system_instruction: str, user_content: str, config: dict[str, Any]
    ) -> str:
        """Call the model and return its reply text, raising ProviderError on failure."""
        ...


class GenAITransport:
    """Calls the Gemini Developer API through `google-genai`, mapping failures to ProviderError."""

    def __init__(self, settings: Settings) -> None:
        from google import genai
        from google.genai.types import HttpOptions

        self._client = genai.Client(
            api_key=settings.api_key, http_options=HttpOptions(timeout=int(settings.timeout * 1000))
        )

    def send(
        self, model: str, system_instruction: str, user_content: str, config: dict[str, Any]
    ) -> str:
        """Call the model and return its reply text, raising ProviderError on failure."""
        from google.genai import errors
        from google.genai.types import GenerateContentConfig

        try:
            response = self._client.models.generate_content(
                model=model,
                contents=user_content,
                config=GenerateContentConfig(system_instruction=system_instruction, **config),
            )
        except errors.APIError as error:
            raise ProviderError(str(error.message or error), error.code) from error
        return _read_reply(response)


class RateLimiter:
    """Spaces out calls so a shared per-minute request budget is never exceeded."""

    def __init__(self, requests_per_minute: int) -> None:
        self._interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def wait(self) -> None:
        """Block until the caller's turn, keeping threads from bunching up."""
        if not self._interval:
            return
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self._interval
        if delay:
            time.sleep(delay)


class ChatBackend:
    """Sends a system+user prompt to the configured model and returns the raw reply."""

    def __init__(self, settings: Settings, transport: Transport | None = None) -> None:
        self._settings = settings
        self._limiter = RateLimiter(settings.requests_per_minute)
        self._transport = transport or GenAITransport(settings)

    def complete(self, system_instruction: str, user_content: str) -> str:
        """Call the model, retrying transient failures with backoff."""
        settings = self._settings
        config: dict[str, Any] = {
            "temperature": settings.temperature,
            "max_output_tokens": settings.max_tokens,
        }
        last_error: ProviderError | None = None
        attempt = 0
        for attempt in range(1, settings.max_retries + 1):
            try:
                self._limiter.wait()
                return self._transport.send(
                    settings.model, system_instruction, user_content, config
                )
            except ProviderError as error:
                last_error = error
                if not error.retryable or attempt == settings.max_retries:
                    break
                delay = settings.backoff_seconds * 2 ** (attempt - 1)
                logger.warning(
                    "Transient inference failure on model=%s attempt=%d/%d, retrying in %.1fs: %s",
                    settings.model,
                    attempt,
                    settings.max_retries,
                    delay,
                    error,
                )
                time.sleep(delay)
        raise InferenceError(
            f"Inference failed for model={settings.model!r} after "
            f"{attempt} attempt(s): {last_error}"
        ) from last_error


def _read_reply(response: Any) -> str:
    """Return the reply text, turning a blocked/truncated/empty response into an exception."""
    block_reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
    if block_reason:
        raise ProviderError(
            f"Prompt was blocked before generation: {block_reason}", retryable=False
        )

    candidates = getattr(response, "candidates", None) or []
    finish_reason = _finish_reason_name(candidates[0]) if candidates else None
    text = (getattr(response, "text", None) or "").strip()

    if text:
        return text
    if finish_reason == "MAX_TOKENS":
        raise ProviderError(
            "Model hit the token budget before answering; raise TICKETAG_MAX_TOKENS, "
            "which reasoning models also spend on hidden thinking",
            retryable=False,
        )
    if finish_reason in PERMANENT_FINISH_REASONS:
        raise ProviderError(
            f"Model declined to answer (finish_reason={finish_reason})", retryable=False
        )
    raise ProviderError(
        f"Model returned an empty reply (finish_reason={finish_reason!r})", retryable=True
    )


def _finish_reason_name(candidate: Any) -> str | None:
    """Normalize a candidate's finish reason to its plain enum name, if present."""
    reason = getattr(candidate, "finish_reason", None)
    if reason is None:
        return None
    return getattr(reason, "name", str(reason))
