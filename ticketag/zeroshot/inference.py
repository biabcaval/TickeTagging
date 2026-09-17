"""Chat completion transport for OpenRouter.

Everything here is about reaching the model reliably: one HTTP POST, a shared
request budget, and retries for the failures worth repeating. Nothing here knows
what a ticket or a category is.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from ..exceptions import InferenceError
from .config import Settings

logger = logging.getLogger(__name__)

TRANSIENT_STATUS = frozenset({408, 409, 424, 429, 500, 502, 503, 504})

# Ticket text is unredacted, so never route it to endpoints that train on prompts.
OPENROUTER_EXTRA_BODY: dict[str, Any] = {"provider": {"data_collection": "deny"}}

# OpenRouter rejects a routing list longer than this, primary model included.
MAX_ROUTED_MODELS = 3


class ProviderError(Exception):
    """A failure reported by OpenRouter or by the model provider behind it."""

    def __init__(
        self, message: str, status: int | None = None, *, retryable: bool | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = status in TRANSIENT_STATUS if retryable is None else retryable


class Transport(Protocol):
    """Sends one chat completion payload and returns the decoded JSON response."""

    def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Post the payload and return the parsed response body."""
        ...


class HttpTransport:
    """Posts chat completions to an OpenAI-compatible endpoint over plain HTTP."""

    def __init__(self, settings: Settings) -> None:
        self._url = f"{settings.base_url.rstrip('/')}/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {settings.api_token}",
            "Content-Type": "application/json",
        }
        self._timeout = settings.timeout

    def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Post the payload, mapping every transport failure onto ProviderError."""
        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise ProviderError(_http_error_message(error), error.code) from error
        except (urllib.error.URLError, TimeoutError) as error:
            # Refused connection, DNS failure or timeout: all worth another attempt.
            raise ProviderError(f"Could not reach {self._url}: {error}", retryable=True) from error
        except json.JSONDecodeError as error:
            raise ProviderError(
                f"Endpoint returned malformed JSON: {error}", retryable=True
            ) from error


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
    """Sends chat messages to the configured endpoint and returns the raw reply."""

    def __init__(self, settings: Settings, transport: Transport | None = None) -> None:
        self._settings = settings
        self._limiter = RateLimiter(settings.requests_per_minute)
        self._extra_body = _build_extra_body(settings)
        self._transport = transport or HttpTransport(settings)

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Call the chat endpoint, retrying transient failures with backoff."""
        settings = self._settings
        payload: dict[str, Any] = {
            "model": settings.model,
            "messages": messages,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
            **self._extra_body,
        }
        last_error: ProviderError | None = None
        attempt = 0
        for attempt in range(1, settings.max_retries + 1):
            try:
                self._limiter.wait()
                return _read_reply(self._transport.send(payload))
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


def _build_extra_body(settings: Settings) -> dict[str, Any]:
    """Assemble the OpenRouter-only request options: privacy and fallback routing."""
    extra = dict(OPENROUTER_EXTRA_BODY)
    if not settings.fallback_models:
        return extra
    routed = [settings.model, *settings.fallback_models]
    if len(routed) > MAX_ROUTED_MODELS:
        logger.warning(
            "OpenRouter accepts %d routed models; ignoring %s",
            MAX_ROUTED_MODELS,
            ", ".join(routed[MAX_ROUTED_MODELS:]),
        )
    extra["models"] = routed[:MAX_ROUTED_MODELS]
    return extra


def _read_reply(payload: dict[str, Any]) -> str:
    """Return the reply text, turning an in-body provider error into an exception.

    OpenRouter reports upstream failures as HTTP 200 with an ``error`` object, so
    this is the only place those become retryable exceptions instead of crashes.
    """
    choices = payload.get("choices")
    if choices:
        choice = choices[0]
        content = (choice.get("message", {}).get("content") or "").strip()
        if content:
            return content
        # A truncated answer is a budget problem the caller must fix; an empty one
        # that finished cleanly is a model hiccup that usually clears on retry.
        reason = choice.get("finish_reason")
        if reason == "length":
            raise ProviderError(
                "Model hit the token budget before answering; raise TICKETAG_MAX_TOKENS, "
                "which reasoning models also spend on hidden thinking",
                retryable=False,
            )
        raise ProviderError(
            f"Model returned an empty reply (finish_reason={reason!r})", retryable=True
        )
    error = payload.get("error")
    if not isinstance(error, dict):
        raise ProviderError(f"Response carried no choices: {error or payload!r}"[:300])
    summary = f"{error.get('message', 'provider error')} (code={error.get('code')})"
    detail = f"{summary} {_raw_detail(error)}".strip()[:300]
    raise ProviderError(detail, _as_status(error.get("code")))


def _http_error_message(error: urllib.error.HTTPError) -> str:
    """Prefer the provider's own explanation over the bare status line."""
    body = error.read().decode("utf-8", errors="replace")
    try:
        reported = json.loads(body).get("error")
    except (json.JSONDecodeError, AttributeError):
        reported = None
    if not isinstance(reported, dict):
        return f"HTTP {error.code}: {body[:200]}"
    detail = f"{reported.get('message', '')} {_raw_detail(reported)}".strip()
    return f"HTTP {error.code}: {detail or body[:200]}"


def _raw_detail(error: dict[str, Any]) -> str:
    """Pull OpenRouter's human-readable upstream explanation, when it sends one."""
    metadata = error.get("metadata")
    return metadata.get("raw", "") if isinstance(metadata, dict) else ""


def _as_status(value: object) -> int | None:
    """Coerce a provider error code to an int status when it looks like one."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
