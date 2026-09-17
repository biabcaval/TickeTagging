"""Zero-shot backend: a chat LLM picks the category and writes the rationale.

Everything specific to talking to a hosted model lives here. The shared core
(``models``, ``taxonomy``, ``pipeline``) knows nothing about prompts or the
Gemini API.
"""

from .classifier import ZeroShotTicketClassifier, classifier_from_env
from .config import Settings
from .inference import ChatBackend, GenAITransport, ProviderError, Transport
from .prompts import build_prompt

__all__ = [
    "ChatBackend",
    "GenAITransport",
    "ProviderError",
    "Settings",
    "Transport",
    "ZeroShotTicketClassifier",
    "build_prompt",
    "classifier_from_env",
]
