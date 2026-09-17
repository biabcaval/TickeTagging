"""Zero-shot backend: a chat LLM picks the category and writes the rationale.

Everything specific to talking to a hosted model lives here. The shared core
(``models``, ``taxonomy``, ``pipeline``) knows nothing about prompts or HTTP.
"""

from .classifier import ZeroShotTicketClassifier, classifier_from_env
from .config import Settings
from .inference import ChatBackend, HttpTransport, ProviderError, Transport
from .prompts import build_messages

__all__ = [
    "ChatBackend",
    "HttpTransport",
    "ProviderError",
    "Settings",
    "Transport",
    "ZeroShotTicketClassifier",
    "build_messages",
    "classifier_from_env",
]
