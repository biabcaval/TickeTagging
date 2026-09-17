"""RAG backend: classify tickets by nearest-neighbor vote over a Chroma knowledge base.

Everything specific to talking to Chroma Cloud lives here. The shared core
(``models``, ``pipeline``) knows nothing about vector stores or embeddings.
Unlike the zero-shot backend, there is no fixed taxonomy: the set of possible
categories is whatever has already been ingested into the knowledge base.
"""

from .classifier import RAGTicketClassifier, classifier_from_env
from .config import Settings
from .store import ChromaTicketStore, CollectionClient, Neighbor, store_from_env

__all__ = [
    "ChromaTicketStore",
    "CollectionClient",
    "Neighbor",
    "RAGTicketClassifier",
    "Settings",
    "classifier_from_env",
    "store_from_env",
]
