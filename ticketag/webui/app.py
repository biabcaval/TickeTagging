"""FastAPI app: classify a ticket on demand via the RAG or zero-shot backend.

Serves one static page (``GET /``) plus a tiny JSON API (``POST
/api/classify``). Each backend's classifier is expensive or impossible to
build without configuration (the RAG backend opens a real connection to
Chroma Cloud; the zero-shot backend needs a Gemini API key), so neither is
built until a request actually needs it -- see ``_PipelineCache`` below.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..exceptions import (
    ClassificationError,
    ConfigurationError,
    EmptyTicketError,
    InferenceError,
    KnowledgeBaseError,
    TickeTagError,
)
from ..logging_setup import configure_logging
from ..pipeline import TicketClassificationPipeline
from ..protocols import TicketClassifier
from ..rag import classifier_from_env as rag_classifier_from_env
from ..zeroshot import classifier_from_env as zeroshot_classifier_from_env

ZEROSHOT_PARKED_DETAIL = (
    "Zero-shot is parked as future work and is disabled in the web UI. Use the RAG backend."
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

Backend = Literal["rag", "zeroshot"]

# Ordered so the first matching (issubclass) entry wins; see `_status_for`.
ERROR_STATUS: tuple[tuple[type[TickeTagError], int], ...] = (
    (EmptyTicketError, 400),
    (ConfigurationError, 503),
    (KnowledgeBaseError, 503),
    (InferenceError, 503),
    (ClassificationError, 502),
)


class ClassifyRequest(BaseModel):
    """Request body for ``POST /api/classify``."""

    text: str = Field(description="Raw ticket text to classify")
    backend: Backend = Field(description="Which classification backend to use")


class ClassifyResponse(BaseModel):
    """Response body for a successful classification."""

    category: str
    justification: str
    confidence: float
    backend: Backend


class ErrorResponse(BaseModel):
    """Shape of a clean, non-traceback error body returned to the frontend."""

    detail: str


def _build_classifier(backend: Backend) -> TicketClassifier:
    """Build the classifier for one backend from environment configuration."""
    if backend == "rag":
        return rag_classifier_from_env()
    return zeroshot_classifier_from_env()


class _PipelineCache:
    """Lazily builds and caches one classification pipeline per backend.

    Guarded by a lock because uvicorn can serve requests from multiple
    threads concurrently. A failed build is never cached, so one backend
    failing to configure (e.g. a missing API key) never poisons the other,
    and the next request for the same backend gets a fresh attempt.
    """

    def __init__(self) -> None:
        self._pipelines: dict[Backend, TicketClassificationPipeline] = {}
        self._lock = threading.Lock()

    def get(self, backend: Backend) -> TicketClassificationPipeline:
        """Return the cached pipeline for `backend`, building it on first use."""
        with self._lock:
            cached = self._pipelines.get(backend)
            if cached is not None:
                return cached
            pipeline = TicketClassificationPipeline(_build_classifier(backend))
            self._pipelines[backend] = pipeline
            return pipeline


_cache = _PipelineCache()

app = FastAPI(
    title="TickeTag Web UI",
    description="Classify a support ticket with the RAG or zero-shot backend.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _status_for(error: TickeTagError) -> int:
    """Map a TickeTagError subclass onto the HTTP status the frontend should see."""
    for error_type, status in ERROR_STATUS:
        if isinstance(error, error_type):
            return status
    return 500


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    """Serve the single-page UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    """Report that the server is reachable, for smoke-testing."""
    return {"status": "ok"}


@app.post(
    "/api/classify",
    response_model=ClassifyResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Classify one ticket with the requested backend and return the full result."""
    if request.backend == "zeroshot":
        raise HTTPException(status_code=403, detail=ZEROSHOT_PARKED_DETAIL)
    try:
        pipeline = _cache.get(request.backend)
        result = pipeline.run(request.text)
    except TickeTagError as error:
        logger.warning("Classification failed backend=%s: %s", request.backend, error)
        raise HTTPException(status_code=_status_for(error), detail=str(error)) from error
    classification = result.classification
    return ClassifyResponse(
        category=classification.category,
        justification=classification.justification,
        confidence=classification.confidence,
        backend=request.backend,
    )


@app.exception_handler(Exception)
def handle_unexpected_error(_request: Request, error: Exception) -> JSONResponse:
    """Log any bug that slips past TickeTagError and still answer with clean JSON.

    Without this, an uncaught exception could surface as a raw traceback to
    the browser depending on how uvicorn is configured; this keeps the
    contract (a JSON body with a ``detail`` message) uniform in every case.
    """
    logger.exception("Unexpected error handling request: %s", error)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def run(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Start the uvicorn server for the web UI."""
    configure_logging()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
