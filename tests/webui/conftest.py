"""Shared fixtures and test doubles for the web UI test suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import ticketag.webui.app as app_module
from ticketag.models import Classification


class StubClassifier:
    """Classifier double that returns a scripted `Classification` and records calls."""

    def __init__(
        self,
        category: str = "Hardware",
        justification: str = "stub justification",
        confidence: float = 0.9,
    ) -> None:
        self.category = category
        self.justification = justification
        self.confidence = confidence
        self.received: list[str] = []

    def classify(self, ticket: str) -> Classification:
        """Record the ticket text and return the scripted classification."""
        self.received.append(ticket)
        return Classification(self.category, self.justification, self.confidence)


@pytest.fixture(autouse=True)
def _reset_pipeline_cache() -> Iterator[None]:
    """Give every test a cold pipeline cache so builds/failures never leak across tests."""
    app_module._cache = app_module._PipelineCache()
    yield


@pytest.fixture
def client() -> TestClient:
    """A `TestClient` bound to the web UI app; callers stub out any network-touching call."""
    return TestClient(app_module.app)
