# RAG Classification Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ticketag.rag`, a k-NN classification backend that stores labeled tickets as embeddings in Chroma Cloud and classifies new tickets by nearest-neighbor vote — no LLM involved.

**Architecture:** `ChromaTicketStore` wraps a Chroma collection (query/upsert/count) behind a small `CollectionClient` protocol so tests never touch the network. `RAGTicketClassifier` queries the store for k neighbors, votes on a category, and implements the existing `TicketClassifier` protocol so it drops straight into `TicketClassificationPipeline` unchanged. Two new CLIs (`ticketag-rag`, `ticketag-rag-ingest`) mirror the existing `ticketag-zeroshot` CLI shape via the shared `cli/common.py` plumbing.

**Tech Stack:** Python 3.12+, `uv`, `pytest`, `ruff`, `chromadb` (Chroma Cloud client, default local ONNX `all-MiniLM-L6-v2` embeddings), stdlib `argparse`/`csv`/`hashlib`/`logging`.

**Spec:** `docs/superpowers/specs/2026-09-17-rag-backend-design.md`

## Global Constraints

- Python 3.12+; ruff `line-length = 100`, `target-version = "py312"`, lint select `E,F,I,N,UP,B,SIM,RET,D` — every new public module/class/function needs a Google-style one-line docstring (tests are exempt per `per-file-ignores`).
- Never use a bare `except:` — catch a specific type, or `except Exception as error:` immediately re-raised as a `TickeTagError` subclass.
- Tests: pytest, Arrange-Act-Assert, named `test_<what>_<expected_behavior>`, and fully offline — no real Chroma Cloud connection and no real ONNX model download anywhere in the suite. Use the `FakeCollection`/`StubStore` doubles and `monkeypatch` for `chromadb.CloudClient`.
- No LLM call anywhere in this backend — classification is a pure k-NN vote over `ChromaTicketStore.query` results.
- This backend takes no fixed taxonomy at classify time (unlike zero-shot); categories come from whatever is already in the knowledge base.
- Collections are always created with `metadata={"hnsw:space": "cosine"}` — cosine is the correct metric for MiniLM sentence embeddings, and Chroma's own default is L2.
- Never pass an explicit embedding function — rely on Chroma's default local ONNX `all-MiniLM-L6-v2`.
- Exact defaults from the spec: `k=5`, `auto_add_threshold=0.8`, ingestion `batch_size=200`.
- The auto-generated justification text must never quote neighbor ticket text (only vote counts/similarity scores).
- `chromadb` is an optional dependency (`ticketag[rag]` extra) and a `dev`-group dependency — never a hard dependency of the base `ticketag` install.
- Commit messages: Conventional Commits (`feat:`/`fix:`/`refactor:`/`docs:`/`test:`/`chore:`), imperative mood, subject line under 72 chars. Never commit `.env` or secrets.

---

## Task 1: Add `chromadb` as a dependency

**Files:**
- Modify: `pyproject.toml:11,16-17`

**Interfaces:**
- Produces: `chromadb` importable in the dev environment and declared as the `ticketag[rag]` optional extra, for every later task to build on.

- [ ] **Step 1: Add the optional extra and dev dependency**

In `pyproject.toml`, change:

```toml
# No runtime dependencies: the OpenRouter call is a plain JSON POST over urllib.
dependencies = []

[project.scripts]
ticketag-zeroshot = "ticketag.cli.zeroshot:main"

[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.6"]
```

to:

```toml
# No runtime dependencies: the OpenRouter call is a plain JSON POST over urllib.
dependencies = []

[project.optional-dependencies]
# Chroma Cloud client for the RAG backend; the zero-shot backend needs none of this.
rag = ["chromadb>=1.0"]

[project.scripts]
ticketag-zeroshot = "ticketag.cli.zeroshot:main"

[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.6", "chromadb>=1.0"]
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync`
Expected: resolves and installs `chromadb`, updates `uv.lock`.

- [ ] **Step 3: Verify chromadb imports**

Run: `uv run python -c "import chromadb; print(chromadb.__version__)"`
Expected: prints a version string (e.g. `1.5.9`) with no error.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add chromadb as an optional rag dependency"
```

---

## Task 2: Add `KnowledgeBaseError`

**Files:**
- Modify: `ticketag/exceptions.py`
- Test: `tests/test_exceptions.py`

**Interfaces:**
- Produces: `ticketag.exceptions.KnowledgeBaseError` (subclass of `TickeTagError`) — used by `ChromaTicketStore` (Task 5) for every Chroma/network failure.

- [ ] **Step 1: Write the failing test**

Create `tests/test_exceptions.py`:

```python
from __future__ import annotations

from ticketag.exceptions import KnowledgeBaseError, TickeTagError


def test_knowledge_base_error_is_a_ticketag_error():
    assert issubclass(KnowledgeBaseError, TickeTagError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_exceptions.py -v`
Expected: FAIL with `ImportError: cannot import name 'KnowledgeBaseError'`

- [ ] **Step 3: Add the exception**

In `ticketag/exceptions.py`, after the `ClassificationError` class at the end of the file, add:

```python
class KnowledgeBaseError(TickeTagError):
    """Raised when the vector store cannot be reached or returns invalid data."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_exceptions.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ticketag/exceptions.py tests/test_exceptions.py
git commit -m "feat: add KnowledgeBaseError for vector store failures"
```

---

## Task 3: Extract the shared `.env` loader into `ticketag/envfile.py`

**Files:**
- Create: `ticketag/envfile.py`
- Modify: `ticketag/zeroshot/config.py:1-19,80-89`
- Test: `tests/test_envfile.py`
- Modify: `tests/zeroshot/test_config.py:1-14,63-73`

**Interfaces:**
- Consumes: nothing new (pure refactor of existing, already-tested behavior).
- Produces: `ticketag.envfile.ENV_FILE` (`Path`), `ticketag.envfile.load_env_file(env_file: Path = ENV_FILE) -> None`. `ticketag.zeroshot.config` keeps re-exporting both names so its existing imports keep working. `ticketag/rag/config.py` (Task 4) imports from `ticketag.envfile` directly.

- [ ] **Step 1: Write the failing test for the new module**

Create `tests/test_envfile.py`:

```python
from __future__ import annotations

import os

from ticketag.envfile import load_env_file


def test_load_env_file_does_not_override_existing_variables(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nMY_TOKEN="from_file"\nMY_MODEL=file/model\n')
    monkeypatch.setenv("MY_TOKEN", "from_shell")
    monkeypatch.delenv("MY_MODEL", raising=False)

    load_env_file(env_file)

    assert os.environ["MY_TOKEN"] == "from_shell"
    assert os.environ["MY_MODEL"] == "file/model"


def test_load_env_file_is_a_no_op_when_the_file_is_absent(tmp_path):
    load_env_file(tmp_path / "absent.env")  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_envfile.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ticketag.envfile'`

- [ ] **Step 3: Create `ticketag/envfile.py` and slim down `zeroshot/config.py`**

Create `ticketag/envfile.py`:

```python
"""Shared ``.env`` file loading, used by every backend's Settings.from_env."""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(".env")


def load_env_file(env_file: Path = ENV_FILE) -> None:
    """Load KEY=VALUE pairs from a .env file without overriding real env vars."""
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
```

In `ticketag/zeroshot/config.py`, replace the top imports and the trailing `ENV_FILE`/`load_env_file` definitions. Change:

```python
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from ..exceptions import ConfigurationError

ENV_FILE = Path(".env")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
```

to:

```python
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from ..envfile import ENV_FILE, load_env_file
from ..exceptions import ConfigurationError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
```

and delete the now-duplicated definition at the bottom of the file:

```python
def load_env_file(env_file: Path = ENV_FILE) -> None:
    """Load KEY=VALUE pairs from a .env file without overriding real env vars."""
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
```

(leave everything else in `zeroshot/config.py` — `Settings`, `ENV_FIELDS`, `Settings.from_env` — untouched; `load_env_file` is now imported, not defined locally).

- [ ] **Step 4: Update the zeroshot config test's import**

In `tests/zeroshot/test_config.py`, the last test currently duplicates coverage now owned by `tests/test_envfile.py`. Change the top of the file:

```python
from __future__ import annotations

import os

import pytest

from ticketag.exceptions import ConfigurationError
from ticketag.zeroshot.config import (
    OPENROUTER_BASE_URL,
    TOKEN_ENV_VAR,
    Settings,
    load_env_file,
)
```

to:

```python
from __future__ import annotations

import pytest

from ticketag.exceptions import ConfigurationError
from ticketag.zeroshot.config import OPENROUTER_BASE_URL, TOKEN_ENV_VAR, Settings
```

(`import os` is dropped because, after this change, nothing else in the file uses it) and delete this test function from the end of the file, since its behavior is now covered by `tests/test_envfile.py`:

```python
def test_load_env_file_does_not_override_existing_variables(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(f'# comment\n{TOKEN_ENV_VAR}="from_file"\nTICKETAG_MODEL=file/model\n')
    monkeypatch.setenv(TOKEN_ENV_VAR, "from_shell")
    monkeypatch.delenv("TICKETAG_MODEL", raising=False)

    load_env_file(env_file)

    assert os.environ[TOKEN_ENV_VAR] == "from_shell"
    assert os.environ["TICKETAG_MODEL"] == "file/model"
```

- [ ] **Step 5: Run the full suite to verify nothing broke**

Run: `uv run pytest -q`
Expected: all tests PASS (zero-shot's public behavior is unchanged; `load_env_file` is now imported rather than defined).

- [ ] **Step 6: Commit**

```bash
git add ticketag/envfile.py ticketag/zeroshot/config.py tests/test_envfile.py tests/zeroshot/test_config.py
git commit -m "refactor: extract shared .env loader into ticketag.envfile"
```

---

## Task 4: `ticketag/rag/config.py` — Settings

**Files:**
- Create: `ticketag/rag/__init__.py` (empty for now, populated in Task 7)
- Create: `ticketag/rag/config.py`
- Test: `tests/rag/__init__.py`
- Test: `tests/rag/test_config.py`

**Interfaces:**
- Consumes: `ticketag.envfile.{ENV_FILE, load_env_file}` (Task 3), `ticketag.exceptions.ConfigurationError` (existing).
- Produces: `ticketag.rag.config.Settings` (dataclass: `chroma_api_key: str`, `chroma_tenant: str | None = None`, `chroma_database: str | None = None`, `collection_name: str = "ticketag_tickets"`, `k: int = 5`, `auto_add_threshold: float = 0.8`) and `Settings.from_env(cls, env_file: Path = ENV_FILE, **overrides: object) -> Self`. Task 5 (`store_from_env`) and Task 6 (`RAGTicketClassifier`) both depend on this exact `Settings` shape.

- [ ] **Step 1: Create the empty package init**

Create `ticketag/rag/__init__.py`:

```python
"""RAG backend: classify tickets by nearest-neighbor vote over a Chroma knowledge base."""
```

Create `tests/rag/__init__.py` (empty file, zero bytes — mirrors `tests/zeroshot/__init__.py`).

- [ ] **Step 2: Write the failing test**

Create `tests/rag/test_config.py`:

```python
from __future__ import annotations

import pytest

from ticketag.exceptions import ConfigurationError
from ticketag.rag.config import API_KEY_ENV_VAR, Settings


def test_settings_default_collection_and_tuning_values():
    settings = Settings(chroma_api_key="ck-test")

    assert settings.collection_name == "ticketag_tickets"
    assert settings.k == 5
    assert settings.auto_add_threshold == pytest.approx(0.8)
    assert settings.chroma_tenant is None
    assert settings.chroma_database is None


def test_settings_from_env_raises_without_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)

    with pytest.raises(ConfigurationError, match="Missing API key"):
        Settings.from_env(env_file=tmp_path / "absent.env")


def test_settings_from_env_reads_the_chroma_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "ck-abc")
    monkeypatch.setenv("CHROMA_TENANT", "tenant-1")
    monkeypatch.setenv("CHROMA_DATABASE", "db-1")

    settings = Settings.from_env(env_file=tmp_path / "absent.env")

    assert settings.chroma_api_key == "ck-abc"
    assert settings.chroma_tenant == "tenant-1"
    assert settings.chroma_database == "db-1"


def test_settings_from_env_reads_rag_tuning_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "ck-abc")
    monkeypatch.setenv("TICKETAG_RAG_COLLECTION", "custom-tickets")
    monkeypatch.setenv("TICKETAG_RAG_K", "3")
    monkeypatch.setenv("TICKETAG_RAG_AUTO_ADD_THRESHOLD", "0.9")

    settings = Settings.from_env(env_file=tmp_path / "absent.env")

    assert settings.collection_name == "custom-tickets"
    assert settings.k == 3
    assert settings.auto_add_threshold == pytest.approx(0.9)


def test_settings_from_env_rejects_an_unparsable_k(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "ck-abc")
    monkeypatch.setenv("TICKETAG_RAG_K", "many")

    with pytest.raises(ConfigurationError, match="TICKETAG_RAG_K"):
        Settings.from_env(env_file=tmp_path / "absent.env")


def test_settings_from_env_rejects_an_unparsable_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "ck-abc")
    monkeypatch.setenv("TICKETAG_RAG_AUTO_ADD_THRESHOLD", "high")

    with pytest.raises(ConfigurationError, match="TICKETAG_RAG_AUTO_ADD_THRESHOLD"):
        Settings.from_env(env_file=tmp_path / "absent.env")


def test_settings_from_env_applies_keyword_overrides_over_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "ck-abc")
    monkeypatch.setenv("TICKETAG_RAG_K", "3")

    settings = Settings.from_env(env_file=tmp_path / "absent.env", k=7)

    assert settings.k == 7
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/rag/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ticketag.rag.config'`

- [ ] **Step 4: Write the implementation**

Create `ticketag/rag/config.py`:

```python
"""Runtime settings: which Chroma Cloud database to use and how to tune retrieval."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from ..envfile import ENV_FILE, load_env_file
from ..exceptions import ConfigurationError

API_KEY_ENV_VAR = "CHROMA_API_KEY"
DEFAULT_COLLECTION_NAME = "ticketag_tickets"


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuration for connecting to Chroma Cloud and tuning the k-NN vote.

    ``chroma_tenant`` and ``chroma_database`` are optional: Chroma Cloud can
    auto-resolve both when the API key is scoped to a single database.
    """

    chroma_api_key: str
    chroma_tenant: str | None = None
    chroma_database: str | None = None
    collection_name: str = DEFAULT_COLLECTION_NAME
    k: int = 5
    auto_add_threshold: float = 0.8

    @classmethod
    def from_env(cls, env_file: Path = ENV_FILE, **overrides: object) -> Self:
        """Build settings from environment variables, loading a local .env first."""
        load_env_file(env_file)
        api_key = os.getenv(API_KEY_ENV_VAR)
        if not api_key:
            raise ConfigurationError(
                f"Missing API key: set {API_KEY_ENV_VAR} in the environment or in .env"
            )
        values: dict[str, object] = {
            "chroma_api_key": api_key,
            "chroma_tenant": os.getenv("CHROMA_TENANT"),
            "chroma_database": os.getenv("CHROMA_DATABASE"),
        }
        collection = os.getenv("TICKETAG_RAG_COLLECTION")
        if collection:
            values["collection_name"] = collection
        raw_k = os.getenv("TICKETAG_RAG_K")
        if raw_k is not None:
            try:
                values["k"] = int(raw_k)
            except ValueError as error:
                raise ConfigurationError(f"Invalid value for TICKETAG_RAG_K: {raw_k!r}") from error
        raw_threshold = os.getenv("TICKETAG_RAG_AUTO_ADD_THRESHOLD")
        if raw_threshold is not None:
            try:
                values["auto_add_threshold"] = float(raw_threshold)
            except ValueError as error:
                raise ConfigurationError(
                    f"Invalid value for TICKETAG_RAG_AUTO_ADD_THRESHOLD: {raw_threshold!r}"
                ) from error
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)  # type: ignore[arg-type]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/rag/test_config.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
git add ticketag/rag/__init__.py ticketag/rag/config.py tests/rag/__init__.py tests/rag/test_config.py
git commit -m "feat: add RAG backend Settings and env loading"
```

---

## Task 5: `ticketag/rag/store.py` — ChromaTicketStore

**Files:**
- Create: `ticketag/rag/store.py`
- Create: `tests/rag/conftest.py`
- Test: `tests/rag/test_store.py`

**Interfaces:**
- Consumes: `ticketag.exceptions.KnowledgeBaseError` (Task 2), `ticketag.rag.config.Settings` (Task 4).
- Produces: `ticketag.rag.store.Neighbor` (frozen dataclass: `ticket_id: str`, `category: str`, `similarity: float`), `ticketag.rag.store.CollectionClient` (Protocol: `query`, `upsert`, `count`), `ticketag.rag.store.ChromaTicketStore` (`__init__(collection: CollectionClient)`, `query(text: str, k: int) -> list[Neighbor]`, `add(ticket_id: str, text: str, category: str) -> None`, `add_many(rows: Iterable[tuple[str, str, str]], batch_size: int = 200) -> int`, `count() -> int`), `ticketag.rag.store.store_from_env(settings: Settings) -> ChromaTicketStore`. Task 6 (`RAGTicketClassifier`) consumes `ChromaTicketStore`, `Neighbor`, `store_from_env`. `tests/rag/conftest.py`'s `FakeCollection` and `StubStore` are consumed by Task 6's tests too.

- [ ] **Step 1: Write the test doubles**

Create `tests/rag/conftest.py`:

```python
"""Shared fixtures and test doubles for the RAG backend test suite."""

from __future__ import annotations

import pytest

from ticketag.rag.config import Settings
from ticketag.rag.store import Neighbor


class FakeCollection:
    """Stands in for a chromadb Collection: scripts query results, records writes.

    Pass ``raise_error`` to make every method raise it instead, simulating a
    Chroma/network failure.
    """

    def __init__(
        self,
        query_result: dict[str, list[list[object]]] | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        self.query_result = query_result or {"ids": [[]], "distances": [[]], "metadatas": [[]]}
        self.raise_error = raise_error
        self.upserted: list[tuple[list[str], list[str], list[dict[str, str]]]] = []

    def query(self, query_texts, n_results, include):
        if self.raise_error:
            raise self.raise_error
        return self.query_result

    def upsert(self, ids, documents, metadatas):
        if self.raise_error:
            raise self.raise_error
        self.upserted.append((list(ids), list(documents), list(metadatas)))

    def count(self):
        if self.raise_error:
            raise self.raise_error
        return sum(len(ids) for ids, _, _ in self.upserted)


class StubStore:
    """A store double returning scripted neighbors and recording write-backs."""

    def __init__(self, neighbors: list[Neighbor]) -> None:
        self.neighbors = neighbors
        self.added: list[tuple[str, str, str]] = []
        self.raise_on_add: Exception | None = None

    def query(self, text: str, k: int) -> list[Neighbor]:
        return self.neighbors[:k]

    def add(self, ticket_id: str, text: str, category: str) -> None:
        if self.raise_on_add:
            raise self.raise_on_add
        self.added.append((ticket_id, text, category))


@pytest.fixture
def settings() -> Settings:
    return Settings(chroma_api_key="ck-test", collection_name="tickets-test")
```

- [ ] **Step 2: Write the failing tests**

Create `tests/rag/test_store.py`:

```python
"""Tests for ChromaTicketStore: mapping Chroma responses and wrapping failures."""

from __future__ import annotations

import pytest

from ticketag.exceptions import KnowledgeBaseError
from ticketag.rag.config import Settings
from ticketag.rag.store import ChromaTicketStore, store_from_env

from .conftest import FakeCollection


def test_query_maps_chroma_response_into_neighbors_nearest_first():
    collection = FakeCollection(
        query_result={
            "ids": [["T-1", "T-2"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{"category": "Hardware"}, {"category": "Access"}]],
        }
    )
    store = ChromaTicketStore(collection)

    neighbors = store.query("printer is broken", k=2)

    assert [n.ticket_id for n in neighbors] == ["T-1", "T-2"]
    assert [n.category for n in neighbors] == ["Hardware", "Access"]
    assert neighbors[0].similarity == pytest.approx(0.9)
    assert neighbors[1].similarity == pytest.approx(0.7)


def test_query_returns_no_neighbors_for_an_empty_knowledge_base():
    store = ChromaTicketStore(FakeCollection())

    assert store.query("ticket", k=5) == []


def test_query_wraps_a_backend_failure_in_knowledge_base_error():
    store = ChromaTicketStore(FakeCollection(raise_error=RuntimeError("connection refused")))

    with pytest.raises(KnowledgeBaseError, match="connection refused"):
        store.query("ticket", k=5)


def test_add_upserts_a_single_row_with_category_metadata():
    collection = FakeCollection()
    store = ChromaTicketStore(collection)

    store.add("T-1", "printer is broken", "Hardware")

    assert collection.upserted == [(["T-1"], ["printer is broken"], [{"category": "Hardware"}])]


def test_add_many_batches_rows_and_returns_the_total_written():
    collection = FakeCollection()
    store = ChromaTicketStore(collection)
    rows = [(f"T-{i}", f"ticket {i}", "Hardware") for i in range(5)]

    total = store.add_many(rows, batch_size=2)

    assert total == 5
    assert [len(ids) for ids, _, _ in collection.upserted] == [2, 2, 1]


def test_add_many_logs_progress_after_each_batch(caplog):
    store = ChromaTicketStore(FakeCollection())
    rows = [(f"T-{i}", f"ticket {i}", "Hardware") for i in range(3)]

    with caplog.at_level("INFO"):
        store.add_many(rows, batch_size=2)

    assert "Upserted 2 ticket(s) so far" in caplog.text
    assert "Upserted 3 ticket(s) so far" in caplog.text


def test_add_many_wraps_a_backend_failure_in_knowledge_base_error():
    store = ChromaTicketStore(FakeCollection(raise_error=RuntimeError("quota exceeded")))

    with pytest.raises(KnowledgeBaseError, match="quota exceeded"):
        store.add_many([("T-1", "text", "Hardware")])


def test_count_returns_the_backend_total():
    store = ChromaTicketStore(FakeCollection())
    store.add_many([("T-1", "text", "Hardware")])

    assert store.count() == 1


def test_store_from_env_creates_a_cosine_collection(monkeypatch):
    calls = {}

    class FakeCloudClient:
        def __init__(self, **kwargs):
            calls["client_kwargs"] = kwargs

        def get_or_create_collection(self, name, metadata):
            calls["name"] = name
            calls["metadata"] = metadata
            return FakeCollection()

    monkeypatch.setattr("ticketag.rag.store.chromadb.CloudClient", FakeCloudClient)
    settings = Settings(
        chroma_api_key="ck-test",
        chroma_tenant="t1",
        chroma_database="d1",
        collection_name="tickets",
    )

    store = store_from_env(settings)

    assert isinstance(store, ChromaTicketStore)
    assert calls["client_kwargs"] == {"api_key": "ck-test", "tenant": "t1", "database": "d1"}
    assert calls["name"] == "tickets"
    assert calls["metadata"] == {"hnsw:space": "cosine"}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/rag/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ticketag.rag.store'`

- [ ] **Step 4: Write the implementation**

Create `ticketag/rag/store.py`:

```python
"""Chroma-backed knowledge base: store labeled tickets, retrieve similar ones.

Everything specific to talking to Chroma lives here. The default local ONNX
``all-MiniLM-L6-v2`` embedding function is used throughout (no embedding
function is ever passed explicitly), so no API key or network call is needed
just to turn text into vectors.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import chromadb

from ..exceptions import KnowledgeBaseError
from .config import Settings

logger = logging.getLogger(__name__)

# Cosine is the metric MiniLM sentence embeddings were trained for; Chroma's
# collection default is L2, so it must be set explicitly.
COLLECTION_METADATA = {"hnsw:space": "cosine"}


@dataclass(frozen=True, slots=True)
class Neighbor:
    """One retrieved match: a past ticket and how it was labeled."""

    ticket_id: str
    category: str
    similarity: float


class CollectionClient(Protocol):
    """The subset of a Chroma collection's interface the store depends on."""

    def query(
        self, query_texts: list[str], n_results: int, include: list[str]
    ) -> dict[str, list[list[object]]]:
        """Return the n_results nearest neighbors for each text in query_texts."""
        ...

    def upsert(
        self, ids: list[str], documents: list[str], metadatas: list[dict[str, str]]
    ) -> None:
        """Insert or overwrite records by id."""
        ...

    def count(self) -> int:
        """Return the number of records in the collection."""
        ...


class ChromaTicketStore:
    """Wraps a Chroma collection with the read/write operations the RAG backend needs."""

    def __init__(self, collection: CollectionClient) -> None:
        self._collection = collection

    def query(self, text: str, k: int) -> list[Neighbor]:
        """Return the k most similar known tickets, nearest first."""
        try:
            result = self._collection.query(
                query_texts=[text], n_results=k, include=["metadatas", "distances"]
            )
        except Exception as error:
            raise KnowledgeBaseError(f"Chroma query failed: {error}") from error
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        return [
            Neighbor(str(ticket_id), str((metadata or {}).get("category", "")), 1.0 - distance)
            for ticket_id, distance, metadata in zip(ids, distances, metadatas, strict=True)
        ]

    def add(self, ticket_id: str, text: str, category: str) -> None:
        """Insert or overwrite a single labeled ticket."""
        self.add_many([(ticket_id, text, category)])

    def add_many(self, rows: Iterable[tuple[str, str, str]], batch_size: int = 200) -> int:
        """Upsert (ticket_id, text, category) rows in chunks, returning the count written."""
        total = 0
        batch: list[tuple[str, str, str]] = []
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                total += self._upsert_batch(batch)
                logger.info("Upserted %d ticket(s) so far", total)
                batch = []
        if batch:
            total += self._upsert_batch(batch)
            logger.info("Upserted %d ticket(s) so far", total)
        return total

    def count(self) -> int:
        """Return how many tickets are currently stored."""
        try:
            return self._collection.count()
        except Exception as error:
            raise KnowledgeBaseError(f"Chroma count failed: {error}") from error

    def _upsert_batch(self, batch: list[tuple[str, str, str]]) -> int:
        ids, documents, categories = (list(field) for field in zip(*batch, strict=True))
        metadatas = [{"category": category} for category in categories]
        try:
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        except Exception as error:
            raise KnowledgeBaseError(f"Chroma upsert failed: {error}") from error
        return len(batch)


def store_from_env(settings: Settings) -> ChromaTicketStore:
    """Connect to Chroma Cloud and wrap the tuned collection."""
    client = chromadb.CloudClient(
        api_key=settings.chroma_api_key,
        tenant=settings.chroma_tenant,
        database=settings.chroma_database,
    )
    collection = client.get_or_create_collection(
        name=settings.collection_name, metadata=COLLECTION_METADATA
    )
    return ChromaTicketStore(collection)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/rag/test_store.py -v`
Expected: PASS (9 tests)

- [ ] **Step 6: Commit**

```bash
git add ticketag/rag/store.py tests/rag/conftest.py tests/rag/test_store.py
git commit -m "feat: add ChromaTicketStore for the RAG knowledge base"
```

---

## Task 6: `ticketag/rag/classifier.py` — RAGTicketClassifier

**Files:**
- Create: `ticketag/rag/classifier.py`
- Test: `tests/rag/test_classifier.py`

**Interfaces:**
- Consumes: `ticketag.exceptions.{ClassificationError, KnowledgeBaseError}` (existing/Task 2), `ticketag.models.Classification` (existing), `ticketag.rag.config.Settings` (Task 4), `ticketag.rag.store.{ChromaTicketStore, Neighbor, store_from_env}` (Task 5), `tests/rag/conftest.py`'s `StubStore` (Task 5).
- Produces: `ticketag.rag.classifier.RAGTicketClassifier` (`__init__(store, k: int = 5, auto_add_threshold: float = 0.8)`, `classify(ticket: str) -> Classification` — satisfies the existing `ticketag.protocols.TicketClassifier` protocol), `ticketag.rag.classifier.classifier_from_env(store: ChromaTicketStore | None = None, **settings_overrides: object) -> RAGTicketClassifier`. Task 7 (`rag/__init__.py`) and Task 9 (`cli/rag.py`) both import these two names.

- [ ] **Step 1: Write the failing tests**

Create `tests/rag/test_classifier.py`:

```python
"""Tests for RAGTicketClassifier: voting, tie-breaking, and write-back behavior."""

from __future__ import annotations

import pytest

from ticketag.exceptions import ClassificationError, KnowledgeBaseError
from ticketag.rag.classifier import RAGTicketClassifier, classifier_from_env
from ticketag.rag.store import Neighbor

from .conftest import StubStore

HARDWARE_NEIGHBORS = [
    Neighbor("T-1", "Hardware", 0.95),
    Neighbor("T-2", "Hardware", 0.90),
    Neighbor("T-3", "Hardware", 0.85),
    Neighbor("T-4", "Access", 0.80),
    Neighbor("T-5", "Miscellaneous", 0.75),
]


def test_classify_returns_the_majority_category_and_vote_share_confidence():
    classifier = RAGTicketClassifier(StubStore(HARDWARE_NEIGHBORS), k=5)

    result = classifier.classify("printer will not turn on")

    assert result.category == "Hardware"
    assert result.confidence == pytest.approx(0.6)


def test_classify_justification_reports_vote_and_similarity_without_quoting_text():
    classifier = RAGTicketClassifier(StubStore(HARDWARE_NEIGHBORS), k=5)

    result = classifier.classify("printer will not turn on")

    assert result.justification == (
        "3/5 nearest tickets (avg similarity 0.90) were classified as 'Hardware'."
    )
    assert "T-1" not in result.justification
    assert "printer" not in result.justification


def test_classify_breaks_ties_toward_the_single_closest_neighbor():
    tied = [
        Neighbor("T-1", "Access", 0.92),
        Neighbor("T-2", "Hardware", 0.88),
        Neighbor("T-3", "Hardware", 0.80),
        Neighbor("T-4", "Access", 0.75),
    ]
    classifier = RAGTicketClassifier(StubStore(tied), k=4)

    result = classifier.classify("ticket")

    assert result.category == "Access"  # T-1 is the single closest neighbor overall


def test_classify_raises_when_the_knowledge_base_is_empty():
    classifier = RAGTicketClassifier(StubStore([]), k=5)

    with pytest.raises(ClassificationError, match="knowledge base is empty"):
        classifier.classify("ticket")


def test_classify_writes_back_high_confidence_predictions():
    unanimous = [Neighbor(f"T-{i}", "Hardware", 0.9) for i in range(4)]
    store = StubStore(unanimous)
    classifier = RAGTicketClassifier(store, k=4, auto_add_threshold=0.8)

    result = classifier.classify("printer broken")

    assert result.confidence == pytest.approx(1.0)
    assert len(store.added) == 1
    ticket_id, text, category = store.added[0]
    assert text == "printer broken"
    assert category == "Hardware"
    assert len(ticket_id) == 16


def test_classify_does_not_write_back_low_confidence_predictions():
    store = StubStore(HARDWARE_NEIGHBORS)
    classifier = RAGTicketClassifier(store, k=5, auto_add_threshold=0.8)

    classifier.classify("printer broken")  # 3/5 = 0.6, below the 0.8 threshold

    assert store.added == []


def test_classify_uses_a_stable_content_hash_id_for_repeated_write_backs():
    store = StubStore([Neighbor("T-1", "Hardware", 1.0)] * 5)
    classifier = RAGTicketClassifier(store, k=5, auto_add_threshold=0.8)

    classifier.classify("printer broken")
    first_id = store.added[0][0]
    store.added.clear()
    classifier.classify("printer broken")

    assert store.added[0][0] == first_id


def test_classify_logs_and_continues_when_write_back_fails(caplog):
    store = StubStore([Neighbor("T-1", "Hardware", 1.0)] * 5)
    store.raise_on_add = KnowledgeBaseError("quota exceeded")
    classifier = RAGTicketClassifier(store, k=5, auto_add_threshold=0.8)

    with caplog.at_level("WARNING"):
        result = classifier.classify("printer broken")

    assert result.category == "Hardware"
    assert "quota exceeded" in caplog.text


def test_classifier_from_env_applies_the_configured_k(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_API_KEY", "ck-test")
    monkeypatch.setenv("TICKETAG_RAG_K", "3")
    neighbors = [
        Neighbor("T-1", "Hardware", 0.95),
        Neighbor("T-2", "Hardware", 0.90),
        Neighbor("T-3", "Access", 0.85),
        Neighbor("T-4", "Access", 0.80),
        Neighbor("T-5", "Access", 0.75),
    ]
    stub_store = StubStore(neighbors)
    monkeypatch.setattr("ticketag.rag.classifier.store_from_env", lambda settings: stub_store)

    classifier = classifier_from_env(env_file=tmp_path / "absent.env")

    # k=3 keeps only the first 3 neighbors (2 Hardware, 1 Access) -> Hardware wins.
    # The default k=5 would have made Access win 3-2 instead.
    assert classifier.classify("ticket").category == "Hardware"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/rag/test_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ticketag.rag.classifier'`

- [ ] **Step 3: Write the implementation**

Create `ticketag/rag/classifier.py`:

```python
"""RAG backend: vote on a category using the k most similar known tickets.

No LLM is involved. The knowledge base already holds labeled tickets; this
module only tallies neighbor votes, breaks ties, and (above a confidence
threshold) writes the new ticket back so the knowledge base keeps growing.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter

from ..exceptions import ClassificationError, KnowledgeBaseError
from ..models import Classification
from .config import Settings
from .store import ChromaTicketStore, Neighbor, store_from_env

logger = logging.getLogger(__name__)


class RAGTicketClassifier:
    """Classifies a ticket by majority vote among its nearest known neighbors."""

    def __init__(
        self,
        store: ChromaTicketStore,
        k: int = 5,
        auto_add_threshold: float = 0.8,
    ) -> None:
        self._store = store
        self._k = k
        self._auto_add_threshold = auto_add_threshold

    def classify(self, ticket: str) -> Classification:
        """Return the category and rationale for a ticket, and grow the knowledge base."""
        neighbors = self._store.query(ticket, self._k)
        if not neighbors:
            raise ClassificationError("knowledge base is empty; run ticketag-rag-ingest first")
        category = _vote(neighbors)
        winners = [neighbor for neighbor in neighbors if neighbor.category == category]
        confidence = len(winners) / len(neighbors)
        avg_similarity = sum(neighbor.similarity for neighbor in winners) / len(winners)
        justification = (
            f"{len(winners)}/{len(neighbors)} nearest tickets (avg similarity "
            f"{avg_similarity:.2f}) were classified as {category!r}."
        )
        classification = Classification(category, justification, confidence)
        if confidence >= self._auto_add_threshold:
            self._remember(ticket, category)
        return classification

    def _remember(self, ticket: str, category: str) -> None:
        """Upsert a high-confidence prediction back into the knowledge base."""
        ticket_id = hashlib.sha256(ticket.encode("utf-8")).hexdigest()[:16]
        try:
            self._store.add(ticket_id, ticket, category)
        except KnowledgeBaseError as error:
            logger.warning(
                "Could not write ticket %s back to the knowledge base: %s", ticket_id, error
            )


def _vote(neighbors: list[Neighbor]) -> str:
    """Pick the majority category, breaking ties toward the single closest neighbor."""
    counts = Counter(neighbor.category for neighbor in neighbors)
    top_count = max(counts.values())
    tied = {category for category, count in counts.items() if count == top_count}
    if len(tied) == 1:
        return next(iter(tied))
    for neighbor in neighbors:  # neighbors is sorted nearest-first
        if neighbor.category in tied:
            return neighbor.category
    raise AssertionError("unreachable: neighbors is non-empty so some category must be tied")


def classifier_from_env(
    store: ChromaTicketStore | None = None, **settings_overrides: object
) -> RAGTicketClassifier:
    """Build the classifier from environment variables, keeping Settings out of the pipeline."""
    settings = Settings.from_env(**settings_overrides)
    return RAGTicketClassifier(
        store or store_from_env(settings),
        k=settings.k,
        auto_add_threshold=settings.auto_add_threshold,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/rag/test_classifier.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add ticketag/rag/classifier.py tests/rag/test_classifier.py
git commit -m "feat: add RAGTicketClassifier with k-NN vote classification"
```

---

## Task 7: `ticketag/rag/__init__.py` — public exports

**Files:**
- Modify: `ticketag/rag/__init__.py`

**Interfaces:**
- Consumes: `RAGTicketClassifier`, `classifier_from_env` (Task 6); `Settings` (Task 4); `ChromaTicketStore`, `CollectionClient`, `Neighbor`, `store_from_env` (Task 5).
- Produces: the package's public surface — `ticketag.rag.{ChromaTicketStore, CollectionClient, Neighbor, RAGTicketClassifier, Settings, classifier_from_env, store_from_env}`. Task 9 (`cli/rag.py`) imports `classifier_from_env` from here; Task 10 (`cli/rag_ingest.py`) imports `Settings` and `store_from_env` from here.

- [ ] **Step 1: Replace the placeholder module docstring with full exports**

Replace the entire contents of `ticketag/rag/__init__.py`:

```python
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
```

- [ ] **Step 2: Verify the package imports cleanly**

Run: `uv run python -c "import ticketag.rag; print(sorted(ticketag.rag.__all__))"`
Expected: prints the sorted list of the 7 names above, no error.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add ticketag/rag/__init__.py
git commit -m "feat: export the RAG backend's public API"
```

---

## Task 8: Make `--category` optional in `build_base_parser`

**Files:**
- Modify: `ticketag/cli/common.py:25-51`
- Modify: `tests/test_cli_common.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `ticketag.cli.common.build_base_parser(prog: str, description: str, *, include_categories: bool = True) -> argparse.ArgumentParser` — the new `include_categories` keyword-only parameter. Task 9 (`cli/rag.py`) calls this with `include_categories=False`; `ticketag/cli/zeroshot.py`'s existing call is unaffected (default `True`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli_common.py`:

```python
from ticketag.cli.common import build_base_parser, read_csv_tickets, write_results
```

(replace the existing `from ticketag.cli.common import read_csv_tickets, write_results` import line with the one above), then add at the end of the file:

```python
def test_build_base_parser_includes_the_category_flag_by_default():
    parser = build_base_parser("prog", "desc")

    args = parser.parse_args(["--text", "a", "--category", "Hardware"])

    assert args.categories == ["Hardware"]


def test_build_base_parser_omits_the_category_flag_when_disabled():
    parser = build_base_parser("prog", "desc", include_categories=False)

    with pytest.raises(SystemExit):
        parser.parse_args(["--text", "a", "--category", "Hardware"])
```

Also add `import pytest` to the top of the file if it is not already imported (check first — `tests/test_cli_common.py` already imports `pytest`, so this step may be a no-op).

- [ ] **Step 2: Run tests to verify the new one fails**

Run: `uv run pytest tests/test_cli_common.py -v`
Expected: `test_build_base_parser_omits_the_category_flag_when_disabled` FAILS because `build_base_parser` doesn't accept `include_categories` yet (`TypeError`); the "includes by default" test passes already.

- [ ] **Step 3: Add the parameter**

In `ticketag/cli/common.py`, change:

```python
def build_base_parser(prog: str, description: str) -> argparse.ArgumentParser:
    """Define the flags every backend accepts; callers add their own on top."""
    parser = argparse.ArgumentParser(prog=prog, description=description)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Ticket body passed inline")
    source.add_argument("--file", type=Path, help="File holding a single ticket body")
    source.add_argument("--csv", type=Path, help="CSV file holding many tickets")
    parser.add_argument("--column", default="Document", help="CSV column with the ticket text")
    parser.add_argument("--id-column", help="CSV column to use as ticket id")
    parser.add_argument("--limit", type=int, help="Classify only the first N CSV rows")
    parser.add_argument("--output", type=Path, help="Write results to this .csv or .json file")
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        metavar="NAME[:DESCRIPTION]",
        help="Override the taxonomy, repeat once per category",
    )
    parser.add_argument("--workers", type=int, default=4, help="Concurrent requests for CSV input")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=MAX_TICKET_CHARS,
        help="Truncate ticket text beyond this length to cap cost",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser
```

to:

```python
def build_base_parser(
    prog: str, description: str, *, include_categories: bool = True
) -> argparse.ArgumentParser:
    """Define the flags every backend accepts; callers add their own on top."""
    parser = argparse.ArgumentParser(prog=prog, description=description)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Ticket body passed inline")
    source.add_argument("--file", type=Path, help="File holding a single ticket body")
    source.add_argument("--csv", type=Path, help="CSV file holding many tickets")
    parser.add_argument("--column", default="Document", help="CSV column with the ticket text")
    parser.add_argument("--id-column", help="CSV column to use as ticket id")
    parser.add_argument("--limit", type=int, help="Classify only the first N CSV rows")
    parser.add_argument("--output", type=Path, help="Write results to this .csv or .json file")
    if include_categories:
        parser.add_argument(
            "--category",
            action="append",
            dest="categories",
            metavar="NAME[:DESCRIPTION]",
            help="Override the taxonomy, repeat once per category",
        )
    parser.add_argument("--workers", type=int, default=4, help="Concurrent requests for CSV input")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=MAX_TICKET_CHARS,
        help="Truncate ticket text beyond this length to cap cost",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_common.py tests/zeroshot/test_cli.py -v`
Expected: all PASS — `ticketag-zeroshot`'s parser is unaffected since it doesn't pass `include_categories`.

- [ ] **Step 5: Commit**

```bash
git add ticketag/cli/common.py tests/test_cli_common.py
git commit -m "refactor: make --category flag optional in build_base_parser"
```

---

## Task 9: `ticketag-rag` CLI

**Files:**
- Create: `ticketag/cli/rag.py`
- Test: `tests/rag/test_cli.py`
- Modify: `pyproject.toml` (`[project.scripts]`)

**Interfaces:**
- Consumes: `ticketag.rag.classifier_from_env` (Task 7), `ticketag.pipeline.TicketClassificationPipeline` (existing), `ticketag.cli.common.{build_base_parser, configure_logging, run_pipeline}` (existing/Task 8), `ticketag.exceptions.TickeTagError` (existing).
- Produces: `ticketag.cli.rag.build_parser() -> argparse.ArgumentParser`, `ticketag.cli.rag.main(argv: list[str] | None = None) -> int`, and the `ticketag-rag` console script.

- [ ] **Step 1: Write the failing tests**

Create `tests/rag/test_cli.py`:

```python
"""Tests for the ticketag-rag CLI argument parsing."""

from __future__ import annotations

import pytest

from ticketag.cli.rag import build_parser
from ticketag.text import MAX_TICKET_CHARS


def test_parser_requires_exactly_one_source():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--text", "a", "--csv", "b.csv"])


def test_parser_applies_the_default_size_guard():
    args = build_parser().parse_args(["--text", "a"])

    assert args.max_chars == MAX_TICKET_CHARS


def test_parser_accepts_the_backend_specific_k_and_collection_flags():
    args = build_parser().parse_args(["--text", "a", "--k", "3", "--collection", "custom"])

    assert args.k == 3
    assert args.collection == "custom"


def test_parser_defaults_k_and_collection_to_none():
    args = build_parser().parse_args(["--text", "a"])

    assert args.k is None
    assert args.collection is None


def test_parser_rejects_the_category_flag():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--text", "a", "--category", "Hardware"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/rag/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ticketag.cli.rag'`

- [ ] **Step 3: Write the implementation**

Create `ticketag/cli/rag.py`:

```python
"""Command line entrypoint for the RAG (nearest-neighbor) backend."""

from __future__ import annotations

import argparse
import sys

from ..exceptions import TickeTagError
from ..pipeline import TicketClassificationPipeline
from ..rag import classifier_from_env
from .common import build_base_parser, configure_logging, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Extend the shared parser with the flags only this backend understands."""
    parser = build_base_parser(
        prog="ticketag-rag",
        description="Classify support tickets by nearest-neighbor vote over a Chroma knowledge base.",
        include_categories=False,
    )
    parser.add_argument("--k", type=int, help="Neighbors to retrieve per ticket")
    parser.add_argument("--collection", help="Chroma collection name to query")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the RAG CLI and return a shell exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        classifier = classifier_from_env(k=args.k, collection_name=args.collection)
        pipeline = TicketClassificationPipeline(classifier, max_chars=args.max_chars)
        return run_pipeline(pipeline, args)
    except TickeTagError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/rag/test_cli.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Register the console script**

In `pyproject.toml`, change:

```toml
[project.scripts]
ticketag-zeroshot = "ticketag.cli.zeroshot:main"
```

to:

```toml
[project.scripts]
ticketag-zeroshot = "ticketag.cli.zeroshot:main"
ticketag-rag = "ticketag.cli.rag:main"
```

Run: `uv sync`
Expected: completes without error; `uv run ticketag-rag --help` prints the CLI help text.

- [ ] **Step 6: Commit**

```bash
git add ticketag/cli/rag.py tests/rag/test_cli.py pyproject.toml uv.lock
git commit -m "feat: add ticketag-rag CLI entrypoint"
```

---

## Task 10: `ticketag-rag-ingest` CLI

**Files:**
- Create: `ticketag/cli/rag_ingest.py`
- Test: `tests/rag/test_ingest_cli.py`
- Modify: `pyproject.toml` (`[project.scripts]`)

**Interfaces:**
- Consumes: `ticketag.rag.{Settings, store_from_env}` (Task 7), `ticketag.cli.common.configure_logging` (existing), `ticketag.exceptions.TickeTagError` (existing).
- Produces: `ticketag.cli.rag_ingest.build_parser() -> argparse.ArgumentParser`, `ticketag.cli.rag_ingest.read_labeled_csv_rows(path, text_column, category_column, id_column, limit) -> Iterator[tuple[str, str, str]]`, `ticketag.cli.rag_ingest.main(argv: list[str] | None = None) -> int`, and the `ticketag-rag-ingest` console script.

- [ ] **Step 1: Write the failing tests**

Create `tests/rag/test_ingest_cli.py`:

```python
"""Tests for the ticketag-rag-ingest CLI: CSV reading and argument parsing."""

from __future__ import annotations

import csv

import pytest

from ticketag.cli.rag_ingest import build_parser, main, read_labeled_csv_rows
from ticketag.exceptions import TickeTagError


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def test_read_labeled_csv_rows_yields_id_text_category_triples(tmp_path):
    path = tmp_path / "tickets.csv"
    write_csv(
        path,
        [["Document", "Topic_group"], ["monitor broken", "Hardware"], ["vpn down", "Access"]],
    )

    rows = list(read_labeled_csv_rows(path, "Document", "Topic_group", None, None))

    assert rows == [("0", "monitor broken", "Hardware"), ("1", "vpn down", "Access")]


def test_read_labeled_csv_rows_skips_blank_categories(tmp_path, caplog):
    path = tmp_path / "tickets.csv"
    write_csv(path, [["Document", "Topic_group"], ["mystery ticket", ""]])

    with caplog.at_level("WARNING"):
        rows = list(read_labeled_csv_rows(path, "Document", "Topic_group", None, None))

    assert rows == []
    assert "Skipping row" in caplog.text


def test_read_labeled_csv_rows_rejects_unknown_column(tmp_path):
    path = tmp_path / "tickets.csv"
    write_csv(path, [["Document", "Topic_group"], ["a", "Hardware"]])

    with pytest.raises(TickeTagError, match="not found"):
        list(read_labeled_csv_rows(path, "Body", "Topic_group", None, None))


def test_read_labeled_csv_rows_honors_the_limit(tmp_path):
    path = tmp_path / "tickets.csv"
    write_csv(
        path,
        [["Document", "Topic_group"], ["a", "Hardware"], ["b", "Access"], ["c", "Storage"]],
    )

    rows = list(read_labeled_csv_rows(path, "Document", "Topic_group", None, 2))

    assert [row[0] for row in rows] == ["0", "1"]


def test_parser_requires_the_csv_flag():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parser_applies_defaults_matching_the_sample_export():
    args = build_parser().parse_args(["--csv", "tickets.csv"])

    assert args.text_column == "Document"
    assert args.category_column == "Topic_group"


def test_main_ingests_and_prints_a_summary(tmp_path, monkeypatch, capsys):
    path = tmp_path / "tickets.csv"
    write_csv(path, [["Document", "Topic_group"], ["monitor broken", "Hardware"]])
    monkeypatch.setenv("CHROMA_API_KEY", "ck-test")

    class StubStore:
        def add_many(self, rows, batch_size):
            self.rows = list(rows)
            return len(self.rows)

        def count(self):
            return 1

    stub_store = StubStore()
    monkeypatch.setattr("ticketag.cli.rag_ingest.store_from_env", lambda settings: stub_store)

    exit_code = main(["--csv", str(path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Ingested 1 ticket(s)" in captured.out
    assert stub_store.rows == [("0", "monitor broken", "Hardware")]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/rag/test_ingest_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ticketag.cli.rag_ingest'`

- [ ] **Step 3: Write the implementation**

Create `ticketag/cli/rag_ingest.py`:

```python
"""Command line entrypoint that loads a labeled CSV into the RAG knowledge base."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

from ..exceptions import TickeTagError
from ..rag import Settings, store_from_env
from .common import configure_logging

logger = logging.getLogger(__name__)

BATCH_SIZE = 200


def build_parser() -> argparse.ArgumentParser:
    """Define the flags for bulk-loading a labeled ticket CSV."""
    parser = argparse.ArgumentParser(
        prog="ticketag-rag-ingest",
        description="Load a labeled ticket CSV into the Chroma knowledge base for the RAG backend.",
    )
    parser.add_argument("--csv", required=True, type=Path, help="CSV file holding labeled tickets")
    parser.add_argument("--text-column", default="Document", help="CSV column with the ticket text")
    parser.add_argument(
        "--category-column", default="Topic_group", help="CSV column with the label"
    )
    parser.add_argument("--id-column", help="CSV column to use as ticket id")
    parser.add_argument("--limit", type=int, help="Ingest only the first N CSV rows")
    parser.add_argument("--collection", help="Chroma collection name to write to")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser


def read_labeled_csv_rows(
    path: Path,
    text_column: str,
    category_column: str,
    id_column: str | None,
    limit: int | None,
) -> Iterator[tuple[str, str, str]]:
    """Stream (ticket_id, text, category) rows from a labeled CSV export."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for column in (text_column, category_column):
            if column not in fields:
                raise TickeTagError(f"Column {column!r} not found in {path}; got {fields}")
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                return
            category = (row.get(category_column) or "").strip()
            ticket_id = row.get(id_column or "", "") or str(index)
            if not category:
                logger.warning("Skipping row %s: blank %r", ticket_id, category_column)
                continue
            yield ticket_id, row.get(text_column, ""), category


def main(argv: list[str] | None = None) -> int:
    """Run the ingestion CLI and return a shell exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        settings = Settings.from_env(collection_name=args.collection)
        store = store_from_env(settings)
        rows = read_labeled_csv_rows(
            args.csv, args.text_column, args.category_column, args.id_column, args.limit
        )
        written = store.add_many(rows, batch_size=BATCH_SIZE)
        total = store.count()
    except TickeTagError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"Ingested {written} ticket(s) into collection {settings.collection_name!r} "
        f"(now {total} total)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/rag/test_ingest_cli.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Register the console script**

In `pyproject.toml`, change:

```toml
[project.scripts]
ticketag-zeroshot = "ticketag.cli.zeroshot:main"
ticketag-rag = "ticketag.cli.rag:main"
```

to:

```toml
[project.scripts]
ticketag-zeroshot = "ticketag.cli.zeroshot:main"
ticketag-rag = "ticketag.cli.rag:main"
ticketag-rag-ingest = "ticketag.cli.rag_ingest:main"
```

Run: `uv sync`
Expected: completes without error; `uv run ticketag-rag-ingest --help` prints the CLI help text.

- [ ] **Step 6: Commit**

```bash
git add ticketag/cli/rag_ingest.py tests/rag/test_ingest_cli.py pyproject.toml uv.lock
git commit -m "feat: add ticketag-rag-ingest CLI entrypoint"
```

---

## Task 11: Document the new environment variables

**Files:**
- Modify: `.env.example`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: nothing consumed by other tasks; this is the last piece needed for a user to actually run the backend end-to-end.

- [ ] **Step 1: Append the Chroma Cloud + RAG section**

Append to the end of `.env.example`:

```bash

# --- RAG backend (ticketag-rag / ticketag-rag-ingest) ---
# Chroma Cloud free-tier credentials from https://trychroma.com — run
# `chroma login && chroma db create <name> && chroma db connect <name> --env-file`
# to generate these three values automatically.
CHROMA_API_KEY=ck-your_key_here
CHROMA_TENANT=your-tenant-id
CHROMA_DATABASE=your-database-name

# Collection to read/write tickets in.
TICKETAG_RAG_COLLECTION=ticketag_tickets

# Nearest neighbors retrieved per ticket for the majority vote.
TICKETAG_RAG_K=5

# Minimum vote-share confidence before a prediction is written back into the
# knowledge base automatically. Below this, ingest the confirmed label by hand.
TICKETAG_RAG_AUTO_ADD_THRESHOLD=0.8
```

- [ ] **Step 2: Verify no secrets leaked**

Run: `git diff .env.example`
Expected: only placeholder values (`ck-your_key_here`, etc.) — nothing from the real local `.env`.

- [ ] **Step 3: Commit**

```bash
git add .env.example
git commit -m "docs: document Chroma Cloud env vars in .env.example"
```

---

## Task 12: Final verification

**Files:** none (verification only).

**Interfaces:** none — this task confirms every prior task's deliverable still holds together.

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests PASS, including every file under `tests/rag/` and the updated `tests/zeroshot/`, `tests/test_cli_common.py`, `tests/test_envfile.py`, `tests/test_exceptions.py`.

- [ ] **Step 2: Run the linter**

Run: `uv run ruff check .`
Expected: no findings. If ruff reports issues in any file touched by this plan (missing docstring, import order, line length, etc.), fix them in that file directly, matching the style already used elsewhere in the module, then re-run this command until it is clean.

- [ ] **Step 3: Confirm the CLIs are wired up**

Run: `uv run ticketag-rag --help` and `uv run ticketag-rag-ingest --help`
Expected: both print usage text with the flags defined in Tasks 9 and 10, with no import errors.

- [ ] **Step 4: Commit any lint fixes, if there were any**

Only if Step 2 required changes:

```bash
git add -A
git commit -m "fix: address ruff findings in the RAG backend"
```
