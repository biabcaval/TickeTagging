# RAG Classification Backend — Design

Date: 2026-09-17
Status: Approved for implementation

## 1. Purpose

Add a third classification backend, `ticketag.rag`, alongside the existing
zero-shot LLM backend. This is "Version 3" from `Escolhas_projeto.md`: a
knowledge base of previously labeled tickets stored as embeddings in Chroma
Cloud, classified by nearest-neighbor vote instead of an LLM call.

Goals:

- Store the existing CSV export (`all_tickets_processed_improved_v3.csv`,
  columns `Document` + `Topic_group`) as a searchable knowledge base in Chroma
  Cloud (free tier).
- Classify new tickets by retrieving the most similar past tickets and voting
  on their category — no LLM call, no per-request cost.
- Support future incoming tickets: high-confidence predictions are written
  back into the knowledge base automatically; low-confidence ones are left for
  a human to confirm and ingest explicitly.
- Slot into the existing `TicketClassificationPipeline` unchanged, by
  implementing the same `TicketClassifier` protocol the zero-shot backend
  implements.

Non-goals:

- No LLM is involved in producing the category or justification (decided:
  pure k-NN vote, not retrieval-augmented *generation*).
- No fixed taxonomy: unlike zero-shot, the RAG backend does not take a closed
  category list at classify time. The set of possible categories is whatever
  already exists in the knowledge base.
- No UI/queueing system for the human-confirmation flow below the
  auto-add threshold — that's future work; this backend only exposes the
  store's `add` operation for that flow to call into.

## 2. Package layout

```
ticketag/
  envfile.py           # NEW — extracted from zeroshot/config.py, shared .env loader
  exceptions.py         # + KnowledgeBaseError
  rag/
    __init__.py          # public exports
    config.py            # Settings, Settings.from_env()
    store.py             # ChromaTicketStore, Neighbor, CollectionClient protocol
    classifier.py         # RAGTicketClassifier, classifier_from_env()
  cli/
    rag.py                # `ticketag-rag` entrypoint (classify)
    rag_ingest.py          # `ticketag-rag-ingest` entrypoint (bulk load)

tests/
  rag/
    __init__.py
    conftest.py           # FakeCollection double, Settings fixture
    test_config.py
    test_store.py
    test_classifier.py
    test_cli.py
    test_ingest_cli.py
```

`ticketag.rag` follows the same isolation rule as `ticketag.zeroshot`: the
core package (`models`, `protocols`, `pipeline`, `taxonomy`, `text`,
`exceptions`) never imports it, and `ticketag/rag/` never imports
`ticketag/zeroshot/`. Both backends import the shared `ticketag/envfile.py`.

### Refactor: extract `load_env_file`

`load_env_file`/`ENV_FILE` currently live in `ticketag/zeroshot/config.py`.
`ticketag/rag/config.py` needs identical `.env`-loading behavior. Rather than
duplicate it or have `rag` depend on `zeroshot`, extract it into a new core
module `ticketag/envfile.py`. `zeroshot/config.py` re-exports from there so
its existing public import path (`from .config import ...` inside the
package; nothing external imports `load_env_file` directly today) keeps
working.

## 3. Chroma store (`ticketag/rag/store.py`)

### Connection

```python
chromadb.CloudClient(
    api_key=settings.chroma_api_key,
    tenant=settings.chroma_tenant,     # optional, auto-resolved if the key is scoped to one DB
    database=settings.chroma_database, # optional, same
)
```

### Collection

- Created (or fetched) with `get_or_create_collection(name, metadata={"hnsw:space": "cosine"})`.
  Cosine is the correct metric for MiniLM sentence embeddings; Chroma's
  collection default is L2, so this is set explicitly rather than relied on.
- No embedding function is passed — Chroma's default local ONNX
  `all-MiniLM-L6-v2` embedding function is used automatically. It runs on
  CPU, requires no API key, and downloads its model files once
  (`~/.cache/chroma/onnx_models/`).

### `CollectionClient` protocol

A minimal `Protocol` (mirroring the existing `Transport` protocol in
`zeroshot/inference.py`) covering only the collection methods the store
calls: `add`, `upsert`, `query`, `count`. This lets tests inject a
`FakeCollection` and never touch the network or the ONNX model, exactly like
`FakeTransport` stands in for real HTTP today.

### `Neighbor`

```python
@dataclass(frozen=True, slots=True)
class Neighbor:
    ticket_id: str
    category: str
    similarity: float  # 1 - cosine_distance, so higher is closer
```

### `ChromaTicketStore` methods

- `query(text: str, k: int) -> list[Neighbor]` — calls
  `collection.query(query_texts=[text], n_results=k, include=["metadatas", "distances"])`
  and maps the result rows into `Neighbor`s, sorted nearest-first (Chroma
  already returns them in that order).
- `add(ticket_id: str, text: str, category: str) -> None` — single upsert.
- `add_many(rows: Iterable[tuple[str, str, str]], batch_size: int = 200) -> int` —
  chunks `(ticket_id, text, category)` rows into batches of `batch_size` and
  upserts each batch (`collection.upsert(ids=..., documents=..., metadatas=...)`),
  returning the total rows written. Batching matters because the sample CSV
  has ~47.8k rows and a single request that size is impractical.
- `count() -> int` — `collection.count()`, useful for CLI progress/sanity
  output.
- Every method catches the underlying Chroma/HTTP exceptions and re-raises as
  `KnowledgeBaseError`, the same way `HttpTransport` maps transport failures
  onto `ProviderError`/`InferenceError` today.

Upsert (not add) is used everywhere so re-running ingestion or re-adding an
already-known ticket is idempotent instead of erroring on duplicate ids.

## 4. Classifier (`ticketag/rag/classifier.py`)

```python
class RAGTicketClassifier:
    def __init__(self, store: ChromaTicketStore, k: int = 5, auto_add_threshold: float = 0.8): ...
    def classify(self, ticket: str) -> Classification: ...
```

`classify()`:

1. `neighbors = store.query(ticket, k)`.
2. If `neighbors` is empty → raise
   `ClassificationError("knowledge base is empty; run ticketag-rag-ingest first")`.
3. Tally votes per `category` among the neighbors.
4. Pick the category with the most votes. **Tie-break**: among categories
   tied for the most votes, pick the one containing the single nearest
   neighbor (lowest distance / highest similarity, i.e. `neighbors[0]`'s
   category if it's among the tied set, otherwise the tied category whose
   best-ranked member is closest).
5. `confidence = winning_votes / len(neighbors)` (vote share, e.g. `4/5 = 0.8`).
6. `justification` is a fixed template, never quoting neighbor ticket text
   (only the vote count and similarity score, to avoid leaking past ticket
   content). `avg_similarity` is the mean similarity of only the neighbors
   that voted for the winning category (not all k), since that's the number
   that actually supports the stated category:
   `f"{winning_votes}/{len(neighbors)} nearest tickets (avg similarity {avg_similarity:.2f}) were classified as {category!r}."`
7. Build `Classification(category, justification, confidence)`.
8. If `confidence >= self._auto_add_threshold`: upsert the ticket back into
   the store under a content-hash id
   (`hashlib.sha256(ticket.encode()).hexdigest()[:16]`) with the *predicted*
   category, so the knowledge base grows unattended only when the prediction
   is trustworthy. The hash id means re-classifying identical text is a no-op
   upsert rather than a duplicate. Failures to write back are logged as a
   warning and do not fail the classification (the caller still gets their
   `Classification`).

`classifier_from_env(store: ChromaTicketStore | None = None, **overrides) -> RAGTicketClassifier`
builds `Settings.from_env()`, then a `store_from_env(settings)` if no store
was given, mirroring `zeroshot.classifier_from_env`.

## 5. Settings (`ticketag/rag/config.py`)

| Field | Env var | Default |
|---|---|---|
| `chroma_api_key` | `CHROMA_API_KEY` | required |
| `chroma_tenant` | `CHROMA_TENANT` | `None` (auto-resolved) |
| `chroma_database` | `CHROMA_DATABASE` | `None` (auto-resolved) |
| `collection_name` | `TICKETAG_RAG_COLLECTION` | `"ticketag_tickets"` |
| `k` | `TICKETAG_RAG_K` | `5` |
| `auto_add_threshold` | `TICKETAG_RAG_AUTO_ADD_THRESHOLD` | `0.8` |

Same `.env`-loading and per-field env-var-override mechanism as
`zeroshot.Settings.from_env`, reusing the extracted `envfile.load_env_file`.

`.env.example` gets a new section documenting these variables (with a comment
pointing at Chroma Cloud's free tier signup, same style as the OpenRouter
section).

## 6. CLI

### `ticketag-rag` (`ticketag/cli/rag.py`)

Same shape as `ticketag-zeroshot`: built on `build_base_parser` +
`run_pipeline` from `cli/common.py` (so `--text/--file/--csv`, `--output`,
`--workers`, `--max-chars`, `-v` all work identically). Adds:

- `--k` — override `Settings.k` for this run.
- `--collection` — override `Settings.collection_name` for this run.

It does **not** accept `--category` (no fixed taxonomy for this backend).

### `ticketag-rag-ingest` (`ticketag/cli/rag_ingest.py`)

New, RAG-only CLI for bulk-loading a labeled CSV into the knowledge base:

```
ticketag-rag-ingest --csv all_tickets_processed_improved_v3.csv \
    [--text-column Document] [--category-column Topic_group] \
    [--id-column ...] [--limit N] [--collection ...] [-v]
```

- Validates both `--text-column` and `--category-column` exist in the CSV
  header (same style as `read_csv_tickets`'s column check).
- Streams rows, batching through `store.add_many`, logging progress every
  `batch_size` rows (`INFO` level, so it's visible without `-v`) and a final
  summary (`Ingested N ticket(s) into collection 'X' (now Y total)`).
- Missing/blank category cells are skipped with a `WARNING` log line rather
  than aborting the whole run.

`pyproject.toml` additions:

```toml
[project.scripts]
ticketag-zeroshot = "ticketag.cli.zeroshot:main"
ticketag-rag = "ticketag.cli.rag:main"
ticketag-rag-ingest = "ticketag.cli.rag_ingest:main"

[project.optional-dependencies]
rag = ["chromadb>=1.0"]
```

`chromadb` is an optional extra, not a hard dependency, so installing
`ticketag` for the zero-shot backend alone stays dependency-free. Users who
want the RAG backend install `ticketag[rag]`.

## 7. Error handling

New exception in `ticketag/exceptions.py`:

```python
class KnowledgeBaseError(TickeTagError):
    """Raised when the vector store cannot be reached or returns invalid data."""
```

Raised by `ChromaTicketStore` for connection/auth/malformed-response
failures. `ClassificationError` continues to cover "the answer doesn't map to
a usable category" cases (here: empty knowledge base). Both are `TickeTagError`
subclasses, so they're already handled by the existing CLI `try/except
TickeTagError` in `cli/rag.py` and `cli/rag_ingest.py`.

## 8. Testing

Mirrors `tests/zeroshot/`, fully offline (no real Chroma connection, no ONNX
model download):

- `tests/rag/conftest.py` — `FakeCollection` (scripted `query`/`add`/`upsert`/
  `count` results, same spirit as `FakeTransport`) and a `Settings` fixture.
- `test_config.py` — env parsing, required-key validation.
- `test_store.py` — `Neighbor` mapping from Chroma's raw query response shape,
  batching behavior of `add_many`, error wrapping into `KnowledgeBaseError`.
- `test_classifier.py` — vote tally, tie-break-by-nearest-neighbor, confidence
  vote-share formula, justification text (asserting it never contains
  neighbor ticket text), auto-add-above-threshold vs. no-write-below-threshold,
  empty-knowledge-base error.
- `test_cli.py` / `test_ingest_cli.py` — argument wiring and CSV column
  validation, with `classifier_from_env`/`store_from_env` monkeypatched to
  avoid any real backend.

## 9. Out of scope for this spec

- A confirmation workflow/UI for human-reviewed low-confidence tickets (the
  store's `add` is the integration point; how a human triggers it is future
  work).
- Automated re-embedding if the embedding model ever changes.
- Deleting/retiring stale knowledge-base entries.
