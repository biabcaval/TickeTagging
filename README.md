# TickeTag

Support-ticket classification: given ticket text, return a category, a short
justification, and a confidence score.

**Live backend: RAG** (k-NN vote over labeled tickets in Chroma Cloud).
Evaluated at 85.5% accuracy / 0.844 macro F1 on a 200-ticket stratified
holdout (`notebooks/backend_evaluation.ipynb`). Zero-shot is parked as
future work — the code stays in the repo and is tested, but the web UI
does not let you run it.

See `docs/ARCHITECTURE.md` for the classification graph and layer map.

## Repository layout

```
ticketag/
  text.py              # Prep: length guard, whitespace collapse
  pipeline.py          # Graph: bound ticket → classifier → result
  protocols.py         # TicketClassifier contract
  models.py            # Classification result types
  taxonomy.py          # Closed label set (zero-shot only)
  logging_setup.py     # Logging: stderr progress, stdout stays machine-readable
  rag/                 # RAG backend (live)
  zeroshot/            # Zero-shot backend (parked)
  webui/               # FastAPI + static UI
  cli/                 # ticketag-rag, ticketag-rag-ingest, ticketag-zeroshot
scripts/
  holdout_test_sample.py   # Prep: stratified holdout + optional Chroma purge
  evaluate.py              # Metrics: accuracy / per-class P/R/F1
notebooks/
  backend_evaluation.ipynb # Metrics: RAG vs zero-shot comparison
docs/
  ARCHITECTURE.md
tests/
```

| Concern | Path |
|---|---|
| Prep | `ticketag/text.py`, `ticketag/cli/rag_ingest.py`, `scripts/holdout_test_sample.py` |
| Graph | `ticketag/pipeline.py` |
| RAG | `ticketag/rag/` |
| Classification | `ticketag/rag/classifier.py`, `ticketag/zeroshot/`, `ticketag/models.py` |
| Metrics | `scripts/evaluate.py`, `notebooks/backend_evaluation.ipynb` |
| Logging | `ticketag/logging_setup.py` |

## Quick start

```bash
# 1. Install (RAG + web UI extras)
uv sync

# 2. Copy .env.example → .env and set CHROMA_API_KEY (and tenant/database).
#    Collection default is `tickets`.

# 3. Ingest labeled tickets (once)
uv run ticketag-rag-ingest --csv all_tickets_processed_improved_v3.csv

# 4. Classify
uv run ticketag-webui          # http://127.0.0.1:8000
uv run ticketag-rag --text "my VPN will not connect"
```

## Tests

```bash
uv run pytest
uv run ruff check .
```
