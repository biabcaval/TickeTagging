# RAG backend

> **Status: main backend.** Evaluated at 85.5% accuracy / 0.844 macro F1 on a 200-ticket
> stratified holdout (see `notebooks/backend_evaluation.ipynb`), with 0 classification
> failures and well-tracking confidence. The zero-shot backend is parked as a future
> experiment after repeated provider instability — see `Escolhas_projeto.md`'s
> "Current status" section.

Classifies a ticket by embedding it, retrieving the `k` most similar tickets already
stored in a Chroma Cloud collection, and returning the majority category among those
neighbors. There is no LLM call in this path. Embeddings use Chroma's default local
ONNX `all-MiniLM-L6-v2` model, so no API key or network round trip is needed to turn
text into a vector — only the Chroma Cloud query itself goes over the network.

Predictions at or above `auto_add_threshold` (vote share, default `0.8`) are written
back into the collection with their predicted label, so the knowledge base grows from
its own confident predictions without a human in the loop.

## Files

- `config.py` — `Settings`, loaded from env vars (`CHROMA_API_KEY`, `TICKETAG_RAG_K`,
  `TICKETAG_RAG_AUTO_ADD_THRESHOLD`, etc.).
- `store.py` — `ChromaTicketStore`: query/add/count against a Chroma collection pinned
  to cosine distance.
- `classifier.py` — `RAGTicketClassifier`: runs the vote, breaks ties, writes back
  high-confidence predictions.

## Approach and why

A ticket's category is decided by nearest-neighbor vote, not by asking a model to
reason about it. This was chosen over a zero-shot/few-shot LLM call for this backend
specifically because:

- **The taxonomy here is open, not fixed.** The set of valid categories is whatever
  labels already exist in the collection. Adding a new category means ingesting a few
  labeled examples of it, not touching code or a prompt.
- **No inference cost or latency per ticket.** A vector query against Chroma Cloud is
  cheaper and faster than a chat completion call, and has no rate limit tied to a free
  model tier.
- **The knowledge base self-improves.** Every high-confidence classification is fed
  back in, so the collection grows denser around common ticket patterns over time
  without manual re-labeling.
- **Deterministic and auditable.** The justification is a plain statement of the vote
  ("4/5 nearest tickets... avg similarity 0.87"), not a generated sentence that has to
  be trusted at face value.

The cost of this approach is that it needs a labeled seed set before it is useful at
all — see `scripts/holdout_test_sample.py` and `ticketag-rag-ingest` for how the
current CSV is loaded and how a held-out test slice is kept out of the collection.

## Main tradeoffs

- **Cold start.** An empty or thin collection produces poor or empty votes
  (`classify()` raises if the collection has zero neighbors). This backend is only as
  good as its seed data, unlike the zero-shot backend which works from category
  names alone.
- **Auto-add can compound errors.** A confidently wrong vote gets written back as if
  it were a confirmed label, and that wrong label can then win future votes. The
  threshold (`0.8`) reduces this but does not eliminate it — there is no periodic
  audit or decay built in.
- **Vote quality depends on neighbor quality**, not on any actual understanding of
  the ticket. Two tickets that are lexically/semantically close under MiniLM but
  differ in the detail that actually matters for triage will still vote together.
- **Fixed embedding model.** `all-MiniLM-L6-v2` is a general-purpose sentence
  encoder, not tuned to this ticketing domain's vocabulary or abbreviations. It is
  not swappable without also re-embedding (re-ingesting) the whole collection.
- **Tie-breaking is arbitrary.** When two categories tie for the top vote, the winner
  is whichever appears first among the neighbors sorted by distance — a heuristic,
  not a principled resolution.
- **External dependency.** Chroma Cloud's free tier has capacity and availability
  limits; this backend cannot run fully offline as currently wired (`chromadb.CloudClient`
  is hardcoded in `store_from_env`).
- **`k` and the auto-add threshold are static**, set once per run, not tuned per query
  or per category. A category with few examples and one with thousands get the same
  `k`.

## What I would do with more time

- Run `scripts/holdout_test_sample.py` output through the classifier and report
  precision/recall per category, then use that to pick `k` and the auto-add threshold
  instead of the current defaults, which are unvalidated.
- Weight votes by similarity instead of a flat per-neighbor count, so a near-identical
  match outweighs four mediocre matches.
- Add a periodic audit: sample a fraction of auto-added rows for human review instead
  of trusting the threshold indefinitely, to catch drift before it compounds.
- Add near-duplicate detection on ingest so the collection does not accumulate many
  near-identical vectors that just reinforce one existing pattern.
- Try a domain-adapted or larger embedding model and measure whether it changes vote
  accuracy enough to justify losing the "no network call" property.
- Add a local (non-Cloud) Chroma persistence path for offline development and testing,
  falling back to Cloud only when configured.
- Turn the retrieved neighbors' text into a real justification (e.g. show the closest
  matching ticket's snippet) instead of only the vote-share statistic.
