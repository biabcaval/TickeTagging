# Zero-shot backend

> **Status: parked as a future experiment**, not the main backend right now (see
> `Escolhas_projeto.md`'s "Current status" section — RAG is main). The code below is
> complete and fully tested, but live runs have hit provider reliability issues on two
> different vendors (OpenRouter's account-wide daily free quota, then Gemini serving a
> deprecated model followed by `503 high demand` on its replacements) without a single
> successful 200-ticket evaluation to compare against RAG yet. Revisit once that's
> resolved (paid tier, different model, or demand settles) rather than building on it.

Classifies a ticket by sending its text plus a closed list of category names and
descriptions to Gemini (via Google's `google-genai` SDK, against the Gemini Developer
API), and reading back a JSON object: `{"category", "justification", "confidence"}`.
There is no vector store and no training step; the categories the model can choose
from are passed in on every call (`taxonomy.py`'s `DEFAULT_CATEGORIES`, or a
caller-supplied override).

## Files

- `config.py` — `Settings`: model id, timeouts, retry/backoff, rate limit, loaded
  from `TICKETAG_*` env vars and `GEMINI_API_KEY`.
- `inference.py` — `ChatBackend`/`GenAITransport`: one `generate_content` call per
  ticket, retry on transient provider errors, per-minute rate limiting.
- `prompts.py` — the system instruction/user content template and category listing
  format.
- `classifier.py` — `ZeroShotTicketClassifier`: builds the prompt, parses the JSON
  reply, resolves the returned category name against the allowed list (exact match,
  then case-fold, then a fuzzy match with `difflib`), and raises if none match closely
  enough.

## Approach and why

A chat LLM is asked directly for the category and a short justification, instead of
building a knowledge base and voting on similarity. This is `Escolhas_projeto.md`'s
"Version 1" and was chosen as the first working backend because:

- **No labeled data is required.** It works from category names and one-line
  descriptions alone, which is the situation at the start of the project — a
  destructively preprocessed corpus with no clean label set to fine-tune or index on.
- **The taxonomy changes without touching data.** Categories are a
  `--category NAME:description` flag or a `tuple[Category, ...]` in code, not rows
  that have to be re-embedded and re-ingested.
- **The justification is generated from the ticket's content**, not derived from
  neighbor statistics — the model states which part of the ticket drove the decision.
- **Fine-tuning was not viable here.** A fine-tuned encoder would likely beat this on
  a fixed taxonomy, but that requires clean training data and hardware neither of
  which were available, per `Escolhas_projeto.md`.

Gemini (via `google-genai`) was chosen over routing through OpenRouter (the original
"Version 1" choice) for stability: OpenRouter's free-tier models rotate in and out of
availability and share a saturated free pool across every user of that vendor, which
made evaluation runs unreliable. Calling Gemini directly means one vendor, one quota to
reason about, and native support for `system_instruction` instead of hand-rolling
OpenAI-style chat messages.

## Main tradeoffs

- **Free-tier limits are per-project, not per-key-per-model.** Unlike OpenRouter's
  per-model routing, Gemini's free tier caps requests per minute *and* per day for the
  whole project; there's no automatic fallback to a second model when the quota is
  exhausted — `TICKETAG_MODEL` has to be changed by hand, and it doesn't help if the
  cap is project-wide rather than model-specific.
- **Per-ticket network round trip.** Every classification is an API call with latency,
  a shared rate limit (`TICKETAG_REQUESTS_PER_MINUTE`), and retry/backoff on failure —
  slower and, on paid usage, costlier per ticket than a vector lookup. `run_batch`
  mitigates latency with a thread pool, not by avoiding the calls.
- **No feedback loop.** Unlike the RAG backend, nothing here is learned or stored
  between calls. Correcting a wrong classification does not change future behavior;
  the only lever is editing the prompt or category descriptions.
- **Free-text output needs a manual safety net.** `_extract_json` scans the reply for
  the first valid JSON object, and `_resolve_category` fuzzy-matches the returned
  category name against the allowed list before falling back to raising
  `ClassificationError`. This works in practice but is not a guarantee the model
  cannot violate — it depends on the model roughly following instructions. Gemini's
  native JSON mode (`response_mime_type="application/json"`) could remove this need
  but isn't wired up yet.
- **Confidence is self-reported, not calibrated.** The model states a number between
  0 and 1; nothing here checks whether that number tracks actual correctness.
- **Justification is capped at 25 words** by the system instruction (and 240
  characters by `MAX_JUSTIFICATION_CHARS`), which keeps output short but also caps how
  much reasoning can be shown.
- **Privacy is partial, not structural.** The raw ticket text — unredacted — is sent
  to Google's API on every call. `Escolhas_projeto.md` flags a planned layer to strip
  personal information before the prompt is built; it is not implemented here, so any
  PII in a ticket currently reaches the model verbatim.

## What I would do with more time

- Run this backend against a labeled holdout set (see `scripts/holdout_test_sample.py`
  and `notebooks/backend_evaluation.ipynb`) to get real per-category precision/recall,
  and use that to pick a model and temperature instead of the current unvalidated
  defaults.
- Add the PII-scrubbing layer that `Escolhas_projeto.md` describes but does not
  implement, so ticket text is redacted before it reaches the prompt, independent of
  which provider is in effect.
- Switch to Gemini's native structured output (`response_mime_type: "application/json"`
  plus a response schema) to remove `_extract_json`'s manual scan-and-parse step
  entirely.
- Cache classifications for identical or near-identical ticket text, since nothing
  currently avoids re-paying for a duplicate ticket.
- Calibrate the self-reported confidence against actual outcomes from the holdout set
  (e.g. bucket by confidence and measure accuracy per bucket), and note where it is
  over- or under-confident.
- Add a hybrid path that falls back to the RAG backend (or a stronger Gemini model)
  when confidence is low, instead of accepting every low-confidence answer as final.
- Support Vertex AI as an alternative to the Gemini Developer API (same `google-genai`
  client, different auth/init) for callers who already have a GCP project instead of a
  standalone API key.
