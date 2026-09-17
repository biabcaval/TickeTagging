r"""Carve out a held-out, category-stratified test sample from the labeled ticket CSV.

Optionally also scrubs those same rows out of a Chroma RAG collection.

Ids are derived exactly the way ``ticketag-rag-ingest`` derives them (the
0-based CSV row index, as a string, whenever ``--id-column`` isn't given), so
the ids written here line up with whatever Chroma already stored for those
rows -- purging is safe to run before, during or after an ingest, regardless
of where in the file the sampled rows actually came from.

Usage:
  # Extract 200 rows, proportional to each category's share of the full CSV
  # (the default strategy -- see select_stratified_holdout_rows).
  uv run scripts/holdout_test_sample.py \\
      --csv all_tickets_processed_improved_v3.csv \\
      --size 200 \\
      --output test_sample_200.csv

  # Also delete those 200 ids from a Chroma collection if they already
  # made it in (e.g. because an ingest without --limit ran to completion).
  uv run scripts/holdout_test_sample.py \\
      --csv all_tickets_processed_improved_v3.csv \\
      --size 200 \\
      --output test_sample_200.csv \\
      --purge-collection tickets

  # Old behavior: the last 200 rows, whatever their category mix happens to
  # be. Kept for comparison; not recommended when the CSV isn't randomly
  # ordered (this dataset drifts toward more Hardware/less Access over time).
  uv run scripts/holdout_test_sample.py \\
      --csv all_tickets_processed_improved_v3.csv \\
      --size 200 \\
      --strategy tail \\
      --output test_sample_200_tail.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

from ticketag.exceptions import TickeTagError
from ticketag.rag import Settings

logger = logging.getLogger("holdout_test_sample")

DEFAULT_SEED = 42


def select_holdout_rows(
    path: Path, text_column: str, category_column: str, size: int
) -> list[tuple[str, str, str]]:
    """Return the last ``size`` (ticket_id, text, category) rows in the CSV.

    Taking the tail does not account for how the categories are distributed
    across the file -- see ``select_stratified_holdout_rows`` for the
    recommended default. This is kept for comparison and for datasets that
    genuinely are randomly ordered, where a tail slice is a fine (and
    slightly simpler) stand-in for a random sample.
    """
    rows = _read_labeled_rows(path, text_column, category_column, skip_blank_category=False)
    if size > len(rows):
        raise TickeTagError(f"Requested {size} holdout rows but {path} only has {len(rows)}")
    if size <= 0:
        raise TickeTagError(f"--size must be positive, got {size}")
    return rows[-size:]


def compute_stratified_target_sizes(
    category_counts: Mapping[str, int], size: int
) -> dict[str, int]:
    """Split ``size`` slots across categories proportional to their share of the total.

    Uses the largest-remainder method (Hare quota): floor each category's exact
    share, then hand out the few leftover slots to the categories whose exact
    share was rounded down the most, so the totals sum to exactly ``size``
    instead of drifting from simple per-category rounding.
    """
    total = sum(category_counts.values())
    if total == 0:
        raise TickeTagError("Cannot stratify an empty or unlabeled dataset")
    exact_shares = {category: count / total * size for category, count in category_counts.items()}
    targets = {category: int(share) for category, share in exact_shares.items()}
    leftover = size - sum(targets.values())
    by_largest_remainder = sorted(
        category_counts,
        key=lambda category: exact_shares[category] - targets[category],
        reverse=True,
    )
    for category in by_largest_remainder[:leftover]:
        targets[category] += 1
    return targets


def select_stratified_holdout_rows(
    path: Path, text_column: str, category_column: str, size: int, seed: int = DEFAULT_SEED
) -> list[tuple[str, str, str]]:
    """Return a `size`-row sample whose per-category mix matches the full CSV's.

    Unlike ``select_holdout_rows``, this draws randomly from every category in
    proportion to how common it is overall (see
    ``compute_stratified_target_sizes``), so a category that is 15% of the
    dataset is ~15% of the holdout too -- rather than whatever happens to sit
    at the tail. Sampling is seeded, so the same ``(path, size, seed)`` always
    returns the same 200 tickets: that's what keeps this run, a later RAG-side
    purge, and the RAG evaluation run all looking at the identical holdout.

    Rows with a blank category are skipped -- there's no gold label to
    stratify by or to evaluate against.
    """
    rows = _read_labeled_rows(path, text_column, category_column, skip_blank_category=True)
    if size <= 0:
        raise TickeTagError(f"--size must be positive, got {size}")
    if size > len(rows):
        raise TickeTagError(
            f"Requested {size} holdout rows but {path} only has {len(rows)} labeled rows"
        )

    by_category: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for ticket_id, text, category in rows:
        by_category[category].append((ticket_id, text))

    counts = {category: len(items) for category, items in by_category.items()}
    targets = compute_stratified_target_sizes(counts, size)

    rng = random.Random(seed)
    sampled: list[tuple[str, str, str]] = []
    for category, target in targets.items():
        available = by_category[category]
        for ticket_id, text in rng.sample(available, min(target, len(available))):
            sampled.append((ticket_id, text, category))

    shortfall = size - len(sampled)
    if shortfall > 0:
        # Only possible if a category is smaller than its proportional target (e.g. a
        # tiny category combined with a large `size`). Top up from whatever's left so
        # the sample still totals `size`, rather than silently returning fewer rows.
        used_ids = {ticket_id for ticket_id, _, _ in sampled}
        remaining = [
            (ticket_id, text, category)
            for category, items in by_category.items()
            for ticket_id, text in items
            if ticket_id not in used_ids
        ]
        sampled.extend(rng.sample(remaining, shortfall))

    sampled.sort(key=lambda row: int(row[0]))
    return sampled


def _read_labeled_rows(
    path: Path, text_column: str, category_column: str, *, skip_blank_category: bool
) -> list[tuple[str, str, str]]:
    """Read every (ticket_id, text, category) row from the CSV, id = 0-based row index."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for column in (text_column, category_column):
            if column not in fields:
                raise TickeTagError(f"Column {column!r} not found in {path}; got {fields}")
        rows = []
        for index, row in enumerate(reader):
            category = (row.get(category_column) or "").strip()
            if skip_blank_category and not category:
                continue
            rows.append((str(index), row.get(text_column, ""), category))
    return rows


def write_holdout_csv(
    rows: list[tuple[str, str, str]], output: Path, text_column: str, category_column: str
) -> None:
    """Write the holdout rows to ``output`` with an explicit ``ticket_id`` column."""
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ticket_id", text_column, category_column])
        writer.writerows(rows)


def purge_ids_from_collection(ids: list[str], collection_name: str, settings: Settings) -> int:
    """Delete ``ids`` from a Chroma Cloud collection, returning how many were removed.

    Chroma's delete is a no-op for ids that were never inserted, so this is
    safe to call whether or not an in-flight ingest has reached these rows.
    """
    try:
        import chromadb  # Imported lazily: only needed for the purge path.

        client = chromadb.CloudClient(
            api_key=settings.chroma_api_key,
            tenant=settings.chroma_tenant,
            database=settings.chroma_database,
        )
        collection = client.get_collection(name=collection_name)
        before = collection.count()
        collection.delete(ids=ids)
        after = collection.count()
    except Exception as error:
        raise TickeTagError(f"Could not purge ids from {collection_name!r}: {error}") from error
    return before - after


def build_parser() -> argparse.ArgumentParser:
    """Define the flags for extracting (and optionally purging) a holdout sample."""
    parser = argparse.ArgumentParser(
        prog="holdout_test_sample",
        description="Carve out a held-out test sample and keep it out of the Chroma RAG "
        "collection.",
    )
    parser.add_argument("--csv", required=True, type=Path, help="Full labeled ticket CSV")
    parser.add_argument("--text-column", default="Document", help="CSV column with the ticket text")
    parser.add_argument(
        "--category-column", default="Topic_group", help="CSV column with the label"
    )
    parser.add_argument("--size", type=int, default=200, help="Number of tickets to hold out")
    parser.add_argument(
        "--strategy",
        choices=("stratified", "tail"),
        default="stratified",
        help="'stratified' (default) matches each category's share of the full CSV; "
        "'tail' just takes the last --size rows",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Random seed for --strategy stratified"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("test_sample_200.csv"), help="Where to write the sample"
    )
    parser.add_argument(
        "--purge-collection",
        metavar="NAME",
        help="Also delete these ids from this Chroma collection if already inserted",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the extraction (and optional purge) and return a shell exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        if args.strategy == "tail":
            rows = select_holdout_rows(args.csv, args.text_column, args.category_column, args.size)
        else:
            rows = select_stratified_holdout_rows(
                args.csv, args.text_column, args.category_column, args.size, args.seed
            )
        write_holdout_csv(rows, args.output, args.text_column, args.category_column)
        logger.info(
            "Wrote %d holdout ticket(s) to %s (strategy=%s)", len(rows), args.output, args.strategy
        )
        logger.info("Category mix: %s", dict(Counter(category for _, _, category in rows)))
        if args.purge_collection:
            settings = Settings.from_env(collection_name=args.purge_collection)
            ids = [ticket_id for ticket_id, _, _ in rows]
            removed = purge_ids_from_collection(ids, args.purge_collection, settings)
            logger.info(
                "Removed %d ticket(s) already present in collection %r",
                removed,
                args.purge_collection,
            )
    except TickeTagError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
