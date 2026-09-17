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
