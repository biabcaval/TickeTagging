"""Command line plumbing shared by every classification backend.

Argument parsing, CSV reading and result writing are identical whatever picks
the category, so each backend only adds its own flags and builds its classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

from ..exceptions import TickeTagError
from ..models import ClassifiedTicket, FailedTicket
from ..pipeline import TicketClassificationPipeline
from ..text import MAX_TICKET_CHARS

OUTPUT_FIELDS = ("ticket_id", "category", "justification", "confidence", "error")


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


def configure_logging(verbose: bool) -> None:
    """Send progress and warnings to stderr so stdout stays machine readable."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def read_csv_tickets(
    path: Path, column: str, id_column: str | None, limit: int | None
) -> Iterator[tuple[str, str]]:
    """Stream (ticket_id, text) pairs from a CSV export."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise TickeTagError(f"Column {column!r} not found in {path}; got {reader.fieldnames}")
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                return
            yield row.get(id_column or "", "") or str(index), row.get(column, "")


def write_results(results: list[ClassifiedTicket | FailedTicket], output: Path | None) -> None:
    """Emit results as JSON on stdout or as CSV/JSON on disk."""
    rows = [result.to_dict() for result in results]
    if output is None:
        json.dump(rows if len(rows) > 1 else rows[0], sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return
    if output.suffix == ".csv":
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    else:
        output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} result(s) to {output}", file=sys.stderr)


def run_pipeline(pipeline: TicketClassificationPipeline, args: argparse.Namespace) -> int:
    """Feed the selected source through the pipeline and return a shell exit code."""
    if args.csv:
        tickets = read_csv_tickets(args.csv, args.column, args.id_column, args.limit)
        results = list(pipeline.run_batch(tickets, max_workers=args.workers))
    else:
        body = args.text or args.file.read_text(encoding="utf-8")
        results = [pipeline.run(body, args.file.name if args.file else "inline")]
    if not results:
        print("error: no tickets to classify", file=sys.stderr)
        return 1
    write_results(results, args.output)
    return 1 if any(isinstance(result, FailedTicket) for result in results) else 0
