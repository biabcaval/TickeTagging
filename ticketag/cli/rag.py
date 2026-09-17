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
        description="Classify support tickets by nearest-neighbor vote over a Chroma "
        "knowledge base.",
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
