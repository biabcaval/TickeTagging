"""Command line entrypoint for the zero-shot LLM backend."""

from __future__ import annotations

import argparse
import sys

from ..exceptions import TickeTagError
from ..pipeline import TicketClassificationPipeline
from ..taxonomy import parse_categories
from ..zeroshot import classifier_from_env
from .common import build_base_parser, configure_logging, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Extend the shared parser with the flags only this backend understands."""
    parser = build_base_parser(
        prog="ticketag-zeroshot",
        description="Classify support tickets with a zero-shot LLM and explain each decision.",
    )
    parser.add_argument("--model", help="Model id to send the prompt to")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the zero-shot CLI and return a shell exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        classifier = classifier_from_env(
            categories=parse_categories(args.categories),
            model=args.model,
        )
        pipeline = TicketClassificationPipeline(classifier, max_chars=args.max_chars)
        return run_pipeline(pipeline, args)
    except TickeTagError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
