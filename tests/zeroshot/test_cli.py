from __future__ import annotations

import pytest

from ticketag.cli.zeroshot import build_parser
from ticketag.text import MAX_TICKET_CHARS


def test_parser_requires_exactly_one_source():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--text", "a", "--csv", "b.csv"])


def test_parser_applies_the_default_size_guard():
    args = build_parser().parse_args(["--text", "a"])

    assert args.max_chars == MAX_TICKET_CHARS
    assert args.categories is None


def test_parser_accepts_the_backend_specific_model_flag():
    args = build_parser().parse_args(["--text", "a", "--model", "gemini-2.5-pro"])

    assert args.model == "gemini-2.5-pro"
