"""Tests for the ticketag-rag CLI argument parsing."""

from __future__ import annotations

import pytest

from ticketag.cli.rag import build_parser
from ticketag.text import MAX_TICKET_CHARS


def test_parser_requires_exactly_one_source():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--text", "a", "--csv", "b.csv"])


def test_parser_applies_the_default_size_guard():
    args = build_parser().parse_args(["--text", "a"])

    assert args.max_chars == MAX_TICKET_CHARS


def test_parser_accepts_the_backend_specific_k_and_collection_flags():
    args = build_parser().parse_args(["--text", "a", "--k", "3", "--collection", "custom"])

    assert args.k == 3
    assert args.collection == "custom"


def test_parser_defaults_k_and_collection_to_none():
    args = build_parser().parse_args(["--text", "a"])

    assert args.k is None
    assert args.collection is None


def test_parser_rejects_the_category_flag():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--text", "a", "--category", "Hardware"])
