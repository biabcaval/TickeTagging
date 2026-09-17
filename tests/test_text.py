from __future__ import annotations

import pytest

from ticketag.text import MAX_TICKET_CHARS, bound_ticket


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  I cannot access the VPN!  ", "I cannot access the VPN!"),
        ("line one\r\nline two", "line one\nline two"),
        ("spaced    out", "spaced out"),
        ("para one\n\n\n\npara two", "para one\n\npara two"),
    ],
)
def test_bound_ticket_only_collapses_spacing(raw, expected):
    assert bound_ticket(raw) == expected


def test_bound_ticket_keeps_casing_punctuation_and_pii():
    raw = "I cannot reach the VPN! Mail ana@corp.com, card 4539578763621486."

    assert bound_ticket(raw) == raw


def test_bound_ticket_truncates_on_a_word_boundary():
    bounded = bound_ticket("word " * 50, max_chars=20)

    assert bounded.endswith("[... truncated ...]")
    assert "wor [" not in bounded


def test_bound_ticket_leaves_text_within_the_default_budget_untouched():
    raw = "Short ticket body."

    assert len(raw) < MAX_TICKET_CHARS
    assert bound_ticket(raw) == raw
