"""Tests for the ticketag-rag-ingest CLI: CSV reading and argument parsing."""

from __future__ import annotations

import csv

import pytest

from ticketag.cli.rag_ingest import build_parser, main, read_labeled_csv_rows
from ticketag.exceptions import TickeTagError


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def test_read_labeled_csv_rows_yields_id_text_category_triples(tmp_path):
    path = tmp_path / "tickets.csv"
    write_csv(
        path,
        [["Document", "Topic_group"], ["monitor broken", "Hardware"], ["vpn down", "Access"]],
    )

    rows = list(read_labeled_csv_rows(path, "Document", "Topic_group", None, None))

    assert rows == [("0", "monitor broken", "Hardware"), ("1", "vpn down", "Access")]


def test_read_labeled_csv_rows_skips_blank_categories(tmp_path, caplog):
    path = tmp_path / "tickets.csv"
    write_csv(path, [["Document", "Topic_group"], ["mystery ticket", ""]])

    with caplog.at_level("WARNING"):
        rows = list(read_labeled_csv_rows(path, "Document", "Topic_group", None, None))

    assert rows == []
    assert "Skipping row" in caplog.text


def test_read_labeled_csv_rows_rejects_unknown_column(tmp_path):
    path = tmp_path / "tickets.csv"
    write_csv(path, [["Document", "Topic_group"], ["a", "Hardware"]])

    with pytest.raises(TickeTagError, match="not found"):
        list(read_labeled_csv_rows(path, "Body", "Topic_group", None, None))


def test_read_labeled_csv_rows_honors_the_limit(tmp_path):
    path = tmp_path / "tickets.csv"
    write_csv(
        path,
        [["Document", "Topic_group"], ["a", "Hardware"], ["b", "Access"], ["c", "Storage"]],
    )

    rows = list(read_labeled_csv_rows(path, "Document", "Topic_group", None, 2))

    assert [row[0] for row in rows] == ["0", "1"]


def test_parser_requires_the_csv_flag():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parser_applies_defaults_matching_the_sample_export():
    args = build_parser().parse_args(["--csv", "tickets.csv"])

    assert args.text_column == "Document"
    assert args.category_column == "Topic_group"


def test_main_ingests_and_prints_a_summary(tmp_path, monkeypatch, capsys):
    path = tmp_path / "tickets.csv"
    write_csv(path, [["Document", "Topic_group"], ["monitor broken", "Hardware"]])
    monkeypatch.setenv("CHROMA_API_KEY", "ck-test")

    class StubStore:
        def add_many(self, rows, batch_size):
            self.rows = list(rows)
            return len(self.rows)

        def count(self):
            return 1

    stub_store = StubStore()
    monkeypatch.setattr("ticketag.cli.rag_ingest.store_from_env", lambda settings: stub_store)

    exit_code = main(["--csv", str(path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Ingested 1 ticket(s)" in captured.out
    assert stub_store.rows == [("0", "monitor broken", "Hardware")]
