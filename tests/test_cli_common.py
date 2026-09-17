from __future__ import annotations

import csv
import json

import pytest

from ticketag.cli.common import build_base_parser, read_csv_tickets, write_results
from ticketag.exceptions import TickeTagError
from ticketag.models import Classification, ClassifiedTicket, FailedTicket


def test_read_csv_tickets_yields_pairs_within_limit(tmp_path):
    path = tmp_path / "tickets.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(
            [["Document", "Topic_group"], ["monitor broken", "Hardware"], ["vpn down", "Access"]]
        )

    assert list(read_csv_tickets(path, "Document", None, 1)) == [("0", "monitor broken")]


def test_read_csv_tickets_rejects_unknown_column(tmp_path):
    path = tmp_path / "tickets.csv"
    path.write_text("Document\nmonitor broken\n")

    with pytest.raises(TickeTagError, match="not found"):
        list(read_csv_tickets(path, "Body", None, None))


def test_write_results_emits_csv_with_failures(tmp_path):
    output = tmp_path / "out.csv"
    results = [
        ClassifiedTicket("1", "vpn down", Classification("Access", "VPN login blocked.", 0.9)),
        FailedTicket("2", "empty ticket"),
    ]

    write_results(results, output)

    rows = list(csv.DictReader(output.open()))
    assert rows[0]["category"] == "Access"
    assert rows[1]["error"] == "empty ticket"


def test_write_results_emits_json_file(tmp_path):
    output = tmp_path / "out.json"
    results = [
        ClassifiedTicket("1", "vpn down", Classification("Access", "VPN login blocked.", 0.9))
    ]

    write_results(results, output)

    assert json.loads(output.read_text())[0]["justification"] == "VPN login blocked."


def test_build_base_parser_includes_the_category_flag_by_default():
    parser = build_base_parser("prog", "desc")

    args = parser.parse_args(["--text", "a", "--category", "Hardware"])

    assert args.categories == ["Hardware"]


def test_build_base_parser_omits_the_category_flag_when_disabled():
    parser = build_base_parser("prog", "desc", include_categories=False)

    with pytest.raises(SystemExit):
        parser.parse_args(["--text", "a", "--category", "Hardware"])
