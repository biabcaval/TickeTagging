from __future__ import annotations

import pytest

from ticketag.models import Classification, ClassifiedTicket, FailedTicket


def test_classification_rejects_out_of_range_confidence():
    with pytest.raises(ValueError, match="between 0 and 1"):
        Classification("Access", "reason", 1.5)


@pytest.mark.parametrize(
    ("category", "justification"),
    [("", "reason"), ("  ", "reason"), ("Access", ""), ("Access", "   ")],
)
def test_classification_rejects_blank_fields(category, justification):
    with pytest.raises(ValueError):
        Classification(category, justification, 0.5)


def test_classified_ticket_to_dict_omits_the_prompt_text():
    ticket = ClassifiedTicket("1", "vpn down", Classification("Access", "Login blocked.", 0.9))

    assert ticket.to_dict() == {
        "ticket_id": "1",
        "category": "Access",
        "justification": "Login blocked.",
        "confidence": 0.9,
    }


def test_failed_ticket_to_dict_reports_a_null_category():
    assert FailedTicket("2", "boom").to_dict() == {
        "ticket_id": "2",
        "category": None,
        "error": "boom",
    }
