from __future__ import annotations

from ticketag.exceptions import KnowledgeBaseError, TickeTagError


def test_knowledge_base_error_is_a_ticketag_error():
    assert issubclass(KnowledgeBaseError, TickeTagError)
