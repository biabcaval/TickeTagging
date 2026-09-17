from __future__ import annotations

import pytest

from ticketag.taxonomy import DEFAULT_CATEGORIES


@pytest.fixture
def categories():
    return DEFAULT_CATEGORIES
