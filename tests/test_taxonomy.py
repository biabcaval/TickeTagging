from __future__ import annotations

import pytest

from ticketag.exceptions import ConfigurationError
from ticketag.taxonomy import DEFAULT_CATEGORIES, Category, parse_categories


def test_parse_categories_falls_back_to_default_taxonomy():
    assert parse_categories(None) == DEFAULT_CATEGORIES


def test_parse_categories_splits_name_and_description():
    parsed = parse_categories(["Access: login and permissions", "Hardware"])

    assert parsed == (Category("Access", "login and permissions"), Category("Hardware"))


def test_parse_categories_rejects_empty_name():
    with pytest.raises(ConfigurationError):
        parse_categories([": no name"])


def test_category_prompt_line_omits_missing_description():
    assert Category("Storage").as_prompt_line() == "- Storage"
    assert Category("Storage", "disks").as_prompt_line() == "- Storage: disks"
