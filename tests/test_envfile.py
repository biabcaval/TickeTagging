from __future__ import annotations

import os

from ticketag.envfile import load_env_file


def test_load_env_file_does_not_override_existing_variables(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nMY_TOKEN="from_file"\nMY_MODEL=file/model\n')
    monkeypatch.setenv("MY_TOKEN", "from_shell")
    monkeypatch.delenv("MY_MODEL", raising=False)

    load_env_file(env_file)

    assert os.environ["MY_TOKEN"] == "from_shell"
    assert os.environ["MY_MODEL"] == "file/model"


def test_load_env_file_is_a_no_op_when_the_file_is_absent(tmp_path):
    load_env_file(tmp_path / "absent.env")  # must not raise
