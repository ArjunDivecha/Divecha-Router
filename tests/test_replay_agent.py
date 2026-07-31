from pathlib import Path

import pytest

from arjun_model_gateway.replay_agent import _execute_tool, _resolve_path


def test_resolve_path_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes workspace"):
        _resolve_path(tmp_path, "../outside")


def test_resolve_path_accepts_canonicalized_workspace(tmp_path: Path) -> None:
    (tmp_path / "inside.txt").write_text("ok", encoding="utf-8")
    assert _resolve_path(tmp_path, "inside.txt").read_text(encoding="utf-8") == "ok"


def test_run_command_is_allowlisted(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not allowlisted"):
        _execute_tool(tmp_path, "run_command", {"argv": ["curl", "https://example.com"]})


def test_replace_requires_unique_match(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("same same", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly once"):
        _execute_tool(
            tmp_path,
            "replace_in_file",
            {"path": "sample.txt", "old_text": "same", "new_text": "changed"},
        )
