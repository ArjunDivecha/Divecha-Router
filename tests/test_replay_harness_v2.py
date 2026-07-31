import json
from pathlib import Path

import pytest

from arjun_model_gateway.replay_harness_v2 import (
    ScriptedBackend,
    ToolExecutor,
    _run_safety_rejections,
    _tool_call,
    preserve_candidate,
    run_harness,
    scripted_stop_response,
    scripted_tool_response,
)


def test_v2_tool_loop_edits_and_preserves_transcript(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "sample.txt").write_text("before\n", encoding="utf-8")
    (workspace / "REPLAY_TASK.md").write_text("offline\n", encoding="utf-8")
    trace = tmp_path / "trace.json"
    backend = ScriptedBackend(
        [
            scripted_tool_response(
                [
                    _tool_call("one", "read_file", {"path": "sample.txt"}),
                    _tool_call(
                        "two",
                        "replace_in_file",
                        {
                            "path": "sample.txt",
                            "old_text": "before",
                            "new_text": "after",
                        },
                    ),
                ]
            ),
            scripted_stop_response(),
        ]
    )
    result = run_harness(workspace, "control", backend, trace)
    assert result.passed
    assert (workspace / "sample.txt").read_text(encoding="utf-8") == "after\n"
    assert json.loads(trace.read_text(encoding="utf-8"))["events"][1]["status"] == "ok"


def test_v2_rejects_escape_and_hidden_dependencies(tmp_path: Path) -> None:
    executor = ToolExecutor(tmp_path)
    with pytest.raises(ValueError, match="escapes workspace"):
        executor.resolve("../outside")
    with pytest.raises(ValueError, match="not model-visible"):
        executor.resolve("node_modules/package/index.js")


def test_v2_validation_is_enum_not_shell(tmp_path: Path) -> None:
    executor = ToolExecutor(tmp_path)
    with pytest.raises(ValueError, match="unknown validation"):
        executor.execute("run_validation", {"check": "curl https://example.com"})


def test_v2_candidate_preservation_uses_canonical_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file.txt").write_text("evidence\n", encoding="utf-8")
    destination = tmp_path / "artifacts" / "candidate"
    hashes = preserve_candidate(workspace, destination)
    assert set(hashes) == {"file.txt"}
    assert (destination.parent / "manifest.json").is_file()


def test_v2_safety_rejections_are_reportable() -> None:
    result = _run_safety_rejections()
    assert result["passed"] is True
