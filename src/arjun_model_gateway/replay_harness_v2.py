"""Auditable v2 harness for provider-free replay controls.

Paid execution is intentionally absent. A future live runner may consume a
preflight report only when the frozen protocol hash matches and every control
passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from arjun_model_gateway.camera_replay import (
    PILOT_COMMITS,
    assert_capsule_safe,
    materialize_capsule,
    prove_goldens,
    verify_capsule,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "evals/camera_history/pilot_protocol_v2.yaml"
SKIP_DIRS = {"node_modules", ".next", ".git"}
VALIDATIONS = {
    "build": ("npm", "run", "build", "--", "--webpack"),
    "types": ("npx", "tsc", "--noEmit"),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List model-visible workspace files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "pattern": {"type": "string", "default": "**/*"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 file with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search model-visible text files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "regex": {"type": "boolean", "default": False},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "Replace one exact unique block in a UTF-8 file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or replace one UTF-8 workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_validation",
            "description": "Run a named offline validation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "check": {
                        "type": "string",
                        "enum": sorted(VALIDATIONS),
                    }
                },
                "required": ["check"],
            },
        },
    },
]


class CompletionBackend(Protocol):
    name: str

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class ScriptedBackend:
    """Deterministic backend used to prove the tool loop without a provider."""

    name = "scripted-control"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def complete(self, _payload: dict[str, Any]) -> dict[str, Any]:
        if not self.responses:
            raise RuntimeError("scripted backend exhausted")
        self.calls += 1
        return self.responses.pop(0)


@dataclass
class HarnessResult:
    passed: bool
    backend: str
    turns: int
    tool_calls: int
    tool_errors: int
    finish_reason: str
    transcript_path: str


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def scripted_tool_response(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": "", "tool_calls": tool_calls},
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0},
    }


def scripted_stop_response(content: str = "done") -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0},
    }


class ToolExecutor:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def resolve(self, relative: str) -> Path:
        if "\x00" in relative:
            raise ValueError("invalid path")
        path = (self.workspace / relative).resolve()
        if path != self.workspace and self.workspace not in path.parents:
            raise ValueError("path escapes workspace")
        if any(part in SKIP_DIRS for part in path.relative_to(self.workspace).parts):
            raise ValueError("path is not model-visible")
        return path

    def visible_files(self, root: Path | None = None) -> list[Path]:
        root = root or self.workspace
        if root.is_file():
            return [root]
        return [
            path
            for path in root.rglob("*")
            if path.is_file()
            and not any(
                part in SKIP_DIRS for part in path.relative_to(self.workspace).parts
            )
            and path.stat().st_size <= 1_000_000
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "list_files":
            root = self.resolve(str(arguments.get("path") or "."))
            pattern = str(arguments.get("pattern") or "**/*")
            candidates = [root] if root.is_file() else list(root.glob(pattern))
            rows = [
                str(path.relative_to(self.workspace))
                for path in candidates
                if not any(
                    part in SKIP_DIRS
                    for part in path.relative_to(self.workspace).parts
                )
            ]
            return "\n".join(sorted(rows)[:1000])
        if name == "read_file":
            path = self.resolve(str(arguments["path"]))
            lines = path.read_text(encoding="utf-8").splitlines()
            start = max(1, int(arguments.get("start_line") or 1))
            end = min(len(lines), int(arguments.get("end_line") or start + 499))
            return "\n".join(
                f"{line_number}: {lines[line_number - 1]}"
                for line_number in range(start, end + 1)
            )
        if name == "search":
            root = self.resolve(str(arguments.get("path") or "."))
            query = str(arguments["query"])
            use_regex = bool(arguments.get("regex", False))
            matcher = re.compile(query) if use_regex else None
            matches: list[str] = []
            for path in self.visible_files(root):
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                except UnicodeDecodeError:
                    continue
                for line_number, line in enumerate(lines, 1):
                    found = bool(matcher.search(line)) if matcher else query in line
                    if found:
                        matches.append(
                            f"{path.relative_to(self.workspace)}:{line_number}:{line[:400]}"
                        )
                        if len(matches) == 300:
                            return "\n".join(matches)
            return "\n".join(matches) or "no matches"
        if name == "replace_in_file":
            path = self.resolve(str(arguments["path"]))
            old = str(arguments["old_text"])
            new = str(arguments["new_text"])
            text = path.read_text(encoding="utf-8")
            count = text.count(old)
            if count != 1:
                raise ValueError(f"old_text must occur exactly once; found {count}")
            path.write_text(text.replace(old, new), encoding="utf-8")
            assert_capsule_safe(self.workspace)
            return f"updated {path.relative_to(self.workspace)}"
        if name == "write_file":
            path = self.resolve(str(arguments["path"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(arguments["content"]), encoding="utf-8")
            assert_capsule_safe(self.workspace)
            return f"wrote {path.relative_to(self.workspace)}"
        if name == "run_validation":
            check = str(arguments["check"])
            if check not in VALIDATIONS:
                raise ValueError(f"unknown validation: {check}")
            result = subprocess.run(
                VALIDATIONS[check],
                cwd=self.workspace,
                text=True,
                capture_output=True,
                timeout=240,
                check=False,
                env={**os.environ, "NEXT_TELEMETRY_DISABLED": "1"},
            )
            output = (result.stdout + "\n" + result.stderr)[-20_000:]
            if result.returncode:
                raise RuntimeError(
                    f"{check} failed with exit code {result.returncode}\n{output}"
                )
            return f"exit_code={result.returncode}\n{output}"
        raise ValueError(f"unknown tool: {name}")


def run_harness(
    workspace: Path,
    prompt: str,
    backend: CompletionBackend,
    transcript_path: Path,
    *,
    max_turns: int = 12,
) -> HarnessResult:
    executor = ToolExecutor(workspace)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Work only inside the sealed offline workspace. Inspect, edit, "
                "validate, and finish. Never use credentials or networks."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    events: list[dict[str, Any]] = []
    tool_calls = 0
    tool_errors = 0
    finish_reason = "turn_limit"
    turns = 0

    for turns in range(1, max_turns + 1):
        raw = backend.complete(
            {
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
            }
        )
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        assistant = {
            "role": "assistant",
            "content": message.get("content") or "",
        }
        calls = message.get("tool_calls") or []
        if calls:
            assistant["tool_calls"] = calls
        messages.append(assistant)
        if not calls:
            finish_reason = str(choice.get("finish_reason") or "stop")
            break
        for call in calls:
            tool_calls += 1
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            arguments: dict[str, Any] = {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
                output = executor.execute(name, arguments)
                status = "ok"
            except Exception as exc:  # noqa: BLE001
                tool_errors += 1
                output = f"tool_error: {type(exc).__name__}: {exc}"
                status = "error"
            events.append(
                {
                    "turn": turns,
                    "tool": name,
                    "arguments": arguments,
                    "status": status,
                    "output": output,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": output,
                }
            )

    assert_capsule_safe(workspace)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "backend": backend.name,
                "messages": messages,
                "events": events,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return HarnessResult(
        passed=finish_reason == "stop" and tool_errors == 0,
        backend=backend.name,
        turns=turns,
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        finish_reason=finish_reason,
        transcript_path=str(transcript_path),
    )


def _visible_text_map(workspace: Path) -> dict[str, str]:
    executor = ToolExecutor(workspace)
    mapping: dict[str, str] = {}
    for path in executor.visible_files():
        try:
            mapping[str(path.relative_to(executor.workspace))] = path.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            continue
    return mapping


def preserve_candidate(workspace: Path, destination: Path) -> dict[str, str]:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        workspace,
        destination,
        ignore=shutil.ignore_patterns("node_modules", ".next", ".git"),
    )
    hashes: dict[str, str] = {}
    for path in ToolExecutor(destination).visible_files():
        hashes[str(path.relative_to(destination))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (destination.parent / "manifest.json").write_text(
        json.dumps({"files": hashes}, indent=2),
        encoding="utf-8",
    )
    return hashes


def _gold_script(base: Path, gold: Path) -> ScriptedBackend:
    base_files = _visible_text_map(base)
    gold_files = _visible_text_map(gold)
    calls: list[dict[str, Any]] = []
    for index, (relative, content) in enumerate(sorted(gold_files.items()), 1):
        if base_files.get(relative) != content:
            calls.append(
                _tool_call(
                    f"gold-{index}",
                    "write_file",
                    {"path": relative, "content": content},
                )
            )
    for check in ("types", "build"):
        calls.append(
            _tool_call(
                f"gold-{len(calls) + 1}",
                "run_validation",
                {"check": check},
            )
        )
    return ScriptedBackend(
        [scripted_tool_response(calls), scripted_stop_response("gold control applied")]
    )


def _run_tool_contract(artifact_root: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="camera-tool-contract-") as temp:
        workspace = Path(temp).resolve()
        (workspace / "sample.txt").write_text("alpha beta\n", encoding="utf-8")
        (workspace / "REPLAY_TASK.md").write_text("offline control\n", encoding="utf-8")
        backend = ScriptedBackend(
            [
                scripted_tool_response(
                    [
                        _tool_call("t1", "list_files", {"path": "."}),
                        _tool_call("t2", "read_file", {"path": "sample.txt"}),
                        _tool_call(
                            "t3",
                            "search",
                            {"path": ".", "query": "alpha", "regex": False},
                        ),
                        _tool_call(
                            "t4",
                            "replace_in_file",
                            {
                                "path": "sample.txt",
                                "old_text": "alpha",
                                "new_text": "gamma",
                            },
                        ),
                        _tool_call(
                            "t5",
                            "write_file",
                            {"path": "created.txt", "content": "created\n"},
                        ),
                    ]
                ),
                scripted_stop_response(),
            ]
        )
        result = run_harness(
            workspace,
            "provider-free tool contract",
            backend,
            artifact_root / "tool_contract_transcript.json",
        )
        return {
            "passed": (
                result.passed
                and (workspace / "sample.txt").read_text(encoding="utf-8")
                == "gamma beta\n"
                and (workspace / "created.txt").read_text(encoding="utf-8")
                == "created\n"
            ),
            "result": asdict(result),
        }


def _run_safety_rejections() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="camera-safety-control-") as temp:
        executor = ToolExecutor(Path(temp))
        outcomes: dict[str, bool] = {}
        try:
            executor.resolve("../outside")
        except ValueError:
            outcomes["path_escape_rejected"] = True
        else:
            outcomes["path_escape_rejected"] = False
        try:
            executor.execute(
                "run_validation",
                {"check": "curl https://example.invalid"},
            )
        except ValueError:
            outcomes["network_shaped_validation_rejected"] = True
        else:
            outcomes["network_shaped_validation_rejected"] = False
        return {"passed": all(outcomes.values()), "checks": outcomes}


def _run_scripted_repair(task_id: str, artifact_root: Path) -> dict[str, object]:
    commits = PILOT_COMMITS[task_id]
    with (
        tempfile.TemporaryDirectory(prefix="camera-scripted-base-") as base_temp,
        tempfile.TemporaryDirectory(prefix="camera-scripted-gold-") as gold_temp,
    ):
        base = materialize_capsule(commits["base"], Path(base_temp))
        gold = materialize_capsule(commits["gold"], Path(gold_temp))
        backend = _gold_script(base, gold)
        result = run_harness(
            base,
            f"scripted reachability control for {task_id}",
            backend,
            artifact_root / task_id / "transcript.json",
        )
        verification = verify_capsule(task_id, commits["base"], base)
        hashes = preserve_candidate(base, artifact_root / task_id / "candidate")
        return {
            "passed": result.passed and verification.passed,
            "harness": asdict(result),
            "verification": asdict(verification),
            "candidate_file_count": len(hashes),
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_preflight(output: Path) -> dict[str, object]:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    artifact_root = output.parent / f"camera_preflight_v2_{timestamp}"
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    controls: dict[str, object] = {}
    controls["tool_contract"] = _run_tool_contract(artifact_root)
    controls["safety_rejections"] = _run_safety_rejections()
    controls["historical_discrimination"] = prove_goldens()
    controls["scripted_repairs"] = {
        task_id: _run_scripted_repair(task_id, artifact_root)
        for task_id in PILOT_COMMITS
    }
    passed = (
        bool(controls["tool_contract"]["passed"])  # type: ignore[index]
        and bool(controls["safety_rejections"]["passed"])  # type: ignore[index]
        and bool(controls["historical_discrimination"]["passed"])  # type: ignore[index]
        and all(
            bool(result["passed"])
            for result in controls["scripted_repairs"].values()  # type: ignore[union-attr]
        )
    )
    report = {
        "schema_version": 2,
        "status": "passed" if passed else "failed",
        "paid_provider_calls": 0,
        "measured_provider_cost_usd": 0,
        "protocol": protocol,
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "artifact_root": str(artifact_root),
        "controls": controls,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight",))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/camera_preflight_v2.json",
    )
    args = parser.parse_args()
    report = run_preflight(args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "paid_provider_calls": report["paid_provider_calls"],
                "protocol_sha256": report["protocol_sha256"],
                "artifact_root": report["artifact_root"],
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
