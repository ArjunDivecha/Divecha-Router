"""Provider-neutral, tool-using agent runner for sealed replay capsules."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from arjun_model_gateway.camera_replay import (
    PILOT_COMMITS,
    VerificationResult,
    assert_capsule_safe,
    materialize_capsule,
    verify_capsule,
)
from arjun_model_gateway.env_loader import load_provider_keys
from arjun_model_gateway.replay_eval import load_replay_suite

ROOT = Path(__file__).resolve().parents[2]
SUITE_PATH = ROOT / "evals/camera_history/suite.yaml"
MODEL_IDS = {
    "kimi": "moonshotai/kimi-k3",
    "fable": "anthropic/claude-fable-5",
}
PILOT_POLICIES = (
    "kimi_solo",
    "fable_solo",
    "kimi_then_fable_on_failure",
)
ALLOWED_COMMANDS = {
    ("pwd",),
    ("npm", "run", "lint"),
    ("npm", "run", "build", "--", "--webpack"),
    ("npx", "tsc", "--noEmit"),
}
SKIP_DIRS = {"node_modules", ".next", ".git"}
SYSTEM_PROMPT = """You are repairing a sealed offline Next.js benchmark capsule.
Work autonomously. Inspect local files, make the smallest robust fix, and run
local checks. Never access networks, credentials, production services, or paths
outside the workspace. Do not ask questions. A hidden verifier tests the
user-visible behavior after you finish."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List workspace files under a relative directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 workspace file.",
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
            "description": "Search text files for a literal string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "Replace one exact unique text block in a file.",
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
            "description": "Create or fully replace a UTF-8 workspace file.",
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
            "name": "run_command",
            "description": "Run one allowlisted offline validation command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "argv": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["argv"],
            },
        },
    },
]


@dataclass
class StageMetrics:
    model_family: str
    model_id: str
    turns: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_seconds: float
    finish_reason: str
    tool_calls: int
    tool_trace: list[dict[str, str]]


@dataclass
class PilotCell:
    task_id: str
    policy_id: str
    passed: bool
    stages: list[StageMetrics]
    verification: VerificationResult
    total_cost_usd: float
    total_latency_seconds: float


def _resolve_path(workspace: Path, relative: str) -> Path:
    if "\x00" in relative:
        raise ValueError("invalid path")
    workspace = workspace.resolve()
    path = (workspace / relative).resolve()
    if path != workspace and workspace not in path.parents:
        raise ValueError("path escapes workspace")
    if any(part in SKIP_DIRS for part in path.relative_to(workspace).parts):
        raise ValueError("path is not model-visible")
    return path


def _text_paths(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part in SKIP_DIRS for part in path.relative_to(root).parts)
        and path.stat().st_size <= 500_000
    ]


def _execute_tool(workspace: Path, name: str, arguments: dict[str, Any]) -> str:
    workspace = workspace.resolve()
    if name == "list_files":
        root = _resolve_path(workspace, str(arguments.get("path", ".")))
        rows = [
            str(path.relative_to(workspace))
            for path in root.rglob("*")
            if not any(part in SKIP_DIRS for part in path.relative_to(workspace).parts)
        ]
        return "\n".join(sorted(rows)[:500])
    if name == "read_file":
        path = _resolve_path(workspace, str(arguments["path"]))
        lines = path.read_text(encoding="utf-8").splitlines()
        start = max(1, int(arguments.get("start_line", 1)))
        end = min(len(lines), int(arguments.get("end_line", start + 399)))
        return "\n".join(f"{index}: {lines[index - 1]}" for index in range(start, end + 1))
    if name == "search":
        root = _resolve_path(workspace, str(arguments.get("path", ".")))
        query = str(arguments["query"])
        matches: list[str] = []
        paths = [root] if root.is_file() else _text_paths(root)
        for path in paths:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for index, line in enumerate(lines, 1):
                if query in line:
                    matches.append(f"{path.relative_to(workspace)}:{index}:{line[:300]}")
                    if len(matches) >= 200:
                        return "\n".join(matches)
        return "\n".join(matches) or "no matches"
    if name == "replace_in_file":
        path = _resolve_path(workspace, str(arguments["path"]))
        old = str(arguments["old_text"])
        new = str(arguments["new_text"])
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once; found {count}")
        path.write_text(text.replace(old, new), encoding="utf-8")
        assert_capsule_safe(workspace)
        return f"updated {path.relative_to(workspace)}"
    if name == "write_file":
        path = _resolve_path(workspace, str(arguments["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(arguments["content"]), encoding="utf-8")
        assert_capsule_safe(workspace)
        return f"wrote {path.relative_to(workspace)}"
    if name == "run_command":
        import subprocess

        argv = tuple(str(value) for value in arguments["argv"])
        if argv not in ALLOWED_COMMANDS:
            raise ValueError(f"command is not allowlisted: {list(argv)}")
        result = subprocess.run(
            list(argv),
            cwd=workspace,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
            env={**os.environ, "NEXT_TELEMETRY_DISABLED": "1"},
        )
        output = (result.stdout + "\n" + result.stderr)[-12_000:]
        return f"exit_code={result.returncode}\n{output}"
    raise ValueError(f"unknown tool: {name}")


def _task_prompt(task_id: str, repair_feedback: str | None = None) -> str:
    suite = load_replay_suite(SUITE_PATH)
    task = next(task for task in suite.tasks if task.id == task_id)
    acceptance = "\n".join(f"- {item}" for item in task.verifier.acceptance)
    extra = ""
    if task_id == "camera_dock_and_ios_regression":
        extra = """
Launcher contract: derive the window-match host from APP_URL. If a dedicated
browser process exists, exit only after confirming that the intended app window
was actually focused; otherwise open a new app window."""
    feedback = f"\nPrior hidden-check failures:\n{repair_feedback}" if repair_feedback else ""
    return f"""Task: {task.prompt}

Acceptance:
{acceptance}
{extra}
{feedback}

Inspect REPLAY_TASK.md and the source. Implement and locally validate the fix."""


def run_agent_stage(
    workspace: Path,
    model_family: str,
    task_id: str,
    *,
    repair_feedback: str | None = None,
    max_turns: int = 16,
    max_stage_cost_usd: float = 3.0,
) -> StageMetrics:
    model_id = MODEL_IDS[model_family]
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        load_provider_keys()
        key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is unavailable")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _task_prompt(task_id, repair_feedback)},
    ]
    prompt_tokens = completion_tokens = tool_count = 0
    tool_trace: list[dict[str, str]] = []
    total_cost = 0.0
    finish_reason = "turn_limit"
    started = time.monotonic()
    actual_turns = 0
    headers = {"Authorization": f"Bearer {key}"}

    for turn in range(1, max_turns + 1):
        actual_turns = turn
        payload = {
            "model": model_id,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0.2,
            "max_tokens": 3500,
            "usage": {"include": True},
        }
        with httpx.Client(timeout=180) as client:
            response = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"{model_id} request failed ({response.status_code}): {response.text[:1000]}"
            )
        raw = response.json()
        usage = raw.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        total_cost += float(usage.get("cost") or 0.0)
        if total_cost > max_stage_cost_usd:
            finish_reason = "cost_limit"
            break
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.get("content") or "",
        }
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)
        if not tool_calls:
            finish_reason = str(choice.get("finish_reason") or "stop")
            break
        for tool_call in tool_calls:
            tool_count += 1
            function = tool_call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
                result = _execute_tool(workspace, str(function.get("name")), arguments)
            except Exception as exc:  # noqa: BLE001
                result = f"tool_error: {type(exc).__name__}: {exc}"
            tool_trace.append(
                {
                    "name": str(function.get("name")),
                    "status": "error" if result.startswith("tool_error:") else "ok",
                    "detail": result[:300] if result.startswith("tool_error:") else "",
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": result,
                }
            )

    assert_capsule_safe(workspace)
    return StageMetrics(
        model_family=model_family,
        model_id=model_id,
        turns=actual_turns,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=round(total_cost, 8),
        latency_seconds=round(time.monotonic() - started, 3),
        finish_reason=finish_reason,
        tool_calls=tool_count,
        tool_trace=tool_trace,
    )


def _failure_feedback(result: VerificationResult) -> str:
    return "\n".join(f"- {name}" for name, passed in result.checks.items() if not passed)


def run_cell(task_id: str, policy_id: str) -> PilotCell:
    if policy_id not in PILOT_POLICIES:
        raise ValueError(f"unsupported pilot policy: {policy_id}")
    base_commit = PILOT_COMMITS[task_id]["base"]
    with tempfile.TemporaryDirectory(prefix="camera-agent-replay-") as temp:
        workspace = materialize_capsule(base_commit, Path(temp))
        stages: list[StageMetrics] = []
        first_family = "fable" if policy_id == "fable_solo" else "kimi"
        stages.append(run_agent_stage(workspace, first_family, task_id))
        verification = verify_capsule(task_id, base_commit, workspace)
        if policy_id == "kimi_then_fable_on_failure" and not verification.passed:
            stages.append(
                run_agent_stage(
                    workspace,
                    "fable",
                    task_id,
                    repair_feedback=_failure_feedback(verification),
                )
            )
            verification = verify_capsule(task_id, base_commit, workspace)
        return PilotCell(
            task_id=task_id,
            policy_id=policy_id,
            passed=verification.passed,
            stages=stages,
            verification=verification,
            total_cost_usd=round(sum(stage.cost_usd for stage in stages), 8),
            total_latency_seconds=round(sum(stage.latency_seconds for stage in stages), 3),
        )


def _pilot_report(cells: list[PilotCell], started: float, complete: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "complete" if complete else "running",
        "started_at_unix": started,
        "completed_at_unix": time.time() if complete else None,
        "models": MODEL_IDS,
        "task_count": len(PILOT_COMMITS),
        "policy_count": len(PILOT_POLICIES),
        "cell_count": len(PILOT_COMMITS) * len(PILOT_POLICIES),
        "completed_cells": len(cells),
        "passed_cells": sum(cell.passed for cell in cells),
        "total_cost_usd": round(sum(cell.total_cost_usd for cell in cells), 8),
        "cells": [asdict(cell) for cell in cells],
    }


def run_pilot(checkpoint_path: Path | None = None) -> dict[str, Any]:
    cells: list[PilotCell] = []
    started = time.time()
    if checkpoint_path and checkpoint_path.exists():
        previous = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if previous.get("status") == "running":
            started = float(previous.get("started_at_unix") or started)
            cells = [PilotCell(**{
                **cell,
                "stages": [StageMetrics(**stage) for stage in cell["stages"]],
                "verification": VerificationResult(**cell["verification"]),
            }) for cell in previous.get("cells", [])]
    completed = {(cell.task_id, cell.policy_id) for cell in cells}
    for task_id in PILOT_COMMITS:
        for policy_id in PILOT_POLICIES:
            if (task_id, policy_id) in completed:
                continue
            cells.append(run_cell(task_id, policy_id))
            if checkpoint_path:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_text(
                    json.dumps(_pilot_report(cells, started, False), indent=2),
                    encoding="utf-8",
                )
    return _pilot_report(cells, started, True)


def _markdown_report(report: dict[str, Any]) -> str:
    rows = [
        "# Camera K3/Fable pilot",
        "",
        f"Cells passed: {report['passed_cells']}/{report['cell_count']}",
        f"Total measured model cost: ${report['total_cost_usd']:.4f}",
        "",
        "| Task | Policy | Pass | Consumptions | Cost | Latency |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        rows.append(
            f"| {cell['task_id']} | {cell['policy_id']} | "
            f"{'yes' if cell['passed'] else 'no'} | {len(cell['stages'])} | "
            f"${cell['total_cost_usd']:.4f} | {cell['total_latency_seconds']:.1f}s |"
        )
    return "\n".join(rows) + "\n"


def rescore_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    adjustments: list[dict[str, str]] = []
    for cell in report.get("cells", []):
        checks = cell["verification"]["checks"]
        if checks.pop("uses_hls_vod", None) is False and checks.get("requested_hls"):
            adjustments.append(
                {
                    "task_id": cell["task_id"],
                    "policy_id": cell["policy_id"],
                    "reason": "removed implementation-specific VodPlayer component-name check",
                }
            )
        passed = all(checks.values())
        cell["passed"] = passed
        cell["verification"]["passed"] = passed
    report["passed_cells"] = sum(cell["passed"] for cell in report["cells"])
    report["score_adjustments"] = adjustments
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    path.with_suffix(".md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run-pilot", "run-cell", "rescore"))
    parser.add_argument("--task", choices=tuple(PILOT_COMMITS))
    parser.add_argument("--policy", choices=PILOT_POLICIES)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/camera_pilot.json")
    args = parser.parse_args()
    if args.command == "rescore":
        payload = rescore_report(args.output)
    elif args.command == "run-cell":
        if not args.task or not args.policy:
            parser.error("run-cell requires --task and --policy")
        payload: dict[str, Any] = asdict(run_cell(args.task, args.policy))
    else:
        payload = run_pilot(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.command == "run-pilot":
        args.output.with_suffix(".md").write_text(_markdown_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
