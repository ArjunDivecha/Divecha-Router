"""Paid v2 Camera replay runner gated by an exact provider-free preflight."""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
import yaml

from arjun_model_gateway.camera_replay import (
    PILOT_COMMITS,
    VerificationResult,
    materialize_capsule,
    verify_capsule,
)
from arjun_model_gateway.env_loader import load_provider_keys
from arjun_model_gateway.replay_eval import load_replay_suite
from arjun_model_gateway.replay_harness_v2 import (
    PROTOCOL_PATH,
    ROOT,
    HarnessResult,
    ToolExecutor,
    _sha256,
    _visible_text_map,
    preserve_candidate,
    run_harness,
)

PREFLIGHT_PATH = ROOT / "reports/camera_preflight_v2.json"
SUITE_PATH = ROOT / "evals/camera_history/suite.yaml"
HARNESS_PATH = ROOT / "src/arjun_model_gateway/replay_harness_v2.py"
RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
MAX_PROVIDER_ATTEMPTS = 8
MAX_RETRY_DELAY_SECONDS = 60.0


def _retry_delay_seconds(response: httpx.Response, attempt: int) -> float:
    backoff = min(MAX_RETRY_DELAY_SECONDS, float(2 ** (attempt - 1)))
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            requested = max(0.0, float(retry_after))
            return min(MAX_RETRY_DELAY_SECONDS, max(backoff, requested))
        except ValueError:
            pass
    return backoff


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_provider_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"provider journal is corrupt at line {line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise RuntimeError(
                f"provider journal line {line_number} is not an object"
            )
        records.append(record)
    return records


class OpenRouterBackend:
    name = "openrouter"

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        max_output_tokens: int,
        max_cost_usd: float,
        temperature: float,
        journal_path: Path,
        audit_context: dict[str, Any],
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.max_cost_usd = max_cost_usd
        self.temperature = temperature
        self.journal_path = journal_path
        self.audit_context = audit_context
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.latency_seconds = 0.0
        self.calls: list[dict[str, Any]] = []
        self.retries: list[dict[str, Any]] = []
        self.cost_limit_reached = False

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.cost_limit_reached:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Stage stopped at the frozen cost limit.",
                        },
                        "finish_reason": "cost_limit",
                    }
                ]
            }
        request = {
            **payload,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "usage": {"include": True},
        }
        response: httpx.Response | None = None
        with httpx.Client(timeout=240) as client:
            for attempt in range(1, MAX_PROVIDER_ATTEMPTS + 1):
                started = time.monotonic()
                response = client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=request,
                )
                latency = time.monotonic() - started
                self.latency_seconds += latency
                if (
                    response.status_code not in RETRYABLE_STATUS_CODES
                    or attempt == MAX_PROVIDER_ATTEMPTS
                ):
                    break
                delay = _retry_delay_seconds(response, attempt)
                self.retries.append(
                    {
                        "attempt": attempt,
                        "status_code": response.status_code,
                        "delay_seconds": delay,
                    }
                )
                time.sleep(delay)
        assert response is not None
        if response.status_code >= 400:
            _append_jsonl(
                self.journal_path,
                {
                    "event": "provider_error",
                    "timestamp_unix": time.time(),
                    **self.audit_context,
                    "model": self.model,
                    "status_code": response.status_code,
                    "retries": len(self.retries),
                    "cost_usd": 0.0,
                },
            )
            raise RuntimeError(
                f"{self.model} failed ({response.status_code}): {response.text[:1200]}"
            )
        raw = response.json()
        usage = raw.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        cost = float(usage.get("cost") or 0.0)
        call_record = {
            "response_id": raw.get("id"),
            "resolved_model": raw.get("model"),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
            "latency_seconds": round(latency, 3),
        }
        _append_jsonl(
            self.journal_path,
            {
                "event": "provider_response",
                "timestamp_unix": time.time(),
                **self.audit_context,
                "model": self.model,
                **call_record,
            },
        )
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost_usd += cost
        self.calls.append(call_record)
        if self.cost_usd > self.max_cost_usd:
            self.cost_limit_reached = True
            choice = (raw.get("choices") or [{}])[0]
            choice["message"] = {
                "role": "assistant",
                "content": "Stage stopped after this response crossed the frozen cost cap.",
            }
            choice["finish_reason"] = "cost_limit"
        return raw

    def metrics(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider_calls": len(self.calls),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "latency_seconds": round(self.latency_seconds, 3),
            "cost_limit_reached": self.cost_limit_reached,
            "retries": self.retries,
            "calls": self.calls,
        }


def _protocol() -> dict[str, Any]:
    payload = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("protocol must be a YAML object")
    return payload


def validate_paid_gate() -> dict[str, Any]:
    protocol = _protocol()
    if protocol.get("paid_calls_enabled") is not True:
        raise RuntimeError("protocol does not enable paid calls")
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed":
        raise RuntimeError("provider-free preflight is not passed")
    expected = {
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "runner_sha256": _sha256(HARNESS_PATH),
    }
    for key, value in expected.items():
        if preflight.get(key) != value:
            raise RuntimeError(
                f"preflight {key} does not match current files; rerun preflight"
            )
    return {
        "protocol": protocol,
        "preflight": preflight,
        **expected,
        "live_runner_sha256": _sha256(Path(__file__)),
    }


def validate_live_catalog(models: dict[str, str]) -> dict[str, Any]:
    with httpx.Client(timeout=30) as client:
        response = client.get("https://openrouter.ai/api/v1/models")
        response.raise_for_status()
        rows = {
            str(item.get("id")): item
            for item in (response.json().get("data") or [])
        }
    result: dict[str, Any] = {}
    for family, model_id in models.items():
        entry = rows.get(model_id)
        if not entry:
            raise RuntimeError(f"model missing from live catalog: {model_id}")
        supported = set(entry.get("supported_parameters") or [])
        if not {"tools", "tool_choice"}.issubset(supported):
            raise RuntimeError(f"model lacks required tool support: {model_id}")
        result[family] = {
            "id": model_id,
            "name": entry.get("name"),
            "pricing": entry.get("pricing"),
            "supported_parameters": sorted(supported),
        }
    return result


def _task_prompt(task_id: str, failed_checks: list[str] | None = None) -> str:
    suite = load_replay_suite(SUITE_PATH)
    task = next(task for task in suite.tasks if task.id == task_id)
    acceptance = "\n".join(f"- {item}" for item in task.verifier.acceptance)
    fixture_contract = ""
    if task_id == "camera_dock_and_ios_regression":
        fixture_contract = """
Offline launcher fixture contract: keep APP_URL, CHROME, PROFILE, PROJECT, and
LOG environment-overridable. The verifier supplies local executable stubs and
tests clean, stale-profile, and already-focused states."""
    feedback = ""
    if failed_checks:
        feedback = "\nFailed hidden checks from the prior stage:\n" + "\n".join(
            f"- {name}" for name in failed_checks
        )
    return f"""Task:
{task.prompt}

Acceptance:
{acceptance}
{fixture_contract}
{feedback}

Work autonomously in the sealed offline capsule. Inspect, edit, run the named
offline validations when useful, and finish. Do not access any network,
credential, production service, or path outside the workspace."""


def _verify_candidate(
    task_id: str,
    base_commit: str,
    workspace: Path,
) -> VerificationResult:
    result = verify_capsule(task_id, base_commit, workspace)
    executor = ToolExecutor(workspace)
    checks = dict(result.checks)
    details = dict(result.details)
    for validation in ("types", "build"):
        try:
            output = executor.execute("run_validation", {"check": validation})
            checks[f"{validation}_passed"] = True
            details[f"{validation}_output"] = output[-4000:]
        except Exception as exc:  # noqa: BLE001
            checks[f"{validation}_passed"] = False
            details[f"{validation}_output"] = f"{type(exc).__name__}: {exc}"[-4000:]
    return VerificationResult(
        task_id=task_id,
        commit=base_commit,
        passed=all(checks.values()),
        checks=checks,
        details=details,
    )


def _workspace_diff(before: dict[str, str], after: dict[str, str]) -> str:
    output: list[str] = []
    for relative in sorted(set(before) | set(after)):
        old = before.get(relative, "")
        new = after.get(relative, "")
        if old == new:
            continue
        output.extend(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    return "".join(output)


def _run_stage(
    workspace: Path,
    task_id: str,
    family: str,
    protocol: dict[str, Any],
    api_key: str,
    transcript_path: Path,
    journal_path: Path,
    audit_context: dict[str, Any],
    failed_checks: list[str] | None = None,
) -> tuple[HarnessResult, dict[str, Any]]:
    limits = protocol["frozen_limits"]
    generation = protocol["generation"]
    backend = OpenRouterBackend(
        protocol["future_models"][family],
        api_key,
        max_output_tokens=int(limits["max_output_tokens_per_turn"]),
        max_cost_usd=float(limits["max_cost_usd_per_stage"]),
        temperature=float(generation["temperature"]),
        journal_path=journal_path,
        audit_context=audit_context,
    )
    harness = run_harness(
        workspace,
        _task_prompt(task_id, failed_checks),
        backend,
        transcript_path,
        max_turns=int(limits["max_turns_per_stage"]),
    )
    return harness, backend.metrics()


def _cell_key(task_id: str, policy_id: str, repetition: int) -> str:
    return f"{task_id}:{policy_id}:r{repetition}"


def run_cell(
    task_id: str,
    policy_id: str,
    repetition: int,
    gate: dict[str, Any],
    api_key: str,
    artifact_root: Path,
    journal_path: Path,
) -> dict[str, Any]:
    protocol = gate["protocol"]
    base_commit = PILOT_COMMITS[task_id]["base"]
    cell_key = _cell_key(task_id, policy_id, repetition)
    cell_dir = artifact_root / task_id / policy_id / f"r{repetition}"
    with tempfile.TemporaryDirectory(prefix="camera-paid-v2-") as temp:
        workspace = materialize_capsule(base_commit, Path(temp))
        before = _visible_text_map(workspace)
        stages: list[dict[str, Any]] = []
        first_family = "fable" if policy_id == "fable_solo" else "kimi"
        harness, metrics = _run_stage(
            workspace,
            task_id,
            first_family,
            protocol,
            api_key,
            cell_dir / f"stage1-{first_family}-transcript.json",
            journal_path,
            {
                "cell_key": cell_key,
                "task_id": task_id,
                "policy_id": policy_id,
                "repetition": repetition,
                "stage": 1,
                "family": first_family,
            },
        )
        stages.append(
            {"family": first_family, "harness": asdict(harness), "metrics": metrics}
        )
        verification = _verify_candidate(task_id, base_commit, workspace)
        if policy_id == "kimi_then_fable_on_failure" and not verification.passed:
            failed = [
                name for name, passed in verification.checks.items() if not passed
            ]
            harness, metrics = _run_stage(
                workspace,
                task_id,
                "fable",
                protocol,
                api_key,
                cell_dir / "stage2-fable-transcript.json",
                journal_path,
                {
                    "cell_key": cell_key,
                    "task_id": task_id,
                    "policy_id": policy_id,
                    "repetition": repetition,
                    "stage": 2,
                    "family": "fable",
                },
                failed,
            )
            stages.append(
                {"family": "fable", "harness": asdict(harness), "metrics": metrics}
            )
            verification = _verify_candidate(task_id, base_commit, workspace)
        after = _visible_text_map(workspace)
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "workspace.diff").write_text(
            _workspace_diff(before, after),
            encoding="utf-8",
        )
        preserve_candidate(workspace, cell_dir / "candidate")
        return {
            "key": cell_key,
            "task_id": task_id,
            "policy_id": policy_id,
            "repetition": repetition,
            "passed": verification.passed,
            "verification": asdict(verification),
            "stages": stages,
            "total_provider_calls": sum(
                stage["metrics"]["provider_calls"] for stage in stages
            ),
            "total_prompt_tokens": sum(
                stage["metrics"]["prompt_tokens"] for stage in stages
            ),
            "total_completion_tokens": sum(
                stage["metrics"]["completion_tokens"] for stage in stages
            ),
            "total_cost_usd": round(
                sum(stage["metrics"]["cost_usd"] for stage in stages), 8
            ),
            "total_latency_seconds": round(
                sum(stage["metrics"]["latency_seconds"] for stage in stages), 3
            ),
            "artifact_dir": str(cell_dir),
        }


def _aggregate(
    cells: list[dict[str, Any]],
    provider_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for cell in cells:
        group = groups.setdefault(
            cell["policy_id"],
            {
                "cells": 0,
                "verified_successes": 0,
                "total_cost_usd": 0.0,
                "total_latency_seconds": 0.0,
            },
        )
        group["cells"] += 1
        group["verified_successes"] += int(cell["passed"])
        if provider_records is None:
            group["total_cost_usd"] += cell["total_cost_usd"]
            group["total_latency_seconds"] += cell["total_latency_seconds"]
    if provider_records is not None:
        for record in provider_records:
            if record.get("event") != "provider_response":
                continue
            policy_id = str(record["policy_id"])
            group = groups.setdefault(
                policy_id,
                {
                    "cells": 0,
                    "verified_successes": 0,
                    "total_cost_usd": 0.0,
                    "total_latency_seconds": 0.0,
                },
            )
            group["total_cost_usd"] += float(record.get("cost_usd") or 0.0)
            group["total_latency_seconds"] += float(
                record.get("latency_seconds") or 0.0
            )
    for group in groups.values():
        successes = group["verified_successes"]
        group["verified_success_rate"] = successes / group["cells"]
        group["cost_per_verified_success"] = (
            group["total_cost_usd"] / successes if successes else math.inf
        )
        group["latency_per_verified_success"] = (
            group["total_latency_seconds"] / successes if successes else math.inf
        )
        group["total_cost_usd"] = round(group["total_cost_usd"], 8)
        group["total_latency_seconds"] = round(
            group["total_latency_seconds"], 3
        )
    return groups


def _checkpoint(
    output: Path,
    *,
    status: str,
    gate: dict[str, Any],
    catalog: dict[str, Any],
    artifact_root: Path,
    cells: list[dict[str, Any]],
    started_at: float,
    journal_path: Path,
) -> dict[str, Any]:
    provider_records = _read_provider_journal(journal_path)
    provider_responses = [
        record
        for record in provider_records
        if record.get("event") == "provider_response"
    ]
    completed_cost = sum(cell["total_cost_usd"] for cell in cells)
    actual_cost = sum(
        float(record.get("cost_usd") or 0.0) for record in provider_responses
    )
    report = {
        "schema_version": 3,
        "status": status,
        "protocol_sha256": gate["protocol_sha256"],
        "runner_sha256": gate["runner_sha256"],
        "live_runner_sha256": gate["live_runner_sha256"],
        "started_at_unix": started_at,
        "updated_at_unix": time.time(),
        "artifact_root": str(artifact_root),
        "provider_journal": str(journal_path),
        "live_catalog": catalog,
        "expected_cells": (
            len(gate["protocol"]["tasks"])
            * len(gate["protocol"]["future_policies"])
            * int(gate["protocol"]["frozen_limits"]["repetitions_per_cell"])
        ),
        "completed_cells": len(cells),
        "passed_cells": sum(int(cell["passed"]) for cell in cells),
        "completed_cell_cost_usd": round(completed_cost, 8),
        "total_cost_usd": round(actual_cost, 8),
        "interrupted_attempt_cost_usd": round(
            max(0.0, actual_cost - completed_cost),
            8,
        ),
        "provider_responses": len(provider_responses),
        "provider_errors": sum(
            record.get("event") == "provider_error"
            for record in provider_records
        ),
        "aggregate_by_policy": _aggregate(cells, provider_records),
        "cells": cells,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_matrix(output: Path) -> dict[str, Any]:
    gate = validate_paid_gate()
    protocol = gate["protocol"]
    load_provider_keys()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is unavailable")
    catalog = validate_live_catalog(protocol["future_models"])
    started_at = time.time()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    artifact_root = output.parent / f"{output.stem}_{timestamp}"
    journal_path = artifact_root / "provider-journal.jsonl"
    cells: list[dict[str, Any]] = []
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("status") == "running":
            for key in ("protocol_sha256", "runner_sha256", "live_runner_sha256"):
                if previous.get(key) != gate[key]:
                    raise RuntimeError(f"cannot resume: {key} changed")
            started_at = float(previous["started_at_unix"])
            artifact_root = Path(previous["artifact_root"])
            journal_path = Path(previous["provider_journal"])
            cells = list(previous.get("cells") or [])
        elif previous.get("status") == "complete":
            raise RuntimeError("paid report is already complete; refusing overwrite")
    completed = {cell["key"] for cell in cells}
    _checkpoint(
        output,
        status="running",
        gate=gate,
        catalog=catalog,
        artifact_root=artifact_root,
        cells=cells,
        started_at=started_at,
        journal_path=journal_path,
    )
    repetitions = int(protocol["frozen_limits"]["repetitions_per_cell"])
    for task_id in protocol["tasks"]:
        for policy_id in protocol["future_policies"]:
            for repetition in range(1, repetitions + 1):
                key = _cell_key(task_id, policy_id, repetition)
                if key in completed:
                    continue
                cells.append(
                    run_cell(
                        task_id,
                        policy_id,
                        repetition,
                        gate,
                        api_key,
                        artifact_root,
                        journal_path,
                    )
                )
                _checkpoint(
                    output,
                    status="running",
                    gate=gate,
                    catalog=catalog,
                    artifact_root=artifact_root,
                    cells=cells,
                    started_at=started_at,
                    journal_path=journal_path,
                )
    return _checkpoint(
        output,
        status="complete",
        gate=gate,
        catalog=catalog,
        artifact_root=artifact_root,
        cells=cells,
        started_at=started_at,
        journal_path=journal_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "gate"))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/camera_paid_v2.json",
    )
    args = parser.parse_args()
    if args.command == "gate":
        gate = validate_paid_gate()
        print(
            json.dumps(
                {
                    "passed": True,
                    "protocol_sha256": gate["protocol_sha256"],
                    "runner_sha256": gate["runner_sha256"],
                    "live_runner_sha256": gate["live_runner_sha256"],
                },
                indent=2,
            )
        )
        return 0
    report = run_matrix(args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "completed_cells": report["completed_cells"],
                "passed_cells": report["passed_cells"],
                "total_cost_usd": report["total_cost_usd"],
                "aggregate_by_policy": report["aggregate_by_policy"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
