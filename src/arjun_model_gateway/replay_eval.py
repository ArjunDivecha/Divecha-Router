"""Validated, leakage-resistant replay plans built from real repository history.

This module does not execute models.  It validates immutable historical task
capsules and expands them against model-consumption policies so paid agent runs
can happen later from a reviewed plan.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


ReplayReadiness = Literal["ready", "partial", "fixture_required"]
ReplayMode = Literal["sequential", "parallel"]
StageTrigger = Literal["always", "after_plan", "on_verification_failure", "after_review"]
StageRole = Literal["plan", "execute", "review", "repair", "independent"]
StageInput = Literal["task", "task_and_plan", "task_and_artifacts", "task_and_review"]

PRIVATE_IPV4_RE = re.compile(
    r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
)
SECRET_RE = re.compile(
    r"(?i)(?:"
    r"\b(?:password|passwd|api[_ -]?key|access[_ -]?token|secret)\s*[:=]\s*\S+|"
    r"\b(?:admin|root):[^@\s/]+|"
    r"\bsk-(?:or-v1-|ant-api|proj-)?[A-Za-z0-9_-]{12,}"
    r")"
)
SHELL_META_RE = re.compile(r"[;&|`$<>]")
ALLOWED_VERIFIER_PREFIXES = {
    ("npm", "run", "build"),
    ("npm", "run", "lint"),
    ("pytest",),
    ("python", "-m", "pytest"),
    ("python3", "-m", "pytest"),
}


class ReplayVerifier(BaseModel):
    commands: list[list[str]] = Field(min_length=1)
    acceptance: list[str] = Field(min_length=1)
    required_fixtures: list[str] = Field(default_factory=list)


class ReplayTask(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    title: str
    task_family: str
    repo: Path
    base_commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    gold_commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    prompt: str = Field(min_length=8)
    source_timestamp: str
    readiness: ReplayReadiness
    verifier: ReplayVerifier
    forbidden_capabilities: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def commits_differ(self) -> "ReplayTask":
        if self.base_commit == self.gold_commit:
            raise ValueError("base_commit and gold_commit must differ")
        return self


class ReplaySuite(BaseModel):
    version: int = 1
    name: str
    private_source_records: list[Path] = Field(default_factory=list)
    tasks: list[ReplayTask] = Field(min_length=1)


class ReplayStage(BaseModel):
    order: int = Field(ge=1)
    model_family: Literal["kimi", "fable"]
    role: StageRole
    trigger: StageTrigger
    input: StageInput


class ReplayPolicy(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    mode: ReplayMode
    selection: Literal["final_stage", "cheapest_verified"]
    stages: list[ReplayStage] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_stage_order(self) -> "ReplayPolicy":
        orders = [stage.order for stage in self.stages]
        if self.mode == "sequential" and orders != sorted(set(orders)):
            raise ValueError("sequential policy stage orders must be unique and ascending")
        if self.mode == "parallel" and len(set(orders)) != 1:
            raise ValueError("parallel policy stages must share one order")
        if self.selection == "cheapest_verified" and self.mode != "parallel":
            raise ValueError("cheapest_verified selection requires a parallel policy")
        return self


class ReplayPolicies(BaseModel):
    version: int = 1
    policies: list[ReplayPolicy] = Field(min_length=1)


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return payload


def load_replay_suite(path: Path) -> ReplaySuite:
    return ReplaySuite.model_validate(_load_yaml(path))


def load_replay_policies(path: Path) -> ReplayPolicies:
    return ReplayPolicies.model_validate(_load_yaml(path))


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"git {' '.join(args)} failed for {repo}: {detail}")
    return result.stdout.strip()


def _contains_sensitive_text(value: str) -> bool:
    return bool(PRIVATE_IPV4_RE.search(value) or SECRET_RE.search(value))


def _command_allowed(command: list[str]) -> bool:
    if not command or any(SHELL_META_RE.search(part) for part in command):
        return False
    command_tuple = tuple(command)
    return any(command_tuple[: len(prefix)] == prefix for prefix in ALLOWED_VERIFIER_PREFIXES)


def validate_replay_plan(suite: ReplaySuite, policies: ReplayPolicies) -> dict:
    errors: list[str] = []
    task_rows: list[dict] = []
    task_ids: set[str] = set()
    policy_ids: set[str] = set()

    for source in suite.private_source_records:
        if not source.expanduser().exists():
            errors.append(f"missing private source record: {source}")

    for policy in policies.policies:
        if policy.id in policy_ids:
            errors.append(f"duplicate policy id: {policy.id}")
        policy_ids.add(policy.id)

    for task in suite.tasks:
        if task.id in task_ids:
            errors.append(f"duplicate task id: {task.id}")
        task_ids.add(task.id)
        repo = task.repo.expanduser().resolve()
        if not repo.is_dir():
            errors.append(f"{task.id}: repo does not exist: {repo}")
            continue
        if _contains_sensitive_text(task.prompt):
            errors.append(f"{task.id}: prompt contains private network or credential material")
        for value in [task.title, *task.verifier.acceptance, *task.notes]:
            if _contains_sensitive_text(value):
                errors.append(f"{task.id}: metadata contains private network or credential material")
        for command in task.verifier.commands:
            if not _command_allowed(command):
                errors.append(f"{task.id}: verifier command is not offline-allowlisted: {command}")
        if "live_network" not in task.forbidden_capabilities:
            errors.append(f"{task.id}: live_network must be explicitly forbidden")
        if "real_credentials" not in task.forbidden_capabilities:
            errors.append(f"{task.id}: real_credentials must be explicitly forbidden")

        changed_paths: list[str] = []
        try:
            _git(repo, "cat-file", "-e", f"{task.base_commit}^{{commit}}")
            _git(repo, "cat-file", "-e", f"{task.gold_commit}^{{commit}}")
            _git(repo, "merge-base", "--is-ancestor", task.base_commit, task.gold_commit)
            changed_paths = [
                line
                for line in _git(
                    repo, "diff", "--name-only", task.base_commit, task.gold_commit
                ).splitlines()
                if line
            ]
            if not changed_paths:
                errors.append(f"{task.id}: base-to-gold diff is empty")
        except ValueError as exc:
            errors.append(f"{task.id}: {exc}")

        task_rows.append(
            {
                "id": task.id,
                "task_family": task.task_family,
                "readiness": task.readiness,
                "base_commit": task.base_commit,
                "gold_commit": task.gold_commit,
                "changed_file_count": len(changed_paths),
                "required_fixtures": task.verifier.required_fixtures,
            }
        )

    policy_rows = [
        {
            "id": policy.id,
            "mode": policy.mode,
            "model_consumptions_per_task": len(policy.stages),
            "selection": policy.selection,
        }
        for policy in policies.policies
    ]
    total_cells = len(suite.tasks) * len(policies.policies)
    total_consumptions = len(suite.tasks) * sum(len(policy.stages) for policy in policies.policies)
    return {
        "schema_version": 1,
        "suite": suite.name,
        "passed": not errors,
        "errors": errors,
        "task_count": len(suite.tasks),
        "policy_count": len(policies.policies),
        "policy_task_cells": total_cells,
        "maximum_model_consumptions": total_consumptions,
        "tasks": task_rows,
        "policies": policy_rows,
    }


def write_replay_report(report: dict, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
