"""Tests for real-history replay plan validation."""

from __future__ import annotations

import subprocess
from pathlib import Path

from arjun_model_gateway.replay_eval import (
    ReplayPolicies,
    ReplayPolicy,
    ReplayStage,
    ReplaySuite,
    ReplayTask,
    ReplayVerifier,
    validate_replay_plan,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _history_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "history"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "replay@example.invalid")
    _git(repo, "config", "user.name", "Replay Test")
    (repo / "app.txt").write_text("before\n", encoding="utf-8")
    _git(repo, "add", "app.txt")
    _git(repo, "commit", "-m", "before")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "app.txt").write_text("after\n", encoding="utf-8")
    _git(repo, "commit", "-am", "after")
    gold = _git(repo, "rev-parse", "HEAD")
    return repo, base, gold


def _policies() -> ReplayPolicies:
    return ReplayPolicies(
        policies=[
            ReplayPolicy(
                id="kimi_solo",
                mode="sequential",
                selection="final_stage",
                stages=[
                    ReplayStage(
                        order=1,
                        model_family="kimi",
                        role="execute",
                        trigger="always",
                        input="task",
                    )
                ],
            ),
            ReplayPolicy(
                id="parallel_oracle",
                mode="parallel",
                selection="cheapest_verified",
                stages=[
                    ReplayStage(
                        order=1,
                        model_family="kimi",
                        role="independent",
                        trigger="always",
                        input="task",
                    ),
                    ReplayStage(
                        order=1,
                        model_family="fable",
                        role="independent",
                        trigger="always",
                        input="task",
                    ),
                ],
            ),
        ]
    )


def _task(repo: Path, base: str, gold: str, prompt: str) -> ReplayTask:
    return ReplayTask(
        id="real_bug",
        title="Repair a visible bug",
        task_family="bugfix",
        repo=repo,
        base_commit=base,
        gold_commit=gold,
        prompt=prompt,
        source_timestamp="2026-07-22T00:00:00Z",
        readiness="ready",
        verifier=ReplayVerifier(
            commands=[["npm", "run", "build"]],
            acceptance=["The visible bug is repaired."],
        ),
        forbidden_capabilities=["live_network", "real_credentials"],
    )


def test_valid_history_plan_expands_policy_cells_and_consumptions(tmp_path: Path):
    repo, base, gold = _history_repo(tmp_path)
    suite = ReplaySuite(name="test", tasks=[_task(repo, base, gold, "Repair the broken view.")])

    report = validate_replay_plan(suite, _policies())

    assert report["passed"] is True
    assert report["policy_task_cells"] == 2
    assert report["maximum_model_consumptions"] == 3
    assert report["tasks"][0]["changed_file_count"] == 1


def test_private_network_or_credentials_fail_closed(tmp_path: Path):
    repo, base, gold = _history_repo(tmp_path)
    private_address = ".".join(["192", "168", "1", "20"])
    synthetic_credential = "admin" + ":default"
    suite = ReplaySuite(
        name="test",
        tasks=[
            _task(
                repo,
                base,
                gold,
                f"Connect to {private_address} using {synthetic_credential}.",
            )
        ],
    )

    report = validate_replay_plan(suite, _policies())

    assert report["passed"] is False
    assert "private network or credential material" in " ".join(report["errors"])


def test_non_allowlisted_verifier_command_is_rejected(tmp_path: Path):
    repo, base, gold = _history_repo(tmp_path)
    task = _task(repo, base, gold, "Repair the broken view.")
    task.verifier.commands = [["curl", "http://example.invalid"]]
    suite = ReplaySuite(name="test", tasks=[task])

    report = validate_replay_plan(suite, _policies())

    assert report["passed"] is False
    assert "not offline-allowlisted" in " ".join(report["errors"])
