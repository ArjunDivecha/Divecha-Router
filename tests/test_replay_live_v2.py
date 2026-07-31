import json
from pathlib import Path

import httpx
import pytest

from arjun_model_gateway.replay_live_v2 import (
    _aggregate,
    _cell_key,
    _retry_delay_seconds,
    _workspace_diff,
)


def test_v2_live_cell_key_is_stable() -> None:
    assert _cell_key("task", "policy", 2) == "task:policy:r2"


def test_v2_live_diff_preserves_candidate_evidence() -> None:
    diff = _workspace_diff({"a.txt": "before\n"}, {"a.txt": "after\n"})
    assert "--- a/a.txt" in diff
    assert "+++ b/a.txt" in diff
    assert "-before" in diff
    assert "+after" in diff


def test_v2_live_aggregate_uses_verified_successes() -> None:
    cells = [
        {
            "policy_id": "solo",
            "passed": True,
            "total_cost_usd": 1.0,
            "total_latency_seconds": 10.0,
        },
        {
            "policy_id": "solo",
            "passed": False,
            "total_cost_usd": 2.0,
            "total_latency_seconds": 20.0,
        },
    ]
    result = _aggregate(cells)["solo"]
    assert result["verified_success_rate"] == 0.5
    assert result["cost_per_verified_success"] == 3.0
    assert result["latency_per_verified_success"] == 30.0


def test_v2_live_retry_delay_honors_retry_after_and_caps() -> None:
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(
        429,
        headers={"Retry-After": "1.5"},
        request=request,
    )
    assert _retry_delay_seconds(response, 1) == 1.5

    capped = httpx.Response(
        429,
        headers={"Retry-After": "999"},
        request=request,
    )
    assert _retry_delay_seconds(capped, 1) == 60.0


def test_v2_live_retry_delay_falls_back_to_bounded_backoff() -> None:
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(503, request=request)
    assert _retry_delay_seconds(response, 1) == 1.0
    assert _retry_delay_seconds(response, 10) == 60.0


def test_v2_live_aggregate_counts_interrupted_journal_spend() -> None:
    cells = [
        {
            "policy_id": "combo",
            "passed": True,
            "total_cost_usd": 1.0,
            "total_latency_seconds": 10.0,
        }
    ]
    journal = [
        {
            "event": "provider_response",
            "policy_id": "combo",
            "cost_usd": 1.0,
            "latency_seconds": 10.0,
        },
        {
            "event": "provider_response",
            "policy_id": "combo",
            "cost_usd": 0.5,
            "latency_seconds": 5.0,
        },
    ]
    result = _aggregate(cells, journal)["combo"]
    assert result["verified_success_rate"] == 1.0
    assert result["total_cost_usd"] == 1.5
    assert result["cost_per_verified_success"] == 1.5


def test_v2_live_refuses_stale_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arjun_model_gateway import replay_live_v2

    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("paid_calls_enabled: true\n", encoding="utf-8")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "status": "passed",
                "protocol_sha256": "stale",
                "runner_sha256": "stale",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(replay_live_v2, "PROTOCOL_PATH", protocol)
    monkeypatch.setattr(replay_live_v2, "PREFLIGHT_PATH", preflight)
    monkeypatch.setattr(replay_live_v2, "HARNESS_PATH", Path(__file__))
    with pytest.raises(RuntimeError, match="does not match"):
        replay_live_v2.validate_paid_gate()
