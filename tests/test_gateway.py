"""
=============================================================================
MODULE: tests/test_gateway.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
Gateway-level tests with the route decision stubbed (no network, no live
catalog): telemetry writing, fallback execution, budget-rejection as HTTP
400, schema auto-verification outcomes, the /v1/outcomes learning endpoint,
and SSE streaming.

INPUT FILES: none (telemetry writes are redirected to pytest tmp_path).
OUTPUT FILES: none outside pytest tmp_path.
=============================================================================
"""

from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from arjun_model_gateway.gateway import (
    app,
    auto_outcomes,
    handle_chat,
    validate_schema_response,
)
from arjun_model_gateway.models import (
    ChatCompletionRequest,
    ChatMessage,
    GatewayMetadata,
    RouteDecision,
)
from arjun_model_gateway.selector import RoutingError
from arjun_model_gateway.telemetry import (
    record_outcome,
    success_counts,
    summarize,
    write_record,
)


def mock_decision(**overrides) -> RouteDecision:
    base = {
        "routing_profile": "cheap",
        "route_type": "simple",
        "provider": "mock",
        "model": "fake/model",
        "estimated_route_cost_usd": 0.001,
    }
    base.update(overrides)
    return RouteDecision(**base)


def test_mock_route_writes_telemetry(tmp_path, monkeypatch):
    telemetry_path = tmp_path / "telemetry.sqlite"
    monkeypatch.setattr(
        "arjun_model_gateway.gateway.write_record", lambda record: write_record(record, telemetry_path)
    )
    monkeypatch.setattr(
        "arjun_model_gateway.gateway.choose_route", lambda request, policy: mock_decision()
    )
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Return JSON please")],
        metadata=GatewayMetadata(repo="test", script="pytest", call_site_id="test.mock"),
    )
    raw = handle_chat(request, force_mock=True)
    assert raw["choices"][0]["message"]["content"]
    assert raw["amg_request_id"].startswith("amg_")  # outcome-callback key

    conn = sqlite3.connect(telemetry_path)
    row = conn.execute(
        "SELECT total_token_count, estimated_route_cost_usd, cost_source FROM model_calls"
    ).fetchone()
    conn.close()
    assert row[0] > 0
    assert row[1] == 0.001
    assert row[2] == "mock"
    summary = summarize(telemetry_path)
    assert summary["totals"]["calls"] == 0  # mock calls excluded from stats


def test_fallback_executes_after_primary_adapter_error(monkeypatch):
    records = []
    monkeypatch.setattr("arjun_model_gateway.gateway.write_record", records.append)
    monkeypatch.setattr("arjun_model_gateway.gateway.record_outcome", lambda *a, **k: None)
    decision = mock_decision(
        provider="missing_provider",
        model="bad-model",
        fallback=[{"provider": "mock", "model": "fallback-model"}],
    )
    monkeypatch.setattr(
        "arjun_model_gateway.gateway.choose_route", lambda request, policy: decision
    )
    request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hello")])
    raw = handle_chat(request)
    assert raw["choices"][0]["message"]["content"]
    assert records[0].chosen_provider == "mock"
    assert records[0].fallback_count == 1


def test_cumulative_budget_blocks_fallback_after_failed_attempt(monkeypatch):
    import pytest

    from arjun_model_gateway.adapters import AdapterError

    monkeypatch.setattr("arjun_model_gateway.gateway.write_record", lambda record: None)
    monkeypatch.setattr("arjun_model_gateway.gateway.record_outcome", lambda *a, **k: None)
    decision = mock_decision(
        provider="missing_provider",
        model="pricey/model",
        budget_usd=0.0015,
        estimated_route_cost_usd=0.001,
        fallback=[{"provider": "mock", "model": "fallback-model"}],
        candidates=[
            {"model": "pricey/model", "estimated_route_cost_usd": 0.001},
            {"model": "fallback-model", "estimated_route_cost_usd": 0.001},
        ],
    )
    monkeypatch.setattr(
        "arjun_model_gateway.gateway.choose_route", lambda request, policy: decision
    )
    request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hello")])
    # primary fails (~$0.001 attempted); fallback (~$0.001) would exceed $0.0015
    with pytest.raises(AdapterError) as exc:
        handle_chat(request)
    assert exc.value.error_type == "cumulative_budget_exceeded"


def test_budget_rejection_returns_400(monkeypatch):
    def raise_budget(request, policy):
        raise RoutingError("budget_exceeded", "estimated $0.42 over the $0.01 budget")

    monkeypatch.setattr("arjun_model_gateway.gateway.choose_route", raise_budget)
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "budget_exceeded"


def test_schema_validation_pass_and_fail():
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "t",
            "schema": {"type": "object", "required": ["x"], "properties": {"x": {"type": "integer"}}},
        },
    }
    ok, _ = validate_schema_response('{"x": 1}', response_format)
    assert ok
    ok, detail = validate_schema_response('{"y": 1}', response_format)
    assert not ok and "schema violation" in detail
    ok, detail = validate_schema_response("not json", response_format)
    assert not ok and "not valid JSON" in detail


def test_auto_outcomes_records_schema_result():
    decision = mock_decision(
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "t", "schema": {"type": "object", "required": ["x"]}},
        }
    )
    request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="data")])
    results = auto_outcomes(request, decision, '{"x": 1}', {"choices": []})
    assert results == [
        {
            "verifier": "schema",
            "success": True,
            "detail": "JSON validates against the requested schema",
        }
    ]


def test_outcomes_endpoint_updates_learning_ledger(tmp_path, monkeypatch):
    telemetry_path = tmp_path / "telemetry.sqlite"
    monkeypatch.setattr(
        "arjun_model_gateway.gateway.record_outcome",
        lambda request_id, **kw: record_outcome(request_id, path=telemetry_path, **kw),
    )
    # write an attempt row so the outcome joins to a (profile, model) pair
    from arjun_model_gateway.models import TelemetryRecord

    write_record(
        TelemetryRecord(
            request_id="amg_test123",
            routing_profile="cheap",
            chosen_route="simple",
            chosen_model="gamma/cheap-4",
            chosen_provider="openrouter",
            latency_ms=10.0,
        ),
        telemetry_path,
    )
    client = TestClient(app)
    response = client.post(
        "/v1/outcomes",
        json={"request_id": "amg_test123", "success": False, "verifier": "tests"},
    )
    assert response.status_code == 200
    scoped, overall = success_counts(telemetry_path)
    assert scoped[("cheap", "gamma/cheap-4")] == (0, 1)
    assert overall["gamma/cheap-4"] == (0, 1)


def test_streaming_endpoint_returns_sse(monkeypatch):
    monkeypatch.setattr(
        "arjun_model_gateway.gateway.choose_route", lambda request, policy: mock_decision()
    )
    monkeypatch.setattr("arjun_model_gateway.gateway.write_record", lambda record: None)
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert "data:" in response.text
    assert "[DONE]" in response.text


def test_responses_api_translation_round_trip():
    from arjun_model_gateway.adapters import responses_output_text, responses_payload

    decision = mock_decision(
        provider="openrouter",
        model="fast/orchestrator",
        use_responses_api=True,
        tool_choice="required",
        tools=[{"type": "openrouter:fusion", "parameters": {"max_tool_calls": 1}}],
    )
    request = ChatCompletionRequest(
        messages=[
            ChatMessage(role="system", content="be brief"),
            ChatMessage(role="user", content="hello"),
        ],
        max_tokens=256,
    )
    payload = responses_payload(request, decision)
    assert payload["model"] == "fast/orchestrator"
    assert payload["input"][0] == {"role": "system", "content": "be brief"}
    assert payload["tools"][0]["type"] == "openrouter:fusion"
    assert payload["tool_choice"] == "required"
    assert payload["max_output_tokens"] == 256

    raw = {
        "id": "resp_1",
        "output": [
            {"type": "reasoning", "content": []},
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "final "},
                    {"type": "output_text", "text": "answer"},
                ],
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "cost": 0.001},
    }
    assert responses_output_text(raw) == "final answer"


def test_models_endpoint_lists_profiles():
    client = TestClient(app)
    response = client.get("/v1/models")
    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert "auto" in ids and "cheap" in ids and "frontier" in ids
