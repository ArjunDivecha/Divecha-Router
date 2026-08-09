"""
=============================================================================
MODULE: tests/test_client.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
Tests for the harness client (arjun_model_gateway.client.DivechaClient). Uses
an httpx.MockTransport so nothing hits the live gateway: verifies the client
contract (profile named in BOTH `model` and metadata.routing_profile), that
only set options are sent, response parsing (text / amg_request_id / model),
outcome reporting, the verified_call auto-report semantics, and gateway-error
mapping to DivechaError.

INPUT FILES: none — this module performs no file I/O (mocked transport).
OUTPUT FILES: none — this module performs no file I/O.
=============================================================================
"""

from __future__ import annotations

import json

import httpx
import pytest

from arjun_model_gateway.client import DivechaClient, DivechaError


def make_client(handler, **kwargs):
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return DivechaClient("http://testserver/v1", http_client=http, **kwargs)


def chat_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "cmpl_x",
            "model": "openai/gpt-5.6-luna",
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            "amg_request_id": "amg_abc123",
        },
    )


def test_chat_builds_client_contract_and_parses(recwarn):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return chat_ok(request)

    client = make_client(handler, repo="my-harness", script="planner.py")
    res = client.chat(
        "cheap",
        [{"role": "user", "content": "hi"}],
        call_site_id="planner.summarize",
    )
    assert seen["path"] == "/v1/chat/completions"
    body = seen["body"]
    # profile named in BOTH places (the client contract)
    assert body["model"] == "cheap"
    assert body["metadata"]["routing_profile"] == "cheap"
    assert body["metadata"]["repo"] == "my-harness"
    assert body["metadata"]["script"] == "planner.py"
    assert body["metadata"]["call_site_id"] == "planner.summarize"
    # unset options are not sent as noise
    assert "quality_floor" not in body["metadata"]
    assert "temperature" not in body
    # parsed result
    assert res.text == "hello"
    assert res.request_id == "amg_abc123"
    assert res.model == "openai/gpt-5.6-luna"


def test_chat_includes_only_set_overrides():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return chat_ok(request)

    client = make_client(handler)
    client.chat(
        "fast",
        [{"role": "user", "content": "x"}],
        quality_floor="high",
        maximum_budget_usd=0.05,
        temperature=0.2,
        tools=[{"type": "function", "function": {"name": "f"}}],
    )
    body = seen["body"]
    assert body["metadata"]["quality_floor"] == "high"
    assert body["metadata"]["maximum_budget_usd"] == 0.05
    assert body["temperature"] == 0.2
    assert body["tools"][0]["function"]["name"] == "f"
    assert "latency_priority" not in body["metadata"]


def test_report_outcome_posts_expected_body():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True, "request_id": "amg_abc123"})

    client = make_client(handler)
    out = client.report_outcome("amg_abc123", True, verifier="tests", detail="42 passed")
    assert seen["path"] == "/v1/outcomes"
    assert seen["body"] == {
        "request_id": "amg_abc123",
        "success": True,
        "verifier": "tests",
        "detail": "42 passed",
    }
    assert out["ok"] is True


def test_verified_call_reports_when_success_set():
    posts = []

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append((request.url.path, json.loads(request.content)))
        if request.url.path.endswith("/chat/completions"):
            return chat_ok(request)
        return httpx.Response(200, json={"ok": True})

    client = make_client(handler)
    with client.verified_call("code", [{"role": "user", "content": "x"}], verifier="tests") as (res, outcome):
        assert res.request_id == "amg_abc123"
        outcome.success = True
        outcome.detail = "green"

    paths = [p for p, _ in posts]
    assert paths == ["/v1/chat/completions", "/v1/outcomes"]
    _, outcome_body = posts[1]
    assert outcome_body["success"] is True
    assert outcome_body["verifier"] == "tests"
    assert outcome_body["detail"] == "green"


def test_verified_call_skips_report_when_success_unset():
    posts = []

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(request.url.path)
        return chat_ok(request)

    client = make_client(handler)
    with client.verified_call("code", [{"role": "user", "content": "x"}]) as (res, outcome):
        pass  # never set outcome.success

    assert posts == ["/v1/chat/completions"]  # no /outcomes call


def test_gateway_error_maps_to_divecha_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"detail": {"code": "budget_exceeded", "message": "over the $0.01 budget"}}
        )

    client = make_client(handler)
    with pytest.raises(DivechaError) as exc:
        client.chat("frontier", [{"role": "user", "content": "x"}], maximum_budget_usd=0.01)
    assert exc.value.code == "budget_exceeded"
    assert exc.value.status == 400


def test_explain_hits_routes_explain():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json={"model": "x", "estimated_route_cost_usd": 0.001, "reasons": []})

    client = make_client(handler)
    plan = client.explain("code", [{"role": "user", "content": "x"}])
    assert seen["path"] == "/v1/routes/explain"
    assert plan["model"] == "x"
