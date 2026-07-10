"""
=============================================================================
MODULE: tests/test_policy.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
Tests for route decisions: policy validation (marketplace profiles may not
pin models), auto-classification, structured-output schema injection,
server-tool blocks with tool_choice=required, latency-priority handling,
and metadata budget enforcement.

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/config/policy.yaml
  (validated and used for real-profile decisions)

OUTPUT FILES: none — this module performs no file I/O.
=============================================================================
"""

from __future__ import annotations

import pytest

from arjun_model_gateway.config import validate_policy
from arjun_model_gateway.models import ChatCompletionRequest, ChatMessage, GatewayMetadata
from arjun_model_gateway.policy import choose_route
from arjun_model_gateway.selector import RoutingError


@pytest.fixture(autouse=True)
def _no_telemetry(monkeypatch):
    monkeypatch.setattr("arjun_model_gateway.policy.success_counts", lambda: ({}, {}))
    monkeypatch.setattr("arjun_model_gateway.policy.manual_seeds", lambda name: {})


def test_policy_validates_and_has_no_hardcoded_marketplace_models():
    policy = validate_policy()
    assert "structured" in policy.profiles
    for name, profile in policy.profiles.items():
        if profile.route_type not in ("local", "direct"):
            assert profile.model is None, f"{name} pins a model"


def test_auto_routes_coding_to_code_profile(fake_catalog):
    policy = validate_policy()
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Debug this repo diff and run pytest")]
    )
    decision = choose_route(request, policy, catalog=fake_catalog)
    assert decision.routing_profile == "code"
    assert decision.route_type == "advisor"
    assert decision.tools and decision.tools[0]["type"] == "openrouter:advisor"
    assert decision.tool_choice == "required"
    # advisor model resolved dynamically = smartest tool-capable fake
    assert decision.tools[0]["parameters"]["model"] == "alpha/frontier-1"


def test_structured_schema_is_injected_and_capability_required(fake_catalog):
    policy = validate_policy()
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="give me data")],
        metadata=GatewayMetadata(
            routing_profile="structured",
            structured_output_schema={"type": "object", "required": ["x"]},
        ),
    )
    decision = choose_route(request, policy, catalog=fake_catalog)
    assert decision.response_format["type"] == "json_schema"
    assert decision.response_format["json_schema"]["schema"]["required"] == ["x"]
    chosen = fake_catalog.get(decision.model)
    assert "structured_outputs" in chosen["supported_parameters"]


def test_fusion_profile_builds_dynamic_panel(fake_catalog):
    policy = validate_policy()
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="research question")],
        metadata=GatewayMetadata(routing_profile="research"),
    )
    decision = choose_route(request, policy, catalog=fake_catalog)
    assert decision.route_type == "fusion"
    tool = decision.tools[0]
    assert tool["type"] == "openrouter:fusion"
    assert len(tool["parameters"]["analysis_models"]) >= 2
    assert decision.estimated_route_cost_usd > 0
    assert any(node["role"] == "judge" for node in decision.route_graph)


def test_latency_priority_switches_to_fastest(fake_catalog):
    policy = validate_policy()
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="quick answer please")],
        metadata=GatewayMetadata(routing_profile="cheap", latency_priority="high"),
    )
    decision = choose_route(request, policy, catalog=fake_catalog)
    assert decision.objective == "fastest"
    assert decision.provider_sort == "latency"


def test_budget_metadata_is_enforced(fake_catalog):
    policy = validate_policy()
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hello")],
        metadata=GatewayMetadata(routing_profile="cheap", maximum_budget_usd=0.0000001),
    )
    with pytest.raises(RoutingError) as exc:
        choose_route(request, policy, catalog=fake_catalog)
    assert exc.value.code == "budget_exceeded"


def test_unknown_profile_rejected(fake_catalog):
    policy = validate_policy()
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hello")],
        metadata=GatewayMetadata(routing_profile="does_not_exist"),
    )
    with pytest.raises(RoutingError) as exc:
        choose_route(request, policy, catalog=fake_catalog)
    assert exc.value.code == "unknown_profile"


def test_local_privacy_forces_private_profile(fake_catalog):
    policy = validate_policy()
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hello")],
        metadata=GatewayMetadata(privacy_level="local_only"),
    )
    decision = choose_route(request, policy, catalog=fake_catalog)
    assert decision.routing_profile == "private"
    assert decision.external_calls_allowed is False
