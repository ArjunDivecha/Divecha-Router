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
from tests.conftest import FakeCatalog, make_model


@pytest.fixture(autouse=True)
def _no_telemetry(monkeypatch):
    monkeypatch.setattr("arjun_model_gateway.policy.success_counts", lambda: ({}, {}))
    monkeypatch.setattr("arjun_model_gateway.policy.manual_seeds", lambda name: {})
    monkeypatch.setattr("arjun_model_gateway.policy.latency_observations", lambda: {})


def test_policy_validates_and_a_pinned_model_is_impossible():
    policy = validate_policy()
    assert "structured" in policy.profiles
    # the schema itself forbids a `model` key — pinning cannot come back quietly
    from arjun_model_gateway.config import PolicyProfile

    with pytest.raises(Exception):
        PolicyProfile.model_validate({"route_type": "simple", "model": "some/model"})
    assert not hasattr(policy.profiles["cheap"], "model")


def test_model_field_names_the_profile_and_beats_the_auto_classifier(fake_catalog):
    """Regression: a standard OpenAI client sends the profile in `model`. The
    metadata default must not shadow it and send the call to the classifier —
    that once routed News's cheap ETF tagging into a finance FUSION panel."""
    policy = validate_policy()
    request = ChatCompletionRequest(
        model="cheap",
        # finance/coding words in the prompt would fool the auto-classifier
        messages=[ChatMessage(role="user", content="Classify this ETF: portfolio, factor, alpha")],
        metadata=GatewayMetadata(repo="News", script="report/tags.py"),
    )
    decision = choose_route(request, policy, catalog=fake_catalog)
    assert decision.routing_profile == "cheap"
    assert decision.route_type == "simple"


def test_explicit_routing_profile_overrides_model_field(fake_catalog):
    policy = validate_policy()
    request = ChatCompletionRequest(
        model="cheap",
        messages=[ChatMessage(role="user", content="hi")],
        metadata=GatewayMetadata(routing_profile="frontier"),
    )
    decision = choose_route(request, policy, catalog=fake_catalog)
    assert decision.routing_profile == "frontier"


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


# Catalog with two DeepSeek tiers for the real `cheap_code` profile. Pro is
# both smarter (earlier = higher rank) and pricier than Flash, so the light
# (cheapest) and heavy (smartest) picks diverge.
_CHEAP_CODE_MODELS = [
    make_model("anthropic/claude-boss", 5.0, 15.0),
    make_model("deepseek/deepseek-v4-pro", 0.43, 0.87),
    make_model("openai/gpt-mid", 2.0, 8.0),
    make_model("deepseek/deepseek-v4-flash-0731", 0.09, 0.18),
    make_model("cheapo/ultra-cheap", 0.01, 0.02),
    make_model("meta/llama-x", 0.5, 1.5),
    make_model("filler/a", 3.0, 9.0),
    make_model("filler/b", 4.0, 12.0),
]


def _cheap_code_catalog() -> FakeCatalog:
    return FakeCatalog(_CHEAP_CODE_MODELS, latency_order=[])


def test_cheap_code_light_default_uses_flash_worker():
    policy = validate_policy()
    request = ChatCompletionRequest(
        model="cheap_code",
        messages=[ChatMessage(role="user", content="reformat this snippet")],
    )
    decision = choose_route(request, policy, catalog=_cheap_code_catalog())
    assert decision.route_type == "subagent"
    assert decision.model == "anthropic/claude-boss"  # smartest boss
    tool = decision.tools[0]
    assert tool["type"] == "openrouter:subagent"
    assert tool["parameters"]["model"] == "deepseek/deepseek-v4-flash-0731"
    assert any("worker_tier=light" in r for r in decision.reasons)


def test_cheap_code_heavy_signal_uses_smartest_deepseek_worker():
    policy = validate_policy()
    request = ChatCompletionRequest(
        model="cheap_code",
        messages=[ChatMessage(role="user", content="refactor this module end to end")],
        metadata=GatewayMetadata(quality_floor="high"),  # heavy signal
    )
    decision = choose_route(request, policy, catalog=_cheap_code_catalog())
    assert decision.route_type == "subagent"
    # heavy = smartest DeepSeek by rank (Pro), not the cheapest
    assert decision.tools[0]["parameters"]["model"] == "deepseek/deepseek-v4-pro"
    assert any("worker_tier=heavy" in r for r in decision.reasons)


def test_local_privacy_forces_private_profile(fake_catalog, monkeypatch):
    monkeypatch.setattr(
        "arjun_model_gateway.provider_models.resolve_direct_model", lambda provider: "local-served"
    )
    policy = validate_policy()
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hello")],
        metadata=GatewayMetadata(privacy_level="local_only"),
    )
    decision = choose_route(request, policy, catalog=fake_catalog)
    assert decision.routing_profile == "private"
    assert decision.model == "local-served"  # resolved live, never pinned
    assert decision.external_calls_allowed is False
