"""
=============================================================================
MODULE: tests/test_worker_tiers.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
Deterministic tests for two-tier subagent workers (the `cheap_code` feature):
a smart boss orchestrates a cheap DeepSeek worker, where the worker tier
(light=Flash / heavy=Pro) is chosen per request. Covers:
  - light tier resolves to the Flash-style worker, heavy to the Pro-style one;
  - a tier's soft author preference beats a globally-cheaper non-preferred
    model, and reaches workers below the boss's quality floor;
  - graceful fallback when the preferred author is absent from the catalog
    (never a hard routing failure);
  - a subagent profile WITHOUT worker_tiers keeps the historical
    cheapest-in-pool behavior;
  - policy.infer_worker_heavy's auto light/heavy signal.

INPUT FILES: none — this module performs no file I/O (deterministic FakeCatalog).
OUTPUT FILES: none — this module performs no file I/O.
=============================================================================
"""

from __future__ import annotations

from arjun_model_gateway.config import PolicyProfile
from arjun_model_gateway.models import ChatCompletionRequest, ChatMessage, GatewayMetadata
from arjun_model_gateway.policy import infer_worker_heavy
from arjun_model_gateway.selector import select
from tests.conftest import FakeCatalog, make_model


# Smartest first (rank = list position). A frontier boss, two DeepSeek tiers,
# a globally-cheaper non-DeepSeek worker, and filler so `high` floor (top 25%)
# yields a small boss pool that EXCLUDES the Pro tier — forcing the heavy tier
# to reach past the floor pool, which is the behavior under test.
TIER_MODELS = [
    make_model("anthropic/claude-boss", 5.0, 15.0),
    make_model("deepseek/deepseek-v4-flash-0731", 0.09, 0.18),
    make_model("deepseek/deepseek-v4-pro", 0.43, 0.87),
    make_model("openai/gpt-mid", 2.0, 8.0),
    make_model("cheapo/ultra-cheap", 0.01, 0.02),
    make_model("deepseek/deepseek-chat", 0.26, 1.0),
    make_model("meta/llama-x", 0.5, 1.5),
    make_model("filler/model-8", 3.0, 9.0),
]


def tiered_profile(**overrides) -> PolicyProfile:
    base = {
        "route_type": "subagent",
        "objective": "smartest",
        "quality_floor": "high",
        "select": {
            "require_parameters": ["tools"],
            "min_context": 32_000,
            "max_prompt_price_per_million": 10.0,
            "pool_size": 8,
        },
        "expected_tokens": {"input": 16_000, "output": 3_000},
        "default_budget_usd": 5.0,
        "worker_tiers": {
            "light": {
                "prefer_authors": ["deepseek"],
                "name_contains": "flash",
                "max_prompt_price_per_million": 0.20,
            },
            "heavy": {
                "prefer_authors": ["deepseek"],
                "name_contains": "pro",
                "max_prompt_price_per_million": 1.0,
            },
        },
    }
    base.update(overrides)
    return PolicyProfile.model_validate(base)


def _tier_catalog(models=None) -> FakeCatalog:
    return FakeCatalog(models if models is not None else TIER_MODELS, latency_order=[])


def test_light_tier_selects_flash_and_boss_stays_frontier():
    sel = select("cheap_code", tiered_profile(), _tier_catalog(), worker_heavy=False)
    # boss = smartest in the floor pool
    assert sel.primary.model_id == "anthropic/claude-boss"
    # worker = the Flash tier, NOT the globally-cheaper non-DeepSeek model
    assert sel.support["worker"]["id"] == "deepseek/deepseek-v4-flash-0731"
    assert sel.support["worker_tier"] == "light"
    assert "worker_tier_fallback" not in sel.support


def test_heavy_tier_reaches_pro_below_the_boss_floor():
    sel = select("cheap_code", tiered_profile(), _tier_catalog(), worker_heavy=True)
    assert sel.primary.model_id == "anthropic/claude-boss"
    # Pro is rank 2 — outside the `high` floor pool — so this proves the tier
    # draws from the full hard-constraint set, not just the floor pool.
    assert sel.support["worker"]["id"] == "deepseek/deepseek-v4-pro"
    assert sel.support["worker_tier"] == "heavy"
    assert "worker_tier_fallback" not in sel.support


def test_tier_falls_back_gracefully_when_author_absent():
    no_deepseek = [m for m in TIER_MODELS if not m["id"].startswith("deepseek/")]
    sel = select("cheap_code", tiered_profile(), _tier_catalog(no_deepseek), worker_heavy=False)
    # still routes — no RoutingError — with a recorded fallback note
    assert sel.support["worker"] is not None
    assert sel.support.get("worker_tier_fallback")
    # falls back to the globally-cheapest tool-capable worker
    assert sel.support["worker"]["id"] == "cheapo/ultra-cheap"


def test_subagent_without_tiers_keeps_cheapest_in_pool():
    prof = tiered_profile(quality_floor="basic", worker_tiers=None)
    sel = select("cheap_code", prof, _tier_catalog(), worker_heavy=True)
    # historical behavior: cheapest model in the (now unrestricted) pool
    assert sel.support["worker"]["id"] == "cheapo/ultra-cheap"
    assert "worker_tier" not in sel.support


def _req(**meta) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="cheap_code",
        messages=[ChatMessage(role="user", content="do a coding subtask")],
        metadata=GatewayMetadata(**meta),
    )


def test_infer_worker_heavy_light_by_default():
    assert infer_worker_heavy(_req()) is False


def test_infer_worker_heavy_on_quality_floor():
    assert infer_worker_heavy(_req(quality_floor="high")) is True
    assert infer_worker_heavy(_req(quality_floor="frontier")) is True
    assert infer_worker_heavy(_req(quality_floor="medium")) is False


def test_infer_worker_heavy_on_large_input():
    assert infer_worker_heavy(_req(context_length_estimate=50_000)) is True
    assert infer_worker_heavy(_req(context_length_estimate=10_000)) is False


def test_infer_worker_heavy_on_tools():
    assert infer_worker_heavy(_req(tool_requirements=["shell"])) is True
    req = ChatCompletionRequest(
        model="cheap_code",
        messages=[ChatMessage(role="user", content="x")],
        tools=[{"type": "function", "function": {"name": "f"}}],
    )
    assert infer_worker_heavy(req) is True
