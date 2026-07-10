"""
=============================================================================
MODULE: tests/test_selector.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
Deterministic tests for the dynamic selection engine: constraint filters,
quality floors as intelligence-rank slices, cost-per-verified-success
scoring, learning from recorded outcomes, budget enforcement (FAIL IS FAIL),
and full route-graph pricing for fusion routes.

INPUT FILES: none — this module performs no file I/O.
OUTPUT FILES: none — this module performs no file I/O.
=============================================================================
"""

from __future__ import annotations

import pytest

from arjun_model_gateway.config import PolicyProfile
from arjun_model_gateway.selector import RoutingError, select


def simple_profile(**overrides) -> PolicyProfile:
    base = {
        "route_type": "simple",
        "objective": "cheapest_success",
        "quality_floor": "basic",
        "select": {"min_context": 32_000, "pool_size": 8},
        "expected_tokens": {"input": 8_000, "output": 1_000},
    }
    base.update(overrides)
    return PolicyProfile.model_validate(base)


def test_cheapest_success_prefers_cheap_model(fake_catalog):
    selection = select("cheap", simple_profile(), fake_catalog)
    assert selection.primary.model_id == "delta/cheapest-5"
    # free teaser variants never qualify
    assert all(":free" not in c.model_id for c in selection.candidates)


def test_recorded_failures_reprice_a_model_out(fake_catalog):
    # 20 verified failures for the cheap model should push selection to the
    # next-best expected-cost-per-success candidate.
    counts = {("cheap", "delta/cheapest-5"): (0, 20)}
    selection = select("cheap", simple_profile(), fake_catalog, counts=counts)
    assert selection.primary.model_id != "delta/cheapest-5"
    losing = next(c for c in selection.candidates if c.model_id == "delta/cheapest-5")
    assert losing.failures == 20
    assert losing.p_success < 0.2


def test_recorded_successes_strengthen_a_model(fake_catalog):
    counts = {("cheap", "gamma/cheap-4"): (30, 0)}
    selection = select("cheap", simple_profile(), fake_catalog, counts=counts)
    winner = next(c for c in selection.candidates if c.model_id == "gamma/cheap-4")
    assert winner.p_success > 0.9


def test_budget_gate_fails_loudly(fake_catalog):
    with pytest.raises(RoutingError) as exc:
        select("cheap", simple_profile(), fake_catalog, budget_usd=0.0000001)
    assert exc.value.code == "budget_exceeded"


def test_quality_floor_high_restricts_to_top_quarter(fake_catalog):
    # 8 models -> "high" = top 25% = top 2 intelligence ranks
    selection = select("frontier", simple_profile(quality_floor="high"), fake_catalog)
    assert all(c.intelligence_rank < 2 for c in selection.candidates)


def test_required_parameters_filter(fake_catalog):
    selection = select(
        "structured",
        simple_profile(),
        fake_catalog,
        extra_required_parameters=["structured_outputs"],
    )
    assert all("no-tools" not in c.model_id for c in selection.candidates)


def test_min_context_filter(fake_catalog):
    selection = select("long", simple_profile(), fake_catalog, min_context_override=100_000)
    ids = [c.model_id for c in selection.candidates]
    assert "delta/cheapest-5" not in ids  # 64k context
    assert "eta/small-context-8" not in ids  # 8k context


def test_impossible_constraints_raise_no_candidates(fake_catalog):
    with pytest.raises(RoutingError) as exc:
        select("cheap", simple_profile(), fake_catalog, min_context_override=10_000_000)
    assert exc.value.code == "no_candidates"


def test_fastest_objective_orders_by_latency_rank(fake_catalog):
    profile = simple_profile(objective="fastest")
    selection = select("fast", profile, fake_catalog)
    ranks = [c.latency_rank for c in selection.candidates]
    assert ranks == sorted(ranks)
    assert selection.primary.model_id == "delta/cheapest-5"  # fastest fake


def test_fastest_objective_learns_from_observed_wall_time(fake_catalog):
    # delta wins the catalog latency sort, but we've MEASURED it slow for this
    # profile (e.g. slow reasoning orchestrator) -> unobserved models win
    profile = simple_profile(objective="fastest")
    slow = {("fast", "delta/cheapest-5"): (3, 600_000.0)}  # 3 calls, 600s avg
    selection = select("fast", profile, fake_catalog, latency_obs=slow)
    assert selection.primary.model_id != "delta/cheapest-5"
    loser = next(c for c in selection.candidates if c.model_id == "delta/cheapest-5")
    assert loser.expected_seconds == 600.0 and loser.observed_calls == 3


def test_smartest_objective_orders_by_intelligence(fake_catalog):
    profile = simple_profile(objective="smartest")
    selection = select("frontier", profile, fake_catalog)
    assert selection.primary.model_id == "alpha/frontier-1"


def test_fusion_route_graph_prices_panel_and_judge(fake_catalog):
    profile = PolicyProfile.model_validate(
        {
            "route_type": "fusion",
            "objective": "smartest",
            "quality_floor": "basic",
            "select": {"min_context": 32_000, "pool_size": 8},
            "expected_tokens": {"input": 10_000, "output": 2_000},
            "fusion": {"panel_size": 3},
        }
    )
    selection = select("research", profile, fake_catalog)
    graph = selection.primary.route_graph
    roles = [node.role for node in graph]
    assert roles.count("panel") == 3
    assert "judge" in roles and "outer" in roles
    # panel members come from distinct authors
    panel_authors = [n.model_id.split("/")[0] for n in graph if n.role == "panel"]
    assert len(panel_authors) == len(set(panel_authors))
    outer_cost = next(n.cost_usd for n in graph if n.role == "outer")
    assert selection.primary.route_cost_usd > outer_cost


def test_reasoning_off_excludes_mandatory_reasoning_models():
    from tests.conftest import FAKE_LATENCY_ORDER, FAKE_MODELS, FakeCatalog, make_model

    thinker = make_model("theta/mandatory-thinker", 0.01, 0.03)
    thinker["reasoning"] = {"mandatory": True}
    catalog = FakeCatalog([*FAKE_MODELS, thinker], FAKE_LATENCY_ORDER)
    profile = simple_profile(reasoning="off")
    selection = select("cheap", profile, catalog)
    ids = [c.model_id for c in selection.candidates]
    assert "theta/mandatory-thinker" not in ids
    # and without the constraint the ultra-cheap thinker would win on cost
    relaxed = select("cheap", simple_profile(), catalog)
    assert relaxed.primary.model_id == "theta/mandatory-thinker"


def test_subagent_route_picks_cheapest_worker(fake_catalog):
    profile = PolicyProfile.model_validate(
        {
            "route_type": "subagent",
            "objective": "cheapest_success",
            "quality_floor": "basic",
            "select": {"min_context": 32_000, "pool_size": 8},
        }
    )
    selection = select("long_context", profile, fake_catalog)
    worker = next(n for n in selection.primary.route_graph if n.role == "worker")
    assert worker.model_id == "delta/cheapest-5"
