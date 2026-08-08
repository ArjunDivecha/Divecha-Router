"""
=============================================================================
MODULE: tests/test_supplemental.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
Tests for the supplemental catalog — non-OpenRouter models (e.g. Celeris)
merged into marketplace selection. Covers: the merge splices a model into the
intelligence/latency rankings at its declared percentiles and flags it
supplemental; a SIMPLE-route profile (fast) can select it and the resulting
route dispatches to the model's own provider; and NON-simple routes (advisor/
subagent) never select it as outer OR as a support model.

INPUT FILES:
- a temporary supplemental_catalog.yaml + snapshot file under pytest tmp_path.
OUTPUT FILES: none — this module performs no file I/O beyond the tmp fixtures.
=============================================================================
"""

from __future__ import annotations

import pytest

from arjun_model_gateway.catalog import Catalog, is_supplemental
from arjun_model_gateway.config import PolicyProfile, validate_policy
from arjun_model_gateway.models import ChatCompletionRequest, ChatMessage, GatewayMetadata
from arjun_model_gateway.policy import choose_route
from arjun_model_gateway.selector import select
from tests.conftest import FAKE_LATENCY_ORDER, FAKE_MODELS

SUPP_YAML = """
models:
  - id: celeris-1
    provider: celeris
    prompt_per_million: 0.20
    completion_per_million: 0.70
    context_length: 131072
    supported_parameters: [tools, structured_outputs, response_format]
    output_modalities: [text]
    intelligence_percentile: 0.35
    latency_percentile: 0.02
"""


def _catalog(tmp_path, monkeypatch) -> Catalog:
    supp = tmp_path / "supp.yaml"
    supp.write_text(SUPP_YAML)
    cat = Catalog(
        ttl_seconds=999, snapshot_path=tmp_path / "snap.json", supplemental_path=supp
    )
    monkeypatch.setattr(cat, "_fetch_live", lambda: (FAKE_MODELS, FAKE_LATENCY_ORDER))
    return cat


def simple_profile(**overrides) -> PolicyProfile:
    base = {
        "route_type": "simple",
        "objective": "fastest",
        "quality_floor": "medium",
        "select": {"min_context": 32_000, "pool_size": 8},
        "expected_tokens": {"input": 4_000, "output": 800},
    }
    base.update(overrides)
    return PolicyProfile.model_validate(base)


def test_supplemental_is_merged_and_flagged(tmp_path, monkeypatch):
    cat = _catalog(tmp_path, monkeypatch)
    ids = [m["id"] for m in cat.models()]
    assert "celeris-1" in ids
    # intelligence percentile 0.35 * 8 == 2.8 -> inserted at index 3
    assert cat.intelligence_rank("celeris-1") == 3
    # latency percentile 0.02 -> front of the latency ranking
    assert cat.latency_rank("celeris-1") == 0
    assert is_supplemental(cat.get("celeris-1"))
    # native models still present and shifted, not dropped
    assert "alpha/frontier-1" in ids


def test_simple_route_can_select_supplemental(tmp_path, monkeypatch):
    cat = _catalog(tmp_path, monkeypatch)
    sel = select("fast", simple_profile(), cat)
    picked = [c.model_id for c in sel.candidates]
    assert "celeris-1" in picked
    # fastest objective + latency_rank 0 -> it should be the winner
    assert sel.primary.model_id == "celeris-1"


def test_non_simple_route_excludes_supplemental(tmp_path, monkeypatch):
    cat = _catalog(tmp_path, monkeypatch)
    advisor = PolicyProfile.model_validate(
        {
            "route_type": "advisor",
            "objective": "cheapest_success",
            "quality_floor": "basic",
            "select": {"require_parameters": ["tools"], "min_context": 32_000, "pool_size": 8},
            "expected_tokens": {"input": 4_000, "output": 800},
        }
    )
    sel = select("code", advisor, cat)
    picked = [c.model_id for c in sel.candidates]
    assert "celeris-1" not in picked  # never an outer on a non-simple route
    assert sel.support.get("advisor", {}).get("id") != "celeris-1"  # nor a support model


def test_choose_route_dispatches_to_supplemental_provider(tmp_path, monkeypatch):
    monkeypatch.setattr("arjun_model_gateway.policy.success_counts", lambda: ({}, {}))
    monkeypatch.setattr("arjun_model_gateway.policy.manual_seeds", lambda name: {})
    monkeypatch.setattr("arjun_model_gateway.policy.latency_observations", lambda: {})
    cat = _catalog(tmp_path, monkeypatch)
    policy = validate_policy()
    request = ChatCompletionRequest(
        model="fast",
        messages=[ChatMessage(role="user", content="classify: great")],
        metadata=GatewayMetadata(routing_profile="fast"),
    )
    decision = choose_route(request, policy, catalog=cat)
    assert decision.model == "celeris-1"
    assert decision.provider == "celeris"  # dispatched to its own adapter
    assert any("supplemental_provider=celeris" in r for r in decision.reasons)
