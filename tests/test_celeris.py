"""
=============================================================================
MODULE: tests/test_celeris.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
Tests for the Celeris-1 direct provider integration (the `fast_celeris`
profile): the env-key whitelist accepts CELERIS_API_KEY with its `ck_`
prefix, adapter_for("celeris") builds an OpenAI-compatible adapter pointed at
the Celeris endpoint, and choose_route on the fast_celeris profile produces a
direct route whose model is resolved live (mocked here — no network).

INPUT FILES:
- a temporary env file created under pytest's tmp_path (env-whitelist test).
OUTPUT FILES: none — this module performs no file I/O beyond the tmp fixture.
=============================================================================
"""

from __future__ import annotations

from arjun_model_gateway.adapters import OpenAICompatibleAdapter, adapter_for
from arjun_model_gateway.env_loader import load_provider_keys
from arjun_model_gateway.models import ChatCompletionRequest, ChatMessage, GatewayMetadata
from arjun_model_gateway.policy import choose_route
from arjun_model_gateway.config import validate_policy


def test_env_whitelist_accepts_celeris_key(tmp_path):
    env = tmp_path / "env.txt"
    env.write_text(
        "CELERIS_API_KEY=ck_live_abc123\n"
        "CELERIS_API_KEY=ck_second_ignored\n"  # first wins
        "BOGUS_KEY=whatever\n"  # not whitelisted
    )
    loaded = load_provider_keys(path=env)
    assert "CELERIS_API_KEY" in loaded
    assert "BOGUS_KEY" not in loaded


def test_env_whitelist_rejects_wrong_prefix(tmp_path):
    env = tmp_path / "env.txt"
    env.write_text("CELERIS_API_KEY=sk-not-a-celeris-key\n")  # wrong prefix
    loaded = load_provider_keys(path=env)
    assert "CELERIS_API_KEY" not in loaded


def test_adapter_for_celeris_is_openai_compatible():
    adapter = adapter_for("celeris")
    assert isinstance(adapter, OpenAICompatibleAdapter)
    assert adapter.provider == "celeris"
    assert adapter.base_url.endswith("celeris-1/v1")
    assert adapter.api_key_env == "CELERIS_API_KEY"


def test_fast_celeris_routes_direct(fake_catalog, monkeypatch):
    # model is resolved live from Celeris /models — mock it, no network
    monkeypatch.setattr(
        "arjun_model_gateway.provider_models.resolve_direct_model",
        lambda provider: "celeris-1",
    )
    policy = validate_policy()
    request = ChatCompletionRequest(
        model="fast_celeris",
        messages=[ChatMessage(role="user", content="classify: great product")],
        metadata=GatewayMetadata(routing_profile="fast_celeris"),
    )
    decision = choose_route(request, policy, catalog=fake_catalog)
    assert decision.route_type == "direct"
    assert decision.provider == "celeris"
    assert decision.model == "celeris-1"  # resolved live, never pinned in policy
