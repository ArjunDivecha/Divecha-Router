import pytest

from arjun_model_gateway.model_intel import ModelIntel, estimate_model_cost


def test_estimate_model_cost_uses_prompt_and_completion_prices():
    model = {"pricing": {"prompt": "0.000001", "completion": "0.000003"}}
    assert estimate_model_cost(model, input_tokens=1000, output_tokens=500) == 0.0025


def test_rank_candidates_uses_success_prior(tmp_path, monkeypatch):
    priors = tmp_path / "priors.yaml"
    priors.write_text(
        """
default_success_prior: 0.50
profiles:
  code:
    expensive-good: 0.90
    cheap-bad: 0.10
"""
    )
    snapshot = tmp_path / "snapshot.json"
    intel = ModelIntel(ttl_seconds=999, snapshot_path=snapshot, success_priors_path=priors)
    monkeypatch.setattr(
        intel,
        "_fetch_catalog",
        lambda: [
            {
                "id": "expensive-good",
                "context_length": 1000000,
                "supported_parameters": ["tools"],
                "pricing": {"prompt": "0.000001", "completion": "0.000001"},
                "architecture": {"output_modalities": ["text"]},
            },
            {
                "id": "cheap-bad",
                "context_length": 1000000,
                "supported_parameters": ["tools"],
                "pricing": {"prompt": "0.0000001", "completion": "0.0000001"},
                "architecture": {"output_modalities": ["text"]},
            },
        ],
    )
    ranked = intel.rank_candidates(
        {
            "candidate_models": ["expensive-good", "cheap-bad"],
            "expected_input_tokens": 1000,
            "expected_output_tokens": 1000,
            "supported_parameters": "tools",
        },
        task_label="code",
    )
    assert ranked[0]["model"]["id"] == "cheap-bad"
    assert ranked[0]["expected_cost_per_success_usd"] == pytest.approx(0.002)
