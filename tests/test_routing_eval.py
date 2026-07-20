"""Deterministic tests for semantic route stability and actor diversity."""

from __future__ import annotations

from pathlib import Path

from arjun_model_gateway.config import validate_policy
from arjun_model_gateway.models import ChatCompletionRequest, ChatMessage
from arjun_model_gateway.policy import resolve_profile_name
from arjun_model_gateway.routing_eval import (
    BehaviorActor,
    BehaviorMatrix,
    PromptVariant,
    RoutingRobustnessCase,
    RoutingRobustnessSuite,
    build_diversity_report,
    evaluate_routing_robustness,
    hierarchic_social_entropy,
    load_routing_suite,
)

ROOT = Path(__file__).parents[1]


def _profile(prompt: str) -> str:
    request = ChatCompletionRequest(messages=[ChatMessage(role="user", content=prompt)])
    return resolve_profile_name(request, validate_policy())


def test_finance_and_code_paraphrases_keep_the_intended_route():
    assert _profile("Analyze the risk in this investment portfolio.") == "finance"
    assert _profile("Assess how exposed this collection of holdings is to loss.") == "finance"
    assert _profile("Debug this repo diff and run pytest.") == "code"
    assert _profile("Find why this software change fails and correct it.") == "code"


def test_default_robustness_suite_passes_all_perturbation_families():
    suite = load_routing_suite(ROOT / "evals/routing_robustness.yaml")
    report = evaluate_routing_robustness(suite, validate_policy())
    assert report["passed"] is True
    assert report["route_robustness"] == 1.0
    assert report["expected_profile_accuracy"] == 1.0
    assert set(report["categories"]) == {
        "character",
        "word",
        "syntax",
        "paraphrase",
        "rambling",
    }


def test_stable_but_wrong_routing_does_not_pass_the_gate():
    suite = RoutingRobustnessSuite(
        minimum_robustness=0.9,
        cases=[
            RoutingRobustnessCase(
                id="wrong-route",
                expected_profile="finance",
                prompt="Shorten this sentence.",
                variants=[PromptVariant(kind="paraphrase", prompt="Condense this wording.")],
            )
        ],
    )
    report = evaluate_routing_robustness(suite, validate_policy())
    assert report["route_robustness"] == 1.0
    assert report["expected_profile_accuracy"] == 0.0
    assert report["passed"] is False


def test_hse_is_zero_for_identical_behavior_and_positive_for_diversity():
    identical = hierarchic_social_entropy([[0.9, 0.7], [0.9, 0.7]])
    diverse = hierarchic_social_entropy([[1.0, 0.0], [0.0, 1.0]])
    assert identical == (0.0, 0.0)
    assert diverse[0] > 0.0
    assert diverse[1] == 1.0


def test_diversity_coreset_enforces_quality_and_cost_gates():
    matrix = BehaviorMatrix(
        actors=[
            BehaviorActor(
                id="balanced",
                scores={"code": 0.88, "research": 0.86},
                cost_per_verified_success_usd=0.10,
            ),
            BehaviorActor(
                id="code-specialist",
                scores={"code": 0.99, "research": 0.72},
                cost_per_verified_success_usd=0.14,
            ),
            BehaviorActor(
                id="too-expensive",
                scores={"code": 0.98, "research": 0.98},
                cost_per_verified_success_usd=1.20,
            ),
            BehaviorActor(
                id="too-weak",
                scores={"code": 0.40, "research": 0.42},
                cost_per_verified_success_usd=0.01,
            ),
        ]
    )
    report = build_diversity_report(
        matrix,
        max_actors=2,
        min_mean_score=0.75,
        max_cost_per_verified_success_usd=0.20,
    )
    assert report["eligible_actor_count"] == 2
    assert set(report["coreset"]["actors"]) == {"balanced", "code-specialist"}
    assert report["coreset"]["mean_cost_per_verified_success_usd"] == 0.12


def test_redundant_actors_do_not_earn_extra_coreset_slots():
    matrix = BehaviorMatrix(
        actors=[
            BehaviorActor(id="first", scores={"a": 0.9, "b": 0.8}),
            BehaviorActor(id="duplicate", scores={"a": 0.9, "b": 0.8}),
        ]
    )
    report = build_diversity_report(matrix, max_actors=8)
    assert len(report["coreset"]["actors"]) == 1
    assert report["redundant_pairs"] == [
        {"left": "first", "right": "duplicate", "cosine_distance": 0.0}
    ]
