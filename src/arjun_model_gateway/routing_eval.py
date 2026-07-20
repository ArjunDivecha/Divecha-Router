"""Offline evaluation gates for route robustness and actor-pool diversity.

The gateway must not spend money merely to decide which route to use. This
module therefore evaluates the deterministic auto-classifier against semantic
prompt perturbations and measures whether an eval-tested candidate pool is
behaviorally diverse enough to justify multiple actors.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .config import Policy
from .models import ChatCompletionRequest, ChatMessage, GatewayMetadata
from .policy import resolve_profile_name


class PromptVariant(BaseModel):
    kind: str
    prompt: str


class RoutingRobustnessCase(BaseModel):
    id: str
    expected_profile: str
    prompt: str
    variants: list[PromptVariant] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingRobustnessSuite(BaseModel):
    version: int = 1
    minimum_robustness: float = Field(default=0.95, ge=0.0, le=1.0)
    cases: list[RoutingRobustnessCase] = Field(min_length=1)


class BehaviorActor(BaseModel):
    id: str
    scores: dict[str, float]
    cost_per_verified_success_usd: float | None = Field(default=None, ge=0.0)


class BehaviorMatrix(BaseModel):
    version: int = 1
    actors: list[BehaviorActor] = Field(min_length=1)


def load_routing_suite(path: Path) -> RoutingRobustnessSuite:
    return RoutingRobustnessSuite.model_validate(yaml.safe_load(path.read_text()) or {})


def load_behavior_matrix(path: Path) -> BehaviorMatrix:
    matrix = BehaviorMatrix.model_validate(yaml.safe_load(path.read_text()) or {})
    eval_names = set(matrix.actors[0].scores)
    if not eval_names:
        raise ValueError("behavior matrix actors must contain at least one eval score")
    for actor in matrix.actors:
        if set(actor.scores) != eval_names:
            raise ValueError("all behavior actors must have the same eval score keys")
        if any(not 0.0 <= score <= 1.0 for score in actor.scores.values()):
            raise ValueError(f"actor {actor.id!r} has a score outside [0, 1]")
    return matrix


def _route_profile(prompt: str, metadata: dict[str, Any], policy: Policy) -> str:
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content=prompt)],
        metadata=GatewayMetadata.model_validate(metadata),
    )
    return resolve_profile_name(request, policy)


def _route_fingerprint(profile_name: str, policy: Policy) -> dict[str, Any]:
    profile = policy.profiles[profile_name]
    return {
        "profile": profile_name,
        "route_type": profile.route_type,
        "objective": profile.objective,
        "quality_floor": profile.quality_floor,
        "verification_required": profile.require_verification,
        "external_calls_allowed": profile.external_calls_allowed,
    }


def evaluate_routing_robustness(
    suite: RoutingRobustnessSuite,
    policy: Policy,
    *,
    minimum: float | None = None,
) -> dict[str, Any]:
    """Test semantic route equivalence without selecting or calling a model."""
    threshold = suite.minimum_robustness if minimum is None else minimum
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("minimum robustness must be between 0 and 1")

    total_variants = 0
    stable_variants = 0
    total_prompts = 0
    expected_prompts = 0
    baseline_failures = 0
    category_counts: dict[str, dict[str, int]] = {}
    case_results: list[dict[str, Any]] = []

    for case in suite.cases:
        baseline_profile = _route_profile(case.prompt, case.metadata, policy)
        baseline_fingerprint = _route_fingerprint(baseline_profile, policy)
        baseline_correct = baseline_profile == case.expected_profile
        baseline_failures += int(not baseline_correct)
        total_prompts += 1
        expected_prompts += int(baseline_correct)

        variants: list[dict[str, Any]] = []
        for variant in case.variants:
            profile_name = _route_profile(variant.prompt, case.metadata, policy)
            fingerprint = _route_fingerprint(profile_name, policy)
            equivalent = fingerprint == baseline_fingerprint
            expected = profile_name == case.expected_profile
            stable_variants += int(equivalent)
            expected_prompts += int(expected)
            total_variants += 1
            total_prompts += 1
            bucket = category_counts.setdefault(variant.kind, {"total": 0, "stable": 0})
            bucket["total"] += 1
            bucket["stable"] += int(equivalent)
            variants.append(
                {
                    "kind": variant.kind,
                    "profile": profile_name,
                    "equivalent_to_baseline": equivalent,
                    "matches_expected": expected,
                }
            )

        case_robustness = sum(v["equivalent_to_baseline"] for v in variants) / len(variants)
        case_accuracy = (
            int(baseline_correct) + sum(v["matches_expected"] for v in variants)
        ) / (len(variants) + 1)
        case_results.append(
            {
                "id": case.id,
                "expected_profile": case.expected_profile,
                "baseline_profile": baseline_profile,
                "baseline_fingerprint": baseline_fingerprint,
                "robustness": round(case_robustness, 6),
                "expected_accuracy": round(case_accuracy, 6),
                "passed": (
                    baseline_correct
                    and case_robustness >= threshold
                    and case_accuracy >= threshold
                ),
                "variants": variants,
            }
        )

    robustness = stable_variants / total_variants
    expected_accuracy = expected_prompts / total_prompts
    categories = {
        kind: {
            **counts,
            "robustness": round(counts["stable"] / counts["total"], 6),
        }
        for kind, counts in sorted(category_counts.items())
    }
    passed = (
        baseline_failures == 0
        and robustness >= threshold
        and expected_accuracy >= threshold
        and all(case["passed"] for case in case_results)
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite_version": suite.version,
        "policy_version": policy.version,
        "minimum": threshold,
        "cases": len(suite.cases),
        "variants": total_variants,
        "baseline_failures": baseline_failures,
        "route_robustness": round(robustness, 6),
        "expected_profile_accuracy": round(expected_accuracy, 6),
        "categories": categories,
        "passed": passed,
        "case_results": case_results,
    }


def cosine_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("behavior vectors must have equal length")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    similarity = sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
    return min(2.0, max(0.0, 1.0 - similarity))


def _cluster_entropy(vectors: list[list[float]], threshold: float) -> float:
    parents = list(range(len(vectors)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            if cosine_distance(vectors[left], vectors[right]) <= threshold + 1e-12:
                union(left, right)

    sizes: dict[int, int] = {}
    for index in range(len(vectors)):
        root = find(index)
        sizes[root] = sizes.get(root, 0) + 1
    count = len(vectors)
    return -sum((size / count) * math.log(size / count) for size in sizes.values())


def hierarchic_social_entropy(vectors: list[list[float]]) -> tuple[float, float]:
    """Return raw and normalized HSE using cosine-distance linkage levels."""
    if len(vectors) < 2:
        return 0.0, 0.0
    distances = [
        cosine_distance(vectors[left], vectors[right])
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    ]
    levels = sorted({0.0, *distances})
    maximum = levels[-1]
    if maximum <= 1e-12:
        return 0.0, 0.0
    raw = sum(
        _cluster_entropy(vectors, left) * (right - left)
        for left, right in zip(levels, levels[1:])
    )
    normalized = raw / (math.log(len(vectors)) * maximum)
    return raw, min(1.0, max(0.0, normalized))


def _actor_vector(actor: BehaviorActor, eval_names: list[str]) -> list[float]:
    return [actor.scores[name] for name in eval_names]


def _mean_score(actor: BehaviorActor) -> float:
    return sum(actor.scores.values()) / len(actor.scores)


def _subset_metrics(actors: list[BehaviorActor], eval_names: list[str]) -> dict[str, float | None]:
    vectors = [_actor_vector(actor, eval_names) for actor in actors]
    raw, normalized = hierarchic_social_entropy(vectors)
    known_costs = [
        actor.cost_per_verified_success_usd
        for actor in actors
        if actor.cost_per_verified_success_usd is not None
    ]
    return {
        "hse": raw,
        "normalized_hse": normalized,
        "mean_score": sum(_mean_score(actor) for actor in actors) / len(actors),
        "mean_cost_per_verified_success_usd": (
            sum(known_costs) / len(known_costs) if known_costs else None
        ),
    }


def build_diversity_report(
    matrix: BehaviorMatrix,
    *,
    max_actors: int = 8,
    min_mean_score: float = 0.0,
    max_cost_per_verified_success_usd: float | None = None,
    quality_weight: float = 0.25,
    minimum_joint_gain: float = 0.001,
) -> dict[str, Any]:
    """Select a small behaviorally distinct, quality-gated actor coreset."""
    if max_actors < 1:
        raise ValueError("max_actors must be at least 1")
    eval_names = sorted(matrix.actors[0].scores)
    eligible = [
        actor
        for actor in matrix.actors
        if _mean_score(actor) >= min_mean_score
        and (
            max_cost_per_verified_success_usd is None
            or (
                actor.cost_per_verified_success_usd is not None
                and actor.cost_per_verified_success_usd <= max_cost_per_verified_success_usd
            )
        )
    ]
    if not eligible:
        raise ValueError("no actor satisfies the quality and cost gates")

    selected: list[BehaviorActor] = []
    remaining = list(eligible)
    current_objective = float("-inf")
    while remaining and len(selected) < max_actors:
        ranked: list[tuple[float, float, float, str, BehaviorActor]] = []
        for actor in remaining:
            trial = [*selected, actor]
            metrics = _subset_metrics(trial, eval_names)
            objective = float(metrics["hse"]) + quality_weight * float(metrics["mean_score"])
            cost = actor.cost_per_verified_success_usd
            ranked.append((objective, _mean_score(actor), -(cost or 0.0), actor.id, actor))
        winner_row = max(ranked, key=lambda item: item[:4])
        winner = winner_row[-1]
        if selected and winner_row[0] <= current_objective + minimum_joint_gain:
            break
        selected.append(winner)
        remaining.remove(winner)
        current_objective = winner_row[0]

    full_metrics = _subset_metrics(eligible, eval_names)
    coreset_metrics = _subset_metrics(selected, eval_names)
    redundant_pairs: list[dict[str, Any]] = []
    for left in range(len(eligible)):
        for right in range(left + 1, len(eligible)):
            distance = cosine_distance(
                _actor_vector(eligible[left], eval_names),
                _actor_vector(eligible[right], eval_names),
            )
            if distance <= 0.05:
                redundant_pairs.append(
                    {
                        "left": eligible[left].id,
                        "right": eligible[right].id,
                        "cosine_distance": round(distance, 6),
                    }
                )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matrix_version": matrix.version,
        "evals": eval_names,
        "actor_count": len(matrix.actors),
        "eligible_actor_count": len(eligible),
        "gates": {
            "min_mean_score": min_mean_score,
            "max_cost_per_verified_success_usd": max_cost_per_verified_success_usd,
            "quality_weight": quality_weight,
            "minimum_joint_gain": minimum_joint_gain,
            "max_actors": max_actors,
        },
        "full_pool": {key: round(value, 8) if value is not None else None for key, value in full_metrics.items()},
        "coreset": {
            "actors": [actor.id for actor in selected],
            **{
                key: round(value, 8) if value is not None else None
                for key, value in coreset_metrics.items()
            },
        },
        "redundant_pairs": redundant_pairs,
    }


def write_versioned_report(report: dict[str, Any], output: Path) -> list[Path]:
    """Write a stable latest report and a timestamped comparison artifact."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output.write_text(rendered)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    history = output.with_name(f"{output.stem}_{stamp}{output.suffix}")
    history.write_text(rendered)
    return [output, history]
