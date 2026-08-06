"""
=============================================================================
MODULE: selector.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
The selection engine. Given a routing profile (pure constraints + an
objective — never a model id) and the live catalog, it:

  1. filters the catalog by hard constraints (context, capabilities, price
     cap, text output, no ':free' teaser variants),
  2. applies the quality floor as a slice of the live intelligence ranking
     (basic = any model, medium = top 60%, high = top 25%, frontier = top 8),
  3. prices the FULL route graph for each candidate — for fusion routes that
     means outer model + analysis panel + judge, not just the outer model,
  4. scores candidates by expected cost per verified success, where
     P(success) is a Beta posterior seeded from the model's intelligence
     rank and updated by real recorded outcomes (telemetry `outcomes` table),
  5. enforces the request budget: candidates whose estimated route cost
     exceeds the budget are dropped; if none survive, RoutingError (FAIL IS
     FAIL — the gateway returns HTTP 400, it does not silently degrade),
  6. orders by the profile objective: cheapest_success / fastest / smartest.

Support models for server-tool routes (advisor model, fusion panel + judge,
subagent worker) are ALSO resolved dynamically from the catalog at decision
time, so they track the current frontier automatically.

INPUT FILES: none directly (catalog and outcome counts are passed in).
OUTPUT FILES: none.
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .catalog import (
    Catalog,
    author_of,
    completion_price,
    is_free_variant,
    outputs_text,
    prompt_price,
)
from .config import PolicyProfile

# Quality floor -> allowed fraction of the intelligence ranking (top slice).
QUALITY_FLOOR_FRACTION = {"basic": 1.0, "medium": 0.60, "high": 0.25}
FRONTIER_TOP_N = 8

# Beta prior seeded from intelligence rank: top-ranked models start near 0.92,
# bottom-ranked near 0.57. Six pseudo-observations, so ~6 real outcomes start
# to dominate the prior.
PRIOR_PSEUDO_COUNT = 6.0
MANUAL_SEED_WEIGHT = 6.0

# Route-graph token heuristics (documented estimates, logged per decision):
# advisor sees a condensed problem statement plus the draft answer;
# the fusion judge reads every panel answer plus a slice of the prompt.
ADVISOR_INPUT_FACTOR = 0.5
JUDGE_PROMPT_FRACTION = 0.25

# `fastest` ordering: models with observed wall times use their mean; models
# never tried under this profile get an optimistic prior so they get tried —
# a slow-observed model loses its slot and the router self-corrects.
# (OpenRouter's latency sort is TTFT-based and mispredicts reasoning models.)
DEFAULT_EXPECTED_SECONDS = 120.0


class RoutingError(RuntimeError):
    """No candidate satisfies the constraints/budget. Surfaced as HTTP 400."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class RouteNode:
    role: str  # outer | advisor | panel | judge | worker
    model_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class Candidate:
    model_id: str
    model: dict[str, Any]
    intelligence_rank: int
    latency_rank: int
    rank_pct: float
    p_success: float
    successes: int
    failures: int
    route_cost_usd: float
    route_graph: list[RouteNode]
    score: float
    expected_seconds: float = DEFAULT_EXPECTED_SECONDS
    observed_calls: int = 0


@dataclass
class Selection:
    profile_name: str
    objective: str
    quality_floor: str
    budget_usd: float | None
    candidates: list[Candidate]
    pool_considered: int
    catalog_as_of: str | None
    catalog_source: str
    support: dict[str, Any] = field(default_factory=dict)

    @property
    def primary(self) -> Candidate:
        return self.candidates[0]


def floor_max_rank(quality_floor: str, catalog_size: int) -> int:
    """Highest allowed intelligence rank (exclusive) for a quality floor."""
    if quality_floor == "frontier":
        return FRONTIER_TOP_N
    fraction = QUALITY_FLOOR_FRACTION.get(quality_floor, 1.0)
    return max(1, int(catalog_size * fraction))


def passes_filters(
    model: dict[str, Any],
    *,
    require_parameters: list[str],
    min_context: int,
    max_prompt_price_per_million: float | None,
    include_free: bool,
    reasoning_off: bool = False,
) -> bool:
    model_id = str(model.get("id") or "")
    if model_id.startswith("openrouter/"):
        return False  # router meta-endpoints (auto/fusion/free/...), not real models
    if prompt_price(model) < 0 or completion_price(model) < 0:
        return False  # sentinel pricing on meta entries
    if not outputs_text(model):
        return False
    if not include_free and is_free_variant(model):
        return False
    if reasoning_off and (model.get("reasoning") or {}).get("mandatory"):
        return False  # profile demands no hidden reasoning; this model can't comply
    if min_context and int(model.get("context_length") or 0) < min_context:
        return False
    supported = set(model.get("supported_parameters") or [])
    if require_parameters and not set(require_parameters).issubset(supported):
        return False
    if max_prompt_price_per_million is not None:
        if prompt_price(model) * 1_000_000 > max_prompt_price_per_million:
            return False
    return True


def prior_alpha_beta(rank_pct: float) -> tuple[float, float]:
    mean = min(0.95, max(0.50, 0.92 - 0.35 * rank_pct))
    return mean * PRIOR_PSEUDO_COUNT, (1.0 - mean) * PRIOR_PSEUDO_COUNT


def posterior_p_success(
    profile_name: str,
    model_id: str,
    rank_pct: float,
    counts: dict[tuple[str, str], tuple[int, int]],
    global_counts: dict[str, tuple[int, int]],
    manual_seeds: dict[str, float] | None = None,
) -> tuple[float, int, int]:
    """Beta posterior mean of P(verified success) for (profile, model)."""
    alpha, beta = prior_alpha_beta(rank_pct)
    seed = (manual_seeds or {}).get(model_id)
    if seed is not None:
        alpha += float(seed) * MANUAL_SEED_WEIGHT
        beta += (1.0 - float(seed)) * MANUAL_SEED_WEIGHT
    successes, failures = counts.get((profile_name, model_id)) or global_counts.get(model_id) or (0, 0)
    mean = (alpha + successes) / (alpha + beta + successes + failures)
    return mean, successes, failures


def node_cost(model: dict[str, Any], input_tokens: int, output_tokens: int) -> float:
    return input_tokens * prompt_price(model) + output_tokens * completion_price(model)


def _cheapest(models: list[dict[str, Any]]) -> dict[str, Any] | None:
    return min(
        models, key=lambda m: prompt_price(m) + completion_price(m), default=None
    )


def select_worker(
    profile: PolicyProfile,
    pool: list[dict[str, Any]],
    filtered_all: list[dict[str, Any]],
    worker_heavy: bool,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Pick the subagent worker, honoring optional two-tier preferences.

    Without `worker_tiers` this is the historical behavior: the single cheapest
    model in the floor-restricted pool. With tiers, the request's difficulty
    picks light vs heavy, and the worker is the cheapest model that matches the
    tier's soft author + name preference (drawn from `filtered_all`, so a cheap
    worker below the boss's quality floor is still eligible). If the tier match
    is empty it degrades to the author-only pool, then to the floor pool —
    never a hard failure just because a preference is currently unmet.

    Returns (worker_model, tier_name, fallback_note)."""
    tiers = profile.worker_tiers
    if tiers is None:
        return _cheapest(pool), None, None

    tier_name = "heavy" if worker_heavy else "light"
    tier = getattr(tiers, tier_name) or (tiers.light if worker_heavy else tiers.heavy)
    if tier is None:
        return _cheapest(pool), None, "no_tier_defined"

    def matches(model: dict[str, Any], *, use_name: bool) -> bool:
        model_id = str(model.get("id") or "")
        if tier.prefer_authors and author_of(model_id) not in tier.prefer_authors:
            return False
        if use_name and tier.name_contains and tier.name_contains.lower() not in model_id.lower():
            return False
        if (
            tier.max_prompt_price_per_million is not None
            and prompt_price(model) * 1_000_000 > tier.max_prompt_price_per_million
        ):
            return False
        return True

    def pick(models: list[dict[str, Any]]) -> dict[str, Any] | None:
        # `filtered_all` is smartest-first (catalog order), so the first match
        # is the smartest; cheapest scans for min route price.
        if tier.pick == "smartest":
            return models[0] if models else None
        return _cheapest(models)

    exact = pick([m for m in filtered_all if matches(m, use_name=True)])
    if exact is not None:
        return exact, tier_name, None

    author_only = pick(
        [
            m
            for m in filtered_all
            if not tier.prefer_authors or author_of(str(m.get("id") or "")) in tier.prefer_authors
        ]
    )
    if author_only is not None:
        return author_only, tier_name, "tier_match_empty->author_pool"

    # last resort: cheapest model meeting the hard constraints (a worker is a
    # cheap helper — the boss's quality floor should not bind it).
    return _cheapest(filtered_all) or _cheapest(pool), tier_name, "tier_match_empty->constraint_pool"


def resolve_support_models(
    profile: PolicyProfile,
    catalog: Catalog,
    pool: list[dict[str, Any]],
    filtered_all: list[dict[str, Any]],
    worker_heavy: bool = False,
) -> dict[str, Any]:
    """Pick advisor/judge/panel/worker models dynamically from the live catalog.

    `pool` is floor-restricted; `filtered_all` satisfies the same hard
    constraints but ignores the quality floor — the fusion panel extends into
    it (smartest-first) when the floor pool lacks author diversity."""
    support: dict[str, Any] = {}
    ranked = [
        m
        for m in catalog.models()  # smartest first
        if not str(m.get("id") or "").startswith("openrouter/")
        and prompt_price(m) >= 0
        and completion_price(m) >= 0
    ]
    if profile.route_type in ("advisor",):
        advisor = next(
            (m for m in ranked if "tools" in (m.get("supported_parameters") or []) and outputs_text(m)),
            None,
        )
        if advisor is None:
            raise RoutingError("no_advisor_model", "no tool-capable model in catalog for advisor route")
        support["advisor"] = advisor
    if profile.route_type == "fusion":
        judge = next((m for m in ranked if outputs_text(m) and not is_free_variant(m)), None)
        if judge is None:
            raise RoutingError("no_judge_model", "no text model in catalog for fusion judge")
        support["judge"] = judge
        panel: list[dict[str, Any]] = []
        seen_authors: set[str] = set()
        for model in [*pool, *filtered_all]:
            author = author_of(str(model.get("id")))
            if author in seen_authors:
                continue
            seen_authors.add(author)
            panel.append(model)
            if len(panel) >= profile.fusion.panel_size:
                break
        if len(panel) < 2:
            raise RoutingError(
                "fusion_panel_too_small",
                f"only {len(panel)} distinct-author models satisfy the fusion constraints",
            )
        support["panel"] = panel
    if profile.route_type == "subagent":
        worker, tier_name, fallback = select_worker(profile, pool, filtered_all, worker_heavy)
        if worker is None:
            raise RoutingError("no_worker_model", "no model satisfies the subagent constraints")
        support["worker"] = worker
        if tier_name:
            support["worker_tier"] = tier_name
        if fallback:
            support["worker_tier_fallback"] = fallback
    return support


def route_graph_for(
    profile: PolicyProfile,
    outer: dict[str, Any],
    support: dict[str, Any],
    expected_in: int,
    expected_out: int,
) -> list[RouteNode]:
    nodes = [
        RouteNode(
            role="outer",
            model_id=str(outer.get("id")),
            input_tokens=expected_in,
            output_tokens=expected_out,
            cost_usd=node_cost(outer, expected_in, expected_out),
        )
    ]
    if profile.route_type == "advisor" and "advisor" in support:
        advisor = support["advisor"]
        adv_in = int(ADVISOR_INPUT_FACTOR * expected_in + expected_out)
        nodes.append(
            RouteNode(
                role="advisor",
                model_id=str(advisor.get("id")),
                input_tokens=adv_in,
                output_tokens=expected_out,
                cost_usd=node_cost(advisor, adv_in, expected_out),
            )
        )
    if profile.route_type == "fusion":
        for member in support.get("panel", []):
            nodes.append(
                RouteNode(
                    role="panel",
                    model_id=str(member.get("id")),
                    input_tokens=expected_in,
                    output_tokens=expected_out,
                    cost_usd=node_cost(member, expected_in, expected_out),
                )
            )
        judge = support.get("judge")
        if judge is not None:
            judge_in = int(
                JUDGE_PROMPT_FRACTION * expected_in + len(support.get("panel", [])) * expected_out
            )
            nodes.append(
                RouteNode(
                    role="judge",
                    model_id=str(judge.get("id")),
                    input_tokens=judge_in,
                    output_tokens=expected_out,
                    cost_usd=node_cost(judge, judge_in, expected_out),
                )
            )
    if profile.route_type == "subagent" and "worker" in support:
        worker = support["worker"]
        nodes.append(
            RouteNode(
                role="worker",
                model_id=str(worker.get("id")),
                input_tokens=expected_in,
                output_tokens=expected_out,
                cost_usd=node_cost(worker, expected_in, expected_out),
            )
        )
    return nodes


def select(
    profile_name: str,
    profile: PolicyProfile,
    catalog: Catalog,
    *,
    quality_floor: str | None = None,
    min_context_override: int = 0,
    extra_required_parameters: list[str] | None = None,
    budget_usd: float | None = None,
    counts: dict[tuple[str, str], tuple[int, int]] | None = None,
    global_counts: dict[str, tuple[int, int]] | None = None,
    manual_seeds: dict[str, float] | None = None,
    latency_obs: dict[tuple[str, str], tuple[int, float]] | None = None,
    worker_heavy: bool = False,
) -> Selection:
    """Resolve a profile's constraints against the live catalog."""
    spec = profile.select
    floor = quality_floor or profile.quality_floor
    if floor not in ("basic", "medium", "high", "frontier"):
        raise RoutingError("invalid_quality_floor", f"unknown quality_floor {floor!r}")
    required = sorted({*spec.require_parameters, *(extra_required_parameters or [])})
    min_context = max(spec.min_context, min_context_override)
    effective_budget = budget_usd if budget_usd is not None else profile.default_budget_usd

    models = catalog.models()
    max_rank = floor_max_rank(floor, len(models))
    filtered_all = [
        model
        for model in models
        if passes_filters(
            model,
            require_parameters=required,
            min_context=min_context,
            max_prompt_price_per_million=spec.max_prompt_price_per_million,
            include_free=spec.include_free,
            reasoning_off=profile.reasoning == "off",
        )
    ]
    pool = [
        model for model in filtered_all if catalog.intelligence_rank(str(model.get("id"))) < max_rank
    ]
    if not pool:
        raise RoutingError(
            "no_candidates",
            f"profile={profile_name}: no model in the live catalog satisfies "
            f"floor={floor} (top {max_rank} of {len(models)}), required={required}, "
            f"min_context={min_context}, "
            f"max_prompt_price=${spec.max_prompt_price_per_million}/M",
        )

    support = resolve_support_models(profile, catalog, pool, filtered_all, worker_heavy)
    expected_in = profile.expected_tokens.input
    expected_out = profile.expected_tokens.output

    # truncate the pool by the OBJECTIVE's ordering, not always by intelligence:
    # a fastest route must see the quick mid-pool models, not just the smartest
    # N (which for frontier floors are all slow reasoners)
    def observed_seconds(model: dict[str, Any]) -> float:
        obs = (latency_obs or {}).get((profile_name, str(model.get("id"))))
        return obs[1] / 1000.0 if obs else DEFAULT_EXPECTED_SECONDS

    if profile.objective == "fastest":
        shortlist = sorted(
            pool, key=lambda m: (observed_seconds(m), catalog.latency_rank(str(m.get("id"))))
        )[: max(spec.pool_size, 1)]
    else:
        shortlist = pool[: max(spec.pool_size, 1)]

    candidates: list[Candidate] = []
    for model in shortlist:
        model_id = str(model.get("id"))
        rank = catalog.intelligence_rank(model_id)
        rank_pct = rank / max(len(models), 1)
        graph = route_graph_for(profile, model, support, expected_in, expected_out)
        route_cost = sum(node.cost_usd for node in graph)
        p, successes, failures = posterior_p_success(
            profile_name, model_id, rank_pct, counts or {}, global_counts or {}, manual_seeds
        )
        observed = (latency_obs or {}).get((profile_name, model_id))
        candidates.append(
            Candidate(
                model_id=model_id,
                model=model,
                intelligence_rank=rank,
                latency_rank=catalog.latency_rank(model_id),
                rank_pct=rank_pct,
                p_success=p,
                successes=successes,
                failures=failures,
                route_cost_usd=route_cost,
                route_graph=graph,
                score=route_cost / max(p, 0.01),
                expected_seconds=(observed[1] / 1000.0) if observed else DEFAULT_EXPECTED_SECONDS,
                observed_calls=observed[0] if observed else 0,
            )
        )

    if effective_budget is not None:
        affordable = [c for c in candidates if c.route_cost_usd <= effective_budget]
        if not affordable:
            cheapest = min(candidates, key=lambda c: c.route_cost_usd)
            raise RoutingError(
                "budget_exceeded",
                f"profile={profile_name}: cheapest qualifying route "
                f"({cheapest.model_id}) is estimated at ${cheapest.route_cost_usd:.4f}, "
                f"over the ${effective_budget:.4f} budget",
            )
        candidates = affordable

    if profile.objective == "fastest":
        candidates.sort(key=lambda c: (c.expected_seconds, c.latency_rank, c.score))
    elif profile.objective == "smartest":
        candidates.sort(key=lambda c: (c.intelligence_rank, c.score))
    else:  # cheapest_success
        candidates.sort(key=lambda c: c.score)

    return Selection(
        profile_name=profile_name,
        objective=profile.objective,
        quality_floor=floor,
        budget_usd=effective_budget,
        candidates=candidates,
        pool_considered=len(pool),
        catalog_as_of=catalog.as_of,
        catalog_source=catalog.source,
        support=support,
    )
