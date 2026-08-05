"""
=============================================================================
MODULE: policy.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
Turns an incoming chat request into a RouteDecision. Steps:

  1. Resolve the routing profile — explicit metadata/model name, or the
     keyword auto-classifier when the caller sends "auto".
  2. For marketplace routes, call selector.select() against the live catalog
     (no hardcoded models anywhere) and translate the winning candidate +
     dynamically resolved support models into the concrete request payload:
     server-tool blocks (fusion/advisor/subagent) with tool_choice="required"
     so the route name guarantees the route behavior, injected
     response_format for structured output, provider latency sort for fast
     routes, and ranked fallbacks.
  3. Enforce request metadata: quality floor override, context estimate,
     tool requirements, budget (selector drops unaffordable candidates and
     raises RoutingError if nothing fits), latency priority.

INPUT FILES: none directly (policy object and catalog are passed in /
constructed; see config.py and catalog.py for the files they read).
OUTPUT FILES: none — this module performs no file I/O.
=============================================================================
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .catalog import Catalog
from .config import CONFIG_DIR, Policy, PolicyProfile
from .models import ChatCompletionRequest, RouteDecision
from .selector import RoutingError, Selection, select
from .telemetry import latency_observations, success_counts

LOCAL_PRIVACY_LEVELS = {"local_only", "private_local", "no_external"}
MANUAL_SEEDS_PATH = CONFIG_DIR / "success_priors.yaml"

# The auto-classifier is intentionally deterministic and free at request time.
# These phrase families are broader than a handful of exact product terms, so
# semantically equivalent prompts do not jump between cheap, Advisor, and
# Fusion routes merely because the user paraphrased the task.
INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "coding": (
        r"\brepo(?:sitory)?\b",
        r"\bdiffs?\b",
        r"\btests?\b",
        r"\bpytest\b",
        r"\bdebug(?:ging)?\b",
        r"\brefactor(?:ing)?\b",
        r"\bbugs?\b",
        r"\bcode\b",
        r"\bsoftware\b",
        r"\bprogram(?:ming)?\b",
        r"\bimplement(?:ation|ing)?\b",
        r"\bpatch(?:es)?\b",
        r"\bstack\s+trace\b",
        r"\bexceptions?\b",
        r"\bcompile[sd]?\b",
        r"\bbuild\s+fail(?:s|ed|ure)?\b",
    ),
    "finance": (
        r"\bportfolios?\b",
        r"\bholdings?\b",
        r"\basset\s+allocation\b",
        r"\basset\s+positions?\b",
        r"\bsecurities\b",
        r"\binvest(?:ment|ing|or)s?\b",
        r"\bdrawdowns?\b",
        r"\brisk[- ]adjusted\b",
        r"\bmarket\s+risk\b",
        r"\bvaluation\b",
        r"\bequit(?:y|ies)\b",
        r"\bbonds?\b",
        r"\bquant(?:itative)?\b",
        r"\balpha\b",
        r"\bfactor\b",
        r"\bbloomberg\b",
        r"\bfinance\b",
    ),
    "high_stakes": (
        r"\blegal\b",
        r"\blaw\b",
        r"\bregulat(?:ion|ory|or)\b",
        r"\bmedical\b",
        r"\bclinical\b",
        r"\bdiagnos(?:is|e|tic)\b",
        r"\bhigh[ -]stakes\b",
    ),
    "research": (
        r"\bresearch\b",
        r"\bevidence\b",
        r"\bsources?\b",
        r"\bcitations?\b",
        r"\bliterature\b",
        r"\binvestigat(?:e|ion)\b",
        r"\bcompar(?:e|ison|ative)\b",
        r"\bdisagree(?:ment)?\b",
        r"\bsynthesi[sz](?:e|ed|ing|s)?\b",
        r"\blatest\b",
    ),
}


def _matches_intent(text: str, label: str) -> bool:
    return any(re.search(pattern, text) for pattern in INTENT_PATTERNS[label])


def manual_seeds(profile_name: str, path: Path = MANUAL_SEEDS_PATH) -> dict[str, float]:
    """Optional hand-set prior seeds (config/success_priors.yaml). Real outcomes
    always dominate these after a handful of observations."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return dict((data.get("profiles") or {}).get(profile_name) or {})


def classify_request(request: ChatCompletionRequest) -> list[str]:
    text = " ".join(str(m.content).lower() for m in request.messages)
    labels: list[str] = []
    if request.tools or re.search(r"\btools?\b", text):
        labels.append("tool_calling")
    if (
        request.response_format
        or request.metadata.structured_output_schema
        or re.search(r"\bjson\b", text)
    ):
        labels.append("structured_output")
    for label in ("coding", "finance", "high_stakes", "research"):
        if _matches_intent(text, label):
            labels.append(label)
    if len(text) < 600 and not any(
        label in labels for label in ("coding", "finance", "high_stakes", "research")
    ):
        labels.append("simple")
    if request.metadata.context_length_estimate and request.metadata.context_length_estimate > 100_000:
        labels.append("long_context")
    return labels or ["general"]


def resolve_profile_name(request: ChatCompletionRequest, policy: Policy) -> str:
    """Precedence: explicit metadata.routing_profile > the standard OpenAI
    `model` field > the policy default. A caller who names a profile in
    `model` (the normal OpenAI-client way) must get that profile — never the
    auto-classifier, which is only for `model="auto"`."""
    name = request.metadata.routing_profile or request.model or policy.default_profile
    if name in (None, ""):
        name = "auto"
    if request.metadata.privacy_level in LOCAL_PRIVACY_LEVELS and "private" in policy.profiles:
        return "private"
    if name == "auto":
        labels = classify_request(request)
        if "finance" in labels:
            name = "finance"
        elif "high_stakes" in labels or "research" in labels:
            name = "research"
        elif "coding" in labels:
            name = "code"
        elif "structured_output" in labels:
            name = "structured"
        elif "long_context" in labels:
            name = "long_context"
        elif "simple" in labels:
            name = "cheap"
        else:
            name = "fast"
    if name not in policy.profiles:
        raise RoutingError("unknown_profile", f"unknown routing profile: {name}")
    return name


def _server_tools(profile: PolicyProfile, selection: Selection) -> list[dict]:
    """Build OpenRouter server-tool blocks from dynamically resolved models."""
    support = selection.support
    if profile.route_type == "advisor" and "advisor" in support:
        return [
            {
                "type": "openrouter:advisor",
                "parameters": {
                    "model": support["advisor"]["id"],
                    "instructions": profile.advisor_instructions
                    or "You are a senior staff engineer. Focus on correctness, tests, and minimal diffs.",
                },
            }
        ]
    if profile.route_type == "fusion":
        return [
            {
                "type": "openrouter:fusion",
                "parameters": {
                    "analysis_models": [m["id"] for m in support.get("panel", [])],
                    "model": support["judge"]["id"],
                    "max_tool_calls": profile.fusion.max_tool_calls,
                },
            }
        ]
    if profile.route_type == "subagent" and "worker" in support:
        return [
            {
                "type": "openrouter:subagent",
                "parameters": {
                    "model": support["worker"]["id"],
                    "instructions": profile.subagent_instructions
                    or "You are a focused worker. Preserve facts, citations, and requested schema exactly.",
                },
            }
        ]
    return []


def _response_format(request: ChatCompletionRequest) -> dict | None:
    """Turn metadata.structured_output_schema into a real response_format."""
    if request.response_format:
        return request.response_format
    schema = request.metadata.structured_output_schema
    if not schema:
        return None
    return {
        "type": "json_schema",
        "json_schema": {"name": "gateway_schema", "strict": True, "schema": schema},
    }


# A subagent request escalates to the HEAVY worker tier when it looks
# demanding: the caller explicitly raised the quality floor, the input is
# large, or the call is agentic (brings its own tools / tool requirements).
# Everything else is handled by the cheaper LIGHT tier. Only consulted when a
# profile defines worker_tiers; ignored otherwise.
HEAVY_WORKER_CONTEXT_TOKENS = 48_000


def infer_worker_heavy(request: ChatCompletionRequest) -> bool:
    """Auto tier signal for two-tier subagent workers (light vs heavy)."""
    meta = request.metadata
    if (meta.quality_floor or "") in ("high", "frontier"):
        return True
    if (meta.context_length_estimate or 0) >= HEAVY_WORKER_CONTEXT_TOKENS:
        return True
    if meta.tool_requirements or request.tools:
        return True
    return False


def choose_route(
    request: ChatCompletionRequest,
    policy: Policy,
    catalog: Catalog | None = None,
) -> RouteDecision:
    profile_name = resolve_profile_name(request, policy)
    profile = policy.profiles[profile_name]
    meta = request.metadata
    privacy = meta.privacy_level or "normal"
    external_allowed = profile.external_calls_allowed and privacy not in LOCAL_PRIVACY_LEVELS

    reasons = [f"profile={profile_name}", f"route_type={profile.route_type}"]

    # local/direct routes skip catalog selection but still resolve their
    # model DYNAMICALLY from the provider's own live models API
    if profile.route_type in ("local", "direct"):
        from .provider_models import resolve_direct_model

        model = resolve_direct_model(profile.provider)
        reasons.append(f"provider_models_api={profile.provider}:{model}")
        return RouteDecision(
            routing_profile=profile_name,
            route_type=profile.route_type,
            provider=profile.provider,
            model=model,
            reasons=reasons,
            privacy_level=privacy,
            external_calls_allowed=external_allowed,
            policy_version=policy.version,
        )

    catalog = catalog or Catalog()
    response_format = _response_format(request)
    extra_params: list[str] = []
    if response_format:
        extra_params.append("structured_outputs")
    if meta.tool_requirements or request.tools:
        extra_params.append("tools")

    objective_override = profile.objective
    if meta.latency_priority == "high" and profile.objective != "fastest":
        profile = profile.model_copy(update={"objective": "fastest"})
        reasons.append("latency_priority=high -> objective=fastest")
        objective_override = "fastest"

    worker_heavy = infer_worker_heavy(request) if profile.route_type == "subagent" else False

    counts, global_counts = success_counts()
    selection = select(
        profile_name,
        profile,
        catalog,
        quality_floor=meta.quality_floor,
        min_context_override=meta.context_length_estimate or 0,
        extra_required_parameters=extra_params,
        budget_usd=meta.maximum_budget_usd,
        counts=counts,
        global_counts=global_counts,
        manual_seeds=manual_seeds(profile_name),
        latency_obs=latency_observations(),
        worker_heavy=worker_heavy,
    )
    if profile.route_type == "subagent" and profile.worker_tiers is not None:
        tier = selection.support.get("worker_tier", "light")
        reasons.append(f"worker_tier={tier}" + (":heavy_signal" if worker_heavy else ""))
        if selection.support.get("worker_tier_fallback"):
            reasons.append(f"worker_tier_fallback={selection.support['worker_tier_fallback']}")

    primary = selection.primary
    reasons.extend(
        [
            f"objective={objective_override}",
            f"quality_floor={selection.quality_floor}",
            f"catalog={selection.catalog_source}:as_of={selection.catalog_as_of}",
            f"pool={selection.pool_considered}",
            (
                f"chosen={primary.model_id}:intel_rank={primary.intelligence_rank}"
                f":est_route_cost=${primary.route_cost_usd:.6f}"
                f":p_success={primary.p_success:.3f}"
                f" (verified {primary.successes}s/{primary.failures}f)"
                f":score={primary.score:.6f}"
            ),
        ]
    )
    if selection.budget_usd is not None:
        reasons.append(f"budget=${selection.budget_usd:.4f}")
    for node in primary.route_graph:
        reasons.append(
            f"graph:{node.role}={node.model_id}"
            f":in={node.input_tokens}:out={node.output_tokens}:${node.cost_usd:.6f}"
        )

    tools = _server_tools(profile, selection)
    # tool_choice=required guarantees server-tool routes actually run the tool,
    # but never override a caller who brought their own tools/tool_choice.
    tool_choice = None
    if tools and not request.tools and request.tool_choice is None:
        tool_choice = "required"

    # fusion runs a panel + judge server-side (beta on chat completions and
    # explicitly slower there) — give server-tool routes a much longer leash
    timeout_seconds = 900.0 if profile.route_type == "fusion" else (
        600.0 if profile.route_type in ("advisor", "subagent") else 300.0
    )

    max_fallbacks = profile.select.max_fallbacks
    fallbacks = [
        {"provider": profile.provider, "model": c.model_id}
        for c in selection.candidates[1 : 1 + max_fallbacks]
    ]

    return RouteDecision(
        routing_profile=profile_name,
        route_type=profile.route_type,
        provider=profile.provider,
        model=primary.model_id,
        objective=objective_override,
        quality_floor=selection.quality_floor,
        fallback=fallbacks,
        tools=tools,
        tool_choice=request.tool_choice or tool_choice,
        provider_sort="latency" if objective_override == "fastest" else None,
        timeout_seconds=timeout_seconds,
        # OpenRouter: fusion on chat/completions is beta and slower; /responses
        # is the recommended path for it
        use_responses_api=profile.route_type == "fusion" and profile.provider == "openrouter",
        reasoning=None if profile.reasoning == "provider_default" else profile.reasoning,
        response_format=response_format,
        estimated_route_cost_usd=primary.route_cost_usd,
        budget_usd=selection.budget_usd,
        route_graph=[
            {
                "role": n.role,
                "model": n.model_id,
                "input_tokens": n.input_tokens,
                "output_tokens": n.output_tokens,
                "cost_usd": round(n.cost_usd, 8),
            }
            for n in primary.route_graph
        ],
        candidates=[
            {
                "model": c.model_id,
                "intelligence_rank": c.intelligence_rank,
                "latency_rank": c.latency_rank,
                "estimated_route_cost_usd": round(c.route_cost_usd, 8),
                "p_success": round(c.p_success, 4),
                "verified_successes": c.successes,
                "verified_failures": c.failures,
                "score": round(c.score, 8),
            }
            for c in selection.candidates
        ],
        reasons=reasons,
        privacy_level=privacy,
        external_calls_allowed=external_allowed,
        require_verification=profile.require_verification,
        catalog_as_of=selection.catalog_as_of,
        catalog_source=selection.catalog_source,
        policy_version=policy.version,
    )
