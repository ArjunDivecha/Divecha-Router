"""
=============================================================================
MODULE: config.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
Policy schema and loader for the gateway. In v2 (policy version 3) a routing
profile is a set of CONSTRAINTS and an OBJECTIVE — it never names a model.
The selector (selector.py) resolves the constraints against the live
OpenRouter catalog at request time, so newly released models are eligible
automatically. The only profiles allowed to pin a model id are `local` and
`direct` route types (diagnostics for the non-marketplace adapters).

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/config/policy.yaml
  (routing profiles; validated by validate_policy())

OUTPUT FILES: none.
=============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
DEFAULT_POLICY_PATH = CONFIG_DIR / "policy.yaml"
DEFAULT_SUPPLEMENTAL_PATH = CONFIG_DIR / "supplemental_catalog.yaml"
DEFAULT_REGISTRY_PATH = DATA_DIR / "registry.json"
DEFAULT_TELEMETRY_PATH = DATA_DIR / "telemetry.sqlite"


RouteType = Literal["simple", "advisor", "fusion", "subagent", "local", "direct"]
Objective = Literal["cheapest_success", "fastest", "smartest"]
QualityFloor = Literal["basic", "medium", "high", "frontier"]
ReasoningMode = Literal["provider_default", "off", "low", "medium", "high"]


class SelectSpec(BaseModel):
    """Hard constraints a model must satisfy to be a candidate."""

    require_parameters: list[str] = Field(default_factory=list)
    min_context: int = 0
    max_prompt_price_per_million: float | None = None
    output_modality: str = "text"
    include_free: bool = False
    pool_size: int = 8
    max_fallbacks: int = 2


class ExpectedTokens(BaseModel):
    """Per-call token estimates used for cost pre-flight (not a limit)."""

    input: int = 8_000
    output: int = 1_000


class FusionSpec(BaseModel):
    panel_size: int = 4
    max_tool_calls: int = 8


class WorkerTier(BaseModel):
    """One subagent-worker tier — still NO pinned model id. A tier narrows the
    live catalog by soft preference (author, optional model-id substring, and
    an optional price cap), then resolves a concrete worker at decision time by
    `pick`: "cheapest" (min route price) or "smartest" (best live intelligence
    rank). `smartest` is deliberately ranking-driven so the tier tracks the
    frontier — when a preferred author ships a stronger model it is selected
    automatically, no config change. `name_contains` is a substring match that
    tolerates version suffixes (deepseek-v4-flash-0731); the tier degrades
    gracefully to the author-only pool, then to the cheapest constraint-meeting
    worker, if nothing matches in the current catalog."""

    model_config = {"extra": "forbid"}

    prefer_authors: list[str] = Field(default_factory=list)
    name_contains: str | None = None
    max_prompt_price_per_million: float | None = None
    pick: Literal["cheapest", "smartest"] = "cheapest"


class WorkerTiers(BaseModel):
    """Two-tier subagent workers: `light` for routine subtasks, `heavy` for
    demanding ones. The router picks the tier per request (see
    policy.infer_worker_heavy). Either tier may be omitted; the present one is
    reused for both."""

    model_config = {"extra": "forbid"}

    light: WorkerTier | None = None
    heavy: WorkerTier | None = None


class PolicyProfile(BaseModel):
    """NO `model` FIELD, BY DESIGN. Nothing in policy may name a model —
    marketplace routes select from the live OpenRouter catalog and
    local/direct routes resolve from the provider's own live models API.
    extra="forbid" makes a pinned `model:` key a validation error."""

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    route_type: RouteType = "simple"
    provider: str = "openrouter"
    objective: Objective = "cheapest_success"
    quality_floor: QualityFloor = "basic"
    select: SelectSpec = Field(default_factory=SelectSpec)
    expected_tokens: ExpectedTokens = Field(default_factory=ExpectedTokens)
    default_budget_usd: float | None = None
    fusion: FusionSpec = Field(default_factory=FusionSpec)
    # subagent routes only: optional two-tier worker selection (light/heavy).
    # When unset the worker stays the single cheapest model in the floor pool.
    worker_tiers: WorkerTiers | None = None
    advisor_instructions: str | None = None
    subagent_instructions: str | None = None
    require_verification: bool = False
    external_calls_allowed: bool = True
    # bulk profiles set "off": hidden reasoning otherwise eats small max_tokens
    # budgets and returns empty content (observed live with deepseek-v4-flash)
    reasoning: ReasoningMode = "provider_default"


class Policy(BaseModel):
    version: int | str = 3
    default_profile: str = "auto"
    profiles: dict[str, PolicyProfile]


def ensure_dirs() -> None:
    for path in (CONFIG_DIR, DATA_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> Policy:
    data = yaml.safe_load(path.read_text()) or {}
    return Policy.model_validate(data)


def load_supplemental_models(path: Path = DEFAULT_SUPPLEMENTAL_PATH) -> list[dict]:
    """Load NON-OpenRouter models to inject into marketplace selection.

    Returns catalog-shaped dicts (per-TOKEN pricing, `supplemental=True`, a
    `provider` for adapter dispatch, and the two percentiles used to slot the
    model into the live intelligence/latency rankings). The `id` must be the
    provider's own model name (what gets sent as `model` to that provider).

    Missing file -> [] (OpenRouter-only, backwards compatible). A malformed
    file raises so the problem is loud rather than silently dropping a model.
    """
    if path is None or not Path(path).exists():
        return []
    raw = yaml.safe_load(Path(path).read_text()) or {}
    out: list[dict] = []
    for entry in raw.get("models", []) or []:
        out.append(
            {
                "id": str(entry["id"]),
                "provider": str(entry["provider"]),
                "supplemental": True,
                "pricing": {
                    "prompt": float(entry["prompt_per_million"]) / 1_000_000,
                    "completion": float(entry["completion_per_million"]) / 1_000_000,
                },
                "context_length": int(entry.get("context_length") or 0),
                "supported_parameters": list(entry.get("supported_parameters") or []),
                "architecture": {
                    "output_modalities": list(entry.get("output_modalities") or ["text"])
                },
                "_intelligence_percentile": float(entry.get("intelligence_percentile", 0.5)),
                "_latency_percentile": float(entry.get("latency_percentile", 0.5)),
            }
        )
    return out


def validate_policy(path: Path = DEFAULT_POLICY_PATH) -> Policy:
    policy = load_policy(path)
    if policy.default_profile != "auto" and policy.default_profile not in policy.profiles:
        raise ValueError(f"default_profile {policy.default_profile!r} is not defined")
    return policy
