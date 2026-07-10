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
DEFAULT_REGISTRY_PATH = DATA_DIR / "registry.json"
DEFAULT_TELEMETRY_PATH = DATA_DIR / "telemetry.sqlite"


RouteType = Literal["simple", "advisor", "fusion", "subagent", "local", "direct"]
Objective = Literal["cheapest_success", "fastest", "smartest"]
QualityFloor = Literal["basic", "medium", "high", "frontier"]


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


class PolicyProfile(BaseModel):
    route_type: RouteType = "simple"
    provider: str = "openrouter"
    objective: Objective = "cheapest_success"
    quality_floor: QualityFloor = "basic"
    select: SelectSpec = Field(default_factory=SelectSpec)
    expected_tokens: ExpectedTokens = Field(default_factory=ExpectedTokens)
    default_budget_usd: float | None = None
    fusion: FusionSpec = Field(default_factory=FusionSpec)
    advisor_instructions: str | None = None
    subagent_instructions: str | None = None
    require_verification: bool = False
    # only meaningful for route_type local/direct (pinned diagnostics):
    model: str | None = None
    external_calls_allowed: bool = True


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


def validate_policy(path: Path = DEFAULT_POLICY_PATH) -> Policy:
    policy = load_policy(path)
    if policy.default_profile != "auto" and policy.default_profile not in policy.profiles:
        raise ValueError(f"default_profile {policy.default_profile!r} is not defined")
    for name, profile in policy.profiles.items():
        if profile.route_type in ("local", "direct") and not profile.model:
            raise ValueError(f"profile {name!r} is {profile.route_type} and must pin a model")
        if profile.route_type not in ("local", "direct") and profile.model:
            raise ValueError(
                f"profile {name!r} hardcodes model {profile.model!r}; "
                "marketplace profiles must select dynamically"
            )
    return policy
