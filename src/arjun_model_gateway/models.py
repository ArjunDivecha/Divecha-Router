"""
=============================================================================
MODULE: models.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
Pydantic data models shared across the gateway: the OpenAI-compatible chat
request (plus gateway routing metadata), the route decision produced by the
selector, adapter responses, telemetry rows, outcome callbacks, and the
OpenRouter registry snapshot types.

INPUT FILES: none — this module performs no file I/O.
OUTPUT FILES: none — this module performs no file I/O.
=============================================================================
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class GatewayMetadata(BaseModel):
    """Routing intent attached by application callers. All fields are ENFORCED
    in v2 (budget, schema, context, tools, quality floor) — none are decorative."""

    repo: str | None = None
    script: str | None = None
    call_site_id: str | None = None
    task_label: str | None = None
    routing_profile: str = "auto"
    privacy_level: str = "normal"  # local_only/private_local/no_external force the local route
    latency_priority: str | None = None  # "high" switches the objective to fastest
    cost_priority: str | None = None
    quality_floor: str | None = None  # overrides the profile floor
    structured_output_schema: dict[str, Any] | None = None  # injected as response_format
    tool_requirements: list[str] = Field(default_factory=list)
    context_length_estimate: int | None = None  # raises the min-context filter
    fallback_permission: bool = True
    maximum_budget_usd: float | None = None  # hard pre-flight gate
    verification_mode: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    metadata: GatewayMetadata = Field(default_factory=GatewayMetadata)


class RouteDecision(BaseModel):
    request_id: str = Field(default_factory=lambda: f"amg_{uuid4().hex}")
    routing_profile: str
    route_type: str
    provider: str
    model: str
    objective: str | None = None
    quality_floor: str | None = None
    fallback: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    provider_sort: str | None = None  # e.g. "latency" -> OpenRouter provider routing
    timeout_seconds: float = 300.0  # upstream HTTP timeout; server-tool routes get longer
    use_responses_api: bool = False  # fusion is materially faster on /responses
    reasoning: str | None = None  # "off" | "low" | "medium" | "high" | None=provider default
    response_format: dict[str, Any] | None = None  # injected structured-output schema
    estimated_route_cost_usd: float | None = None
    budget_usd: float | None = None
    route_graph: list[dict[str, Any]] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    privacy_level: str = "normal"
    external_calls_allowed: bool = True
    require_verification: bool = False
    catalog_as_of: str | None = None
    catalog_source: str | None = None
    policy_version: int | str = 3


class AdapterResponse(BaseModel):
    provider: str
    model: str
    raw: dict[str, Any]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float | None = None
    output_text: str | None = None


class OutcomeReport(BaseModel):
    """POST /v1/outcomes payload — the router's learning signal."""

    request_id: str
    success: bool
    verifier: str = "external"  # schema | tool_call | tests | transport | human | external
    detail: str | None = None


class TelemetryRecord(BaseModel):
    request_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    repo: str | None = None
    script: str | None = None
    call_site_id: str | None = None
    routing_profile: str
    inferred_task_type: str | None = None
    chosen_route: str
    chosen_model: str  # catalog id used for routing/learning joins
    resolved_model: str | None = None  # provider-resolved slug (may be dated)
    chosen_provider: str
    fallback_models: list[dict[str, Any]] = Field(default_factory=list)
    prompt_token_count: int = 0
    completion_token_count: int = 0
    total_token_count: int = 0
    total_cost_usd: float | None = None
    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    cost_source: str | None = None
    cost_per_1k_tokens_usd: float | None = None
    output_input_token_ratio: float | None = None
    estimated_route_cost_usd: float | None = None
    latency_ms: float
    time_to_first_token_ms: float | None = None
    generation_time_ms: float | None = None
    throughput_tokens_per_second: float | None = None
    prompt_tokens_per_second: float | None = None
    completion_tokens_per_second: float | None = None
    schema_pass: bool | None = None
    tool_pass: bool | None = None
    tests_pass: bool | None = None
    retries: int = 0
    fallback_count: int = 0
    error_type: str | None = None
    verification_result: str | None = None
    output_hash: str | None = None
    prompt_hash: str | None = None
    missing_metadata: list[str] = Field(default_factory=list)


class ModelRegistryEntry(BaseModel):
    id: str
    name: str | None = None
    provider: str = "openrouter"
    context_length: int | None = None
    input_modalities: list[str] = Field(default_factory=list)
    output_modalities: list[str] = Field(default_factory=list)
    supported_parameters: list[str] | dict[str, Any] = Field(default_factory=list)
    pricing: dict[str, Any] = Field(default_factory=dict)
    top_provider: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class RegistrySnapshot(BaseModel):
    version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str
    models: list[ModelRegistryEntry]


RouteType = Literal["simple", "advisor", "fusion", "subagent", "local", "direct"]
