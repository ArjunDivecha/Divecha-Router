"""
=============================================================================
MODULE: gateway.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
The FastAPI service — an OpenAI-compatible endpoint that routes every request
through the dynamic selector and closes the learning loop:

- POST /v1/chat/completions  OpenAI-compatible chat. Routing constraints are
  enforced (budget -> HTTP 400 when nothing affordable qualifies; structured
  output schemas are injected AND validated; context/capability filters).
  `stream: true` is real upstream SSE passthrough.
- POST /v1/outcomes          Verification callback (tests passed, human
  verdict, ...). This is the router's learning signal: outcomes update the
  Beta posteriors that drive model selection.
- POST /v1/routes/explain    Full route decision with candidate scores and
  the route-graph cost breakdown, without executing anything.
- GET  /v1/models            Routing profiles as an OpenAI-compatible model
  list (so SDK model pickers work against the gateway).
- GET  /v1/telemetry/summary Totals, per-model, per-profile, verified outcomes.
- GET  /health               Includes live catalog age/source/size.

Automatic outcomes recorded without any caller action: JSON-schema
validation of structured responses, tool-call presence when the caller
required tools, and transport failures. After each OpenRouter call a
background thread queries /api/v1/generation for true time-to-first-token
and generation time and back-fills telemetry (disable: AMG_GENERATION_STATS=0).

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/config/policy.yaml
  (routing profiles, via config.load_policy)
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/data/catalog_snapshot.json
  (offline catalog fallback, via catalog.Catalog)

OUTPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/data/telemetry.sqlite
  (attempt + outcome ledgers)
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/data/catalog_snapshot.json
  (refreshed on live catalog fetches)
=============================================================================
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .adapters import AdapterError, adapter_for, parse_stream_stats
from .catalog import Catalog
from .config import load_policy
from .models import (
    ChatCompletionRequest,
    OutcomeReport,
    RouteDecision,
    TelemetryRecord,
)
from .policy import choose_route, classify_request
from .pricing import estimate_cost
from .selector import RoutingError
from .telemetry import (
    missing_metadata,
    record_outcome,
    stable_hash,
    summarize,
    update_generation_stats,
    write_record,
)

app = FastAPI(title="Arjun Model Gateway", version="2.0.0")

EXTERNAL_PROVIDERS = {"openrouter", "openai", "anthropic", "celeris"}
GENERATION_URL = "https://openrouter.ai/api/v1/generation"


def candidate_allowed(decision: RouteDecision) -> bool:
    if decision.external_calls_allowed:
        return True
    return decision.provider not in EXTERNAL_PROVIDERS


# -- automatic verification ----------------------------------------------------


def validate_schema_response(text: str | None, response_format: dict[str, Any]) -> tuple[bool, str]:
    """Validate a structured response against the requested JSON schema."""
    if not text:
        return False, "empty response content"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, f"response is not valid JSON: {exc}"
    schema = (response_format.get("json_schema") or {}).get("schema")
    if not schema:
        return True, "valid JSON (no schema body to validate against)"
    try:
        import jsonschema

        jsonschema.validate(parsed, schema)
        return True, "JSON validates against the requested schema"
    except jsonschema.ValidationError as exc:
        return False, f"schema violation: {exc.message}"


def auto_outcomes(
    request: ChatCompletionRequest,
    decision: RouteDecision,
    output_text: str | None,
    raw: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Outcomes the gateway can certify on its own, synchronously."""
    results: list[dict[str, Any]] = []
    if decision.response_format and decision.response_format.get("type") == "json_schema":
        ok, detail = validate_schema_response(output_text, decision.response_format)
        results.append({"verifier": "schema", "success": ok, "detail": detail})
    if request.tools and request.tool_choice == "required" and raw is not None:
        choices = raw.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        has_tool_call = bool(message.get("tool_calls"))
        results.append(
            {
                "verifier": "tool_call",
                "success": has_tool_call,
                "detail": "tool call present" if has_tool_call else "tool call missing",
            }
        )
    return results


# -- generation-stats back-fill -------------------------------------------------


def fetch_generation_stats(request_id: str, generation_id: str) -> None:
    """Query OpenRouter for true TTFT/generation time; best-effort, background."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return
    headers = {"Authorization": f"Bearer {api_key}"}
    # stats typically materialize ~5-10s after the generation completes
    for delay in (2.0, 3.0, 5.0, 8.0, 8.0):
        time.sleep(delay)
        try:
            with httpx.Client(timeout=15) as client:
                response = client.get(GENERATION_URL, params={"id": generation_id}, headers=headers)
                if response.status_code == 404:
                    continue  # stats not materialized yet
                response.raise_for_status()
                data = response.json().get("data") or {}
        except httpx.HTTPError:
            continue
        latency_ms = data.get("latency")  # time to first token
        generation_time_ms = data.get("generation_time")
        if latency_ms is None and generation_time_ms is None:
            continue
        update_generation_stats(
            request_id,
            time_to_first_token_ms=float(latency_ms) if latency_ms is not None else None,
            generation_time_ms=float(generation_time_ms) if generation_time_ms is not None else None,
        )
        return


def schedule_generation_stats(decision: RouteDecision, provider: str, raw: dict[str, Any] | None) -> None:
    if provider != "openrouter" or not raw:
        return
    if os.environ.get("AMG_GENERATION_STATS", "1") == "0":
        return
    generation_id = raw.get("id")
    if not generation_id:
        return
    threading.Thread(
        target=fetch_generation_stats, args=(decision.request_id, generation_id), daemon=True
    ).start()


# -- core request handling -------------------------------------------------------


def handle_chat(request: ChatCompletionRequest, *, force_mock: bool = False) -> dict:
    policy = load_policy()
    decision = choose_route(request, policy)
    candidates = [{"provider": decision.provider, "model": decision.model}]
    if request.metadata.fallback_permission:
        candidates.extend(decision.fallback)
    if force_mock:
        candidates = [{"provider": "mock", "model": decision.model}]

    started = time.perf_counter()
    response = None
    error_type = None
    provider_used = str(candidates[0]["provider"])
    model_used = str(candidates[0]["model"])
    fallback_count = 0
    last_error: AdapterError | None = None
    outcome_records: list[dict[str, Any]] = []
    # budget is CUMULATIVE across fallback attempts: a failed attempt may still
    # have billed upstream (e.g. a timeout after generation started), so each
    # attempt's estimate counts against the budget before the next one runs
    estimate_by_model = {
        str(c.get("model")): float(c.get("estimated_route_cost_usd") or 0)
        for c in decision.candidates
    }
    spent_estimate = 0.0
    try:
        for fallback_count, candidate in enumerate(candidates):
            provider_used = str(candidate["provider"])
            model_used = str(candidate["model"])
            attempt_estimate = estimate_by_model.get(
                model_used, decision.estimated_route_cost_usd or 0.0
            )
            if (
                decision.budget_usd is not None
                and fallback_count > 0
                and spent_estimate + attempt_estimate > decision.budget_usd
            ):
                last_error = AdapterError(
                    provider_used,
                    "cumulative_budget_exceeded",
                    f"fallback {model_used} (~${attempt_estimate:.4f}) would take the "
                    f"attempt total past the ${decision.budget_usd:.4f} budget "
                    f"(~${spent_estimate:.4f} already attempted)",
                )
                error_type = last_error.error_type
                break
            candidate_decision = decision.model_copy(
                update={"provider": provider_used, "model": model_used}
            )
            if not candidate_allowed(candidate_decision):
                last_error = AdapterError(
                    provider_used,
                    "privacy_external_call_blocked",
                    f"privacy_level={decision.privacy_level} disallows external provider {provider_used}",
                )
                error_type = last_error.error_type
                continue
            try:
                response = adapter_for(provider_used).chat(request, candidate_decision)
                provider_used = response.provider
                # keep model_used = the catalog id we requested, NOT the
                # provider-resolved dated slug (e.g. "-20260423"), so verified
                # outcomes join back to catalog candidates for learning
                error_type = None
                break
            except AdapterError as exc:
                spent_estimate += attempt_estimate
                last_error = exc
                error_type = exc.error_type
                if not request.metadata.fallback_permission:
                    break
        if response is None:
            raise last_error or AdapterError(provider_used, "no_response", "no route produced a response")
        outcome_records = auto_outcomes(request, decision, response.output_text, response.raw)
        schedule_generation_stats(decision, provider_used, response.raw)
        if isinstance(response.raw, dict):
            # callers post verification to /v1/outcomes keyed by this id
            response.raw["amg_request_id"] = decision.request_id
        return response.raw
    except AdapterError as exc:
        error_type = exc.error_type
        outcome_records = [
            {"verifier": "transport", "success": False, "detail": f"{exc.provider}:{exc.error_type}"}
        ]
        raise
    finally:
        _write_attempt(
            request,
            decision,
            response,
            provider_used,
            model_used,
            error_type,
            fallback_count,
            started,
            outcome_records,
        )


def _write_attempt(
    request: ChatCompletionRequest,
    decision: RouteDecision,
    response,
    provider_used: str,
    model_used: str,
    error_type: str | None,
    fallback_count: int,
    started: float,
    outcome_records: list[dict[str, Any]],
) -> None:
    latency_ms = (time.perf_counter() - started) * 1000
    prompt_tokens = response.prompt_tokens if response else 0
    completion_tokens = response.completion_tokens if response else 0
    total_tokens = prompt_tokens + completion_tokens
    total_cost = response.total_cost_usd if response else None
    input_cost = None
    output_cost = None
    cost_source = "provider_usage" if total_cost is not None else None
    if provider_used == "mock":
        cost_source = "mock"
    elif response and total_cost is None:
        input_cost, output_cost, total_cost, cost_source = estimate_cost(
            provider_used, response.model, prompt_tokens, completion_tokens
        )
    cost_per_1k = None
    token_ratio = None
    if total_cost is not None and total_tokens:
        cost_per_1k = total_cost / (total_tokens / 1000)
    if prompt_tokens:
        token_ratio = completion_tokens / prompt_tokens

    schema_pass = None
    tool_pass = None
    for record in outcome_records:
        if record["verifier"] == "schema":
            schema_pass = record["success"]
        if record["verifier"] == "tool_call":
            tool_pass = record["success"]
    if outcome_records:
        worst = all(r["success"] for r in outcome_records)
        verification = f"auto:{'pass' if worst else 'fail'}"
    elif decision.require_verification:
        verification = "pending_external"
    else:
        verification = "not_requested"

    write_record(
        TelemetryRecord(
            request_id=decision.request_id,
            repo=request.metadata.repo,
            script=request.metadata.script,
            call_site_id=request.metadata.call_site_id,
            routing_profile=decision.routing_profile,
            inferred_task_type=",".join(classify_request(request)),
            chosen_route=decision.route_type,
            chosen_model=model_used,
            resolved_model=response.model if response else None,
            chosen_provider=provider_used,
            fallback_models=decision.fallback,
            prompt_token_count=prompt_tokens,
            completion_token_count=completion_tokens,
            total_token_count=total_tokens,
            total_cost_usd=total_cost,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            cost_source=cost_source,
            cost_per_1k_tokens_usd=cost_per_1k,
            output_input_token_ratio=token_ratio,
            estimated_route_cost_usd=decision.estimated_route_cost_usd,
            latency_ms=latency_ms,
            schema_pass=schema_pass,
            tool_pass=tool_pass,
            retries=fallback_count,
            fallback_count=fallback_count,
            error_type=error_type,
            verification_result=verification,
            output_hash=stable_hash(response.output_text) if response else None,
            prompt_hash=stable_hash([m.model_dump() for m in request.messages]),
            missing_metadata=missing_metadata(request),
        )
    )
    if provider_used != "mock":
        for record in outcome_records:
            record_outcome(
                decision.request_id,
                success=record["success"],
                verifier=record["verifier"],
                detail=record.get("detail"),
            )


# -- streaming -------------------------------------------------------------------


def stream_chat_response(request: ChatCompletionRequest) -> Iterator[str]:
    """Real upstream SSE passthrough, with telemetry after the stream ends."""
    policy = load_policy()
    decision = choose_route(request, policy)
    adapter = adapter_for(decision.provider)
    if not adapter.supports_streaming or not candidate_allowed(decision) or decision.use_responses_api:
        # providers without streaming (anthropic direct, mock) and Responses-API
        # routes (fusion) get a synthetic stream over the completed response
        yield from synthetic_stream(request)
        return
    started = time.perf_counter()
    collected: list[str] = []
    error_type = None

    def stream_row(verification: str, latency_ms: float) -> TelemetryRecord:
        stats = parse_stream_stats(collected)
        usage = stats["usage"] or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        cost = usage.get("cost")
        return TelemetryRecord(
            request_id=decision.request_id,
            repo=request.metadata.repo,
            script=request.metadata.script,
            call_site_id=request.metadata.call_site_id,
            routing_profile=decision.routing_profile,
            inferred_task_type=",".join(classify_request(request)),
            chosen_route=decision.route_type,
            chosen_model=decision.model,
            resolved_model=stats["model"],
            chosen_provider=decision.provider,
            fallback_models=decision.fallback,
            prompt_token_count=prompt_tokens,
            completion_token_count=completion_tokens,
            total_token_count=prompt_tokens + completion_tokens,
            total_cost_usd=float(cost) if cost is not None else None,
            cost_source="provider_usage" if cost is not None else None,
            estimated_route_cost_usd=decision.estimated_route_cost_usd,
            latency_ms=latency_ms,
            error_type=error_type,
            verification_result=verification,
            output_hash=stable_hash(stats["content"]) if stats["content"] else None,
            prompt_hash=stable_hash([m.model_dump() for m in request.messages]),
            missing_metadata=missing_metadata(request),
        )

    # durable attempt row BEFORE streaming; replaced with final numbers after —
    # a client disconnect mid-stream can never lose the attempt entirely
    write_record(stream_row("streaming_in_progress", 0.0))
    try:
        for line in adapter.stream_chat(request, decision):
            collected.append(line)
            yield f"{line}\n\n"
    except AdapterError as exc:
        error_type = exc.error_type
        yield f'data: {{"error": {{"type": "{exc.error_type}", "provider": "{exc.provider}"}}}}\n\n'
    finally:
        stats = parse_stream_stats(collected)
        outcome_records = (
            auto_outcomes(request, decision, stats["content"], None) if not error_type else []
        )
        write_record(stream_row("streamed", (time.perf_counter() - started) * 1000))
        for record in outcome_records:
            record_outcome(
                decision.request_id,
                success=record["success"],
                verifier=record["verifier"],
                detail=record.get("detail"),
            )
        if stats["id"]:
            schedule_generation_stats(decision, decision.provider, {"id": stats["id"]})


def synthetic_stream(request: ChatCompletionRequest) -> Iterator[str]:
    non_streaming_request = request.model_copy(update={"stream": False})
    raw = handle_chat(non_streaming_request)
    content = ""
    choices = raw.get("choices") or []
    if choices:
        content = ((choices[0].get("message") or {}).get("content")) or ""
    chunk = {
        "id": raw.get("id"),
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}
        ],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    done = {
        "id": raw.get("id"),
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": raw.get("usage"),
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


# -- HTTP endpoints ---------------------------------------------------------------


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    try:
        if request.stream:
            return StreamingResponse(stream_chat_response(request), media_type="text/event-stream")
        return handle_chat(request)
    except RoutingError as exc:
        raise HTTPException(status_code=400, detail={"code": exc.code, "message": str(exc)})
    except AdapterError as exc:
        raise HTTPException(
            status_code=502,
            detail={"provider": exc.provider, "error_type": exc.error_type},
        )


@app.post("/v1/routes/explain")
def explain_route(request: ChatCompletionRequest) -> RouteDecision:
    try:
        return choose_route(request, load_policy())
    except RoutingError as exc:
        raise HTTPException(status_code=400, detail={"code": exc.code, "message": str(exc)})


@app.post("/v1/outcomes")
def report_outcome(report: OutcomeReport) -> dict:
    record_outcome(
        report.request_id,
        success=report.success,
        verifier=report.verifier,
        detail=report.detail,
    )
    return {"ok": True, "request_id": report.request_id}


@app.get("/v1/models")
def list_models() -> dict:
    """Routing profiles exposed as an OpenAI-compatible model list."""
    policy = load_policy()
    profiles = [
        {"id": name, "object": "model", "owned_by": "arjun-model-gateway"}
        for name in ["auto", *policy.profiles.keys()]
    ]
    return {"object": "list", "data": profiles}


@app.get("/v1/telemetry/summary")
def telemetry_summary() -> dict:
    return summarize()


@app.get("/health")
def health() -> dict:
    catalog = Catalog()
    try:
        size = catalog.size()
        return {
            "ok": True,
            "catalog_models": size,
            "catalog_source": catalog.source,
            "catalog_as_of": catalog.as_of,
        }
    except Exception as exc:  # catalog unavailable: report, don't lie
        return {"ok": False, "catalog_error": str(exc)}
