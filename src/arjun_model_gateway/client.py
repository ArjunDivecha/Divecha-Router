"""
=============================================================================
MODULE: client.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
A thin, dependency-light client for wiring an external harness into the
Divecha Router. It wraps the gateway's OpenAI-compatible HTTP API so a caller
never rebuilds the boilerplate:

  - always names the routing profile in BOTH `model` and
    `metadata.routing_profile` (the client contract that prevents a prompt
    from being silently re-classified into an expensive route),
  - threads repo / script / call_site_id metadata for per-call-site learning,
  - captures the `amg_request_id` off every response so outcomes can be
    reported back, and
  - offers `report_outcome()` plus a `verified_call()` context manager that
    reports the outcome automatically when your harness knows whether the
    step actually succeeded (tests passed, tool ran, human approved).

It also exposes `explain()` (a no-cost dry run of the routing decision).

This talks to the gateway over HTTP only — it does NOT import the server. A
harness in any repo can `from arjun_model_gateway.client import DivechaClient`
(if this package is installed) or simply copy this file.

INPUT FILES: none — network I/O only (HTTP to the gateway base URL).
OUTPUT FILES: none — this module performs no file I/O.

USAGE:
    from arjun_model_gateway.client import DivechaClient

    amg = DivechaClient(repo="my-harness", script="planner.py")
    res = amg.chat("cheap", [{"role": "user", "content": "..."}],
                   call_site_id="planner.summarize")
    print(res.text)

    # close the learning loop once you've verified the step:
    with amg.verified_call("code", msgs, verifier="tests") as (res, outcome):
        outcome.success = run_tests(res.text)   # reported automatically on exit
=============================================================================
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8765/v1"


class DivechaError(RuntimeError):
    """A gateway error surfaced to the caller. `code` is the machine code the
    gateway returned (e.g. 'budget_exceeded', 'no_candidates', an adapter
    error_type, or 'http_error')."""

    def __init__(self, code: str, message: str, *, status: int | None = None):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass
class ChatResult:
    """The parsed result of a chat call."""

    text: str | None
    request_id: str | None  # amg_request_id — pass to report_outcome()
    model: str | None  # the concrete model the gateway actually used
    raw: dict[str, Any]


@dataclass
class Outcome:
    """Mutable outcome holder yielded by verified_call(); set `.success`."""

    success: bool | None = None
    detail: str | None = None


class DivechaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        repo: str | None = None,
        script: str | None = None,
        timeout: float = 600.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.repo = repo
        self.script = script
        # allow dependency injection (tests pass an httpx.MockTransport client)
        self._http = http_client or httpx.Client(timeout=timeout)
        self._owns_http = http_client is None

    # -- payload construction ------------------------------------------------

    def _metadata(
        self,
        profile: str,
        *,
        call_site_id: str | None,
        quality_floor: str | None,
        maximum_budget_usd: float | None,
        latency_priority: str | None,
        context_length_estimate: int | None,
        structured_output_schema: dict[str, Any] | None,
        tool_requirements: list[str] | None,
        privacy_level: str | None,
        extra_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        # routing_profile is ALWAYS set (the client contract); everything else
        # only when provided, so we never send noise.
        meta: dict[str, Any] = {"routing_profile": profile}
        optional = {
            "repo": self.repo,
            "script": self.script,
            "call_site_id": call_site_id,
            "quality_floor": quality_floor,
            "maximum_budget_usd": maximum_budget_usd,
            "latency_priority": latency_priority,
            "context_length_estimate": context_length_estimate,
            "structured_output_schema": structured_output_schema,
            "tool_requirements": tool_requirements,
            "privacy_level": privacy_level,
        }
        meta.update({k: v for k, v in optional.items() if v is not None})
        if extra_metadata:
            meta.update(extra_metadata)
        return meta

    def _payload(
        self,
        profile: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any | None,
        response_format: dict[str, Any] | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": profile,  # a PROFILE, not a model name
            "messages": messages,
            "metadata": metadata,
        }
        for key, value in (
            ("temperature", temperature),
            ("max_tokens", max_tokens),
            ("tools", tools),
            ("tool_choice", tool_choice),
            ("response_format", response_format),
        ):
            if value is not None:
                payload[key] = value
        return payload

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._http.post(f"{self.base_url}{path}", json=payload)
        except httpx.HTTPError as exc:
            raise DivechaError("network_error", str(exc)) from exc
        if response.status_code >= 400:
            detail: Any = None
            try:
                detail = response.json().get("detail")
            except Exception:  # noqa: BLE001 — non-JSON error body
                detail = response.text
            if isinstance(detail, dict):
                code = str(detail.get("code") or detail.get("error_type") or "http_error")
                message = str(detail.get("message") or detail)
            else:
                code, message = "http_error", str(detail)
            raise DivechaError(code, message, status=response.status_code)
        return response.json()

    # -- public API ----------------------------------------------------------

    def chat(
        self,
        profile: str,
        messages: list[dict[str, Any]],
        *,
        call_site_id: str | None = None,
        quality_floor: str | None = None,
        maximum_budget_usd: float | None = None,
        latency_priority: str | None = None,
        context_length_estimate: int | None = None,
        structured_output_schema: dict[str, Any] | None = None,
        tool_requirements: list[str] | None = None,
        privacy_level: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        response_format: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Route a chat completion through the gateway and return a ChatResult."""
        meta = self._metadata(
            profile,
            call_site_id=call_site_id,
            quality_floor=quality_floor,
            maximum_budget_usd=maximum_budget_usd,
            latency_priority=latency_priority,
            context_length_estimate=context_length_estimate,
            structured_output_schema=structured_output_schema,
            tool_requirements=tool_requirements,
            privacy_level=privacy_level,
            extra_metadata=extra_metadata,
        )
        payload = self._payload(
            profile,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            metadata=meta,
        )
        raw = self._post("/chat/completions", payload)
        choices = raw.get("choices") or []
        text = None
        if choices:
            text = (choices[0].get("message") or {}).get("content")
        return ChatResult(
            text=text,
            request_id=raw.get("amg_request_id"),
            model=raw.get("model"),
            raw=raw,
        )

    def explain(
        self,
        profile: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """No-cost dry run: the full route decision (chosen model, candidates,
        priced route graph, reasons) without sending the request upstream.
        Accepts the same keyword options as chat()."""
        meta = self._metadata(
            profile,
            call_site_id=kwargs.get("call_site_id"),
            quality_floor=kwargs.get("quality_floor"),
            maximum_budget_usd=kwargs.get("maximum_budget_usd"),
            latency_priority=kwargs.get("latency_priority"),
            context_length_estimate=kwargs.get("context_length_estimate"),
            structured_output_schema=kwargs.get("structured_output_schema"),
            tool_requirements=kwargs.get("tool_requirements"),
            privacy_level=kwargs.get("privacy_level"),
            extra_metadata=kwargs.get("extra_metadata"),
        )
        payload = self._payload(
            profile,
            messages,
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
            response_format=kwargs.get("response_format"),
            metadata=meta,
        )
        return self._post("/routes/explain", payload)

    def report_outcome(
        self,
        request_id: str,
        success: bool,
        *,
        verifier: str = "external",
        detail: str | None = None,
    ) -> dict[str, Any]:
        """Teach the router whether a prior call actually succeeded. `verifier`
        is one of schema | tool_call | tests | human | external."""
        return self._post(
            "/outcomes",
            {
                "request_id": request_id,
                "success": bool(success),
                "verifier": verifier,
                "detail": detail,
            },
        )

    @contextmanager
    def verified_call(
        self,
        profile: str,
        messages: list[dict[str, Any]],
        *,
        verifier: str = "external",
        **kwargs: Any,
    ) -> Iterator[tuple[ChatResult, Outcome]]:
        """Make a call, hand back (result, outcome); when the block exits, if
        you set outcome.success it is reported automatically. Left as None ->
        nothing reported (the auto-verifiers still run server-side)."""
        result = self.chat(profile, messages, **kwargs)
        outcome = Outcome()
        try:
            yield result, outcome
        finally:
            if outcome.success is not None and result.request_id:
                self.report_outcome(
                    result.request_id,
                    outcome.success,
                    verifier=verifier,
                    detail=outcome.detail,
                )

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "DivechaClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
