"""
=============================================================================
MODULE: adapters.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
Provider adapters that execute a RouteDecision: OpenRouter (primary
marketplace runtime), direct OpenAI, direct Anthropic, any local
OpenAI-compatible server (e.g. Ollama), and a mock adapter for tests.

v2 changes:
- OpenRouter payloads always send `usage: {"include": true}` so every
  response carries exact cost and token accounting (no price-table guessing).
- The decision's dynamically-selected server tools, tool_choice,
  response_format (structured output), and provider sort preference
  (`provider: {"sort": "latency"}` for fast routes) are applied to the payload.
- `stream_chat()` provides REAL SSE passthrough from the upstream provider
  (v1 faked streaming by wrapping a completed response).

INPUT FILES: none — network I/O only (api keys come from environment
variables: OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY,
LOCAL_OPENAI_API_KEY / LOCAL_OPENAI_BASE_URL).
OUTPUT FILES: none — this module performs no file I/O.
=============================================================================
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import httpx

from .models import AdapterResponse, ChatCompletionRequest, RouteDecision


class AdapterError(RuntimeError):
    def __init__(self, provider: str, error_type: str, message: str):
        super().__init__(message)
        self.provider = provider
        self.error_type = error_type


class ProviderAdapter(ABC):
    provider: str
    supports_streaming: bool = False

    @abstractmethod
    def chat(self, request: ChatCompletionRequest, decision: RouteDecision) -> AdapterResponse:
        raise NotImplementedError

    def stream_chat(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> Iterator[str]:
        raise NotImplementedError


def build_payload(request: ChatCompletionRequest, decision: RouteDecision) -> dict[str, Any]:
    data = request.model_dump(exclude={"metadata"}, exclude_none=True)
    data["model"] = decision.model
    merged_tools = [*(request.tools or []), *decision.tools]
    if merged_tools:
        data["tools"] = merged_tools
    if decision.tool_choice is not None:
        data["tool_choice"] = decision.tool_choice
    if decision.response_format is not None:
        data["response_format"] = decision.response_format
    return data


def _usage(raw: dict[str, Any]) -> tuple[int, int]:
    usage = raw.get("usage") or {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def _text(raw: dict[str, Any]) -> str | None:
    choices = raw.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else None


def _resolved_model(raw: dict[str, Any], fallback: str) -> str:
    return str(raw.get("model") or raw.get("model_id") or fallback)


def _cost(raw: dict[str, Any]) -> float | None:
    usage = raw.get("usage") or {}
    for key in ("cost", "total_cost", "total_cost_usd"):
        value = usage.get(key) or raw.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


class OpenAICompatibleAdapter(ProviderAdapter):
    provider = "openai_compatible"
    supports_streaming = True

    def __init__(
        self,
        provider: str,
        base_url: str,
        api_key_env: str,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.default_headers = default_headers or {}

    def _headers(self) -> dict[str, str]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise AdapterError(self.provider, "missing_api_key", f"{self.api_key_env} is not set")
        return {"Authorization": f"Bearer {api_key}", **self.default_headers}

    def _payload(self, request: ChatCompletionRequest, decision: RouteDecision) -> dict[str, Any]:
        payload = build_payload(request, decision)
        if self.provider == "openrouter":
            payload["usage"] = {"include": True}  # exact cost accounting per response
            if decision.provider_sort:
                payload["provider"] = {"sort": decision.provider_sort}
        return payload

    def chat(self, request: ChatCompletionRequest, decision: RouteDecision) -> AdapterResponse:
        headers = self._headers()
        payload = self._payload(request, decision)
        payload["stream"] = False
        try:
            with httpx.Client(timeout=decision.timeout_seconds) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
                response.raise_for_status()
                raw = response.json()
        except httpx.HTTPStatusError as exc:
            raise AdapterError(
                self.provider, f"http_{exc.response.status_code}", exc.response.text
            ) from exc
        except httpx.HTTPError as exc:
            raise AdapterError(self.provider, "network_error", str(exc)) from exc
        prompt_tokens, completion_tokens = _usage(raw)
        return AdapterResponse(
            provider=self.provider,
            model=_resolved_model(raw, decision.model),
            raw=raw,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_cost_usd=_cost(raw),
            output_text=_text(raw),
        )

    def stream_chat(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> Iterator[str]:
        """Yield raw upstream SSE lines (real passthrough streaming)."""
        headers = self._headers()
        payload = self._payload(request, decision)
        payload["stream"] = True
        try:
            with httpx.Client(timeout=decision.timeout_seconds) as client:
                with client.stream(
                    "POST", f"{self.base_url}/chat/completions", json=payload, headers=headers
                ) as response:
                    if response.status_code >= 400:
                        body = response.read().decode(errors="replace")
                        raise AdapterError(self.provider, f"http_{response.status_code}", body)
                    for line in response.iter_lines():
                        if line:
                            yield line
        except httpx.HTTPError as exc:
            raise AdapterError(self.provider, "network_error", str(exc)) from exc


def parse_stream_stats(lines: list[str]) -> dict[str, Any]:
    """Extract id/model/usage/content from collected SSE lines after a stream ends."""
    stats: dict[str, Any] = {"id": None, "model": None, "usage": {}, "content": ""}
    for line in lines:
        if not line.startswith("data:"):
            continue
        body = line[len("data:") :].strip()
        if not body or body == "[DONE]":
            continue
        try:
            chunk = json.loads(body)
        except json.JSONDecodeError:
            continue
        stats["id"] = chunk.get("id") or stats["id"]
        stats["model"] = chunk.get("model") or stats["model"]
        if chunk.get("usage"):
            stats["usage"] = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                stats["content"] += content
    return stats


class AnthropicAdapter(ProviderAdapter):
    provider = "anthropic"

    def chat(self, request: ChatCompletionRequest, decision: RouteDecision) -> AdapterResponse:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise AdapterError(self.provider, "missing_api_key", "ANTHROPIC_API_KEY is not set")
        system = None
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "system":
                system = str(message.content)
            else:
                messages.append({"role": message.role, "content": str(message.content)})
        payload: dict[str, Any] = {
            "model": decision.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,
        }
        if system:
            payload["system"] = system
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            with httpx.Client(timeout=300) as client:
                response = client.post(
                    "https://api.anthropic.com/v1/messages", json=payload, headers=headers
                )
                response.raise_for_status()
                raw = response.json()
        except httpx.HTTPStatusError as exc:
            raise AdapterError(
                self.provider, f"http_{exc.response.status_code}", exc.response.text
            ) from exc
        except httpx.HTTPError as exc:
            raise AdapterError(self.provider, "network_error", str(exc)) from exc
        input_tokens = int((raw.get("usage") or {}).get("input_tokens") or 0)
        output_tokens = int((raw.get("usage") or {}).get("output_tokens") or 0)
        text = "".join(
            part.get("text", "") for part in raw.get("content", []) if part.get("type") == "text"
        )
        compatible = {
            "id": raw.get("id"),
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
            "provider_raw": raw,
        }
        return AdapterResponse(
            provider=self.provider,
            model=decision.model,
            raw=compatible,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            output_text=text,
        )


class MockAdapter(ProviderAdapter):
    provider = "mock"

    def chat(self, request: ChatCompletionRequest, decision: RouteDecision) -> AdapterResponse:
        time.sleep(0.001)
        content = f"mock response via {decision.routing_profile}:{decision.route_type}"
        raw = {
            "id": decision.request_id,
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {
                "prompt_tokens": len(str(request.messages)),
                "completion_tokens": len(content.split()),
            },
        }
        prompt_tokens, completion_tokens = _usage(raw)
        return AdapterResponse(
            provider="mock",
            model=decision.model,
            raw=raw,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            output_text=content,
        )


def adapter_for(provider: str) -> ProviderAdapter:
    if provider == "openrouter":
        return OpenAICompatibleAdapter(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            default_headers={
                "HTTP-Referer": "http://127.0.0.1:8765",
                "X-Title": "Arjun Model Gateway",
            },
        )
    if provider == "openai":
        return OpenAICompatibleAdapter("openai", "https://api.openai.com/v1", "OPENAI_API_KEY")
    if provider == "local_openai":
        base = os.environ.get("LOCAL_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
        return OpenAICompatibleAdapter("local_openai", base, "LOCAL_OPENAI_API_KEY")
    if provider == "anthropic":
        return AnthropicAdapter()
    if provider == "mock":
        return MockAdapter()
    raise AdapterError(provider, "unknown_provider", f"no adapter for {provider}")
