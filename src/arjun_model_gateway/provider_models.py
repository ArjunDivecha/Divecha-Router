"""
=============================================================================
MODULE: provider_models.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
Dynamic model resolution for DIRECT provider routes (OpenAI, Anthropic,
local OpenAI-compatible servers). Nothing in this repo names a model — the
direct routes ask each provider's own live /models API at decision time:

- anthropic:    GET https://api.anthropic.com/v1/models (newest-first per
                Anthropic docs); prefer the newest small ("haiku"-tier)
                model for cheap diagnostics, else the newest model.
- openai:       GET https://api.openai.com/v1/models; consider only text
                chat models (excludes audio/realtime/image/tts/embedding/
                moderation ids), prefer the newest "mini"-tier, else the
                newest chat model.
- local_openai: GET {LOCAL_OPENAI_BASE_URL}/models; first model served.

"mini"/"haiku" are TIER patterns, not version pins — when a provider ships
a newer generation, these resolve to it automatically. Results are cached
in-process for AMG_PROVIDER_MODELS_TTL_SECONDS (default 900s). If a
provider's models API is unreachable, RoutingError — FAIL IS FAIL, no
guessed fallback model.

INPUT FILES: none — network I/O only (api keys from environment variables
OPENAI_API_KEY / ANTHROPIC_API_KEY / LOCAL_OPENAI_API_KEY).
OUTPUT FILES: none — this module performs no file I/O.
=============================================================================
"""

from __future__ import annotations

import os
import time

import httpx

from .selector import RoutingError

_CACHE: dict[str, tuple[str, float]] = {}

# id fragments that mark non-chat OpenAI models
_OPENAI_EXCLUDE = (
    "audio",
    "realtime",
    "image",
    "tts",
    "transcribe",
    "whisper",
    "embed",
    "moderation",
    "dall-e",
    "sora",
    "computer-use",
)


def _ttl() -> int:
    return int(os.environ.get("AMG_PROVIDER_MODELS_TTL_SECONDS", "900"))


def resolve_direct_model(provider: str) -> str:
    cached = _CACHE.get(provider)
    now = time.monotonic()
    if cached and now - cached[1] < _ttl():
        return cached[0]
    if provider == "anthropic":
        model = _anthropic_newest()
    elif provider == "openai":
        model = _openai_newest_chat()
    elif provider == "local_openai":
        model = _local_first()
    else:
        raise RoutingError("unknown_direct_provider", f"no dynamic resolver for provider {provider!r}")
    _CACHE[provider] = (model, now)
    return model


def reset_cache() -> None:
    """Test hook."""
    _CACHE.clear()


def _anthropic_newest() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RoutingError("missing_api_key", "ANTHROPIC_API_KEY is not set")
    try:
        with httpx.Client(timeout=15) as client:
            response = client.get(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                params={"limit": 100},
            )
            response.raise_for_status()
            data = response.json().get("data", [])
    except httpx.HTTPError as exc:
        raise RoutingError("provider_models_unavailable", f"anthropic models API failed: {exc}")
    if not data:
        raise RoutingError("provider_models_unavailable", "anthropic models API returned no models")
    # newest-first; prefer the newest small tier for cheap diagnostics
    for item in data:
        if "haiku" in str(item.get("id", "")):
            return str(item["id"])
    return str(data[0]["id"])


def _openai_newest_chat() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RoutingError("missing_api_key", "OPENAI_API_KEY is not set")
    try:
        with httpx.Client(timeout=15) as client:
            response = client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json().get("data", [])
    except httpx.HTTPError as exc:
        raise RoutingError("provider_models_unavailable", f"openai models API failed: {exc}")
    chat = [
        m
        for m in data
        if str(m.get("id", "")).startswith("gpt-")
        and not any(frag in str(m.get("id", "")) for frag in _OPENAI_EXCLUDE)
    ]
    if not chat:
        raise RoutingError("provider_models_unavailable", "no chat models in openai models API")
    chat.sort(key=lambda m: int(m.get("created") or 0), reverse=True)
    minis = [m for m in chat if "mini" in str(m.get("id", ""))]
    return str((minis[0] if minis else chat[0])["id"])


def _local_first() -> str:
    base = os.environ.get("LOCAL_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{base}/models")
            response.raise_for_status()
            data = response.json().get("data", [])
    except httpx.HTTPError as exc:
        raise RoutingError(
            "provider_models_unavailable", f"local OpenAI-compatible server at {base} failed: {exc}"
        )
    if not data:
        raise RoutingError("provider_models_unavailable", f"no models served at {base}")
    return str(data[0]["id"])
