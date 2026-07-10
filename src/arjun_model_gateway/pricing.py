from __future__ import annotations

from pathlib import Path

import yaml

from .config import CONFIG_DIR, DEFAULT_REGISTRY_PATH
from .registry import load_registry


DEFAULT_PRICE_PATH = CONFIG_DIR / "model_prices.yaml"


def load_prices(path: Path = DEFAULT_PRICE_PATH) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def estimate_cost(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    path: Path = DEFAULT_PRICE_PATH,
) -> tuple[float | None, float | None, float | None, str | None]:
    # try the live OpenRouter registry first; for direct providers attempt the
    # "{provider}/{model}" marketplace id as a best-effort mapping
    for candidate in ([model] if provider == "openrouter" else [f"{provider}/{model}", model]):
        input_cost, output_cost, total_cost = estimate_openrouter_registry_cost(
            candidate, prompt_tokens, completion_tokens
        )
        if total_cost is not None:
            return input_cost, output_cost, total_cost, "openrouter_registry"
    catalog = load_prices(path).get("prices", {})
    price = catalog.get(f"{provider}:{model}")
    if not price:
        return None, None, None, None
    input_cost = prompt_tokens * float(price["input_per_million_usd"]) / 1_000_000
    output_cost = completion_tokens * float(price["output_per_million_usd"]) / 1_000_000
    return input_cost, output_cost, input_cost + output_cost, "manual_price_catalog"


def estimate_openrouter_registry_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> tuple[float | None, float | None, float | None]:
    if not registry_path.exists():
        return None, None, None
    try:
        registry = load_registry(registry_path)
    except Exception:
        return None, None, None
    entry = next((item for item in registry.models if item.id == model), None)
    if entry is None and model == "openrouter/auto":
        return None, None, None
    if entry is None:
        return None, None, None
    prompt_price = entry.pricing.get("prompt") or entry.pricing.get("input")
    completion_price = entry.pricing.get("completion") or entry.pricing.get("output")
    try:
        input_cost = prompt_tokens * float(prompt_price)
        output_cost = completion_tokens * float(completion_price)
    except (TypeError, ValueError):
        return None, None, None
    return input_cost, output_cost, input_cost + output_cost
