from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import httpx
import yaml

from .config import CONFIG_DIR, DATA_DIR, ensure_dirs
from .registry import OPENROUTER_MODELS_URL


DEFAULT_MODEL_INTEL_SNAPSHOT = DATA_DIR / "model_intel_snapshot.json"
DEFAULT_SUCCESS_PRIORS_PATH = CONFIG_DIR / "success_priors.yaml"


@dataclass(frozen=True)
class ModelQuery:
    category: str | None = None
    supported_parameters: str | None = None
    output_modalities: str | None = None
    input_modalities: str | None = None
    sort: str | None = None
    q: str | None = None
    context: int | None = None
    max_price: float | None = None
    min_price: float | None = None
    model_authors: str | None = None
    providers: str | None = None
    zdr: bool = False

    def params(self) -> dict[str, Any]:
        params = {
            "category": self.category,
            "supported_parameters": self.supported_parameters,
            "output_modalities": self.output_modalities,
            "input_modalities": self.input_modalities,
            "sort": self.sort,
            "q": self.q,
            "context": self.context,
            "max_price": self.max_price,
            "min_price": self.min_price,
            "model_authors": self.model_authors,
            "providers": self.providers,
        }
        if self.zdr:
            params["zdr"] = "true"
        return {key: value for key, value in params.items() if value is not None}


class ModelIntelProvider(Protocol):
    source: str
    as_of: str | None

    def get_model(self, model_id: str) -> dict[str, Any] | None: ...

    def list_models(self, query: ModelQuery, limit: int = 10) -> list[dict[str, Any]]: ...

    def rank_candidates(
        self,
        selection: dict[str, Any],
        task_label: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]: ...


class ModelIntel:
    source = "openrouter_models_api_cache"
    _catalog: list[dict[str, Any]] | None = None
    _loaded_at_monotonic: float | None = None
    _as_of: str | None = None

    def __init__(
        self,
        ttl_seconds: int | None = None,
        snapshot_path: Path = DEFAULT_MODEL_INTEL_SNAPSHOT,
        success_priors_path: Path = DEFAULT_SUCCESS_PRIORS_PATH,
    ) -> None:
        self.ttl_seconds = ttl_seconds or int(os.environ.get("AMG_MODEL_INTEL_TTL_SECONDS", "900"))
        self.snapshot_path = snapshot_path
        self.success_priors_path = success_priors_path

    @property
    def as_of(self) -> str | None:
        self.catalog()
        return self._as_of

    def catalog(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        now = time.monotonic()
        if (
            not refresh
            and self.__class__._catalog is not None
            and self.__class__._loaded_at_monotonic is not None
            and now - self.__class__._loaded_at_monotonic < self.ttl_seconds
        ):
            return self.__class__._catalog

        try:
            catalog = self._fetch_catalog()
            self._write_snapshot(catalog)
            self.__class__._catalog = catalog
            self.__class__._loaded_at_monotonic = now
            self.__class__._as_of = datetime.now(timezone.utc).isoformat()
            return catalog
        except httpx.HTTPError:
            catalog, as_of = self._read_snapshot()
            self.__class__._catalog = catalog
            self.__class__._loaded_at_monotonic = now
            self.__class__._as_of = as_of
            return catalog

    def _fetch_catalog(self) -> list[dict[str, Any]]:
        with httpx.Client(timeout=30) as client:
            response = client.get(OPENROUTER_MODELS_URL)
            response.raise_for_status()
            return response.json().get("data", [])

    def _write_snapshot(self, catalog: list[dict[str, Any]]) -> None:
        ensure_dirs()
        self.snapshot_path.write_text(
            json.dumps(
                {
                    "source": OPENROUTER_MODELS_URL,
                    "as_of": datetime.now(timezone.utc).isoformat(),
                    "models": catalog,
                },
                indent=2,
            )
        )

    def _read_snapshot(self) -> tuple[list[dict[str, Any]], str | None]:
        if not self.snapshot_path.exists():
            raise RuntimeError("model intel fetch failed and no snapshot exists")
        payload = json.loads(self.snapshot_path.read_text())
        return payload.get("models", []), payload.get("as_of")

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        for model in self.catalog():
            if model.get("id") == model_id:
                return model
        return None

    def list_models(self, query: ModelQuery, limit: int = 10) -> list[dict[str, Any]]:
        models = [model for model in self.catalog() if model_matches_query(model, query)]
        models.sort(key=lambda model: sort_key(model, query.sort))
        return models[:limit]

    def rank_candidates(
        self,
        selection: dict[str, Any],
        task_label: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        candidate_ids = selection.get("candidate_models") or selection.get("preferred_models") or []
        candidates = [model for model_id in candidate_ids if (model := self.get_model(model_id))]
        if not candidates:
            candidates = self.list_models(query_from_policy(selection), limit=limit)

        expected_input = int(selection.get("expected_input_tokens") or 8_000)
        expected_output = int(selection.get("expected_output_tokens") or 1_000)
        min_context = int(selection.get("context") or 0)
        required_parameters = {
            item.strip()
            for item in str(selection.get("supported_parameters") or "").split(",")
            if item.strip()
        }
        priority = {model_id: idx for idx, model_id in enumerate(candidate_ids)}
        priors = load_success_priors(self.success_priors_path)
        default_prior = float(priors.get("default_success_prior", 0.60))
        profile_priors = (priors.get("profiles") or {}).get(task_label or "", {})

        ranked = []
        for model in candidates:
            model_id = model.get("id")
            if min_context and int(model.get("context_length") or 0) < min_context:
                continue
            supported = set(model.get("supported_parameters") or [])
            if required_parameters and not required_parameters.issubset(supported):
                continue
            estimated_cost = estimate_model_cost(model, expected_input, expected_output)
            success_prior = float(profile_priors.get(model_id, default_prior))
            expected_cost_per_success = estimated_cost / max(success_prior, 0.01)
            prior_rank = priority.get(model_id, len(priority))
            score = expected_cost_per_success + prior_rank * float(
                selection.get("tie_break_weight", 0.000001)
            )
            ranked.append(
                {
                    "model": model,
                    "score": score,
                    "estimated_cost_usd": estimated_cost,
                    "success_prior": success_prior,
                    "expected_cost_per_success_usd": expected_cost_per_success,
                    "prior_rank": prior_rank,
                    "context_length": model.get("context_length"),
                    "pricing": model.get("pricing") or {},
                    "intel_source": self.source,
                    "intel_as_of": self.as_of,
                }
            )
        ranked.sort(key=lambda item: item["score"])
        return ranked

    def choose_model(self, query: ModelQuery) -> dict[str, Any] | None:
        models = self.list_models(query, limit=1)
        return models[0] if models else None


def query_from_policy(selection: dict[str, Any]) -> ModelQuery:
    return ModelQuery(
        category=selection.get("category"),
        supported_parameters=selection.get("supported_parameters"),
        output_modalities=selection.get("output_modalities"),
        input_modalities=selection.get("input_modalities"),
        sort=selection.get("sort"),
        q=selection.get("q"),
        context=selection.get("context"),
        max_price=selection.get("max_price"),
        min_price=selection.get("min_price"),
        model_authors=selection.get("model_authors"),
        providers=selection.get("providers"),
        zdr=bool(selection.get("zdr", False)),
    )


def load_success_priors(path: Path = DEFAULT_SUCCESS_PRIORS_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"default_success_prior": 0.60, "profiles": {}}
    return yaml.safe_load(path.read_text()) or {"default_success_prior": 0.60, "profiles": {}}


def estimate_model_cost(model: dict[str, Any], input_tokens: int, output_tokens: int) -> float:
    pricing = model.get("pricing") or {}
    try:
        prompt = float(pricing.get("prompt") or pricing.get("input") or 0)
        completion = float(pricing.get("completion") or pricing.get("output") or 0)
    except (TypeError, ValueError):
        return 999_999.0
    return input_tokens * prompt + output_tokens * completion


def model_matches_query(model: dict[str, Any], query: ModelQuery) -> bool:
    model_id = str(model.get("id") or "")
    name = str(model.get("name") or "")
    if query.q and query.q.lower() not in f"{model_id} {name}".lower():
        return False
    if query.context and int(model.get("context_length") or 0) < query.context:
        return False
    supported = set(model.get("supported_parameters") or [])
    if query.supported_parameters:
        required = {item.strip() for item in query.supported_parameters.split(",") if item.strip()}
        if not required.issubset(supported):
            return False
    architecture = model.get("architecture") or {}
    input_modalities = set(architecture.get("input_modalities") or [])
    output_modalities = set(architecture.get("output_modalities") or [])
    if query.input_modalities and query.input_modalities not in input_modalities:
        return False
    if query.output_modalities and query.output_modalities not in output_modalities:
        return False
    prompt_per_million = price_per_million(model, "prompt")
    if query.max_price is not None and prompt_per_million > query.max_price:
        return False
    if query.min_price is not None and prompt_per_million < query.min_price:
        return False
    return True


def sort_key(model: dict[str, Any], sort: str | None) -> tuple[float, str]:
    model_id = str(model.get("id") or "")
    if sort in {"pricing-low-to-high", "latency-low-to-high"}:
        return (price_per_million(model, "prompt") + price_per_million(model, "completion"), model_id)
    if sort == "context-high-to-low":
        return (-float(model.get("context_length") or 0), model_id)
    return (0.0, model_id)


def price_per_million(model: dict[str, Any], key: str) -> float:
    pricing = model.get("pricing") or {}
    value = pricing.get(key) or pricing.get("input" if key == "prompt" else "output") or 0
    try:
        return float(value) * 1_000_000
    except (TypeError, ValueError):
        return 999_999.0
