"""
=============================================================================
MODULE: tests/conftest.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
Shared pytest fixtures: a FakeCatalog (deterministic stand-in for the live
OpenRouter catalog so tests never hit the network) and an autouse fixture
that clears the real Catalog's class-level cache between tests.

INPUT FILES: none — this module performs no file I/O.
OUTPUT FILES: none — this module performs no file I/O.
=============================================================================
"""

from __future__ import annotations

from typing import Any

import pytest

from arjun_model_gateway.catalog import Catalog


def make_model(
    model_id: str,
    prompt_per_million: float,
    completion_per_million: float,
    *,
    context: int = 200_000,
    params: tuple[str, ...] = ("tools", "structured_outputs", "response_format"),
    output_modalities: tuple[str, ...] = ("text",),
) -> dict[str, Any]:
    return {
        "id": model_id,
        "name": model_id,
        "context_length": context,
        "supported_parameters": list(params),
        "architecture": {"output_modalities": list(output_modalities)},
        "pricing": {
            "prompt": str(prompt_per_million / 1_000_000),
            "completion": str(completion_per_million / 1_000_000),
        },
    }


# Smartest first (intelligence rank = list position).
FAKE_MODELS = [
    make_model("alpha/frontier-1", 10.0, 30.0),
    make_model("beta/strong-2", 2.0, 8.0),
    make_model("alpha/mid-3", 1.0, 3.0),
    make_model("gamma/cheap-4", 0.2, 0.6),
    make_model("delta/cheapest-5", 0.05, 0.2, context=64_000),
    make_model("epsilon/tiny-6:free", 0.0, 0.0),
    make_model("zeta/no-tools-7", 0.1, 0.3, params=()),
    make_model("eta/small-context-8", 0.1, 0.3, context=8_000),
]

# Fastest first.
FAKE_LATENCY_ORDER = [
    "delta/cheapest-5",
    "gamma/cheap-4",
    "eta/small-context-8",
    "zeta/no-tools-7",
    "alpha/mid-3",
    "beta/strong-2",
    "alpha/frontier-1",
    "epsilon/tiny-6:free",
]


class FakeCatalog:
    """Duck-typed Catalog replacement with a fixed, deterministic model set."""

    as_of = "2026-07-10T00:00:00+00:00"
    source = "fake_catalog"

    def __init__(self, models: list[dict[str, Any]] | None = None, latency_order: list[str] | None = None):
        self._models = models if models is not None else FAKE_MODELS
        self._latency = latency_order if latency_order is not None else FAKE_LATENCY_ORDER
        self._rank = {m["id"]: i for i, m in enumerate(self._models)}
        self._latency_rank = {mid: i for i, mid in enumerate(self._latency)}

    def models(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        return self._models

    def get(self, model_id: str) -> dict[str, Any] | None:
        idx = self._rank.get(model_id)
        return self._models[idx] if idx is not None else None

    def intelligence_rank(self, model_id: str) -> int:
        return self._rank.get(model_id, len(self._rank))

    def latency_rank(self, model_id: str) -> int:
        return self._latency_rank.get(model_id, len(self._latency_rank))

    def size(self) -> int:
        return len(self._models)


@pytest.fixture
def fake_catalog() -> FakeCatalog:
    return FakeCatalog()


@pytest.fixture(autouse=True)
def _reset_catalog_cache():
    Catalog.reset_cache()
    yield
    Catalog.reset_cache()
