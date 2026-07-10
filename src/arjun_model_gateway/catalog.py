"""
=============================================================================
MODULE: catalog.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
Live, ranked view of the OpenRouter model catalog. This module is the reason
the router never hardcodes model IDs: every routing decision starts from the
current catalog, fetched from OpenRouter's public Models API with two
server-side sort orders:

  1. sort=intelligence-high-to-low  -> each model gets an "intelligence rank"
     (rank 0 = smartest model available today).
  2. sort=latency-low-to-high       -> each model gets a "latency rank"
     (rank 0 = fastest model available today).

New models released on OpenRouter appear here automatically on the next
refresh — nothing in config needs to change.

Caching: the catalog is cached in-process for AMG_CATALOG_TTL_SECONDS
(default 900s) and persisted to a JSON snapshot so the router still works
offline (snapshot is the explicit, logged fallback — not a silent one).

INPUT FILES:
- (network) https://openrouter.ai/api/v1/models  (public, no API key needed)
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/data/catalog_snapshot.json
  (fallback when the network fetch fails)

OUTPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/data/catalog_snapshot.json
  (rewritten on every successful live fetch)
=============================================================================
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .config import DATA_DIR, ensure_dirs

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_CATALOG_SNAPSHOT = DATA_DIR / "catalog_snapshot.json"


class CatalogError(RuntimeError):
    """Raised when no live catalog and no snapshot are available. FAIL IS FAIL."""


class Catalog:
    """Cached, doubly-ranked OpenRouter model catalog."""

    # class-level cache shared across requests in one process
    _models: list[dict[str, Any]] | None = None
    _intelligence_rank: dict[str, int] | None = None
    _latency_rank: dict[str, int] | None = None
    _loaded_at: float | None = None
    _as_of: str | None = None
    _source: str = "unloaded"

    def __init__(
        self,
        ttl_seconds: int | None = None,
        snapshot_path: Path = DEFAULT_CATALOG_SNAPSHOT,
    ) -> None:
        self.ttl_seconds = ttl_seconds or int(os.environ.get("AMG_CATALOG_TTL_SECONDS", "900"))
        self.snapshot_path = snapshot_path

    # -- loading ------------------------------------------------------------

    def _ensure_loaded(self, *, refresh: bool = False) -> None:
        cls = self.__class__
        now = time.monotonic()
        if (
            not refresh
            and cls._models is not None
            and cls._loaded_at is not None
            and now - cls._loaded_at < self.ttl_seconds
        ):
            return
        try:
            intelligence, latency_ids = self._fetch_live()
            cls._source = "openrouter_live"
            cls._as_of = datetime.now(timezone.utc).isoformat()
            self._write_snapshot(intelligence, latency_ids, cls._as_of)
        except httpx.HTTPError:
            intelligence, latency_ids, as_of = self._read_snapshot()
            cls._source = "snapshot_fallback"
            cls._as_of = as_of
        cls._models = intelligence
        cls._intelligence_rank = {m["id"]: i for i, m in enumerate(intelligence)}
        cls._latency_rank = {mid: i for i, mid in enumerate(latency_ids)}
        cls._loaded_at = now

    def _fetch_live(self) -> tuple[list[dict[str, Any]], list[str]]:
        with httpx.Client(timeout=30) as client:
            smart = client.get(OPENROUTER_MODELS_URL, params={"sort": "intelligence-high-to-low"})
            smart.raise_for_status()
            fast = client.get(OPENROUTER_MODELS_URL, params={"sort": "latency-low-to-high"})
            fast.raise_for_status()
        intelligence = smart.json().get("data", [])
        latency_ids = [m.get("id") for m in fast.json().get("data", [])]
        if not intelligence:
            raise httpx.HTTPError("empty catalog from OpenRouter")
        return intelligence, latency_ids

    def _write_snapshot(self, models: list[dict[str, Any]], latency_ids: list[str], as_of: str) -> None:
        ensure_dirs()
        self.snapshot_path.write_text(
            json.dumps(
                {"as_of": as_of, "source": OPENROUTER_MODELS_URL, "models": models, "latency_order": latency_ids}
            )
        )

    def _read_snapshot(self) -> tuple[list[dict[str, Any]], list[str], str | None]:
        if not self.snapshot_path.exists():
            raise CatalogError(
                "OpenRouter catalog fetch failed and no snapshot exists at "
                f"{self.snapshot_path}. Run `amg intel refresh` while online."
            )
        payload = json.loads(self.snapshot_path.read_text())
        return payload.get("models", []), payload.get("latency_order", []), payload.get("as_of")

    # -- accessors ----------------------------------------------------------

    def models(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        """All models, ordered smartest-first."""
        self._ensure_loaded(refresh=refresh)
        return self.__class__._models or []

    def get(self, model_id: str) -> dict[str, Any] | None:
        self._ensure_loaded()
        idx = (self.__class__._intelligence_rank or {}).get(model_id)
        return self.__class__._models[idx] if idx is not None else None

    def intelligence_rank(self, model_id: str) -> int:
        """0 = smartest. Unknown models rank last."""
        self._ensure_loaded()
        ranks = self.__class__._intelligence_rank or {}
        return ranks.get(model_id, len(ranks))

    def latency_rank(self, model_id: str) -> int:
        """0 = fastest. Unknown models rank last."""
        self._ensure_loaded()
        ranks = self.__class__._latency_rank or {}
        return ranks.get(model_id, len(ranks))

    def size(self) -> int:
        return len(self.models())

    @property
    def as_of(self) -> str | None:
        self._ensure_loaded()
        return self.__class__._as_of

    @property
    def source(self) -> str:
        self._ensure_loaded()
        return self.__class__._source

    @classmethod
    def reset_cache(cls) -> None:
        """Test hook: drop the in-process cache."""
        cls._models = None
        cls._intelligence_rank = None
        cls._latency_rank = None
        cls._loaded_at = None
        cls._as_of = None
        cls._source = "unloaded"


# -- pricing helpers (shared by selector and telemetry) -----------------------


def prompt_price(model: dict[str, Any]) -> float:
    """USD per prompt token; unparsable pricing sorts to the bottom."""
    return _price(model, "prompt")


def completion_price(model: dict[str, Any]) -> float:
    return _price(model, "completion")


def _price(model: dict[str, Any], key: str) -> float:
    pricing = model.get("pricing") or {}
    value = pricing.get(key) or pricing.get("input" if key == "prompt" else "output") or 0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0  # $1/token: guarantees the model never wins a cost ranking


def is_free_variant(model: dict[str, Any]) -> bool:
    """':free' variants are rate-limited teasers; excluded unless opted in."""
    model_id = str(model.get("id") or "")
    if model_id.endswith(":free"):
        return True
    return prompt_price(model) == 0.0 and completion_price(model) == 0.0


def outputs_text(model: dict[str, Any]) -> bool:
    arch = model.get("architecture") or {}
    return "text" in (arch.get("output_modalities") or [])


def author_of(model_id: str) -> str:
    return model_id.split("/", 1)[0] if "/" in model_id else model_id
