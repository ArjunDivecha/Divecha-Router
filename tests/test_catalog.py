"""
=============================================================================
MODULE: tests/test_catalog.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
Tests for the live-catalog layer: rank assignment from the two sorted
fetches, snapshot persistence, and explicit snapshot fallback when the
network fetch fails (with a loud CatalogError when neither is available).

INPUT FILES: none — no file I/O outside pytest tmp_path (snapshot files are
created under the per-test temporary directory only).
OUTPUT FILES: none — no file I/O outside pytest tmp_path.
=============================================================================
"""

from __future__ import annotations

import httpx
import pytest

from arjun_model_gateway.catalog import Catalog, CatalogError
from tests.conftest import FAKE_LATENCY_ORDER, FAKE_MODELS


def test_ranks_come_from_sorted_fetches(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot.json"
    catalog = Catalog(ttl_seconds=999, snapshot_path=snapshot, supplemental_path=None)
    monkeypatch.setattr(catalog, "_fetch_live", lambda: (FAKE_MODELS, FAKE_LATENCY_ORDER))
    assert catalog.models()[0]["id"] == "alpha/frontier-1"
    assert catalog.intelligence_rank("alpha/frontier-1") == 0
    assert catalog.intelligence_rank("gamma/cheap-4") == 3
    assert catalog.latency_rank("delta/cheapest-5") == 0
    assert catalog.intelligence_rank("unknown/model") == len(FAKE_MODELS)
    assert catalog.source == "openrouter_live"
    assert snapshot.exists()


def test_snapshot_fallback_when_fetch_fails(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot.json"
    live = Catalog(ttl_seconds=999, snapshot_path=snapshot, supplemental_path=None)
    monkeypatch.setattr(live, "_fetch_live", lambda: (FAKE_MODELS, FAKE_LATENCY_ORDER))
    live.models()  # writes the snapshot

    Catalog.reset_cache()
    offline = Catalog(ttl_seconds=999, snapshot_path=snapshot, supplemental_path=None)

    def fail():
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(offline, "_fetch_live", fail)
    assert offline.models()[0]["id"] == "alpha/frontier-1"
    assert offline.source == "snapshot_fallback"
    assert offline.latency_rank("delta/cheapest-5") == 0


def test_no_catalog_and_no_snapshot_fails_loudly(tmp_path, monkeypatch):
    catalog = Catalog(ttl_seconds=999, snapshot_path=tmp_path / "missing.json")

    def fail():
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(catalog, "_fetch_live", fail)
    with pytest.raises(CatalogError):
        catalog.models()
