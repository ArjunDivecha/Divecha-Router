from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .config import DEFAULT_REGISTRY_PATH, ensure_dirs
from .models import ModelRegistryEntry, RegistrySnapshot


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_IMAGE_MODELS_URL = "https://openrouter.ai/api/v1/images/models"


def _entry(item: dict[str, Any]) -> ModelRegistryEntry:
    architecture = item.get("architecture") or {}
    return ModelRegistryEntry(
        id=item["id"],
        name=item.get("name"),
        context_length=item.get("context_length"),
        input_modalities=architecture.get("input_modalities") or [],
        output_modalities=architecture.get("output_modalities") or [],
        supported_parameters=item.get("supported_parameters") or [],
        pricing=item.get("pricing") or {},
        top_provider=item.get("top_provider") or {},
        capabilities=[],
        raw=item,
    )


def refresh_openrouter_registry(path: Path = DEFAULT_REGISTRY_PATH) -> RegistrySnapshot:
    ensure_dirs()
    with httpx.Client(timeout=60) as client:
        response = client.get(OPENROUTER_MODELS_URL)
        response.raise_for_status()
        data = response.json()
    models = [_entry(item) for item in data.get("data", [])]
    snapshot = RegistrySnapshot(
        version=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        source=OPENROUTER_MODELS_URL,
        models=models,
    )
    path.write_text(snapshot.model_dump_json(indent=2))
    return snapshot


def load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> RegistrySnapshot:
    return RegistrySnapshot.model_validate(json.loads(path.read_text()))


def validate_registry(path: Path = DEFAULT_REGISTRY_PATH) -> RegistrySnapshot:
    snapshot = load_registry(path)
    if not snapshot.models:
        raise ValueError("registry has no models")
    ids = [model.id for model in snapshot.models]
    if len(ids) != len(set(ids)):
        raise ValueError("registry contains duplicate model ids")
    return snapshot

