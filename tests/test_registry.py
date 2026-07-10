import json

from arjun_model_gateway.registry import validate_registry


def test_registry_validation(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "version": "test",
                "source": "test",
                "models": [{"id": "openrouter/auto", "provider": "openrouter"}],
            }
        )
    )
    snapshot = validate_registry(path)
    assert snapshot.models[0].id == "openrouter/auto"

