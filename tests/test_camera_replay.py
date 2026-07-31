from pathlib import Path

import pytest

from arjun_model_gateway.camera_replay import assert_capsule_safe


def test_capsule_safety_rejects_private_address(tmp_path: Path) -> None:
    address = ".".join(["192", "168", "2", "44"])
    (tmp_path / "bad.txt").write_text("host=" + address, encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden private material"):
        assert_capsule_safe(tmp_path)


def test_capsule_safety_rejects_private_tailnet_name(tmp_path: Path) -> None:
    suffix = ".".join(["ts", "net"])
    (tmp_path / "bad.txt").write_text(f"camera.example.{suffix}", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden private material"):
        assert_capsule_safe(tmp_path)


def test_capsule_safety_allows_offline_fixture(tmp_path: Path) -> None:
    (tmp_path / "safe.txt").write_text(
        "APP_URL=http://camera.test and no credentials", encoding="utf-8"
    )
    assert_capsule_safe(tmp_path)
