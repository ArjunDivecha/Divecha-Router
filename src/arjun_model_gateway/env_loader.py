from __future__ import annotations

import os
import re
from pathlib import Path


DEFAULT_ENV_FILE = Path("/Users/arjundivecha/Dropbox/AAA Backup/.env.txt")
SAFE_KEY_PATTERN = re.compile(
    r"^(OPENROUTER_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|CELERIS_API_KEY)$"
)
KEY_PREFIXES = {
    "OPENROUTER_API_KEY": "sk-or-",
    "OPENAI_API_KEY": "sk-proj-",
    "ANTHROPIC_API_KEY": "sk-ant-",
    "CELERIS_API_KEY": "ck_",
}


def load_provider_keys(path: Path = DEFAULT_ENV_FILE, override: bool = True) -> list[str]:
    loaded: set[str] = set()
    if not path.exists():
        return loaded
    for line in path.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if " #" in value:
            value = value.split(" #", 1)[0].strip()
        expected_prefix = KEY_PREFIXES.get(key)
        if not SAFE_KEY_PATTERN.match(key) or not value:
            continue
        if expected_prefix and not value.startswith(expected_prefix):
            continue
        if key in loaded:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            loaded.add(key)
    return sorted(loaded)
