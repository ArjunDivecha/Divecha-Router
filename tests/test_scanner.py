from pathlib import Path

from arjun_model_gateway.scanner import scan


def test_scanner_finds_model_calls_and_skips_env(tmp_path):
    (tmp_path / "app.py").write_text(
        """
from openai import OpenAI
MODEL = "gpt-4.1-mini"
client = OpenAI()
client.chat.completions.create(model=MODEL, messages=[])
"""
    )
    (tmp_path / ".env").write_text("OPENAI_API_KEY=do_not_read")
    report = scan(tmp_path, tmp_path / "scan.json")
    assert report["model_call_sites"]
    assert report["hardcoded_model_report"]
    assert "do_not_read" not in Path(tmp_path / "scan.json").read_text()

