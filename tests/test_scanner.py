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


def test_scanner_skips_generated_artifacts_and_its_own_output(tmp_path):
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n")
    for directory in ("data", "reports", "runs"):
        path = tmp_path / directory
        path.mkdir()
        (path / "generated.json").write_text('{"model": "gpt-generated-only"}')

    output = tmp_path / "scan.json"
    first = scan(tmp_path, output)
    second = scan(tmp_path, output)

    assert {item["path"] for item in first["model_call_sites"]} == {"app.py"}
    assert first["model_call_sites"] == second["model_call_sites"]
    assert first["hardcoded_model_report"] == second["hardcoded_model_report"] == []
