from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import REPORTS_DIR, ensure_dirs


SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "data",
    "dist",
    "build",
    "reports",
    "runs",
}
SECRET_NAMES = {".env", ".env.local", ".env.production", "id_rsa", "id_ed25519"}
TEXT_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".md",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ipynb",
}

PATTERNS = {
    "openai_sdk": re.compile(r"\bOpenAI\(|openai\.(chat|responses|Completion|Embedding)", re.I),
    "anthropic_sdk": re.compile(r"\bAnthropic\(|anthropic\.messages|claude-", re.I),
    "openrouter": re.compile(r"openrouter|OPENROUTER_API_KEY", re.I),
    "gemini": re.compile(r"gemini|google\.genai|GenerativeModel", re.I),
    "litellm": re.compile(r"litellm", re.I),
    "local_model": re.compile(r"ollama|lm studio|mlx|vllm|localhost:\d+/(v1|api)", re.I),
    "hardcoded_model": re.compile(
        r"(gpt-[\w.\-]+|claude-[\w.\-]+|gemini-[\w.\-]+|llama[-/\w.]+|qwen[-/\w.]+|mistral[-/\w.]+)",
        re.I,
    ),
    "json_schema": re.compile(r"BaseModel|response_format|json_schema|pydantic", re.I),
    "tool_calling": re.compile(r"tools\s*=|function_call|tool_choice", re.I),
    "eval_candidate": re.compile(r"eval|judge|golden|expected|pytest|unittest", re.I),
}


@dataclass
class Finding:
    path: str
    line: int
    kind: str
    snippet: str
    workload: str


def _workload(kinds: set[str], line: str) -> str:
    lower = line.lower()
    if "json_schema" in kinds:
        return "structured_output"
    if "tool_calling" in kinds:
        return "tool_calling"
    if "eval_candidate" in kinds:
        return "model_as_judge_or_eval"
    if any(word in lower for word in ("bloomberg", "portfolio", "finance", "factor", "alpha")):
        return "finance_reasoning"
    if any(word in lower for word in ("diff", "repo", "pytest", "refactor", "debug")):
        return "coding_repo"
    if "local_model" in kinds:
        return "local_private_inference"
    return "general_model_call"


def iter_files(root: Path):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_dir():
            continue
        if path.name in SECRET_NAMES:
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        yield path


def scan(root: Path, output: Path | None = None) -> dict:
    ensure_dirs()
    output = output or REPORTS_DIR / "repo_scan.json"
    output_resolved = output.resolve()
    findings: list[Finding] = []
    for path in iter_files(root):
        if path.resolve() == output_resolved:
            continue
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        for idx, line in enumerate(lines, start=1):
            kinds = {name for name, pattern in PATTERNS.items() if pattern.search(line)}
            if not kinds:
                continue
            rel = str(path.relative_to(root))
            for kind in sorted(kinds):
                findings.append(
                    Finding(
                        path=rel,
                        line=idx,
                        kind=kind,
                        snippet=line.strip()[:240],
                        workload=_workload(kinds, line),
                    )
                )
    report = {
        "root": str(root),
        "model_call_sites": [asdict(f) for f in findings if f.kind not in {"hardcoded_model"}],
        "hardcoded_model_report": [asdict(f) for f in findings if f.kind == "hardcoded_model"],
        "prompt_inventory": [asdict(f) for f in findings if "prompt" in f.snippet.lower()],
        "schema_inventory": [asdict(f) for f in findings if f.kind == "json_schema"],
        "tool_inventory": [asdict(f) for f in findings if f.kind == "tool_calling"],
        "eval_candidates": [asdict(f) for f in findings if f.kind == "eval_candidate"],
        "migration_plan": [
            {
                "path": f.path,
                "line": f.line,
                "recommended_profile": "structured" if f.workload == "structured_output" else "auto",
                "metadata_needed": ["repo", "script", "call_site_id", "routing_profile"],
            }
            for f in findings
            if f.kind in {"openai_sdk", "anthropic_sdk", "openrouter", "gemini", "litellm", "local_model"}
        ],
        "risk_report": [
            {"risk": "secret_files_skipped", "policy": "scanner never reads .env/private key filenames"},
            {"risk": "hardcoded_models", "count": len([f for f in findings if f.kind == "hardcoded_model"])},
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return report


def generate_eval_candidates(scan_report: dict, output: Path | None = None) -> dict:
    ensure_dirs()
    candidates = []
    for item in scan_report.get("eval_candidates", [])[:25]:
        candidates.append(
            {
                "name": f"{item['workload']}:{item['path']}:{item['line']}",
                "source": item,
                "input": "derive from source workflow",
                "expected_behavior": "preserve existing deterministic behavior",
                "scoring_method": "repo-native test/schema/golden check",
                "hard_fail_conditions": ["exception", "schema failure", "changed expected output"],
                "cost_budget_usd": 0.05,
                "latency_budget_ms": 60_000,
                "candidate_routes": ["auto", "cheap", "code", "structured", "frontier"],
                "verification_gate": "to be concretized during pilot migration",
            }
        )
    report = {"evals": candidates, "count": len(candidates)}
    output = output or REPORTS_DIR / "eval_candidates.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return report
