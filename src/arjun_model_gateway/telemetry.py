"""
=============================================================================
MODULE: telemetry.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
SQLite persistence for the router's two ledgers:

  1. `model_calls` — one row per attempt (append-as-you-go: written in the
     request's `finally` block, so crashes still leave a record).
  2. `outcomes`   — append-only verification ledger. Rows arrive either
     automatically from the gateway (schema validation, tool-call checks,
     transport failures) or externally via POST /v1/outcomes (test suites,
     humans). This table is what turns the router from "picks by price" into
     "learns which models actually succeed": success_counts() feeds the Beta
     posteriors in selector.py.

Also provides summarize() / write_summary_report() for reporting. Mock calls
are excluded from performance and cost aggregates. Throughput aggregates are
computed as sum(completion_tokens) / sum(generation_time) — never an average
of per-call ratios.

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/data/telemetry.sqlite
  (read for summaries and success counts; created on first use)

OUTPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/data/telemetry.sqlite
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/reports/telemetry_summary.md
  (when write_summary_report() is called)
=============================================================================
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DEFAULT_TELEMETRY_PATH, ensure_dirs
from .models import ChatCompletionRequest, TelemetryRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_calls (
  request_id TEXT PRIMARY KEY,
  timestamp TEXT NOT NULL,
  repo TEXT,
  script TEXT,
  call_site_id TEXT,
  routing_profile TEXT NOT NULL,
  inferred_task_type TEXT,
  chosen_route TEXT NOT NULL,
  chosen_model TEXT NOT NULL,
  resolved_model TEXT,
  chosen_provider TEXT NOT NULL,
  fallback_models TEXT NOT NULL,
  prompt_token_count INTEGER NOT NULL,
  completion_token_count INTEGER NOT NULL,
  total_token_count INTEGER NOT NULL,
  total_cost_usd REAL,
  input_cost_usd REAL,
  output_cost_usd REAL,
  cost_source TEXT,
  cost_per_1k_tokens_usd REAL,
  output_input_token_ratio REAL,
  estimated_route_cost_usd REAL,
  latency_ms REAL NOT NULL,
  time_to_first_token_ms REAL,
  generation_time_ms REAL,
  throughput_tokens_per_second REAL,
  prompt_tokens_per_second REAL,
  completion_tokens_per_second REAL,
  schema_pass INTEGER,
  tool_pass INTEGER,
  tests_pass INTEGER,
  retries INTEGER NOT NULL,
  fallback_count INTEGER NOT NULL,
  error_type TEXT,
  verification_result TEXT,
  output_hash TEXT,
  prompt_hash TEXT,
  missing_metadata TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS outcomes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  verifier TEXT NOT NULL,
  success INTEGER NOT NULL,
  detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_outcomes_request ON outcomes(request_id);
"""


def connect(path: Path = DEFAULT_TELEMETRY_PATH) -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(model_calls)").fetchall()}
    migrations = {
        "total_token_count": "ALTER TABLE model_calls ADD COLUMN total_token_count INTEGER NOT NULL DEFAULT 0",
        "input_cost_usd": "ALTER TABLE model_calls ADD COLUMN input_cost_usd REAL",
        "output_cost_usd": "ALTER TABLE model_calls ADD COLUMN output_cost_usd REAL",
        "cost_source": "ALTER TABLE model_calls ADD COLUMN cost_source TEXT",
        "cost_per_1k_tokens_usd": "ALTER TABLE model_calls ADD COLUMN cost_per_1k_tokens_usd REAL",
        "output_input_token_ratio": "ALTER TABLE model_calls ADD COLUMN output_input_token_ratio REAL",
        "estimated_route_cost_usd": "ALTER TABLE model_calls ADD COLUMN estimated_route_cost_usd REAL",
        "resolved_model": "ALTER TABLE model_calls ADD COLUMN resolved_model TEXT",
        "prompt_tokens_per_second": "ALTER TABLE model_calls ADD COLUMN prompt_tokens_per_second REAL",
        "completion_tokens_per_second": (
            "ALTER TABLE model_calls ADD COLUMN completion_tokens_per_second REAL"
        ),
    }
    for column, statement in migrations.items():
        if column not in existing:
            conn.execute(statement)
    conn.commit()
    return conn


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def missing_metadata(request: ChatCompletionRequest) -> list[str]:
    meta = request.metadata
    required = ["repo", "script", "call_site_id"]
    return [field for field in required if getattr(meta, field) in (None, "")]


def write_record(record: TelemetryRecord, path: Path = DEFAULT_TELEMETRY_PATH) -> None:
    conn = connect(path)
    data = record.model_dump()
    data["timestamp"] = record.timestamp.isoformat()
    data["fallback_models"] = json.dumps(record.fallback_models, sort_keys=True)
    data["missing_metadata"] = json.dumps(record.missing_metadata, sort_keys=True)
    for field in ("schema_pass", "tool_pass", "tests_pass"):
        if data[field] is not None:
            data[field] = int(data[field])
    columns = ", ".join(data)
    placeholders = ", ".join(f":{k}" for k in data)
    conn.execute(f"INSERT OR REPLACE INTO model_calls ({columns}) VALUES ({placeholders})", data)
    conn.commit()
    conn.close()


def update_generation_stats(
    request_id: str,
    *,
    time_to_first_token_ms: float | None,
    generation_time_ms: float | None,
    path: Path = DEFAULT_TELEMETRY_PATH,
) -> None:
    """Back-fill true timing stats from OpenRouter's /v1/generation endpoint."""
    conn = connect(path)
    throughput = None
    if generation_time_ms:
        row = conn.execute(
            "SELECT completion_token_count FROM model_calls WHERE request_id = ?", (request_id,)
        ).fetchone()
        if row and row[0]:
            throughput = row[0] / (generation_time_ms / 1000.0)
    conn.execute(
        """
        UPDATE model_calls
        SET time_to_first_token_ms = COALESCE(?, time_to_first_token_ms),
            generation_time_ms = COALESCE(?, generation_time_ms),
            completion_tokens_per_second = COALESCE(?, completion_tokens_per_second)
        WHERE request_id = ?
        """,
        (time_to_first_token_ms, generation_time_ms, throughput, request_id),
    )
    conn.commit()
    conn.close()


# -- outcomes ledger ----------------------------------------------------------


def record_outcome(
    request_id: str,
    *,
    success: bool,
    verifier: str,
    detail: str | None = None,
    path: Path = DEFAULT_TELEMETRY_PATH,
) -> None:
    conn = connect(path)
    conn.execute(
        "INSERT INTO outcomes (request_id, timestamp, verifier, success, detail) VALUES (?, ?, ?, ?, ?)",
        (request_id, datetime.now(timezone.utc).isoformat(), verifier, int(success), detail),
    )
    # keep the attempt row's verification summary in sync (latest outcome wins)
    conn.execute(
        "UPDATE model_calls SET verification_result = ? WHERE request_id = ?",
        (f"{verifier}:{'pass' if success else 'fail'}", request_id),
    )
    conn.commit()
    conn.close()


def success_counts(
    path: Path = DEFAULT_TELEMETRY_PATH,
) -> tuple[dict[tuple[str, str], tuple[int, int]], dict[str, tuple[int, int]]]:
    """Verified (successes, failures) per (profile, model) and per model overall."""
    conn = connect(path)
    rows = conn.execute(
        """
        SELECT mc.routing_profile, mc.chosen_model, o.success, COUNT(*)
        FROM outcomes o
        JOIN model_calls mc ON mc.request_id = o.request_id
        WHERE mc.chosen_provider != 'mock'
        GROUP BY mc.routing_profile, mc.chosen_model, o.success
        """
    ).fetchall()
    conn.close()
    scoped: dict[tuple[str, str], list[int]] = {}
    overall: dict[str, list[int]] = {}
    for profile, model, success, count in rows:
        pair = scoped.setdefault((profile, model), [0, 0])
        pair[0 if success else 1] += count
        agg = overall.setdefault(model, [0, 0])
        agg[0 if success else 1] += count
    return (
        {key: (s, f) for key, (s, f) in scoped.items()},
        {key: (s, f) for key, (s, f) in overall.items()},
    )


def latency_observations(
    path: Path = DEFAULT_TELEMETRY_PATH,
) -> dict[tuple[str, str], tuple[int, float]]:
    """Observed (count, mean wall-time ms) per (profile, model) for successful
    non-mock calls. Feeds the selector's `fastest` ordering — OpenRouter's
    latency sort is TTFT-based and badly mispredicts end-to-end time for
    reasoning models orchestrating server tools."""
    conn = connect(path)
    rows = conn.execute(
        """
        SELECT routing_profile, chosen_model, COUNT(*), AVG(latency_ms)
        FROM model_calls
        WHERE chosen_provider != 'mock' AND error_type IS NULL AND latency_ms > 0
        GROUP BY routing_profile, chosen_model
        """
    ).fetchall()
    conn.close()
    return {(profile, model): (count, avg) for profile, model, count, avg in rows}


# -- reporting ---------------------------------------------------------------

NON_MOCK = "chosen_provider != 'mock'"


def summarize(path: Path = DEFAULT_TELEMETRY_PATH) -> dict:
    conn = connect(path)
    conn.row_factory = sqlite3.Row
    totals = conn.execute(
        f"""
        SELECT
          COUNT(*) AS calls,
          COALESCE(SUM(prompt_token_count), 0) AS prompt_tokens,
          COALESCE(SUM(completion_token_count), 0) AS completion_tokens,
          COALESCE(SUM(total_token_count), 0) AS total_tokens,
          SUM(total_cost_usd) AS total_cost_usd,
          AVG(latency_ms) AS avg_latency_ms,
          CASE WHEN SUM(generation_time_ms) > 0
               THEN SUM(completion_token_count) * 1000.0 / SUM(generation_time_ms)
          END AS true_completion_tokens_per_second,
          AVG(time_to_first_token_ms) AS avg_ttft_ms
        FROM model_calls WHERE {NON_MOCK}
        """
    ).fetchone()
    by_model = conn.execute(
        f"""
        SELECT
          chosen_provider, chosen_model,
          COUNT(*) AS calls,
          COALESCE(SUM(total_token_count), 0) AS total_tokens,
          SUM(total_cost_usd) AS total_cost_usd,
          AVG(latency_ms) AS avg_latency_ms,
          CASE WHEN SUM(generation_time_ms) > 0
               THEN SUM(completion_token_count) * 1000.0 / SUM(generation_time_ms)
          END AS true_completion_tokens_per_second
        FROM model_calls WHERE {NON_MOCK}
        GROUP BY chosen_provider, chosen_model
        ORDER BY calls DESC, chosen_provider, chosen_model
        """
    ).fetchall()
    by_profile = conn.execute(
        f"""
        SELECT
          routing_profile,
          COUNT(*) AS calls,
          COALESCE(SUM(total_token_count), 0) AS total_tokens,
          SUM(total_cost_usd) AS total_cost_usd,
          AVG(latency_ms) AS avg_latency_ms
        FROM model_calls WHERE {NON_MOCK}
        GROUP BY routing_profile
        ORDER BY calls DESC, routing_profile
        """
    ).fetchall()
    verification = conn.execute(
        """
        SELECT mc.routing_profile, mc.chosen_model,
               SUM(o.success) AS successes,
               SUM(1 - o.success) AS failures
        FROM outcomes o
        JOIN model_calls mc ON mc.request_id = o.request_id
        WHERE mc.chosen_provider != 'mock'
        GROUP BY mc.routing_profile, mc.chosen_model
        ORDER BY mc.routing_profile, mc.chosen_model
        """
    ).fetchall()
    conn.close()
    return {
        "totals": dict(totals),
        "by_provider_model": [dict(row) for row in by_model],
        "by_profile": [dict(row) for row in by_profile],
        "verified_outcomes": [dict(row) for row in verification],
    }


def _fmt(value: Any) -> str:
    """Render report cells: never 'n/a'; genuinely-undefined renders as em dash."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_summary_report(output: Path, path: Path = DEFAULT_TELEMETRY_PATH) -> dict:
    summary = summarize(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    totals = summary["totals"]
    lines = [
        "# Arjun Model Gateway Telemetry Summary",
        "",
        "Mock calls are excluded. `—` = no value has ever existed for that cell.",
        "",
        "## Totals",
        "",
        f"- Calls: {_fmt(totals['calls'])}",
        f"- Prompt tokens: {_fmt(totals['prompt_tokens'])}",
        f"- Completion tokens: {_fmt(totals['completion_tokens'])}",
        f"- Total tokens: {_fmt(totals['total_tokens'])}",
        f"- Total cost USD: {_fmt(totals['total_cost_usd'])}",
        f"- Average gateway latency ms: {_fmt(totals['avg_latency_ms'])}",
        f"- True completion tokens/sec: {_fmt(totals['true_completion_tokens_per_second'])}",
        f"- Average time-to-first-token ms: {_fmt(totals['avg_ttft_ms'])}",
        "",
        "## By Provider And Model",
        "",
        "| Provider | Model | Calls | Total Tokens | Total Cost USD | Avg Latency ms | True Tokens/sec |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["by_provider_model"]:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                *(
                    _fmt(row[k])
                    for k in (
                        "chosen_provider",
                        "chosen_model",
                        "calls",
                        "total_tokens",
                        "total_cost_usd",
                        "avg_latency_ms",
                        "true_completion_tokens_per_second",
                    )
                )
            )
        )
    lines.extend(
        [
            "",
            "## By Profile",
            "",
            "| Profile | Calls | Total Tokens | Total Cost USD | Avg Latency ms |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["by_profile"]:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                *(
                    _fmt(row[k])
                    for k in (
                        "routing_profile",
                        "calls",
                        "total_tokens",
                        "total_cost_usd",
                        "avg_latency_ms",
                    )
                )
            )
        )
    lines.extend(
        [
            "",
            "## Verified Outcomes (learning signal)",
            "",
            "| Profile | Model | Successes | Failures |",
            "|---|---|---:|---:|",
        ]
    )
    for row in summary["verified_outcomes"]:
        lines.append(
            "| {} | {} | {} | {} |".format(
                *(_fmt(row[k]) for k in ("routing_profile", "chosen_model", "successes", "failures"))
            )
        )
    output.write_text("\n".join(lines) + "\n")
    return summary
