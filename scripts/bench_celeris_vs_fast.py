"""
=============================================================================
SCRIPT: bench_celeris_vs_fast.py (Divecha Router v2)
=============================================================================

WHAT THIS DOES (for someone with no context):
Runs a head-to-head latency benchmark between two ways the Divecha Router can
serve short, latency-sensitive work:

  A) Celeris-1  — a diffusion LM, reached via the `fast_celeris` DIRECT route
                  (OpenAI-compatible endpoint at inference.celeris.ai).
  B) The `fast` profile's current marketplace WINNER on OpenRouter (selected
     live by the router's `fastest` objective; pinned once at start so every
     prompt hits the same model for a clean A/B).

It sends a fixed set of representative SHORT prompts (classification, JSON
extraction, query rewrite) to BOTH contenders, N trials each, interleaved to
avoid time-of-day bias, and measures END-TO-END wall time (the metric that
actually matters), plus token usage and per-call cost. Results are written
incrementally (crash-safe) and summarized with an xlsx, a parquet, a JSON
summary, and a light-mode PDF chart.

REAL API CALLS: this spends real money — OpenRouter (per-token) for the fast
winner and prepaid Celeris credits. Outputs are short, so total cost is a few
cents. Keys are read from the environment via the gateway's env_loader.

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/.env.txt   (API keys, via env_loader)
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/config/policy.yaml
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/data/catalog_snapshot.json
  (live OpenRouter catalog; refreshed on load)

OUTPUT FILES (all under a timestamped run dir inside the project):
- <run_dir>/trials.jsonl        one line per trial, appended live
- <run_dir>/results.parquet     canonical tabular results
- <run_dir>/results.xlsx        same, for eyeballing
- <run_dir>/summary.json        aggregate metrics + config
- <run_dir>/latency.pdf         light-mode bar chart (mean latency per prompt)
- <run_dir>/run.log             tee'd console log
where <run_dir> =
  /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/runs/<ts>_celeris_vs_fast/

DEPENDENCIES: arjun_model_gateway (this repo), pandas, openpyxl, pyarrow,
matplotlib.

USAGE:
  source .venv/bin/activate
  python scripts/bench_celeris_vs_fast.py [--trials 3]

NOTES:
- The fast winner is pinned at start via a fixed RouteDecision so latency
  learning can't swap models mid-run. Celeris resolves its single model live.
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from arjun_model_gateway.adapters import AdapterError, adapter_for
from arjun_model_gateway.catalog import Catalog
from arjun_model_gateway.config import validate_policy
from arjun_model_gateway.env_loader import load_provider_keys
from arjun_model_gateway.models import ChatCompletionRequest, ChatMessage, GatewayMetadata
from arjun_model_gateway.policy import choose_route
from arjun_model_gateway.selector import prompt_price, completion_price

PROJECT = Path("/Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router")

# (category, prompt, max_tokens) — short, bursty tasks: Celeris's stated lane.
PROMPTS = [
    ("classify", "Classify sentiment as positive/negative/neutral (one word): The delivery was late and the box was crushed.", 8),
    ("classify", "Is this a question or a statement? Reply one word: How do I reset my password", 8),
    ("classify", "Label the topic (one word) — sports/politics/tech/food: The new GPU doubles inference throughput.", 8),
    ("classify", "Spam or not spam? One word: Congratulations, you won a free cruise, click here now!!!", 8),
    ("extract", "Extract as JSON {name, amount, currency}: Invoice from Acme for 4,200 USD.", 128),
    ("extract", "Extract as JSON {city, temp_c, condition}: It's 18C and drizzling in Dublin right now.", 128),
    ("extract", "Extract all email addresses as a JSON array: contact bob@x.io or sales@y.com for info.", 128),
    ("rewrite", "Rewrite as a concise web search query: I'm trying to figure out why my sourdough starter isn't rising anymore.", 48),
    ("rewrite", "Rewrite this politely and professionally: send me the report NOW, you forgot again.", 48),
    ("rewrite", "Turn this into an imperative git commit subject (<=60 chars): I added a caching layer to speed up the fast route.", 48),
]


def log(msg: str, fh) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    fh.write(line + "\n")
    fh.flush()


def price_of(catalog: Catalog, model_id: str) -> tuple[float, float]:
    for m in catalog.models():
        if str(m.get("id")) == model_id:
            return prompt_price(m) * 1e6, completion_price(m) * 1e6
    return float("nan"), float("nan")


def run_once(request, decision, provider) -> dict:
    t = time.time()
    try:
        resp = adapter_for(provider).chat(request, decision)
        dt = time.time() - t
        return {
            "ok": True,
            "seconds": dt,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "text": (resp.output_text or "")[:200],
        }
    except AdapterError as e:
        return {"ok": False, "seconds": time.time() - t, "error": f"{e.error_type}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT / "runs" / f"{ts}_celeris_vs_fast"
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = open(run_dir / "run.log", "w")
    trials_path = run_dir / "trials.jsonl"

    load_provider_keys()
    policy = validate_policy()
    catalog = Catalog()

    # Pin the fast winner once (fixed RouteDecision) so it can't change mid-run.
    warm = ChatCompletionRequest(
        model="fast", messages=[ChatMessage(role="user", content="hi")],
        metadata=GatewayMetadata(routing_profile="fast"),
    )
    fast_decision = choose_route(warm, policy, catalog=catalog)
    fast_model = fast_decision.model
    cel_warm = ChatCompletionRequest(
        model="fast_celeris", messages=[ChatMessage(role="user", content="hi")],
        metadata=GatewayMetadata(routing_profile="fast_celeris"),
    )
    cel_decision = choose_route(cel_warm, policy, catalog=catalog)
    cel_model = cel_decision.model

    fast_pp, fast_cp = price_of(catalog, fast_model)
    cel_pp, cel_cp = 0.20, 0.70  # Celeris docs, 2026-08 (not on OpenRouter)

    contenders = [
        ("celeris", "fast_celeris", cel_decision, cel_model, cel_pp, cel_cp),
        ("fast_winner", "fast", fast_decision, fast_model, fast_pp, fast_cp),
    ]
    log(f"run dir: {run_dir}", logf)
    log(f"contenders: celeris={cel_model} (${cel_pp}/${cel_cp} per M)  "
        f"fast_winner={fast_model} (${fast_pp:.2f}/${fast_cp:.2f} per M)", logf)
    log(f"prompts={len(PROMPTS)} trials={args.trials} "
        f"(total calls={len(PROMPTS)*args.trials*2})", logf)

    rows = []
    with open(trials_path, "w") as tf:
        for pi, (cat, prompt, max_tok) in enumerate(PROMPTS):
            for trial in range(args.trials):
                # interleave contenders each trial
                for name, profile, decision, model, pp, cp in contenders:
                    req = ChatCompletionRequest(
                        model=profile,
                        messages=[ChatMessage(role="user", content=prompt)],
                        max_tokens=max_tok,
                        metadata=GatewayMetadata(routing_profile=profile),
                    )
                    r = run_once(req, decision, decision.provider)
                    cost = None
                    if r["ok"]:
                        cost = (r["prompt_tokens"] * pp + r["completion_tokens"] * cp) / 1e6
                    row = {
                        "prompt_idx": pi, "category": cat, "trial": trial,
                        "contender": name, "model": model,
                        "ok": r["ok"], "seconds": round(r["seconds"], 4),
                        "prompt_tokens": r.get("prompt_tokens"),
                        "completion_tokens": r.get("completion_tokens"),
                        "cost_usd": cost, "error": r.get("error"),
                        "text": r.get("text"),
                    }
                    rows.append(row)
                    tf.write(json.dumps(row) + "\n")
                    tf.flush()
                    status = "OK" if r["ok"] else f"FAIL({r.get('error')})"
                    log(f"  p{pi}[{cat}] t{trial} {name:11s} {r['seconds']:.3f}s {status}", logf)

    df = pd.DataFrame(rows)
    ok = df[df["ok"]]
    # per-contender aggregates on successful calls
    agg = ok.groupby("contender")["seconds"].agg(["count", "mean", "median", "min", "max"]).round(3)
    cost_agg = ok.groupby("contender")["cost_usd"].mean()
    fail = df[~df["ok"]]

    df.to_parquet(run_dir / "results.parquet", index=False)
    with pd.ExcelWriter(run_dir / "results.xlsx") as xw:
        df.to_excel(xw, sheet_name="trials", index=False)
        agg.to_excel(xw, sheet_name="latency_summary")
        # per-prompt mean latency, contenders side by side
        pivot = ok.pivot_table(index=["prompt_idx", "category"], columns="contender",
                               values="seconds", aggfunc="mean").round(3)
        pivot.to_excel(xw, sheet_name="per_prompt")

    summary = {
        "run_dir": str(run_dir),
        "timestamp": ts,
        "celeris_model": cel_model,
        "fast_winner_model": fast_model,
        "prices_per_million": {
            "celeris": [cel_pp, cel_cp], "fast_winner": [fast_pp, fast_cp]},
        "trials_per_prompt": args.trials,
        "n_prompts": len(PROMPTS),
        "latency_seconds": agg.to_dict(orient="index"),
        "mean_cost_usd": {k: (None if pd.isna(v) else round(v, 6)) for k, v in cost_agg.items()},
        "failures": len(fail),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # light-mode PDF chart: mean latency per prompt, contenders side by side
    pivot = ok.pivot_table(index="prompt_idx", columns="contender",
                           values="seconds", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")
    ax.set_facecolor("white")
    pivot.plot(kind="bar", ax=ax, color={"celeris": "#2563eb", "fast_winner": "#f59e0b"})
    ax.set_xlabel("prompt index")
    ax.set_ylabel("mean end-to-end latency (s)")
    ax.set_title(f"Celeris ({cel_model}) vs fast winner ({fast_model}) — lower is better")
    ax.legend(title="")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "latency.pdf")

    log("", logf)
    log("=== LATENCY SUMMARY (successful calls) ===", logf)
    for c in agg.index:
        log(f"  {c:11s} mean={agg.loc[c,'mean']:.3f}s  median={agg.loc[c,'median']:.3f}s  "
            f"min={agg.loc[c,'min']:.3f}s  n={int(agg.loc[c,'count'])}", logf)
    log(f"failures: {len(fail)}", logf)
    log(f"outputs: {run_dir}", logf)
    logf.close()
    print(f"\nDONE. Results in: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
