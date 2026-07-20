<!--
=============================================================================
DOCUMENT: README.md — Arjun Model Gateway (Divecha Router v2)
=============================================================================
Project overview and operating guide. See REDESIGN.md for the full v2 design.
INPUT FILES: none (documentation). OUTPUT FILES: none (documentation).
=============================================================================
-->

# Arjun Model Gateway (v2)

Local, provider-neutral model router. One OpenAI-compatible endpoint for all
of Arjun's repos; the gateway picks the model.

**v2 core idea (2026-07-10 redesign):** no hardcoded model IDs anywhere.
Profiles declare *constraints + an objective*; the selector resolves them
against the **live OpenRouter catalog** (fetched with server-side
intelligence and latency rankings, TTL-cached, snapshot fallback), prices the
**full route graph** (fusion panel + judge included), and scores candidates
by **expected cost per verified success**, where P(success) is a Beta
posterior learned from the **outcomes ledger**. New models (Sol, Terra,
Luna, ...) become candidates automatically on the next catalog refresh.

## Deployment (how this actually runs)

The gateway is DEPLOYED as a macOS LaunchAgent — always on, starts at login,
auto-restarts on crash:

- plist: `/Users/arjundivecha/Library/LaunchAgents/com.arjundivecha.amg.plist`
- logs: `runs/launchd.out.log` / `runs/launchd.err.log` in this project
- restart: `launchctl kickstart -k gui/$(id -u)/com.arjundivecha.amg`
- stop:    `launchctl bootout gui/$(id -u)/com.arjundivecha.amg`
- after `git pull` or code edits: kickstart to pick up the new code

Any repo's code uses it with a stock OpenAI client — no OpenRouter/Anthropic
SDKs, no keys in application code (keys live only in the gateway):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="local")
reply = client.chat.completions.create(
    model="cheap",  # a routing profile, or "auto" — never a model name
    messages=[{"role": "user", "content": "Summarize this log ..."}],
    extra_body={"metadata": {"repo": "News", "script": "daily.py",
                             "call_site_id": "daily.summarize"}},
)
```

## Quick Start

```bash
. .venv/bin/activate            # python 3.12 venv, pip install -e '.[dev]'
amg policy validate
amg intel refresh               # live catalog + snapshot
amg intel choose --profile cheap
amg intel scorecard --profile code
amg eval routing-robustness --minimum 0.95
amg eval diversity --matrix evals/behavior_matrix.example.yaml \
  --max-cost-per-verified-success-usd 0.50
amg serve --load-env            # gateway on http://127.0.0.1:8765/v1
pytest
```

Application code passes routing intent through `metadata` (all fields are
**enforced**, none are decorative):

```json
{
  "model": "auto",
  "messages": [{"role": "user", "content": "Summarize this log"}],
  "metadata": {
    "repo": "News", "script": "daily.py", "call_site_id": "daily.summarize",
    "routing_profile": "cheap",
    "maximum_budget_usd": 0.05,
    "quality_floor": "medium",
    "structured_output_schema": {"type": "object", "required": ["summary"]}
  }
}
```

- `maximum_budget_usd` — pre-flight gate. If no qualifying route fits the
  budget the gateway returns **HTTP 400** with the estimate (fail is fail).
- `structured_output_schema` — injected as a real `response_format`
  json_schema AND validated on the way back; the result is recorded as an
  automatic outcome.
- `quality_floor` — basic / medium (top 60% of live intelligence ranking) /
  high (top 25%) / frontier (top 8 models).
- `latency_priority: "high"` — switches to the latency objective + OpenRouter
  provider latency sort.

## The learning loop

Every call writes an attempt row; every verification writes an outcome row:

- automatic: JSON-schema validation, required-tool-call checks, transport failures
- external: `POST /v1/outcomes {"request_id": "...", "success": true, "verifier": "tests"}`
  or `amg outcomes report REQUEST_ID --success/--no-success`

P(success) per (profile, model) is a Beta posterior seeded from the model's
live intelligence rank (~6 pseudo-observations) and updated by real outcomes,
so a model that keeps failing gets priced out of its route automatically.

## Routing quality gates

The auto-classifier is deterministic and free at request time. Its policy is
tested against `evals/routing_robustness.yaml`, where the same workload is
rewritten with character, word, syntax, paraphrase, and irrelevant-context
perturbations. This operationalizes the robustness test in
[When is Routing Meaningful?](https://arxiv.org/abs/2607.09197). The gate
requires both route equivalence and the expected profile, so a classifier
that is consistently wrong cannot pass:

```bash
amg eval routing-robustness --minimum 0.95
```

Candidate-model diversity is evaluated from private per-workload scores, not
provider labels or benchmark reputation. `amg eval diversity` computes
hierarchic social entropy over behavioral score vectors, flags near-duplicate
actors, applies verified-quality and cost-per-verified-success constraints,
and emits a small coreset, stopping when another actor adds no material joint
diversity-plus-quality value. The example file documents the input schema; replace
its illustrative actors with real eval results before using its report to
change production policy. Both eval commands write a stable latest report and
a timestamped comparison artifact under `reports/`.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible; real SSE passthrough streaming |
| `POST /v1/outcomes` | verification callback (the learning signal) |
| `POST /v1/routes/explain` | decision + candidate scores + route-graph costs, no execution |
| `GET /v1/models` | profiles as an OpenAI-compatible model list |
| `GET /v1/telemetry/summary` | totals, per-model, per-profile, verified outcomes |
| `GET /health` | includes catalog age/source/size |

## Profiles (config/policy.yaml, version 3)

cheap · fast · code (advisor) · research (fusion) · finance (fusion) ·
structured · long_context (subagent) · frontier · verify · private (local) ·
direct smokes. Server-tool routes resolve their support models dynamically
too: the advisor/judge is the current top intelligence model, the fusion
panel is the top distinct-author models under the profile's price cap, the
subagent worker is the cheapest qualifying model — all with
`tool_choice: "required"` so the route name guarantees the behavior.

Fusion tuning (measured 2026-07-10): the OUTER orchestrator model dominates
fusion wall-time — heavy reasoners (grok-4.5, gpt-5.6-sol) took 9–15 min as
orchestrators, while a fast orchestrator runs the same frontier panel in
~40s–5min. Fusion profiles use `objective: fastest`, where "fastest" is
learned from observed per-(profile, model) wall times in telemetry
(OpenRouter's latency sort is TTFT-based and mispredicts reasoning models),
with `max_tool_calls: 1`. Fusion executes via OpenRouter's **Responses API**
(their recommended path; the gateway translates to/from chat-completions
shape, so callers see no difference) — measured live: full frontier panel
with web search + Fable judge + synthesized final answer in ~5 min, $1.51,
vs the chat-completions beta path which returned unsynthesized panel dumps.
Budget cap ($2.50 research / $3.00 finance) and the 900s timeout apply, and
the budget is cumulative across fallback attempts.

## Telemetry (data/telemetry.sqlite)

`model_calls` records exact provider-billed cost (`usage.include`), the
catalog id (`chosen_model`, used for learning joins) and the
provider-resolved dated slug (`resolved_model`), the estimated route cost,
and — back-filled a few seconds after each call from OpenRouter's
`/v1/generation` — true time-to-first-token, generation time, and
throughput. Aggregates always compute throughput as
`sum(completion_tokens)/sum(generation_time)` and exclude mock calls.
`—` in reports means genuinely undefined; stale values carry a `*`.

## Files

- `REDESIGN.md` — v2 design rationale (post ChatGPT-Sol review)
- `src/arjun_model_gateway/catalog.py` — live doubly-ranked catalog
- `src/arjun_model_gateway/selector.py` — constraint filtering, floors, scoring, route graphs
- `src/arjun_model_gateway/policy.py` — decision assembly
- `src/arjun_model_gateway/routing_eval.py` — route robustness + behavioral diversity gates
- `src/arjun_model_gateway/gateway.py` — FastAPI service, enforcement, outcomes, streaming
- `config/policy.yaml` — profiles (zero model IDs; validated)
- `evals/routing_robustness.yaml` — semantic route-equivalence cases
- `config/success_priors.yaml` — optional manual seed overrides (empty by default)
- `contracts/mvp.spec.md` — implementation contract + execution ledger
- `data/` — telemetry.sqlite, catalog_snapshot.json, registry.json
- Legacy (superseded by v2, kept pending explicit deletion approval):
  `src/arjun_model_gateway/model_intel.py`, `tests/test_model_intel.py`,
  `data/model_intel_snapshot.json`

## Verified live (2026-07-10)

All 35 unit tests pass; live end-to-end against real OpenRouter: dynamic
selection (cheap→deepseek-v4-flash, fast→gpt-5.6-luna released this week,
frontier→claude-fable-5 with gpt-5.6-sol fallback), schema enforcement +
auto-outcome, budget rejection (HTTP 400), real streaming, advisor route,
outcome→posterior learning join, generation-stats back-fill. See
`runs/2026-07-10_v2_live_test/`.

## Verified after routing-robustness update (2026-07-14)

46 unit tests pass. The offline gate passes 6 workload cases and 30 semantic
variants at 100% route robustness and 100% expected-profile accuracy. After a
LaunchAgent restart, `/health` reported 343 live OpenRouter models from the
July 14 catalog; live no-execution route explanations kept both finance
phrasings on Fusion and both coding phrasings on Advisor. No paid provider
generation was used for these checks.
