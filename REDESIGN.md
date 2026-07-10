<!--
=============================================================================
DOCUMENT: REDESIGN.md
=============================================================================
WHAT THIS DOCUMENT IS:
Design document for Divecha Router v2 ("Arjun Model Gateway v2"). It explains,
in plain language, how the redesigned router works: how it dynamically picks
models from the live OpenRouter catalog instead of hardcoded lists, how it
learns which models actually succeed from recorded outcomes, and which request
controls are now enforced rather than decorative.

INPUT FILES: none (this is documentation).
OUTPUT FILES: none (this is documentation).
=============================================================================
-->

# Divecha Router v2 — Design

*2026-07-10. Supersedes the v1 "candidate_models" design after the ChatGPT Sol
review. Per Arjun: there is NO sensitive data — all ZDR/privacy machinery is
dropped except the explicit `private` (local-only) profile.*

## Objective

> Minimize expected cost to a verified-successful answer, subject to:
> quality floor, request budget, context/capability constraints.

Two hard requirements from Arjun:
1. **No hardcoded model IDs anywhere.** New models (Sol, Terra, Luna, Grok 4.5,
   next month's releases) must enter the candidate pool automatically.
2. **The router must learn.** Success priors come from recorded outcomes, not
   hand-typed YAML numbers.

## How dynamic selection works

The **catalog** layer fetches the live OpenRouter Models API twice per TTL
(default 15 min), with the snapshot on disk as offline fallback:

- `GET /api/v1/models?sort=intelligence-high-to-low` → every model gets an
  **intelligence rank** (verified working live: rank 1 = claude-fable-5 today).
- `GET /api/v1/models?sort=latency-low-to-high` → every model gets a
  **latency rank**.

The **selector** turns a profile (pure constraints, zero model IDs) into a
ranked candidate list:

1. **Filter** (hard gates): required parameters (`tools`, `structured_outputs`, …),
   min context (≥ profile floor and ≥ the request's `context_length_estimate`),
   max prompt price $/M, output modality, text-output models only.
2. **Quality floor** (from profile or request metadata): a model must sit in the
   top slice of the intelligence ranking —
   `basic` = any, `medium` = top 60%, `high` = top 25%, `frontier` = top 8 models.
3. **Score** every survivor: `expected_route_cost_usd / p_success`, lowest wins.
   - `expected_route_cost_usd` prices the **whole route graph**, not just the
     outer model (see below).
   - `p_success` is a Beta posterior (see "Learning loop").
4. **Objective ordering**: profiles declare `objective: cheapest_success`
   (score order), `fastest` (latency-rank order over the filtered pool, and the
   request also carries `provider: {"sort": "latency"}`), or `smartest`
   (intelligence-rank order, price-capped).
5. **Budget gate**: candidates whose estimated route cost exceeds
   `maximum_budget_usd` (request metadata, else profile default) are dropped.
   If nothing fits: **HTTP 400 with the estimate — FAIL IS FAIL**, no silent
   downgrade.
6. Top candidate = primary; next `max_fallbacks` = executed fallbacks.

### Route graphs (fusion / advisor / subagent priced fully)

Server-tool routes are priced as a graph, with panel/judge/advisor models
**also resolved dynamically at decision time**:

| Route | Graph | Dynamic resolution |
|---|---|---|
| simple | outer | — |
| advisor | outer + advisor | advisor = top intelligence model with `tools` |
| fusion | outer + N panel + judge | panel = top-intelligence models from N **distinct authors** under the price cap; judge = top intelligence model |
| subagent | outer + worker | worker = cheapest model passing the profile filter |

Estimated graph cost = Σ over nodes of (expected_input × prompt_price +
expected_output × completion_price), with documented per-node token
multipliers. The breakdown is logged in the route decision's `reasons`.
Server-tool profiles set `tool_choice: "required"` so the route name
guarantees the tool actually runs.

## Learning loop (replaces success_priors.yaml)

- Every call writes an **attempt** row (existing `model_calls` table).
- New **`outcomes`** table (append-only): `request_id, verifier, success, detail, ts`.
- **Automatic outcomes**, recorded synchronously by the gateway:
  - schema requested → response parsed + validated against the JSON schema
    (`verifier="schema"`);
  - tools required → did the response contain a tool call (`verifier="tool_call"`);
  - transport/HTTP error → failure (`verifier="transport"`).
- **External outcomes**: `POST /v1/outcomes {request_id, success, verifier, detail}`
  — for test suites, diff gates, or Arjun saying "that was wrong".
- `p_success(profile, model)` = Beta posterior:
  - **Seed prior from intelligence rank** (new models start optimistic in
    proportion to their rank): `prior_mean = clamp(0.92 − 0.35 × rank_pct, 0.50, 0.95)`,
    pseudo-count 6.
  - Posterior mean = `(α₀ + successes) / (α₀ + β₀ + successes + failures)`,
    scoped per (profile, model), falling back to per-model across profiles.
  - Real outcomes dominate after ~6 observations. A model that keeps failing
    schema validation gets priced out of the route automatically.
- `config/success_priors.yaml` is retained only as optional manual seed
  overrides (merged as extra pseudo-counts); it is no longer required.

## Enforced request controls (previously decorative)

| Control | v2 behavior |
|---|---|
| `maximum_budget_usd` | Pre-flight filter + 400 if no candidate fits; actual vs estimate logged |
| `structured_output_schema` | Injected as real `response_format: {type: json_schema, strict: true}`; `structured_outputs` capability auto-required; response validated; outcome recorded |
| `tool_requirements` | Auto-adds `tools` to required parameters |
| `context_length_estimate` | Raises the min-context filter |
| `quality_floor` | Request metadata overrides profile floor |
| `latency_priority: high` | Switches objective to `fastest` + provider latency sort |

## Telemetry fixes

- OpenRouter payloads send `usage: {"include": true}` → exact cost + token
  details in every response (no more registry guessing when live).
- After each OpenRouter call, a **background thread** queries
  `GET /v1/generation?id=…` (retry ~3× over a few seconds, best-effort) and
  back-fills `time_to_first_token_ms`, `generation_time_ms`, and
  **true throughput = completion_tokens / generation_time**.
- Mock calls are excluded from stats (`cost_source='mock'` filter).
- Streaming is now **real SSE passthrough** (httpx stream → client), with the
  final usage chunk captured for telemetry.

## API surface (v2)

- `POST /v1/chat/completions` — OpenAI-compatible, real streaming.
- `POST /v1/outcomes` — outcome callback (the learning signal).
- `POST /v1/routes/explain` — full decision: candidates, scores, graph cost breakdown.
- `GET /v1/models` — OpenAI-compatible list of routing profiles (so SDK
  model-pickers work) plus catalog metadata.
- `GET /v1/telemetry/summary` — totals, by-model, by-profile, success rates.
- `GET /health` — includes catalog age and model count.

Deferred (recorded, not built): images/embeddings/Responses API passthrough,
auth/rate limits. The gateway is single-user on localhost.

## What stayed

FastAPI app, adapter pattern (OpenRouter / OpenAI / Anthropic / local / mock),
TTL cache + snapshot fallback, SQLite telemetry, repo scanner, Typer CLI,
auto-classifier routing (`model="auto"`), gateway-side fallback execution.

## Key files

- `src/arjun_model_gateway/catalog.py` — live ranked catalog (new)
- `src/arjun_model_gateway/selector.py` — filtering/scoring/route graphs (new)
- `src/arjun_model_gateway/policy.py` — route decision (rewritten)
- `src/arjun_model_gateway/gateway.py` — enforcement + outcomes + streaming (rewritten)
- `src/arjun_model_gateway/telemetry.py` — outcomes + posteriors (extended)
- `config/policy.yaml` — v3 schema, **zero model IDs**
- `data/telemetry.sqlite` — attempts + outcomes ledger
