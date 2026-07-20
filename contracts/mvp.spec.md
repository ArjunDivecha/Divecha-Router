# Arjun Model Gateway MVP Contract

## Scope

Allowed write paths:

- `README.md`
- `pyproject.toml`
- `config/**`
- `contracts/**`
- `evals/**`
- `src/**`
- `tests/**`
- `reports/**`
- `data/**`

Forbidden paths:

- `.env`
- `*.pem`
- `*.key`
- `*credential*`
- unrelated repos outside this folder
- production data outside this folder

Expected outputs:

- runnable local gateway
- valid default policy
- OpenRouter registry refresh command
- speed polling command
- telemetry SQLite store
- repo scanner report
- eval candidate report
- deterministic tests

## Falsifiable Bet

The MVP is useful if a local OpenAI-compatible endpoint can accept a request, route it through a provider-neutral policy decision, write complete telemetry, and validate registry/policy/scanner artifacts without touching forbidden files.

## Invariants

- Application callers request profiles/capabilities, not provider-specific route internals.
- Production inference is through gateway adapters, not MCP.
- Missing metadata is logged as a migration issue.
- Success is certified by tests, schema validation, route smoke checks, and diff scope, not model self-report.
- Secrets are never scanned as prompt content.

## Verification Gates

- `amg policy validate`
- `amg eval routing-robustness --minimum 0.95`
- `amg routes test --mock`
- `amg registry validate`
- `amg scanner scan --root .`
- `pytest`
- no forbidden files modified

## Budget

- max implementation turns: 6
- max consecutive failures: 3
- dollar cap for gates: $0 unless the user explicitly enables live provider tests
- live provider calls require API keys and explicit command choice

## Kill Conditions

- gateway adds opaque routing without logged decisions
- adapter errors cannot be normalized
- telemetry cannot be written
- scanner cannot avoid obvious secret files

## Graduate Conditions

- local mock route passes
- telemetry row is written
- policy and registry schemas validate
- scanner reports model call sites and hard-coded models
- tests pass
- semantic prompt variants preserve the expected routing profile at or above 95%

## Scale Conditions

- run scanner on three pilot repos
- add at least five real evals from discovered workflows
- enable one live OpenRouter route and one direct-provider fallback
- migrate a single scoped call site in a pilot repo

## Execution Ledger

- 2026-07-02: Created MVP contract from PRD.
- 2026-07-02: Added Opus-requested router hardening: cached model-intel snapshot, success-prior scorecards, privacy blocking, fallback execution, server-tool/client-tool merge, synthetic SSE streaming, and live Advisor/Fusion/Subagent smoke checks.
- 2026-07-10: v2 redesign after ChatGPT-Sol review (see REDESIGN.md). Dynamic
  model selection from the live OpenRouter catalog — zero hardcoded model IDs
  (validate_policy rejects pinned marketplace models); quality floors as live
  intelligence-rank slices; full route-graph cost estimation (fusion panel +
  judge priced); Beta-posterior success learning from a new append-only
  outcomes ledger (auto schema/tool/transport outcomes + POST /v1/outcomes);
  enforced budget (HTTP 400), schema injection+validation, context/tool
  filters; real SSE passthrough streaming with durable pre-stream telemetry;
  exact provider cost accounting (usage.include) and /v1/generation TTFT /
  generation-time back-fill. ZDR/privacy machinery dropped per Arjun (no
  sensitive data); local-only escape hatch retained. Verified live against
  real OpenRouter (runs/2026-07-10_v2_live_test): 35 unit tests + 8/8 live
  end-to-end checks pass. Known follow-up: fusion via Responses API (beta and
  slow on chat completions upstream).
- 2026-07-10 (later): DEPLOYED as macOS LaunchAgent (com.arjundivecha.amg,
  KeepAlive, RunAtLoad; logs in runs/). Deleted legacy v1 files
  (model_intel.py, test_model_intel.py, model_intel_snapshot.json) with
  Arjun's permission. Budget now enforced cumulatively across fallback
  attempts (cumulative_budget_exceeded). Fusion migrated to OpenRouter's
  Responses API with chat-completions-shape normalization — verified live
  through the deployed service: synthesized frontier-panel answer in 291s,
  $1.51 (the chat-completions beta path returned unsynthesized panel dumps).
  36 tests pass.
- 2026-07-14: Added paper-derived routing quality gates from arXiv:2607.09197:
  deterministic route-equivalence tests across character, word, syntax,
  paraphrase, and rambling perturbations; expected-profile accuracy so a
  stable but wrong route cannot pass; broader intent phrase families; and an
  HSE-based, quality/cost-gated actor coreset evaluator for private eval
  matrices. These checks are offline and add no classifier-model cost or
  request latency. Verification: 46 unit tests pass; 6 cases / 30 variants
  pass at 100% route robustness and expected-profile accuracy; deployed
  `/v1/routes/explain` preserved Fusion for finance paraphrases and Advisor
  for coding paraphrases against a refreshed 343-model catalog. Also fixed
  scanner self-ingestion of `reports/`, `runs/`, and catalog `data/` (384,783
  false findings reduced to 406 source findings).
