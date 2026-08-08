<!--
=============================================================================
DOCUMENT: WHAT_THIS_ROUTER_DOES.md — friend-facing overview of the Divecha Router
=============================================================================
A plain-language explanation of what the Divecha Router (Arjun Model Gateway)
is, why it exists, how it decides, and how to use it. Written to be handed to
someone with no prior context.
INPUT FILES: none (documentation). OUTPUT FILES: none (documentation).
=============================================================================
-->

# The Divecha Router — what it is, in plain terms

## The one-sentence version

It's a **local "traffic controller" for AI models**: your code talks to a single
OpenAI-compatible endpoint and asks for a *kind of job* (cheap, fast, coding,
research…), and the router picks the **best actual model for that job right now**
— then bills, logs, and learns from the result.

## The problem it solves

If you build anything on LLMs, you hit three annoyances fast:

1. **Model names rotate constantly.** You hardcode `gpt-5.6-luna` today; next
   month there's a better/cheaper one and every script is stale.
2. **You keep re-deciding the same thing.** "Is this worth a frontier model or
   will a cheap one do?" — over and over, in every repo.
3. **Keys and providers leak everywhere.** Every project ends up with its own
   API keys and vendor SDKs.

The router removes all three. **Your code never names a model and never holds a
key.** It names *intent*, and one service resolves that to a concrete model,
holds the keys, and keeps a ledger of what actually worked.

## The core idea: you route by *intent*, not by *model*

You call it with a **profile** — a named intent — instead of a model:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="local")

reply = client.chat.completions.create(
    model="cheap",                       # <-- a PROFILE, never a model name
    messages=[{"role": "user", "content": "Summarize this log ..."}],
    extra_body={"metadata": {"routing_profile": "cheap"}},
)
```

A profile is **not** a model. It's a set of *constraints + an objective*, e.g.
"cheap = cheapest model that succeeds, ≥32k context, under $1/M tokens, no hidden
reasoning." Nothing in the config ever names a model — so when a new model shows
up, it becomes a candidate **automatically**, no code change.

## How it actually decides (the interesting part)

On every request the router:

1. **Pulls the live model catalog** from OpenRouter — the current list of models,
   already ranked two ways: by *intelligence* and by *latency*. (Cached, with an
   offline snapshot fallback.)
2. **Filters** to models that satisfy the profile's hard constraints (context
   size, price cap, needed capabilities like tool use or JSON schema, a quality
   floor expressed as a slice of the ranking — e.g. "top 25%").
3. **Prices the whole route**, not just one call — for multi-model routes it
   includes the helper models too (see route types below).
4. **Scores** each candidate by *expected cost per verified success*. The
   "probability it succeeds" starts from the model's rank and is then **updated
   by real recorded outcomes** — so the router literally learns which models work
   for which jobs over time.
5. **Enforces the budget.** If nothing fits the budget, it fails loudly rather
   than silently downgrading.
6. **Orders by the objective** — cheapest, fastest, or smartest — and sends the
   request, with ranked fallbacks if the first choice errors.

## Route types (how many models a profile uses)

- **simple** — one model. (cheap, fast, structured, frontier)
- **advisor** — a strong model does the work, a second model sanity-checks it.
  (code, verify)
- **fusion** — a *panel* of models from different labs answer, and a judge model
  synthesizes. (research, finance)
- **subagent** — a smart "boss" model plans and delegates the grunt work to a
  cheap "worker" model. (long_context, and cheap_code below)
- **direct / local / private** — talk straight to one provider (or a local
  model), bypassing the marketplace — used for privacy or a specific endpoint.

## The profiles it ships with

| Profile | What it's for |
|---|---|
| `cheap` | bulk work — summaries, extraction, log scanning |
| `fast` | latency-sensitive, short interactive calls |
| `structured` | guaranteed-JSON output (schema injected *and* validated) |
| `code` | coding — strong model + a senior-engineer advisor |
| `cheap_code` | coding on a budget: smart boss (Claude today) + **DeepSeek workers**, a light tier for routine subtasks and a heavier tier for hard ones |
| `research` / `finance` | multi-model panel + judge for synthesis |
| `long_context` | very large inputs, delegated to a cheap worker |
| `frontier` | just the single smartest model available right now |
| `verify` | high-scrutiny answers with external verification expected |
| `private` | local-only model, no external calls |
| `fast_celeris` | a diffusion model (Celeris-1) for ultra-low-latency short work |

Two recent additions worth calling out:

- **Two-tier cheap coding** — `cheap_code` uses a top-ranked model as the
  orchestrator and cheap **DeepSeek** models as the workhorses, automatically
  escalating from a light worker to the smartest available DeepSeek when the task
  is big or tool-heavy.
- **Supplemental catalog** — models that *aren't* on OpenRouter (like Celeris,
  reached via its own API) can be dropped into a config file and then **compete
  in the normal selection** on merit. In a live head-to-head, Celeris came out
  ~3.8× faster than the incumbent "fast" model on short tasks, so it now wins the
  fast lane automatically.

## What you get back, and what it records

- A normal OpenAI-style response — drop-in, no SDK changes.
- Every call is **logged with real cost and token accounting** and its route
  decision.
- Automatic quality checks (did the JSON validate? did the tool get called?) feed
  an **outcomes ledger**, which is the learning signal that reweights future
  routing. You can also POST your own verdicts (tests passed / human approved) to
  `/v1/outcomes` to teach it.

## Using it (the short version)

It runs as an always-on local service (a macOS LaunchAgent on the author's
machine) listening on `http://127.0.0.1:8765`. Endpoints:

- `POST /v1/chat/completions` — the main one (OpenAI-compatible)
- `POST /v1/routes/explain` — "why did you pick that?" — full decision + scores
- `GET  /v1/models` — the profiles, as if they were models
- `POST /v1/outcomes` — report whether a call actually succeeded
- `GET  /v1/telemetry/summary` — totals per model / per profile

The **one rule for callers**: name the profile in **both** the `model` field and
`metadata.routing_profile`. (Belt and suspenders — it prevents a caller's prompt
from being silently re-classified into an expensive route.)

## Honest caveats

- It's a **personal research tool**, wired to the author's machine, paths, and
  API keys — not a packaged product. You'd need your own OpenRouter (and
  optionally Anthropic/OpenAI/Celeris) keys and to run the service yourself.
- The marketplace layer leans on **OpenRouter** as the model source of truth.
- The "supplemental catalog" model competes in latency-sorted routes; wiring it
  into the cheapest-first routes needs a small extra tweak (noted in the code).

---

*Repo:* `Divecha Router` (a.k.a. Arjun Model Gateway) · Python 3.12 + FastAPI ·
one endpoint, many models, no hardcoded model names.
