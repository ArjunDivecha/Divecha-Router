# Camera real-history replay suite

This suite converts one private, real Claude Code session and the resulting
Camera repository history into sanitized counterfactual replay tasks.

The task prompts contain no camera addresses, credentials, footage, private
hostnames, or access tokens. The private source records are provenance only:
the replay runner must never include them in a model prompt.

## Current boundary

`amg eval replay-plan` validates task provenance, git ancestry, offline
verifier commands, policy structure, and leakage controls. It does not spend
money or execute agent models.

Most tasks still require deterministic fixtures before paid comparison runs:

- a synthetic RTSP/NVR service;
- a mocked Frigate-compatible API;
- browser checks using desktop Chromium and iOS-sized WebKit;
- launcher and camera-clock simulators.

The build and lint commands are necessary but not sufficient proof of the
user-visible behavior.

## Intended evaluation

Every policy receives the same sealed checkout, prompt, fixtures, tool
allowlist, wall-clock limit, and dollar budget. Results are scored by hidden
behavioral checks plus total cost, tokens, latency, retries, and model
consumptions.

The `parallel_oracle` policy establishes the cheapest-correct ceiling. The
other policies are compared by quality regret, cost regret, and latency regret
against that ceiling.

## Pilot golden proof

The first executable pilot uses two narrow, user-reported regressions:

- `camera_dock_and_ios_regression`
- `camera_ios_clip_playback`

Run the sealed offline fixture and iPhone WebKit verifier:

```bash
.venv/bin/python -m arjun_model_gateway.camera_replay prove-goldens
```

Each historical base must fail its hidden checks and each corresponding gold
commit must pass. The capsule contains only allowlisted application files and
is rejected if private addresses, tailnet names, or credential-like material
are found.

## Six-cell live pilot

After the golden proof passes, run two tasks against three consumption
policies (K3 only, Fable only, and K3 then Fable on hidden-check failure):

```bash
.venv/bin/python -m arjun_model_gateway.replay_agent run-pilot
```

The common harness gives both model families the same local file tools, offline
command allowlist, 16-turn limit, per-stage cost cap, and sealed verifier.
It writes machine-readable and Markdown reports under `reports/`.

## V2 provider-free preflight

The original six-cell paid run is invalidated and must not inform routing. The
replacement harness does not expose paid execution until this command passes:

```bash
.venv/bin/python -m arjun_model_gateway.replay_harness_v2 preflight \
  --output reports/camera_preflight_v2.json
```

The v2 preflight runs zero provider calls. It proves the common tool loop,
negative safety controls, historical base/gold discrimination, scripted repair
reachability, TypeScript/build validation, full transcript capture, and final
candidate preservation under a hashed protocol.

## Paid matrix

After the exact paid gate passes, run the frozen matrix:

```bash
.venv/bin/python -m arjun_model_gateway.replay_live_v2 gate
.venv/bin/python -m arjun_model_gateway.replay_live_v2 run \
  --output reports/camera_paid_v3.json
```

The clean 2026-07-23 run completed all 18 cells with zero provider errors and
zero verified successes. See `PAID_V3_RESULTS.md` for the result and integrity
evidence. The append-only provider journal is authoritative for cost even
between cell-level report checkpoints.
