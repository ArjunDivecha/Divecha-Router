# INVALIDATED — Camera K3/Fable pilot results

Do not use this run for routing decisions. Paid execution began before the
provider-free tool-loop controls passed, the harness rejected legitimate agent
operations, scoring changed after outputs were observed, and candidate
workspaces were not preserved. The table below is retained only as an audit
record of the invalid run.

Run date: 2026-07-23

Models:

- `moonshotai/kimi-k3`
- `anthropic/claude-fable-5`

The pilot replayed two real Camera repository regressions against three
consumption policies, one repetition per task-policy cell. Every cell used the
same sanitized historical capsule, offline tool sandbox, 16-turn stage limit,
per-stage dollar cap, and hidden iPhone WebKit verifier.

| Task | Policy | Verified | Cost | Latency |
|---|---|---:|---:|---:|
| Dock launcher and iOS detections | K3 only | no | $0.1663 | 250.6s |
| Dock launcher and iOS detections | Fable only | no | $1.2239 | 157.0s |
| Dock launcher and iOS detections | K3 then Fable on failure | no | $0.6708 | 366.2s |
| iOS clip playback | K3 only | no | $0.1888 | 372.4s |
| iOS clip playback | Fable only | no | $0.7409 | 99.3s |
| iOS clip playback | K3 then Fable on failure | yes | $3.0775 | 670.4s |

Total measured model cost was $6.0681. The combination policy produced the
only verified success, but it was also the most expensive and slowest cell.
K3 was materially cheaper than Fable in solo runs, but neither solo model
completed either task.

One verifier correction was made after the paid run: the playback verifier had
required the historical component name `VodPlayer` even though the combination
implementation demonstrably requested HLS, avoided direct MP4, and passed the
remaining fallback checks. The component-name check was removed, the historical
base/gold calibration was rerun successfully, and the saved behavioral evidence
was rescored without another model call.

No model or policy conclusion may be drawn from this run.
