# Camera replay v2 preflight

Run date: 2026-07-23

Status: **PASSED**

- Paid provider calls: `0`
- Measured provider cost: `$0`
- Protocol SHA-256:
  `6e4a4082f9e86ded4018a59d12d4ed4ffae3ccc9c3f1264729678f1131337147`
- Runner SHA-256:
  `f09964f722891b22409de7901c1e3855bd4b6c7ffcb912041d4b02e70fb18693`
- Full report: `reports/camera_preflight_v2.json`
- Preserved artifacts:
  `reports/camera_preflight_v2_20260723T190514Z`

Provider-free controls:

- Common scripted agent loop: passed with five tool calls and zero tool errors.
- Path escape rejection: passed.
- Network-shaped validation rejection: passed.
- Historical dock/iOS base failed and gold passed.
- Historical playback base failed and gold passed.
- Scripted dock/iOS repair passed TypeScript, production build, launcher
  scenarios, mobile CSS contract, and iPhone WebKit verification.
- Scripted playback repair passed TypeScript, production build, working-HLS
  playback, failed-HLS thumbnail fallback, and iPhone WebKit verification.
- Both complete tool transcripts were preserved.
- Both final candidate source workspaces and SHA-256 manifests were preserved.

The original paid pilot remains invalidated and is not routing evidence.
Protocol v2 was subsequently enabled without changing its tools, limits,
prompts, or verifier. The completed clean paid run is summarized in
`PAID_V3_RESULTS.md`.
