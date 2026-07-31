# Camera real-history paid matrix

Run date: 2026-07-23

Status: **COMPLETE — NO POLICY PRODUCED A VERIFIED SUCCESS**

This is a valid negative result. All 18 frozen cells ran against the two real
historical Camera tasks, and every final candidate was exercised by the hidden
behavioral verifier plus TypeScript and the Next.js production build.

## Result

| Policy | Cells | Verified | Success rate | Cost | Provider latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kimi K3 solo | 6 | 0 | 0% | $0.8395746 | 1,337.684 s |
| Fable 5 solo | 6 | 0 | 0% | $6.7400400 | 749.585 s |
| Kimi then Fable on failure | 6 | 0 | 0% | $6.0261114 | 1,785.658 s |
| **Total** | **18** | **0** | **0%** | **$13.6057260** | **3,872.927 s** |

Cost per verified success and latency per verified success are undefined
because every policy had zero verified successes.

## Task breakdown

| Task | Policy | Verified | Cost | Provider latency | Calls |
| --- | --- | ---: | ---: | ---: | ---: |
| Dock launcher + iOS | Kimi solo | 0/3 | $0.3095178 | 534.724 s | 20 |
| Dock launcher + iOS | Fable solo | 0/3 | $2.6198400 | 367.791 s | 24 |
| Dock launcher + iOS | Kimi→Fable | 0/3 | $2.8749358 | 917.126 s | 53 |
| iOS clip playback | Kimi solo | 0/3 | $0.5300568 | 807.485 s | 29 |
| iOS clip playback | Fable solo | 0/3 | $4.1202000 | 381.795 s | 29 |
| iOS clip playback | Kimi→Fable | 0/3 | $3.1511756 | 874.066 s | 48 |

## What failed

- All nine dock/iOS candidates missed the same five behavioral contracts:
  launcher reuse with an existing profile, focus without duplication,
  configured-host matching, dynamic mobile viewport units, and the iOS safe
  area.
- All nine playback candidates missed the failed-HLS fallback behavior.
- The Kimi→Fable playback policy was closest: all three final candidates
  failed only `missing_event_shows_fallback`.
- TypeScript and production build checks passed for every final candidate.
  Compilation was therefore not treated as behavioral success.

## Integrity evidence

- Machine report: `reports/camera_paid_v3.json`
- Artifact root: `reports/camera_paid_v3_20260723T194049Z`
- Provider journal: `reports/camera_paid_v3_20260723T194049Z/provider-journal.jsonl`
- Provider responses: `203`, all with unique response IDs
- Provider errors: `0`
- Interrupted-attempt cost: `$0`
- Completed-cell cost: `$13.605726`
- Journal cost: `$13.605726`
- Candidate workspaces and SHA-256 manifests: `18`
- Workspace diffs: `18`
- Stage transcripts: `24`
- Preserved Playwright browser log files: `76`
- Protocol SHA-256:
  `6e4a4082f9e86ded4018a59d12d4ed4ffae3ccc9c3f1264729678f1131337147`
- Common harness SHA-256:
  `f09964f722891b22409de7901c1e3855bd4b6c7ffcb912041d4b02e70fb18693`
- Live runner SHA-256:
  `55343492122bc26bb50d1891d0304486ce33051947e0d2e819a06bde3b68c824`

The earlier unjournaled partial run is preserved as
`reports/camera_paid_v2_invalidated_unjournaled_20260723.json` but is excluded
from every result above. It had `$3.6585022` of recorded completed-cell spend
plus an unrecoverable amount from the unfinished cell. The still-earlier
invalid pilot recorded `$6.0681`; it is also excluded. These are sunk testing
costs, not part of the clean `$13.605726` comparison.

## Router conclusion

Do not promote any of these three policies as a verified-success route from
this sample. Kimi is the cheapest failure mode. Fable is faster in provider
latency but roughly eight times Kimi's total cost. Escalating Kimi to Fable
improved the playback candidates but did not cross the behavioral success
threshold and added substantial latency.
