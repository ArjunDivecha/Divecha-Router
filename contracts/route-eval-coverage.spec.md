---
divecha: 3
id: ROUTE-EVAL-COVERAGE-001
status: green
objective: 'add routing-robustness eval coverage for the two auto-routable profiles
  that currently have none: long_context and fast'
scope:
  write:
  - evals/**
  - tests/**
  - contracts/route-eval-coverage.spec.md
  forbid:
  - src/**
  - config/**
  - .env
  - '*.key'
  - '*.pem'
context:
  repo: /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router
  read_first:
  - evals/routing_robustness.yaml
  - src/arjun_model_gateway/routing_eval.py
  - src/arjun_model_gateway/policy.py
  - tests/test_routing_eval.py
  facts:
  - 'run everything through the project venv: .venv/bin/python and .venv/bin/amg'
  - 'the auto-classifier can only produce these profiles: finance, research, code,
    structured, long_context, cheap, fast (policy.py resolve_profile_name)'
  - "evals/routing_robustness.yaml currently covers finance, code, research x2, structured,\
    \ cheap \u2014 long_context and fast have zero coverage"
  - long_context is reached ONLY via metadata.context_length_estimate > 100000, never
    from prompt text (policy.py classify_request)
  - 'fast is the catch-all: a prompt of >=600 chars that matches no intent family
    and needs no tools/json (policy.py resolve_profile_name else-branch)'
  - cheap is the <600-char generic path, so a fast case must stay long AND keyword-free
  - RoutingCase already supports a per-case `metadata` dict that is passed to the
    router (routing_eval.py RoutingCase.metadata)
  - 'each existing case carries exactly five variants, one per perturbation kind:
    character, word, syntax, paraphrase, rambling'
  - 'src/** and config/** are forbidden: this is a coverage change, not a classifier
    or policy change'
  - 'baseline before this change: 46 tests pass; robustness 1.0 and expected_profile_accuracy
    1.0 over 6 cases / 30 variants'
interfaces:
- kind: file
  signature: "evals/routing_robustness.yaml \u2014 cases[]: {id, expected_profile,\
    \ prompt, metadata?, variants[]: {kind, prompt}}"
  location: evals/routing_robustness.yaml
  notes: extend the existing cases list; do not restructure the schema
- kind: cli
  signature: amg eval routing-robustness --minimum 0.95 -> exit 0 when robustness
    and expected-profile accuracy both hold
  location: src/arjun_model_gateway/cli.py (read-only here)
behaviors:
- id: B1
  given: the expanded corpus
  when: the offline routing-robustness gate runs at the 0.95 minimum
  then: "it still exits 0 \u2014 every added prompt and all of its variants route\
    \ to the case's expected_profile, so coverage grew without dropping robustness\
    \ or expected-profile accuracy"
  examples:
  - in: .venv/bin/amg eval routing-robustness --minimum 0.95
    out: exit 0; cases >= 8, variants >= 40, robustness 1.0, expected_profile_accuracy
      1.0
  check: .venv/bin/amg eval routing-robustness --minimum 0.95
- id: B2
  given: the two previously-uncovered auto-routable profiles
  when: the corpus is inspected for coverage and perturbation completeness
  then: a long_context case and a fast case both exist, each carrying all five perturbation
    kinds, and the long_context case drives its route from metadata.context_length_estimate
    > 100000 rather than from keywords in its prompt
  examples:
  - in: 'case expected_profile=long_context, prompt=''Summarize the attached document.'',
      metadata={context_length_estimate: 150000}'
    out: routes to long_context; kinds present = {character, word, syntax, paraphrase,
      rambling}
  - in: case expected_profile=fast, prompt=an everyday >=600-char request with no
      coding/finance/research/json/tool keywords
    out: routes to fast (not cheap, not research); kinds present = {character, word,
      syntax, paraphrase, rambling}
  check: .venv/bin/python -m pytest tests/test_eval_coverage.py -q
- id: B3
  when: the full existing test suite runs
  then: "no regression \u2014 every previously passing test still passes"
  examples:
  - in: .venv/bin/python -m pytest -q
    out: exit 0, >= 46 tests passed
  check: .venv/bin/python -m pytest -q
loop:
  max_turns: 6
  max_consecutive_failures: 3
ledger:
- at: '2026-07-20T07:09:01.924049+00:00'
  turn: 1
  failing:
  - B2
  note: B2 rc=4
- at: '2026-07-20T07:11:43.855938+00:00'
  turn: 1
  failing: []
  note: green
- at: '2026-07-20T07:16:02.551725+00:00'
  turn: 1
  failing: []
  note: green
---

Anti-overfit note for the builder: the point is genuine surface-form diversity, not
variants engineered to pass. Keep each variant semantically equivalent to its case
prompt and clearly distinct in surface form. If a legitimate paraphrase cannot route
correctly, that is a real classifier finding — record it as a ledger agent_note
instead of weakening the variant or editing forbidden src/config paths.
