"""
=============================================================================
SCRIPT NAME: test_eval_coverage.py
=============================================================================

DESCRIPTION:
Verifies that the routing-robustness eval corpus
(evals/routing_robustness.yaml) has coverage for the two auto-routable
profiles that previously had none: long_context and fast. For each of
those two profiles, checks that a case exists whose variants carry all
five perturbation kinds (character, word, syntax, paraphrase, rambling),
and that the long_context case routes via metadata.context_length_estimate
rather than via prompt keywords.

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/evals/routing_robustness.yaml

OUTPUT FILES:
- none (this is a pytest test module; it performs no file writes)

VERSION: 1.0
LAST UPDATED: 2026-07-20

DEPENDENCIES:
- pytest
- arjun_model_gateway.routing_eval

USAGE:
.venv/bin/python -m pytest tests/test_eval_coverage.py -q
=============================================================================
"""

from __future__ import annotations

from pathlib import Path

from arjun_model_gateway.routing_eval import load_routing_suite

ROOT = Path(__file__).parents[1]
PERTURBATION_KINDS = {"character", "word", "syntax", "paraphrase", "rambling"}


def _case_for(suite, expected_profile: str):
    matches = [case for case in suite.cases if case.expected_profile == expected_profile]
    assert matches, f"no case with expected_profile={expected_profile!r}"
    return matches[0]


def test_long_context_case_covers_all_perturbation_kinds_and_routes_from_metadata():
    suite = load_routing_suite(ROOT / "evals/routing_robustness.yaml")
    case = _case_for(suite, "long_context")
    assert {variant.kind for variant in case.variants} == PERTURBATION_KINDS
    assert case.metadata.get("context_length_estimate", 0) > 100_000


def test_fast_case_covers_all_perturbation_kinds():
    suite = load_routing_suite(ROOT / "evals/routing_robustness.yaml")
    case = _case_for(suite, "fast")
    assert {variant.kind for variant in case.variants} == PERTURBATION_KINDS
