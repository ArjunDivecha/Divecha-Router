"""
=============================================================================
MODULE: cli.py (Divecha Router v2)
=============================================================================

DESCRIPTION:
Typer CLI (`amg`) for operating the gateway:

  amg init                      create data/report directories
  amg serve --load-env          run the FastAPI gateway on localhost
  amg intel refresh             force-refresh the live OpenRouter catalog
  amg intel choose PROFILE      show which model the profile picks right now
  amg intel scorecard PROFILE   ranked candidates with costs and posteriors
  amg eval routing-robustness   gate route stability across prompt variants
  amg eval diversity            select a quality/cost-gated actor coreset
  amg routes test [--mock]      end-to-end smoke through the gateway
  amg outcomes report ...       record a manual verification outcome
  amg telemetry summary|report  usage/cost/outcome reporting
  amg registry refresh|validate raw OpenRouter registry snapshot
  amg policy validate           validate config/policy.yaml
  amg scanner scan              find model call sites in a repo

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/config/policy.yaml
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/data/telemetry.sqlite
- /Users/arjundivecha/Dropbox/AAA Backup/.env.txt (only with --load-env)

OUTPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/data/catalog_snapshot.json
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/data/registry.json
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/reports/telemetry_summary.md
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Divecha Router/reports/repo_scan.json
=============================================================================
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn

from .adapters import AdapterError
from .catalog import Catalog
from .config import DEFAULT_REGISTRY_PATH, DEFAULT_TELEMETRY_PATH, ensure_dirs, validate_policy
from .env_loader import load_provider_keys
from .gateway import handle_chat
from .models import ChatCompletionRequest, ChatMessage, GatewayMetadata
from .policy import manual_seeds
from .registry import refresh_openrouter_registry, validate_registry
from .routing_eval import (
    build_diversity_report,
    evaluate_routing_robustness,
    load_behavior_matrix,
    load_routing_suite,
    write_versioned_report,
)
from .scanner import generate_eval_candidates, scan
from .selector import RoutingError, select
from .telemetry import (
    latency_observations,
    record_outcome,
    success_counts,
    summarize,
    write_summary_report,
)

app = typer.Typer(help="Arjun Model Gateway CLI")
registry_app = typer.Typer(help="Registry commands")
policy_app = typer.Typer(help="Policy commands")
scanner_app = typer.Typer(help="Repo scanner commands")
routes_app = typer.Typer(help="Route smoke-test commands")
telemetry_app = typer.Typer(help="Telemetry reporting commands")
intel_app = typer.Typer(help="Live model intelligence commands")
outcomes_app = typer.Typer(help="Verification outcome commands")
eval_app = typer.Typer(help="Offline routing evaluation gates")
app.add_typer(registry_app, name="registry")
app.add_typer(policy_app, name="policy")
app.add_typer(scanner_app, name="scanner")
app.add_typer(routes_app, name="routes")
app.add_typer(telemetry_app, name="telemetry")
app.add_typer(intel_app, name="intel")
app.add_typer(outcomes_app, name="outcomes")
app.add_typer(eval_app, name="eval")


@app.command()
def init() -> None:
    ensure_dirs()
    typer.echo(f"initialized data dirs; telemetry={DEFAULT_TELEMETRY_PATH}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8765, load_env: bool = False) -> None:
    if load_env:
        loaded = load_provider_keys()
        typer.echo(f"loaded provider env keys: {', '.join(loaded) if loaded else 'none'}")
    uvicorn.run("arjun_model_gateway.gateway:app", host=host, port=port, reload=False)


@registry_app.command("refresh")
def registry_refresh(output: Path = DEFAULT_REGISTRY_PATH) -> None:
    snapshot = refresh_openrouter_registry(output)
    typer.echo(f"registry refreshed: {len(snapshot.models)} models -> {output}")


@registry_app.command("validate")
def registry_validate(path: Path = DEFAULT_REGISTRY_PATH) -> None:
    snapshot = validate_registry(path)
    typer.echo(f"registry valid: {len(snapshot.models)} models, version={snapshot.version}")


@policy_app.command("validate")
def policy_validate() -> None:
    policy = validate_policy()
    typer.echo(f"policy valid: {len(policy.profiles)} profiles, version={policy.version}")


@scanner_app.command("scan")
def scanner_scan(root: Path = Path("."), output: Path = Path("reports/repo_scan.json")) -> None:
    report = scan(root.resolve(), output)
    evals = generate_eval_candidates(report)
    typer.echo(
        json.dumps(
            {
                "model_call_sites": len(report["model_call_sites"]),
                "hardcoded_models": len(report["hardcoded_model_report"]),
                "eval_candidates": evals["count"],
                "output": str(output),
            },
            indent=2,
        )
    )


@routes_app.command("test")
def routes_test(mock: bool = False, load_env: bool = False, profile: str = "cheap") -> None:
    loaded = load_provider_keys() if load_env else []
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="Return a short smoke response.")],
        metadata=GatewayMetadata(
            repo="Divecha Router",
            script="amg routes test",
            call_site_id="cli.routes_test",
            routing_profile=profile,
        ),
    )
    try:
        raw = handle_chat(request, force_mock=mock)
    except (AdapterError, RoutingError) as exc:
        detail = (
            {"provider": exc.provider, "error_type": exc.error_type}
            if isinstance(exc, AdapterError)
            else {"code": exc.code, "message": str(exc)}
        )
        typer.echo(json.dumps({"ok": False, **detail, "loaded_env_keys": loaded}, indent=2), err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        json.dumps(
            {
                "ok": True,
                "response_id": raw.get("id"),
                "model": raw.get("model"),
                "mock": mock,
                "loaded_env_keys": loaded,
            },
            indent=2,
        )
    )


@telemetry_app.command("summary")
def telemetry_summary(path: Path = DEFAULT_TELEMETRY_PATH) -> None:
    typer.echo(json.dumps(summarize(path), indent=2, default=str))


@telemetry_app.command("report")
def telemetry_report(
    output: Path = Path("reports/telemetry_summary.md"),
    path: Path = DEFAULT_TELEMETRY_PATH,
) -> None:
    summary = write_summary_report(output, path)
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "calls": summary["totals"]["calls"],
                "total_tokens": summary["totals"]["total_tokens"],
                "total_cost_usd": summary["totals"]["total_cost_usd"],
            },
            indent=2,
            default=str,
        )
    )


@outcomes_app.command("report")
def outcomes_report(
    request_id: str,
    success: bool = typer.Option(..., help="Did the output verify as correct?"),
    verifier: str = "human",
    detail: str = "",
) -> None:
    record_outcome(request_id, success=success, verifier=verifier, detail=detail or None)
    typer.echo(json.dumps({"ok": True, "request_id": request_id, "success": success}, indent=2))


@eval_app.command("routing-robustness")
def eval_routing_robustness(
    cases: Path = Path("evals/routing_robustness.yaml"),
    output: Path = Path("reports/routing_robustness.json"),
    minimum: float | None = None,
) -> None:
    """Fail when semantic prompt variants change or miss the intended route."""
    suite = load_routing_suite(cases)
    report = evaluate_routing_robustness(suite, validate_policy(), minimum=minimum)
    outputs = write_versioned_report(report, output)
    typer.echo(
        json.dumps(
            {
                "passed": report["passed"],
                "route_robustness": report["route_robustness"],
                "expected_profile_accuracy": report["expected_profile_accuracy"],
                "cases": report["cases"],
                "variants": report["variants"],
                "outputs": [str(path) for path in outputs],
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise typer.Exit(1)


@eval_app.command("diversity")
def eval_diversity(
    matrix: Path = typer.Option(..., "--matrix", help="Verified per-workload actor scores"),
    output: Path = Path("reports/route_diversity.json"),
    max_actors: int = 8,
    min_mean_score: float = 0.0,
    max_cost_per_verified_success_usd: float | None = None,
    minimum_joint_gain: float = 0.001,
) -> None:
    """Measure behavioral HSE and select a small quality/cost-gated pool."""
    report = build_diversity_report(
        load_behavior_matrix(matrix),
        max_actors=max_actors,
        min_mean_score=min_mean_score,
        max_cost_per_verified_success_usd=max_cost_per_verified_success_usd,
        minimum_joint_gain=minimum_joint_gain,
    )
    outputs = write_versioned_report(report, output)
    typer.echo(
        json.dumps(
            {
                "actor_count": report["actor_count"],
                "eligible_actor_count": report["eligible_actor_count"],
                "coreset": report["coreset"],
                "outputs": [str(path) for path in outputs],
            },
            indent=2,
        )
    )


def _selection_for(profile_name: str, limit: int):
    policy = validate_policy()
    if profile_name not in policy.profiles:
        raise typer.BadParameter(f"unknown profile {profile_name!r}")
    profile = policy.profiles[profile_name]
    if profile.route_type in ("local", "direct"):
        from .provider_models import resolve_direct_model

        return None, (profile, resolve_direct_model(profile.provider))
    counts, global_counts = success_counts()
    selection = select(
        profile_name,
        profile,
        Catalog(),
        counts=counts,
        global_counts=global_counts,
        manual_seeds=manual_seeds(profile_name),
        latency_obs=latency_observations(),
    )
    selection.candidates = selection.candidates[:limit]
    return selection, profile


@intel_app.command("refresh")
def intel_refresh() -> None:
    catalog = Catalog()
    models = catalog.models(refresh=True)
    typer.echo(
        json.dumps(
            {
                "models": len(models),
                "source": catalog.source,
                "as_of": catalog.as_of,
                "snapshot": str(catalog.snapshot_path),
                "smartest_right_now": [m["id"] for m in models[:5]],
            },
            indent=2,
        )
    )


@intel_app.command("choose")
def intel_choose(profile: str = "fast", load_env: bool = False) -> None:
    if load_env:
        load_provider_keys()
    try:
        selection, profile_obj = _selection_for(profile, limit=3)
    except RoutingError as exc:
        typer.echo(json.dumps({"ok": False, "code": exc.code, "message": str(exc)}, indent=2), err=True)
        raise typer.Exit(1) from exc
    if selection is None:
        _, resolved = profile_obj
        typer.echo(
            json.dumps(
                {"profile": profile, "model": resolved, "source": "provider_models_api"}, indent=2
            )
        )
        return
    primary = selection.primary
    typer.echo(
        json.dumps(
            {
                "profile": profile,
                "objective": selection.objective,
                "quality_floor": selection.quality_floor,
                "catalog": f"{selection.catalog_source}:{selection.catalog_as_of}",
                "selected_model": primary.model_id,
                "intelligence_rank": primary.intelligence_rank,
                "estimated_route_cost_usd": round(primary.route_cost_usd, 6),
                "p_success": round(primary.p_success, 4),
                "verified": f"{primary.successes}s/{primary.failures}f",
                "route_graph": [
                    {"role": n.role, "model": n.model_id, "cost_usd": round(n.cost_usd, 6)}
                    for n in primary.route_graph
                ],
                "fallbacks": [c.model_id for c in selection.candidates[1:]],
            },
            indent=2,
        )
    )


@intel_app.command("scorecard")
def intel_scorecard(profile: str = "code", limit: int = 10, load_env: bool = False) -> None:
    if load_env:
        load_provider_keys()
    try:
        selection, profile_obj = _selection_for(profile, limit=limit)
    except RoutingError as exc:
        typer.echo(json.dumps({"ok": False, "code": exc.code, "message": str(exc)}, indent=2), err=True)
        raise typer.Exit(1) from exc
    if selection is None:
        _, resolved = profile_obj
        typer.echo(
            json.dumps(
                {"profile": profile, "model": resolved, "source": "provider_models_api"}, indent=2
            )
        )
        return
    typer.echo(
        json.dumps(
            {
                "profile": profile,
                "objective": selection.objective,
                "quality_floor": selection.quality_floor,
                "pool_considered": selection.pool_considered,
                "catalog": f"{selection.catalog_source}:{selection.catalog_as_of}",
                "ranked": [
                    {
                        "rank": idx + 1,
                        "model": c.model_id,
                        "intelligence_rank": c.intelligence_rank,
                        "latency_rank": c.latency_rank,
                        "estimated_route_cost_usd": round(c.route_cost_usd, 6),
                        "p_success": round(c.p_success, 4),
                        "verified": f"{c.successes}s/{c.failures}f",
                        "score": round(c.score, 6),
                    }
                    for idx, c in enumerate(selection.candidates)
                ],
            },
            indent=2,
        )
    )
