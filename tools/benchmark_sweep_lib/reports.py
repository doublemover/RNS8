from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import SweepCommand

def scenario_manifest(entries: list[SweepCommand], args: argparse.Namespace) -> dict[str, Any]:
    scenario_entries = [entry for entry in entries if entry.scenario is not None]
    families = sorted({str(entry.scenario["family"]) for entry in scenario_entries if entry.scenario})
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_request": list(getattr(args, "scenario", []) or []),
        "scenario_families": families,
        "capture_count": len(scenario_entries),
        "review_mode": args.review_mode,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "seed": args.seed,
        "entries": [
            {
                **(entry.scenario or {}),
                "capture_name": entry.name,
                "capture_path": str(entry.output),
                "command": entry.command,
            }
            for entry in scenario_entries
        ],
    }


def write_scenario_manifest(entries: list[SweepCommand], args: argparse.Namespace, out_root: Path) -> dict[str, str] | None:
    if not any(entry.scenario is not None for entry in entries):
        return None
    manifest = scenario_manifest(entries, args)
    json_path = out_root / "scenario_manifest.json"
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# RNS8 Scenario Benchmark Manifest",
        "",
        f"- schema_version: `{manifest['schema_version']}`",
        f"- generated_utc: `{manifest['generated_utc']}`",
        f"- scenario_request: `{','.join(manifest['scenario_request'])}`",
        f"- scenario_families: `{','.join(manifest['scenario_families'])}`",
        f"- captures: `{manifest['capture_count']}`",
        f"- review_mode: `{manifest['review_mode']}`",
        f"- warmups: `{manifest['warmups']}`",
        f"- repeats: `{manifest['repeats']}`",
        f"- seed: `{manifest['seed']}`",
        "",
        "| family | item | semantics | shape | backend | pack | review | promotion | output_domain | evidence_scope | capture |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in manifest["entries"]:
        shape = entry["shape"]
        lines.append(
            "| {family} | {name} | {semantics} | {m}x{n}x{k} | {backend} | {pack} | {review} | {promotion} | {domain} | {scope} | {capture} |".format(
                family=entry["family"],
                name=entry["name"],
                semantics=entry["semantics"],
                m=shape["m"],
                n=shape["n"],
                k=shape["k"],
                backend=entry["backend"],
                pack=entry["pack_mode"],
                review=entry["review_mode_expectation"],
                promotion=entry["promotion_eligibility"],
                domain=entry["output_domain"],
                scope=entry["evidence_scope"],
                capture=entry["capture_name"],
            )
        )
    markdown_path = out_root / "scenario_manifest.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"scenario_manifest": str(json_path), "scenario_markdown": str(markdown_path)}


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# RNS8 Benchmark Sweep Review",
        "",
        f"- schema_version: `{report.get('schema_version')}`",
        f"- review_mode: `{report.get('review_mode')}`",
        f"- reviewed_utc: `{report.get('reviewed_utc')}`",
        f"- groups: `{report.get('group_count')}`",
        f"- promotable_autotune_entries: `{len(report.get('promotable_autotune_entries', []))}`",
        f"- cache_write: `{report.get('cache_write', {}).get('status')}`",
        "",
    ]
    for group in report.get("groups", []):
        shape = group.get("shape", {})
        modulus = group.get("finite_modulus")
        title = f"{group.get('semantics')} {shape.get('m')}x{shape.get('n')}x{shape.get('k')}"
        if modulus is not None:
            title += f" mod {modulus}"
        lines.extend([f"## {title}", ""])
        missing = group.get("missing_required_baselines") or []
        missing_targets = group.get("missing_gpu_targets") or []
        lines.append(f"- missing_required_baselines: `{','.join(missing) if missing else 'none'}`")
        lines.append(f"- missing_gpu_targets: `{','.join(missing_targets) if missing_targets else 'none'}`")
        lines.append(f"- gpu_target_compatible: `{group.get('gpu_target_compatible')}`")
        missing_configured = group.get("missing_configured_gpu_targets") or []
        lines.append(
            f"- missing_configured_gpu_targets: `{','.join(missing_configured) if missing_configured else 'none'}`"
        )
        lines.append(f"- configured_target_compatible: `{group.get('configured_target_compatible')}`")
        missing_versions = group.get("missing_hip_toolchain_versions") or []
        lines.append(
            f"- missing_hip_toolchain_versions: `{','.join(missing_versions) if missing_versions else 'none'}`"
        )
        lines.append(f"- hip_toolchain_version_compatible: `{group.get('hip_toolchain_version_compatible')}`")
        missing_runtime = group.get("missing_hip_runtime_versions") or []
        lines.append(
            f"- missing_hip_runtime_versions: `{','.join(missing_runtime) if missing_runtime else 'none'}`"
        )
        lines.append(f"- hip_runtime_version_compatible: `{group.get('hip_runtime_version_compatible')}`")
        missing_driver = group.get("missing_hip_driver_versions") or []
        lines.append(f"- missing_hip_driver_versions: `{','.join(missing_driver) if missing_driver else 'none'}`")
        lines.append(f"- hip_driver_version_compatible: `{group.get('hip_driver_version_compatible')}`")
        missing_compilers = group.get("missing_compiler_identities") or []
        lines.append(f"- missing_compiler_identities: `{','.join(missing_compilers) if missing_compilers else 'none'}`")
        lines.append(f"- compiler_identity_compatible: `{group.get('compiler_identity_compatible')}`")
        missing_git = group.get("missing_git_commits") or []
        lines.append(f"- missing_git_commits: `{','.join(missing_git) if missing_git else 'none'}`")
        lines.append(f"- git_commit_identity_compatible: `{group.get('git_commit_identity_compatible')}`")
        missing_warmups = group.get("missing_warmup_counts") or []
        lines.append(f"- missing_warmup_counts: `{','.join(missing_warmups) if missing_warmups else 'none'}`")
        lines.append(f"- warmup_count_compatible: `{group.get('warmup_count_compatible')}`")
        missing_repeats = group.get("missing_repeat_counts") or []
        lines.append(f"- missing_repeat_counts: `{','.join(missing_repeats) if missing_repeats else 'none'}`")
        lines.append(f"- repeat_count_compatible: `{group.get('repeat_count_compatible')}`")
        duplicates = group.get("duplicate_backends") or []
        lines.append(f"- duplicate_backends: `{','.join(duplicates) if duplicates else 'none'}`")
        lines.append(f"- checksum_reference_backend: `{group.get('checksum_reference_backend') or 'none'}`")
        lines.append(f"- checksum_consistent: `{group.get('checksum_consistent')}`")
        checksum_mismatches = group.get("checksum_mismatches") or []
        lines.append(
            f"- checksum_mismatches: `{','.join(checksum_mismatches) if checksum_mismatches else 'none'}`"
        )
        scenario_scopes = group.get("scenario_promotion_scopes") or []
        lines.append(
            f"- scenario_promotion_scopes: `{','.join(scenario_scopes) if scenario_scopes else 'none'}`"
        )
        lines.append(f"- release_review_satisfied: `{group.get('release_review_satisfied')}`")
        fastest = group.get("fastest_promotable")
        if fastest:
            lines.append(f"- fastest_promotable: `{fastest['backend']}/{fastest['selected_kernel']}`")
            lines.append(f"- winner_rationale: `{fastest.get('promotion_reason')}`")
        else:
            lines.append("- fastest_promotable: `none`")
        lines.append("")
        lines.append(
            "| backend | kernel | target | e2e median us | bottleneck | promotable | cache | blockers | primary loss phase |"
        )
        lines.append("|---|---|---|---:|---|---|---|---|---|")
        for candidate in group.get("candidates", []):
            blockers = ",".join(candidate.get("promotion_blockers") or [])
            source = candidate.get("source_metadata") if isinstance(candidate.get("source_metadata"), dict) else {}
            bottleneck = candidate.get("bottleneck") if isinstance(candidate.get("bottleneck"), dict) else {}
            bottleneck_text = bottleneck.get("class") or "unknown"
            if bottleneck.get("phase"):
                bottleneck_text += f"/{bottleneck.get('phase')}"
            lines.append(
                "| {backend} | {kernel} | {target} | {median} | {bottleneck} | {promotable} | {cache} | {blockers} | {loss} |".format(
                    backend=candidate.get("backend"),
                    kernel=candidate.get("selected_kernel"),
                    target=source.get("target_id"),
                    median=candidate.get("median_end_to_end_us"),
                    bottleneck=bottleneck_text,
                    promotable=candidate.get("promotable"),
                    cache=candidate.get("cache_write_status"),
                    blockers=blockers or "none",
                    loss=candidate.get("primary_loss_phase_vs_direct_hip") or "",
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


