#!/usr/bin/env python3
"""Run the promotion-ledger and cache-install gates as one closeout step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import install_autotune_cache
import promotion_ledger


DEFAULT_OUT_DIR = Path("temp") / "cache-promotion-closeouts"


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Cache Promotion Closeout",
        "",
        f"- Complete: `{summary['complete']}`",
        f"- Dry run: `{summary['dry_run']}`",
        f"- Sources: `{summary['source_count']}`",
        f"- Captures: `{summary['capture_count']}`",
        f"- Ledger blocked rows: `{summary['ledger_blocked_count']}`",
        f"- Install gate: `{summary['install_status']}`",
        "",
        "## Blockers",
        "",
    ]
    if summary["blockers"]:
        lines.extend(f"- `{blocker}`" for blocker in summary["blockers"])
    else:
        lines.append("- none")
    install_summary = summary.get("install_summary")
    if isinstance(install_summary, dict):
        lines.extend(
            [
                "",
                "## Install Summary",
                "",
                f"- Destination: `{install_summary.get('destination')}`",
                f"- Added entries: `{install_summary.get('added_entries')}`",
                f"- Replaced entries: `{install_summary.get('replaced_entries')}`",
                f"- Replacement history rows: `{len(install_summary.get('replacement_history') or [])}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_closeout(
    *,
    sources: list[Path],
    destination: Path,
    captures: list[Path],
    out_dir: Path,
    review_reports: list[Path] | None = None,
    variance_reports: list[Path] | None = None,
    target_validation_reports: list[Path] | None = None,
    shape_family_shadow_reports: list[Path] | None = None,
    install: bool = False,
    replace_existing: bool = False,
    require_variance_gate: bool = True,
    require_target_validation_gate: bool = False,
    allow_selector_review_cache: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_entries: list[dict[str, Any]] = []
    for source in sources:
        candidate_entries.extend(install_autotune_cache.read_cache(source))
    candidate_cache_path = out_dir / "candidate-cache.json"
    install_autotune_cache.write_cache(candidate_cache_path, candidate_entries)
    ledger = promotion_ledger.build_ledger(
        captures,
        candidate_cache_path,
        review_reports or [],
        variance_reports or [],
        target_validation_reports or [],
        shape_family_shadow_reports or [],
        require_variance_gate=require_variance_gate,
    )
    ledger_paths = promotion_ledger.write_outputs(ledger, out_dir / "ledger")
    ledger_json = Path(ledger_paths["json"])

    blockers = []
    if ledger["blocked_count"]:
        blockers.append(f"promotion_ledger_blocked_rows:{ledger['blocked_count']}")
    install_summary: dict[str, Any] | None = None
    install_error: str | None = None
    try:
        install_summary = install_autotune_cache.install_cache(
            sources,
            destination,
            dry_run=not install,
            replace_existing=replace_existing,
            promotion_ledgers=[ledger_json],
            require_variance_gate=require_variance_gate,
            require_target_validation_gate=require_target_validation_gate,
            allow_selector_review_cache=allow_selector_review_cache,
        )
    except install_autotune_cache.AutotuneCacheInstallError as exc:
        install_error = str(exc)
        blockers.append(f"cache_install_blocked:{install_error}")

    summary = {
        "schema": "rns8_cache_promotion_closeout_v1",
        "policy": "reviewed_captures_variance_target_validation_then_cache_install_gate",
        "complete": not blockers,
        "dry_run": not install,
        "source_count": len(sources),
        "capture_count": len(captures),
        "review_report_count": len(review_reports or []),
        "variance_report_count": len(variance_reports or []),
        "target_validation_report_count": len(target_validation_reports or []),
        "shape_family_shadow_report_count": len(shape_family_shadow_reports or []),
        "ledger_path": str(ledger_json),
        "ledger_markdown_path": ledger_paths["markdown"],
        "candidate_cache_path": str(candidate_cache_path),
        "candidate_entry_count": len(candidate_entries),
        "ledger_blocked_count": ledger["blocked_count"],
        "destination": str(destination),
        "install_status": "passed" if install_summary is not None else "blocked",
        "install_summary": install_summary,
        "install_error": install_error,
        "blockers": blockers,
    }
    json_path = out_dir / "cache-promotion-closeout.json"
    md_path = out_dir / "cache-promotion-closeout.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(summary, md_path)
    summary["json_path"] = str(json_path)
    summary["markdown_path"] = str(md_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True, help="reviewed cache JSON")
    parser.add_argument("--destination", type=Path, required=True, help="destination cache JSON")
    parser.add_argument("--capture", type=Path, action="append", required=True, help="reviewed benchmark capture")
    parser.add_argument("--review-report", type=Path, action="append", default=[])
    parser.add_argument("--variance-report", type=Path, action="append", default=[])
    parser.add_argument("--target-validation-report", type=Path, action="append", default=[])
    parser.add_argument("--shape-family-shadow-report", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--install", action="store_true", help="write destination; default is dry-run only")
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--no-require-variance-gate", dest="require_variance_gate", action="store_false")
    parser.set_defaults(require_variance_gate=True)
    parser.add_argument("--require-target-validation-gate", action="store_true")
    parser.add_argument("--allow-selector-review-cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    summary = build_closeout(
        sources=args.source,
        destination=args.destination,
        captures=args.capture,
        out_dir=args.out_dir,
        review_reports=args.review_report,
        variance_reports=args.variance_report,
        target_validation_reports=args.target_validation_report,
        shape_family_shadow_reports=args.shape_family_shadow_report,
        install=args.install,
        replace_existing=args.replace_existing,
        require_variance_gate=args.require_variance_gate,
        require_target_validation_gate=args.require_target_validation_gate,
        allow_selector_review_cache=args.allow_selector_review_cache,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(summary["json_path"])
    if args.require_complete and not summary["complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
