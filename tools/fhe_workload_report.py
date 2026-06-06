#!/usr/bin/env python3
"""Group FHE/lattice proxy captures without making library compatibility claims."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark_schema import load_capture, validate_capture


DEFAULT_OUT_DIR = Path("temp") / "fhe-workload-reports"


def row_for_capture(path: Path) -> dict[str, Any]:
    capture = load_capture(path)
    validate_capture(capture, path)
    proxy = capture.get("workload_proxy") if isinstance(capture.get("workload_proxy"), dict) else {}
    scenario = capture.get("scenario_metadata") if isinstance(capture.get("scenario_metadata"), dict) else {}
    return {
        "path": str(path),
        "label": proxy.get("label") or scenario.get("workflow_name") or "unlabeled",
        "family": proxy.get("family") or scenario.get("algebra_family"),
        "tower_role": proxy.get("tower_role") or scenario.get("phase_label"),
        "reuse_profile": proxy.get("reuse_profile") or scenario.get("reuse_profile"),
        "transform_role": proxy.get("transform_role") or scenario.get("lowering_role"),
        "output_domain_requirement": proxy.get("output_domain_requirement") or scenario.get("output_domain_requirement"),
        "compatibility_claim": bool(proxy.get("compatibility_claim")),
        "backend_selected": capture.get("backend_selected"),
        "selected_kernel": capture.get("selected_kernel"),
        "semantics": capture.get("semantics"),
        "shape": {"m": capture.get("m"), "n": capture.get("n"), "k": capture.get("k")},
    }


def build_report(paths: list[Path]) -> dict[str, Any]:
    rows = [row_for_capture(path) for path in paths]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["label"], row["output_domain_requirement"])].append(row)
    return {
        "schema_version": 1,
        "policy": "proxy_workload_grouping_only_no_fhe_library_compatibility_claim",
        "groups": [{"key": key, "rows": value} for key, value in sorted(groups.items(), key=lambda item: str(item[0]))],
    }


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fhe-workload-report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(args.captures)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(write_report(report, args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
