#!/usr/bin/env python3
"""Summarize target validation without cross-target inference."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark_schema import load_capture, validate_capture


DEFAULT_OUT_DIR = Path("temp") / "target-validation-reports"


def capture_target(path: Path) -> dict[str, Any]:
    capture = load_capture(path)
    validate_capture(capture, path)
    device = capture.get("device") if isinstance(capture.get("device"), dict) else {}
    target = capture.get("target_variant") if isinstance(capture.get("target_variant"), dict) else {}
    toolchain = capture.get("hip_toolchain") if isinstance(capture.get("hip_toolchain"), dict) else {}
    return {
        "path": str(path),
        "backend_selected": capture.get("backend_selected"),
        "selected_kernel": capture.get("selected_kernel"),
        "semantics": capture.get("semantics"),
        "target_id": target.get("target_id") or device.get("gcn_arch"),
        "target_namespace": target.get("target_namespace"),
        "review_group_key": target.get("review_group_key"),
        "device_name": device.get("name"),
        "hip_runtime_version": device.get("hip_runtime_version"),
        "hip_driver_version": device.get("hip_driver_version"),
        "hip_sdk_or_rocm_version": toolchain.get("hip_sdk_or_rocm_version"),
        "configured_amdgpu_targets": capture.get("configured_amdgpu_targets"),
        "validated_scope": "this_capture_only_no_cross_target_inference",
    }


def build_report(paths: list[Path]) -> dict[str, Any]:
    captures = [capture_target(path) for path in paths]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for capture in captures:
        groups[str(capture.get("review_group_key") or capture.get("target_id") or "unknown")].append(capture)
    return {
        "schema_version": 1,
        "policy": "target_validation_is_per_os_gpu_toolchain_group_only",
        "capture_count": len(captures),
        "groups": [
            {"review_group_key": key, "capture_count": len(value), "captures": value}
            for key, value in sorted(groups.items())
        ],
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "target-validation-report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"json": str(json_path)}


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
        for label, path in write_outputs(report, args.out_dir).items():
            print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
