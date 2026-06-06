"""Command-line entrypoint for benchmark schema validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import BenchmarkSchemaError, load_capture, validate_capture


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate rns8-bench JSON capture files.")
    parser.add_argument("captures", type=Path, nargs="+", help="benchmark JSON capture files")
    parser.add_argument("--json", action="store_true", help="emit machine-readable validation result")
    args = parser.parse_args()

    results = []
    all_errors: list[str] = []
    for path in args.captures:
        try:
            data = load_capture(path)
            result = validate_capture(data, path)
            results.append({"path": str(path), "valid": True, **result})
        except BenchmarkSchemaError as exc:
            messages = str(exc).splitlines()
            all_errors.extend(messages)
            results.append({"path": str(path), "valid": False, "errors": messages})

    if args.json:
        print(json.dumps({"valid": not all_errors, "captures": results}, indent=2, sort_keys=True))
    else:
        for item in results:
            if item["valid"]:
                print(f"{item['path']}: valid schema v{item['schema_version']}")
            else:
                for message in item["errors"]:
                    print(message)
    return 0 if not all_errors else 1
