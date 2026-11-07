"""Select expensive native checks from an actual bounded Git tree difference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re

from process_progress import capture_succeeded, run_capture

ROOT = Path(__file__).resolve().parents[1]


def analysis_changed_paths(paths):
    # One JSON line is safe for GitHub outputs and environment transport. Large
    # inventories deliberately select the full lane instead of being truncated.
    encoded = json.dumps(paths, ensure_ascii=True, separators=(",", ":"))
    return (
        encoded
        if len(encoded.encode("utf-8")) <= 16384 and (paths is None or len(paths) <= 1024)
        else "null"
    )


def requires_source_checks(paths):
    if not isinstance(paths, list) or len(paths) > 100000:
        raise ValueError("bounded explicit changed paths required")
    for name in paths:
        if type(name) is not str or not name or "\\" in name or "\0" in name:
            raise ValueError("nonempty Git-relative changed path required")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("changed path escapes the repository")
    return any(
        not (
            (
                name.startswith("docs/")
                and PurePosixPath(name).suffix.lower()
                in {".md", ".png", ".svg", ".jpg", ".jpeg", ".webp", ".gif", ".pdf"}
            )
            or name == "docs/documentation-inventory.json"
            or ("/" not in name and name.endswith(".md"))
            or name in {"LICENSE", "third_party/README.md"}
            or (name.startswith("src/") and name.endswith("/README.md"))
            or name.startswith((".github/ISSUE_TEMPLATE/", ".github/PULL_REQUEST_TEMPLATE"))
        )
        for name in paths
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event", required=True, choices=("pull_request", "schedule", "workflow_dispatch")
    )
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--github-output", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = None
    try:
        if args.event == "pull_request":
            if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", args.base_sha):
                raise ValueError("actual PR base SHA required")
            capture = run_capture(
                [
                    "git",
                    "--no-optional-locks",
                    "diff",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--no-renames",
                    "--name-only",
                    "-z",
                    args.base_sha,
                    "HEAD",
                    "--",
                ],
                cwd=ROOT,
                label="native-ci-change-scope",
                timeout_seconds=30,
                max_output_bytes=8 * 1024**2,
                return_on_interrupt=True,
            )
            if not capture_succeeded(capture):
                raise ValueError("native CI changed-source inventory unavailable")
            text = capture["stdout"]
            if text and not text.endswith("\0"):
                raise ValueError("native Git path inventory is truncated")
            paths = text[:-1].split("\0") if text else []
        required = True if paths is None else requires_source_checks(paths)
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write("source_required=" + str(required).lower() + "\n")
            output.write("analysis_changed_paths=" + analysis_changed_paths(paths) + "\n")
        print(
            json.dumps(
                dict(
                    source_required=required,
                    event=args.event,
                    paths=paths,
                    scope="execution_selection_not_validation_or_qualification",
                )
            )
        )
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print("native CI scope: " + str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
