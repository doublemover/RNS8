"""Bounded Coverage.py producer and process-profile admission, without tracing."""

from __future__ import annotations

import ast
from pathlib import Path
import re

from host_instrumentation_session import reference


def raw_data(directory: Path) -> list[dict]:
    paths = sorted(directory.iterdir())
    if not 1 <= len(paths) <= 10000:
        raise ValueError("nonempty bounded Python process data required")
    result = []
    for path in paths:
        parts = path.name.rsplit(".", 2)
        if (
            path.is_symlink()
            or not path.is_file()
            or len(parts) != 3
            or not parts[0].startswith(".coverage.")
            or not re.fullmatch(r"[1-9][0-9]*", parts[1])
            or not re.fullmatch(r"[A-Za-z0-9_]{1,128}", parts[2])
        ):
            raise ValueError("owned native Coverage.py process data filenames required")
        row = reference(path)
        if not 1 <= row["bytes"] <= 64 * 1024**2:
            raise ValueError("empty/oversized Python process profile")
        row["process_id"] = int(parts[1])
        result.append(row)
    if sum(row["bytes"] for row in result) > 1024**3:
        raise ValueError("Python process profiles exceed total budget")
    return result


def marker_lines(path: Path) -> dict[str, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"parent_path", "child_path"}:
            expected = 23 if node.name == "parent_path" else 37
            matches = [
                item.lineno
                for item in ast.walk(node)
                if isinstance(item, ast.Return)
                and isinstance(item.value, ast.Constant)
                and type(item.value.value) is int
                and item.value.value == expected
            ]
            if len(matches) != 1:
                raise ValueError("unique actual parent/child body marker required")
            result[node.name] = matches[0]
    if set(result) != {"parent_path", "child_path"}:
        raise ValueError("parent/child body markers missing")
    return result


def process_control(
    capture: dict, document: dict, profiles: list[dict], validation: dict, markers: dict
) -> dict:
    from process_progress import capture_succeeded

    child = document.get("child") if isinstance(document, dict) else None
    child_capture = document.get("child_process") if isinstance(document, dict) else None
    parent_id = capture.get("process_id")
    if (
        not capture_succeeded(capture)
        or not isinstance(child, dict)
        or not isinstance(child_capture, dict)
        or document.get("schema") != "rns8_python_coverage_process_control_v1"
        or type(parent_id) is not int
        or parent_id <= 0
        or document.get("process_id") != parent_id
        or type(document.get("value")) is not int
        or document["value"] != 23
        or not capture_succeeded(child_capture)
        or type(child.get("process_id")) is not int
        or child["process_id"] <= 0
        or child["process_id"] == parent_id
        or child["process_id"] != child_capture.get("process_id")
        or child.get("role") != "child"
        or type(child.get("value")) is not int
        or child["value"] != 37
    ):
        raise ValueError("actual parent/child process control failed")
    if (
        not isinstance(validation, dict)
        or validation.get("schema") != "rns8_python_coverage_data_validation_v1"
    ):
        raise ValueError("native per-process branch data inspection required")
    files = validation.get("files")
    if not isinstance(files, list) or len(files) != len(profiles):
        raise ValueError("native inspected profile inventory differs")
    expected = {row["path"]: row for row in profiles}
    observations = {}
    for row in files:
        if not isinstance(row, dict) or row.get("path") not in expected:
            raise ValueError("unknown process profile")
        original = expected[row["path"]]
        if (
            row.get("sha256") != original["sha256"]
            or row.get("has_arcs") is not True
            or row["path"] in observations
        ):
            raise ValueError("unbound or nonbranch process profile")
        observations[row["path"]] = row
    for pid, marker in (
        (parent_id, markers["parent_path"]),
        (child["process_id"], markers["child_path"]),
    ):
        matched = [observations[row["path"]] for row in profiles if row["process_id"] == pid]
        if not matched or not any(
            marker in row.get("marker_lines_executed", []) for row in matched
        ):
            raise ValueError("actual process body was not traced: " + str(pid))
    return dict(
        status="passed",
        scope="bounded_parent_child_tracing_control_not_all_subprocesses",
        parent_process_id=parent_id,
        child_process_id=child["process_id"],
        application_pass_count=0,
    )


def branch_report(document, *, root: Path, sources: list[dict], version: str) -> dict:
    meta = document.get("meta") if isinstance(document, dict) else None
    files = document.get("files") if isinstance(document, dict) else None
    if (
        not isinstance(meta, dict)
        or type(meta.get("format")) is not int
        or meta["format"] != 3
        or meta.get("version") != version
        or meta.get("branch_coverage") is not True
        or not isinstance(files, dict)
        or not files
        or len(files) > 10000
    ):
        raise ValueError("version-bound native Python branch report required")
    expected = {Path(row["path"]).resolve(): row for row in sources}
    if not expected or len(expected) != len(sources):
        raise ValueError("nonempty unique Python source denominator required")
    rows = {}
    totals = dict(statements=0, covered_statements=0, branches=0, covered_branches=0)
    for name, row in files.items():
        path = (root / Path(name)).resolve()
        if path not in expected or path in rows or not isinstance(row, dict):
            raise ValueError("Python report file outside exact selected source denominator")
        if reference(path)["sha256"] != expected[path]["sha256"]:
            raise ValueError("Python coverage source changed")
        values = row.get("summary")
        fields = (
            "covered_lines",
            "num_statements",
            "missing_lines",
            "excluded_lines",
            "num_branches",
            "covered_branches",
            "missing_branches",
        )
        if not isinstance(values, dict) or any(
            type(values.get(key)) is not int or values[key] < 0 for key in fields
        ):
            raise ValueError("invalid native Python branch counts")
        if (
            values["covered_lines"] + values["missing_lines"] != values["num_statements"]
            or values["covered_branches"] + values["missing_branches"] != values["num_branches"]
        ):
            raise ValueError("native Python coverage contradicts source denominator")
        lines = {}
        for key in ("executed_lines", "missing_lines", "excluded_lines"):
            value = row.get(key)
            if (
                not isinstance(value, list)
                or any(type(line) is not int or line <= 0 for line in value)
                or len(set(value)) != len(value)
            ):
                raise ValueError("invalid native Python line inventory")
            lines[key] = value
        if (
            len(lines["missing_lines"]) != values["missing_lines"]
            or len(lines["excluded_lines"]) != values["excluded_lines"]
            or set(lines["executed_lines"]) & set(lines["missing_lines"])
        ):
            raise ValueError("native Python missing lines contradict counts")
        arcs = {}
        for key in ("executed_branches", "missing_branches"):
            value = row.get(key)
            if not isinstance(value, list) or any(
                not isinstance(pair, list)
                or len(pair) != 2
                or any(type(v) is not int or v == 0 for v in pair)
                for pair in value
            ):
                raise ValueError("native Python branch arcs missing/malformed")
            pairs = [tuple(pair) for pair in value]
            if len(set(pairs)) != len(pairs):
                raise ValueError("duplicate native Python branch arc")
            arcs[key] = value
        if set(map(tuple, arcs["executed_branches"])) & set(map(tuple, arcs["missing_branches"])):
            raise ValueError("native branch is both executed and missing")
        rows[path] = dict(
            **expected[path],
            relative_path=path.relative_to(root).as_posix(),
            summary=values,
            **lines,
            **arcs,
        )
        totals["statements"] += values["num_statements"]
        totals["covered_statements"] += values["covered_lines"]
        totals["branches"] += values["num_branches"]
        totals["covered_branches"] += values["covered_branches"]
    if set(rows) != set(expected):
        raise ValueError("selected production modules missing from native report")
    if not totals["covered_statements"] or not totals["branches"] or not totals["covered_branches"]:
        raise ValueError("no executed production statement/branch coverage")
    return dict(
        schema="rns8_python_production_branch_coverage_v1",
        status="collected",
        files=list(rows.values()),
        totals=totals,
        reviewed_percentage_threshold=None,
        mutation_detection="not_run",
        production_latency=False,
    )


def invariant_observations(document, invariants: list[dict], *, root: Path) -> list[dict]:
    if not invariants:
        raise ValueError("explicit source/test invariant mapping required")
    files = {(root / Path(name)).resolve(): row for name, row in document["files"].items()}
    result = []
    for invariant in invariants:
        path = (root / invariant["source"]).resolve(strict=True)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == invariant["function"]
        ]
        if len(functions) != 1 or path not in files:
            raise ValueError("invariant production function missing from source/native mapping")
        function = functions[0]
        native = files[path].get("functions", {}).get(invariant["function"])
        contexts = files[path].get("contexts")
        if (
            not isinstance(native, dict)
            or native.get("start_line") != function.lineno
            or not isinstance(contexts, dict)
        ):
            raise ValueError("native invariant function/context mapping missing")
        test_id = invariant["test_id"]
        lines = [
            line
            for line in native.get("executed_lines", [])
            if type(line) is int
            and function.lineno < line <= function.end_lineno
            and any(
                type(context) is str and (context == test_id or context.endswith("|" + test_id))
                for context in contexts.get(str(line), [])
            )
        ]
        if (
            not lines
            or type(native.get("summary", {}).get("covered_branches")) is not int
            or native["summary"]["covered_branches"] <= 0
        ):
            raise ValueError(
                "required test did not observe production invariant body/branches: "
                + invariant["name"]
            )
        result.append(
            dict(
                invariant,
                source=reference(path),
                observed_body_lines=lines,
                native_missing_branches=native.get("missing_branches"),
                scope="source_body_and_test_context_not_mutation_detection",
            )
        )
    return result
