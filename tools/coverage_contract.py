"""Read LLVM source coverage against a complete, source-bound host denominator."""

from __future__ import annotations

import math
import hashlib
from pathlib import Path

from benchmark_request import file_digest

METRICS = ("functions", "regions", "lines", "branches")


def invariant_mapping(native, summary, seeds, routes, tests, *, root: Path):
    """Locate reviewed operations, native API counts and selected assertions.

    Counts cover the selected suite. A passing test and executed API are not a
    per-test call-edge proof or evidence that a production mutant was detected.
    """
    if (
        not isinstance(routes, dict)
        or routes.get("schema") != "rns8_native_invariant_routes_v1"
        or not isinstance(routes.get("routes"), dict)
        or set(routes["routes"]) != {seed["id"] for seed in seeds}
        or not 1 <= len(seeds) <= 16
        or tests.get("schema") != "rns8_native_test_execution_inventory_v1"
    ):
        raise ValueError("complete bounded native invariant/test mapping required")
    root = root.resolve()
    files = {row["relative_path"]: row for row in summary["files"]}
    functions = native["data"][0]["functions"]
    rows = []
    test_sets = {key: {row["name"] for row in tests[key]} for key in ("discovered", "selected")}
    test_sets.update({key: set(tests[key]) for key in ("executed", "passed")})
    text_cache = {}

    def source_text(name):
        if name in text_cache:
            return text_cache[name]
        reference = files[name]
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or path != Path(reference["path"]).resolve():
            raise ValueError("native invariant source path differs from coverage")
        with path.open("rb") as handle:
            raw = handle.read(16 * 1024**2 + 1)
        if len(raw) > 16 * 1024**2 or hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("native invariant source drift or byte budget exceeded")
        text_cache[name] = raw.decode("utf-8").replace("\r\n", "\n")
        return text_cache[name]

    for seed in seeds:
        route = routes["routes"][seed["id"]]
        if (
            not isinstance(route, dict)
            or set(route) != {"invariant", "entrypoint", "source"}
            or type(route["invariant"]) is not str
            or not route["invariant"]
            or ((route["entrypoint"] is None) != (route["source"] is None))
            or (
                route["entrypoint"] is not None
                and (
                    type(route["entrypoint"]) is not str
                    or not route["entrypoint"].startswith("rns8_")
                    or type(route["source"]) is not str
                    or route["source"] not in files
                )
            )
        ):
            raise ValueError("explicit native route or helper-only scope required")
        reference = files.get(seed["source"])
        if reference is None or reference["sha256"] != seed["source_sha256"]:
            raise ValueError("native invariant production source drift")
        text = source_text(seed["source"])
        anchor, operation = seed["anchor"], seed["before"]
        if (
            text.count(anchor) != 1
            or text.count(operation) != 1
            or text.index(operation) < text.index(anchor)
        ):
            raise ValueError("native invariant operation/anchor is not uniquely bound")
        start = text.count("\n", 0, text.index(anchor)) + 1
        operation_start = text.count("\n", 0, text.index(operation)) + 1
        end = operation_start + operation.count("\n")
        branches = [
            row for row in reference["branches"] if row["line"] <= end and row["end_line"] >= start
        ]
        api = []
        if route["entrypoint"] is not None:
            target = (root / route["source"]).resolve()
            source_text(route["source"])
            for function in functions:
                if function.get("name") != route["entrypoint"]:
                    continue
                names, count = function.get("filenames"), function.get("count")
                if (
                    not isinstance(names, list)
                    or any(type(name) is not str or not Path(name).is_absolute() for name in names)
                    or type(count) is not int
                    or count < 0
                ):
                    raise ValueError("native invariant function mapping/count missing")
                if target not in {Path(name).resolve() for name in names}:
                    raise ValueError("native invariant entrypoint maps to a different source")
                api.append(count)
        mapped_tests = [
            dict(test, **{key: test["name"] in names for key, names in test_sets.items()})
            for test in seed["tests"]
        ]
        gaps = []
        if not all(all(row[key] for key in test_sets) for row in mapped_tests):
            gaps.append("mapped_assertion_case_not_executed_and_passed")
        if not branches:
            gaps.append("no_native_branch_mapping_in_reviewed_source_prefix")
        elif any(not row["true_covered"] or not row["false_covered"] for row in branches):
            gaps.append("uncovered_native_branch_outcomes_in_reviewed_source_prefix")
        if route["entrypoint"] is not None and (not api or not sum(api)):
            gaps.append("public_entrypoint_not_observed_in_native_suite")
        rows.append(
            dict(
                seed_id=seed["id"],
                invariant=route["invariant"],
                source=reference["path"],
                source_sha256=reference["sha256"],
                reviewed_prefix=dict(start_line=start, end_line=end),
                changed_operation=dict(start_line=operation_start, end_line=end),
                branches=branches,
                public_entrypoint=route["entrypoint"],
                entrypoint_source={
                    key: files[route["source"]][key] for key in ("path", "sha256", "state")
                }
                if route["source"]
                else None,
                native_entrypoint_calls=sum(api) if api else None,
                scope="public_cpu_route"
                if route["entrypoint"]
                else "host_template_fake_operations_only",
                tests=mapped_tests,
                coverage_gaps=gaps,
                per_test_callchain_proven=False,
                mutation_detection="separate_native_mutation_lane",
            )
        )
    return dict(
        schema="rns8_native_coverage_invariants_v1",
        invariants=rows,
        status="gaps" if any(row["coverage_gaps"] for row in rows) else "observed",
        assertion_strength="not_established_by_coverage",
        production_latency=False,
    )


def mapped_sources(value, admitted: set[Path]) -> list[str]:
    if (
        not isinstance(value, dict)
        or value.get("type") != "llvm.coverage.json.export"
        or value.get("version") != "3.1.0"
    ):
        raise ValueError("selected native mapping producer required")
    data = value.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise ValueError("one native mapping discovery required")
    files = data[0].get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= 100000:
        raise ValueError("nonempty bounded native file mappings required")
    result, seen = [], set()
    for row in files:
        if not isinstance(row, dict) or type(row.get("filename")) is not str:
            raise ValueError("native mapped filename missing")
        path = Path(row["filename"])
        if not path.is_absolute():
            raise ValueError("native mapped paths must be absolute")
        path = path.resolve()
        if path in seen:
            raise ValueError("duplicate native mapped filename")
        seen.add(path)
        if path in admitted:
            result.append(row["filename"])
    if not result:
        raise ValueError("nonempty first-party native mapping required")
    return result


def nonnegative(value, name):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("invalid native coverage " + name)
    return value


def summary(value):
    if not isinstance(value, dict):
        raise ValueError("native coverage summary missing")
    result = {}
    for name in METRICS:
        row = value.get(name)
        if not isinstance(row, dict):
            raise ValueError("native coverage metric missing: " + name)
        count, covered = nonnegative(row.get("count"), name), nonnegative(row.get("covered"), name)
        percent = row.get("percent")
        if (
            covered > count
            or type(percent) not in (int, float)
            or not math.isfinite(percent)
            or not 0 <= percent <= 100
        ):
            raise ValueError("native coverage metric contradicts denominator")
        if "notcovered" in row and (
            type(row["notcovered"]) is not int or row["notcovered"] != count - covered
        ):
            raise ValueError("native uncovered count contradicts denominator")
        # Native LLVM emits 0 percent for an empty denominator. Recompute useful
        # percentages from integer counts rather than trusting display rounding.
        expected = 100 * covered / count if count else 0.0
        if abs(percent - expected) > 0.011:
            raise ValueError("native coverage percentage contradicts counts")
        result[name] = dict(
            count=count,
            covered=covered,
            not_covered=count - covered,
            percent=expected if count else None,
        )
    return result


def branch(row):
    if not isinstance(row, list) or len(row) != 9:
        raise ValueError("LLVM3.1 branch tuple required")
    for value in row:
        nonnegative(value, "branch coordinate/count")
    if min(row[:4]) < 1 or (row[2], row[3]) < (row[0], row[1]):
        raise ValueError("invalid branch source span")
    return dict(
        line=row[0],
        column=row[1],
        end_line=row[2],
        end_column=row[3],
        true_count=row[4],
        false_count=row[5],
        true_covered=row[4] > 0,
        false_covered=row[5] > 0,
        file_id=row[6],
        expanded_file_id=row[7],
        kind=row[8],
    )


def llvm_export(
    value,
    *,
    root: Path,
    sources: list[dict],
    compiled_sources: set[Path],
    generated: set[str],
    required_modules: tuple[str, ...],
) -> dict:
    if (
        not isinstance(value, dict)
        or value.get("type") != "llvm.coverage.json.export"
        or value.get("version") != "3.1.0"
    ):
        raise ValueError("selected LLVM coverage JSON3.1.0 producer required")
    data = value.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise ValueError("one native coverage mapping required")
    files = data[0].get("files")
    functions = data[0].get("functions")
    if (
        not isinstance(files, list)
        or not 1 <= len(files) <= 100000
        or not isinstance(functions, list)
        or not functions
    ):
        raise ValueError("nonempty detailed file/function coverage required")
    root = root.resolve()
    indexed = {Path(row["path"]).resolve(): row for row in sources}
    if len(indexed) != len(sources):
        raise ValueError("duplicate coverage source denominator")
    rows, excluded, seen = {}, [], set()
    for row in files:
        if not isinstance(row, dict) or type(row.get("filename")) is not str:
            raise ValueError("native coverage filename missing")
        path = Path(row["filename"])
        if not path.is_absolute():
            raise ValueError("coverage filenames must be bound absolute source paths")
        path = path.resolve()
        if path in seen:
            raise ValueError("duplicate native coverage file mapping")
        seen.add(path)
        if path not in indexed:
            excluded.append(
                dict(path=str(path), reason="outside_selected_first_party_source_denominator")
            )
            continue
        reference = indexed[path]
        if file_digest(path) != reference["sha256"]:
            raise ValueError("coverage source changed since build inventory")
        relative = path.relative_to(root).as_posix()
        metrics = summary(row.get("summary"))
        branches = row.get("branches")
        if not isinstance(branches, list):
            raise ValueError("detailed native branches missing")
        native_branches = [branch(item) for item in branches]
        # Macro expansions keep their own native source/filename mapping in the
        # raw export. File branch rows are not falsely treated as all expansions.
        expansions = row.get("expansions")
        if not isinstance(expansions, list):
            raise ValueError("native expansion mapping missing")
        segments = row.get("segments")
        if not isinstance(segments, list):
            raise ValueError("detailed native source segments missing")
        if (
            not any(item["count"] for item in metrics.values())
            and not native_branches
            and not expansions
            and not segments
        ):
            # llvm-cov can emit an empty row for an explicitly requested source
            # with no mapping. It does not establish that a disabled TU compiled.
            continue
        role = (
            "generated"
            if relative in generated
            else "production"
            if relative.startswith(("src/", "include/"))
            else "test_or_tool"
        )
        rows[path] = dict(
            **reference,
            relative_path=relative,
            role=role,
            metrics=metrics,
            branches=native_branches,
            expansion_records=len(expansions),
            state="compiled_and_covered"
            if metrics["lines"]["covered"]
            else "compiled_and_uncovered",
        )
    for path, reference in indexed.items():
        if path in rows:
            continue
        relative = path.relative_to(root).as_posix()
        if path in compiled_sources:
            state = "compiled_without_observed_source_mapping"
        elif path.suffix.lower() in {".cpp", ".cc", ".cxx", ".c", ".hip"}:
            state = "translation_unit_not_selected_in_this_host_build"
        else:
            state = "header_or_fragment_without_observed_mapping"
        rows[path] = dict(
            **reference,
            relative_path=relative,
            role="generated"
            if relative in generated
            else "production"
            if relative.startswith(("src/", "include/"))
            else "test_or_tool",
            state=state,
            metrics=None,
            branches=[],
            expansion_records=0,
        )
    totals = {}
    for role in ("production", "test_or_tool", "generated"):
        selected = [
            row for row in rows.values() if row["role"] == role and row["metrics"] is not None
        ]
        totals[role] = {
            name: dict(
                count=sum(row["metrics"][name]["count"] for row in selected),
                covered=sum(row["metrics"][name]["covered"] for row in selected),
            )
            for name in METRICS
        }
    blockers = []
    for module in required_modules:
        row = rows.get((root / module).resolve())
        if row is None or row["metrics"] is None or not row["metrics"]["lines"]["covered"]:
            blockers.append("required_production_module_not_observed:" + module)
    if (
        not totals["production"]["branches"]["count"]
        or not totals["production"]["branches"]["covered"]
    ):
        blockers.append("no_executed_production_branch_outcomes")
    return dict(
        schema="rns8_native_host_source_coverage_v1",
        status="failed" if blockers else "collected",
        files=sorted(rows.values(), key=lambda row: row["relative_path"]),
        excluded=excluded,
        totals=totals,
        native_function_records=len(functions),
        blockers=blockers,
        required_modules=list(required_modules),
        missing_mapping_count=sum(row["metrics"] is None for row in rows.values()),
        scope="selected_native_host_objects_not_all_gpu_or_platform_code",
        production_latency=False,
        reviewed_percentage_threshold=None,
        mutation_detection="not_run",
    )
