"""Read LLVM source coverage against a complete, source-bound host denominator."""

from __future__ import annotations

import math
from pathlib import Path

from benchmark_request import file_digest

METRICS = ("functions", "regions", "lines", "branches")


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
