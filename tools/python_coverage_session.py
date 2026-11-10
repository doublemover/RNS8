"""Plan or explicitly collect isolated Python production branch/subprocess coverage."""

from __future__ import annotations

import argparse
import configparser
import json
import os
from pathlib import Path
import re
import sys

from compiled_binary_identity import canonical_digest
from execution_inventory import bounded_read
from host_instrumentation_session import (
    Session,
    check_bookends,
    reference,
    invocation_path,
    source_inventory,
    write_json,
    valid_total_budget,
)
from json_evidence import parse_evidence_json
from python_coverage_contract import (
    branch_report,
    invariant_observations,
    marker_lines,
    process_control,
    raw_data,
)
from quality_inventory import ROOT, inventory
from quality_policy import load_quality_lock

POLICY = ROOT / "metadata/python-coverage-policy.json"
WORKER = ROOT / "tools/python_coverage_worker.py"
JOURNAL = ROOT / "tools/python_test_inventory.py"
CONTROL = ROOT / "tests/controls/python_coverage_process.py"


def policy():
    value = parse_evidence_json(bounded_read(POLICY)[0])
    if (
        not isinstance(value, dict)
        or value.get("schema") != "rns8_python_branch_coverage_policy_v1"
        or value.get("all_selected_cases_must_pass") is not True
        or value.get("percentage_threshold") is not None
    ):
        raise ValueError("reviewed explicit Python coverage policy required")
    for field, pattern in (
        ("production_modules", r"tools/[A-Za-z0-9_]+\.py"),
        ("test_modules", r"test_[A-Za-z0-9_]+"),
        ("required_test_ids", r"test_[A-Za-z0-9_]+\.[A-Za-z0-9_.]+"),
    ):
        rows = value.get(field)
        if (
            not isinstance(rows, list)
            or not rows
            or len(rows) > 128
            or any(type(row) is not str or not re.fullmatch(pattern, row) for row in rows)
            or len(set(rows)) != len(rows)
        ):
            raise ValueError("invalid native Python coverage policy " + field)
    for name in value["production_modules"]:
        if Path(name).name.startswith("test_") or not (ROOT / name).resolve(
            strict=True
        ).is_relative_to(ROOT / "tools"):
            raise ValueError("production denominator contains test scaffolding or escaped path")
    invariants = value.get("invariants")
    if not isinstance(invariants, list) or not invariants or len(invariants) > 128:
        raise ValueError("bounded explicit Python invariant mapping required")
    names = set()
    for row in invariants:
        if (
            not isinstance(row, dict)
            or set(row) != {"name", "source", "function", "test_id"}
            or any(type(item) is not str or not item for item in row.values())
            or row["name"] in names
            or row["source"] not in value["production_modules"]
            or row["test_id"] not in value["required_test_ids"]
            or not row["function"].isidentifier()
        ):
            raise ValueError("invalid source/test invariant identity")
        names.add(row["name"])
    return value


def configuration(path: Path, data_file: Path, source: Path, include: list[Path], context: str):
    def escaped(value):
        value = str(value)
        if "\n" in value or "\r" in value:
            raise ValueError("coverage paths cannot contain newlines")
        return value.replace("$", "$$")

    config = configparser.ConfigParser(interpolation=None)
    config["run"] = {
        "branch": "True",
        "parallel": "True",
        "core": "ctrace",
        "dynamic_context": "test_function",
        "patch": "subprocess\n_exit",
        "source": escaped(source),
        "data_file": escaped(data_file),
        "context": context,
    }
    config["report"] = {
        "include": "\n".join(escaped(path) for path in include),
        "skip_empty": "False",
        "ignore_errors": "False",
    }
    config["json"] = {"show_contexts": "True", "pretty_print": "True"}
    with path.open("x", encoding="utf-8") as handle:
        config.write(handle)
    return reference(path)


def cov_command(python: Path, action: str, config: Path, *args) -> list[str]:
    return [str(python), "-m", "coverage", action, "--rcfile=" + str(config), *map(str, args)]


def validated_data(
    session: Session, python: Path, directory: Path, *, name: str, marker_source=None
):
    command = [str(python), str(WORKER), "data", "--directory", str(directory)]
    if marker_source:
        command.extend(["--marker-source", str(marker_source)])
    _, value = session.json(name + "-native-data-check", command)
    refs = raw_data(directory)
    if (
        not isinstance(value, dict)
        or value.get("schema") != "rns8_python_coverage_data_validation_v1"
        or not isinstance(value.get("files"), list)
        or len(value["files"]) != len(refs)
    ):
        raise ValueError("complete native per-process Python data inspection required")
    expected = {row["path"]: row for row in refs}
    seen = set()
    for row in value["files"]:
        if (
            not isinstance(row, dict)
            or row.get("path") not in expected
            or row["path"] in seen
            or row.get("has_arcs") is not True
            or row.get("sha256") != expected[row["path"]]["sha256"]
        ):
            raise ValueError("native Python data check omitted/changed a process profile")
        seen.add(row["path"])
    return refs, value


def combine(session, python, config, directory, output, name):
    destination = output / (name + "-combined")
    destination.mkdir()
    merged = destination / ".coverage"
    session.run(
        name + "-combine",
        cov_command(
            python, "combine", config, "--keep", "--data-file=" + str(merged), str(directory)
        ),
    )
    before = reference(merged)
    report = output / (name + "-native.json")
    session.run(
        name + "-json",
        cov_command(
            python,
            "json",
            config,
            "--data-file=" + str(merged),
            "--output=" + str(report),
            "--show-contexts",
        ),
    )
    check_bookends(before)
    document = parse_evidence_json(bounded_read(report)[0])
    return document, dict(merged=before, native_report=reference(report))


def execute(args, plan, selected_policy, version):
    output = Path(plan["output"])
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COVERAGE_", "ASAN_", "UBSAN_", "TSAN_", "LSAN_", "LLVM_PROFILE_"))
    }
    if any(environment.get(key) for key in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_AUDIT")):
        raise ValueError("isolated Python coverage rejects inherited interpreter/runtime injection")
    environment.update(PYTHONOPTIMIZE="", PYTHONNOUSERSITE="1")
    output.mkdir(parents=True)
    session = Session(
        output,
        environment,
        args.timeout_seconds,
        args.artifact_mib * 1024**2,
        total_seconds=getattr(args, "total_seconds", None),
    )
    result = dict(
        schema="rns8_python_branch_coverage_session_v1",
        status="blocked",
        plan=plan,
        blockers=[],
        scope="explicit_python_production_modules_not_native_or_gpu_code",
        production_latency=False,
        mutation_detection="not_run",
    )
    try:
        result["source_inventory"] = source_inventory()
        result["policy"] = reference(POLICY)
        result["tool_lock"] = reference(ROOT / "metadata/quality-tools.json")
        capture, identity = session.json(
            "coverage-package-identity", [str(args.python), str(WORKER), "identity"]
        )
        if (
            not isinstance(identity, dict)
            or identity.get("schema") != "rns8_python_coverage_package_identity_v1"
            or identity.get("version") != version
            or identity.get("assertions_enabled") is not True
            or identity.get("python_version") != plan["python_version"]
        ):
            raise ValueError("selected isolated Python/Coverage.py identity or assertions differ")
        result["package_identity"] = identity
        preflight = output / "preflight-data"
        preflight.mkdir()
        child_cwd = output / "child-cwd"
        child_cwd.mkdir()
        config = output / "preflight.ini"
        result["preflight_configuration"] = configuration(
            config, preflight / ".coverage", CONTROL.parent, [CONTROL], "rns8-parent-child-control"
        )
        captured = session.run(
            "parent-child-tracing-control",
            cov_command(
                args.python, "run", config, str(CONTROL), "parent", "--child-cwd", str(child_cwd)
            ),
        )
        if "CoverageWarning" in captured["stderr"]:
            raise ValueError("native parent/child tracing warning")
        control = parse_evidence_json(captured["stdout"])
        raw, validation = validated_data(
            session, args.python, preflight, name="preflight", marker_source=CONTROL
        )
        result["process_control"] = process_control(
            captured, control, raw, validation, marker_lines(CONTROL)
        )
        result["preflight_raw"] = raw
        result["preflight_data_validation"] = validation
        # Sensitivity profiles stay separate and cannot inflate application scores.
        _, result["preflight_artifacts"] = combine(
            session, args.python, config, preflight, output, "preflight"
        )
        application = output / "application-data"
        application.mkdir()
        config = output / "application.ini"
        source_refs = [reference(ROOT / name) for name in selected_policy["production_modules"]]
        result["configuration"] = configuration(
            config,
            application / ".coverage",
            ROOT / "tools",
            [Path(row["path"]) for row in source_refs],
            "rns8-python-application",
        )
        discover_path = output / "test-discovery.json"
        command = [
            str(args.python),
            str(JOURNAL),
            "--discover-only",
            "--output",
            str(discover_path),
        ]
        for name in selected_policy["test_modules"]:
            command.extend(["--module", name])
        discover_capture = session.run("native-test-discovery", command)
        discovered = parse_evidence_json(bounded_read(discover_path)[0])
        if (
            not isinstance(discovered, dict)
            or discovered.get("schema") != "rns8_native_unittest_discovery_v1"
            or discovered.get("assertions_enabled") is not True
            or discovered.get("process_id") != discover_capture.get("process_id")
            or not isinstance(discovered.get("discovered"), list)
            or not discovered["discovered"]
        ):
            raise ValueError("nonempty native Python test discovery required")
        if not set(selected_policy["required_test_ids"]).issubset(discovered["discovered"]):
            raise ValueError("mandatory real Python evidence tests missing")
        result["test_discovery"] = reference(discover_path)
        selected_file = output / "selected-tests.json"
        write_json(selected_file, discovered["discovered"])
        result["selected_tests"] = reference(selected_file)
        test_path = output / "test-execution.json"
        command = cov_command(
            args.python,
            "run",
            config,
            str(JOURNAL),
            "--output",
            str(test_path),
            "--selected",
            str(selected_file),
        )
        for name in selected_policy["test_modules"]:
            command.extend(["--module", name])
        execution = session.run("native-test-execution", command, required=False)
        tests = parse_evidence_json(bounded_read(test_path)[0])
        result["test_execution"] = reference(test_path)
        if (
            not isinstance(tests, dict)
            or tests.get("schema") != "rns8_native_unittest_inventory_v1"
            or tests.get("process_id") != execution.get("process_id")
            or tests.get("assertions_enabled") is not True
            or tests.get("discovered") != discovered["discovered"]
            or tests.get("selected") != discovered["discovered"]
            or tests.get("status") != "passed"
            or set(tests.get("passed", [])) != set(discovered["discovered"])
        ):
            result["blockers"].append("native_python_case_inventory_failed")
        if not session.stages[-1]["succeeded"] or "CoverageWarning" in execution["stderr"]:
            result["blockers"].append("native_python_test_process_incomplete_or_tracing_warning")
        raw, validation = validated_data(session, args.python, application, name="application")
        result["raw_profiles"] = raw
        result["data_validation"] = validation
        native, result["artifacts"] = combine(
            session, args.python, config, application, output, "application"
        )
        summary = branch_report(native, root=ROOT, sources=source_refs, version=version)
        result["invariant_observations"] = invariant_observations(
            native, selected_policy["invariants"], root=ROOT
        )
        result["summary"] = write_json(output / "branch-summary.json", summary)
        session.run(
            "application-html",
            cov_command(
                args.python,
                "html",
                config,
                "--data-file=" + result["artifacts"]["merged"]["path"],
                "--directory=" + str(output / "html"),
            ),
        )
        html = [reference(path) for path in sorted((output / "html").rglob("*")) if path.is_file()]
        if not html or not (output / "html/index.html").is_file():
            raise ValueError("Python coverage HTML missing")
        result["html_inventory"] = html
        selected = set(selected_policy["production_modules"])
        result["excluded_python_sources"] = [
            row.path for row in inventory() if row.role == "python" and row.path not in selected
        ]
        if raw_data(application) != raw or raw_data(preflight) != result["preflight_raw"]:
            raise ValueError("Python process profile set changed during reporting")
        check_bookends([result, tests, discovered, summary, session.stages])
        result["status"] = "failed" if result["blockers"] else "collected"
    except (OSError, ValueError, TypeError, KeyError) as error:
        result["blockers"].append(str(error))
    result["stages"] = session.stages
    result["identity_sha256"] = canonical_digest(result)
    write_json(output / "report.json", result)
    print(
        json.dumps(
            dict(
                status=result["status"],
                report=str(output / "report.json"),
                blockers=result["blockers"],
            )
        )
    )
    return 0 if result["status"] == "collected" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--total-seconds", type=int, help="total owned child-process budget")
    parser.add_argument("--artifact-mib", type=int, default=2048)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.python = invocation_path(args.python)
        output = args.out.resolve()
        if output == ROOT / "temp" or not output.is_relative_to(ROOT / "temp") or output.exists():
            raise ValueError("fresh ignored Python coverage output required")
        if (
            not 1 <= args.timeout_seconds <= 7200
            or not 128 <= args.artifact_mib <= 8192
            or not valid_total_budget(args.total_seconds)
        ):
            raise ValueError("Python coverage budgets outside supported bounds")
        selected = policy()
        lock = load_quality_lock(ROOT)
        version = lock["coverage"]["wheels"][0]["version"]
        dependencies = parse_evidence_json(
            (ROOT / "metadata/dependencies.json").read_text(encoding="utf-8")
        )
        python_version = next(
            row["selected"]["version"]
            for row in dependencies["components"]
            if row["name"] == "python"
        )
        plan = dict(
            schema="rns8_python_branch_coverage_plan_v1",
            total_seconds=args.total_seconds,
            output=str(output),
            python=str(args.python),
            python_version=[int(part) for part in python_version.split(".")],
            coverage_version=version,
            policy=reference(POLICY),
            test_modules=selected["test_modules"],
            production_modules=selected["production_modules"],
            parent_child_control_separate=True,
            runtime_validation_executed=False,
            production_latency=False,
        )
        if not args.execute:
            print(json.dumps(plan, indent=2))
            return 0
        if sys.flags.optimize:
            raise ValueError("Python driver assertions must be enabled")
        return execute(args, plan, selected, version)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print("Python coverage: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
