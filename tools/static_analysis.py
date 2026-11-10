#!/usr/bin/env python3
"""Plan or execute bounded native host/HIP static analysis; never fixes source."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time
import uuid

from analysis_diagnostics import read_diagnostics, SUPPORTED_TOOL
from analysis_inputs import Compilation, load_inputs
from benchmark_request import file_digest
from compiled_binary_identity import canonical_digest
from compiler_context import compiler_environment
from gpu_qualification import owned_artifact_files, write_json
from json_evidence import load_evidence_json
from process_progress import run_capture, capture_succeeded
from quality_inventory import ROOT, inventory
from quality_policy import load_quality_lock

SCHEMA = "rns8_static_analysis_plan_v2"
RESULT_SCHEMA = "rns8_static_analysis_results_v2"
CONTROL_CHECK = "clang-analyzer-core.NullDereference"


@dataclass(frozen=True)
class AnalysisTask:
    context: Compilation
    stage: str
    target: str | None

    @property
    def identity(self) -> str:
        return canonical_digest(
            dict(context=self.context.identity, stage=self.stage, target=self.target)
        )


def selected_tasks(contexts: tuple[Compilation, ...], lane: str) -> tuple[AnalysisTask, ...]:
    if lane == "host":
        tasks = [
            AnalysisTask(context, "host-analyzer", None)
            for context in contexts
            if context.kind == "host"
        ]
    elif lane == "hip-syntax":
        tasks = [
            AnalysisTask(context, "hip-syntax-only", None)
            for context in contexts
            if context.kind == "hip"
        ]
    elif lane == "hip-analyzer":
        tasks = []
        for context in contexts:
            if context.kind == "hip":
                tasks.append(AnalysisTask(context, "hip-host-analyzer", None))
                tasks.extend(
                    AnalysisTask(context, "hip-device-analyzer", target)
                    for target in context.targets
                )
    else:
        raise ValueError("unknown analysis lane")
    if not tasks:
        raise ValueError("zero selected translation-unit contexts cannot pass an analysis lane")
    return tuple(tasks)


def select_changed_tasks(contexts, lane, changed_paths=None):
    """Narrow only complete TU-only changes; headers conservatively affect every TU."""
    tasks = selected_tasks(contexts, lane)
    reasons = []
    if changed_paths is not None:
        if type(changed_paths) is not list or len(changed_paths) > 1024:
            raise ValueError("bounded explicit changed-source inventory required")
        if len(json.dumps(changed_paths, ensure_ascii=True).encode("utf-8")) > 16384:
            raise ValueError("changed-source inventory exceeds 16 KiB; request a full lane")
        for name in changed_paths:
            if type(name) is not str or not name or "\\" in name or "\0" in name:
                raise ValueError("canonical repository-relative changed source required")
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != name or ":" in name:
                raise ValueError("changed source escapes canonical repository paths")
        changed_paths = sorted(set(changed_paths))
        if not changed_paths:
            reasons.append("empty_change_inventory")
        sources = {task.context.source for task in tasks}
        requested = set()
        for name in changed_paths:
            source = (ROOT / name).resolve()
            if not source.is_relative_to(ROOT.resolve()) or not source.is_file():
                reasons.append("deleted_missing_or_external_source:" + name)
            elif PurePosixPath(name).suffix.lower() not in {".c", ".cc", ".cpp", ".cxx", ".hip"}:
                reasons.append("header_fragment_configuration_or_unknown_input:" + name)
            elif source not in sources:
                reasons.append("changed_source_not_configured_in_lane:" + name)
            else:
                requested.add(source)
        if not reasons:
            selected = tuple(task for task in tasks if task.context.source in requested)
            return selected, dict(
                mode="changed_translation_units",
                changed_paths=changed_paths,
                fallback_reasons=[],
                configured_pass_count=len(tasks),
                unselected_pass_count=len(tasks) - len(selected),
                header_dependency_policy="full_lane_on_any_non_translation_unit_change",
            )
    return tasks, dict(
        mode="full_configured_lane",
        changed_paths=changed_paths,
        fallback_reasons=reasons,
        configured_pass_count=len(tasks),
        unselected_pass_count=0,
        header_dependency_policy="full_lane_on_any_non_translation_unit_change",
    )


def planned_tasks(inputs, plan):
    tasks, selection = select_changed_tasks(
        inputs.contexts, plan["lane"], plan["selection"]["changed_paths"]
    )
    if selection != plan["selection"]:
        raise ValueError("retained changed-source selection differs from configured inputs")
    return tasks


def hip_projection(task: AnalysisTask) -> dict:
    record = task.context.native_record()
    if task.context.kind != "hip":
        return record
    argv = list(record["arguments"])
    if task.stage == "hip-device-analyzer":
        argv = [argument for argument in argv if not argument.startswith("--offload-arch=")]
        argv += ["--offload-arch=" + task.target, "--offload-device-only"]
    elif task.stage == "hip-host-analyzer":
        argv.append("--offload-host-only")
    elif task.stage == "hip-syntax-only":
        result, position = [], 0
        while position < len(argv):
            argument = argv[position]
            if argument == "-c":
                position += 1
                continue
            if argument == "-o":
                if position + 1 == len(argv):
                    raise ValueError("custom HIP output flag has no operand")
                position += 2
                continue
            result.append(argument)
            position += 1
        argv = [*result, "-fsyntax-only"]
    record["arguments"] = argv
    return record


def task_record(task: AnalysisTask) -> dict:
    return dict(
        identity=task.identity,
        context=task.context.identity,
        stage=task.stage,
        target=task.target,
        original_native_record=task.context.native_record(),
        projected_native_record=hip_projection(task),
        registered_targets=task.context.registered_targets,
        generated_source=task.context.generated_source,
        explicit_environment=dict(task.context.environment),
        targets=task.context.targets,
    )


def verify_diagnostic_decoder() -> None:
    decoder = next(
        row for row in load_quality_lock(ROOT)["analysis"]["wheels"] if row["name"] == "ruamel.yaml"
    )
    try:
        observed = importlib.metadata.version("ruamel.yaml")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("install the locked optional native diagnostic decoder") from error
    if observed != decoder["version"]:
        raise ValueError("native diagnostic decoder differs from the locked version")


def control_header() -> tuple[Path, dict[str, int]]:
    path = ROOT / "tests/analysis/branch_control.hpp"
    lines = path.read_text(encoding="utf-8").splitlines()
    locations = {}
    for branch in ("host", "device"):
        found = [
            index + 1
            for index, line in enumerate(lines)
            if "RNS8_" + branch.upper() + "_BRANCH" in line
        ]
        if len(found) != 1:
            raise ValueError("analyzer branch control source marker is missing or duplicated")
        locations[branch] = found[0]
    return path.resolve(strict=True), locations


def native_analyzer(tool: Path | None, tool_bin: Path | None) -> Path:
    if tool is not None:
        return tool.resolve(strict=True)
    if tool_bin is None:
        raise ValueError(
            "supply the actual --clang-tidy binary or the locked --tool-bin environment"
        )
    prefix = tool_bin.resolve(strict=True).parent
    candidates = [prefix / "Lib/site-packages/clang_tidy/data/bin/clang-tidy.exe"]
    candidates += list(prefix.glob("lib/python*/site-packages/clang_tidy/data/bin/clang-tidy"))
    candidates = [path.resolve() for path in candidates if path.is_file()]
    if len(candidates) != 1:
        raise ValueError(
            "locked native clang-tidy binary is absent or ambiguous; do not bind only its Python launcher"
        )
    return candidates[0]


def build_plan(
    build: Path,
    out: Path,
    *,
    lane="host",
    tool: Path | None = None,
    tool_bin: Path | None = None,
    timeout_seconds=120,
    total_seconds=3600,
    max_artifact_bytes=1024**3,
    minimum_free_bytes=5 * 1024**3,
    changed_paths=None,
) -> dict:
    for value in (timeout_seconds, total_seconds):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("positive finite analysis time budgets required")
    if total_seconds <= timeout_seconds + 30:
        raise ValueError(
            "analysis session requires a 30-second export reserve and preparation time"
        )
    for value in (max_artifact_bytes, minimum_free_bytes):
        if type(value) is not int or value < 1:
            raise ValueError("positive analysis artifact/free-space byte budgets required")
    out = out.resolve()
    if (
        out.exists()
        or out == (ROOT / "temp").resolve()
        or not out.is_relative_to((ROOT / "temp").resolve())
    ):
        raise ValueError("analysis needs a fresh child of repository temp/")
    inputs = load_inputs(build)
    tasks, selection = select_changed_tasks(inputs.contexts, lane, changed_paths)
    lock = load_quality_lock(ROOT)
    analyzer = None if lane == "hip-syntax" else native_analyzer(tool, tool_bin)
    checker = next(row for row in lock["analysis"]["wheels"] if row["name"] == "clang-tidy")
    if checker["version"] != SUPPORTED_TOOL:
        raise ValueError("analyzer policy and native diagnostic adapter versions disagree")
    missing = sorted(
        {
            str(path)
            for task in tasks
            for path in (*task.context.generated_inputs, task.context.source)
            if not path.is_file()
        }
    )
    plan = dict(
        schema=SCHEMA,
        lane=lane,
        build_directory=str(inputs.build),
        output_directory=str(out),
        configuration=inputs.configuration,
        source_root=str(ROOT),
        input_artifacts=dict(inputs.artifact_hashes),
        analyzer=str(analyzer) if analyzer else None,
        analyzer_sha256=file_digest(analyzer) if analyzer else None,
        tool_version=SUPPORTED_TOOL,
        config_sha256=file_digest(ROOT / ".clang-tidy"),
        policy_sha256=file_digest(ROOT / "metadata/quality-tools.json"),
        selected_context_count=len({task.context.identity for task in tasks}),
        selected_pass_count=len(tasks),
        selection=selection,
        tasks=[task_record(task) for task in tasks],
        excluded_records=inputs.excluded_records,
        missing_generated_inputs=missing,
        unconfigured_first_party_translation_units=[
            source.path
            for source in inventory()
            if source.role == "cpp"
            and Path(source.path).suffix.lower() in {".cpp", ".c", ".cc", ".cxx"}
            and (ROOT / source.path).resolve()
            not in {context.source for context in inputs.contexts}
        ],
        budgets=dict(
            timeout_seconds=timeout_seconds,
            total_seconds=total_seconds,
            export_reserve_seconds=30,
            max_artifact_bytes=max_artifact_bytes,
            minimum_free_bytes=minimum_free_bytes,
        ),
        mode="plan_only_no_tool_or_compiler_executed",
        hardware_qualification=False,
        scope="configured_static_contexts_not_GPU_correctness_or_performance",
    )
    plan["identity_sha256"] = canonical_digest(plan)
    return plan


def _header_filter() -> str:
    parts = []
    for character in ROOT.as_posix():
        if character == "/":
            parts.append(r"[/\\]")
        elif os.name == "nt" and character.isascii() and character.isalpha():
            parts.append("[" + character.lower() + character.upper() + "]")
        else:
            parts.append(re.escape(character))
    return "^" + "".join(parts) + r"[/\\](include|src|tools|tests|benchmarks|examples)[/\\].*$"


def included_headers(stderr: str, directory: Path, known: dict[Path, str]) -> list[dict]:
    result = {}
    pattern = _header_filter()
    for line in stderr.splitlines():
        match = re.fullmatch(r"\.+ (.+)", line)
        if match is None:
            continue
        native = directory / match[1]
        path = native.resolve()
        if path not in known:
            continue
        if re.match(pattern, str(native)) is None:
            raise ValueError("header filter does not cover an observed first-party include path")
        result[str(path)] = dict(path=str(path), sha256=known[path])
    return [result[key] for key in sorted(result)]


def _analysis_flags(compiler: Path, path: Path) -> list[str]:
    if compiler.stem.lower() in {"cl", "clang-cl"}:
        return [
            "/FI" + str(path),
            "/clang:-H",
            "/clang:-Xclang",
            "/clang:-analyzer-opt-analyze-headers",
        ]
    # Standalone functions in the injected header are otherwise excluded from
    # the analyzer. Keep this diagnostic-only option in the recorded argv.
    return ["-include", str(path), "-H", "-Xclang", "-analyzer-opt-analyze-headers"]


def analyzer_command(
    plan: dict, directory: Path, source: Path, diagnostics: Path, injected: list[str]
) -> list[str]:
    return [
        plan["analyzer"],
        "-p",
        str(directory),
        str(source),
        "--config-file=" + str(ROOT / ".clang-tidy"),
        "--header-filter=" + _header_filter(),
        "--export-fixes=" + str(diagnostics),
        *("--extra-arg=" + argument for argument in injected),
    ]


def classify_diagnostics(
    diagnostics: list[dict], process: dict, stage: str, control: Path, lines: dict[str, int]
) -> tuple[str, str | None, list[dict]]:
    if not capture_succeeded(process) or any(
        row["parse_failure"] or row["level"] == "Error" for row in diagnostics
    ):
        return (
            "parse_or_process_failure",
            "native analyzer did not complete the configured program",
            diagnostics,
        )
    branch = "device" if stage == "hip-device-analyzer" else "host"
    controls = [
        row
        for row in diagnostics
        if row["check"] == CONTROL_CHECK
        and row["level"] == "Warning"
        and row["message"]["location"]["path"] == str(control)
        and row["message"]["location"]["line"] == lines[branch]
    ]
    if len(controls) != 1:
        return (
            "unsupported_toolchain",
            "native analyzer did not observe exactly one expected branch/header control",
            diagnostics,
        )
    tagged = [
        dict(row, role="deliberate_branch_control" if row is controls[0] else "program_diagnostic")
        for row in diagnostics
    ]
    return "analyzed_findings" if len(tagged) > 1 else "analyzed_clean", None, tagged


def retained_process(root: Path, label: str, artifacts: dict) -> dict:
    path = root / (label + ".process.json")
    if str(path) not in artifacts:
        raise ValueError("native analyzer process receipt not retained")
    value = _read_record(path)
    if (
        type(value) is not dict
        or type(value.get("process_id")) is not int
        or value["process_id"] <= 0
        or type(value.get("command")) is not list
        or not value["command"]
        or any(type(arg) is not str for arg in value["command"])
    ):
        raise ValueError("native analyzer process identity or command invalid")
    result = dict(value)
    if type(value.get("streams")) is not dict or set(value["streams"]) != {"stdout", "stderr"}:
        raise ValueError("complete native analyzer streams required")
    for stream in ("stdout", "stderr"):
        expected = root / (label + "." + stream + ".log")
        if (
            value["streams"][stream]
            != dict(path=str(expected), sha256=artifacts.get(str(expected)))
            or str(expected) not in artifacts
            or expected.stat().st_size > 16 * 1024**2
        ):
            raise ValueError("native analyzer stream custody or budget invalid")
        result[stream] = expected.read_text(encoding="utf-8")
    return result


def parse_control_rejected(process: dict, diagnostics: list[dict], missing: Path) -> bool:
    return (
        type(process.get("returncode")) is int
        and process["returncode"] > 0
        and process.get("timed_out") is False
        and process.get("interrupted") is False
        and process.get("limit_exceeded") is None
        and process.get("capture_error") is None
        and any(
            row["parse_failure"]
            and row["level"] == "Error"
            and missing.name in row["message"]["message"]
            for row in diagnostics
        )
    )


def verify_analyzer_controls(
    result: dict, plan: dict, root: Path, artifacts: dict, compiler: Path
) -> None:
    if plan["tool_version"] != SUPPORTED_TOOL:
        raise ValueError("retained analyzer version differs from supported native adapter")
    for label, arguments in (
        ("analyzer-version", ["--version"]),
        ("verify-config", ["--config-file=" + str(ROOT / ".clang-tidy"), "--verify-config"]),
        ("list-checks", ["--config-file=" + str(ROOT / ".clang-tidy"), "--list-checks"]),
        ("dump-config", ["--config-file=" + str(ROOT / ".clang-tidy"), "--dump-config"]),
    ):
        process = retained_process(root, label, artifacts)
        if (
            process["command"] != [plan["analyzer"], *arguments]
            or process.get("working_directory") != str(ROOT)
            or not capture_succeeded(process)
        ):
            raise ValueError("retained native analyzer/configuration control did not succeed")
        if (
            label == "analyzer-version"
            and re.search(
                r"(?<![\w.])" + re.escape(SUPPORTED_TOOL) + r"(?![\w.+-])",
                process["stdout"] + process["stderr"],
            )
            is None
        ):
            raise ValueError("retained native analyzer version not observed")
        if label == "list-checks" and CONTROL_CHECK not in process["stdout"]:
            raise ValueError("retained analyzer control check was not enabled")
    controls = result.get("preflight_controls")
    if (
        type(controls) is not list
        or len(controls) != 1
        or type(controls[0]) is not dict
        or controls[0].get("kind") != "missing_header_parse_error_rejected"
    ):
        raise ValueError("one retained native missing-header sensitivity control required")
    item = controls[0]
    directory = root / "parse-error-control"
    diagnostic = directory / "diagnostics.yaml"
    missing = directory / ("missing-header-" + result["run_id"] + ".hpp")
    record = plan["tasks"][0]["projected_native_record"]
    source = (Path(record["directory"]) / record["file"]).resolve()
    process = retained_process(root, "parse-error-control", artifacts)
    prefix = [
        plan["analyzer"],
        "-p",
        str(directory),
        str(source),
        "--config-file=" + str(ROOT / ".clang-tidy"),
        "--export-fixes=" + str(diagnostic),
    ]
    command = prefix + ["--extra-arg=" + part for part in _analysis_flags(compiler, missing)]
    if (
        missing.exists()
        or item.get("missing_header") != str(missing)
        or item.get("source") != str(source)
        or _read_record(directory / "compile_commands.json") != [record]
        or process["command"] != command
        or process.get("working_directory") != str(ROOT)
        or canonical_digest(item.get("process"))
        != canonical_digest(
            {key: value for key, value in process.items() if key not in {"stdout", "stderr"}}
        )
    ):
        raise ValueError("native missing-header control command/context custody invalid")
    if (
        item.get("diagnostic_artifact")
        != dict(path=str(diagnostic), sha256=artifacts.get(str(diagnostic)))
        or str(diagnostic) not in artifacts
    ):
        raise ValueError("native parse-control diagnostics not retained")
    diagnostics = read_diagnostics(
        diagnostic,
        source=source,
        directory=Path(record["directory"]),
        tool_version=plan["tool_version"],
    )
    if not parse_control_rejected(process, diagnostics, missing):
        raise ValueError("native parse control did not detect the intended missing header")


def verify_analyzed_pass(
    detailed: dict,
    task: dict,
    plan: dict,
    root: Path,
    artifacts: dict,
    known_headers: dict[Path, str],
    compiler: Path,
) -> tuple[str, int]:
    process = retained_process(root, task["identity"], artifacts)
    summary = {key: value for key, value in process.items() if key not in {"stdout", "stderr"}}
    if canonical_digest(summary) != canonical_digest(detailed["process"]):
        raise ValueError("analysis pass process differs from native receipt")
    directory = root / task["identity"]
    if _read_record(directory / "compile_commands.json") != [task["projected_native_record"]]:
        raise ValueError("actual analyzer database differs from planned compiler context")
    record = task["projected_native_record"]
    if (
        any(
            canonical_digest(detailed.get(key)) != canonical_digest(task[key])
            for key in ("original_native_record", "projected_native_record")
        )
        and task["stage"] != "hip-syntax-only"
    ):
        raise ValueError("declared compiler projection differs from retained native context")
    source = (Path(record["directory"]) / record["file"]).resolve()
    if detailed["source"] != str(source):
        raise ValueError("native analyzer source differs from planned compilation")
    if task["stage"] == "hip-syntax-only":
        if (
            process["command"] != record["arguments"]
            or process.get("working_directory") != record["directory"]
            or not capture_succeeded(process)
            or detailed.get("analyzed") is not False
            or detailed.get("device_kernel_control_observed") is not False
        ):
            raise ValueError("syntax-only claim differs from actual native invocation")
        return "syntax_only", 0
    control, lines = control_header()
    injected = detailed.get("diagnostic_flags")
    if (
        injected != _analysis_flags(compiler, control)
        or detailed.get("injected_diagnostic_header") != str(control)
        or detailed.get("injected_header_sha256") != file_digest(control)
    ):
        raise ValueError("native analyzer branch/header injection invalid")
    diagnostic = directory / "diagnostics.yaml"
    if (
        detailed.get("diagnostic_artifact")
        != dict(path=str(diagnostic), sha256=artifacts.get(str(diagnostic)))
        or str(diagnostic) not in artifacts
    ):
        raise ValueError("native analyzer diagnostics not retained by actual pass")
    if process["command"] != analyzer_command(
        plan, directory, source, diagnostic, injected
    ) or process.get("working_directory") != str(ROOT):
        raise ValueError("actual analyzer command differs from declared context/control")
    raw = read_diagnostics(
        diagnostic,
        source=source,
        directory=Path(record["directory"]),
        tool_version=plan["tool_version"],
    )
    status, reason, tagged = classify_diagnostics(raw, process, task["stage"], control, lines)
    if (
        canonical_digest(tagged) != canonical_digest(detailed.get("diagnostics"))
        or detailed.get("reason") != reason
        or detailed.get("device_kernel_control_observed")
        is not (status.startswith("analyzed_") and task["stage"] == "hip-device-analyzer")
    ):
        raise ValueError("derived analyzer findings differ from retained native diagnostics")
    headers = included_headers(process["stderr"], Path(record["directory"]), known_headers)
    if (
        detailed.get("observed_first_party_headers") != headers
        or detailed.get("header_coverage_scope")
        != "observed_include_trace_not_all_potential_headers"
    ):
        raise ValueError("declared header coverage differs from native include trace")
    return status, sum(row["role"] == "program_diagnostic" for row in tagged) if status.startswith(
        "analyzed_"
    ) else 0


def _read_record(path: Path):
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("analysis receipt exceeds its read budget")
    return load_evidence_json(path)


def verify_receipt(path: Path) -> dict:
    """Reopen a retained session without launching tools or admitting GPU evidence."""
    path = path.resolve(strict=True)
    result = _read_record(path)
    if not isinstance(result, dict) or result.get("schema") != RESULT_SCHEMA:
        raise ValueError("unsupported analysis receipt")
    if result.get("status") not in {
        "in_progress",
        "blocked_or_incomplete",
        "incomplete_or_unsupported",
        "findings_require_triage",
        "syntax_only_not_analyzer_coverage",
        "selected_static_contexts_clean",
        "interrupted",
    }:
        raise ValueError("unknown static analysis receipt disposition")
    try:
        if str(uuid.UUID(result["run_id"])) != result["run_id"]:
            raise ValueError("noncanonical native analysis run identity")
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("canonical native analysis run identity required") from error
    payload = {key: value for key, value in result.items() if key != "identity_sha256"}
    if canonical_digest(payload) != result.get("identity_sha256"):
        raise ValueError("analysis receipt seal differs")
    root = path.parent
    artifacts = result.get("retained_artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("analysis receipt has no retained artifact inventory")
    if str(root / "plan.json") not in artifacts:
        raise ValueError("analysis receipt does not bind its plan artifact")
    for name, digest in artifacts.items():
        artifact = Path(name).resolve(strict=True)
        if not artifact.is_relative_to(root) or artifact == path or file_digest(artifact) != digest:
            raise ValueError("analysis artifact is outside its session or changed: " + name)
    plan = _read_record(root / "plan.json")
    if (
        not isinstance(plan, dict)
        or plan.get("schema") != SCHEMA
        or canonical_digest({key: value for key, value in plan.items() if key != "identity_sha256"})
        != plan.get("identity_sha256")
        or plan.get("identity_sha256") != result.get("plan_identity_sha256")
    ):
        raise ValueError("analysis receipt and retained plan differ")
    inputs = load_inputs(Path(plan["build_directory"]))
    actual_tasks = planned_tasks(inputs, plan)
    if result.get("selection") != plan["selection"]:
        raise ValueError("analysis result differs from retained changed-source scope")
    if (
        type(plan.get("selected_context_count")) is not int
        or plan["selected_context_count"] != len({task.context.identity for task in actual_tasks})
        or type(plan.get("selected_pass_count")) is not int
        or plan["selected_pass_count"] != len(actual_tasks)
    ):
        raise ValueError("planned analysis coverage differs from configured context denominator")
    if (
        canonical_digest([task_record(task) for task in actual_tasks])
        != canonical_digest(plan.get("tasks"))
        or dict(inputs.artifact_hashes) != plan["input_artifacts"]
        or inputs.configuration != plan["configuration"]
        or inputs.build != Path(plan["build_directory"]).resolve()
    ):
        raise ValueError("retained analysis plan differs from actual configured compiler inventory")
    native_tasks = {task.identity: task for task in actual_tasks}
    for name, digest in result["source_artifacts"].items():
        if file_digest(name) != digest:
            raise ValueError("analysis source/tool input changed: " + name)
    if plan.get("source_root") != str(ROOT) or any(
        result["source_artifacts"].get(str(ROOT / name)) != plan.get(key)
        for name, key in (
            (".clang-tidy", "config_sha256"),
            ("metadata/quality-tools.json", "policy_sha256"),
        )
    ):
        raise ValueError("analysis policy/source root not bound to retained inputs")
    if (
        any(
            result["source_artifacts"].get(name) != digest
            for name, digest in plan["input_artifacts"].items()
        )
        or plan["analyzer"]
        and result["source_artifacts"].get(plan["analyzer"]) != plan["analyzer_sha256"]
    ):
        raise ValueError("planned analyzer/configuration inputs not retained by actual session")
    if plan["analyzer"] and str(control_header()[0]) not in result["source_artifacts"]:
        raise ValueError("native branch control source absent from analysis input custody")
    for task in plan["tasks"]:
        record = task["projected_native_record"]
        source = (Path(record["directory"]) / record["file"]).resolve()
        if str(source) not in result["source_artifacts"]:
            raise ValueError("analyzed program source missing from input custody")
    known_headers = {
        Path(name).resolve(): digest for name, digest in result["source_artifacts"].items()
    }
    rows = result["results"]
    if any(row.get("status") in {"analyzed_clean", "analyzed_findings"} for row in rows):
        verify_diagnostic_decoder()
    identities = [row["identity"] for row in rows]
    planned = {task["identity"]: task for task in plan["tasks"]}
    if len(set(identities)) != len(identities) or not set(identities) <= planned.keys():
        raise ValueError("analysis pass inventory is duplicated or unplanned")
    for row in rows:
        reference = row["receipt"]
        if artifacts.get(reference["path"]) != reference["sha256"]:
            raise ValueError("analysis pass is not bound to a retained artifact")
        detailed = _read_record(Path(reference["path"]))
        if any(
            detailed[key] != row[key]
            for key in ("identity", "context", "stage", "target", "source", "status")
        ):
            raise ValueError("analysis compact pass differs from its full receipt")
        task = planned[row["identity"]]
        if any(row[key] != task[key] for key in ("identity", "context", "stage", "target")):
            raise ValueError("analysis pass differs from its planned context")
        if row["status"] in {"analyzed_clean", "analyzed_findings", "syntax_only"}:
            native_status, count = verify_analyzed_pass(
                detailed,
                task,
                plan,
                root,
                artifacts,
                known_headers,
                inputs.compiler
                if native_tasks[row["identity"]].context.kind == "host"
                else Path(task["projected_native_record"]["arguments"][0]),
            )
            if (
                row["status"] != native_status
                or type(row.get("program_diagnostic_count")) is not int
                or row["program_diagnostic_count"] != count
            ):
                raise ValueError("analysis coverage/findings differ from retained native pass")
            if canonical_digest(row["process"]) != canonical_digest(
                {key: detailed["process"].get(key) for key in row["process"]}
            ):
                raise ValueError("compact analyzer process differs from native observation")
    if any(row["status"] in {"analyzed_clean", "analyzed_findings"} for row in rows):
        first = actual_tasks[0]
        verify_analyzer_controls(
            result,
            plan,
            root,
            artifacts,
            inputs.compiler
            if first.context.kind == "host"
            else Path(first.context.native_record()["arguments"][0]),
        )
    expected = dict(
        selected_contexts=plan["selected_context_count"],
        selected_passes=len(plan["tasks"]),
        attempted_passes=len(rows),
        unattempted_passes=len(plan["tasks"]) - len(rows),
        analyzed_passes=sum(
            row["status"] in {"analyzed_clean", "analyzed_findings"} for row in rows
        ),
        syntax_checked_passes=sum(row["status"] == "syntax_only" for row in rows),
    )
    if any(
        type(result.get(key)) is not int or result[key] != value for key, value in expected.items()
    ):
        raise ValueError("analysis coverage counts differ from retained passes")
    if result["status"] == "selected_static_contexts_clean" and (
        not rows
        or len(rows) != len(planned)
        or any(row["status"] != "analyzed_clean" for row in rows)
    ):
        raise ValueError("clean analysis verdict requires every planned analyzer pass")
    if result["status"] == "syntax_only_not_analyzer_coverage" and (
        not rows or len(rows) != len(planned) or any(row["status"] != "syntax_only" for row in rows)
    ):
        raise ValueError("syntax verdict requires every planned syntax pass")
    if result["status"] == "findings_require_triage" and (
        len(rows) != len(planned)
        or not any(row["status"] == "analyzed_findings" for row in rows)
        or any(row["status"] not in {"analyzed_clean", "analyzed_findings"} for row in rows)
    ):
        raise ValueError("finding verdict requires complete observed analyzer coverage")
    if result["status"] in {
        "selected_static_contexts_clean",
        "syntax_only_not_analyzer_coverage",
        "findings_require_triage",
    } and (
        result.get("error") is not None
        or result.get("retained_artifact_inventory_error") is not None
    ):
        raise ValueError("completed static coverage cannot erase a session/export failure")
    if (
        result.get("hardware_qualification") is not False
        or result.get("performance_promotion_allowed") is not False
    ):
        raise ValueError("static analysis cannot qualify hardware or performance")
    return result


def execute(
    plan: dict,
    *,
    prepare=False,
    advisory=False,
    runner=run_capture,
    environment: dict[str, str] | None = None,
    clock=time.monotonic,
) -> dict:
    # Reopen the exact configured inputs; do not execute a stale edited plan.
    rebuilt = build_plan(
        Path(plan["build_directory"]),
        Path(plan["output_directory"]),
        lane=plan["lane"],
        changed_paths=plan["selection"]["changed_paths"],
        tool=Path(plan["analyzer"]) if plan["analyzer"] else None,
        **{
            key: plan["budgets"][key]
            for key in (
                "timeout_seconds",
                "total_seconds",
                "max_artifact_bytes",
                "minimum_free_bytes",
            )
        },
    )
    if canonical_digest(rebuilt) != canonical_digest(plan):
        raise ValueError("analysis plan no longer matches current configured inputs/tools")
    inputs = load_inputs(Path(plan["build_directory"]))
    tasks = planned_tasks(inputs, plan)
    out = Path(plan["output_directory"])
    volume = next(path for path in out.parents if path.exists())
    if shutil.disk_usage(volume).free < plan["budgets"]["minimum_free_bytes"]:
        raise ValueError("insufficient free space for analysis capture")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "plan.json", plan)
    child = dict(os.environ if environment is None else environment)
    started, run_id, rows, preflight_controls = clock(), str(uuid.uuid4()), [], []
    witnesses = dict(plan["input_artifacts"])
    retained = {}

    def retain(path):
        retained[str(path)] = file_digest(path)

    retain(out / "plan.json")
    witnesses.update(
        {
            str(ROOT / name): plan[key]
            for name, key in (
                (".clang-tidy", "config_sha256"),
                ("metadata/quality-tools.json", "policy_sha256"),
            )
        }
    )
    if plan["analyzer"]:
        witnesses[plan["analyzer"]] = plan["analyzer_sha256"]
    initial_environment = compiler_environment(child)

    def checkpoint(status, error=None):
        # Discover partial native outputs even when a decoder/preflight fails.
        # Existing hashes are never refreshed: later edits remain detectable.
        inventory_error = None
        try:
            artifacts = owned_artifact_files(out, out.parent)
            if (
                sum(artifact.stat().st_size for artifact in artifacts)
                > plan["budgets"]["max_artifact_bytes"]
            ):
                raise ValueError("retained analysis artifacts exceed the session byte budget")
            for artifact in artifacts:
                if artifact != out / "results.json" and str(artifact) not in retained:
                    retain(artifact)
        except (OSError, ValueError) as failure:
            inventory_error = str(failure)
            if status != "interrupted":
                status = "blocked_or_incomplete"
        analyzed = sum(row.get("status") in {"analyzed_clean", "analyzed_findings"} for row in rows)
        syntax = sum(row.get("status") == "syntax_only" for row in rows)
        result = dict(
            schema=RESULT_SCHEMA,
            run_id=run_id,
            plan_identity_sha256=plan["identity_sha256"],
            status=status,
            error=error,
            selected_contexts=plan["selected_context_count"],
            selected_passes=len(tasks),
            selection=plan["selection"],
            attempted_passes=len(rows),
            analyzed_passes=analyzed,
            syntax_checked_passes=syntax,
            unattempted_passes=len(tasks) - len(rows),
            results=rows,
            elapsed_seconds=clock() - started,
            compiler_environment=initial_environment,
            advisory_requested=advisory,
            source_artifacts=witnesses,
            preflight_controls=preflight_controls,
            retained_artifacts=dict(retained),
            retained_artifact_inventory_error=inventory_error,
            hardware_qualification=False,
            performance_promotion_allowed=False,
            external_dependency_identity="configured_paths_and_manifest_not_full_external_tree_rehash",
            peak_process_memory_bytes=None,
            memory_observation="not_available_from_owned_process_runner",
        )
        result["identity_sha256"] = canonical_digest(result)
        write_json(out / "results.json", result)
        return result

    def retain_pass(row, directory):
        for path in directory.iterdir():
            if path.is_file():
                retain(path)
        path = directory / "pass.json"
        write_json(path, row)
        retain(path)
        summary = {
            key: row[key] for key in ("identity", "context", "stage", "target", "source", "status")
        }
        summary["process"] = {
            key: row["process"].get(key)
            for key in (
                "process_id",
                "returncode",
                "timed_out",
                "interrupted",
                "limit_exceeded",
                "capture_error",
            )
        }
        summary["receipt"] = dict(path=str(path), sha256=file_digest(path))
        summary["program_diagnostic_count"] = sum(
            item.get("role") == "program_diagnostic" for item in row.get("diagnostics", [])
        )
        rows.append(summary)

    def call(argv, name, *, env=None, cwd=ROOT, context_pass=False):
        remaining = plan["budgets"]["total_seconds"] - (clock() - started) - 30
        if remaining <= 0:
            raise ValueError("analysis session time budget exhausted")
        if shutil.disk_usage(out).free < plan["budgets"]["minimum_free_bytes"]:
            raise ValueError("analysis free-space budget exhausted")
        value = runner(
            argv,
            cwd=cwd,
            env=child if env is None else env,
            label=name,
            timeout_seconds=min(plan["budgets"]["timeout_seconds"], remaining),
            max_output_bytes=16 * 1024**2,
            artifact_roots=[out],
            max_artifact_bytes=plan["budgets"]["max_artifact_bytes"],
            return_on_interrupt=True,
        )
        value["working_directory"] = str(cwd)
        streams = {}
        for stream in ("stdout", "stderr"):
            path = out / (name + "." + stream + ".log")
            path.write_text(value[stream], encoding="utf-8")
            retain(path)
            streams[stream] = dict(path=str(path), sha256=file_digest(path))
        summary = {key: item for key, item in value.items() if key not in {"stdout", "stderr"}}
        summary["streams"] = streams
        process_path = out / (name + ".process.json")
        write_json(process_path, summary)
        retain(process_path)
        value["receipt"] = summary
        if value.get("interrupted") and not context_pass:
            raise KeyboardInterrupt
        return value

    checkpoint("in_progress")
    try:
        if prepare:
            wrapper = (
                [sys.executable, str(ROOT / "tools/windows_dev.py")] if os.name == "nt" else []
            )
            value = call(
                [
                    *wrapper,
                    "cmake",
                    "--build",
                    str(inputs.build),
                    "--target",
                    "rns8_analysis_inputs",
                ],
                "prepare-inputs",
            )
            if not capture_succeeded(value):
                raise ValueError("real generated analysis input preparation failed")
            refreshed = load_inputs(inputs.build)
            if refreshed.artifact_hashes != inputs.artifact_hashes:
                raise ValueError(
                    "configured compilation contexts changed during preparation; replan"
                )
        for task in tasks:
            for path in (*task.context.generated_inputs, task.context.source):
                if not path.is_file():
                    raise ValueError("real configured analysis input is missing: " + str(path))
                witnesses[str(path)] = file_digest(path)
        for source in inventory():
            if source.role in {"cpp", "python"}:
                witnesses[str(ROOT / source.path)] = file_digest(ROOT / source.path)
        generated = json.loads(
            (ROOT / "metadata/first-party-inventory.json").read_text(encoding="utf-8")
        )["generated"]
        for name in generated:
            witnesses[str(ROOT / name)] = file_digest(ROOT / name)
        if plan["analyzer"]:
            control, _ = control_header()
            witnesses[str(control)] = file_digest(control)
        if plan["analyzer"]:
            version = call([plan["analyzer"], "--version"], "analyzer-version")
            if (
                not capture_succeeded(version)
                or re.search(
                    r"(?<![\w.])" + re.escape(SUPPORTED_TOOL) + r"(?![\w.+-])",
                    version["stdout"] + version["stderr"],
                )
                is None
            ):
                raise ValueError("native analyzer release differs from the pinned adapter")
            verify_diagnostic_decoder()
            for name, flag in (
                ("verify-config", "--verify-config"),
                ("list-checks", "--list-checks"),
                ("dump-config", "--dump-config"),
            ):
                value = call(
                    [
                        plan["analyzer"],
                        "--config-file=" + str(ROOT / ".clang-tidy"),
                        flag,
                    ],
                    name,
                )
                if not capture_succeeded(value):
                    raise ValueError(
                        "analyzer does not accept the maintained configuration: " + name
                    )
                if name == "list-checks" and CONTROL_CHECK not in value["stdout"]:
                    raise ValueError("analyzer control check is not enabled")
            negative = out / "parse-error-control"
            negative.mkdir()
            first = tasks[0]
            projected = hip_projection(first)
            write_json(negative / "compile_commands.json", [projected])
            missing = negative / ("missing-header-" + run_id + ".hpp")
            diagnostic_path = negative / "diagnostics.yaml"
            compiler = (
                inputs.compiler if first.context.kind == "host" else Path(projected["arguments"][0])
            )
            value = call(
                [
                    plan["analyzer"],
                    "-p",
                    str(negative),
                    str(first.context.source),
                    "--config-file=" + str(ROOT / ".clang-tidy"),
                    "--export-fixes=" + str(diagnostic_path),
                    *("--extra-arg=" + argument for argument in _analysis_flags(compiler, missing)),
                ],
                "parse-error-control",
                env={**child, **dict(first.context.environment)},
            )
            diagnostics = read_diagnostics(
                diagnostic_path,
                source=first.context.source,
                directory=first.context.directory,
            )
            if not parse_control_rejected(value, diagnostics, missing):
                raise ValueError(
                    "analyzer did not reject and retain the deliberate missing-header parse error"
                )
            preflight_controls.append(
                dict(
                    kind="missing_header_parse_error_rejected",
                    source=str(first.context.source),
                    missing_header=str(missing),
                    diagnostic_artifact=dict(
                        path=str(diagnostic_path), sha256=file_digest(diagnostic_path)
                    ),
                    process=value["receipt"],
                )
            )
            retain(negative / "compile_commands.json")
            retain(diagnostic_path)
        else:
            compilers = sorted(
                {Path(task.context.native_record()["arguments"][0]) for task in tasks}
            )
            for index, compiler in enumerate(compilers):
                value = call([str(compiler), "--version"], "hip-compiler-version-" + str(index))
                if not capture_succeeded(value):
                    raise ValueError("configured HIP compiler identity query failed")
        known_headers = {Path(name).resolve(): digest for name, digest in witnesses.items()}
        for task in tasks:
            directory = out / task.identity
            directory.mkdir()
            projected = hip_projection(task)
            write_json(directory / "compile_commands.json", [projected])
            env = {**child, **dict(task.context.environment)}
            if task.stage == "hip-syntax-only":
                value = call(
                    projected["arguments"],
                    task.identity,
                    env=env,
                    cwd=task.context.directory,
                    context_pass=True,
                )
                status = (
                    "interrupted"
                    if value.get("interrupted")
                    else "syntax_only"
                    if capture_succeeded(value)
                    else "failed_syntax"
                )
                retain_pass(
                    dict(
                        identity=task.identity,
                        context=task.context.identity,
                        stage=task.stage,
                        target=task.target,
                        source=str(task.context.source),
                        status=status,
                        process=value["receipt"],
                        analyzed=False,
                        device_kernel_control_observed=False,
                    ),
                    directory,
                )
                checkpoint("in_progress")
                if (
                    value.get("interrupted")
                    or value.get("timed_out")
                    or value.get("limit_exceeded")
                    or value.get("capture_error")
                ):
                    break
                continue
            control, lines = control_header()
            diagnostics_path = directory / "diagnostics.yaml"
            compiler = (
                inputs.compiler if task.context.kind == "host" else Path(projected["arguments"][0])
            )
            injected = _analysis_flags(compiler, control)
            argv = analyzer_command(
                plan, directory, task.context.source, diagnostics_path, injected
            )
            value = call(argv, task.identity, env=env, context_pass=True)
            diagnostics, headers, reason = [], [], None
            status = (
                "unsupported_toolchain" if capture_succeeded(value) else "parse_or_process_failure"
            )
            try:
                diagnostics = read_diagnostics(
                    diagnostics_path,
                    source=task.context.source,
                    directory=task.context.directory,
                )
                headers = included_headers(value["stderr"], task.context.directory, known_headers)
                status, reason, diagnostics = classify_diagnostics(
                    diagnostics, value, task.stage, control, lines
                )
            except (OSError, ValueError) as error:
                reason = str(error)
            retain_pass(
                dict(
                    identity=task.identity,
                    context=task.context.identity,
                    stage=task.stage,
                    target=task.target,
                    source=str(task.context.source),
                    status=status,
                    reason=reason,
                    diagnostics=diagnostics,
                    process=value["receipt"],
                    original_native_record=task.context.native_record(),
                    diagnostic_artifact=(
                        dict(
                            path=str(diagnostics_path),
                            sha256=file_digest(diagnostics_path),
                        )
                        if diagnostics_path.is_file()
                        else None
                    ),
                    observed_first_party_headers=headers,
                    header_coverage_scope="observed_include_trace_not_all_potential_headers",
                    projected_native_record=projected,
                    injected_diagnostic_header=str(control),
                    injected_header_sha256=file_digest(control),
                    diagnostic_flags=injected,
                    device_kernel_control_observed=status.startswith("analyzed_")
                    and task.stage == "hip-device-analyzer",
                ),
                directory,
            )
            checkpoint("in_progress")
            if (
                value.get("interrupted")
                or value.get("timed_out")
                or value.get("limit_exceeded")
                or value.get("capture_error")
                or status == "unsupported_toolchain"
            ):
                break
        for path, digest in witnesses.items():
            if file_digest(path) != digest:
                raise ValueError("analysis input changed during the session: " + path)
        for path, digest in retained.items():
            if file_digest(path) != digest:
                raise ValueError("retained analysis artifact changed during the session: " + path)
        if (
            sum(path.stat().st_size for path in owned_artifact_files(out, out.parent))
            > plan["budgets"]["max_artifact_bytes"]
        ):
            raise ValueError("analysis artifact budget exhausted during receipt export")
        if compiler_environment(child) != initial_environment:
            raise ValueError("analysis compiler environment changed during the session")
        failures = any(
            row["status"] not in {"analyzed_clean", "analyzed_findings", "syntax_only"}
            for row in rows
        )
        findings = any(row["status"] == "analyzed_findings" for row in rows)
        interrupted = any(row["process"].get("interrupted") for row in rows)
        return checkpoint(
            "interrupted"
            if interrupted
            else "incomplete_or_unsupported"
            if failures or len(rows) != len(tasks)
            else "findings_require_triage"
            if findings
            else "syntax_only_not_analyzer_coverage"
            if plan["lane"] == "hip-syntax"
            else "selected_static_contexts_clean"
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        importlib.metadata.PackageNotFoundError,
    ) as error:
        return checkpoint("blocked_or_incomplete", str(error))
    except KeyboardInterrupt:
        return checkpoint("interrupted", "owned child stopped; no complete analysis verdict")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--verify-receipt", type=Path)
    parser.add_argument("--lane", choices=("host", "hip-syntax", "hip-analyzer"), default="host")
    parser.add_argument("--clang-tidy", type=Path)
    parser.add_argument("--tool-bin", type=Path)
    parser.add_argument(
        "--changed-paths-json",
        help="bounded JSON path list; TU-only edits narrow contexts, other edits use the full lane",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--advisory", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--total-seconds", type=float, default=3600)
    parser.add_argument("--max-artifact-bytes", type=int, default=1024**3)
    parser.add_argument("--minimum-free-bytes", type=int, default=5 * 1024**3)
    args = parser.parse_args(argv)
    try:
        if args.verify_receipt is not None:
            if (
                args.execute
                or args.prepare
                or args.build_dir is not None
                or args.out_dir is not None
                or args.changed_paths_json is not None
            ):
                raise ValueError(
                    "receipt verification does not configure or execute an analysis session"
                )
            result = verify_receipt(args.verify_receipt)
            print(
                json.dumps(
                    dict(
                        receipt_integrity="verified",
                        status=result["status"],
                        hardware_qualification=False,
                    ),
                    indent=2,
                )
            )
            return 0
        if args.build_dir is None or args.out_dir is None:
            raise ValueError("analysis planning requires --build-dir and --out-dir")
        plan = build_plan(
            args.build_dir,
            args.out_dir,
            lane=args.lane,
            tool=args.clang_tidy,
            tool_bin=args.tool_bin,
            timeout_seconds=args.timeout_seconds,
            total_seconds=args.total_seconds,
            max_artifact_bytes=args.max_artifact_bytes,
            minimum_free_bytes=args.minimum_free_bytes,
            changed_paths=(
                json.loads(args.changed_paths_json)
                if args.changed_paths_json is not None
                and len(args.changed_paths_json.encode("utf-8")) <= 16384
                else None
            ),
        )
        if not args.execute:
            print(json.dumps(plan, indent=2))
            return 0
        result = execute(plan, prepare=args.prepare, advisory=args.advisory)
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "results"},
                indent=2,
            )
        )
        if result["status"] == "interrupted":
            return 130
        accepted = {
            "selected_static_contexts_clean",
            "syntax_only_not_analyzer_coverage",
        }
        if args.advisory:
            accepted.add("findings_require_triage")
        return 0 if result["status"] in accepted else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
