"""Optional native Mull0.37 pilot contracts; no mutation or compiler runs here.

Native schema/status source is mull-project/mull at
04f5c409878d602e1527ffae4c8d21c61a33d8ea, rust/mull-reporters/src/sqlite_reporter.rs
and rust/mull-state/src/execution_status.rs. Elements' Killed bucket is not used:
it also folds unexecuted/invalid states and cannot prove an intended assertion.
"""

from __future__ import annotations

from collections import Counter
from contextlib import closing
from pathlib import Path
import hashlib
import re
import sqlite3
import shlex
import time

from benchmark_request import file_digest

VERSION = "0.37.0"
SOURCES = ("src/backend_common/checked_arithmetic.hpp", "src/core/api_integer_split.cpp")
COLUMNS = (
    "mutant_id",
    "mutator",
    "filename",
    "directory",
    "line_number",
    "column_number",
    "end_line_number",
    "end_column_number",
    "execution_status",
    "exit_status",
    "duration",
    "stdout",
    "stderr",
    "mutation_replacement",
)
STATES = {
    0: "invalid_native_result",
    1: "native_failure_requires_assertion_review",
    2: "survived",
    3: "timeout",
    4: "crashed",
    5: "abnormal_exit",
    6: "not_executed_dry_run",
    7: "not_executed_fail_fast",
    8: "not_covered",
}


def compatibility(compiler_version: str, runner_version: str, plugin: Path):
    compiler = re.search(r"\bclang version ([0-9]+\.[0-9]+\.[0-9]+)\b", compiler_version)
    runner = re.search(r"^mull-runner " + re.escape(VERSION) + r"\s*$", runner_version, re.M)
    plugins = re.findall(r"^\s*(.+?) \(mull ([0-9.]+), LLVM ([0-9.]+)\)\s*$", runner_version, re.M)
    selected = [
        (version, llvm)
        for path, version, llvm in plugins
        if Path(path).is_absolute() and Path(path).resolve() == plugin.resolve()
    ]
    if not compiler or not runner or selected != [(VERSION, compiler.group(1))]:
        raise ValueError(
            "Mull runner/plugin and native Clang need exact observed version/path agreement"
        )
    return dict(
        mull=VERSION,
        compiler_llvm=compiler.group(1),
        plugin=str(plugin.resolve()),
        native_plugin_loading="requires_actual_build",
        qualified=False,
    )


def configuration(source: Path):
    return dict(
        mutators=["cxx_boundary", "cxx_comparison"],
        includePaths=["^" + re.escape(str((source / name).resolve())) + "$" for name in SOURCES],
        excludePaths=["/tests/", "/third_party/"],
        junkDetectionDisabled=False,
        strict=True,
        parallelization=dict(workers=1, executionWorkers=1),
    )


def inspect_installation(session, args):
    from host_instrumentation_session import reference
    from native_mutation_session import ROOT
    from execution_inventory import bounded_read
    from json_evidence import parse_evidence_json

    policy_text, policy = bounded_read(ROOT / "metadata/mull-pilot.json")
    selected = parse_evidence_json(policy_text)
    if (
        not isinstance(selected, dict)
        or selected.get("schema") != "rns8_optional_mull_pilot_policy_v1"
        or selected.get("version") != VERSION
        or selected.get("mutator_scope") != list(SOURCES)
    ):
        raise ValueError("reviewed Mull version/scope policy differs from the pilot")
    tools = {
        key: reference(getattr(args, key)) for key in ("compiler", "mull_runner", "mull_plugin")
    }
    compiler = session.run("mull-clang-version", [str(args.compiler), "--version"])["stdout"]
    runner = session.run("mull-runner-version", [str(args.mull_runner), "--version"])["stdout"]
    return dict(
        tools=tools,
        versions=compatibility(compiler, runner, args.mull_plugin),
        policy=policy,
        reader=reference(ROOT / "tools/mull_pilot.py"),
    )


def collect(session, args, inputs, tests, admission):
    """Reuse the existing snapshot/build/baseline owner for one optional pilot."""
    from native_mutation_session import ROOT, variant
    from host_instrumentation_session import check_bookends, reference, write_json
    from json_evidence import parse_evidence_json
    from process_progress import capture_succeeded

    if any(not re.fullmatch(r"[A-Za-z0-9 _-]+", name) for name in tests):
        raise ValueError("Mull Catch selector needs reviewed literal native test names")
    check_bookends(admission)
    source = session.output / "mull-instrumented/source"
    config = write_json(session.output / "mull-config.json", configuration(source))
    flags = shlex.join(["-fpass-plugin=" + str(args.mull_plugin), "-g", "-grecord-command-line"])
    environment = session.environment
    session.environment = {**environment, "MULL_CONFIG": config["path"]}
    try:
        baseline, _ = variant(
            session, args, inputs, name="mull-instrumented", tests=tests, compiler_flags=flags
        )
        if baseline["classification"] != "baseline_passed":
            raise ValueError("Mull instrumented unchanged baseline is inconclusive")
        database = parse_evidence_json(
            Path(baseline["compilation_inventory"]["path"]).read_text(encoding="utf-8")
        )
        selected_flags = "-fpass-plugin=" + str(args.mull_plugin)
        for row in database:
            arguments = row.get("arguments")
            if arguments is None:
                arguments = shlex.split(row["command"])
            if selected_flags not in arguments or "-grecord-command-line" not in arguments:
                raise ValueError(
                    "actual Mull compilation lacks selected plugin/debug command recording"
                )
        result_dir = session.output / "mull-native"
        result_dir.mkdir()
        binary = baseline["artifacts"]["rns8_tests"]["path"]
        capture = session.run(
            "mull-pilot",
            [
                str(args.mull_runner),
                "--strict",
                "--workers",
                "1",
                "--timeout",
                "10000",
                "--include-not-covered",
                "--allow-surviving",
                "--reporters",
                "SQLite",
                "--report-dir",
                str(result_dir),
                "--report-name",
                "pilot",
                binary,
                ",".join(tests),
            ],
            required=False,
        )
        if not capture_succeeded(capture):
            raise ValueError("Mull pilot did not finish; native partial results remain unqualified")
        sources = {
            str((source / name).resolve()): dict(reference(source / name), relative_path=name)
            for name in SOURCES
        }
        result = read_database(result_dir / "pilot.sqlite", sources)
        result.update(
            instrumented_baseline=baseline,
            config=config,
            admission=admission,
            reader=reference(ROOT / "tools/mull_pilot.py"),
            selected_tests=tests,
            policy=reference(ROOT / "metadata/mull-pilot.json"),
        )
        check_bookends([result, config, admission])
        return result
    finally:
        session.environment = environment


def read_database(path: Path, sources: dict):
    """Read one finalized bounded native report; never turn a crash into a kill."""
    path = path.resolve(strict=True)
    if len(sources) != 2 or {row.get("relative_path") for row in sources.values()} != set(SOURCES):
        raise ValueError("exact two-file native Mull production scope required")
    source_lines = {}
    for name, reference in sources.items():
        source = Path(name)
        if (
            not source.is_absolute()
            or str(source.resolve()) != name
            or str(Path(reference["path"]).resolve()) != name
        ):
            raise ValueError("native Mull source reference is not canonical")
        with source.open("rb") as stream:
            raw = stream.read(16 * 1024**2 + 1)
        if len(raw) > 16 * 1024**2 or hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("native Mull production source drift or byte budget exceeded")
        source_lines[name] = raw.splitlines()
    if not path.is_file() or not 1 <= path.stat().st_size <= 128 * 1024**2:
        raise ValueError("bounded nonempty native Mull database required")
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise ValueError("Mull database must be a finalized standalone snapshot")
    before = file_digest(path)
    deadline = time.monotonic() + 20
    rows, seen, total_logs = [], set(), 0
    try:
        with closing(
            sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True, timeout=1)
        ) as connection:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.enable_load_extension(False)
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 4 * 1024**2)
            connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            tables = connection.execute(
                "SELECT name,type FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            ).fetchmany(4)
            if set(tables) != {("mutant", "table"), ("information", "table")}:
                raise ValueError("unsupported native Mull database objects")
            for table, columns in (("mutant", COLUMNS), ("information", ("key", "value"))):
                info = connection.execute("PRAGMA table_xinfo(" + table + ")").fetchmany(32)
                if tuple(row[1] for row in info) != columns or any(row[6] != 0 for row in info):
                    raise ValueError("native Mull0.37 database schema differs")
            information = connection.execute("SELECT key,value FROM information").fetchmany(65)
            if len(information) > 64 or any(
                type(k) is not str or type(v) is not str or len(k) + len(v) > 65536
                for k, v in information
            ):
                raise ValueError("bounded native Mull producer metadata required")
            if (
                len(dict(information)) != len(information)
                or dict(information).get("Mull Version") != VERSION
            ):
                raise ValueError("native Mull producer version missing, duplicated or unsupported")
            cursor = connection.execute("SELECT " + ",".join(COLUMNS) + " FROM mutant")
            for values in cursor:
                if len(rows) >= 5000:
                    raise ValueError("Mull pilot exceeds5000 native result rows")
                row = dict(zip(COLUMNS, values))
                if (
                    any(
                        type(row[key]) is not str or not row[key]
                        for key in ("mutant_id", "mutator", "filename")
                    )
                    or row["directory"] != ""
                ):
                    raise ValueError("native Mull mutation identity missing")
                if (
                    len(row["mutant_id"]) > 8192
                    or len(row["mutator"]) > 128
                    or len(row["filename"]) > 4096
                ):
                    raise ValueError("native Mull mutation identity exceeds bounds")
                if row["mutant_id"] in seen:
                    raise ValueError("duplicate native Mull mutant identity")
                seen.add(row["mutant_id"])
                source = Path(row["filename"])
                if not source.is_absolute() or str(source.resolve()) not in sources:
                    raise ValueError("native Mull mutant outside the two admitted production files")
                source = source.resolve()
                for key in ("line_number", "column_number", "end_line_number", "end_column_number"):
                    if type(row[key]) is not int or not 1 <= row[key] < 1 << 31:
                        raise ValueError("native Mull source location missing")
                if (row["end_line_number"], row["end_column_number"]) < (
                    row["line_number"],
                    row["column_number"],
                ):
                    raise ValueError("native Mull source span is reversed")
                lines = source_lines[str(source)]
                if any(
                    row[prefix + "line_number"] > len(lines)
                    or row[prefix + "column_number"]
                    > len(lines[row[prefix + "line_number"] - 1]) + 1
                    for prefix in ("", "end_")
                ):
                    raise ValueError("native Mull source span exceeds its bound production bytes")
                status, exit_code, duration = (
                    row[key] for key in ("execution_status", "exit_status", "duration")
                )
                if (
                    type(status) is not int
                    or status not in STATES
                    or type(exit_code) is not int
                    or not -(1 << 31) <= exit_code < 1 << 31
                    or type(duration) is not int
                    or not 0 <= duration < 1 << 63
                ):
                    raise ValueError("native Mull execution status/duration missing")
                if (status == 2 and exit_code != 0) or (status == 1 and exit_code == 0):
                    raise ValueError("native Mull status contradicts exit code")
                for key in ("stdout", "stderr", "mutation_replacement"):
                    if type(row[key]) is not str:
                        raise ValueError("native Mull result text missing")
                    count = len(row[key].encode("utf-8"))
                    if count > 1024**2:
                        raise ValueError("native Mull text exceeds bounded pilot scope")
                    total_logs += count
                if total_logs > 32 * 1024**2:
                    raise ValueError("native Mull retained result text exceeds32MiB")
                row.update(
                    classification=STATES[status],
                    source_sha256=sources[str(source)]["sha256"],
                    intended_assertion_kill=False,
                )
                rows.append(row)
    except sqlite3.Error as error:
        raise ValueError("native Mull database could not be read: " + str(error)) from error
    if not rows:
        raise ValueError("empty native Mull mutation inventory")
    if file_digest(path) != before or any(
        file_digest(Path(name)) != ref["sha256"] for name, ref in sources.items()
    ):
        raise ValueError("native Mull report or production sources changed during review")
    return dict(
        schema="rns8_native_mull_pilot_v1",
        database=dict(path=str(path), sha256=before),
        producer=dict(information),
        results=rows,
        native_duration_unit="milliseconds",
        counts=dict(Counter(row["classification"] for row in rows)),
        scope="two_native_cpu_production_files_no_device_or_test_mutation",
        assertion_strength="requires_review_not_inferred_from_native_failure_bucket",
        qualified=False,
        production_latency=False,
        promotion_eligible=False,
    )
