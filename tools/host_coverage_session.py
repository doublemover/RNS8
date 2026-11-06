"""Plan or explicitly collect native Linux LLVM host source/branch coverage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import sys

from analysis_inputs import cache_values
from compiled_binary_identity import canonical_digest, validate_document
from coverage_contract import llvm_export, mapped_sources
from execution_inventory import bounded_read, discovery
from host_instrumentation_session import (
    MANDATORY,
    Session,
    check_bookends,
    configured_inventory,
    reference,
    source_inventory,
    version_number,
    write_json,
)
from json_evidence import parse_evidence_json
from quality_inventory import ROOT, inventory

REQUIRED_MODULES = (
    "src/core/api_gemm.cpp",
    "src/core/api_pack.cpp",
    "src/core/api_export.cpp",
    "src/core/exact_range.hpp",
)


def profiles(directory: Path) -> list[dict]:
    entries = list(directory.iterdir())
    if not 1 <= len(entries) <= 10000:
        raise ValueError("nonempty bounded raw profile inventory required")
    result = []
    for path in sorted(entries):
        if (
            path.is_symlink()
            or not path.is_file()
            or not re.fullmatch(r"[0-9]+(?:_[0-9]+)?-[0-9]+\.profraw", path.name)
        ):
            raise ValueError("raw profiles need owned native module/process filenames")
        item = reference(path)
        if not 1 <= item["bytes"] <= 128 * 1024**2:
            raise ValueError("empty/oversized raw profile")
        result.append(item)
    if sum(row["bytes"] for row in result) > 2 * 1024**3:
        raise ValueError("raw profile inventory exceeds total budget")
    return result


def execute(args, plan):
    output = Path(plan["output"])
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            ("ASAN_", "UBSAN_", "TSAN_", "LSAN_", "LLVM_PROFILE_", "COVERAGE_", "DEBUGINFOD_")
        )
    }
    if any(
        environment.get(key)
        for key in (
            "LD_PRELOAD",
            "LD_AUDIT",
            "CCC_OVERRIDE_OPTIONS",
            "CLANG_CONFIG_FILE_USER_DIR",
            "CLANG_CONFIG_FILE_SYSTEM_DIR",
        )
    ):
        raise ValueError("unreviewed compiler/runtime injection in coverage lane")
    environment.update(
        PYTHONOPTIMIZE="",
        CTEST_NO_TESTS_ACTION="error",
        DEBUGINFOD_URLS="",
        LLVM_PROFILE_FILE=str(output / "setup-profiles/%m-%p.profraw"),
    )
    output.mkdir(parents=True)
    (output / "setup-profiles").mkdir()
    (output / "test-profiles").mkdir()
    session = Session(output, environment, args.timeout_seconds, args.artifact_mib * 1024**2)
    result = dict(
        schema="rns8_native_host_coverage_session_v1",
        status="blocked",
        plan=plan,
        blockers=[],
        production_latency=False,
        python_branch_coverage="not_run",
        mutation_detection="not_run",
        scope="selected_native_catch_cpu_api_tests_not_all_platforms",
    )
    try:
        result["source_inventory"] = source_inventory()
        result["tools"] = {
            key: reference(getattr(args, key))
            for key in ("cmake", "ctest", "compiler", "nm", "llvm_cov", "llvm_profdata")
        }
        if args.toolchain:
            result["toolchain"] = reference(args.toolchain)
        versions = {
            key: session.run(key + "-version", [str(getattr(args, key)), "--version"])["stdout"]
            for key in result["tools"]
        }
        if (
            len(
                {
                    version_number(versions[key])
                    for key in ("compiler", "nm", "llvm_cov", "llvm_profdata")
                }
            )
            != 1
        ):
            raise ValueError(
                "compiler and native LLVM profile/coverage/object tools must match exactly"
            )
        if versions["cmake"].splitlines()[0].replace("cmake ", "") != versions[
            "ctest"
        ].splitlines()[0].replace("ctest ", ""):
            raise ValueError("matching native CMake/CTest required")
        result["versions"] = versions
        session.run("configure", plan["configure"])
        build = Path(plan["build"])
        cache = cache_values(build / "CMakeCache.txt")
        if (
            cache.get("RNS8_ENABLE_HOST_COVERAGE") != "ON"
            or cache.get("RNS8_HOST_SANITIZER") != "none"
            or cache.get("RNS8_ENABLE_HIP") != "OFF"
            or cache.get("RNS8_ENABLE_CPU_PARALLEL") != "OFF"
        ):
            raise ValueError("separate native CPU coverage configuration required")
        session.run("build", plan["build_command"])
        objects = configured_inventory(session, build, "coverage", args.nm)
        result["object_inventory"] = write_json(output / "objects.json", objects)
        binary = objects["artifacts"]["rns8_tests"]["path"]
        capture, identity = session.json("test-build-identity", [binary, "--build-identity"])
        validate_document(identity)
        if type(identity.get("process_id")) is not int or identity["process_id"] != capture.get(
            "process_id"
        ):
            raise ValueError("coverage identity process mismatch")
        result["compiled_identity"] = identity
        _, native = session.json(
            "candidate-native-test-scope",
            [str(args.ctest), "--test-dir", str(build), "--no-tests=error", "--show-only=json-v1"],
        )
        selected = [
            name
            for name, test in discovery(native).items()
            if Path(test.command[0]).resolve() == Path(binary).resolve()
        ]
        allowed = None
        if args.allowed_skips:
            source, ref = bounded_read(args.allowed_skips)
            allowed = parse_evidence_json(source)
            result["skip_policy"] = ref
        session.environment["LLVM_PROFILE_FILE"] = str(output / "test-profiles/%m-%p.profraw")
        test_ref, tests = session.tests(
            "native-api",
            build,
            args.ctest,
            required=MANDATORY,
            allowed=allowed,
            selection_names=selected,
        )
        result["test_inventory"] = test_ref
        result["test_selection_scope"] = "all_discovered_cases_of_actual_rns8_tests_executable"
        raw = profiles(output / "test-profiles")
        result["raw_profiles"] = raw
        profile_list = output / "profiles.txt"
        if any("\n" in row["path"] or "\r" in row["path"] for row in raw):
            raise ValueError("profile paths cannot contain newlines")
        with profile_list.open("x", encoding="utf-8") as handle:
            handle.write("\n".join(row["path"] for row in raw) + "\n")
        result["merge_inputs"] = reference(profile_list)
        merged = output / "host.profdata"
        session.run(
            "profile-merge",
            [
                str(args.llvm_profdata),
                "merge",
                "--instr",
                "--failure-mode=any",
                "--sparse=false",
                "--num-threads=" + str(args.jobs),
                "--input-files=" + str(profile_list),
                "--output=" + str(merged),
            ],
        )
        result["merged_profile"] = reference(merged)
        artifacts = list(objects["artifacts"].values())
        # Supplying the static archive as well retains compiled but unlinked
        # code in the mapping. Native coverage merges duplicate function records.
        bins = [row["path"] for row in artifacts if row["path"] != binary]
        common = [
            str(binary),
            "--instr-profile=" + str(merged),
            "--check-binary-ids",
            "--debuginfod=false",
            "--num-threads=" + str(args.jobs),
        ]
        for path in bins:
            common.extend(["--object", path])
        policy = parse_evidence_json(
            (ROOT / "metadata/first-party-inventory.json").read_text(encoding="utf-8")
        )
        generated = {
            name
            for name in policy["generated"]
            if Path(name).suffix in {".hpp", ".h", ".inc", ".cpp"}
        }
        source_names = sorted({row.path for row in inventory() if row.role == "cpp"} | generated)
        source_refs = [reference(ROOT / name) for name in source_names]
        mapped = session.run(
            "coverage-mapping-discovery",
            [str(args.llvm_cov), "export", *common, "--summary-only"],
            max_output_bytes=64 * 1024**2,
        )
        if re.search(r"\b(?:warning|error):", mapped["stderr"], re.I):
            raise ValueError("native coverage mapping discovery emitted a warning")
        mapped_document = parse_evidence_json(mapped["stdout"])
        write_json(output / "mapping-discovery.json", mapped_document)
        result["mapping_discovery"] = reference(output / "mapping-discovery.json")
        admitted = {Path(row["path"]) for row in source_refs}
        sources = mapped_sources(mapped_document, admitted)
        export = session.run(
            "coverage-export",
            [str(args.llvm_cov), "export", *common, "--sources", *sources],
            max_output_bytes=128 * 1024**2,
        )
        if re.search(r"\b(?:warning|error):", export["stderr"], re.I):
            raise ValueError("native coverage emitted a mapping/profile warning")
        export_path = output / "llvm-coverage.json"
        with export_path.open("x", encoding="utf-8") as handle:
            handle.write(export["stdout"])
        result["native_export"] = reference(export_path)
        document = parse_evidence_json(export["stdout"])
        summary = llvm_export(
            document,
            root=ROOT,
            sources=source_refs,
            compiled_sources={Path(row["source"]["path"]) for row in objects["objects"]},
            generated=generated,
            required_modules=REQUIRED_MODULES,
        )
        result["coverage_summary"] = write_json(output / "coverage-summary.json", summary)
        session.run(
            "coverage-html",
            [
                str(args.llvm_cov),
                "show",
                *common,
                "--format=html",
                "--output-dir=" + str(output / "html"),
                "--show-branches=count",
                "--show-expansions",
                "--sources",
                *sources,
            ],
        )
        html = [reference(path) for path in sorted((output / "html").rglob("*")) if path.is_file()]
        if not html or not (output / "html/index.html").is_file():
            raise ValueError("native coverage HTML output missing")
        result["html_inventory"] = html
        if profiles(output / "test-profiles") != raw:
            raise ValueError("raw profile inventory changed during merge/export")
        check_bookends([result, objects, tests, summary, session.stages])
        if tests["status"] != "passed":
            result["blockers"].append("native_test_inventory_failed")
        result["blockers"].extend(summary["blockers"])
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
    parser.add_argument("--out", type=Path, required=True)
    for name in ("cmake", "ctest", "compiler", "nm", "llvm-cov", "llvm-profdata"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--toolchain", type=Path)
    parser.add_argument("--allowed-skips", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--artifact-mib", type=int, default=32768)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        output = args.out.resolve()
        if output == ROOT / "temp" or not output.is_relative_to(ROOT / "temp") or output.exists():
            raise ValueError("fresh ignored repository temp child required")
        if (
            not 1 <= args.jobs <= 64
            or not 1 <= args.timeout_seconds <= 14400
            or not 256 <= args.artifact_mib <= 65536
        ):
            raise ValueError("native coverage budget outside supported bounds")
        for name in (
            "cmake",
            "ctest",
            "compiler",
            "nm",
            "llvm_cov",
            "llvm_profdata",
            "toolchain",
            "allowed_skips",
        ):
            path = getattr(args, name)
            if path is not None:
                path = path.resolve(strict=True)
                if not path.is_file():
                    raise ValueError("regular native coverage tool/input required")
                setattr(args, name, path)
        build = output / "build"
        command = [
            str(args.cmake),
            "--preset",
            "cpu-coverage-debug",
            "-S",
            str(ROOT),
            "-B",
            str(build),
            "-DCMAKE_CXX_COMPILER=" + str(args.compiler),
            "-DPython3_EXECUTABLE=" + sys.executable,
            "-DRNS8_BUILD_SHARED=OFF",
            "-DRNS8_BUILD_FUZZERS=OFF",
            "-DRNS8_BUILD_SANITIZER_CONTROLS=OFF",
        ]
        if args.toolchain:
            command.append("-DCMAKE_TOOLCHAIN_FILE=" + str(args.toolchain))
        plan = dict(
            schema="rns8_native_host_coverage_plan_v1",
            output=str(output),
            build=str(build),
            configure=command,
            build_command=[
                str(args.cmake),
                "--build",
                str(build),
                "--parallel",
                str(args.jobs),
                "--verbose",
            ],
            native_catch_corpus_only=True,
            required_modules=list(REQUIRED_MODULES),
            runtime_validation_executed=False,
            production_latency=False,
        )
        if not args.execute:
            print(json.dumps(plan, indent=2))
            return 0
        if platform.system() != "Linux":
            raise ValueError("native Linux Clang coverage required; no Windows/HIP coverage claim")
        return execute(args, plan)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print("host coverage: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
