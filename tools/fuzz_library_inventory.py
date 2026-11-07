"""Inspect every configured production fuzz-library object, not a synthetic proxy."""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import re

from analysis_inputs import cache_values, load_inputs, target_artifacts
from execution_inventory import bounded_read
from host_instrumentation_contract import native_argv, object_path
from host_instrumentation_session import reference, version_number
from json_evidence import parse_evidence_json
from quality_inventory import ROOT

TARGETS = (
    "rns8_fuzz_plan_descriptor",
    "rns8_fuzz_export_contract",
    "rns8_fuzz_json_dependency",
)


def fuzz_flags(argv: list[str], *, windows: bool) -> dict:
    # clang-cl forwards /clang: options; examine the effective spelling too.
    flags = [arg.lower().removeprefix("/clang:") for arg in argv[1:]]
    selected = set()
    for arg in flags:
        if arg.startswith("-fsanitize="):
            selected.update(arg.split("=", 1)[1].split(","))
        elif arg == "/fsanitize=address":
            selected.add("address")
        if (
            arg.startswith(
                (
                    "-fno-sanitize",
                    "-fsanitize-ignorelist",
                    "-fsanitize-blacklist",
                    "-fsanitize-recover",
                    "--config",
                    "-config",
                )
            )
            and arg != "-fno-sanitize-recover=undefined"
        ):
            raise ValueError("fuzz object instrumentation is suppressed or overridden")
        if arg.startswith(("-flto", "-fprofile", "-fcoverage", "--coverage")) or arg in {
            "/gl",
            "-mllvm",
        }:
            raise ValueError("fuzz objects cannot share IPO or source-coverage identities")
        if windows and arg in {"/md", "/mdd"}:
            raise ValueError("Windows libFuzzer requires the selected static CRT")
    expected = {"fuzzer-no-link", "address"} | (set() if windows else {"undefined"})
    if selected != expected:
        raise ValueError("production object lacks exact fuzz feedback/sanitizer flags")
    if windows and not {"/mt", "/mtd"}.intersection(flags):
        raise ValueError("Windows fuzz object lacks an explicit static CRT")
    if not windows and "-fno-sanitize-recover=undefined" not in flags:
        raise ValueError("fuzz undefined-behavior checks must be nonrecovering")
    return dict(sanitizers=sorted(selected), configured_flags_verified=True)


def feedback_symbols(text: str) -> bool:
    return bool(re.search(r"\b_?__sanitizer_cov_|\b_?__sancov_", text))


def library_feedback(text: str, sources: list[dict], *, root: Path = ROOT) -> dict:
    """Attribute libFuzzer's symbolized PC-edge observations to retained source.

    These are SanitizerCoverage edges, not LLVM source branch percentages or
    proof that both outcomes of each C++ conditional executed.
    """
    owned, line_counts = {}, {}
    windows = os.name == "nt"
    for source in sources:
        path = Path(source["path"])
        if path.is_relative_to(root / "src"):
            owned[path.as_posix().casefold() if windows else path.as_posix()] = source
    marker = " " + root.as_posix() + "/"
    observed, seen, excluded = [], {}, 0
    pattern = re.compile(r"^COVERED_FUNC: hits: ([0-9]+) edges: ([0-9]+)/([0-9]+) (.+)$")
    for line in text.splitlines():
        match = pattern.fullmatch(line)
        if not match:
            continue
        hits, edges, total = map(int, match.groups()[:3])
        tail = match[4].replace("\\", "/")
        searchable, prefix = (tail.casefold(), marker.casefold()) if windows else (tail, marker)
        position = searchable.find(prefix)
        if position < 0:
            excluded += 1
            continue
        function = tail[:position]
        location = tail[position + 1 :]
        path, separator, number = location.rpartition(":")
        source = owned.get(path.casefold() if windows else path)
        if source is None:
            excluded += 1
            continue
        if source["path"] not in line_counts:
            line_counts[source["path"]] = len(
                Path(source["path"]).read_text(encoding="utf-8").splitlines()
            )
        if (
            not separator
            or not re.fullmatch(r"[0-9]+", number)
            or int(number) <= 0
            or not function
            or not 0 < hits
            or not 0 < edges <= total
            or int(number) > line_counts[source["path"]]
        ):
            raise ValueError("symbolized production fuzz feedback is contradictory")
        row = dict(
            function=function,
            source=source,
            line=int(number),
            hits=hits,
            observed_pc_edges=edges,
            instrumented_pc_edges=total,
        )
        key = function, path, int(number)
        if key in seen and seen[key] != row:
            raise ValueError("conflicting symbolized fuzz function observations")
        if key not in seen:
            seen[key] = row
            observed.append(row)
    nontrivial = [row for row in observed if row["observed_pc_edges"] >= 2]
    if len(nontrivial) < 2 or not any(row["observed_pc_edges"] >= 3 for row in nontrivial):
        raise ValueError("fuzz feedback lacks multiple executed production control-flow edges")
    if not any("rns8_create_plan" in row["function"] for row in nontrivial):
        raise ValueError("fuzz feedback lacks the actual public plan production route")
    return dict(
        scope="symbolized_first_party_library_function_pc_edges_not_source_branch_coverage",
        functions=observed,
        excluded_function_records=excluded,
        source_branch_percentage=None,
    )


def configured_objects(build: Path, *, windows: bool):
    inputs = load_inputs(build)
    cache = cache_values(build / "CMakeCache.txt")
    if cache.get("RNS8_ENABLE_HIP", "OFF").upper() not in {"OFF", "0", "FALSE", "NO"}:
        raise ValueError("fuzz discovery requires a separate CPU-only build")
    metadata = build / ("rns8-fuzz-instrumentation-" + inputs.configuration + ".json")
    document = parse_evidence_json(bounded_read(metadata)[0])
    if (
        not isinstance(document, dict)
        or document.get("schema") != "rns8_fuzz_instrumentation_v1"
        or document.get("configuration") != inputs.configuration
        or document.get("scope") != "configured_not_executed"
        or document.get("gpu_instrumented") is not False
        or document.get("external_dependencies_instrumented") is not False
        or Path(document.get("compiler", "")).resolve() != inputs.compiler
        or not isinstance(document.get("sources"), list)
        or not document["sources"]
        or any(
            type(path) is not str or not Path(path).is_absolute() for path in document["sources"]
        )
    ):
        raise ValueError("configured production fuzz denominator is malformed")
    sources = [Path(path).resolve(strict=True) for path in document["sources"]]
    if len(set(sources)) != len(sources):
        raise ValueError("duplicate production fuzz translation unit")
    rows, excluded, seen = [], [], set()
    for context in inputs.contexts:
        if context.kind != "host":
            raise ValueError("device compilation cannot count as CPU fuzz feedback")
        record = context.native_record()
        argv = native_argv(record, windows=windows)
        path = object_path(record, argv, build, windows=windows)
        if path in seen:
            raise ValueError("configured compilation contexts alias one object")
        seen.add(path)
        parts = path.relative_to(build).parts
        owned = any(
            parts[i : i + 2] == ("CMakeFiles", "rns8_fuzz_library.dir")
            for i in range(len(parts) - 1)
        )
        if not owned:
            excluded.append(
                dict(source=str(context.source), object=str(path), reason="other_target")
            )
            continue
        if (
            "rns8_fuzz_library" not in context.registered_targets
            or context.source not in sources
            or Path(argv[0]).resolve() != inputs.compiler
        ):
            raise ValueError("fuzz object has contradictory source/target/compiler ownership")
        rows.append(dict(context=context, object=path, flags=fuzz_flags(argv, windows=windows)))
    if Counter(row["context"].source for row in rows) != Counter(sources):
        raise ValueError("fuzz commands do not cover every actual production translation unit")
    targets = target_artifacts(
        parse_evidence_json(
            bounded_read(build / ("rns8-analysis-targets-" + inputs.configuration + ".json"))[0]
        ),
        inputs.configuration,
    )
    required = {"rns8_fuzz_library", *TARGETS}
    if not required.issubset(targets) or any(
        not targets[name][0].is_relative_to(build) for name in required
    ):
        raise ValueError("configured fuzz library/entrypoint artifact inventory is incomplete")
    if Path(document.get("library", "")).resolve() != targets["rns8_fuzz_library"][0]:
        raise ValueError("configured fuzz library artifact identity differs")
    return inputs, metadata, rows, targets, excluded


def inspect(session, build: Path, nm: Path, *, windows: bool) -> dict:
    inputs, metadata, rows, targets, excluded = configured_objects(build, windows=windows)
    tools = dict(compiler=reference(inputs.compiler), nm=reference(nm))
    compiler = session.run("fuzz-compiler-version", [str(inputs.compiler), "--version"])
    inspector = session.run("fuzz-nm-version", [str(nm), "--version"])
    if version_number(compiler["stdout"]) != version_number(inspector["stdout"]):
        raise ValueError("fuzz object inspector differs from the actual compiler version")
    objects = []
    for row in rows:
        context, path = row["context"], row["object"]
        before = reference(path)
        symbols = session.run("fuzz-object-" + str(len(objects)), [str(nm), str(path)])
        if before != reference(path):
            raise ValueError("production fuzz object changed during inspection")
        objects.append(
            dict(
                source=reference(context.source),
                object=before,
                context_sha256=context.identity,
                flags=row["flags"],
                feedback_sites_observed=feedback_symbols(symbols["stdout"]),
            )
        )
    if not any(row["feedback_sites_observed"] for row in objects):
        raise ValueError("no actual production library object contains fuzz feedback sites")
    return dict(
        schema="rns8_fuzz_library_objects_v1",
        tools=tools,
        configured_metadata=reference(metadata),
        configured_inputs=[
            dict(path=path, sha256=digest) for path, digest in inputs.artifact_hashes
        ],
        generated_inputs=[
            reference(path)
            for path in sorted({path for row in rows for path in row["context"].generated_inputs})
        ],
        objects=objects,
        objects_without_feedback_sites=[
            row["source"]["path"] for row in objects if not row["feedback_sites_observed"]
        ],
        artifacts={name: reference(targets[name][0]) for name in ("rns8_fuzz_library", *TARGETS)},
        excluded_records=excluded
        + [dict(source=path, reason=reason) for path, reason in inputs.excluded_records],
        scope="all_configured_production_fuzz_objects_not_execution_or_branch_coverage",
        external_dependencies_instrumented=False,
        gpu_instrumented=False,
    )
