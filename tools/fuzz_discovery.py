"""Plan or explicitly run bounded CPU libFuzzer regression and discovery."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import time

from fuzz_cases import MAX_BYTES, admitted, encode, seeds
from fuzz_library_inventory import TARGETS, inspect, library_feedback
from host_instrumentation_session import (
    ROOT,
    Session,
    check_bookends,
    child_environment,
    reference,
    source_inventory,
    version_number,
    write_json,
)


class DiscoverySession(Session):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deadline = time.monotonic() + self.timeout

    def run(self, name, command, **kwargs):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("aggregate fuzz session deadline exhausted; no retry")
        kwargs["timeout"] = min(kwargs.get("timeout", remaining), remaining)
        return super().run(name, command, **kwargs)


def corpus_inputs():
    arithmetic = []
    counts = Counter()
    for name, case in seeds():
        raw = encode(case)
        valid = admitted(case)
        counts["admitted" if valid else "range_rejected"] += 1
        arithmetic.append((name, raw))
    descriptors = [("empty", b"")]
    descriptors.extend(
        (
            f"per-tile-semantic-{semantic}-invalid-{invalid}",
            bytes([semantic, 0, 0, 0]) + bytes(8) + bytes([1]) + bytes(8) + bytes([invalid]),
        )
        for semantic in range(2)
        for invalid in range(15)
    )
    descriptors.extend(
        [
            (
                f"semantic-{semantic}-invalid-{invalid}",
                bytes([semantic, 0, 0, 0])
                + (bytes(9) if semantic < 2 else bytes(1) if semantic == 2 else b"")
                + bytes([invalid]),
            )
            for semantic in range(6)
            for invalid in range(15)
        ]
    )
    # Valid descriptor prefix followed by the explicit bounded R8H1 history.
    # Keep the original descriptor-only seeds as independent regression inputs.
    history = b"R8H1" + bytes([3, 4, 9, 0, 5, 3, 6, 5, 3, 7, 8, 4, 5, 3, 10, 11, 1, 10, 11])
    descriptors.extend(
        (
            f"history-semantic-{semantic}",
            bytes([semantic, 0, 0, 0])
            + (bytes(9) if semantic < 2 else bytes(1) if semantic == 2 else b"")
            + bytes([0])
            + history,
        )
        for semantic in range(6)
    )
    metadata = [
        ("empty-object", b"{}"),
        ("object", b'{"schema":"unknown","m":1,"n":1,"k":1}'),
        ("wrong-type", b'{"schema":[],"m":{},"n":null,"k":true}'),
        ("array", b"[]"),
        ("null", b"null"),
        ("malformed", b"{"),
    ]
    if not counts["admitted"] or not arithmetic or not descriptors:
        raise ValueError("nonempty semantic regression controls required")
    return dict(zip(TARGETS, (descriptors, arithmetic, metadata))), dict(counts)


def discovery_stats(text: str) -> dict:
    matches = re.findall(r"^stat::number_of_executed_units:\s*([0-9]+)\s*$", text, re.M)
    coverage = re.findall(r"^#[0-9]+\s+DONE\s+cov:\s*([0-9]+)\s+ft:\s*([0-9]+)", text, re.M)
    if len(matches) != 1 or int(matches[0]) <= 0 or not coverage or int(coverage[-1][0]) <= 0:
        raise ValueError("libFuzzer lacks a completed nonempty discovery/feedback report")
    return dict(
        executed_units=int(matches[0]),
        feedback_edges=int(coverage[-1][0]),
        features=int(coverage[-1][1]),
    )


def execute(args, plan):
    out = args.out.resolve()
    if out.exists():
        raise ValueError("fuzz output must be fresh")
    environment = child_environment(
        dict(os.environ),
        args.symbolizer,
        "windows-address" if os.name == "nt" else "address-undefined",
    )
    out.mkdir(parents=True)
    session = DiscoverySession(out, environment, args.seconds, args.artifact_mib * 1024**2)
    report = dict(
        schema="rns8_cpu_fuzz_session_v1",
        state="incomplete",
        plan=plan,
        targets=[],
        blockers=[],
        hardware_qualification=False,
        production_latency=False,
        promotion_eligible=False,
    )
    interrupted = False
    try:
        report["source_inventory"] = source_inventory()
        report["instrumentation"] = inspect(
            session, args.build.resolve(), args.nm.resolve(), windows=os.name == "nt"
        )
        instrumentation = report["instrumentation"]
        symbolizer = session.run("fuzz-symbolizer-version", [str(args.symbolizer), "--version"])
        nm_version = session.run("fuzz-nm-bookend-version", [str(args.nm), "--version"])
        if version_number(symbolizer["stdout"]) != version_number(nm_version["stdout"]):
            raise ValueError("fuzz symbolizer must match the object inspector/compiler")
        report["symbolizer"] = reference(args.symbolizer)
        corpus, report["arithmetic_seed_counts"] = corpus_inputs()
        for name in TARGETS:
            path = Path(instrumentation["artifacts"][name]["path"])
            root = out / name
            originals, discovered, crashes = (
                root / part for part in ("seeds", "discovered", "crashes")
            )
            for directory in (originals, discovered, crashes):
                directory.mkdir(parents=True)
            references = []
            for seed, raw in corpus[name]:
                destination = originals / seed
                with destination.open("xb") as stream:
                    stream.write(raw)
                references.append(reference(destination))
            row = dict(
                target=name,
                seeds=references,
                deterministic_replay=[],
                discovery=None,
                reachability="third_party_json_only"
                if name == TARGETS[2]
                else "first_party_cpu_library",
            )
            report["targets"].append(row)
            common = [
                str(path),
                "-seed=20261106",
                "-timeout=10",
                "-rss_limit_mb=" + str(args.rss_mib),
                "-malloc_limit_mb=256",
                "-max_len=" + str(MAX_BYTES if name == TARGETS[1] else 4096),
                "-artifact_prefix=" + str(crashes) + os.sep,
            ]
            # File operands replay every named input without coverage-based corpus pruning.
            for start in range(0, len(references), 24):
                selected = references[start : start + 24]
                session.run(
                    name + "-replay-" + str(start), [*common, *[item["path"] for item in selected]]
                )
                row["deterministic_replay"].extend(item["sha256"] for item in selected)
            result = session.run(
                name + "-discovery",
                [
                    *common,
                    "-runs=" + str(args.runs),
                    "-max_total_time=" + str(args.discovery_seconds),
                    "-print_final_stats=1",
                    "-print_coverage=1",
                    str(discovered),
                    str(originals),
                ],
            )
            row["discovery"] = discovery_stats(result["stderr"] + "\n" + result["stdout"])
            if name != TARGETS[2]:
                row["library_feedback"] = library_feedback(
                    result["stderr"] + "\n" + result["stdout"], report["source_inventory"]
                )
            row["discovered_corpus"] = [
                reference(item) for item in sorted(discovered.iterdir()) if item.is_file()
            ]
            if any(crashes.iterdir()):
                raise ValueError(
                    "libFuzzer retained a crash/timeout artifact despite successful exit"
                )
            check_bookends(report)
        if len(report["targets"]) != len(TARGETS):
            raise ValueError("incomplete fuzz target execution inventory")
        check_bookends(report)
        report["state"] = "completed"
    except (KeyboardInterrupt, InterruptedError):
        interrupted = True
        report["blockers"].append("interrupted; no automatic retry")
    except (OSError, ValueError, TypeError, KeyError) as error:
        report["blockers"].append(str(error))
    report["stages"] = session.stages
    try:
        check_bookends(report)
    except (OSError, ValueError) as error:
        report["state"] = "incomplete"
        report["blockers"].append(str(error))
    report["interrupted"] = interrupted
    report["accepted"] = report["state"] == "completed"
    report["library_branch_execution_attribution"] = (
        "symbolized_production_function_pc_edges_required; source_branch_percentage_not_measured"
    )
    try:
        write_json(out / "results.json", report)
    except OSError:
        if not interrupted:
            raise
    print(
        json.dumps(
            dict(
                state=report["state"], blockers=report["blockers"], report=str(out / "results.json")
            )
        )
    )
    return 130 if interrupted else 0 if report["accepted"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build", "nm", "symbolizer", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--discovery-seconds", type=int, default=20)
    parser.add_argument("--rss-mib", type=int, default=2048)
    parser.add_argument("--artifact-mib", type=int, default=512)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.out = args.out.resolve()
        if (
            args.out.exists()
            or args.out == ROOT / "temp"
            or not args.out.is_relative_to(ROOT / "temp")
            or not 1 <= args.seconds <= 3600
            or not 1 <= args.runs <= 100000
            or not 1 <= args.discovery_seconds <= min(300, args.seconds)
            or not 256 <= args.rss_mib <= 8192
            or not 32 <= args.artifact_mib <= 2048
        ):
            raise ValueError("fresh ignored output and bounded CPU discovery budgets required")
        plan = dict(
            schema="rns8_cpu_fuzz_plan_v1",
            build=str(args.build.resolve()),
            targets=list(TARGETS),
            seconds=args.seconds,
            discovery_runs=args.runs,
            discovery_seconds=args.discovery_seconds,
            rss_mib=args.rss_mib,
            artifact_mib=args.artifact_mib,
            seed=20261106,
            execute=args.execute,
            gpu_enabled=False,
            stop_on_first_failure=True,
            automatic_shrinking=False,
            production_latency=False,
        )
        if not args.execute:
            print(json.dumps(plan, indent=2))
            return 0
        return execute(args, plan)
    except (OSError, ValueError) as error:
        print("CPU fuzz session: " + str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
