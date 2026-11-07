"""Feature-level AMD profiling policy and explicit, bounded read-only preflight.

Planning never executes a command. Preflight uses the existing dependency
resolver and owned-process runner; it never launches an application workload,
GUI, service, driver installer or command that changes permissions/clocks.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import re
import shutil
from typing import Callable, Literal, Mapping

from benchmark_request import file_digest
from check_dependencies_lib.system import find_command
from compiled_binary_identity import canonical_digest
from process_progress import run_capture, capture_succeeded
from session_environment import build_environment, snapshot
from amd_profiler_evidence import EvidenceReader

State = Literal[
    "supported",
    "unsupported",
    "unknown",
    "trace_only",
    "not_run",
    "available",
    "denied",
    "failed",
    "not_required",
]
FEATURES = (
    "timeline",
    "counters",
    "att",
    "att_counters",
    "pc_host_trap",
    "pc_stochastic",
    "compute_profile",
)
CDNA = frozenset(("gfx90a", "gfx942", "gfx950"))
RDNA = frozenset(("gfx1030", "gfx1100", "gfx1200", "gfx1201"))
SUPPORTED_VERSIONS = {"rocprofv3": "1.4.1", "RadeonDeveloperPanelCLI": "1.0.0"}
SCHEMA = "rns8_amd_profiler_capabilities_v1"


@dataclass(frozen=True)
class Feature:
    official_support: State
    advertised: State = "unknown"
    permission: State = "unknown"
    capture: State = "not_run"
    decoded: State = "not_run"
    evidence_bound: State = "not_run"
    reason: str = ""


def support_matrix(system: str, target: str) -> dict[str, Feature]:
    """Conservative documented tuples; installation is a separate observation."""
    result = {
        name: Feature("unsupported", reason="no maintained official HIP tuple") for name in FEATURES
    }
    if system == "Linux" and target in CDNA | RDNA:
        for name in ("timeline", "counters"):
            result[name] = Feature(
                "supported",
                permission="not_required" if name == "timeline" else "unknown",
            )
        if target in CDNA:
            for name in ("att", "att_counters", "compute_profile", "pc_host_trap"):
                result[name] = Feature("supported")
            if target in {"gfx942", "gfx950"}:
                result["pc_stochastic"] = Feature("supported")
        elif target == "gfx1100":
            result["att"] = Feature("trace_only", reason="no Radeon ATT performance counters")
    elif system == "Windows" and target == "gfx1100":
        for name in ("timeline", "counters", "att"):
            result[name] = Feature(
                "supported",
                reason="RDP/RGP driver/tool tuple must be qualified independently",
            )
    return result


def tool_inventory(system: str, resolver: Callable[[str], str | None] = find_command) -> dict:
    names = (
        ("rocprofv3", "rocprofv3-avail", "rocpd", "rocprof-compute")
        if system == "Linux"
        else ("RadeonDeveloperPanelCLI", "RadeonDeveloperServiceCLI")
        if system == "Windows"
        else ()
    )
    result = {}
    for name in names:
        found = resolver(name)
        result[name] = dict(
            path=str(Path(found).resolve()) if found else None,
            located=found is not None,
            version=None,
            version_state="not_run",
        )
    return result


def probe_plan(
    system: str,
    target: str,
    *,
    resolver=find_command,
    agent: int | None = None,
    counters: tuple[str, ...] = (),
) -> dict:
    if system not in ("Windows", "Linux") or target not in CDNA | RDNA:
        raise ValueError("explicit maintained native OS/architecture tuple required")
    if agent is not None and (type(agent) is not int or agent < 0):
        raise ValueError("profiler agent selector must be a nonnegative integer")
    if (
        len(counters) > 8
        or len(set(counters)) != len(counters)
        or any(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None for name in counters)
    ):
        raise ValueError("request at most eight unique explicit counter names")
    if counters and agent is None:
        raise ValueError("counter collectability requires an explicit profiler agent selector")
    tools = tool_inventory(system, resolver)
    commands = []

    def add(name, tool, *args):
        if tools[tool]["located"]:
            commands.append(
                dict(
                    name=name,
                    tool=tool,
                    argv=[tools[tool]["path"], *args],
                    timeout_seconds=30,
                )
            )

    if system == "Linux":
        add("version", "rocprofv3", "--version")
        add("help", "rocprofv3", "--help")
        add("avail-help", "rocprofv3-avail", "--help")
        # Node/agent indices are retained as their own namespace. They are
        # never interpreted as HIP ordinals, even if their numbers coincide.
        select = ("-d", str(agent)) if agent is not None else ()
        add("agents", "rocprofv3-avail", "info", "--agent", *select)
        add("pmc", "rocprofv3-avail", "info", "--pmc", *select)
        add("pc-sampling", "rocprofv3-avail", "info", "--pc-sampling")
        if counters:
            add("pmc-check", "rocprofv3-avail", "pmc-check", "-d", str(agent), *counters)
        add("converter-help", "rocpd", "convert", "--help")
        add("compute-version", "rocprof-compute", "--version")
    else:
        add("version", "RadeonDeveloperPanelCLI", "--version")
        add("help", "RadeonDeveloperPanelCLI", "--help=profiling")
        # --system-info can attach to a service; do not start one implicitly.
        # Its absence leaves runtime target/driver capability unknown.
    result = dict(
        schema=SCHEMA,
        system=system,
        target=target,
        tools=tools,
        features={name: asdict(value) for name, value in support_matrix(system, target).items()},
        commands=commands,
        profiler_agent_selector=agent,
        agent_namespace="rocprofiler_logical_node_selector_not_HIP_ordinal"
        if agent is not None
        else None,
        requested_counters=list(counters),
        actual_runtime_agent="unknown",
        GPU_work_performed=False,
        qualification="plan_only",
        promotion_eligible=False,
    )
    result["identity_sha256"] = canonical_digest(result)
    return result


def _version(text: str, tool: str) -> str | None:
    # Avoid extracting a driver/build date as a profiler version.
    pattern = (
        r"(?:rocprofv3|rocprofiler(?:-sdk)?)\s*(?:version|v)?\s*:?\s*"
        if tool == "rocprofv3"
        else r"(?:RadeonDeveloperPanelCLI|Radeon Developer Panel(?: CLI)?)\s*(?:version|v)?\s*:?\s*"
    )
    matches = set(re.findall(pattern + r"(\d+\.\d+\.\d+)(?![\w.+-])", text, re.I))
    return next(iter(matches)) if len(matches) == 1 else None


def probe_completed(process: dict, command: dict) -> bool:
    if capture_succeeded(process):
        return True
    # The pinned RDP bundle 3.5.1 CLI has its own 1.0.0 information surface.
    # ParseCommandLine returns false after valid help/version; main returns 1.
    # Admit only those exact information commands and complete recognized text.
    if (
        command["tool"] != "RadeonDeveloperPanelCLI"
        or not capture_succeeded(process, expected_exit_code=1)
        or process.get("stderr") != ""
    ):
        return False
    output = process.get("stdout")
    if not isinstance(output, str):
        return False
    if command["name"] == "version" and command["argv"][1:] == ["--version"]:
        return output.strip() == "RadeonDeveloperPanelCLI 1.0.0"
    return (
        command["name"] == "help"
        and command["argv"][1:] == ["--help=profiling"]
        and output.startswith("RadeonDeveloperPanelCLI - GPU Capture Tool\n")
        and "Usage:" in output
        and all(
            flag in output
            for flag in ("--rgp-auto-capture", "--rgp-capture-mode", "--rgp-render-op-count")
        )
    )


def availability_inventory(output: str) -> list[dict]:
    """Read the SDK 1.4.1 'info --agent' text, without assigning HIP ordinals."""
    agents, current = [], None
    for line in output.splitlines():
        match = re.fullmatch(r"\s*GPU\s*:\s*(\d+)\s*", line)
        if match:
            current = dict(selector=int(match[1]), properties={})
            agents.append(current)
            continue
        field = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.*?)\s*", line)
        if field and current is not None:
            name = field[1].lower()
            if name in current["properties"]:
                raise ValueError("duplicate native availability agent property")
            current["properties"][name] = field[2]
    if not agents or len({item["selector"] for item in agents}) != len(agents):
        raise ValueError("native availability agent inventory is empty or ambiguous")
    for agent in agents:
        name = agent["properties"].get("name")
        if (
            not isinstance(name, str)
            or re.fullmatch(r"gfx[0-9a-f]+(?::[A-Za-z0-9_+-]+)*", name) is None
        ):
            raise ValueError("native availability agent architecture is missing")
        agent["architecture"] = name
        agent["namespace"] = "rocprofiler_logical_node_selector_not_HIP_ordinal"
    return agents


def _apply_availability(result: dict) -> None:
    observations = {row["name"]: row for row in result["observations"]}
    tool = result["tools"].get("rocprofv3", {})
    if result["system"] != "Linux" or tool.get("version_state") != "supported":
        return
    agents = observations.get("agents", {})
    if agents.get("completed"):
        try:
            result["runtime_agent_inventory"] = availability_inventory(agents["result"]["stdout"])
            selected = [
                item
                for item in result["runtime_agent_inventory"]
                if item["selector"] == result["profiler_agent_selector"]
            ]
            if len(selected) == 1 and selected[0]["architecture"].split(":")[0] == result["target"]:
                result["actual_runtime_agent"] = selected[0]
                # This establishes a native profiler target, not a HIP mapping.
                # Dispatch/device association is still required after capture.
                help_row = observations.get("help", {})
                if help_row.get("completed") and all(
                    flag in help_row["result"]["stdout"]
                    for flag in ("--kernel-trace", "--hip-trace", "--output-format")
                ):
                    result["features"]["timeline"]["advertised"] = "available"
        except ValueError as error:
            result["availability_error"] = str(error)
    check = observations.get("pmc-check", {})
    if check and not check.get("completed"):
        message = check["result"].get("stderr", "").lower()
        if any(
            term in message
            for term in (
                "permission denied",
                "operation not permitted",
                "access denied",
            )
        ):
            result["features"]["counters"]["permission"] = "denied"
    pmc = observations.get("pmc", {})
    if (
        isinstance(result["actual_runtime_agent"], dict)
        and pmc.get("completed")
        and check.get("completed")
    ):
        # Counter definitions remain raw vendor text. Descriptions/formulas are
        # not unit definitions and must not become numeric metric heuristics.
        sections = re.split(r"(?m)^\s*GPU\s*:\s*(\d+)\s*$", pmc["result"]["stdout"])
        if len(sections) % 2 == 1:
            selected = [
                sections[index + 1]
                for index in range(1, len(sections), 2)
                if int(sections[index]) == result["profiler_agent_selector"]
            ]
            if len(selected) == 1:
                names = re.findall(
                    r"(?mi)^\s*Counter_Name\s*:\s*([A-Za-z][A-Za-z0-9_]*)\s*$",
                    selected[0],
                )
                if len(names) == len(set(names)) and set(result["requested_counters"]) <= set(
                    names
                ):
                    result["features"]["counters"]["advertised"] = "available"
    result["profiler_to_HIP_device_binding"] = "not_run"


def preflight(
    plan: dict,
    output: Path,
    *,
    environment: Mapping[str, str],
    runner=run_capture,
    minimum_free_bytes: int = 1024**3,
) -> dict:
    if plan.get("schema") != SCHEMA or plan.get("identity_sha256") != canonical_digest(
        {key: value for key, value in plan.items() if key != "identity_sha256"}
    ):
        raise ValueError("unchanged versioned profiler plan required")
    if platform.system() != plan["system"]:
        raise ValueError("preflight must run on the selected native OS")
    if type(minimum_free_bytes) is not int or minimum_free_bytes < 1:
        raise ValueError("positive output capacity requirement needed")
    # Reconstruct the allowed command inventory; a resealed arbitrary argv is
    # not a read-only preflight. Tool paths are checked again before execution.
    paths = {name: row["path"] for name, row in plan["tools"].items()}
    expected = probe_plan(
        plan["system"],
        plan["target"],
        resolver=paths.get,
        agent=plan["profiler_agent_selector"],
        counters=tuple(plan["requested_counters"]),
    )
    if plan != expected:
        raise ValueError("preflight command/policy differs from the maintained plan")
    output.mkdir(parents=True, exist_ok=False)
    result = dict(plan)
    result.pop("identity_sha256")
    result["tools"] = {name: dict(row) for name, row in plan["tools"].items()}
    result["features"] = {name: dict(row) for name, row in plan["features"].items()}
    result["observations"] = []
    result["environment"] = snapshot(environment)
    result["free_bytes"] = shutil.disk_usage(output).free
    result["output_capacity"] = (
        "available" if result["free_bytes"] >= minimum_free_bytes else "denied"
    )
    if result["output_capacity"] == "denied":
        result["qualification"] = "blocked_output_capacity"
    else:
        env = build_environment(environment)
        for command in plan["commands"]:
            path = Path(command["argv"][0]).resolve(strict=True)
            before = file_digest(path)
            observed = runner(
                command["argv"],
                cwd=output,
                env=env,
                label=command["name"],
                timeout_seconds=command["timeout_seconds"],
            )
            artifacts = {}
            for stream in ("stdout", "stderr"):
                raw = output / (command["name"] + "." + stream + ".log")
                raw.write_text(observed[stream], encoding="utf-8")
                artifacts[str(raw.resolve())] = file_digest(raw)
            stable = file_digest(path) == before
            complete = probe_completed(observed, command) and stable
            result["observations"].append(
                dict(
                    name=command["name"],
                    command=command["argv"],
                    result=observed,
                    completed=complete,
                    executable_sha256=before,
                    unchanged_during_probe=stable,
                    artifacts=artifacts,
                )
            )
            if command["name"] == "version":
                tool = result["tools"][command["tool"]]
                version = (
                    _version(observed["stdout"] + "\n" + observed["stderr"], command["tool"])
                    if complete
                    else None
                )
                tool.update(
                    version=version,
                    executable_sha256=before,
                    version_state="supported"
                    if version == SUPPORTED_VERSIONS[command["tool"]]
                    else "unknown"
                    if version is None
                    else "unsupported",
                )
        # Raw availability inventories are retained. A name in help or a zero
        # exit code is deliberately not upgraded to permissions or bound data.
        result["qualification"] = "read_only_preflight_completed_no_capture"
        _apply_availability(result)
    result["identity_sha256"] = canonical_digest(result)
    (output / "capabilities.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def validate_preflight(reference: dict, *, store: EvidenceReader, parent: Path) -> dict:
    """Recompute preflight decisions from retained probe bytes; never execute."""
    path, value = store.json(reference, parent)
    if value.get("schema") != SCHEMA or value.get("identity_sha256") != canonical_digest(
        {key: item for key, item in value.items() if key != "identity_sha256"}
    ):
        raise ValueError("sealed profiler preflight receipt required")
    if value.get("qualification") != "read_only_preflight_completed_no_capture":
        raise ValueError("profiler preflight did not complete its read-only phase")
    tools = value.get("tools")
    if not isinstance(tools, dict) or any(not isinstance(item, dict) for item in tools.values()):
        raise ValueError("profiler tool inventory is malformed")
    plan = probe_plan(
        value["system"],
        value["target"],
        resolver=lambda name: tools.get(name, {}).get("path"),
        agent=value["profiler_agent_selector"],
        counters=tuple(value["requested_counters"]),
    )
    if value.get("commands") != plan["commands"] or set(tools) != set(plan["tools"]):
        raise ValueError("profiler preflight differs from maintained read-only command policy")
    if (
        value.get("output_capacity") != "available"
        or type(value.get("free_bytes")) is not int
        or value["free_bytes"] < 1
    ):
        raise ValueError("preflight output capacity was not available")
    observed = value.get("observations")
    if not isinstance(observed, list) or len(observed) != len(plan["commands"]):
        raise ValueError("preflight probe inventory is incomplete")
    rebuilt = dict(
        plan,
        observations=observed,
        tools={name: dict(row) for name, row in plan["tools"].items()},
        features={name: dict(row) for name, row in plan["features"].items()},
    )
    executable_hashes = {}
    for command, row in zip(plan["commands"], observed):
        if (
            not isinstance(row, dict)
            or row.get("name") != command["name"]
            or row.get("command") != command["argv"]
        ):
            raise ValueError("profiler preflight probes are reordered or substituted")
        process = row.get("result")
        if not isinstance(process, dict) or process.get("command") != command["argv"]:
            raise ValueError("preflight lacks actual launched process arguments")
        elapsed = process.get("elapsed_seconds")
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("preflight process timing is malformed")
        if type(process.get("timed_out")) is not bool:
            raise ValueError("preflight timeout state is malformed")
        returned = process.get("returncode")
        if returned is not None and type(returned) is not int:
            raise ValueError("preflight exit status is malformed")
        if returned is not None and (
            type(process.get("process_id")) is not int or process["process_id"] < 1
        ):
            raise ValueError("preflight launched process identity is missing")
        executable = Path(command["argv"][0]).resolve(strict=True)
        expected_hash = row.get("executable_sha256")
        store.verify(executable, expected_hash)
        if executable in executable_hashes and executable_hashes[executable] != expected_hash:
            raise ValueError("profiler tool changed between preflight probes")
        executable_hashes[executable] = expected_hash
        if row.get("unchanged_during_probe") is not True:
            raise ValueError("profiler tool stability was not established")
        artifacts = row.get("artifacts")
        expected_paths = {
            str((path.parent / (command["name"] + "." + stream + ".log")).resolve())
            for stream in ("stdout", "stderr")
        }
        if not isinstance(artifacts, dict) or set(artifacts) != expected_paths:
            raise ValueError("preflight raw stream inventory is incomplete")
        for stream in ("stdout", "stderr"):
            raw = (path.parent / (command["name"] + "." + stream + ".log")).resolve(strict=True)
            store.verify(raw, artifacts[str(raw)])
            if (
                not isinstance(process.get(stream), str)
                or raw.read_text(encoding="utf-8") != process[stream]
            ):
                raise ValueError(
                    "preflight stream summary differs from its retained process output"
                )
        complete = probe_completed(process, command)
        if row.get("completed") is not complete:
            raise ValueError("preflight completion label contradicts its process status")
        if command["name"] == "version":
            name = command["tool"]
            version = (
                _version(process["stdout"] + "\n" + process["stderr"], name) if complete else None
            )
            rebuilt["tools"][name].update(
                version=version,
                executable_sha256=expected_hash,
                version_state="supported"
                if version == SUPPORTED_VERSIONS[name]
                else "unknown"
                if version is None
                else "unsupported",
            )
    _apply_availability(rebuilt)
    if rebuilt["tools"] != tools or rebuilt["features"] != value.get("features"):
        raise ValueError("preflight tool/feature summary differs from raw probe evidence")
    for key in (
        "actual_runtime_agent",
        "runtime_agent_inventory",
        "availability_error",
        "profiler_to_HIP_device_binding",
    ):
        if rebuilt.get(key) != value.get(key):
            raise ValueError("preflight native agent summary differs from retained probe output")
    return dict(
        source=str(path),
        identity_sha256=value["identity_sha256"],
        system=value["system"],
        target=value["target"],
        tools=tools,
        features=value["features"],
        actual_runtime_agent=value.get("actual_runtime_agent"),
        profiler_agent_selector=value["profiler_agent_selector"],
        requested_counters=value["requested_counters"],
        tool_hashes={str(key): item for key, item in executable_hashes.items()},
        scope="read_only_feature_inventory_not_capture_permission_or_device_binding",
        promotion_eligible=False,
    )


def main() -> int:
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("Windows", "Linux"), default=platform.system())
    parser.add_argument("--target", required=True, choices=sorted(CDNA | RDNA))
    parser.add_argument("--agent", type=int)
    parser.add_argument("--counter", action="append", default=[])
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="run bounded read-only tool probes, never a workload",
    )
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    try:
        plan = probe_plan(args.system, args.target, agent=args.agent, counters=tuple(args.counter))
        if args.preflight:
            if args.out_dir is None:
                parser.error("--preflight requires a fresh --out-dir")
            plan = preflight(plan, args.out_dir, environment=os.environ)
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
        return 2 if plan["qualification"].startswith("blocked") else 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
