"""Source-bound scalar proof inventory and strict pinned-CBMC result admission."""

from __future__ import annotations

from pathlib import Path
import math
import re

from execution_inventory import bounded_read
from json_evidence import parse_evidence_json, same_evidence_value

ROOT = Path(__file__).resolve().parents[1]
VERSION = "6.8.0"
REVISION = "cdee49cb1a32c6d6703cebf6ae67161977264ad4"
SCHEMA = "rns8_scalar_bitvector_pilot_v1"
SOURCES = [
    f"https://github.com/diffblue/cbmc/blob/{REVISION}/" + path
    for path in (
        "src/goto-programs/show_properties.cpp",
        "src/goto-programs/json_goto_trace.cpp",
        "src/goto-programs/json_expr.cpp",
        "src/goto-checker/report_util.cpp",
        "doc/cprover-manual/api.md",
    )
]
HARNESS = "tests/proofs/scalar_arithmetic.cpp"
HEADERS = (
    "src/backend_common/checked_arithmetic.hpp",
    "src/backend_common/finite_u8_reducer.hpp",
)
FAMILIES = {
    "checked": (1, ("success_iff_fits", "exact_value", "failure_preserves_output")),
    "reducer": (
        2,
        (
            "reciprocal_construction",
            "one_correction_precondition",
            "exact_modulo",
            "canonical_interval",
        ),
    ),
    "native": (
        3,
        (
            "unsigned_modulo",
            "unsigned_interval",
            "radix",
            "signed_bias",
            "unsigned_center",
            "signed_center",
            "centered_interval",
        ),
    ),
}
SAFETY = (
    "--bounds-check",
    "--pointer-check",
    "--pointer-primitive-check",
    "--div-by-zero-check",
    "--signed-overflow-check",
    "--undefined-shift-check",
    "--unwind",
    "258",
    "--unwinding-assertions",
)


def labels(family: str) -> set[str]:
    return {"rns8.bv." + family + "." + name for name in FAMILIES[family][1]}


def command_prefix(selected: dict, model: str, family: str, source: Path) -> list[str]:
    return [
        selected["tool_invocations"]["cbmc"],
        "--cpp11",
        "--" + model,
        "--function",
        "main",
        "-DRNS8_BV_CBMC=1",
        f"-DRNS8_BV_FAMILY={FAMILIES[family][0]}",
        f"-DRNS8_BV_LLP64={int(model == 'LLP64')}",
        "-I",
        str(source / "src"),
    ]


def proof_command(
    selected: dict, name: str, model: str, command: list[str]
) -> tuple[list[str], Path]:
    result = Path(selected["output"]) / (name + "-" + model + ".child-resources.json")
    return [
        selected["proof_python_invocation"],
        str(ROOT / "tools/scalar_proof_worker.py"),
        "--result",
        str(result),
        "--",
        *command,
        "--trace",
        "--all-properties",
    ], result


def resource_observation(
    document: dict, process: dict, command: list[str], *, selected: dict
) -> dict:
    if (
        not isinstance(document, dict)
        or set(document)
        != {
            "schema",
            "command",
            "os",
            "process_id",
            "returncode",
            "elapsed_seconds",
            "elapsed_scope",
            "status",
            "raw_peak",
            "peak_memory_bytes",
            "resource_error",
            "source",
            "units",
            "scope",
            "worker_python_executable",
            "worker_python_version",
        }
        or document.get("schema") != "rns8_scalar_proof_child_resources_v1"
        or type(document.get("os")) is not str
        or document["os"] != selected["native_host"]["os"]
        or document.get("worker_python_executable") != selected["proof_python"]["path"]
        or document.get("worker_python_version") != selected["proof_python_version"]
        or not complete_process(process, (0, 10))
        or not same_evidence_value(document.get("command"), command)
        or not same_evidence_value(document.get("returncode"), process.get("returncode"))
        or type(document.get("process_id")) is not int
        or not 0 < document["process_id"] < 2**32
        or type(process.get("process_id")) is not int
        or document["process_id"] == process["process_id"]
        or document.get("elapsed_scope") != "owned_child_spawn_to_terminal_resource_observation"
    ):
        raise ValueError("owned verifier child command/PID/termination resource binding differs")
    expected = {
        "Linux": (
            "linux_wait4_ru_maxrss",
            "KiB",
            "waited_child_and_accounted_descendants_maxrss_not_concurrent_tree_total",
            1024,
        ),
        "Windows": (
            "windows_K32GetProcessMemoryInfo_PeakWorkingSetSize",
            "bytes",
            "single_owned_child_peak_working_set_excludes_descendants",
            1,
        ),
    }.get(document.get("os"))
    if (
        expected is None
        or (document["source"], document["units"], document["scope"]) != expected[:3]
    ):
        raise ValueError("unknown or mislabeled platform child resource observation")
    elapsed, outer = document["elapsed_seconds"], process.get("elapsed_seconds")
    if (
        any(
            type(value) not in (int, float) or not 0 <= value <= 86400 or not math.isfinite(value)
            for value in (elapsed, outer)
        )
        or elapsed > outer + 0.01
    ):
        raise ValueError("child resource interval is not bound by its complete process capture")
    if document["status"] == "observed":
        if (
            type(document["raw_peak"]) is not int
            or not 0 < document["raw_peak"] < 2**54
            or type(document["peak_memory_bytes"]) is not int
            or document["peak_memory_bytes"] != document["raw_peak"] * expected[3]
            or document["resource_error"] is not None
        ):
            raise ValueError("invalid child peak-memory units/value observation")
    elif document["status"] == "not_observed":
        if (
            document["peak_memory_bytes"] is not None
            or not (
                document["raw_peak"] is None
                or type(document["raw_peak"]) is int
                and document["raw_peak"] == 0
            )
            or type(document["resource_error"]) is not str
            or not document["resource_error"]
        ):
            raise ValueError("unavailable child resources cannot claim a peak")
    else:
        raise ValueError("unknown child resource observation status")
    return {
        key: document[key]
        for key in (
            "os",
            "status",
            "source",
            "units",
            "scope",
            "raw_peak",
            "peak_memory_bytes",
            "resource_error",
            "process_id",
            "elapsed_seconds",
            "elapsed_scope",
        )
    }


def bind_resources(
    facts: dict, document: dict | None, process: dict, command: list[str], *, selected: dict
) -> dict:
    try:
        return resource_observation(document, process, command, selected=selected)
    except (ValueError, TypeError) as error:
        facts.update(verdict="unknown", blocker="child_resource_receipt_unbound: " + str(error))
        return dict(status="not_observed", peak_memory_bytes=None, blocker=str(error))


def registry() -> dict:
    value = parse_evidence_json(bounded_read(ROOT / "metadata/scalar-proof-pilot.json")[0])
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "cbmc_version", "cbmc_revision", "seeds"}
        or value["schema"] != SCHEMA
        or value["cbmc_version"] != VERSION
        or value["cbmc_revision"] != REVISION
        or not isinstance(value["seeds"], list)
        or not 1 <= len(value["seeds"]) <= 16
    ):
        raise ValueError("reviewed pinned scalar proof registry required")
    seen = set()
    for seed in value["seeds"]:
        if (
            not isinstance(seed, dict)
            or set(seed)
            != {
                "id",
                "family",
                "source",
                "source_sha256",
                "anchor",
                "before",
                "after",
                "intended_properties",
            }
            or not isinstance(seed.get("id"), str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", seed["id"])
            or seed["id"] in seen
            or seed.get("family") not in FAMILIES
            or seed.get("source") not in HEADERS
            or not isinstance(seed.get("source_sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", seed["source_sha256"])
            or any(
                type(seed.get(key)) is not str or not seed[key] or len(seed[key]) > 4096
                for key in ("anchor", "before", "after")
            )
            or not isinstance(seed["intended_properties"], list)
            or not seed["intended_properties"]
            or any(
                type(item) is not str or item not in labels(seed["family"])
                for item in seed["intended_properties"]
            )
            or len(set(seed["intended_properties"])) != len(seed["intended_properties"])
        ):
            raise ValueError("invalid or duplicate source-bound scalar mutation")
        seen.add(seed["id"])
    return value


def complete_process(result: dict, codes=(0,)) -> bool:
    return (
        type(result.get("returncode")) is int
        and result["returncode"] in codes
        and result.get("timed_out") is False
        and result.get("limit_exceeded") is None
        and result.get("capture_error") is None
        and (result.get("interrupted") is None or result.get("interrupted") is False)
    )


def json_stdout(result: dict):
    value = result.get("stdout")
    if type(value) is not str or not value or len(value) > 32 * 1024**2:
        raise ValueError("bounded verifier/replay JSON text required")
    try:
        return parse_evidence_json(value)
    except (TypeError, RecursionError) as error:
        raise ValueError("malformed verifier/replay JSON text") from error


def messages(result: dict, codes=(0,)) -> list[dict]:
    if not complete_process(result, codes):
        raise ValueError("incomplete or unsuccessful verifier process")
    value = json_stdout(result)
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 100_000
        or any(not isinstance(row, dict) for row in value)
    ):
        raise ValueError("bounded pinned-CBMC JSON message array required")
    if any(row.get("messageType") == "ERROR" for row in value):
        raise ValueError("verifier emitted an error diagnostic")
    return value


def property_inventory(result: dict, family: str) -> dict:
    documents = messages(result)
    inventories = [row["properties"] for row in documents if "properties" in row]
    if (
        len(inventories) != 1
        or not isinstance(inventories[0], list)
        or not 1 <= len(inventories[0]) <= 100_000
    ):
        raise ValueError("one complete verifier property inventory required")
    found = {}
    for row in inventories[0]:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("name"), str)
            or not row["name"]
            or row["name"] in found
            or any(
                type(row.get(key)) is not str or not row[key]
                for key in ("class", "description", "expression")
            )
            or not isinstance(row.get("sourceLocation"), dict)
            or not row["sourceLocation"].get("file")
        ):
            raise ValueError("incomplete, duplicate or unlocated verifier property")
        found[row["name"]] = row
    actual = [
        row["description"] for row in found.values() if row["description"].startswith("rns8.bv.")
    ]
    if len(actual) != len(set(actual)) or set(actual) != labels(family):
        raise ValueError("scalar proof obligations are missing or duplicated")
    return found


def counterexample(row: dict) -> list[int]:
    trace = row.get("trace")
    if not isinstance(trace, list) or not 1 <= len(trace) <= 100_000:
        raise ValueError("bounded concrete refutation trace required")
    inputs = [
        step.get("values")
        for step in trace
        if isinstance(step, dict)
        and step.get("stepType") == "output"
        and step.get("outputID") == "rns8.bv.inputs"
    ]
    failures = [
        index
        for index, step in enumerate(trace)
        if isinstance(step, dict)
        and step.get("stepType") == "failure"
        and step.get("property") == row["property"]
    ]
    input_positions = [
        index
        for index, step in enumerate(trace)
        if isinstance(step, dict)
        and step.get("stepType") == "output"
        and step.get("outputID") == "rns8.bv.inputs"
    ]
    if (
        len(inputs) != 1
        or len(failures) != 1
        or not input_positions
        or input_positions[0] >= failures[0]
        or not isinstance(inputs[0], list)
        or len(inputs[0]) != 6
    ):
        raise ValueError("refutation has no unique input tuple/intended failure")
    values = []
    for item in inputs[0]:
        if (
            not isinstance(item, dict)
            or item.get("name") != "integer"
            or type(item.get("width")) is not int
            or item["width"] != 64
            or not isinstance(item.get("binary"), str)
            or not re.fullmatch("[01]{64}", item["binary"])
        ):
            raise ValueError("counterexample requires concrete unsigned 64-bit input fields")
        values.append(int(item["binary"], 2))
    return values


def proof_result(result: dict, inventory: dict, family: str, *, intended=()) -> dict:
    """An engine success alone never replaces complete property admission."""
    output = dict(verdict="unknown", blocker=None, properties=[], counterexamples=[])
    try:
        documents = messages(result, (0, 10))
        results = [row["result"] for row in documents if "result" in row]
        statuses = [row["cProverStatus"] for row in documents if "cProverStatus" in row]
        if (
            len(results) != 1
            or not isinstance(results[0], list)
            or not 1 <= len(results[0]) <= 100_000
            or len(statuses) != 1
        ):
            raise ValueError("complete per-property results and one terminal status required")
        seen, failed = set(), []
        for row in results[0]:
            if (
                not isinstance(row, dict)
                or type(row.get("property")) is not str
                or row["property"] in seen
                or row.get("status") not in {"SUCCESS", "FAILURE"}
            ):
                raise ValueError("duplicate, unchecked, unreachable, unknown or malformed property")
            key = row["property"]
            seen.add(key)
            if key in inventory:
                if row.get("description") != inventory[key]["description"]:
                    raise ValueError("property result differs from its discovered obligation")
            elif ".unwind." not in key or row.get("description") != "unwinding assertion":
                raise ValueError("undiscovered verifier property")
            output["properties"].append(
                dict(id=key, description=row["description"], status=row["status"])
            )
            if row["status"] == "FAILURE":
                failed.append(row)
        if not set(inventory) <= seen:
            raise ValueError("verifier omitted discovered properties")
        if not failed:
            if result["returncode"] != 0 or statuses != ["success"]:
                raise ValueError("property success and verifier termination disagree")
            output["verdict"] = "proven"
        else:
            if result["returncode"] != 10 or statuses != ["failure"]:
                raise ValueError("property refutation and verifier termination disagree")
            if any(
                row["property"] not in inventory
                or inventory[row["property"]]["class"] != "assertion"
                for row in failed
            ):
                raise ValueError("machine safety or unwind failure is not an arithmetic refutation")
            relevant = [
                row
                for row in failed
                if row["description"] in (set(intended) if intended else labels(family))
            ]
            if not relevant:
                raise ValueError("no intended arithmetic property was refuted")
            for row in relevant:
                values = counterexample(row)
                if (
                    values[3] > 255
                    or values[4] >= (2 if family == "checked" else 5)
                    or values[5] > 2
                ):
                    raise ValueError("counterexample is outside the stated input domain")
                output["counterexamples"].append(dict(property=row["property"], inputs=values))
            output["verdict"] = "refuted"
    except (ValueError, TypeError, KeyError) as error:
        output["blocker"] = str(error)
    return output


def replay_result(result: dict, family: str, *, intended=()) -> bool:
    expected = not intended
    if not complete_process(result, (0 if expected else 1,)):
        return False
    try:
        value = json_stdout(result)
        if (
            not isinstance(value, dict)
            or set(value) != {"exact_reference_passed", "failed_properties"}
            or value["exact_reference_passed"] is not expected
            or not isinstance(value["failed_properties"], list)
            or any(
                type(item) is not str or item not in labels(family)
                for item in value["failed_properties"]
            )
            or len(set(value["failed_properties"])) != len(value["failed_properties"])
        ):
            return False
        return (
            not value["failed_properties"]
            if expected
            else bool(set(intended) & set(value["failed_properties"]))
        )
    except ValueError:
        return False
