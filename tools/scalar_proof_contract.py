"""Source-bound scalar proof inventory and strict pinned-CBMC result admission."""

from __future__ import annotations

from pathlib import Path
import re

from execution_inventory import bounded_read
from json_evidence import parse_evidence_json

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
