"""Inspect exact generated symbols using existing ISA/resource extraction helpers."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from json_evidence import parse_evidence_json

from generate_native_wmma import Variant, generate
from gpu_isa_report import (
    code_object_metadata,
    merge_resource_metadata,
    readobj_for_objdump,
    scan_disassembly,
)
from isa_common import (
    IsaToolConfig,
    device_function_symbols,
    device_function_sizes,
    matrix_symbol_reports,
    extracted_device_code_object,
    sibling_tool,
    forbidden_mnemonic_lines,
    FORBIDDEN_DIVIDE_RE,
    FORBIDDEN_INT32_GLOBAL_STORE_RE,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_generated_source(directory, root=None):
    manifest = json.loads((directory / "native_variants.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "rns8_generated_native_wmma_v1":
        raise ValueError("unknown generated family manifest")
    variants = [Variant(**entry["spec"]) for entry in manifest["variants"]]
    # Regenerate from current implementation/proof dependencies; comparing the
    # complete artifacts rejects spec, certificate, header and helper-source drift.
    with tempfile.TemporaryDirectory() as scratch:
        regenerated = Path(scratch)
        expected = generate(regenerated, variants, root)
        if manifest != expected:
            raise ValueError("generated manifest no longer matches its source/proof dependencies")
        for name in ("native_variants.hip", "native_variants.hpp"):
            if (directory / name).read_bytes() != (regenerated / name).read_bytes():
                raise ValueError("generated source/header drift")
    return manifest


def load_bound_inspection(path, manifest, executable_sha256):
    """Read diagnostic claims only for the exact executable being collected.

    A separate .o inspection cannot qualify a linked binary by association.
    Existing extraction supports .hip_fatbin sections in host inputs; inspect
    the launched executable itself when requesting this join.
    """
    with path.open("rb") as stream:
        raw = stream.read(8 * 1024**2 + 1)
    if len(raw) > 8 * 1024**2:
        raise ValueError("native inspection report exceeds byte budget")
    report = parse_evidence_json(raw.decode("utf-8"))
    from native_wmma_lowering import fingerprint

    if (
        not isinstance(report, dict)
        or report.get("schema") != "rns8_generated_native_isa_v2"
        or report.get("generated_manifest_sha256") != fingerprint(manifest)
        or report.get("generated_source_sha256") != manifest["source_sha256"]
        or report.get("generated_header_sha256") != manifest["header_sha256"]
        or report.get("checker_sha256") != sha(Path(__file__).read_bytes())
        or not isinstance(report.get("artifact"), dict)
        or report["artifact"].get("host_object_sha256") != executable_sha256
        or type(report.get("inspection_passed")) is not bool
        or report.get("device_differential_executed") is not False
        or report.get("performance_qualified") is not False
        or report.get("promotion_eligible") is not False
    ):
        raise ValueError("native inspection source/checker/executable binding mismatch")
    expected = {entry["id"]: entry for entry in manifest["variants"]}
    rows, seen = report.get("symbols"), set()
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("native inspection must cover the exact generated family")
    for row in rows:
        if (
            not isinstance(row, dict)
            or type(row.get("id")) is not str
            or row["id"] not in expected
            or row["id"] in seen
            or row.get("symbol") != expected[row["id"]]["kernel_symbol"]
            or report.get("target") != expected[row["id"]]["spec"]["target"]
        ):
            raise ValueError("native inspection variant/symbol/target binding mismatch")
        seen.add(row["id"])
    failures = report.get("inspection_failures")
    if (
        not isinstance(failures, list)
        or len(failures) > 10000
        or any(type(item) is not str or not item for item in failures)
        or report["inspection_passed"] is not (not failures)
    ):
        raise ValueError("native inspection verdict contradicts failures")
    return dict(
        path=str(path.resolve()),
        sha256=sha(raw),
        report=report,
        source_current=True,
        executable_bound=True,
        independently_replayed_during_ingestion=False,
        promotion_eligible=False,
    )


def inspect(config, directory, logical_report=None, compile_receipt=None):
    manifest = verify_generated_source(directory)
    from check_native_wmma_logic import load_bound_report, logical_inventory

    logical = load_bound_report(directory, logical_report) if logical_report else None
    logical_coverage = logical["inventory"] if logical else logical_inventory(manifest, None)
    logical_rows = {row["id"]: row for row in logical_coverage["variants"]}
    from compile_receipt import load_compile_cost

    compile_cost = (
        load_compile_cost(
            compile_receipt,
            source=directory / "native_variants.hip",
            host_object=config.host_object,
            target=config.target,
        )
        if compile_receipt
        else dict(status="not_recorded", seconds=None)
    )
    if any(entry["spec"]["target"] != config.target for entry in manifest["variants"]):
        raise ValueError("inspection target does not match generated target")
    reports, failures = [], []
    object_hash = sha(config.host_object.read_bytes())
    owned_functions = set()
    sizes = {}
    with extracted_device_code_object(config, "rns8-native-variants-", "native.fatbin") as obj:
        symbols = set(device_function_symbols(config.objdump, obj))
        sizes = device_function_sizes(config.objdump, obj)
        metadata = code_object_metadata(readobj_for_objdump(config.objdump), obj)
        artifact = {
            "host_object_sha256": object_hash,
            "host_object_bytes": config.host_object.stat().st_size,
            "code_object_sha256": sha(obj.read_bytes()),
            "code_object_bytes": obj.stat().st_size,
        }
        selected = [
            entry["kernel_symbol"]
            for entry in manifest["variants"]
            if entry["kernel_symbol"] in symbols
        ]
        attributed = (
            {
                row["symbol"]: row
                for row in matrix_symbol_reports(
                    config.objdump, obj, config.target, selected, ["v_wmma_i32_16x16x16_iu8"]
                )
            }
            if selected
            else {}
        )
        for entry in manifest["variants"]:
            symbol = entry["kernel_symbol"]
            if symbol not in symbols:
                failures.append(symbol + ":missing_symbol")
                continue
            attribution = attributed[symbol]
            functions = attribution["functions"]
            owned_functions.update(function["symbol"] for function in functions)
            disassembly = "\n".join(function["disassembly"] for function in functions)
            counts = merge_resource_metadata(scan_disassembly(disassembly), metadata.get(symbol))
            failures.extend(symbol + ":" + reason for reason in attribution["attribution_failures"])
            if not attribution["matrix_count"]:
                failures.append(symbol + ":missing_integer_wmma")
            resources = (
                "wavefront_size",
                "lds_bytes",
                "vgpr_count",
                "sgpr_count",
                "scratch_bytes",
                "vgpr_spill_count",
                "sgpr_spill_count",
                "max_flat_workgroup_size",
            )
            if (
                counts.get("wavefront_size") != 32
                or counts.get("lds_bytes") != 0
                or any(type(counts.get(key)) is not int or counts[key] < 0 for key in resources)
                or counts.get("max_flat_workgroup_size", 0) < 32
                or counts.get("vgpr_count", 0) < 1
                or counts.get("sgpr_count", 0) < 1
            ):
                failures.append(symbol + ":missing_or_incompatible_resource_metadata")
            extents = [sizes.get(function["symbol"]) for function in functions]
            size_complete = bool(extents) and all(
                item is not None and item["bytes"] > 0 for item in extents
            )
            if not size_complete:
                failures.append(symbol + ":missing_or_zero_native_function_extent")
            stores = forbidden_mnemonic_lines(disassembly, FORBIDDEN_INT32_GLOBAL_STORE_RE)
            if stores:
                failures.append(symbol + ":unexpected_global_int32_store")
            reports.append(
                {
                    "id": entry["id"],
                    "symbol": symbol,
                    "counts": counts,
                    "disassembly_sha256": sha(disassembly.encode()),
                    "attributed_matrix_instructions": attribution["matrix_count"],
                    "reachable_code_bytes": sum(item["bytes"] for item in extents)
                    if size_complete
                    else None,
                    "instruction_scope": "selected_entry_and_resolved_reachable_functions_once_each",
                    "logical_memory_coverage": {
                        key: value
                        for key, value in logical_rows[entry["id"]].items()
                        if key != "configurations"
                    }
                    if entry["id"] in logical_rows
                    else None,
                    "functions": [
                        {key: value for key, value in function.items() if key != "disassembly"}
                        for function in functions
                    ],
                    "global_int32_store_diagnostics": stores,
                    "remainder_diagnostics": forbidden_mnemonic_lines(
                        disassembly, FORBIDDEN_DIVIDE_RE
                    ),
                }
            )
        if sha(config.host_object.read_bytes()) != object_hash:
            raise RuntimeError("generated host object changed during native inspection")
    all_sizes_known = len(reports) == len(manifest["variants"]) and all(
        row["reachable_code_bytes"] is not None for row in reports
    )
    from native_wmma_lowering import fingerprint

    return {
        "schema": "rns8_generated_native_isa_v2",
        "target": config.target,
        "artifact": artifact,
        "generated_source_sha256": manifest["source_sha256"],
        "generated_header_sha256": manifest["header_sha256"],
        "generated_manifest_sha256": fingerprint(manifest),
        "symbols": reports,
        "logical_assurance": {
            key: value for key, value in logical.items() if key not in ("report", "inventory")
        }
        if logical
        else None,
        "family_inventory": {
            "generated_variants": len(manifest["variants"]),
            "inspected_variants": len(reports),
            "unique_reachable_functions": len(owned_functions),
            "unique_reachable_code_bytes": sum(sizes[name]["bytes"] for name in owned_functions)
            if all_sizes_known
            else None,
            "compile_cost": compile_cost,
            "logical_memory_coverage": logical_coverage,
        },
        "inspection_failures": failures,
        "inspection_passed": not failures,
        "checker_sha256": sha(Path(__file__).read_bytes()),
        "device_differential_executed": False,
        "performance_qualified": False,
        "promotion_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--logical-report", type=Path)
    parser.add_argument("--compile-receipt", type=Path)
    parser.add_argument("--hipcc", type=Path)
    parser.add_argument("--scratch-root", type=Path, default=Path("temp"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = IsaToolConfig(
        args.object,
        args.target,
        sibling_tool(args.hipcc, "llvm-objcopy"),
        sibling_tool(args.hipcc, "llvm-objdump"),
        sibling_tool(args.hipcc, "clang-offload-bundler"),
        args.scratch_root,
    )
    result = inspect(config, args.generated_dir, args.logical_report, args.compile_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("Generated native WMMA ISA:", "PASS" if result["inspection_passed"] else "FAIL")
    return 0 if result["inspection_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
