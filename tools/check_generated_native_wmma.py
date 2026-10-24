"""Inspect exact generated symbols using existing ISA/resource extraction helpers."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from generate_native_wmma import Variant, generate
from gpu_isa_report import code_object_metadata, merge_resource_metadata, readobj_for_objdump, scan_disassembly
from isa_common import (IsaToolConfig, device_function_symbols, disassemble_code_object,
                        extracted_device_code_object, sibling_tool, forbidden_mnemonic_lines,
                        FORBIDDEN_DIVIDE_RE, FORBIDDEN_INT32_GLOBAL_STORE_RE)


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


def inspect(config, directory):
    manifest = verify_generated_source(directory)
    if any(entry["spec"]["target"] != config.target for entry in manifest["variants"]):
        raise ValueError("inspection target does not match generated target")
    reports, failures = [], []
    with extracted_device_code_object(config, "rns8-native-variants-", "native.fatbin") as obj:
        symbols = set(device_function_symbols(config.objdump, obj))
        metadata = code_object_metadata(readobj_for_objdump(config.objdump), obj)
        artifact = {"host_object_sha256": sha(config.host_object.read_bytes()),
                    "host_object_bytes": config.host_object.stat().st_size,
                    "code_object_sha256": sha(obj.read_bytes()), "code_object_bytes": obj.stat().st_size}
        for entry in manifest["variants"]:
            symbol = entry["kernel_symbol"]
            if symbol not in symbols:
                failures.append(symbol + ":missing_symbol")
                continue
            disassembly = disassemble_code_object(config.objdump, obj, config.target, symbol)
            counts = merge_resource_metadata(scan_disassembly(disassembly), metadata.get(symbol))
            if not counts["wmma_dense_integer"]:
                failures.append(symbol + ":missing_integer_wmma")
            if counts.get("wavefront_size") != 32 or counts.get("lds_bytes") != 0:
                failures.append(symbol + ":missing_or_incompatible_resource_metadata")
            stores = forbidden_mnemonic_lines(disassembly, FORBIDDEN_INT32_GLOBAL_STORE_RE)
            if stores:
                failures.append(symbol + ":unexpected_global_int32_store")
            reports.append({"id": entry["id"], "symbol": symbol, "counts": counts,
                            "disassembly_sha256": sha(disassembly.encode()),
                            "global_int32_store_diagnostics": stores,
                            "remainder_diagnostics": forbidden_mnemonic_lines(disassembly, FORBIDDEN_DIVIDE_RE)})
    return {"schema": "rns8_generated_native_isa_v1", "target": config.target, "artifact": artifact,
            "generated_source_sha256": manifest["source_sha256"], "symbols": reports,
            "inspection_failures": failures, "inspection_passed": not failures,
            "checker_sha256": sha(Path(__file__).read_bytes()),
            "device_differential_executed": False, "performance_qualified": False, "promotion_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--hipcc", type=Path)
    parser.add_argument("--scratch-root", type=Path, default=Path("temp"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = IsaToolConfig(args.object, args.target, sibling_tool(args.hipcc, "llvm-objcopy"),
                           sibling_tool(args.hipcc, "llvm-objdump"), sibling_tool(args.hipcc, "clang-offload-bundler"),
                           args.scratch_root)
    result = inspect(config, args.generated_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("Generated native WMMA ISA:", "PASS" if result["inspection_passed"] else "FAIL")
    return 0 if result["inspection_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
