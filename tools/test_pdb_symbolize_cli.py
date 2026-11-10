"""Native helper rejection and optional real test-only optimized PDB/line lookup."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from benchmark_request import file_digest
from json_evidence import parse_evidence_json
from process_progress import run_capture, capture_succeeded
from pe_symbol_identity import symbol_pair, native_image
from etl_source_contract import decode_lookup

ROOT = Path(__file__).resolve().parents[1]


def positive_control(binary, control, pdb, dbghelp, path):
    pair = symbol_pair(control, pdb)
    library = native_image(dbghelp)
    if library["machine"] != pair["executable"]["machine"]:
        raise ValueError("native PDB control and DbgHelp architecture differ")
    process = run_capture(
        [str(control), "--describe"],
        cwd=ROOT,
        label="pdb-test-only-control",
        timeout_seconds=15,
        max_output_bytes=16384,
        return_on_interrupt=True,
    )
    if not capture_succeeded(process):
        raise ValueError("test-only optimized PDB control did not describe itself")
    value = parse_evidence_json(process["stdout"])
    source = ROOT / "tools/rns8_pdb_symbol_control.cpp"
    if (
        not isinstance(value, dict)
        or value.get("schema") != "rns8_native_pdb_source_control_v1"
        or type(value.get("process_id")) is not int
        or value["process_id"] != process["process_id"]
        or type(value.get("pointer_bytes")) is not int
        or value["pointer_bytes"] != 8
        or value.get("source_sha256") != file_digest(source)
        or value.get("test_only") is not True
        or value.get("benchmark_executed") is not False
        or type(value.get("image_size")) is not int
        or value["image_size"] != pair["executable"]["image_size"]
        or type(value.get("rva")) is not int
        or not 0 < value["rva"] < value["image_size"]
        or any(type(value.get(k)) is not int for k in ("first_line", "last_line"))
        or not 0 < value["first_line"] < value["last_line"]
        or value.get("symbol_name") != "rns8_pdb_symbol_control"
        or type(value.get("source_file")) is not str
        or Path(value["source_file"]).resolve(strict=True) != source
    ):
        raise ValueError("test-only optimized PDB control source/process/image identity differs")
    request = dict(
        pdb=pair["symbols"]["path"],
        dbghelp=library["path"],
        image_size=value["image_size"],
        rvas=[value["rva"]],
    )
    path.write_text(json.dumps(request), encoding="utf-8")
    process = run_capture(
        [str(binary), "--input", str(path)],
        cwd=ROOT,
        label="pdb-real-local-line-control",
        timeout_seconds=15,
        max_output_bytes=1024**2,
        return_on_interrupt=True,
    )
    if not capture_succeeded(process):
        raise ValueError("native local PDB positive lookup did not complete")
    rows = decode_lookup(
        parse_evidence_json(process["stdout"]),
        request=request,
        pair=pair,
        process_id=process["process_id"],
        source_sha256=file_digest(ROOT / "tools/rns8_pdb_symbolize.cpp"),
    )
    row = rows[value["rva"]]
    if (
        row["symbol"] is None
        or value["symbol_name"] not in row["symbol"]["name"]
        or row["source"] is None
        or Path(row["source"]["path"]).resolve(strict=True) != source
        or not value["first_line"] <= row["source"]["line"] <= value["last_line"]
    ):
        raise ValueError(
            "optimized test control did not resolve its actual native PDB symbol/source interval"
        )
    for file in (pair["executable"], pair["symbols"], library):
        if file_digest(file["path"]) != file["sha256"]:
            raise ValueError("native PDB positive control input changed during lookup")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--control-binary", type=Path)
    parser.add_argument("--control-pdb", type=Path)
    parser.add_argument("--dbghelp", type=Path)
    args = parser.parse_args(argv)
    binary = args.binary.resolve(strict=True)
    if (args.control_binary is None) != (args.control_pdb is None):
        raise ValueError("native PDB control needs both executable and matching linker PDB")
    if args.dbghelp is not None and args.control_binary is None:
        raise ValueError("DbgHelp selection requires the native PDB positive control")
    before = file_digest(binary)
    result = run_capture(
        [str(binary), "--build-identity"],
        cwd=ROOT,
        label="pdb-helper-identity",
        timeout_seconds=15,
        max_output_bytes=16384,
        return_on_interrupt=True,
    )
    if not capture_succeeded(result):
        raise ValueError("PDB helper identity query failed")
    identity = parse_evidence_json(result["stdout"])
    if (
        identity.get("schema") != "rns8_pdb_symbolizer_identity_v1"
        or type(identity.get("process_id")) is not int
        or identity["process_id"] != result["process_id"]
        or type(identity.get("pointer_bytes")) is not int
        or identity["pointer_bytes"] != 8
        or identity.get("source_sha256") != file_digest(ROOT / "tools/rns8_pdb_symbolize.cpp")
    ):
        raise ValueError("PDB helper compiled source/ABI identity differs")
    (ROOT / "temp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pdb-cli-", dir=ROOT / "temp") as directory:
        path = Path(directory) / "request.json"
        for rva in (-1, True, 1, 2**64):
            path.write_text(
                json.dumps(
                    dict(pdb="missing.pdb", dbghelp="missing.dll", image_size=1, rvas=[rva])
                ),
                encoding="utf-8",
            )
            result = run_capture(
                [str(binary), "--input", str(path)],
                cwd=ROOT,
                label="pdb-helper-invalid-rva",
                timeout_seconds=15,
                max_output_bytes=16384,
                return_on_interrupt=True,
            )
            # The intended RVA guard must precede any path canonicalization or DLL loading.
            if (
                type(result.get("returncode")) is not int
                or result["returncode"] != 2
                or not capture_succeeded(dict(result, returncode=0))
                or result["stdout"]
                or "RVA outside the explicit image span" not in result["stderr"]
            ):
                raise ValueError("invalid RVA did not reach the native pre-load guard")
        if args.control_binary is not None:
            positive_control(
                binary,
                args.control_binary.resolve(strict=True),
                args.control_pdb.resolve(strict=True),
                args.dbghelp
                or Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/dbghelp.dll",
                path,
            )
    if file_digest(binary) != before:
        raise ValueError("PDB helper changed during CLI checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
