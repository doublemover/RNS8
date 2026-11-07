"""Native helper identity and pre-load request rejection; no PDB is loaded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from benchmark_request import file_digest
from json_evidence import parse_evidence_json
from process_progress import run_capture, capture_succeeded

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args(argv)
    binary = args.binary.resolve(strict=True)
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
    if file_digest(binary) != before:
        raise ValueError("PDB helper changed during CLI checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
