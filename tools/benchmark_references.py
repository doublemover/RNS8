"""Restore or package complete, content-bound integer GEMM reference fixtures.

The header count is the number of canonical uint64 words (cells times limbs).
Keys bind bounded, exact-wide, finite-ring/field and strict-wrap contracts.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "tests/fixtures/native-reference/index.json"
DEFAULT_CACHE = ROOT / "temp/benchmark-references"
LIMIT = 128 * 1024 * 1024


def reference_cache_supported(arguments):
    def option(flag,default):
        return arguments[arguments.index(flag)+1] if flag in arguments else default
    return (all(option(flag,"1")=="1" for flag in ("--residue-chain-length","--host-api-batch-size","--grouped-dispatch")) and
        not any(flag in arguments for flag in ("--native-to-rns-bridge","--vector-to-rns-chain",
            "--force-native-to-rns-bridge","--vector-native-to-rns-chain","--vector-to-rns-chain-host-repack-control",
            "--vector-native-to-rns-chain-host-repack-control","--transient-uniform-small-inputs",
            "--transient-uniform-small-i8-inputs","--residue-channel-fusion","--streaming-overlap","--hip-graph-replay")))


def cached_arguments(arguments,directory):
    if directory is None or not reference_cache_supported(arguments):return list(arguments)
    if any(arg in arguments for arg in ("--reference-cache-dir","--populate-reference-cache")):
        raise ValueError("collector owns its read-only reference cache arguments")
    return [*arguments,"--reference-cache-dir",str(Path(directory).resolve())]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_reference(data, key):
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("reference key must be lowercase SHA256")
    if not 144 <= len(data) <= LIMIT or data[:8] != b"RNS8REF1" or data[8:72] != key.encode("ascii"):
        raise ValueError("reference format or key mismatch")
    cells = struct.unpack_from("<Q", data, 136)[0]
    if not cells or len(data) != 144 + cells * 8 or data[72:136] != digest(data[144:]).encode("ascii"):
        raise ValueError("reference size or payload mismatch")
    return cells


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(index=DEFAULT_INDEX, output=DEFAULT_CACHE):
    index, output = Path(index).resolve(), Path(output).resolve()
    if output.is_relative_to(ROOT) and (output == ROOT or output.relative_to(ROOT).parts[0] not in ("temp", "build", "out")):
        raise ValueError("expanded references must stay outside tracked source directories")
    manifest = json.loads(index.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rns8_native_reference_fixtures_v1" or not isinstance(manifest.get("fixtures"), list):
        raise ValueError("reference fixture manifest schema")
    seen, total, validated = set(), 0, []
    for entry in manifest["fixtures"]:
        key = entry["key"]
        if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key) or key in seen:
            raise ValueError("invalid or duplicate reference key")
        seen.add(key)
        if type(entry["bytes"]) is not int or not 144 <= entry["bytes"] <= LIMIT:
            raise ValueError("reference expansion budget")
        total += entry["bytes"]
        if total > LIMIT:
            raise ValueError("reference fixture set expansion budget")
        compressed_path = index.parent / (key + ".bin.gz")
        if (type(entry["compressed_bytes"]) is not int or not 0 < entry["compressed_bytes"] <= LIMIT or
                compressed_path.stat().st_size != entry["compressed_bytes"]):
            raise ValueError("compressed reference size budget")
        compressed = compressed_path.read_bytes()
        if len(compressed) != entry["compressed_bytes"] or digest(compressed) != entry["compressed_sha256"]:
            raise ValueError("compressed reference digest mismatch")
        # Bound decompression before allocating an untrusted expanded payload.
        import io
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
            data = stream.read(entry["bytes"] + 1)
        if len(data) != entry["bytes"] or digest(data) != entry["sha256"]:
            raise ValueError("expanded reference digest mismatch")
        if validate_reference(data, key) != entry["cells"]:
            raise ValueError("reference cell count mismatch")
        validated.append((output / (key + ".bin"), data))
    # Verify the entire authored set before publishing any expanded file.
    for path, data in validated:
        if not path.is_file() or path.read_bytes() != data:
            atomic_write(path, data)
    return dict(fixtures=len(validated), bytes=total, output=str(output), manifest_sha256=digest(index.read_bytes()))


def pack(raw, output, provenance):
    raw, output = Path(raw), Path(output)
    document = json.loads(Path(provenance).read_text(encoding="utf-8"))
    if document.get("oracle") != "boost_cpp_int_full_integer_reference" or not document.get("cases"):
        raise ValueError("full independent reference generation provenance required")
    rows, total = [], 0
    for case in document["cases"]:
        key = case["key"]
        # Validate keys before using them as filenames.
        if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("invalid reference key")
        path = raw / (key + ".bin")
        if path.stat().st_size > LIMIT:
            raise ValueError("reference size budget")
        data = path.read_bytes()
        cells = validate_reference(data, key)
        total += len(data)
        if total > LIMIT or any(row["key"] == key for row in rows):
            raise ValueError("reference set budget or duplicate key")
        compressed = gzip.compress(data, compresslevel=9, mtime=0)
        atomic_write(output / (key + ".bin.gz"), compressed)
        rows.append(dict(key=key, cells=cells, bytes=len(data), sha256=digest(data),
            compressed_bytes=len(compressed), compressed_sha256=digest(compressed)))
    result = dict(schema="rns8_native_reference_fixtures_v1", format="RNS8REF1 header + canonical little-endian uint64 words, gzip",
        provenance=document, fixtures=rows)
    atomic_write(output / "index.json", (json.dumps(result, indent=2) + "\n").encode())
    return dict(fixtures=len(rows), raw_bytes=total, compressed_bytes=sum(row["compressed_bytes"] for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--pack", type=Path, help="directory of independently generated raw references")
    parser.add_argument("--provenance", type=Path, help="required generation receipt when packaging")
    args = parser.parse_args()
    if args.pack is not None:
        if args.provenance is None:
            parser.error("--pack requires --provenance")
        report = pack(args.pack, args.output, args.provenance)
    else:
        if args.provenance is not None:
            parser.error("--provenance requires --pack")
        report = prepare(args.index, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
