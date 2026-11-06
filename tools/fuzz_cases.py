#!/usr/bin/env python3
"""Expand named arithmetic-case-v1 seeds into an ignored, reproducible corpus."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
HEADER = struct.Struct("<4sBBBBIBBBB H 6x")
MAX_VALUES = 262144
MAX_BYTES = 24 + 8 * MAX_VALUES


@dataclass(frozen=True)
class Case:
    semantic: int
    m: int
    n: int
    k: int
    a: tuple[int, ...]
    b: tuple[int, ...]
    operations: int = 3
    padding_a: int = 1
    padding_b: int = 3
    padding_c: int = 2
    prefix: int = 20
    modulus: int = 0


def encode(case: Case) -> bytes:
    dimensions = (case.m, case.n, case.k)
    if any(type(value) is not int for value in dimensions) or not (
        1 <= case.m <= 8 and 1 <= case.n <= 8 and 1 <= case.k <= 65537
    ):
        raise ValueError("bounded arithmetic dimensions required")
    if case.m * case.n * case.k > 262144 or (case.m + case.n) * case.k > MAX_VALUES:
        raise ValueError("arithmetic operation/input budget exceeded")
    if len(case.a) != case.m * case.k or len(case.b) != case.k * case.n:
        raise ValueError("arithmetic value counts differ from dimensions")
    if (
        type(case.semantic) is not int
        or not 0 <= case.semantic <= 5
        or type(case.operations) is not int
        or not 0 <= case.operations <= 7
    ):
        raise ValueError("unknown arithmetic semantics or operation bits")
    if any(
        type(value) is not int or not 0 <= value <= 7
        for value in (case.padding_a, case.padding_b, case.padding_c)
    ):
        raise ValueError("bounded row-major padding required")
    if type(case.prefix) is not int or type(case.modulus) is not int:
        raise ValueError("explicit integer prefix/modulus required")
    if case.semantic in (2, 3):
        if case.prefix or case.operations & 4:
            raise ValueError("finite/wrap cases have no RNS prefix or prepared-RNS operation")
    elif not 1 <= case.prefix <= 20:
        raise ValueError("RNS prefix must be in [1,20]")
    if (case.semantic == 2 and not 2 <= case.modulus <= 256) or (
        case.semantic != 2 and case.modulus != 0
    ):
        raise ValueError("explicit finite ring modulus required only for finite semantics")
    minimum = -(1 << 63) if case.semantic in (0, 4) else 0
    maximum = (
        (1 << 63) - 1 if case.semantic in (0, 4) else 255 if case.semantic == 2 else (1 << 64) - 1
    )
    if any(
        type(value) is not int or not minimum <= value <= maximum for value in (*case.a, *case.b)
    ):
        raise ValueError("arithmetic value does not fit its explicit input domain")
    header = HEADER.pack(
        b"R8F1",
        case.semantic,
        case.operations,
        case.m,
        case.n,
        case.k,
        case.padding_a,
        case.padding_b,
        case.padding_c,
        case.prefix,
        case.modulus,
    )
    values = b"".join(struct.pack("<Q", value & ((1 << 64) - 1)) for value in (*case.a, *case.b))
    return header + values


def decode(raw: bytes) -> Case:
    if type(raw) is not bytes or not 24 <= len(raw) <= MAX_BYTES:
        raise ValueError("arithmetic input byte budget exceeded")
    magic, semantic, operations, m, n, k, pa, pb, pc, prefix, modulus = HEADER.unpack_from(raw)
    if magic != b"R8F1" or raw[18:24] != bytes(6):
        raise ValueError("unsupported arithmetic case version or reserved bytes")
    count = (m + n) * k
    if count > MAX_VALUES or len(raw) != 24 + 8 * count:
        raise ValueError("arithmetic byte count differs from dimensions")
    values = tuple(item[0] for item in struct.iter_unpack("<Q", raw[24:]))
    if semantic in (0, 4):
        values = tuple(value if value < 1 << 63 else value - (1 << 64) for value in values)
    case = Case(
        semantic, m, n, k, values[: m * k], values[m * k :], operations, pa, pb, pc, prefix, modulus
    )
    # One encoder owns cross-language admission rules; it also rejects bool/int
    # confusion on programmatic inputs. Serialized roundtrip is exact.
    if encode(case) != raw:
        raise ValueError("noncanonical arithmetic case")
    return case


def expected(case: Case) -> tuple[int, ...]:
    """Independent Python integer oracle; no production arithmetic imports."""
    encode(case)
    result = []
    for row in range(case.m):
        for col in range(case.n):
            value = sum(case.a[row * case.k + x] * case.b[x * case.n + col] for x in range(case.k))
            if case.semantic == 2:
                value %= case.modulus
            elif case.semantic == 3:
                value %= 1 << 64
            result.append(value)
    return tuple(result)


def admitted(case: Case) -> bool:
    """Independent integer envelope for valid replay, including wide input packs."""
    cells = expected(case)  # Also validates the complete bounded wire contract.
    if case.semantic in (2, 3):
        return True
    signed = case.semantic in (0, 4)
    if case.semantic < 4 and any(
        not (-(1 << 63) if signed else 0) <= cell <= ((1 << 63) - 1 if signed else (1 << 64) - 1)
        for cell in cells
    ):
        return False
    moduli = (
        256,
        255,
        253,
        251,
        247,
        239,
        233,
        229,
        227,
        223,
        217,
        211,
        199,
        197,
        193,
        191,
        181,
        179,
        173,
        167,
    )
    product = 1
    for modulus in moduli[: case.prefix]:
        product *= modulus
    factor = 2 if signed else 1
    if case.semantic >= 4:
        max_a, max_b = max(map(abs, case.a)), max(map(abs, case.b))
        return factor * max(max_a, max_b, case.k * max_a * max_b) < product
    return factor * max(map(abs, cells)) < product


def seeds(*, large_k=False):
    for semantic in range(6):
        ring = semantic in (2, 3)
        yield (
            f"semantic-{semantic}-tiny",
            Case(
                semantic,
                1,
                1,
                1,
                (1,),
                (1,),
                operations=3 if ring else 7,
                prefix=0 if ring else 20,
                modulus=256 if semantic == 2 else 0,
            ),
        )
    for semantic in (0, 1, 4, 5):
        for prefix in range(1, 21):
            yield (
                f"semantic-{semantic}-prefix-{prefix}",
                Case(semantic, 1, 1, 1, (1,), (1,), prefix=prefix),
            )
    for modulus in range(2, 257):
        yield (
            f"finite-q{modulus}",
            Case(2, 1, 3, 1, (255,), (0, 128, 255), prefix=0, modulus=modulus),
        )
    for value in (-(1 << 63), -(1 << 63) + 1, -1, 0, 1, (1 << 63) - 2, (1 << 63) - 1):
        name = "negative-" + str(-value) if value < 0 else str(value)
        yield "signed-boundary-" + name, Case(0, 1, 1, 1, (value,), (1,))
        yield "wide-signed-" + name, Case(4, 1, 1, 1, (value,), (value,))
    for value in (0, 1, (1 << 63) - 1, 1 << 63, (1 << 64) - 2, (1 << 64) - 1):
        for semantic in (1, 3, 5):
            yield (
                f"unsigned-{semantic}-{value}",
                Case(
                    semantic,
                    1,
                    1,
                    1,
                    (value,),
                    (1 if semantic == 1 else value,),
                    prefix=0 if semantic == 3 else 20,
                ),
            )
    yield (
        "signed-cancellation",
        Case(0, 1, 1, 4, (1 << 62, -(1 << 62), 17, -17), (1, 1, 1, 1), operations=7),
    )
    yield (
        "rectangular-padding",
        Case(
            0,
            2,
            3,
            2,
            (1, -2, -3, 4),
            (3, -4, 5, -6, 7, -8),
            operations=7,
            padding_a=7,
            padding_b=5,
            padding_c=3,
        ),
    )
    if large_k:
        for k in (65535, 65536, 65537):
            for semantic in (0, 2, 3, 4):
                ring = semantic in (2, 3)
                yield (
                    f"semantic-{semantic}-k{k}",
                    Case(
                        semantic,
                        1,
                        1,
                        k,
                        (1,) * k,
                        (1,) * k,
                        prefix=0 if ring else 20,
                        modulus=256 if semantic == 2 else 0,
                    ),
                )


def expand(directory: Path, *, large_k=False) -> dict:
    directory = directory.resolve()
    if directory.exists() or not any(
        directory.is_relative_to(ROOT / root) and directory != ROOT / root
        for root in ("temp", "build", "out")
    ):
        raise ValueError("corpus expansion requires a fresh ignored repository directory")
    manifest_path = directory.parent / (directory.name + ".manifest.json")
    if manifest_path.exists():
        raise ValueError("corpus manifest already exists")
    directory.mkdir(parents=True)
    rows = []
    for name, case in seeds(large_k=large_k):
        raw = encode(case)
        path = directory / (name + ".r8f")
        with path.open("xb") as stream:
            stream.write(raw)
        rows.append(
            dict(
                name=name,
                path=path.name,
                sha256=hashlib.sha256(raw).hexdigest(),
                semantic=case.semantic,
                shape=[case.m, case.n, case.k],
                bytes=len(raw),
            )
        )
    manifest = dict(
        schema="rns8_arithmetic_corpus_v1",
        cases=rows,
        large_k=large_k,
        corpus_sha256=hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        execution="not_run",
    )
    # Keep the manifest outside the corpus: libFuzzer treats every regular file
    # in its input directory as a candidate input.
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--large-k", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(expand(args.out, large_k=args.large_k), indent=2))
        return 0
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
