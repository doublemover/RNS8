"""Division-free one-level Strassen oracle; no GPU or speedup qualification."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class Product:
    a: tuple[int, int, int, int]
    b: tuple[int, int, int, int]
    c: tuple[int, int, int, int]


# Block order 00, 01, 10, 11. Each record owns one transformed product and
# immediately adds it into its declared outputs, so products need not coexist.
STRASSEN = (
    Product((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
    Product((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
    Product((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1)),
    Product((0, 0, 0, 1), (-1, 0, 1, 0), (1, 0, 1, 0)),
    Product((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
    Product((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 1)),
    Product((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),
)


def verify_program(program: tuple[Product, ...]) -> dict:
    if not program or len(program) > 32:
        raise ValueError("bounded nonempty program required")
    if any(len(v) != 4 or any(type(x) is not int for x in v)
           for p in program for v in (p.a, p.b, p.c)):
        raise ValueError("four integer coefficients required per operand/output")
    failures = []
    for c in range(4):
        row, col = divmod(c, 2)
        for a in range(4):
            for b in range(4):
                expected = int(a // 2 == row and b % 2 == col and a % 2 == b // 2)
                actual = sum(p.c[c] * p.a[a] * p.b[b] for p in program)
                if actual != expected:
                    failures.append([c, a, b, actual, expected])
    payload = [[p.a, p.b, p.c] for p in program]
    return {"polynomial_identity_verified": not failures, "failures": failures,
            "sha256": hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest(),
            "division_free": True, "hardware_qualified": False}


def _shape(a: list[list[int]]) -> tuple[int, int]:
    if not a or not a[0] or any(len(row) != len(a[0]) for row in a):
        raise ValueError("nonempty rectangular matrix required")
    if any(type(x) is not int for row in a for x in row):
        raise ValueError("integer elements required")
    return len(a), len(a[0])


def centered(value: int, modulus: int) -> int:
    residue = value % modulus
    return residue - modulus if residue >= (modulus + 1) // 2 else residue


def one_level(a: list[list[int]], b: list[list[int]], modulus: int) -> tuple[list[list[int]], dict]:
    m, k = _shape(a)
    bk, n = _shape(b)
    if k != bk or type(modulus) is not int or not 2 <= modulus <= 256:
        raise ValueError("compatible matrices and byte modulus required")
    tm, tn, tk = (m + 1) // 2, (n + 1) // 2, (k + 1) // 2
    output = [[0] * n for _ in range(m)]

    def transform(source, rows, cols, tile_rows, tile_cols, coefficients):
        transformed = [[0] * tile_cols for _ in range(tile_rows)]
        for i in range(tile_rows):
            for j in range(tile_cols):
                value = 0
                for block, coefficient in enumerate(coefficients):
                    row, col = i + (block // 2) * tile_rows, j + (block % 2) * tile_cols
                    if coefficient and row < rows and col < cols:
                        value += coefficient * centered(source[row][col], modulus)
                # Reduce pre-additions before any signed-byte GEMM, including m=256.
                transformed[i][j] = centered(value, modulus)
        return transformed

    for p in STRASSEN:
        left = transform(a, m, k, tm, tk, p.a)
        right = transform(b, k, n, tk, tn, p.b)
        for i in range(tm):
            for j in range(tn):
                # Exact host reference reduces every term; no inherited GPU K cap.
                value = 0
                for inner in range(tk):
                    value = (value + left[i][inner] * right[inner][j]) % modulus
                for block, coefficient in enumerate(p.c):
                    row, col = i + (block // 2) * tm, j + (block % 2) * tn
                    if coefficient and row < m and col < n:
                        output[row][col] = centered(output[row][col] + coefficient * value, modulus)
    cap = modulus // 2
    return output, {"program": verify_program(STRASSEN), "modulus": modulus,
                    "padded_shape": [2 * tm, 2 * tn, 2 * tk], "subproducts": 7,
                    "max_transform_abs_bound": 2 * cap,
                    "packed_operand_abs_bound": cap,
                    "term_reduction_accumulator_abs_bound": cap * cap + modulus - 1,
                    "max_live_transform_elements": tm * tk + tk * tn,
                    "stored_product_matrices": 0, "runtime_dispatch_eligible": False}
