"""Deterministic semantic shrinking for admitted arithmetic-case-v1 failures.

The caller supplies an explicitly selected, bounded replay adapter. This module
never chooses a backend, launches a process, or treats unsupported as a failure.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import time
from typing import Callable

from fuzz_cases import Case, admitted, encode


@dataclass(frozen=True)
class Verdict:
    kind: str
    # A replay adapter supplies a stable semantic category and phase, optionally
    # expected/observed status. Do not include coordinates which shrinking moves.
    signature: tuple[str, ...] = ()

    def __post_init__(self):
        if self.kind not in {"passed", "unsupported", "rejected", "failed"}:
            raise ValueError("unknown arithmetic replay disposition")
        if (
            type(self.signature) is not tuple
            or len(self.signature) > 8
            or any(type(item) is not str or not item or len(item) > 256 for item in self.signature)
        ):
            raise ValueError("bounded immutable semantic failure signature required")
        if (self.kind == "failed") != bool(self.signature):
            raise ValueError("only actual semantic failures carry a failure signature")


@dataclass(frozen=True)
class ShrinkResult:
    original: Case
    minimized: Case
    signature: tuple[str, ...]
    attempts: tuple[dict, ...]
    status: str
    elapsed_seconds: float

    def summary(self) -> dict:
        return dict(
            schema="rns8_arithmetic_shrink_v1",
            status=self.status,
            original_sha256=hashlib.sha256(encode(self.original)).hexdigest(),
            minimized_sha256=hashlib.sha256(encode(self.minimized)).hexdigest(),
            failure_signature=list(self.signature),
            replay_count=len(self.attempts),
            reduction_count=sum(
                row["accepted"] and row["stage"] != "original" for row in self.attempts
            ),
            elapsed_seconds=self.elapsed_seconds,
            attempts=list(self.attempts),
            scope="caller_supplied_replay_verdicts_not_native_execution_attestation",
            production_latency=False,
        )


def _complexity(case: Case) -> tuple:
    # Strictly descending measure prevents cycles, including -1 <-> +1.
    values = (*case.a, *case.b)
    return (
        case.operations.bit_count(),
        case.m * case.n * case.k,
        case.m + case.n + case.k,
        sum(abs(value).bit_length() for value in values),
        sum(value != 0 for value in values),
        sum(value < 0 for value in values),
    )


def _remove(case: Case, axis: str, first: int, end: int) -> Case:
    """Remove matching logical coordinates, never reinterpret physical padding."""
    if axis == "m":
        return replace(
            case, m=case.m - (end - first), a=case.a[: first * case.k] + case.a[end * case.k :]
        )
    if axis == "n":
        return replace(
            case,
            n=case.n - (end - first),
            b=tuple(
                case.b[row * case.n + col]
                for row in range(case.k)
                for col in range(case.n)
                if not first <= col < end
            ),
        )
    return replace(
        case,
        k=case.k - (end - first),
        a=tuple(
            case.a[row * case.k + col]
            for row in range(case.m)
            for col in range(case.k)
            if not first <= col < end
        ),
        b=case.b[: first * case.n] + case.b[end * case.n :],
    )


def minimize(
    original: Case,
    replay: Callable[[Case], Verdict],
    *,
    max_replays: int = 256,
    seconds: float = 60,
    clock: Callable[[], float] = time.monotonic,
) -> ShrinkResult:
    """Keep only smaller admitted cases reproducing the same semantic failure.

    Every candidate is re-encoded, independently range-admitted and re-executed.
    The supplied adapter must enforce its own per-call timeout/output budgets;
    a Python deadline cannot cancel a callback already executing native work.
    Adapter exceptions propagate; they never become accepted semantic failures.
    """
    if type(max_replays) is not int or not 1 <= max_replays <= 4096:
        raise ValueError("replay budget must be in [1,4096]")
    if type(seconds) not in (int, float) or not 0 < seconds <= 3600:
        raise ValueError("finite shrink deadline must be in (0,3600]")
    if type(original) is not Case or type(original.a) is not tuple or type(original.b) is not tuple:
        raise ValueError("immutable arithmetic case and operand tuples required")
    if not admitted(original):
        raise ValueError("semantic shrinking requires a valid arithmetic envelope")
    start = clock()
    best, attempts, seen = original, [], set()
    signature = ()
    exhausted = False

    def attempt(candidate: Case, stage: str, *, baseline=False) -> bool:
        nonlocal best, signature, exhausted
        if len(attempts) >= max_replays or clock() - start >= seconds:
            exhausted = True
            return False
        if not baseline and _complexity(candidate) >= _complexity(best):
            return False
        raw = encode(candidate)
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen:
            return False
        seen.add(digest)
        if not admitted(candidate):
            return False
        result = replay(candidate)
        if type(result) is not Verdict:
            raise TypeError("replay must return an explicit arithmetic Verdict")
        if baseline:
            if result.kind != "failed":
                raise ValueError("original case did not reproduce a semantic failure")
            signature = result.signature
        accepted = result.kind == "failed" and result.signature == signature
        attempts.append(
            dict(
                stage=stage,
                sha256=digest,
                bytes=len(raw),
                shape=[candidate.m, candidate.n, candidate.k],
                operations=candidate.operations,
                disposition=result.kind,
                signature=list(result.signature),
                accepted=accepted,
            )
        )
        if accepted:
            best = candidate
        return accepted

    attempt(original, "original", baseline=True)
    if not signature:
        raise ValueError("deadline expired before the original failure was reproduced")
    changed = True
    while changed and not exhausted:
        before = best
        # Preserve the wire semantics; simplify API history before matrix work.
        for bit in (4, 1, 2):
            if best.operations & bit:
                attempt(replace(best, operations=best.operations & ~bit), "operations")
            if exhausted:
                break
        for axis in ("m", "n", "k"):
            if exhausted:
                break
            # Coarse-to-fine interval deletion; all operands share a K deletion.
            chunk = max(1, getattr(best, axis) // 2)
            while not exhausted:
                first = 0
                while first < getattr(best, axis) and not exhausted:
                    end = min(getattr(best, axis), first + chunk)
                    if end - first < getattr(best, axis) and attempt(
                        _remove(best, axis, first, end), "dimension-" + axis
                    ):
                        first = 0
                    else:
                        first += chunk
                if chunk == 1:
                    break
                chunk = max(1, chunk // 2)
        # Clear value intervals before individual alphabets, avoiding an
        # obligatory O(input values) sequence of native replays for large K.
        for operand in ("a", "b"):
            if exhausted:
                break
            chunk = max(1, len(getattr(best, operand)) // 2)
            while not exhausted:
                for first in range(0, len(getattr(best, operand)), chunk):
                    values = getattr(best, operand)
                    end = min(len(values), first + chunk)
                    if any(values[first:end]):
                        attempt(
                            replace(
                                best,
                                **{operand: values[:first] + (0,) * (end - first) + values[end:]},
                            ),
                            "zeros-" + operand,
                        )
                    if exhausted:
                        break
                if chunk == 1:
                    break
                chunk = max(1, chunk // 2)
            for index in range(len(getattr(best, operand))):
                if exhausted:
                    break
                value = getattr(best, operand)[index]
                half = abs(value) // 2 * (-1 if value < 0 else 1)
                for replacement in dict.fromkeys((0, 1, -1 if value < 0 else 1, half)):
                    values = getattr(best, operand)
                    if replacement != values[index]:
                        attempt(
                            replace(
                                best,
                                **{operand: values[:index] + (replacement,) + values[index + 1 :]},
                            ),
                            "alphabet-" + operand,
                        )
                    if exhausted:
                        break
        changed = best != before
    return ShrinkResult(
        original,
        best,
        signature,
        tuple(attempts),
        "budget_exhausted" if exhausted else "local_fixed_point",
        max(0.0, clock() - start),
    )
