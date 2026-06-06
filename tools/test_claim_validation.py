#!/usr/bin/env python3
"""Self-test durable documentation claim validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import claim_validation


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def write_doc(tmp: Path, name: str, text: str) -> Path:
    path = tmp / name
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    registry_errors = claim_validation.validate_registry_claim_labels()
    expect(not registry_errors, "\n".join(registry_errors))

    current_violations = claim_validation.validate_paths(claim_validation.DEFAULT_PATHS)
    expect(not current_violations, "\n".join(item.format() for item in current_violations))

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        overclaim = write_doc(
            tmp,
            "overclaim.md",
            "Linux ROCm and Instinct are production-ready from a Windows 2.0x speedup.\n",
        )
        violations = claim_validation.validate_paths([overclaim])
        reasons = [item.reason for item in violations]
        expect(
            any("non-Windows target readiness claim" in reason for reason in reasons),
            "expected Linux/Instinct readiness overclaim to be rejected",
        )
        expect(
            any("speedup/winner wording" in reason for reason in reasons),
            "expected unqualified speedup wording to be rejected",
        )

        scoped = write_doc(
            tmp,
            "scoped.md",
            (
                "Windows gfx1100 local reviewed release same-contract evidence shows a 2.0x speedup.\n"
                "Linux ROCm and Instinct readiness require live validation on those platforms.\n"
            ),
        )
        scoped_violations = claim_validation.validate_paths([scoped])
        expect(not scoped_violations, "\n".join(item.format() for item in scoped_violations))

    print("claim validation self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
