#!/usr/bin/env python3
"""Validate durable documentation claim boundaries."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import metadata_registry_constants as registry


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PATHS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "backend-notes.md",
    REPO_ROOT / "docs" / "performance-wins.md",
    REPO_ROOT / "docs" / "performance-model.md",
    REPO_ROOT / "docs" / "platform-linux.md",
    REPO_ROOT / "docs" / "platform-readiness.md",
    REPO_ROOT / "docs" / "reviewed-local-evidence.md",
    REPO_ROOT / "docs" / "roadmap-status.md",
]

TARGET_RE = re.compile(r"\b(linux|rocm|instinct|cdna[0-9]*|rdna4)\b", re.IGNORECASE)
PRESENT_READINESS_RE = re.compile(
    r"\b(validated|validation-passed|ready|readiness-passed|production-ready|ship-ready|promoted|shipping)\b",
    re.IGNORECASE,
)
BOUNDARY_RE = re.compile(
    r"\b("
    r"after|before|blocked|boundary|claiming|future|gate|gated|must|no|not|only|pending|planned|provisional|"
    r"remain|remains|require|required|requires|scope|separate|separately|target|unproven|unvalidated|until"
    r")\b",
    re.IGNORECASE,
)
SPEED_RE = re.compile(
    r"\b("
    r"[0-9]+(?:\.[0-9]+)?x|faster|fastest|speedup|speedups|win|winner|winning|wins"
    r")\b",
    re.IGNORECASE,
)
EVIDENCE_RE = re.compile(
    r"\b("
    r"baseline|cache|current|decision|direct hip|evidence|event-valid|experimental|exploratory|gfx1100|"
    r"installed|local|median|not a claim|none|promoted|release|required|reviewed|same-backend|"
    r"same-contract|schema-valid|status|unreviewed"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClaimViolation:
    path: Path
    line_number: int
    reason: str
    text: str

    def format(self) -> str:
        rel = self.path.relative_to(REPO_ROOT) if self.path.is_relative_to(REPO_ROOT) else self.path
        return f"{rel}:{self.line_number}: {self.reason}: {self.text}"


def validate_registry_claim_labels() -> list[str]:
    required_labels = {"windows_gfx1100_local", "linux_rocm_required", "instinct_required"}
    missing = sorted(required_labels.difference(registry.TARGET_CLAIM_LABELS))
    if missing:
        return [f"metadata claim labels missing required target boundary labels: {', '.join(missing)}"]
    return []


def validate_line(path: Path, line_number: int, line: str, file_has_reviewed_local_context: bool) -> list[ClaimViolation]:
    text = line.strip()
    if not text:
        return []
    if text.startswith("#"):
        return []
    violations: list[ClaimViolation] = []
    has_target = TARGET_RE.search(text) is not None
    has_readiness = PRESENT_READINESS_RE.search(text) is not None
    has_boundary = BOUNDARY_RE.search(text) is not None
    if has_target and has_readiness and not has_boundary:
        violations.append(
            ClaimViolation(
                path,
                line_number,
                "non-Windows target readiness claim lacks a required/future/unvalidated boundary",
                text,
            )
        )
    has_speed = SPEED_RE.search(text) is not None
    has_evidence = EVIDENCE_RE.search(text) is not None
    if has_speed and not has_evidence and not file_has_reviewed_local_context:
        violations.append(
            ClaimViolation(
                path,
                line_number,
                "speedup/winner wording lacks reviewed/local/evidence qualifier",
                text,
            )
        )
    if has_target and has_speed and "windows" in text.lower() and not has_boundary and "gfx1100" not in text.lower():
        violations.append(
            ClaimViolation(
                path,
                line_number,
                "Windows speedup wording mentions non-Windows targets without gfx1100/local boundary",
                text,
            )
        )
    return violations


def validate_paths(paths: list[Path]) -> list[ClaimViolation]:
    violations: list[ClaimViolation] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        normalized = text.lower()
        file_has_reviewed_local_context = "reviewed" in normalized and "gfx1100" in normalized
        for line_number, line in enumerate(text.splitlines(), start=1):
            violations.extend(validate_line(path, line_number, line, file_has_reviewed_local_context))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=DEFAULT_PATHS)
    args = parser.parse_args()
    registry_errors = validate_registry_claim_labels()
    violations = validate_paths(args.paths)
    if registry_errors or violations:
        for error in registry_errors:
            print(error)
        for violation in violations:
            print(violation.format())
        return 1
    print("claim validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
