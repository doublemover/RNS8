from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from benchmark_sweep_lib.cli import load_required_isa_index, review_capture_paths


with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    captures = tmp / "captures"
    captures.mkdir()
    capture = captures / "candidate.json"
    duplicate = captures / "duplicate.json"
    explicit = tmp / "explicit.json"
    failed = captures / "candidate.failed.json"
    review = captures / "review_report.json"
    cache = captures / "candidate-autotune-cache.json"
    isa_sidecar = captures / "backend-gfx1100-ck-isa-summary.json"
    for path in [capture, duplicate, explicit, failed, review, cache, isa_sidecar]:
        path.write_text("{}\n", encoding="utf-8")

    args = argparse.Namespace(
        capture=[explicit, duplicate],
        capture_root=[captures],
        review_only=True,
        out_root=tmp / "unused",
    )
    paths = review_capture_paths(args)
    assert paths == [explicit, duplicate, capture]

    default_root = tmp / "review-root"
    default_capture = default_root / "scenarios" / "default.json"
    default_capture.parent.mkdir(parents=True)
    default_capture.write_text("{}\n", encoding="utf-8")
    default_args = argparse.Namespace(
        capture=[],
        capture_root=[],
        review_only=True,
        out_root=default_root,
    )
    assert review_capture_paths(default_args) == [default_capture]

    non_review_args = argparse.Namespace(
        capture=[],
        capture_root=[],
        review_only=False,
        out_root=default_root,
    )
    assert review_capture_paths(non_review_args) == []

    missing = tmp / "missing-isa"
    try:
        load_required_isa_index([missing])
    except SystemExit as exc:
        assert "--isa-report path does not exist" in str(exc)
    else:
        raise AssertionError("missing --isa-report path should fail")

    empty = tmp / "empty-isa"
    empty.mkdir()
    try:
        load_required_isa_index([empty])
    except SystemExit as exc:
        assert "--isa-report found no *-isa-summary.json reports" in str(exc)
    else:
        raise AssertionError("empty --isa-report directory should fail")

    valid = tmp / "valid-isa" / "ck-gfx1100-ck-isa-summary.json"
    valid.parent.mkdir()
    valid.write_text(
        json.dumps(
            {
                "backend": "ck",
                "target": "gfx1100",
                "instruction_totals": {
                    "matrix_instruction_histogram": {"v_wmma_i32_16x16x16_iu8": 2},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index = load_required_isa_index([valid.parent])
    assert "ck|gfx1100" in index
    assert index["ck|gfx1100"][0]["isa_matrix_instruction_histogram"] == {
        "v_wmma_i32_16x16x16_iu8": 2,
    }
