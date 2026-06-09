from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from benchmark_sweep_lib.cli import (
    list_scenarios_payload,
    load_required_isa_index,
    release_preflight_readiness,
    review_capture_paths,
    write_command_plan,
)
from benchmark_sweep_lib.config import SweepCommand


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

    scenarios = list_scenarios_payload()
    assert scenarios["schema_version"] == 1
    assert "release-candidates" in scenarios["special_scenarios"]
    assert "skinny-gemv" in scenarios["scenarios"]
    assert scenarios["count"] == len(scenarios["scenarios"])

    release_entry = SweepCommand(
        name="release-candidate",
        command=["rns8-bench", "--backend", "hip-direct"],
        output=tmp / "release-candidate.json",
        scenario={"promotion_eligibility": "release_review_candidate"},
    )
    evidence_only_entry = SweepCommand(
        name="evidence-only",
        command=["rns8-bench", "--backend", "hip-direct"],
        output=tmp / "evidence-only.json",
        scenario={"promotion_eligibility": "execution_path_evidence"},
    )
    smoke_readiness = release_preflight_readiness(
        argparse.Namespace(review_mode="smoke", warmups=1, repeats=3),
        [release_entry, evidence_only_entry],
    )
    assert smoke_readiness["release_candidate_captures"] == 1
    assert smoke_readiness["release_ready"] is False
    assert {warning["code"] for warning in smoke_readiness["warnings"]} == {
        "review_mode_not_release",
        "warmups_below_release_minimum",
        "repeats_below_release_minimum",
    }
    release_readiness = release_preflight_readiness(
        argparse.Namespace(review_mode="release", warmups=3, repeats=9),
        [release_entry],
    )
    assert release_readiness == {
        "release_candidate_captures": 1,
        "release_ready": True,
        "warnings": [],
    }
    nonrelease_readiness = release_preflight_readiness(
        argparse.Namespace(review_mode="smoke", warmups=1, repeats=3),
        [evidence_only_entry],
    )
    assert nonrelease_readiness == {
        "release_candidate_captures": 0,
        "release_ready": True,
        "warnings": [],
    }

    plan_root = tmp / "plan"
    command_paths = write_command_plan(
        [
            SweepCommand(
                name="example",
                command=["rns8-bench", "--backend", "hip-direct"],
                output=plan_root / "example.json",
                env={"HIP_VISIBLE_DEVICES": "0"},
                scenario={"family": "fixture", "name": "example"},
            )
        ],
        plan_root,
    )
    plan_json = json.loads(Path(command_paths["command_plan"]).read_text(encoding="utf-8"))
    assert plan_json["capture_count"] == 1
    assert plan_json["entries"][0]["environment"] == {"HIP_VISIBLE_DEVICES": "0"}
    plan_text = Path(command_paths["command_plan_text"]).read_text(encoding="utf-8")
    assert "HIP_VISIBLE_DEVICES=0 rns8-bench --backend hip-direct" in plan_text
