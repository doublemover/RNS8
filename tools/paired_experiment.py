"""Explicit two-build experiments; ordinary comparison source matching stays strict.

Plans declare the allowed implementation delta before collection. Receipts bind
the observed sequential collection order and every raw production/diagnostic
process. They describe collector observations, not hardware attestation or cache
promotion authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import json
from pathlib import Path
import re
from types import MappingProxyType

from benchmark_request import file_digest
from benchmark_sweep_lib.execution import cli_backend
from collect_diagnostic_pairs import paired_plan
from diagnostic_companion import build_binding, digest, normalize_arguments
from observation_cohorts import comparison_environment, evidence
from review_receipt import ReviewReceiptReader
from timing_eligibility import capture_identity

PLAN_SCHEMA = "rns8_paired_source_experiment_plan_v1"
RECEIPT_SCHEMA = "rns8_paired_source_experiment_collection_v1"
ARMS = ("baseline", "candidate")


def file_reference(path: Path) -> dict:
    path = path.resolve(strict=True)
    return dict(path=str(path), sha256=file_digest(path))


def arm_identity(capture: dict) -> dict:
    manifest = capture.get("build_manifest") or {}
    artifact = capture.get("artifact_capture") or {}
    return dict(
        source_commit=manifest.get("source_commit"),
        source_digest=manifest.get("source_digest"),
        build_manifest_sha256=digest(manifest),
        executable_sha256=artifact.get("executable_sha256"),
        shared_library_sha256=artifact.get("shared_library_sha256"),
        backend_selected=capture.get("backend_selected"),
        selected_kernel=capture.get("selected_kernel"),
    )


def _seal(plan: dict) -> str:
    return digest({key: value for key, value in plan.items() if key != "identity_sha256"})


def _sequence(runs: int):
    for run in range(runs):
        for arm in ARMS if run % 2 == 0 else reversed(ARMS):
            for mode in (
                ("production", "instrumented") if run % 2 == 0 else ("instrumented", "production")
            ):
                yield run, arm, mode


def validate_plan(plan: dict) -> None:
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("paired experiment plan schema")
    runs = plan.get("runs")
    if type(runs) is not int or not 2 <= runs <= 100 or runs % 2:
        raise ValueError("an even 2..100 independent runs per arm are required")
    if not isinstance(plan.get("collection_id"), str) or not re.fullmatch(
        r"[a-zA-Z0-9_-]{1,80}", plan["collection_id"]
    ):
        raise ValueError("safe collection identifier required")
    if (
        not isinstance(plan.get("allowed_delta"), str)
        or not 16 <= len(plan["allowed_delta"]) <= 4096
    ):
        raise ValueError("describe the intentionally allowed source/implementation change")
    if plan.get("identity_sha256") != _seal(plan):
        raise ValueError("paired experiment plan identity changed")
    arms = plan.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("exactly baseline and candidate arms required")
    identities = []
    for arm in ARMS:
        value = arms[arm]
        if not isinstance(value, dict) or set(value) != {"bench", "manifest", "identity"}:
            raise ValueError("paired experiment arm shape")
        if not isinstance(value["bench"], str) or not Path(value["bench"]).is_absolute():
            raise ValueError("absolute declared executable path required")
        if not isinstance(value["manifest"], dict) or set(value["manifest"]) != {"path", "sha256"}:
            raise ValueError("hash-bound build manifest reference required")
        identity = value["identity"]
        if not isinstance(identity, dict) or set(identity) != {
            "source_commit",
            "source_digest",
            "build_manifest_sha256",
            "executable_sha256",
            "shared_library_sha256",
            "backend_selected",
            "selected_kernel",
        }:
            raise ValueError("explicit source/build/artifact/backend/kernel identity required")
        for key in ("source_digest", "build_manifest_sha256", "executable_sha256"):
            if not isinstance(identity[key], str) or not re.fullmatch(
                r"[0-9a-f]{64}", identity[key]
            ):
                raise ValueError("unbound paired experiment arm: " + key)
        if not isinstance(identity["source_commit"], str) or not re.fullmatch(
            r"[0-9a-f]{40}", identity["source_commit"]
        ):
            raise ValueError("exact source commit required")
        if any(
            not isinstance(identity[key], str) or not identity[key]
            for key in ("backend_selected", "selected_kernel")
        ):
            raise ValueError("explicit backend and selected kernel required")
        if identity["shared_library_sha256"] is not None:
            raise ValueError("paired source collection currently requires static RNS8 linkage")
        identities.append((identity["source_commit"], identity["source_digest"]))
    if identities[0] == identities[1]:
        raise ValueError("use ordinary same-source comparison for identical source identities")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 4 * runs:
        raise ValueError("complete counterbalanced two-arm plan required")
    outputs = set()
    arguments = None
    for ordinal, (job, expected) in enumerate(zip(jobs, _sequence(runs))):
        if (
            not isinstance(job, dict)
            or set(job) != {"ordinal", "run", "arm", "mode", "output", "command"}
            or any(type(job.get(key)) is not int for key in ("ordinal", "run"))
        ):
            raise ValueError("paired plan job shape")
        if (job["ordinal"], job["run"], job["arm"], job["mode"]) != (ordinal, *expected):
            raise ValueError("paired plan collection order is not counterbalanced")
        output = job["output"]
        if not isinstance(output, str) or not Path(output).is_absolute() or output in outputs:
            raise ValueError("distinct absolute capture outputs required")
        outputs.add(output)
        command = job["command"]
        arm = arms[job["arm"]]
        if not isinstance(command, list) or len(command) < 5 or command[0] != arm["bench"]:
            raise ValueError("command does not name declared arm executable")
        if command[-4:] != [
            "--backend",
            cli_backend(arm["identity"]["backend_selected"]),
            "--host-timing",
            job["mode"],
        ]:
            raise ValueError("command backend/timing differs from declared arm")
        common = normalize_arguments(command[1:-4])
        if common != command[1:-4]:
            raise ValueError("host timing is declared only by the planned job mode")
        if any(arg == "--backend" or arg.startswith("--backend=") for arg in common):
            raise ValueError("duplicate backend option")
        if arguments is not None and common != arguments:
            raise ValueError("paired arms must use the same explicit mathematical request")
        arguments = common


def build_plan(
    definition: dict,
    parent: Path,
    output: Path,
    runs: int,
    collection_id: str,
    arguments: list[str],
    reference_directory: Path | None = None,
) -> dict:
    """Prepare an immutable plan without executing a compiler or benchmark."""
    if not isinstance(definition, dict) or set(definition) != {"allowed_delta", "arms"}:
        raise ValueError("definition requires allowed_delta and arms")
    if not isinstance(definition["arms"], dict) or set(definition["arms"]) != set(ARMS):
        raise ValueError("two explicit experiment arms required")
    if any(arg == "--backend" or arg.startswith("--backend=") for arg in arguments):
        raise ValueError("declare backend per arm, not in common arguments")
    reader, arms, jobs_by_arm = ReviewReceiptReader(), {}, {}
    for arm in ARMS:
        value = definition["arms"][arm]
        if not isinstance(value, dict) or set(value) != {
            "bench",
            "manifest",
            "backend_selected",
            "selected_kernel",
        }:
            raise ValueError("definition arm shape")
        path, manifest = reader.document(value["manifest"], parent)
        binary = (parent / value["bench"]).resolve(strict=True)
        identity = dict(
            source_commit=manifest.get("source_commit"),
            source_digest=manifest.get("source_digest"),
            build_manifest_sha256=digest(manifest),
            executable_sha256=file_digest(binary),
            shared_library_sha256=None,
            backend_selected=value["backend_selected"],
            selected_kernel=value["selected_kernel"],
        )
        arms[arm] = dict(bench=str(binary), manifest=file_reference(path), identity=identity)
        # Reuse the ordinary production/event-pair request policy and cache args.
        rows = paired_plan(
            binary,
            normalize_arguments(arguments),
            output.resolve(),
            runs,
            collection_id + "-" + arm,
            reference_directory,
        )
        jobs_by_arm[arm] = {
            (row["independent_run"], item["mode"]): item for row in rows for item in row["captures"]
        }
    jobs = []
    for ordinal, (run, arm, mode) in enumerate(_sequence(runs)):
        item = jobs_by_arm[arm][(run, mode)]
        # Keep arm options at the end and do not infer implementation from paths.
        command = item["command"][:-2] + [
            "--backend",
            cli_backend(arms[arm]["identity"]["backend_selected"]),
            "--host-timing",
            mode,
        ]
        jobs.append(
            dict(
                ordinal=ordinal, run=run, arm=arm, mode=mode, output=item["output"], command=command
            )
        )
    plan = dict(
        schema=PLAN_SCHEMA,
        collection_id=collection_id,
        runs=runs,
        allowed_delta=definition["allowed_delta"],
        arms=arms,
        jobs=jobs,
    )
    plan["identity_sha256"] = _seal(plan)
    validate_plan(plan)
    reader.finish()
    return plan


def check_arm(capture: dict, plan: dict, arm: str) -> None:
    if arm_identity(capture) != plan["arms"][arm]["identity"]:
        raise ValueError("capture differs from its declared source/build/implementation arm")
    if (capture.get("artifact_capture") or {}).get("rns8_linkage") != "static":
        raise ValueError("paired experiment requires a content-bound static executable")


@dataclass(frozen=True)
class VerifiedPairedExperiment:
    plan_identity: str
    receipt_sha256: str
    allowed_delta: str
    membership: Mapping[str, str]

    def arm(self, capture: dict) -> str | None:
        if capture is None:
            return None
        arm = self.membership.get(capture_identity(capture))
        if arm is None:
            return None
        records = evidence(capture).get("observations", [])
        identities = (
            [record["capture_sha256"] for record in records]
            if records
            else [capture_identity(capture)]
        )
        arms = {self.membership.get(identity) for identity in identities}
        return arm if arms == {arm} else None

    def environment(self, capture: dict, identity: dict) -> dict:
        if self.arm(capture) is None:
            return identity
        return {
            key: value
            for key, value in identity.items()
            if key not in ("source_commit", "source_digest")
        } | {"paired_experiment": self.plan_identity}

    def blockers(self, candidate: dict, baseline: dict) -> list[str]:
        return (
            []
            if self.arm(candidate) == "candidate" and self.arm(baseline) == "baseline"
            else ["paired_experiment_arm_membership_mismatch"]
        )

    def metadata(self, capture: dict) -> dict:
        return dict(
            plan_identity_sha256=self.plan_identity,
            receipt_sha256=self.receipt_sha256,
            arm=self.arm(capture),
            allowed_delta=self.allowed_delta,
            scope="collector_recorded_order_not_hardware_attestation_or_cache_authority",
        )


def load_experiment(plan_path: Path, receipt_path: Path):
    """Replay collection custody; all timing/numerical gates still run downstream."""
    reader = ReviewReceiptReader()
    _, plan = reader.document(file_reference(plan_path))
    validate_plan(plan)
    receipt_reference = file_reference(receipt_path)
    _, receipt = reader.document(receipt_reference)
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("completed") is not True:
        raise ValueError("completed paired collection receipt required")
    if receipt.get("plan") != file_reference(plan_path):
        raise ValueError("collection receipt does not bind this exact plan")
    rows = receipt.get("observations")
    if not isinstance(rows, list) or len(rows) != len(plan["jobs"]):
        raise ValueError("incomplete paired collection")
    captures, membership, executions, pairs = [], {}, set(), {}
    last_finish = -1
    for job, row in zip(plan["jobs"], rows):
        if not isinstance(row, dict) or set(row) != {
            "ordinal",
            "run",
            "arm",
            "mode",
            "started_ns",
            "finished_ns",
            "capture",
            "capture_sha256",
            "execution_id",
        }:
            raise ValueError("collection observation shape")
        if any(
            type(row[key]) is not int or row[key] < 0
            for key in ("ordinal", "run", "started_ns", "finished_ns")
        ):
            raise ValueError("integer collection sequence and monotonic interval required")
        if any(row[key] != job[key] for key in ("ordinal", "run", "arm", "mode")):
            raise ValueError("observed order differs from declared counterbalanced order")
        if row["started_ns"] < last_finish or row["finished_ns"] <= row["started_ns"]:
            raise ValueError("paired collection must be sequential and non-overlapping")
        last_finish = row["finished_ns"]
        capture = reader.capture(row["capture"], receipt_path.parent)
        identity = capture_identity(capture)
        run_id = capture.get("execution_id")
        if identity != row["capture_sha256"] or not run_id or run_id != row["execution_id"]:
            raise ValueError("collection observation content/execution identity changed")
        if identity in membership or run_id in executions:
            raise ValueError("copied capture or reused process is not an independent observation")
        check_arm(capture, plan, row["arm"])
        if (capture.get("request_identity") or {}).get("arguments") != job["command"][1:]:
            raise ValueError("actual process request differs from planned request")
        executions.add(run_id)
        pairs.setdefault((row["run"], row["arm"]), {})[row["mode"]] = capture
        # Diagnostics never enter the independent production cohort.
        if row["mode"] == "production":
            membership[identity] = row["arm"]
            captures.append(capture)
    bindings = {}
    for pair in pairs.values():
        production, diagnostic = pair["production"], pair["instrumented"]
        binding = build_binding(production["_path"], diagnostic["_path"])
        bindings[production["_path"]] = binding
    environments = set()
    for capture in captures:
        value = json.loads(comparison_environment(capture))
        environments.add(
            digest(
                {
                    key: field
                    for key, field in value.items()
                    if key not in ("source_commit", "source_digest")
                }
            )
        )
    if len(environments) != 1:
        raise ValueError(
            "paired collection changed host/device/toolchain/runtime/timing environment"
        )
    reader.finish()
    return (
        VerifiedPairedExperiment(
            plan["identity_sha256"],
            receipt_reference["sha256"],
            plan["allowed_delta"],
            MappingProxyType(membership),
        ),
        captures,
        bindings,
        reader,
    )
