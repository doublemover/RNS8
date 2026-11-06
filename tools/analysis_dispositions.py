"""Narrow, reviewed static-analysis dispositions; never an automatic baseline."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re

from analysis_diagnostics import SUPPORTED_TOOL
from benchmark_request import file_digest
from compiled_binary_identity import canonical_digest
from json_evidence import load_evidence_json
from quality_inventory import ROOT
from static_analysis import verify_receipt

SIGNATURE_FIELDS = {
    "context",
    "stage",
    "target",
    "check",
    "level",
    "path",
    "anchor_hex",
    "offset_in_anchor",
    "message",
}
CLASSIFICATIONS = {"intentional_arithmetic", "tool_false_positive", "third_party_false_positive"}
MAX_BYTES = 16 * 1024**2


def source_path(raw: object) -> Path:
    if type(raw) is not str or not raw or len(raw) > 4096 or "\\" in raw:
        raise ValueError("explicit repository-relative source path required")
    path = PurePosixPath(raw)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or path.parts[0]
        not in {"include", "src", "tests", "tools", "benchmarks", "examples", "docs"}
    ):
        raise ValueError("disposition cannot match a path glob or external tree")
    result = (ROOT / raw).resolve(strict=True)
    if (
        not result.is_relative_to(ROOT.resolve())
        or not result.is_file()
        or result.stat().st_size > MAX_BYTES
    ):
        raise ValueError("disposition source escaped repository or byte budget")
    return result


def read_record(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("bounded disposition/analysis record required")
    value = load_evidence_json(path)
    if type(value) is not dict:
        raise ValueError("disposition/analysis record must be an object")
    return value


def source_bytes(path: Path, cache: dict[Path, bytes] | None) -> bytes:
    if cache is not None and path in cache:
        return cache[path]
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("diagnostic source exceeds byte budget")
    if cache is not None:
        if len(cache) >= 4096 or len(data) + sum(map(len, cache.values())) > 8 * MAX_BYTES:
            raise ValueError("diagnostic source cache exceeds review memory budget")
        cache[path] = data
    return data


def load_policy(
    path: Path, *, today: date, sources: dict[Path, bytes] | None = None
) -> tuple[dict, dict[str, dict]]:
    policy = read_record(path)
    if (
        set(policy) != {"schema", "owner_issue", "tool_version", "config_sha256", "entries"}
        or policy["schema"] != "rns8_analysis_dispositions_v1"
        or type(policy["owner_issue"]) is not int
        or policy["owner_issue"] != 99
        or policy["tool_version"] != SUPPORTED_TOOL
        or policy["config_sha256"] != file_digest(ROOT / ".clang-tidy")
        or type(policy["entries"]) is not list
        or len(policy["entries"]) > 1000
    ):
        raise ValueError("disposition policy/tool/config identity invalid")
    indexed = {}
    for entry in policy["entries"]:
        if type(entry) is not dict or set(entry) != SIGNATURE_FIELDS | {
            "id",
            "classification",
            "owner",
            "reason",
            "upstream",
            "revisit_before",
            "evidence",
        }:
            raise ValueError("complete narrow diagnostic disposition required")
        signature = {key: entry[key] for key in SIGNATURE_FIELDS}
        if (
            entry["id"] != canonical_digest(signature)
            or entry["id"] in indexed
            or entry["classification"] not in CLASSIFICATIONS
            or entry["level"] not in {"Warning", "Remark"}
            or type(entry["check"]) is not str
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", entry["check"])
            or entry["check"] == "clang-diagnostic-error"
            or type(entry["context"]) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", entry["context"])
            or entry["stage"] not in {"host-analyzer", "hip-host-analyzer", "hip-device-analyzer"}
        ):
            raise ValueError("duplicate, broad, parse-error or unreviewed disposition")
        if (
            (entry["stage"] == "hip-device-analyzer") != (type(entry["target"]) is str)
            or entry["target"] is not None
            and (not entry["target"] or len(entry["target"]) > 128)
        ):
            raise ValueError("disposition needs its exact host/device branch target")
        for key, limit in (("owner", 128), ("reason", 2048), ("message", 65536)):
            if type(entry[key]) is not str or not entry[key].strip() or len(entry[key]) > limit:
                raise ValueError("reviewable diagnostic owner/reason/message required")
        if entry["upstream"] is not None and (
            type(entry["upstream"]) is not str
            or not entry["upstream"].startswith("https://")
            or len(entry["upstream"]) > 2048
        ):
            raise ValueError("upstream reference must be an explicit HTTPS link or null")
        if (
            type(entry["revisit_before"]) is not str
            or date.fromisoformat(entry["revisit_before"]).isoformat() != entry["revisit_before"]
            or today >= date.fromisoformat(entry["revisit_before"])
        ):
            raise ValueError("diagnostic disposition expired or has no canonical revisit date")
        if (
            type(entry["anchor_hex"]) is not str
            or not re.fullmatch(r"(?:[0-9a-f]{2}){16,256}", entry["anchor_hex"])
            or type(entry["offset_in_anchor"]) is not int
        ):
            raise ValueError("bounded exact diagnostic byte anchor required")
        anchor = bytes.fromhex(entry["anchor_hex"])
        source = source_bytes(source_path(entry["path"]), sources)
        if not 0 <= entry["offset_in_anchor"] <= len(anchor) or source.count(anchor) != 1:
            raise ValueError("diagnostic byte anchor changed or is ambiguous")
        evidence = entry["evidence"]
        if (
            type(evidence) is not dict
            or set(evidence) != {"path", "sha256"}
            or file_digest(source_path(evidence["path"])) != evidence["sha256"]
        ):
            raise ValueError("reduced control or source-invariant evidence changed")
        indexed[entry["id"]] = entry
    return policy, indexed


def signature_for(
    diagnostic: dict,
    detailed: dict,
    *,
    sources: dict[Path, bytes] | None = None,
    anchors: dict | None = None,
) -> dict:
    location = diagnostic["message"]["location"]
    if (
        diagnostic["parse_failure"]
        or diagnostic["level"] not in {"Warning", "Remark"}
        or location["state"] != "located"
        or type(location["path"]) is not str
    ):
        raise ValueError("parse/error/unlocated diagnostics cannot receive a disposition")
    path = Path(location["path"]).resolve(strict=True)
    try:
        relative = path.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            "external-only diagnostic needs separate source custody; no broad dependency exclusion"
        ) from error
    path = source_path(relative)
    raw, offset = source_bytes(path, sources), location["byte_offset"]
    if type(offset) is not int or not 0 <= offset <= len(raw):
        raise ValueError("diagnostic byte location invalid")
    cached = None if anchors is None else anchors.get((path, offset))
    if cached is None:
        begin, end = max(0, offset - 128), min(len(raw), offset + 128)
        anchor = raw[begin:end]
        if len(anchor) < 16 or raw.count(anchor) != 1:
            raise ValueError("diagnostic source anchor is too short or ambiguous")
        cached = anchor.hex(), offset - begin
        if anchors is not None:
            anchors[path, offset] = cached
    return dict(
        context=detailed["context"],
        stage=detailed["stage"],
        target=detailed["target"],
        check=diagnostic["check"],
        level=diagnostic["level"],
        path=relative,
        anchor_hex=cached[0],
        offset_in_anchor=cached[1],
        message=diagnostic["message"]["message"],
    )


def evaluate(receipt: Path, policy_path: Path, *, today: date | None = None) -> dict:
    today = today or datetime.now(timezone.utc).date()
    initial = {
        str(path.resolve(strict=True)): file_digest(path)
        for path in (receipt, policy_path, ROOT / ".clang-tidy")
    }
    result = verify_receipt(receipt)
    sources, anchors = {}, {}
    policy, entries = load_policy(policy_path, today=today, sources=sources)
    if (
        result["status"] not in {"selected_static_contexts_clean", "findings_require_triage"}
        or result["unattempted_passes"]
        or result["analyzed_passes"] != result["selected_passes"]
        or not result["analyzed_passes"]
    ):
        raise ValueError("only complete observed analyzer coverage can enter a findings gate")
    seen, observations, unclassified, evidence = set(), [], [], {}
    retained_bytes = 0

    def append_finding(target: list, value: dict) -> None:
        nonlocal retained_bytes
        retained_bytes += len(
            json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if retained_bytes > MAX_BYTES - 2 * 1024**2:
            raise ValueError("static finding report exceeds its retained byte budget")
        target.append(value)

    for summary in result["results"]:
        detailed = read_record(Path(summary["receipt"]["path"]))
        for diagnostic in detailed["diagnostics"]:
            if diagnostic["role"] != "program_diagnostic":
                continue
            if len(observations) + len(unclassified) >= 10000:
                raise ValueError("static finding inventory exceeds its bounded gate")
            try:
                signature = signature_for(diagnostic, detailed, sources=sources, anchors=anchors)
            except ValueError as error:
                append_finding(
                    unclassified,
                    dict(
                        pass_identity=summary["identity"],
                        check=diagnostic["check"],
                        reason=str(error),
                    ),
                )
                continue
            key = canonical_digest(signature)
            entry = entries.get(key)
            if entry is None:
                append_finding(
                    unclassified,
                    dict(
                        id=key, signature=signature, reason="new_or_changed_unreviewed_diagnostic"
                    ),
                )
                continue
            seen.add(key)
            evidence[str(source_path(entry["evidence"]["path"]))] = entry["evidence"]["sha256"]
            append_finding(
                observations,
                dict(
                    id=key,
                    classification=entry["classification"],
                    owner=entry["owner"],
                    pass_identity=summary["identity"],
                ),
            )
    stale = sorted(entries.keys() - seen)
    references = {
        **result["source_artifacts"],
        **result["retained_artifacts"],
        **initial,
        **evidence,
    }
    for path, digest in references.items():
        if file_digest(path) != digest:
            raise ValueError("analysis/disposition input changed during review: " + path)
    report = dict(
        schema="rns8_analysis_disposition_gate_v1",
        status="unclassified_findings"
        if unclassified
        else "stale_dispositions"
        if stale
        else "reviewed_findings_no_new"
        if observations
        else "selected_static_contexts_clean",
        receipt=dict(path=str(receipt.resolve()), sha256=initial[str(receipt.resolve())]),
        policy=dict(path=str(policy_path.resolve()), sha256=initial[str(policy_path.resolve())]),
        tool_version=policy["tool_version"],
        config_sha256=policy["config_sha256"],
        selected_passes=result["selected_passes"],
        analyzed_passes=result["analyzed_passes"],
        reviewed_observations=observations,
        unclassified=unclassified,
        stale_dispositions=stale,
        date=today.isoformat(),
        hardware_qualification=False,
        performance_promotion_allowed=False,
    )
    report["identity_sha256"] = canonical_digest(report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=ROOT / "metadata/analysis-dispositions.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--advisory", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output:
            out = args.output.resolve()
            if out.exists() or not out.is_relative_to((ROOT / "temp").resolve()):
                raise ValueError("disposition output requires a fresh ignored temp child")
        report = evaluate(args.receipt, args.policy)
        if args.output:
            encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
            if len(encoded.encode("utf-8")) > MAX_BYTES:
                raise ValueError("disposition report exceeds byte budget")
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if key not in {"reviewed_observations", "unclassified"}
                },
                indent=2,
            )
        )
        accepted = {"selected_static_contexts_clean", "reviewed_findings_no_new"}
        if args.advisory and not report["stale_dispositions"]:
            accepted.add("unclassified_findings")
        return 0 if report["status"] in accepted else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(
            json.dumps(
                dict(
                    status="blocked_or_incomplete",
                    error=str(error),
                    hardware_qualification=False,
                    performance_promotion_allowed=False,
                )
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
