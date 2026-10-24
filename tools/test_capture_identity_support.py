"""Complete, explicitly synthetic provenance for authored test captures only."""
import uuid


def synthetic_identity(capture: dict, *, commit: str = "c" * 40) -> None:
    capture["git_commit"] = commit
    capture["execution_id"] = str(uuid.uuid4())
    capture["build_manifest"] = {"fixture_only": True, "version": 2, "source_digest": "a" * 64,
                                 "source_commit": commit, "source_dirty": False,
                                 "configuration": "synthetic fixture", "dependency_digest": "d" * 64,
                                 "linkage": "static-rns8", "identity_scope": "source-config-and-external-headers",
                                 "external_dependencies": {}}
    for name in ("CK", "ROCWMMA", "HIPBLASLT"):
        capture["build_manifest"]["configuration"] += f"\n{name.lower()}=ON\ndependency_tree.{name}=synthetic/{name}"
        capture["build_manifest"]["external_dependencies"][name] = {
            "root": f"synthetic/{name}", "sha256": "e" * 64, "files": 1, "bytes": 1}
    capture["artifact_capture"] = {"executable_sha256": "b" * 64, "unchanged_during_run": True,
                                   "rns8_linkage": "static", "fixture_only": True}


def synthetic_host_measurements(capture: dict) -> None:
    """Declare authored samples explicitly; never use this on collected evidence."""
    metadata = capture.setdefault("timing_metadata", {})
    metadata.update(host_timing_contract_version=2, host_result_kind="single_operation",
                    clock_period_numerator=1, clock_period_denominator=1000000000,
                    timer_overhead_samples=1001, timer_pair_median_us=0.02,
                    timer_pair_p95_us=0.04, censor_threshold_us=0.2,
                    synthetic_fixture_only=True)
    metadata["phase_states"] = {
        phase: ("measured" if samples and min(samples) > 0.2 else "below_resolution")
        for phase, samples in capture.get("raw_timings_us", {}).items()}


def synthetic_static_isa_report(capture: dict, counts: dict) -> dict:
    """Authored static association only; never evidence from a real code object."""
    import hashlib
    import json
    if "artifact_capture" not in capture or "build_manifest" not in capture:
        synthetic_identity(capture)
    return {"fixture_only": True, "schema": "rns8_static_isa_v2",
        "object": "synthetic-kernel-object.o", "object_sha256": "1" * 64,
        "code_object_sha256": "2" * 64, "backend": capture["backend_selected"],
        "target": capture["device"]["gcn_arch"], "device_symbol_count": 1,
        "reported_symbol_count": 1, "tools": {"rga_status": "not_run_optional"},
        "symbols": [{"symbol": "synthetic_selected_symbol", "counts": counts}],
        "instruction_totals": counts,
        "capture_binding": {"schema": "rns8_selected_static_isa_binding_v1",
            "artifact_kind": "executable_sha256",
            "artifact_sha256": capture["artifact_capture"]["executable_sha256"],
            "code_object_sha256": "2" * 64,
            "build_manifest_sha256": hashlib.sha256(json.dumps(capture["build_manifest"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "target": capture["device"]["gcn_arch"], "selected_kernel": capture["selected_kernel"],
            "symbols": ["synthetic_selected_symbol"], "selection_evidence": "explicit_static_symbol_mapping",
            "artifact_membership_verified": True, "dynamic_execution_proven": False}}


def synthetic_isa_index(capture: dict, histogram: dict) -> dict:
    from pathlib import Path
    from evidence_database_lib.isa import summarize_isa_report, normalized_backend
    report = synthetic_static_isa_report(capture, {"matrix_instruction_histogram": histogram})
    return {f"{normalized_backend(capture['backend_selected'])}|{capture['device']['gcn_arch']}":
            [summarize_isa_report(Path("synthetic-isa-summary.json"), report)]}


def synthetic_independent_runs(captures: list[dict], count: int = 2) -> list[dict]:
    """Author repeated-process controls; never apply to collected evidence."""
    from copy import deepcopy
    result = []
    for source in captures:
        base = deepcopy(source)
        synthetic_identity(base, commit="a" * 40)
        # Report unit fixtures can define medians directly. Materialize explicit
        # synthetic samples rather than pretending stale arrays match them.
        for phase, summary in (base.get("timing_summary_us") or {}).items():
            if not isinstance(summary, dict) or type(summary.get("median")) not in (int, float): continue
            value = summary["median"]
            length = 1 if phase in ("planning", "scheduling", "tile_bound_scan", "global_bound_scan", "matrix_alloc", "prepack_setup") else base.get("repeats", 9)
            base.setdefault("raw_timings_us", {})[phase] = [value] * length
            summary.update(avg=value, median=value, p95=value)
            base[f"avg_{phase}_us"] = value
        synthetic_host_measurements(base)
        for ordinal in range(count):
            capture = deepcopy(base)
            capture["execution_id"] = str(uuid.uuid4())
            capture["_authored_process_ordinal"] = ordinal
            result.append(capture)
    return result


def write_independent_fixture_pair(path, capture: dict):
    import json
    from pathlib import Path
    path = Path(path)
    for ordinal, record in enumerate(synthetic_independent_runs([capture])):
        target = path if ordinal == 0 else path.with_name(path.stem + ".independent.json")
        target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def independent_fixture_paths(paths):
    from pathlib import Path
    result = []
    for value in paths:
        path = Path(value); result.append(path)
        sibling = path.with_name(path.stem + ".independent.json")
        if path.is_file() and sibling.exists(): result.append(sibling)
    return result
