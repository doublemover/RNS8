"""Complete, explicitly synthetic provenance for authored test captures only."""
import uuid


def synthetic_identity(capture: dict, *, commit: str = "c" * 40) -> None:
    capture["git_commit"] = commit
    capture["execution_id"] = str(uuid.uuid4())
    capture["build_manifest"] = {"fixture_only": True, "version": 1, "source_digest": "a" * 64,
                                 "source_commit": commit, "source_dirty": False,
                                 "configuration": "synthetic fixture", "dependency_digest": "d" * 64}
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
