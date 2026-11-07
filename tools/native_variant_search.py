"""Bounded experimental-family search reports, separate from promotion authority."""

from __future__ import annotations

from collections import defaultdict
import re
import statistics

from evidence_scalars import finite_number


def regression_scope(corpus, *, component, target, features, version):
    if (
        not isinstance(corpus, dict)
        or corpus.get("schema") != "rns8_native_known_regressions_v1"
        or set(corpus)
        != {
            "schema",
            "source",
            "component",
            "reported_fixed_release",
            "reported_bundle",
            "affected_release_range",
            "qualification_inherited",
            "records",
        }
        or any(
            type(corpus.get(key)) is not str or not corpus[key]
            for key in ("source", "component", "reported_fixed_release", "reported_bundle")
        )
        or corpus.get("qualification_inherited") is not False
        or corpus.get("affected_release_range") is not None
        or not isinstance(corpus.get("records"), list)
        or not 1 <= len(corpus["records"]) <= 64
        or not isinstance(features, set)
        or any(type(item) is not str for item in features)
        or any(type(item) is not str or not item for item in (component, target, version))
    ):
        raise ValueError("bounded explicit native regression scope required")
    rows, identities = [], set()
    for item in corpus["records"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"id", "features", "reported_targets", "failure"}
            or type(item["id"]) is not str
            or not item["id"]
            or item["id"] in identities
            or not isinstance(item["features"], list)
            or not item["features"]
            or any(type(feature) is not str or not feature for feature in item["features"])
            or len(set(item["features"])) != len(item["features"])
            or type(item["failure"]) is not str
            or not item["failure"]
            or (
                item["reported_targets"] is not None
                and (
                    not isinstance(item["reported_targets"], list)
                    or not item["reported_targets"]
                    or any(
                        type(value) is not str or not value for value in item["reported_targets"]
                    )
                )
            )
        ):
            raise ValueError("invalid native known-regression record")
        identities.add(item["id"])
        if component != corpus.get("component"):
            state = "different_component_no_inherited_vendor_qualification"
        elif not set(item["features"]) <= features:
            state = "different_path_no_inherited_vendor_qualification"
        elif item["reported_targets"] is not None and target not in item["reported_targets"]:
            state = "target_not_reported_not_proven_unaffected"
        elif version == corpus.get("reported_fixed_release"):
            state = "reported_fix_release_requires_actual_regression"
        else:
            state = "affected_version_range_unknown_requires_actual_regression"
        rows.append(dict(id=item["id"], state=state, executed=False, qualified=False))
    return dict(
        schema="rns8_native_regression_scope_v1",
        component=component,
        target=target,
        version=version,
        features=sorted(features),
        source=corpus.get("source"),
        records=rows,
        qualification_inherited=False,
    )


def finite(value, *, positive=False):
    if not finite_number(value) or value < 0 or (positive and value == 0):
        raise ValueError("finite native search objective required")
    return value


def summarize(manifest, observations, *, required_cohorts, baseline_ids):
    """Pareto retirement requires full declared coverage and all three objectives.

    The caller owns source/capture admission. Unknown memory or preparation
    measurements cannot become zero. This report never edits shipped variants.
    """
    entries = manifest.get("variants") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "rns8_generated_native_wmma_v1"
        or not isinstance(entries, list)
        or not 1 <= len(entries) <= 6
        or not isinstance(observations, list)
        or len(observations) > 8000
        or not isinstance(required_cohorts, dict)
        or not 1 <= len(required_cohorts) <= 80
        or not isinstance(baseline_ids, set)
        or not baseline_ids
    ):
        raise ValueError("bounded generated family/cohort/baseline policy required")
    if any(
        not isinstance(item, dict)
        or type(item.get("id")) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", item["id"])
        for item in entries
    ):
        raise ValueError("source-bound native variant digests required")
    ids = {item["id"] for item in entries}
    if len(ids) != len(entries) or not baseline_ids <= ids:
        raise ValueError("duplicate/missing native baseline identity")
    for cohort, expected in required_cohorts.items():
        if (
            type(cohort) is not str
            or not cohort
            or not isinstance(expected, set)
            or not expected
            or not expected <= ids
        ):
            raise ValueError("explicit compatible variant set required for every cohort")
    groups, seen, seen_paths, seen_digests = defaultdict(list), set(), set(), set()
    for item in observations:
        if not isinstance(item, dict) or set(item) != {
            "variant_id",
            "cohort",
            "run_id",
            "latency_ns",
            "payload_bytes",
            "prepare_ns",
            "capture",
        }:
            raise ValueError("complete source-bound native search observation required")
        variant, cohort, run = (item[key] for key in ("variant_id", "cohort", "run_id"))
        if (
            type(variant) is not str
            or type(cohort) is not str
            or cohort not in required_cohorts
            or variant not in required_cohorts[cohort]
            or type(run) is not str
            or not run
            or run in seen
        ):
            raise ValueError("duplicate run or foreign native variant/cohort")
        seen.add(run)
        finite(item["latency_ns"], positive=True)
        for field in ("payload_bytes", "prepare_ns"):
            if item[field] is not None:
                finite(item[field])
        if item["payload_bytes"] is not None and (
            type(item["payload_bytes"]) is not int or item["payload_bytes"] >= 1 << 64
        ):
            raise ValueError("observed payload bytes must be uint64")
        reference = item["capture"]
        if (
            not isinstance(reference, dict)
            or set(reference) != {"path", "sha256"}
            or any(type(reference[key]) is not str or not reference[key] for key in reference)
            or not re.fullmatch(r"[0-9a-f]{64}", reference["sha256"])
        ):
            raise ValueError("retained native capture reference required")
        if reference["path"] in seen_paths or reference["sha256"] in seen_digests:
            raise ValueError("copied native captures cannot create independent observations")
        seen_paths.add(reference["path"])
        seen_digests.add(reference["sha256"])
        groups[(cohort, variant)].append(item)
    points, coverage, partial_points = {}, [], []
    for cohort, expected in sorted(required_cohorts.items()):
        for variant in sorted(expected):
            samples = groups.get((cohort, variant), [])
            reason = None
            if len(samples) < 5:
                reason = "fewer_than_five_owned_process_observations"
            elif any(
                row[field] is None for row in samples for field in ("payload_bytes", "prepare_ns")
            ):
                reason = "missing_payload_or_preparation_observation"
            if reason:
                coverage.append(dict(cohort=cohort, variant_id=variant, state=reason))
                if samples:
                    times = [row["latency_ns"] for row in samples]
                    partial_points.append(
                        dict(
                            cohort=cohort,
                            variant_id=variant,
                            samples=len(samples),
                            lower_ns=min(times),
                            upper_ns=max(times),
                            median_ns=statistics.median(times),
                            captures=[row["capture"] for row in samples],
                        )
                    )
                continue
            latencies = [row["latency_ns"] for row in samples]
            points[(cohort, variant)] = dict(
                lower_ns=min(latencies),
                upper_ns=max(latencies),
                median_ns=statistics.median(latencies),
                payload_bytes=max(row["payload_bytes"] for row in samples),
                payload_bytes_lower=min(row["payload_bytes"] for row in samples),
                prepare_ns=max(row["prepare_ns"] for row in samples),
                prepare_ns_lower=min(row["prepare_ns"] for row in samples),
                captures=[row["capture"] for row in samples],
            )
    decisions = []
    for variant in sorted(ids):
        cohorts = [cohort for cohort, expected in required_cohorts.items() if variant in expected]
        dominators = set(ids - {variant})
        if not cohorts or any((cohort, variant) not in points for cohort in cohorts):
            dominators.clear()
        for cohort in cohorts:
            if (cohort, variant) not in points:
                break
            candidate = points[(cohort, variant)]
            admitted = set()
            for other in required_cohorts[cohort] - {variant}:
                peer = points.get((cohort, other))
                if peer is not None and (
                    peer["upper_ns"] < candidate["lower_ns"]
                    and peer["payload_bytes"] <= candidate["payload_bytes_lower"]
                    and peer["prepare_ns"] <= candidate["prepare_ns_lower"]
                ):
                    admitted.add(other)
            dominators &= admitted
        state = (
            "retain_explicit_baseline"
            if variant in baseline_ids
            else "conditional_retirement_candidate"
            if dominators
            else "retain_no_complete_dominance"
        )
        decisions.append(
            dict(
                variant_id=variant,
                state=state,
                dominating_variants=sorted(dominators),
                required_cohorts=sorted(cohorts),
            )
        )
    return dict(
        schema="rns8_generated_family_search_v1",
        decisions=decisions,
        coverage_gaps=coverage,
        observed_points=[dict(cohort=c, variant_id=v, **p) for (c, v), p in sorted(points.items())],
        partial_objective_points=partial_points,
        objective_scope="declared_native_microcase_latency_payload_and_preparation_not_public_pipeline_or_ISA_qualification",
        estimator="conservative_process_min_max_separation_not_a_confidence_interval",
        qualification_required=[
            "source_bound_logical_and_memory_checks",
            "actual_variant_ISA_resources",
            "device_differential",
            "known_regressions",
            "held_out_cohorts",
            "ordinary_strongest_control",
        ],
        automatic_retirement=False,
        promotion_allowed=False,
    )
