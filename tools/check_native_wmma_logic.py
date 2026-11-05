"""Bounded source-bound native WMMA logic checker; never device qualification.

Each physical fragment slot is compared symbolically with an independent scalar
residue specification. Equality holds for all admitted operand values, not just
random numeric samples, for ONLY the exact listed configurations. The matrix
instruction's documented lane algebra and native encoder are trusted contracts.
"""

import argparse
from collections import Counter
from functools import lru_cache
import copy
import hashlib
import json
import operator
from pathlib import Path
import time

from modulus_set_search import DEFAULT_MODULI
from native_wmma_lowering import expression, fingerprint, mutations, render, validate
from json_evidence import parse_evidence_json


class Refuted(ValueError):
    pass


class Unsupported(ValueError):
    pass


CONFIGURATIONS = [
    {"m": 16, "n": 16, "k": 16, "lda": 16, "ldb": 16, "ldc": 16, "prefix": 4},
    {"m": 17, "n": 19, "k": 31, "lda": 34, "ldb": 22, "ldc": 24, "prefix": 5},
    {"m": 1, "n": 3, "k": 17, "lda": 21, "ldb": 7, "ldc": 8, "prefix": 1},
    *[
        {"m": 1, "n": 1, "k": k, "lda": k + 3, "ldb": 4, "ldc": 5, "prefix": 3}
        for k in (65535, 65536, 65537)
    ],
    {"m": 3, "n": 2, "k": 7, "lda": 11, "ldb": 5, "ldc": 6, "prefix": 20},
]
SUPPORT = [
    "nonnegative_INT64_indices",
    "native64_symbolic_operands",
    "masked_zero_load",
    "initialized_register_fragments",
    "signed_INT32_WMMA_contract",
    "centered_remainder_carry",
    "centered_byte_store",
    "single_wave_program_order",
    "nonaliasing_output",
]
TRUSTED = [
    "RDNA3 signed IU8 wave32 physical-lane-to-logical-matrix instruction contract",
    "source-bound native64 centered reducer is congruent and within [-128,127]",
    "compiler implements emitted C++ without miscompilation",
    "live A/B allocations initialized at every declared logical input cell; no concurrent mutation",
    "launch uses exactly the manifested grid and 32 threads; output does not alias inputs",
]
MEMORY_SCOPE = "logical_single_wave_program_order_not_emitted_barriers"
MEMORY_DEPENDENCIES = [
    "initialized input -> masked load -> native encode -> fragment",
    "initialized accumulator + fragment -> WMMA -> centered remainder carry",
    "final carry -> centered store -> unique output cell",
]


def fail(condition, detail):
    if not condition:
        raise Refuted(detail)


_INDEX_OPERATIONS = {
    "+": operator.add,
    "*": operator.mul,
    "//": operator.floordiv,
    "%": operator.mod,
    "<": lambda a, b: int(a < b),
    "and": lambda a, b: int(bool(a) and bool(b)),
    "min": min,
}


def eval_index(node, env):
    # Every intermediate operation fits the exact nonnegative signed-64 domain.
    # Render casts each variable to int64_t before arithmetic. No signed overflow
    # or floor-vs-truncating division simplification is assumed.
    if isinstance(node, list):
        # Reuse the checked child values. Reinterpreting the entire subtree at
        # every ancestor repeats the same work for millions of fragment slots.
        if len(node) != 3 or node[0] not in _INDEX_OPERATIONS:
            raise Unsupported("unsupported index expression")
        a, b = eval_index(node[1], env), eval_index(node[2], env)
        value = _INDEX_OPERATIONS[node[0]](a, b)
    else:
        value = expression(node, env)
    if not 0 <= value <= (1 << 63) - 1:
        raise Unsupported("index intermediate outside nonnegative signed-64 domain")
    return value


def configuration(c):
    keys = {"m", "n", "k", "lda", "ldb", "ldc", "prefix"}
    if set(c) != keys or any(type(v) is not int or not 1 <= v <= 2147483647 for v in c.values()):
        raise Unsupported(
            "configuration must contain exact positive INT32 dimensions/pitches/prefix"
        )
    if c["prefix"] > 20 or c["lda"] < c["k"] or c["ldb"] < c["n"] or c["ldc"] < c["n"]:
        raise Unsupported("configuration outside generated launch contract")
    for size in (c["m"] * c["lda"] * 8, c["k"] * c["ldb"] * 8, c["prefix"] * c["m"] * c["ldc"]):
        if size > 2147483647:
            raise Unsupported("configuration outside qualified full-storage byte domain")


@lru_cache(maxsize=32)
def _input_fragment_proof(key):
    # Only successful proofs are cached. The complete evaluated input expressions,
    # loop schedule and configuration are bound; group/output expressions and
    # signedness cannot affect these physical input-cell identities (validate
    # rejects those variables in input expressions). Accumulator/range guards
    # and all output ownership checks are still evaluated on every check.
    ir, c = json.loads(key)
    m, n, k = c["m"], c["n"], c["k"]
    work = Counter()
    # A symbolic token identifies the original native cell. Equality of each
    # lane's token with the reference implies equality after the same exact
    # modular encoder, for every native bit pattern in the declared domain.
    for by in range((m + 15) // 16):
        for bx in range((n + 15) // 16):
            contributions = Counter()
            env = dict(c, bx=bx, by=by, bz=0, channel=0, reg=0, lane=0, word=0, byte=0, kk=0)
            for slab in range(0, k, ir["slab_k"] * ir["slab_step"]):
                work["slabs"] += 1
                for _ in range(ir["slab_repetitions"]):
                    for kk in range(slab, min(k, slab + ir["slab_k"]), ir["k_step"]):
                        env["kk"] = kk
                        for q in range(kk, min(k, kk + 16)):
                            contributions[q] += 1
                        for lane in range(32):
                            env["lane"] = lane
                            for word in range(4):
                                env["word"] = word
                                for byte in range(4):
                                    env["byte"] = byte
                                    q = kk + 4 * word + byte
                                    # Independent scalar specification; NOT
                                    # evaluated through the lowering's indices.
                                    row, col = by * 16 + lane % 16, bx * 16 + lane % 16
                                    for operand, valid, expected, rows, cols, pitch in (
                                        (
                                            "a",
                                            row < m and q < k,
                                            row * c["lda"] + q,
                                            m,
                                            k,
                                            c["lda"],
                                        ),
                                        (
                                            "b",
                                            q < k and col < n,
                                            q * c["ldb"] + col,
                                            k,
                                            n,
                                            c["ldb"],
                                        ),
                                    ):
                                        active = bool(eval_index(ir[operand + "_mask"], env))
                                        fail(
                                            active == valid,
                                            operand
                                            + " mask disagrees with scalar zero-tail specification",
                                        )
                                        if active:
                                            address = eval_index(ir[operand + "_index"], env)
                                            fail(
                                                address < rows * pitch and address % pitch < cols,
                                                operand
                                                + " reads out-of-allocation or poison padding",
                                            )
                                            fail(
                                                address == expected,
                                                operand + " symbolic operand index/stride mismatch",
                                            )
                                            work["initialized_global_reads"] += 1
                                        else:
                                            work["initialized_zero_fragments"] += 1
                                        work["fragment_slots"] += 1
                        work["instructions_per_channel"] += 1
            fail(
                len(contributions) == k and all(contributions[q] == 1 for q in range(k)),
                "missing or duplicate K contribution; scalar residue polynomial differs",
            )
            work["exact_symbolic_contraction_terms_per_output"] += k
    return dict(work)


def _check(ir, c, budget):
    validate(ir)
    configuration(c)
    m, n, k, prefix, group = c["m"], c["n"], c["k"], c["prefix"], ir["group"]
    tiles = ((m + 15) // 16) * ((n + 15) // 16)
    instruction_count = tiles * ((k + 15) // 16) * ir["slab_repetitions"]
    if instruction_count * 1024 > budget or prefix * m * n > 500000:
        raise Unsupported("explicit physical-fragment checking budget exceeded")
    fail(
        ir["accumulator_initialized"], "read of uninitialized accumulator before first instruction"
    )
    fail(
        ir["slab_k"] * ir["slab_repetitions"] * 128 * 128 + 255 <= 2147483647,
        "INT32 partial/carry overflow cannot be simplified as integer algebra",
    )
    fail(
        not ir["reset_each_slab"] or k <= ir["slab_k"],
        "stale partial: previous slab contribution discarded",
    )
    work = Counter()
    input_ir = {
        name: ir[name]
        for name in (
            "a_index",
            "a_mask",
            "b_index",
            "b_mask",
            "slab_k",
            "slab_step",
            "slab_repetitions",
            "k_step",
        )
    }
    key = json.dumps([input_ir, c], sort_keys=True, separators=(",", ":"))
    previous_hits = _input_fragment_proof.cache_info().hits
    work.update(_input_fragment_proof(key))
    work["reused_input_fragment_proof"] = int(
        _input_fragment_proof.cache_info().hits > previous_hits
    )
    writers = Counter()
    for bz in range((prefix + group - 1) // group):
        for by in range((m + 15) // 16):
            for bx in range((n + 15) // 16):
                for channel in range(group):
                    env = dict(
                        c, bx=bx, by=by, bz=bz, channel=channel, reg=0, lane=0, word=0, byte=0, kk=0
                    )
                    plane = bz * group + channel
                    active = bool(eval_index(ir["plane_mask"], env))
                    fail(active == (plane < prefix), "plane/group tail mask mismatch")
                    if not active:
                        continue
                    fail(eval_index(ir["plane"], env) == plane, "wrong logical residue plane")
                    mod_index = eval_index(ir["modulus_index"], env)
                    fail(
                        mod_index < 20 and DEFAULT_MODULI[mod_index] == DEFAULT_MODULI[plane],
                        "wrong modulus domain",
                    )
                    for lane in range(32):
                        env["lane"] = lane
                        for reg in range(8):
                            env["reg"] = reg
                            row, col = by * 16 + 2 * reg + lane // 16, bx * 16 + lane % 16
                            valid = row < m and col < n
                            fail(
                                bool(eval_index(ir["c_mask"], env)) == valid,
                                "output edge mask mismatch",
                            )
                            if not valid:
                                continue
                            index = eval_index(ir["c_index"], env)
                            fail(
                                index < prefix * m * c["ldc"] and index % c["ldc"] < n,
                                "output store outside logical output region",
                            )
                            fail(
                                index == (plane * m + row) * c["ldc"] + col,
                                "output ownership/address mismatch",
                            )
                            writers[index] += 1
    fail(
        len(writers) == prefix * m * n and all(v == 1 for v in writers.values()),
        "output coverage or exactly-one-final-writer violated",
    )
    work["final_stores"] = len(writers)
    return dict(work)


def check(ir, c, budget=40000000):
    started = time.perf_counter()
    result = {
        "configuration": copy.deepcopy(c),
        "lowering_sha256": fingerprint(ir),
        "logical_equivalence": "unknown",
        "memory_model": "unknown",
        "physical_layout_correctness": "unknown",
        "target_execution": "not_run",
        "unresolved_nodes": [],
        "operations": SUPPORT,
        "trusted_assumptions": TRUSTED,
        "qualification": False,
        "promotion_eligible": False,
        "memory_dependencies": list(MEMORY_DEPENDENCIES),
        "memory_scope": MEMORY_SCOPE,
    }
    try:
        result["work"] = _check(ir, c, budget)
        result.update(logical_equivalence="proven", memory_model="proven")
    except Refuted as error:
        result.update(
            logical_equivalence="refuted", memory_model="unknown", counterexample=str(error)
        )
    except (ValueError, KeyError, TypeError, ZeroDivisionError, RecursionError) as error:
        result["unresolved_nodes"] = [str(error)]
    result["checking_seconds"] = time.perf_counter() - started
    return result


def inspect(directory, configurations=None, budget=40000000, mutation_check=False):
    from check_generated_native_wmma import verify_generated_source

    manifest = verify_generated_source(directory)
    configurations = CONFIGURATIONS if configurations is None else configurations
    if not isinstance(configurations, list) or not configurations:
        raise ValueError("explicit nonempty configuration list required")
    entries, cache, hits, mutation_reports = [], {}, 0, []
    for entry in manifest["variants"]:
        ir = entry["lowering"]
        if fingerprint(ir) != entry["lowering_sha256"]:
            raise ValueError("lowering identity drift")
        reports = []
        for config in configurations:
            # Cache key is FULL structural equality, including signedness,
            # masks, pitches and bounds; no hash-only or same-tile inference.
            # Empty variable renaming is the only admitted reuse in this slice.
            key = json.dumps([ir, config, budget], sort_keys=True, separators=(",", ":"))
            if key in cache:
                report = copy.deepcopy(cache[key])
                hits += 1
                report["reuse"] = {
                    "structural_equality": True,
                    "variable_renaming": {},
                    "matching_preconditions": True,
                }
            else:
                report = check(ir, config, budget)
                cache[key] = copy.deepcopy(report)
            reports.append(report)
        entries.append(
            {"id": entry["id"], "kernel_symbol": entry["kernel_symbol"], "checks": reports}
        )
        if mutation_check:
            for name, mutated in mutations(ir).items():
                checked = [check(mutated, c, budget) for c in configurations]
                mutation_reports.append(
                    {
                        "variant": entry["id"],
                        "mutation": name,
                        "lowering_sha256": fingerprint(mutated),
                        "emitted_body_sha256": hashlib.sha256(render(mutated).encode()).hexdigest(),
                        "detected": any(r["logical_equivalence"] == "refuted" for r in checked),
                        "unresolved": any(r["logical_equivalence"] == "unknown" for r in checked),
                        "checks": checked,
                    }
                )
    all_checks = [c for e in entries for c in e["checks"]]
    return {
        "schema": "rns8_generated_native_logical_v1",
        "generated_source_sha256": manifest["source_sha256"],
        "generated_header_sha256": manifest["header_sha256"],
        "variants": entries,
        "generated_manifest_sha256": fingerprint(manifest),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "lowering_implementation_sha256": hashlib.sha256(
            Path(__file__).with_name("native_wmma_lowering.py").read_bytes()
        ).hexdigest(),
        "scope": "only_exact_configurations_under_named_trusted_contracts",
        "checking_seconds": sum(c["checking_seconds"] for c in cache.values()),
        "cache_hits": hits,
        "unique_checks": len(cache),
        "logical_checks_passed": all(
            c["logical_equivalence"] == "proven" and c["memory_model"] == "proven"
            for c in all_checks
        ),
        "mutations": mutation_reports,
        "mutations_passed": bool(mutation_reports)
        and all(m["detected"] or m["unresolved"] for m in mutation_reports),
        "coverage_gaps": [
            "unlisted configurations",
            "compiler lowering",
            "actual lane mapping",
            "device scheduling",
            "all AMD execution",
        ],
        "target_execution": "not_run",
        "qualified": False,
        "promotion_eligible": False,
    }


def load_bound_report(directory, path):
    """Attach source-current assurance to existing ISA/evidence records.

    Reopens and binds report bytes; does not turn a logical proof into runtime
    qualification or replay the expensive checker during evidence ingestion.
    """
    from check_generated_native_wmma import verify_generated_source

    manifest = verify_generated_source(directory)
    with path.open("rb") as stream:
        raw = stream.read(8 * 1024**2 + 1)
    if len(raw) > 8 * 1024**2:
        raise ValueError("logical report exceeds its byte budget")
    report = parse_evidence_json(raw.decode("utf-8"))
    if type(report) is not dict:
        raise ValueError("logical report must be a JSON object")
    if (
        report.get("schema") != "rns8_generated_native_logical_v1"
        or report.get("generated_manifest_sha256") != fingerprint(manifest)
        or report.get("generated_source_sha256") != manifest["source_sha256"]
        or report.get("generated_header_sha256") != manifest["header_sha256"]
        or report.get("checker_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        or report.get("lowering_implementation_sha256")
        != hashlib.sha256(
            Path(__file__).with_name("native_wmma_lowering.py").read_bytes()
        ).hexdigest()
    ):
        raise ValueError("logical report source/lowering/checker binding mismatch")
    expected = {entry["id"]: entry for entry in manifest["variants"]}
    seen = set()
    for variant in report.get("variants", []):
        identity = variant.get("id")
        if (
            identity not in expected
            or identity in seen
            or variant.get("kernel_symbol") != expected[identity]["kernel_symbol"]
        ):
            raise ValueError("logical report variant binding mismatch")
        seen.add(identity)
        if not variant.get("checks"):
            raise ValueError("empty logical coverage")
        for checked in variant["checks"]:
            configuration(checked["configuration"])
            if (
                checked.get("lowering_sha256") != expected[identity]["lowering_sha256"]
                or checked.get("target_execution") != "not_run"
                or checked.get("qualification") is not False
                or checked.get("promotion_eligible") is not False
                or checked.get("trusted_assumptions") != TRUSTED
                or checked.get("operations") != SUPPORT
            ):
                raise ValueError("logical report contract mismatch")
    if (
        seen != set(expected)
        or report.get("qualified") is not False
        or report.get("promotion_eligible") is not False
    ):
        raise ValueError("logical report incomplete coverage or invalid qualification claim")
    checks = [c for v in report["variants"] for c in v["checks"]]
    if report.get("logical_checks_passed") != all(
        c.get("logical_equivalence") == "proven" and c.get("memory_model") == "proven"
        for c in checks
    ):
        raise ValueError("logical report aggregate contradicts individual verdicts")
    bound = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source_current": True,
        "independently_replayed_during_ingestion": False,
        "report": report,
    }
    bound["inventory"] = logical_inventory(manifest, bound)
    return bound


def logical_inventory(manifest, bound):
    """Summarize source-bound claims per exact configuration without replaying.

    Reused checks count as records, never additional configuration coverage.
    Storage envelopes are descriptor arithmetic, not measured device traffic.
    """
    if bound is None:
        return dict(
            status="not_recorded",
            source_current=False,
            variants=[],
            covered_variants=0,
            check_records=0,
            distinct_variant_configurations=0,
            hardware_qualification=False,
        )
    if type(bound) is not dict or bound.get("source_current") is not True:
        raise ValueError("logical inventory requires source-current bound evidence")
    by_id = {entry["id"]: entry for entry in manifest["variants"]}
    variants = []
    seen = set()
    for variant in bound["report"]["variants"]:
        identity = variant["id"]
        if (
            identity not in by_id
            or identity in seen
            or variant["kernel_symbol"] != by_id[identity]["kernel_symbol"]
        ):
            raise ValueError("logical inventory variant/symbol coverage differs")
        seen.add(identity)
        groups = {}
        checks = variant["checks"]
        if not checks or len(checks) > 10000:
            raise ValueError("logical inventory check count is empty/over budget")
        for checked in checks:
            c = checked["configuration"]
            configuration(c)
            logical, memory = checked.get("logical_equivalence"), checked.get("memory_model")
            if (
                (logical, memory)
                not in (("proven", "proven"), ("refuted", "unknown"), ("unknown", "unknown"))
                or checked.get("physical_layout_correctness") != "unknown"
                or checked.get("memory_scope") != MEMORY_SCOPE
                or checked.get("memory_dependencies") != MEMORY_DEPENDENCIES
            ):
                raise ValueError("logical inventory has unsupported proof/memory claims")
            work = checked.get("work")
            if work is not None and (
                type(work) is not dict
                or any(
                    type(key) is not str or type(value) is not int or value < 0
                    for key, value in work.items()
                )
            ):
                raise ValueError("logical work counters must be nonnegative integers")
            if logical == "proven" and (
                work is None or work.get("final_stores") != c["prefix"] * c["m"] * c["n"]
            ):
                raise ValueError("logical final-writer count differs from its exact configuration")
            key = fingerprint(c)
            row = groups.setdefault(
                tuple(sorted(c.items())),
                dict(
                    configuration=dict(c),
                    configuration_sha256=key,
                    check_records=0,
                    logical_verdicts=[],
                    memory_verdicts=[],
                    work_groups={},
                    qualified_full_pitch_byte_envelope=dict(
                        a_native=c["m"] * c["lda"] * 8,
                        b_native=c["k"] * c["ldb"] * 8,
                        c_residues=c["prefix"] * c["m"] * c["ldc"],
                    ),
                    expected_unique_final_writers=c["prefix"] * c["m"] * c["n"],
                ),
            )
            row["check_records"] += 1
            row["logical_verdicts"].append(logical)
            row["memory_verdicts"].append(memory)
            if work is not None:
                work_id = fingerprint(work)
                work_row = row["work_groups"].setdefault(
                    tuple(sorted(work.items())),
                    dict(sha256=work_id, counters=dict(work), check_records=0),
                )
                work_row["check_records"] += 1
        configurations = []
        for row in groups.values():
            for kind in ("logical", "memory"):
                verdicts = row.pop(kind + "_verdicts")
                row[kind + "_verdict"] = (
                    "refuted"
                    if "refuted" in verdicts
                    else "unknown"
                    if "unknown" in verdicts
                    else "proven"
                )
            row["claimed_work"] = list(row.pop("work_groups").values())
            configurations.append(row)
        variants.append(
            dict(
                id=identity,
                kernel_symbol=variant["kernel_symbol"],
                check_records=len(checks),
                distinct_configurations=len(configurations),
                configurations=configurations,
            )
        )
    if seen != set(by_id):
        raise ValueError("logical inventory lacks generated variants")
    return dict(
        status="recorded_exact_configurations",
        source_current=True,
        variants=variants,
        covered_variants=len(variants),
        check_records=sum(row["check_records"] for row in variants),
        distinct_variant_configurations=sum(row["distinct_configurations"] for row in variants),
        logical_proven_configurations=sum(
            c["logical_verdict"] == "proven" for row in variants for c in row["configurations"]
        ),
        memory_proven_configurations=sum(
            c["memory_verdict"] == "proven" for row in variants for c in row["configurations"]
        ),
        memory_scope=MEMORY_SCOPE,
        physical_layout_correctness="unknown",
        independently_replayed_during_ingestion=False,
        hardware_qualification=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--configurations", type=Path)
    parser.add_argument("--slot-budget", type=int, default=40000000)
    parser.add_argument("--mutations", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configs = json.loads(args.configurations.read_text()) if args.configurations else None
    result = inspect(args.generated_dir, configs, args.slot_budget, args.mutations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return (
        0
        if result["logical_checks_passed"] and (not args.mutations or result["mutations_passed"])
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
