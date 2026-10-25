"""Source/IR binding, bounded universal polynomial checks and seeded mutations.

Intentionally authored without executing as part of the frontloaded development.
"""
import copy
import json
from pathlib import Path
import tempfile

from generate_native_wmma import Variant, generate
from native_wmma_lowering import recipe, render, mutations
from check_native_wmma_logic import CONFIGURATIONS, check, inspect, load_bound_report
from promotion_ledger import build_ledger


def main():
    for unsigned in (False, True):
        for group in (1, 2, 4):
            ir = recipe(group, unsigned)
            for config in CONFIGURATIONS:
                report = check(ir, config)
                assert report["logical_equivalence"] == "proven", report
                assert report["memory_model"] == "proven"
                assert report["target_execution"] == "not_run"
                assert not report["qualification"]
            body = render(ir)
            for name, mutated in mutations(ir).items():
                assert render(mutated) != body, name  # mutations affect actual emitted source
                results = []
                for config in CONFIGURATIONS:
                    results.append(check(mutated, config))
                    if results[-1]["logical_equivalence"] == "refuted":
                        break
                assert any(r["logical_equivalence"] == "refuted" for r in results), (name, results)
    ir = recipe(1, False)
    unsupported = copy.deepcopy(ir); unsupported["instruction"] = "unknown_mfma"
    assert check(unsupported, CONFIGURATIONS[0])["logical_equivalence"] == "unknown"
    hidden = copy.deepcopy(ir); hidden["a_index"] = ["+", hidden["a_index"], "channel"]
    assert check(hidden, CONFIGURATIONS[0])["logical_equivalence"] == "unknown"
    assert check(ir, CONFIGURATIONS[0], budget=1)["logical_equivalence"] == "unknown"
    bad = dict(CONFIGURATIONS[0], lda=1)
    assert check(ir, bad)["logical_equivalence"] == "unknown"
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch); directory = root / "generated"
        manifest = generate(directory, [Variant()])
        configurations = [CONFIGURATIONS[0], CONFIGURATIONS[0]]
        report = inspect(directory, configurations)
        assert report["cache_hits"] == 1 and report["unique_checks"] == 1
        assert report["logical_checks_passed"] and not report["qualified"]
        assert report["variants"][0]["checks"][1]["reuse"]["matching_preconditions"]
        source = (directory / "native_variants.hip").read_text()
        assert render(manifest["variants"][0]["lowering"]) in source
        path = root / "logical.json"; path.write_text(json.dumps(report))
        bound = load_bound_report(directory, path)
        assert bound["source_current"] and not bound["independently_replayed_during_ingestion"]
        ledger = build_ledger([], None, generated_logical_reports=[(directory, path)])
        assert ledger["generated_logical_assurance"][0]["sha256"] == bound["sha256"]
        assert ledger["accelerator_promotable_count"] == 0
        stale = copy.deepcopy(report); stale["generated_source_sha256"] = "0" * 64
        path.write_text(json.dumps(stale))
        try: load_bound_report(directory, path)
        except ValueError: pass
        else: raise AssertionError("source drift was accepted")
        (directory / "native_variants.hip").write_text(source.replace("kk+=16", "kk+=32"))
        try: inspect(directory, [CONFIGURATIONS[0]])
        except ValueError: pass
        else: raise AssertionError("source unbound to logical lowering accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
