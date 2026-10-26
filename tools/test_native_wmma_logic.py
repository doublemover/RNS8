"""Source/IR binding, bounded universal polynomial checks and seeded mutations.

Intentionally authored without executing as part of the frontloaded development.
"""
import copy
import json
from pathlib import Path
import tempfile

from generate_native_wmma import Variant, generate
from native_wmma_lowering import recipe, render, mutations
from check_native_wmma_logic import CONFIGURATIONS, Unsupported, check, eval_index, inspect, load_bound_report
from native_wmma_lowering import expression
from promotion_ledger import build_ledger


def main():
    # Independent legacy traversal checks every intermediate before interpreting
    # a whole subtree; the optimized interpreter must retain the same decisions.
    def reference(node, env):
        if isinstance(node, list):
            for child in node[1:]: reference(child, env)
        result = expression(node, env)
        if not 0 <= result <= (1 << 63) - 1:
            raise Unsupported("reference intermediate overflow")
        return result

    for config in CONFIGURATIONS:
        env = dict(config, bx=0, by=0, bz=0, channel=0, reg=7, lane=31, word=3, byte=3, kk=config['k']-1)
        for key, node in recipe(4, False).items():
            if key.endswith(('_index', '_mask')) or key == 'plane':
                assert eval_index(node, env) == reference(node, env), key
        for op in ('+', '*', '//', '%', '<', 'and', 'min'):
            node = [op, ['+', 'lane', 1], 3]
            assert eval_index(node, env) == reference(node, env), op
    # A bounded root cannot hide overflow in a child, even if min masks it.
    overflowing = ['min', ['+', 'lda', 1], 0]
    for interpreter in (reference, eval_index):
        try: interpreter(overflowing, {'lda': (1 << 63) - 1})
        except Unsupported: pass
        else: raise AssertionError('hidden intermediate overflow accepted')
        try: interpreter(['//', 1, 0], {})
        except ZeroDivisionError: pass
        else: raise AssertionError('division by zero accepted')
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
    first = check(ir, CONFIGURATIONS[0])
    reused = check(recipe(4, True), CONFIGURATIONS[0])
    assert first["logical_equivalence"] == reused["logical_equivalence"] == "proven"
    assert reused["work"]["reused_input_fragment_proof"] == 1
    # Reusing an input-cell proof cannot conceal changed addressing or skip
    # an output-ownership mutation, range guards or a smaller work budget.
    changed = copy.deepcopy(ir); changed["a_index"] = ["+", changed["a_index"], 1]
    assert check(changed, CONFIGURATIONS[0])["logical_equivalence"] == "refuted"
    changed = copy.deepcopy(ir); changed["c_index"] = ["+", changed["c_index"], 1]
    assert check(changed, CONFIGURATIONS[0])["logical_equivalence"] == "refuted"
    assert check(ir, CONFIGURATIONS[0], budget=1)["logical_equivalence"] == "unknown"
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
