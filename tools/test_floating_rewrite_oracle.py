"""Offline fixtures; no device or floating-instruction qualification."""
from fractions import Fraction
from floating_rewrite_oracle import (evaluate, round_binary32, rational, decode_element,
                                     decode_scale, _reduce, _soft, INF)
from arithmetic_certificate import centered_int8, verify
from arithmetic_composition import Node, verify_composition


def main():
    # E2M1 codes: 7=6, 15=-6, 4=2, 12=-2, 1=0.5.
    a, b = [7]*29+[15,12,0], [7]*30+[4,0]
    fold = evaluate(a,b,120,252,"fold")
    balanced = evaluate(a,b,120,252,"balanced")
    assert fold["outputs"]["rational"] == {"direct":INF,"hoisted":0x7f7b0000}
    assert balanced["outputs"]["rational"] == {"direct":0x7f7b0000,"hoisted":0x7f7b0000}
    assert fold["verdicts"]["floating_rewrite_equivalence"] == "refuted"
    both = evaluate([7]*32,[7]*32,120,252)
    assert both["outputs"]["rational"] == {"direct":INF,"hoisted":INF}
    assert both["verdicts"]["floating_rewrite_equivalence"] == "proven"
    assert both["verdicts"]["finite_exact_recoverability"] == "refuted"
    underflow = evaluate([1,1],[1,1],53,53)
    assert underflow["outputs"]["rational"] == {"direct":0,"hoisted":1}
    for report in (fold,balanced,both,underflow):
        assert report["evaluators_agree"]
        assert not report["runtime_dispatch_eligible"]
        assert report["verdicts"]["target_execution"] == "not_run"
    assert fold["sha256"] != balanced["sha256"]
    for bad in (-1,253,254,255,True):
        try: decode_scale(bad)
        except ValueError: pass
        else: raise AssertionError("invalid scale admitted")
    for tree in ((0,0),(0,2),(0,),"fold"):
        try: _reduce([0,0],tree,_soft)
        except ValueError: pass
        else: raise AssertionError("malformed tree admitted")
    # Exhaustive decoded-element pairs at fixed scale one, independent evaluator.
    for a in range(16):
        for b in range(16):
            report=evaluate([a],[b],127,127)
            assert report["evaluators_agree"]
            assert rational(report["outputs"]["rational"]["direct"]) == decode_element(a)*decode_element(b)
    for k in (1,2,31,32,33,64):
        for topology in ("fold","balanced"):
            for scales in ((0,0),(53,53),(120,252),(127,127),(252,252)):
                report=evaluate([7,15]*(k//2)+([7] if k%2 else []),[7]*k,*scales,topology)
                assert report["evaluators_agree"]
    assert round_binary32(Fraction(1,1<<150)) == 0  # halfway to first subnormal
    assert round_binary32(Fraction(3,1<<150)) == 2  # ties to even
    assert round_binary32(-Fraction(1,1<<151)) == 0 # declared canonical zero
    integer=verify(centered_int8(251))
    assert integer["verdicts"]["floating_rewrite_equivalence"] == "unsupported"
    assert integer["verdicts"]["target_execution"] == "not_run"
    composition=verify_composition((Node("a","input",interval=(0,1)),))
    assert composition["verdicts"]["physical_layout_correctness"] == "unknown"
    print("floating rewrite oracle self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
