"""Synthetic algebra/resource scheduling tests, not hardware calibration."""
from dataclasses import replace
from fractions import Fraction

from analytical_planner import Rate, Stage, predict
from arithmetic_certificate import centered_int8


def main() -> int:
    rates = {name: Rate(Fraction(1), Fraction(2), "operations", "synthetic-only")
             for name in ("conversion", "matrix", "reconstruction")}
    stages = (Stage("pack", "pack_kernel", (), ("hbm", "conversion"), (("conversion", 100),), 10, bytes_allocated=100),
              Stage("gemm", "gemm_kernel", ("pack",), ("matrix",), (("matrix", 200),), 10, bytes_allocated=50),
              Stage("export", "crt_kernel", ("gemm",), ("conversion",), (("reconstruction", 50),), 10, retain_output=True))
    result = predict(stages, rates, centered_int8(255), 150, "synthetic-target")
    assert result["latency_lower_ns"] == 205
    assert result["latency_upper_ns"] == 380
    assert result["peak_workspace_bytes"] == 150
    assert not result["promotion_eligible"]
    for broken in (stages + (stages[0],), (replace(stages[0], dependencies=("export",)),) + stages[1:]):
        try:
            predict(broken, rates, centered_int8(255), 150, "synthetic-target")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid graph admitted")
    try:
        predict(stages, rates, centered_int8(255), 149, "synthetic-target")
    except ValueError:
        pass
    else:
        raise AssertionError("insufficient workspace accepted")
    # Neither all-fast nor all-slow endpoint overlaps these two buffers, but
    # the allowed interior rate 20/11 does. Admission must cover it.
    envelope_rates = {"delay": Rate(Fraction(1), Fraction(5, 2), "operations", "synthetic-only"),
                      "fixed": Rate(Fraction(1), Fraction(1), "operations", "synthetic-only")}
    witness = (Stage("a_delay", "delay", (), ("a",), (("delay", 100),), 0),
               Stage("b_buffer", "buffer", ("a_delay",), ("a",), (("fixed", 0),), 10, bytes_allocated=100),
               Stage("x_delay", "delay", (), ("x",), (("fixed", 0),), 55),
               Stage("y_buffer", "buffer", ("x_delay",), ("x",), (("fixed", 0),), 10, bytes_allocated=100))
    result = predict(witness, envelope_rates, centered_int8(255), 200, "synthetic-target")
    assert result["peak_workspace_bytes"] == 200
    assert result["workspace_admission"] == "robust_lifetime_envelopes"
    try:
        predict(witness, envelope_rates, centered_int8(255), 100, "synthetic-target")
    except ValueError:
        pass
    else:
        raise AssertionError("interior-rate allocation overlap ignored")
    interior = dict(envelope_rates, delay=Rate(Fraction(20, 11), Fraction(20, 11), "operations", "synthetic-only"))
    assert predict(witness, interior, centered_int8(255), 200, "synthetic-target")["peak_workspace_bytes"] == 200
    # One producer allocation serves two consumers, with no double counting.
    consumers = (Stage("p", "produce", (), ("p",), (("fixed", 0),), 1, bytes_allocated=100),
                 Stage("c1", "consume", ("p",), ("c1",), (("fixed", 0),), 10),
                 Stage("c2", "consume", ("p",), ("c2",), (("fixed", 0),), 20),
                 Stage("q", "produce", ("c1",), ("q",), (("fixed", 0),), 1, bytes_allocated=50))
    assert predict(consumers, envelope_rates, centered_int8(255), 150, "synthetic-target")["peak_workspace_bytes"] == 150
    # Boundary overlap is intentionally conservative, including instantaneous
    # allocations. The report never labels this bound a tight observed peak.
    boundary = (Stage("p", "p", (), ("p",), (("fixed", 0),), 1, bytes_allocated=10),
                Stage("q", "q", (), ("p",), (("fixed", 0),), 1, bytes_allocated=20))
    assert predict(boundary, envelope_rates, centered_int8(255), 30, "synthetic-target")["peak_workspace_bytes"] == 30
    print("analytical planner self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
