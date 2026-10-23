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
    print("analytical planner self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
