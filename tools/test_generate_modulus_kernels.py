"""Deterministic source/manifest and bounded-domain generator regressions."""
import hashlib
from pathlib import Path
import tempfile

from generate_modulus_kernels import gen


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = gen(root / "a", (256, 255, 251))
        second = gen(root / "b", (256, 255, 251))
        assert first == second
        a = (root / "a/rns8_generated_centered_reducers.cuh").read_bytes()
        b = (root / "b/rns8_generated_centered_reducers.cuh").read_bytes()
        assert a == b
        assert b"\r" not in a
        assert first["source_sha256"] == hashlib.sha256(a).hexdigest()
        assert not first["runtime_dispatch_eligible"]
        for moduli, cap in (((256, 256), 65536), ((1,), 1), ((257,), 1), ((256,), 65537), ((), 1)):
            try:
                gen(root / "bad", moduli, cap)
            except ValueError:
                pass
            else:
                raise AssertionError("illegal variant was generated")
        assert not (root / "bad").exists()
    print("modulus generator self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
