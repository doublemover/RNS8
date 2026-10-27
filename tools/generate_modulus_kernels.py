#!/usr/bin/env python3
"""Generate bounded, algebraically certified reducer specializations and manifest."""
import argparse
import hashlib
import json
from pathlib import Path

from arithmetic_certificate import centered_int8, verify
from modulus_set_search import DEFAULT_MODULI

MODS = tuple(DEFAULT_MODULI)


def gen(out: Path, moduli=MODS, safe_k=65536):
    moduli = tuple(moduli)
    if (not moduli or len(moduli) > 28 or len(set(moduli)) != len(moduli)
            or any(type(m) is not int or not 2 <= m <= 256 for m in moduli)):
        raise ValueError("expected 1..28 distinct byte moduli")
    if type(safe_k) is not int or not 1 <= safe_k <= 65536:
        raise ValueError("generated baseline K slab must lie in 1..65536")
    certificates = [verify(centered_int8(m, safe_k)) for m in moduli]
    if any(not c["algebraically_verified"] for c in certificates):
        raise ValueError("unsafe arithmetic certificate")
    lines = ["// Generated reducer baseline; GPU execution qualification is separate.",
             "#pragma once", "#include <cstdint>",
             "#if defined(__HIPCC__)", "#define RNS8_GENERATED_HD __host__ __device__ constexpr",
             "#else", "#define RNS8_GENERATED_HD constexpr", "#endif",
             "namespace rns8::generated {", "template<uint16_t Modulus> struct CenteredReducer;",
             "inline constexpr uint16_t moduli[] = {" + ",".join(map(str, moduli)) + "};"]
    for m, certificate in zip(moduli, certificates):
        lines += [f"template<> struct CenteredReducer<{m}> {{",
                  f'  inline static constexpr char certificate_sha256[] = "{certificate["sha256"]}";',
                  f"  inline static constexpr uint32_t safe_k = {safe_k};",
                  "  static RNS8_GENERATED_HD int8_t apply(int32_t value) {",
                  f"    int32_t residue = value % {m};",
                  f"    if (residue < 0) residue += {m};",
                  f"    return static_cast<int8_t>(residue >= {(m + 1) // 2} ? residue - {m} : residue);",
                  "  }", "};"]
    lines += ["}", "#undef RNS8_GENERATED_HD", ""]
    source = "\n".join(lines)
    manifest = {"schema": "rns8_generated_reducers_v1", "moduli": moduli, "safe_k": safe_k,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "arithmetic_certificates": [c["sha256"] for c in certificates],
                "algebraically_verified": True, "compiled": False, "instruction_measured": False,
                "end_to_end_qualified": False, "runtime_dispatch_eligible": False,
                "scope": "constant-modulus centered reducers; no generated matrix instruction family"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "rns8_generated_centered_reducers.cuh").write_bytes(source.encode("utf-8"))
    (out / "rns8_generated_centered_reducers.json").write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("temp/generated-reducers"))
    parser.add_argument("--moduli", default=",".join(map(str, MODS)))
    parser.add_argument("--safe-k", type=int, default=65536)
    args = parser.parse_args()
    gen(args.out_dir, [int(m) for m in args.moduli.split(",")], args.safe_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
