"""Bounded family reproducibility, identity and fail-closed specification fixtures."""
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
from generate_native_wmma import Variant, generate, composition
from check_generated_native_wmma import verify_generated_source


def main():
    baseline = Variant()
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        variants = [Variant(unsigned_inputs=u, group=g) for u in (False, True) for g in (1, 2, 4)]
        first = generate(root / "a", variants)
        second = generate(root / "b", variants)
        assert first == second
        assert len({x["id"] for x in first["variants"]}) == 6
        for name in ("native_variants.hip", "native_variants.hpp", "native_variants.json"):
            assert (root / "a" / name).read_bytes() == (root / "b" / name).read_bytes()
        source = (root / "a/native_variants.hip").read_bytes()
        assert hashlib.sha256(source).hexdigest() == first["source_sha256"]
        for entry in first["variants"]:
            assert entry["kernel_symbol"].encode() in source
            assert len(entry["arithmetic_certificates"]) == 20
            assert len(entry["composition_certificates"]) == 20
            assert entry["resource_plan"]["global_temporary_bytes"] == 0
            assert not entry["compatibility"]["cross_feature_runtime_qualified"]
        try:
            composition(replace(baseline, safe_k=262144), 256)
        except ValueError:
            pass
        else:
            raise AssertionError("overflowing carried slab composition accepted")
        assert not first["qualified"] and not first["promotion_eligible"]
        assert verify_generated_source(root / "a") == first
        (root / "a/native_variants.hpp").write_text("// drift\n", encoding="utf-8")
        try:
            verify_generated_source(root / "a")
        except ValueError:
            pass
        else:
            raise AssertionError("generated header drift accepted")
        for bad in (replace(baseline, group=3), replace(baseline, target="gfx942"),
                    replace(baseline, unsigned_inputs=1), replace(baseline, tile_m=32),
                    replace(baseline, vector_width=4), replace(baseline, pipeline_depth=2),
                    replace(baseline, safe_k=65537), replace(baseline, split_k=2),
                    replace(baseline, output="wrap64"), replace(baseline, layout="column_major"),
                    replace(baseline, tails="unmasked"), replace(baseline, lds_budget=1)):
            try:
                generate(root / "bad", [bad])
            except ValueError:
                pass
            else:
                raise AssertionError("unsupported variant emitted")
        assert not (root / "bad").exists()
        try:
            generate(root / "duplicate", [baseline, baseline])
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate variant identity accepted")
    print("native WMMA generator self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
