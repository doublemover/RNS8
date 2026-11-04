"""Allocation-free configure/body contract regressions; no HIP compilation."""

from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "target.cmake"
        script.write_text(
            f'include("{ROOT.as_posix()}/cmake/RocwmmaTargetContract.cmake")\n'
            'rns8_rocwmma_target_contract(wave ${TARGETS})\n'
            'message(STATUS "wave=${wave}")\n', encoding="utf-8")
        for targets, wave in (("gfx1100", 32), ("gfx942", 64),
                              ("gfx1100;gfx1102:xnack-", 32),
                              ("gfx942:sramecc+:xnack-;gfx950", 64)):
            result = subprocess.run(["cmake", f"-DTARGETS={targets}", "-P", str(script)],
                                    capture_output=True, text=True, timeout=10)
            assert result.returncode == 0, result.stderr
            assert f"wave={wave}" in result.stdout
        for targets in ("gfx1100;gfx942", "gfx942;gfx1100", "gfx1103", "gfxfuture", ""):
            result = subprocess.run(["cmake", f"-DTARGETS={targets}", "-P", str(script)],
                                    capture_output=True, text=True, timeout=10)
            assert result.returncode != 0 and "rocWMMA" in result.stderr
    header = (ROOT / "src/backend_common/matrix_target.hpp").read_text(encoding="utf-8")
    kernels = (ROOT / "src/backend_amdgpu_builtins/amdgpu_builtins_kernels.hip").read_text(encoding="utf-8")
    for family, targets in {"CDNA3": {"gfx942"}, "RDNA3": {"gfx1100", "gfx1101", "gfx1102"},
                            "RDNA4": {"gfx1200", "gfx1201"}}.items():
        guard = re.search(rf"#if ([^\n]+)\n#define RNS8_BUILTIN_BODY_{family} 1", header)
        assert guard and set(re.findall(r"__(gfx[0-9a-f]+)__", guard[1])) == targets
        assert f"#if RNS8_BUILTIN_BODY_{family}" in kernels
    assert "std::strncmp(arch" not in kernels
    print("matrix target contracts: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
