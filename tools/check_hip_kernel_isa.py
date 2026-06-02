#!/usr/bin/env python3
"""Validate selected direct-HIP kernel ISA properties from a compiled object."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


GEMM_SYMBOL_MARKERS = [
    "rns8_ring_gemm_i8_i32_tiled_kernel",
    "rns8_ring_gemm_i8_i32_scheduled_kernel",
]

FORBIDDEN_MNEMONIC_RE = re.compile(r"\b[sv]_(?:div|rem|rcp)\w*\b")
REQUIRED_RECIPROCAL_MNEMONIC = "v_mul_hi_u32"
MNEMONIC_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\b")


def run_command(command: list[str]) -> str:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed with exit code "
            f"{completed.returncode}: {' '.join(command)}\n{completed.stderr.strip()}"
        )
    return completed.stdout


def sibling_tool(hipcc: Path | None, name: str) -> str:
    suffixes = [".exe", ".bat", ".cmd", ""] if sys.platform == "win32" else ["", ".exe"]
    candidates: list[Path] = []
    if hipcc is not None:
        candidates.extend(hipcc.parent / f"{name}{suffix}" for suffix in suffixes)
    for suffix in suffixes:
        found = shutil.which(f"{name}{suffix}")
        if found:
            candidates.append(Path(found))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(f"required HIP/LLVM tool not found: {name}")


def bundle_target_for(targets: list[str], amdgpu_target: str) -> str:
    for target in targets:
        if target.startswith("hip") and target.endswith(f"--{amdgpu_target}"):
            return target
    raise RuntimeError(
        f"fatbin does not contain HIP bundle for {amdgpu_target}; found: {', '.join(targets) or '<none>'}"
    )


def extract_code_object(
    objcopy: str,
    bundler: str,
    host_object: Path,
    amdgpu_target: str,
    scratch: Path,
) -> Path:
    fatbin = scratch / "hip_direct_kernels.fatbin"
    run_command([objcopy, f"--dump-section=.hip_fatbin={fatbin}", str(host_object)])
    listed = run_command([bundler, "--list", "--type=o", f"--input={fatbin}"])
    targets = [line.strip() for line in listed.splitlines() if line.strip()]
    device_target = bundle_target_for(targets, amdgpu_target)

    outputs: list[Path] = []
    for index, target in enumerate(targets):
        extension = ".co" if target == device_target else ".bundle"
        outputs.append(scratch / f"bundle_{index}{extension}")
    command = [
        bundler,
        "--unbundle",
        "--type=o",
        f"--targets={','.join(targets)}",
        f"--input={fatbin}",
    ]
    for output in outputs:
        command.append(f"--output={output}")
    run_command(command)
    return outputs[targets.index(device_target)]


def symbol_table_symbols(objdump: str, code_object: Path) -> list[str]:
    symbols = run_command([objdump, "-t", str(code_object)])
    names: list[str] = []
    for line in symbols.splitlines():
        if " F .text" not in line:
            continue
        for marker in GEMM_SYMBOL_MARKERS:
            if marker in line:
                names.append(line.split()[-1])
                break
    missing = [marker for marker in GEMM_SYMBOL_MARKERS if not any(marker in name for name in names)]
    if missing:
        raise RuntimeError(f"missing expected GEMM kernel symbols: {', '.join(missing)}")
    return names


def scan_disassembly(objdump: str, code_object: Path, amdgpu_target: str, symbols: list[str]) -> list[str]:
    reports: list[str] = []
    for symbol in symbols:
        disassembly = run_command(
            [
                objdump,
                "-d",
                f"--mcpu={amdgpu_target}",
                f"--disassemble-symbols={symbol}",
                str(code_object),
            ]
        )
        forbidden: list[str] = []
        mnemonics: set[str] = set()
        for line in disassembly.splitlines():
            match = MNEMONIC_RE.match(line)
            if not match:
                continue
            mnemonic = match.group(1)
            mnemonics.add(mnemonic)
            if FORBIDDEN_MNEMONIC_RE.search(mnemonic):
                forbidden.append(line.strip())
        if forbidden:
            raise RuntimeError(
                f"{symbol} contains forbidden divide/remainder/rcp instructions:\n" + "\n".join(forbidden)
            )
        if REQUIRED_RECIPROCAL_MNEMONIC not in mnemonics:
            raise RuntimeError(f"{symbol} does not contain required {REQUIRED_RECIPROCAL_MNEMONIC} instruction")
        reports.append(f"{symbol}: no div/rem/rcp mnemonics; contains {REQUIRED_RECIPROCAL_MNEMONIC}")
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, type=Path, help="Compiled HIP host object containing .hip_fatbin")
    parser.add_argument("--target", required=True, help="AMDGPU target id, for example gfx1100")
    parser.add_argument("--hipcc", type=Path, help="HIP compiler path; sibling LLVM tools are preferred")
    parser.add_argument("--llvm-objcopy", help="Override llvm-objcopy path")
    parser.add_argument("--llvm-objdump", help="Override llvm-objdump path")
    parser.add_argument("--clang-offload-bundler", help="Override clang-offload-bundler path")
    parser.add_argument("--scratch-root", type=Path, default=Path("temp"), help="Ignored scratch directory")
    args = parser.parse_args()

    host_object = args.object
    if not host_object.exists():
        raise RuntimeError(f"HIP object does not exist: {host_object}")

    objcopy = args.llvm_objcopy or sibling_tool(args.hipcc, "llvm-objcopy")
    objdump = args.llvm_objdump or sibling_tool(args.hipcc, "llvm-objdump")
    bundler = args.clang_offload_bundler or sibling_tool(args.hipcc, "clang-offload-bundler")

    args.scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rns8-hip-isa-", dir=args.scratch_root) as scratch_dir:
        scratch = Path(scratch_dir)
        code_object = extract_code_object(objcopy, bundler, host_object, args.target, scratch)
        symbols = symbol_table_symbols(objdump, code_object)
        reports = scan_disassembly(objdump, code_object, args.target, symbols)

    print("HIP ISA check: PASS")
    print(f"object: {host_object}")
    print(f"target: {args.target}")
    for report in reports:
        print(f"- {report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - command-line tool should print concise failure cause.
        print(f"HIP ISA check: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
