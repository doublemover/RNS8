#!/usr/bin/env python3
"""Validate strict wrap64 direct-HIP byte-GEMM36 kernel ISA evidence."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


WRAP64_SYMBOL_MARKERS = [
    "rns8_wrap64_pack_u64_kernel",
    "rns8_wrap64_byte_gemm36_tiled_kernel",
    "rns8_wrap64_export_u64_kernel",
]

FORBIDDEN_DIVIDE_RE = re.compile(r"\b[sv]_(?:div|rem|rcp)[_a-z0-9]*\b")
FORBIDDEN_MATRIX_RE = re.compile(r"\bv_(?:wmma|mfma)[_a-z0-9]*\b")
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
  fatbin = scratch / "wrap64_hip_kernels.fatbin"
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


def wrap64_symbols(objdump: str, code_object: Path) -> list[str]:
  symbols = run_command([objdump, "-t", str(code_object)])
  names: list[str] = []
  for line in symbols.splitlines():
    if " F .text" not in line:
      continue
    for marker in WRAP64_SYMBOL_MARKERS:
      if marker in line:
        names.append(line.split()[-1])
        break
  missing = [marker for marker in WRAP64_SYMBOL_MARKERS if not any(marker in name for name in names)]
  if missing:
    raise RuntimeError(f"missing expected wrap64 kernel symbols: {', '.join(missing)}")
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
    forbidden_divides: list[str] = []
    forbidden_matrix: list[str] = []
    for line in disassembly.splitlines():
      match = MNEMONIC_RE.match(line)
      if not match:
        continue
      mnemonic = match.group(1)
      if FORBIDDEN_DIVIDE_RE.search(mnemonic):
        forbidden_divides.append(line.strip())
      if FORBIDDEN_MATRIX_RE.search(mnemonic):
        forbidden_matrix.append(line.strip())
    if forbidden_divides:
      raise RuntimeError(
          f"{symbol} contains forbidden divide/remainder/rcp instructions:\n"
          + "\n".join(forbidden_divides[:20])
      )
    if forbidden_matrix:
      raise RuntimeError(
          f"{symbol} unexpectedly contains matrix-engine instructions:\n" + "\n".join(forbidden_matrix[:20])
      )
    reports.append(f"{symbol}: no div/rem/rcp mnemonics; no matrix-engine mnemonics")
  return reports


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--object", required=True, type=Path, help="Compiled wrap64 HIP host object containing .hip_fatbin")
  parser.add_argument("--target", required=True, help="AMDGPU target id, for example gfx1100")
  parser.add_argument("--hipcc", type=Path, help="HIP compiler path; sibling LLVM tools are preferred")
  parser.add_argument("--llvm-objcopy", help="Override llvm-objcopy path")
  parser.add_argument("--llvm-objdump", help="Override llvm-objdump path")
  parser.add_argument("--clang-offload-bundler", help="Override clang-offload-bundler path")
  parser.add_argument("--scratch-root", type=Path, default=Path("temp"), help="Ignored scratch directory")
  args = parser.parse_args()

  host_object = args.object
  if not host_object.exists():
    raise RuntimeError(f"wrap64 HIP object does not exist: {host_object}")

  objcopy = args.llvm_objcopy or sibling_tool(args.hipcc, "llvm-objcopy")
  objdump = args.llvm_objdump or sibling_tool(args.hipcc, "llvm-objdump")
  bundler = args.clang_offload_bundler or sibling_tool(args.hipcc, "clang-offload-bundler")

  args.scratch_root.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix="rns8-wrap64-isa-", dir=args.scratch_root) as scratch_dir:
    scratch = Path(scratch_dir)
    code_object = extract_code_object(objcopy, bundler, host_object, args.target, scratch)
    symbols = wrap64_symbols(objdump, code_object)
    reports = scan_disassembly(objdump, code_object, args.target, symbols)

  print("wrap64 ISA check: PASS")
  print(f"object: {host_object}")
  print(f"target: {args.target}")
  for report in reports:
    print(f"- {report}")
  return 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except Exception as exc:  # noqa: BLE001 - command-line tool should print concise failure cause.
    print(f"wrap64 ISA check: FAIL: {exc}", file=sys.stderr)
    raise SystemExit(1)
