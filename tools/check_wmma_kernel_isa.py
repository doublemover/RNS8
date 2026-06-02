#!/usr/bin/env python3
"""Validate rocWMMA accelerator kernel ISA evidence from a compiled HIP object."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


WMMA_SYMBOL_MARKER = "rocwmma_i8_residue_gemm"
REQUIRED_WMMA_MNEMONIC = "v_wmma_i32_16x16x16_iu8"
FORBIDDEN_INT32_GLOBAL_STORE_RE = re.compile(r"\b(?:global|buffer)_store_dword(?:x[234])?\b")
FORBIDDEN_RECIPROCAL_OR_DIVIDE_RE = re.compile(r"\b[vs]_(?:rcp|div|rem)[_a-z0-9]*\b")


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
  fatbin = scratch / "wmma_backend_kernels.fatbin"
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


def wmma_symbol_count(objdump: str, code_object: Path) -> int:
  symbols = run_command([objdump, "-t", str(code_object)])
  return sum(1 for line in symbols.splitlines() if " F .text" in line and WMMA_SYMBOL_MARKER in line)


def scan_disassembly(objdump: str, code_object: Path, amdgpu_target: str) -> tuple[int, list[str], list[str]]:
  disassembly = run_command([objdump, "-d", f"--mcpu={amdgpu_target}", str(code_object)])
  wmma_count = disassembly.count(REQUIRED_WMMA_MNEMONIC)
  forbidden_stores = [
      line.strip()
      for line in disassembly.splitlines()
      if FORBIDDEN_INT32_GLOBAL_STORE_RE.search(line)
  ]
  forbidden_divides = [
      line.strip()
      for line in disassembly.splitlines()
      if FORBIDDEN_RECIPROCAL_OR_DIVIDE_RE.search(line)
  ]
  return wmma_count, forbidden_stores, forbidden_divides


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--object", required=True, type=Path, help="Compiled rocWMMA HIP host object containing .hip_fatbin")
  parser.add_argument("--target", required=True, help="AMDGPU target id, for example gfx1100")
  parser.add_argument("--hipcc", type=Path, help="HIP compiler path; sibling LLVM tools are preferred")
  parser.add_argument("--llvm-objcopy", help="Override llvm-objcopy path")
  parser.add_argument("--llvm-objdump", help="Override llvm-objdump path")
  parser.add_argument("--clang-offload-bundler", help="Override clang-offload-bundler path")
  parser.add_argument("--scratch-root", type=Path, default=Path("temp"), help="Ignored scratch directory")
  args = parser.parse_args()

  host_object = args.object
  if not host_object.exists():
    raise RuntimeError(f"rocWMMA HIP object does not exist: {host_object}")

  objcopy = args.llvm_objcopy or sibling_tool(args.hipcc, "llvm-objcopy")
  objdump = args.llvm_objdump or sibling_tool(args.hipcc, "llvm-objdump")
  bundler = args.clang_offload_bundler or sibling_tool(args.hipcc, "clang-offload-bundler")

  args.scratch_root.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix="rns8-wmma-isa-", dir=args.scratch_root) as scratch_dir:
    scratch = Path(scratch_dir)
    code_object = extract_code_object(objcopy, bundler, host_object, args.target, scratch)
    symbol_count = wmma_symbol_count(objdump, code_object)
    if symbol_count <= 0:
      raise RuntimeError(f"missing rocWMMA {WMMA_SYMBOL_MARKER} device kernel symbols")
    wmma_count, forbidden_stores, forbidden_divides = scan_disassembly(objdump, code_object, args.target)
    if wmma_count <= 0:
      raise RuntimeError(f"rocWMMA object does not contain required {REQUIRED_WMMA_MNEMONIC} matrix instructions")
    if forbidden_stores:
      raise RuntimeError(
          "rocWMMA object contains forbidden INT32 global/buffer stores:\n" + "\n".join(forbidden_stores[:20])
      )
    if forbidden_divides:
      raise RuntimeError(
          "rocWMMA object contains forbidden reciprocal/divide/remainder instructions:\n"
          + "\n".join(forbidden_divides[:20])
      )

  print("rocWMMA ISA check: PASS")
  print(f"object: {host_object}")
  print(f"target: {args.target}")
  print(f"- rocWMMA fused kernel symbols: {symbol_count}")
  print(f"- {REQUIRED_WMMA_MNEMONIC} instructions: {wmma_count}")
  print("- no global_store_dword/buffer_store_dword instructions")
  print("- no v/s reciprocal, divide, or remainder instructions")
  return 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except Exception as exc:  # noqa: BLE001 - command-line tool should print concise failure cause.
    print(f"rocWMMA ISA check: FAIL: {exc}", file=sys.stderr)
    raise SystemExit(1)
