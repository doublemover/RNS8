"""Shared helpers for RNS8 HIP/AMDGPU ISA inspection scripts."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEVICE_FUNCTION_MARKER = " F .text"
MNEMONIC_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\b")
FORBIDDEN_DIVIDE_RE = re.compile(r"\b[sv]_(?:div|rem|rcp)[_a-z0-9]*\b")
FORBIDDEN_INT32_GLOBAL_STORE_RE = re.compile(r"\b(?:global|buffer)_store_dword(?:x[234])?\b")
FORBIDDEN_MATRIX_ENGINE_RE = re.compile(r"\bv_(?:wmma|mfma)[_a-z0-9]*\b")


@dataclass(frozen=True)
class IsaToolConfig:
    host_object: Path
    target: str
    objcopy: str
    objdump: str
    bundler: str
    scratch_root: Path


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
    fatbin_name: str,
) -> Path:
    fatbin = scratch / fatbin_name
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


def parse_isa_tool_config(description: str | None, object_help: str, missing_label: str) -> IsaToolConfig:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--object", required=True, type=Path, help=object_help)
    parser.add_argument("--target", required=True, help="AMDGPU target id, for example gfx1100")
    parser.add_argument("--hipcc", type=Path, help="HIP compiler path; sibling LLVM tools are preferred")
    parser.add_argument("--llvm-objcopy", help="Override llvm-objcopy path")
    parser.add_argument("--llvm-objdump", help="Override llvm-objdump path")
    parser.add_argument("--clang-offload-bundler", help="Override clang-offload-bundler path")
    parser.add_argument("--scratch-root", type=Path, default=Path("temp"), help="Ignored scratch directory")
    args = parser.parse_args()

    host_object = args.object
    if not host_object.exists():
        raise RuntimeError(f"{missing_label} does not exist: {host_object}")

    return IsaToolConfig(
        host_object=host_object,
        target=args.target,
        objcopy=args.llvm_objcopy or sibling_tool(args.hipcc, "llvm-objcopy"),
        objdump=args.llvm_objdump or sibling_tool(args.hipcc, "llvm-objdump"),
        bundler=args.clang_offload_bundler or sibling_tool(args.hipcc, "clang-offload-bundler"),
        scratch_root=args.scratch_root,
    )


@contextmanager
def extracted_device_code_object(
    config: IsaToolConfig,
    temp_prefix: str,
    fatbin_name: str,
):
    config.scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=temp_prefix, dir=config.scratch_root) as scratch_dir:
        yield extract_code_object(
            config.objcopy,
            config.bundler,
            config.host_object,
            config.target,
            Path(scratch_dir),
            fatbin_name,
        )


def device_function_symbols(objdump: str, code_object: Path) -> list[str]:
    symbols = run_command([objdump, "-t", str(code_object)])
    return [line.split()[-1] for line in symbols.splitlines() if DEVICE_FUNCTION_MARKER in line]


def symbols_matching_markers(
    objdump: str,
    code_object: Path,
    markers: list[str],
    description: str,
) -> list[str]:
    names = [
        name
        for name in device_function_symbols(objdump, code_object)
        if any(marker in name for marker in markers)
    ]
    missing = [marker for marker in markers if not any(marker in name for name in names)]
    if missing:
        raise RuntimeError(f"missing expected {description}: {', '.join(missing)}")
    return names


def symbol_count_matching_marker(objdump: str, code_object: Path, marker: str) -> int:
    return sum(1 for name in device_function_symbols(objdump, code_object) if marker in name)


def disassemble_code_object(
    objdump: str,
    code_object: Path,
    amdgpu_target: str,
    symbol: str | None = None,
) -> str:
    command = [objdump, "-d", f"--mcpu={amdgpu_target}"]
    if symbol is not None:
        command.append(f"--disassemble-symbols={symbol}")
    command.append(str(code_object))
    return run_command(command)


def mnemonic_lines(disassembly: str) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    for line in disassembly.splitlines():
        match = MNEMONIC_RE.match(line)
        if match:
            lines.append((line.strip(), match.group(1)))
    return lines


def forbidden_mnemonic_lines(disassembly: str, pattern: re.Pattern[str]) -> list[str]:
    return [line for line, mnemonic in mnemonic_lines(disassembly) if pattern.search(mnemonic)]
