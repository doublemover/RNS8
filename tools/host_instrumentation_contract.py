"""Native host object and detector observations, separate from test PASS counts."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shlex

from process_progress import capture_succeeded

LANES = {"address-undefined", "thread", "windows-address", "coverage"}
EXIT_CODES = {"heap-overread": 86, "signed-overflow": 87, "race": 88, "openmp-race": 88}
DIAGNOSTIC = {
    "heap-overread": r"ERROR: AddressSanitizer: heap-buffer-overflow",
    "signed-overflow": r"runtime error: signed integer overflow",
    "race": r"WARNING: ThreadSanitizer: data race",
    "openmp-race": r"WARNING: ThreadSanitizer: data race",
}
SYMBOLS = {
    "address-undefined": {"address": r"\b_?__asan_", "undefined": r"\b_?__ubsan_"},
    "thread": {"thread": r"\b_?__tsan_"},
    "windows-address": {"address": r"\b_?__asan_"},
    "coverage": {"profile": r"\b_?__llvm_pr(?:of|f)_|\b_?__llvm_profile_|\b_?__prof[cdv]_"},
}


def native_argv(record: dict, *, windows: bool) -> list[str]:
    args, command = record.get("arguments"), record.get("command")
    if args is not None:
        if (
            command is not None
            or not isinstance(args, list)
            or any(type(x) is not str for x in args)
        ):
            raise ValueError("ambiguous native compilation argv")
        result = list(args)
    elif type(command) is str and command:
        if windows:
            if os.name != "nt":
                raise ValueError("Windows command lines require the native Windows parser")
            import ctypes
            from ctypes import wintypes

            shell = ctypes.WinDLL("shell32", use_last_error=True)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            parse = shell.CommandLineToArgvW
            parse.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
            parse.restype = ctypes.POINTER(wintypes.LPWSTR)
            kernel.LocalFree.argtypes = [wintypes.HLOCAL]
            kernel.LocalFree.restype = wintypes.HLOCAL
            count = ctypes.c_int()
            pointer = parse(command, ctypes.byref(count))
            if not pointer:
                raise ValueError("native Windows compilation command parse failed")
            try:
                result = [pointer[i] for i in range(count.value)]
            finally:
                kernel.LocalFree(ctypes.cast(pointer, wintypes.HLOCAL))
        else:
            result = shlex.split(command, posix=True)
    else:
        raise ValueError("native compilation command missing")
    if not result or not result[0] or len(result) > 16384 or any("\0" in arg for arg in result):
        raise ValueError("invalid native compilation argv")
    if any(arg in {"&&", "||", "|", ";", ">", "<"} or arg.startswith("@") for arg in result):
        raise ValueError("shell composition/response files need separately bound expansion")
    return result


def object_path(record: dict, argv: list[str], build: Path, *, windows: bool) -> Path:
    output = record.get("output")
    outputs = []
    for i, arg in enumerate(argv):
        if arg == "-o" and i + 1 < len(argv):
            outputs.append(argv[i + 1])
        elif windows and arg.lower().startswith("/fo") and len(arg) > 3:
            outputs.append(arg[3:])
    if len(outputs) != 1:
        raise ValueError("one explicit compiler object output required")
    directory = Path(record["directory"])
    actual = (directory / outputs[0]).resolve()
    if output is not None and (type(output) is not str or (directory / output).resolve() != actual):
        raise ValueError("compile database output differs from actual argv")
    if not actual.is_relative_to(build.resolve()) or actual.suffix.lower() not in {".o", ".obj"}:
        raise ValueError("object output is outside the selected build")
    return actual


def selected_flags(argv: list[str], lane: str) -> dict:
    if lane not in LANES:
        raise ValueError("unknown host instrumentation lane")
    lower = [arg.lower() for arg in argv]
    if any(
        arg != "-fno-sanitize-recover=all"
        and arg.startswith(
            (
                "-fno-sanitize",
                "-fsanitize-ignorelist",
                "-fsanitize-blacklist",
                "-fsanitize-recover",
                "-fno-profile",
                "-fno-coverage",
                "--config",
                "-config",
            )
        )
        for arg in lower
    ):
        raise ValueError("unreviewed instrumentation override or suppression")
    selected = set()
    for arg in lower:
        if arg.startswith("-fsanitize="):
            selected.update(arg.split("=", 1)[1].split(","))
        elif arg in {"/fsanitize=address", "/fsanitize:address"}:
            selected.add("address")
    expected = {
        "address-undefined": {"address", "undefined"},
        "thread": {"thread"},
        "windows-address": {"address"},
        "coverage": set(),
    }[lane]
    if selected != expected:
        raise ValueError("effective native command lacks the exact selected sanitizer")
    if lane == "coverage":
        if not {"-fprofile-instr-generate", "-fcoverage-mapping"}.issubset(lower):
            raise ValueError("native command lacks complete source coverage instrumentation")
    elif any(
        arg.startswith(("-fprofile-instr-generate", "-fcoverage-mapping", "--coverage"))
        for arg in lower
    ):
        raise ValueError("sanitizer and coverage commands may not share objects")
    if lane == "address-undefined" and "-fno-sanitize-recover=all" not in lower:
        raise ValueError("required UB checks must be nonrecovering")
    return dict(lane=lane, sanitizer_classes=sorted(selected), configured_flags_verified=True)


def symbol_classes(source: str, lane: str) -> list[str]:
    if lane not in LANES:
        raise ValueError("unknown host instrumentation lane")
    return sorted(name for name, pattern in SYMBOLS[lane].items() if re.search(pattern, source))


def detector_sites(path: Path) -> dict[str, int]:
    sites = {}
    for line, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.search(r"// rns8-detector-site:([a-z-]+)\s*$", text)
        if match:
            mode = match[1]
            if mode not in DIAGNOSTIC or mode in sites:
                raise ValueError("invalid or repeated native detector source site")
            sites[mode] = line
    if set(sites) != set(DIAGNOSTIC):
        raise ValueError("native detector source sites incomplete")
    return sites


def detector_result(mode: str, result: dict, *, source_line: int) -> dict:
    if mode not in DIAGNOSTIC:
        raise ValueError("unknown detector control")
    rc = result.get("returncode")
    if type(source_line) is not int or source_line <= 0:
        raise ValueError("positive native detector source line required")
    source = str(result.get("stdout", "")) + "\n" + str(result.get("stderr", ""))
    if (
        type(rc) is not int
        or result.get("timed_out") is not False
        or result.get("limit_exceeded") is not None
        or result.get("capture_error") is not None
        or result.get("interrupted") not in (None, False)
    ):
        status, reason = "blocked", "incomplete_owned_process_capture"
    elif rc == 77:
        status, reason = "unsupported", "control_reported_unsupported"
    elif rc == 0:
        status, reason = "failed", "intended_defect_was_not_detected"
    elif (
        rc != EXIT_CODES[mode]
        or not re.search(DIAGNOSTIC[mode], source)
        or not re.search(
            (
                r"host_sanitizer_control\.cpp:"
                if mode == "signed-overflow"
                else r"#0[^\n]*host_sanitizer_control\.cpp:"
            )
            + str(source_line)
            + r"(?=[:\s]|$)",
            source,
        )
    ):
        status, reason = "failed", "exit_or_diagnostic_does_not_match_intended_defect"
    else:
        pid = result.get("process_id")
        if type(pid) is not int or pid <= 0:
            status, reason = "blocked", "actual_control_process_identity_missing"
        elif mode == "heap-overread" and not re.search(
            r"==" + str(pid) + r"==\s*ERROR: AddressSanitizer", source
        ):
            status, reason = "failed", "asan_diagnostic_process_mismatch"
        elif mode in {"race", "openmp-race"} and not re.search(
            r"\(pid=" + str(pid) + r"\)", source
        ):
            status, reason = "failed", "tsan_diagnostic_process_mismatch"
        else:
            status, reason = "passed", "intended_host_defect_detected"
    return dict(
        mode=mode,
        status=status,
        reason=reason,
        expected_exit_code=EXIT_CODES[mode],
        actual_returncode=rc,
        expected_source_line=source_line,
        application_pass_count=0,
        production_latency=False,
        scope="disposable_host_control_not_hip_device",
    )


def valid_result(result: dict, document: object) -> dict:
    valid = (
        capture_succeeded(result)
        and isinstance(document, dict)
        and document.get("schema") == "rns8_host_detector_valid_v1"
        and type(document.get("value")) is int
        and document["value"] == 128
        and type(document.get("expected")) is int
        and document["expected"] == 128
    )
    return dict(
        status="passed" if valid else "failed",
        mode="synchronized_host_control",
        application_pass_count=0,
    )
