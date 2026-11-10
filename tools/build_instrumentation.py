"""Read compiled build instrumentation without confusing provenance with latency."""

from __future__ import annotations

import re

_FLAGS = re.compile(
    r"(?:^|[\s;\"'])(?:-fsanitize(?:=|-)|/fsanitize[:=]|-fprofile-instr-generate(?:=|\s|$)|"
    r"-fcoverage-mapping(?:\s|$)|--coverage(?:\s|$)|-fprofile-arcs(?:\s|$)|-ftest-coverage(?:\s|$)|"
    r"-fpass-plugin(?:=|\s|$))",
    re.IGNORECASE,
)


def instrumentation_reasons(manifest: object) -> list[str]:
    # Older receipts have no explicit selector. Existing provenance consumers
    # still own missing-manifest rejection; inspect their recorded flags here.
    if not isinstance(manifest, dict):
        return []
    configuration = manifest.get("configuration")
    if not isinstance(configuration, str):
        return ["build_instrumentation_configuration_unreadable"]
    reasons: set[str] = set()
    for line in configuration.splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key, value = key.strip().lower(), value.strip()
        if key == "host_sanitizer" and value.lower() != "none":
            reasons.add("host_sanitizer_build")
        if key in {"host_coverage", "windows_clang_asan", "device_asan"} and value.lower() not in {
            "off",
            "false",
            "0",
            "no",
        }:
            reasons.add(key + "_build")
        if (key == "flags" or key.startswith("flags_") or key.endswith("_flags")) and _FLAGS.search(
            value
        ):
            reasons.add("compiler_instrumentation_flags")
    return sorted(reasons)
