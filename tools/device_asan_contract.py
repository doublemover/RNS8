"""ROCm GPU-ASan prerequisites and artifact observations, never a device PASS."""

from __future__ import annotations

import re

from host_instrumentation_contract import selected_flags

RELEASE = "10.1.0"
TARGETS = ("gfx942:xnack+", "gfx950:xnack+")
HMM_KEYS = (
    "CONFIG_HMM_MIRROR",
    "CONFIG_DEVICE_PRIVATE",
    "CONFIG_ZONE_DEVICE",
    "CONFIG_MMU_NOTIFIER",
    "CONFIG_MIGRATION",
)


def support(system: str, release: str, target: str) -> dict:
    supported = system == "Linux" and release == RELEASE and target in TARGETS
    return dict(
        official_support="supported" if supported else "unsupported",
        release=release,
        target=target,
        system=system,
        device_execution="not_run",
        detector_sensitivity="not_run_separate_authorization_required",
        production_latency=False,
        sources=[
            "https://rocm.docs.amd.com/en/docs-10.1.0/install/asan.html",
            "https://rocmdocs.amd.com/projects/llvm-project/en/latest/conceptual/using-gpu-sanitizer.html",
        ],
    )


def host_prerequisites(kernel: str, configuration: str, environment: dict) -> dict:
    match = re.match(r"(\d+)\.(\d+)(?:\.|-)", kernel)
    if (
        not match
        or tuple(map(int, match.groups())) < (5, 6)
        or re.search(r"microsoft|wsl", kernel, re.I)
    ):
        raise ValueError("native Linux kernel >=5.6 required; version alone does not prove HMM")
    values = {}
    for line in configuration.splitlines():
        if line.startswith("CONFIG_") and "=" in line:
            key, value = line.split("=", 1)
            if key in values:
                raise ValueError("ambiguous native kernel configuration")
            values[key] = value
    missing = [key for key in HMM_KEYS if values.get(key) != "y"]
    if missing:
        raise ValueError("native kernel HMM prerequisites unavailable: " + ",".join(missing))
    if environment.get("HSA_XNACK") != "1" or environment.get("HSA_OVERRIDE_GFX_VERSION"):
        raise ValueError("actual HSA_XNACK=1 and no fabricated target override required")
    return dict(
        kernel=kernel,
        configuration={key: values[key] for key in HMM_KEYS},
        HSA_XNACK="1",
        scope="kernel_build_prerequisites_not_successful_gpu_execution",
    )


def agent_observation(text: str, target: str, index: int) -> dict:
    if target not in TARGETS or type(index) is not int or index < 0:
        raise ValueError("explicit supported GPU agent required")
    headers = list(re.finditer(r"^\s*Agent\s+(\d+)\s*$", text, re.M))
    if (
        not headers
        or len(headers) > 1024
        or len({int(item[1]) for item in headers}) != len(headers)
    ):
        raise ValueError("native rocminfo agent inventory missing or ambiguous")
    blocks = {
        int(item[1]): text[
            item.end() : headers[i + 1].start() if i + 1 < len(headers) else len(text)
        ]
        for i, item in enumerate(headers)
    }
    block = blocks.get(index, "")
    base = target.split(":")[0]
    names = re.findall(r"^\s*Name:\s*(gfx[0-9a-f]+)\s*$", block, re.M)
    uuids = re.findall(
        r"^\s*Uuid:\s*(GPU-(?:[0-9A-Fa-f]{32}|[0-9A-Fa-f]{16}))\s*$", block, re.M | re.I
    )
    isas = re.findall(r"^\s*Name:\s*(amdgcn-amd-amdhsa--\S+)\s*$", block, re.M)
    eligible = []
    for isa in isas:
        parts = isa.removeprefix("amdgcn-amd-amdhsa--").split(":")
        if parts[0] == base and "xnack+" in parts[1:] and "xnack-" not in parts[1:]:
            if (
                len(parts[1:]) != len(set(parts[1:]))
                or any(feature not in {"xnack+", "sramecc+", "sramecc-"} for feature in parts[1:])
                or {"sramecc+", "sramecc-"}.issubset(parts[1:])
            ):
                raise ValueError("ambiguous native agent ISA feature list")
            eligible.append(isa)
    if (
        names != [base]
        or len(uuids) != 1
        or int(uuids[0][4:], 16) == 0
        or not eligible
        or len(eligible) != len(set(eligible))
    ):
        raise ValueError("selected native rocminfo GPU/UUID/xnack+ ISA not observed")
    return dict(
        agent_index=index,
        uuid=uuids[0],
        target=base,
        advertised_isas=isas,
        selected_xnack_isas=eligible,
        scope="native_agent_advertisement_not_workload_device_binding",
    )


def compiler_flags(argv: list[str], *, target: str | None = None) -> dict:
    flags = selected_flags(argv, "device-address")
    arches = [
        arg.removeprefix("--offload-arch=") for arg in argv if arg.startswith("--offload-arch=")
    ]
    if target is not None and (target not in TARGETS or arches != [target]):
        raise ValueError("actual HIP command must select exactly the reviewed xnack+ target")
    if any(
        arg
        in {
            "-g0",
            "-static-libsan",
            "-static-libasan",
            "-static",
            "--cuda-host-only",
            "--hip-host-only",
        }
        or arg.startswith(("-fno-sanitize", "--offload-arch"))
        and arg
        not in {
            "-fno-sanitize-recover=address",
            "-fno-sanitize-recover=all",
            "--offload-arch=" + str(target),
        }
        for arg in argv
    ):
        raise ValueError("device sanitizer disabled, host-only or incompatible runtime override")
    return flags


def dynamic_dependencies(text: str) -> list[str]:
    rows = re.findall(r"\(NEEDED\)\s+Shared library:\s*\[([^\]\r\n]+)\]", text)
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("native ELF dynamic dependencies missing or ambiguous")
    if not any(re.fullmatch(r"libclang_rt\.asan-[A-Za-z0-9_]+\.so", name) for name in rows):
        raise ValueError("application does not directly depend on shared ASan runtime")
    return rows
