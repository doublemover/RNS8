"""ROCm GPU-ASan prerequisites and artifact observations, never a device PASS."""

from __future__ import annotations

import re
from pathlib import Path
from device_observation import runtime_identity

from host_instrumentation_contract import selected_flags

RELEASE = "10.1.0"
TARGETS = ("gfx942:xnack+", "gfx950:xnack+")
CORRECTNESS_CASES = (
    "qualification pack covers dispatch thresholds and plane tails",
    "minimal host allocations pack logical cells for every integer storage contract",
    "finite exports cover scalar quad and block tails with logical padding canaries",
    "HIP ragged exact extension feeds a GEMM crossing source tile boundaries",
    "qualification persistent GEMM spans multiple safe K blocks",
    "private HIP buffer moves and rejected replacement preserve allocation ownership",
    "direct HIP explicit split K reduces byte partials through bounded merge levels",
)


def runtime_binding(value: dict, contract: dict, loaded: dict, *, entering: bool) -> dict:
    fields = {
        "schema",
        "visible_hip_devices",
        "current_hip_ordinal",
        "runtime_version",
        "driver_version",
        "architecture",
        "runtime_device_identity",
        "hsa_gpu_count",
        "hsa_uuid",
        "hsa_name",
        "hsa_bdfid",
        "hsa_domain",
        "free_device_bytes",
        "total_device_bytes",
        "hip_symbol_owner",
        "hsa_symbol_owner",
        "scope",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or value["schema"] != "rns8_device_asan_runtime_snapshot_v1"
        or value["scope"] != "single_visible_hsa_uuid_and_pci_join_to_observed_hip_ordinal_zero"
    ):
        raise ValueError("complete native device-ASan runtime observation required")
    if (
        type(contract) is not dict
        or set(contract) != {"target", "agent_uuid", "runtimes", "minimum_free_bytes"}
        or contract["target"] not in TARGETS
        or type(contract["minimum_free_bytes"]) is not int
        or not 64 * 1024**2 <= contract["minimum_free_bytes"] <= 2**40
    ):
        raise ValueError("explicit reviewed device runtime contract required")
    for key, limit in (
        ("visible_hip_devices", 2**32),
        ("current_hip_ordinal", 2**31),
        ("hsa_gpu_count", 2**32),
        ("runtime_version", 2**32),
        ("driver_version", 2**32),
        ("hsa_bdfid", 2**16),
        ("hsa_domain", 2**16),
        ("free_device_bytes", 2**64),
        ("total_device_bytes", 2**64),
    ):
        if type(value[key]) is not int or not 0 <= value[key] < limit:
            raise ValueError(
                "native device witness integer or unsupported partition/domain encoding"
            )
    base = contract["target"].split(":")[0]
    architecture = value["architecture"]
    parts = architecture.split(":") if type(architecture) is str else []
    if (
        value["visible_hip_devices"] != 1
        or value["current_hip_ordinal"] != 0
        or value["hsa_gpu_count"] != 1
        or not value["runtime_version"]
        or not value["driver_version"]
        or not parts
        or parts[0] != base
        or "xnack+" not in parts[1:]
        or len(parts) != len(set(parts))
        or any(part not in {"xnack+", "sramecc+", "sramecc-"} for part in parts[1:])
        or {"sramecc+", "sramecc-"}.issubset(parts[1:])
        or value["hsa_name"] != base
        or type(value["hsa_uuid"]) is not str
        or not re.fullmatch(r"GPU-[0-9a-fA-F]{16}", value["hsa_uuid"])
        or value["hsa_uuid"].lower() != str(contract["agent_uuid"]).lower()
        or int(value["hsa_uuid"][4:], 16) == 0
        or not 0 < value["free_device_bytes"] <= value["total_device_bytes"]
        or entering
        and value["free_device_bytes"] < contract["minimum_free_bytes"]
    ):
        raise ValueError(
            "native device/runtime/agent/features or available-memory admission disagree"
        )
    identity = runtime_identity(value["runtime_device_identity"])
    bdf, domain = value["hsa_bdfid"], value["hsa_domain"]
    hsa_pci = f"{domain:04x}:{bdf >> 8:02x}:{(bdf >> 3) & 31:02x}.{bdf & 7}"
    if identity.runtime_ordinal != 0 or identity.pci_bdf != hsa_pci:
        raise ValueError("native HSA agent PCI does not identify workload HIP ordinal zero")
    files = {row["path"]: row for row in loaded["files"]}
    runtimes = contract["runtimes"]
    if type(runtimes) is not dict or set(runtimes) != {"hip", "hsa", "asan"}:
        raise ValueError("exact selected HIP/HSA/ASan runtime identities required")
    for name, reference in runtimes.items():
        path = Path(reference["path"]).resolve(strict=True)
        if str(path) not in files or files[str(path)]["sha256"] != reference["sha256"]:
            raise ValueError("actual device application did not load selected runtime: " + name)
        owner = (
            value.get(name + "_symbol_owner")
            if name != "asan"
            else loaded["symbol_owners"]["sanitizer"]
        )
        if type(owner) is not str or Path(owner).resolve(strict=True) != path:
            raise ValueError(
                "actual device runtime symbol owner differs from selected runtime: " + name
            )
    return dict(
        target=base,
        architecture=architecture,
        agent_uuid=value["hsa_uuid"],
        pci_bdf=hsa_pci,
        runtime_identity=identity.to_json(),
        runtime_version=value["runtime_version"],
        driver_version=value["driver_version"],
        total_device_bytes=value["total_device_bytes"],
    )


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


def hip_component_version(native: str, header_version: str | None) -> str:
    # Core SDK releases and HIP component versions have independent numbering.
    versions = re.findall(
        r"^\s*HIP version:\s*(\d+\.\d+\.\d+)(?:[-+][A-Za-z0-9._+-]+)?\s*$",
        native,
        re.I | re.M,
    )
    if len(versions) != 1 or not header_version or versions[0] != header_version:
        raise ValueError("selected hipcc and its actual HIP version header disagree")
    return versions[0]
