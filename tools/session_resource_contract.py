"""Read-only native Linux resource/access preflight before session builds."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
from json_evidence import same_evidence_value

SCHEMA = "rns8_linux_session_resources_v1"
KFD_SOURCE = "https://github.com/torvalds/linux/blob/47f05f71ad988c3180bdba807151e6e86ed75aa3/drivers/gpu/drm/amd/amdkfd/kfd_topology.c"
MAX_FILE_BYTES = 128 * 1024


def read_text(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("native preflight file exceeds its byte allowance")
    return raw.decode("utf-8", errors="strict")


def decimal(text, *, positive=False):
    if type(text) is not str or not re.fullmatch(r"[0-9]{1,20}", text):
        raise ValueError("native resource field is not an unsigned decimal")
    value = int(text)
    if not int(positive) <= value < 1 << 64:
        raise ValueError("native resource field is outside uint64")
    return value


def cgroup_directories(cgroup, mountinfo):
    """Resolve the current unified hierarchy, including its visible ancestors."""
    groups = [line[3:] for line in cgroup.splitlines() if line.startswith("0::")]
    if len(groups) != 1:
        raise ValueError("one native cgroup-v2 membership required; v1 is unsupported")

    def absolute(value):
        # mountinfo escapes spaces, tabs, newlines and backslashes. Control
        # paths are unsupported rather than passed to another interpreter.
        value = re.sub(r"\\(040|011|012|134)", lambda m: chr(int(m[1], 8)), value)
        path = PurePosixPath(value)
        if not value.startswith("/") or any(ord(c) < 32 for c in value) or ".." in path.parts:
            raise ValueError("unsupported native cgroup path")
        return path

    group = absolute(groups[0])
    matches = []
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(" - ")
        if not separator or after.split(" ", 1)[0] != "cgroup2":
            continue
        fields = before.split(" ")
        if len(fields) < 6:
            raise ValueError("truncated cgroup mountinfo")
        root, mount = absolute(fields[3]), absolute(fields[4])
        if group.is_relative_to(root):
            matches.append((root, mount))
    if len(matches) != 1:
        raise ValueError("current cgroup has no unambiguous visible unified mount")
    root, mount = matches[0]
    current = mount / group.relative_to(root)
    result = []
    while True:
        result.append(Path(str(current)))
        if current == mount:
            return result
        if len(result) >= 64:
            raise ValueError("native cgroup ancestor budget exhausted")
        current = current.parent


def cgroup_limits(rows):
    """Limits include ancestors; a root's unlimited value cannot waive a child."""
    if not isinstance(rows, list) or not 1 <= len(rows) <= 64:
        raise ValueError("bounded current native cgroup hierarchy required")
    cpu, memory = [], []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("typed native cgroup observation required")
        if any(
            row.get(key) is not None and type(row[key]) is not str
            for key in ("cpu_max", "memory_max", "memory_current")
        ):
            raise ValueError("typed native cgroup file values required")
        if row.get("cpu_max") is not None:
            fields = row["cpu_max"].strip().split()
            if len(fields) != 2:
                raise ValueError("invalid native cpu.max")
            period = decimal(fields[1], positive=True)
            if fields[0] != "max":
                cpu.append(Fraction(decimal(fields[0], positive=True), period))
        if row.get("memory_max") is not None and row["memory_max"].strip() != "max":
            if row.get("memory_current") is None:
                raise ValueError("finite native memory limit lacks current usage")
            limit = decimal(row["memory_max"].strip())
            current = decimal(row["memory_current"].strip())
            memory.append(max(0, limit - current))
    quota = min(cpu) if cpu else None
    return dict(
        cpu_quota=None
        if quota is None
        else dict(numerator=quota.numerator, denominator=quota.denominator),
        memory_headroom_bytes=min(memory) if memory else None,
        scope="visible_cgroup_v2_ancestors_point_in_time_not_reserved_capacity",
    )


def kfd_properties(text):
    result = {}
    wanted = {
        "gfx_target_version",
        "vendor_id",
        "device_id",
        "location_id",
        "domain",
        "drm_render_minor",
        "fw_version",
        "sdma_fw_version",
        "unique_id",
    }
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] not in wanted:
            continue
        if len(fields) != 2 or fields[0] in result:
            raise ValueError("duplicate or malformed native KFD property")
        result[fields[0]] = decimal(fields[1])
    return result


def observe(output, *, gpu_required):
    """Only explicit execution calls this function; no tools or devices opened."""
    if platform.system() != "Linux":
        raise ValueError("native Linux preflight required")
    memory = re.findall(r"^MemAvailable:\s+([0-9]+) kB$", read_text("/proc/meminfo"), re.M)
    if len(memory) != 1:
        raise ValueError("native MemAvailable observation unavailable")
    result = dict(
        system="Linux",
        kernel=platform.release(),
        machine=platform.machine(),
        allowed_cpu_count=len(os.sched_getaffinity(0)),
        mem_available_bytes=decimal(memory[0]) * 1024,
        cgroup_state="unavailable",
        cgroup_limits=None,
        cgroup_observations=[],
        amdgpu_loaded=False,
        kfd_access=False,
        render_access=False,
        kfd_nodes=[],
        target_inventory_state="not_requested",
        driver_identity={},
    )
    filesystem = os.statvfs(Path(output).resolve(strict=True))
    result.update(
        disk_available_bytes=filesystem.f_bavail * filesystem.f_frsize,
        available_inodes=filesystem.f_favail,
    )
    try:
        directories = cgroup_directories(
            read_text("/proc/self/cgroup"), read_text("/proc/self/mountinfo")
        )
        rows = []
        for directory in directories:
            row = dict(path=str(directory))
            for key, filename in (
                ("cpu_max", "cpu.max"),
                ("memory_max", "memory.max"),
                ("memory_current", "memory.current"),
            ):
                path = directory / filename
                row[key] = read_text(path).strip() if path.is_file() else None
            rows.append(row)
        result.update(
            cgroup_state="observed", cgroup_limits=cgroup_limits(rows), cgroup_observations=rows
        )
    except (OSError, ValueError) as error:
        result["cgroup_error"] = str(error)
    if gpu_required:
        result["amdgpu_loaded"] = Path("/sys/module/amdgpu").is_dir()
        for name in ("version", "srcversion"):
            path = Path("/sys/module/amdgpu") / name
            if path.is_file():
                result["driver_identity"][name] = read_text(path).strip()[:256]

        def accessible(path):
            try:
                return stat.S_ISCHR(path.stat().st_mode) and os.access(path, os.R_OK | os.W_OK)
            except OSError:
                return False

        result["kfd_access"] = accessible(Path("/dev/kfd"))
        try:
            render = []
            for count, path in enumerate(Path("/dev/dri").iterdir()):
                if count >= 128:
                    raise ValueError("render device inventory budget exhausted")
                if re.fullmatch(r"renderD[0-9]{1,10}", path.name):
                    render.append(path)
            result["render_access"] = any(accessible(path) for path in render)
        except OSError:
            pass
        try:
            nodes = []
            for count, path in enumerate(Path("/sys/class/kfd/kfd/topology/nodes").iterdir()):
                if count >= 128:
                    raise ValueError("KFD node inventory budget exhausted")
                if not re.fullmatch(r"[0-9]{1,10}", path.name):
                    continue
                gpu_id = decimal(read_text(path / "gpu_id").strip())
                if not gpu_id:
                    continue
                nodes.append(
                    dict(
                        node_id=decimal(path.name),
                        gpu_id=gpu_id,
                        properties=kfd_properties(read_text(path / "properties")),
                    )
                )
            result.update(kfd_nodes=nodes, target_inventory_state="observed")
        except (OSError, ValueError) as error:
            result.update(target_inventory_state="unavailable", target_inventory_error=str(error))
    return result


def resource_policy(
    *,
    jobs,
    minimum_free_bytes,
    minimum_memory_bytes=0,
    peak_job_memory_bytes=0,
    memory_reserve_bytes=0,
    gpu_required=False,
    architecture=None,
):
    for name, value in (
        ("jobs", jobs),
        ("minimum_free_bytes", minimum_free_bytes),
        ("minimum_memory_bytes", minimum_memory_bytes),
        ("peak_job_memory_bytes", peak_job_memory_bytes),
        ("memory_reserve_bytes", memory_reserve_bytes),
    ):
        if (
            type(value) is not int
            or not (1 if name in {"jobs", "minimum_free_bytes"} else 0) <= value < 1 << 64
        ):
            raise ValueError("invalid session resource policy: " + name)
    if jobs > 256:
        raise ValueError("session resource jobs must be 1..256")
    if (
        type(gpu_required) is not bool
        or architecture not in {None, "gfx942"}
        or gpu_required != (architecture is not None)
    ):
        raise ValueError("explicit supported session target/access policy required")
    return dict(
        jobs=jobs,
        minimum_free_bytes=minimum_free_bytes,
        minimum_memory_bytes=minimum_memory_bytes,
        peak_job_memory_bytes=peak_job_memory_bytes,
        memory_reserve_bytes=memory_reserve_bytes,
        gpu_required=gpu_required,
        architecture=architecture,
    )


def evaluate(
    observed,
    *,
    jobs,
    minimum_free_bytes,
    minimum_memory_bytes=0,
    peak_job_memory_bytes=0,
    memory_reserve_bytes=0,
    gpu_required=False,
    architecture=None,
):
    policy = resource_policy(
        jobs=jobs,
        minimum_free_bytes=minimum_free_bytes,
        minimum_memory_bytes=minimum_memory_bytes,
        peak_job_memory_bytes=peak_job_memory_bytes,
        memory_reserve_bytes=memory_reserve_bytes,
        gpu_required=gpu_required,
        architecture=architecture,
    )
    if not isinstance(observed, dict) or observed.get("system") != "Linux":
        raise ValueError("native Linux resource observation required")
    for key in (
        "allowed_cpu_count",
        "mem_available_bytes",
        "disk_available_bytes",
        "available_inodes",
    ):
        if type(observed.get(key)) is not int or not 0 <= observed[key] < 1 << 64:
            raise ValueError("typed bounded native resource observation required: " + key)
    blockers, warnings = [], []
    cpus, memory = observed["allowed_cpu_count"], observed["mem_available_bytes"]
    limits = observed.get("cgroup_limits")
    if observed.get("cgroup_state") == "observed" and isinstance(limits, dict):
        if not same_evidence_value(limits, cgroup_limits(observed.get("cgroup_observations"))):
            raise ValueError("derived cgroup limits differ from actual recorded file values")
        quota = limits["cpu_quota"]
        if quota is not None:
            numerator, denominator = quota["numerator"], quota["denominator"]
            if (
                type(numerator) is not int
                or type(denominator) is not int
                or not 0 < numerator < 1 << 64
                or not 0 < denominator < 1 << 64
            ):
                raise ValueError("invalid observed cgroup CPU ratio")
            cpus = min(cpus, (numerator + denominator - 1) // denominator)
        if limits["memory_headroom_bytes"] is not None:
            value = limits["memory_headroom_bytes"]
            if type(value) is not int or not 0 <= value < 1 << 64:
                raise ValueError("invalid observed cgroup memory headroom")
            memory = min(memory, value)
    else:
        blockers.append(
            "native cgroup limits unavailable; inspect container/cgroup namespace before building"
        )
    required_memory = max(minimum_memory_bytes, jobs * peak_job_memory_bytes + memory_reserve_bytes)
    if jobs > cpus:
        blockers.append(
            "requested build jobs exceed observed affinity/quota; choose --jobs " + str(cpus)
            if cpus
            else "no allowed build CPU capacity; request an operator-approved CPU allocation"
        )
    if not memory or memory < required_memory:
        blockers.append(
            "insufficient observed host/cgroup memory for declared build/workload model"
        )
    if observed["disk_available_bytes"] < minimum_free_bytes or not observed["available_inodes"]:
        blockers.append("insufficient available disk bytes or inodes for declared session floor")
    if not peak_job_memory_bytes and not minimum_memory_bytes:
        warnings.append(
            "host peak-memory model not supplied; positive available memory does not prove build headroom"
        )
    if gpu_required:
        for key, remedy in (
            ("amdgpu_loaded", "load a supported driver through the operator"),
            ("kfd_access", "request operator-approved /dev/kfd access"),
            ("render_access", "request operator-approved render-device access"),
        ):
            if type(observed.get(key)) is not bool:
                raise ValueError("typed native device-access observation required")
            if not observed[key]:
                blockers.append(key + ": " + remedy)
        if observed.get("target_inventory_state") != "observed" or not any(
            row.get("properties", {}).get("vendor_id") == 0x1002
            and row.get("properties", {}).get("gfx_target_version") == 90402
            for row in observed.get("kfd_nodes", [])
            if isinstance(row, dict)
        ):
            blockers.append("no positive native AMD gfx942 KFD target observation")
        warnings.append(
            "KFD node identity does not map selected HIP ordinals; native post-build inspection must verify each requested device"
        )
    return dict(
        schema=SCHEMA,
        process_id=os.getpid(),
        state="blocked" if blockers else "resource_access_checks_passed",
        observed=observed,
        blockers=blockers,
        warnings=warnings,
        policy=policy,
        effective_build_cpu_ceiling=cpus,
        observed_memory_headroom_bytes=memory,
        required_memory_bytes=required_memory,
        memory_is_reserved=False,
        kfd_producer_source=KFD_SOURCE,
        supported_driver_OS_tuple_qualified=False,
        selected_runtime_device_mapping_verified=False,
        hardware_qualified=False,
        promotion_allowed=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=int)
    parser.add_argument("--minimum-free-bytes", required=True, type=int)
    parser.add_argument("--minimum-memory-bytes", default=0, type=int)
    parser.add_argument("--peak-job-memory-bytes", default=0, type=int)
    parser.add_argument("--memory-reserve-bytes", default=0, type=int)
    parser.add_argument("--architecture", choices=("gfx942",))
    args = parser.parse_args(argv)
    try:
        policy = resource_policy(
            jobs=args.jobs,
            minimum_free_bytes=args.minimum_free_bytes,
            minimum_memory_bytes=args.minimum_memory_bytes,
            peak_job_memory_bytes=args.peak_job_memory_bytes,
            memory_reserve_bytes=args.memory_reserve_bytes,
            gpu_required=args.architecture is not None,
            architecture=args.architecture,
        )
        report = evaluate(observe(args.output_root, gpu_required=policy["gpu_required"]), **policy)
        print(json.dumps(report, allow_nan=False))
        return int(bool(report["blockers"]))
    except (OSError, ValueError) as error:
        print(
            json.dumps(
                dict(
                    schema=SCHEMA,
                    process_id=os.getpid(),
                    state="blocked",
                    error=str(error),
                    hardware_qualified=False,
                    promotion_allowed=False,
                )
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
