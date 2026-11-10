"""Bounded capture with terminal resource observations from one owned child."""

from __future__ import annotations

import math
from pathlib import Path
import platform
import sys

from json_evidence import load_evidence_json
from owned_child_resources import SCHEMA
from process_progress import run_capture


def resource_record(document, command, supervisor):
    """Validate native measurements without conflating child and supervisor PIDs."""
    if (
        not isinstance(document, dict)
        or type(supervisor.get("process_id")) is not int
        or not 0 < supervisor["process_id"] < 2**32
        or document.get("schema") != SCHEMA
        or document.get("command") != command
        or type(document.get("process_id")) is not int
        or not 0 < document["process_id"] < 2**32
        or document["process_id"] == supervisor.get("process_id")
        or type(document.get("returncode")) is not int
        or document["returncode"] != supervisor.get("returncode")
        or document.get("elapsed_scope") != "owned_child_spawn_to_terminal_resource_observation"
        or type(document.get("elapsed_seconds")) not in (float, int)
        or not math.isfinite(document["elapsed_seconds"])
        or document["elapsed_seconds"] <= 0
        or document["elapsed_seconds"] > supervisor.get("elapsed_seconds", 0)
        or type(document.get("worker_python_executable")) is not str
        or not Path(document["worker_python_executable"]).is_absolute()
        or type(supervisor.get("command")) is not list
        or not supervisor["command"]
        or str(Path(supervisor["command"][0]).resolve()) != document["worker_python_executable"]
        or not isinstance(document.get("worker_python_version"), str)
    ):
        raise ValueError("owned child command, PID, termination or duration differs")
    expected = {
        "Windows": (
            "windows_K32GetProcessMemoryInfo_PeakWorkingSetSize",
            "bytes",
            "single_owned_child_peak_working_set_excludes_descendants",
            1,
        ),
        "Linux": (
            "linux_wait4_ru_maxrss",
            "KiB",
            "waited_child_and_accounted_descendants_maxrss_not_concurrent_tree_total",
            1024,
        ),
    }.get(document.get("os"))
    if (
        expected is None
        or tuple(document.get(k) for k in ("source", "units", "scope")) != expected[:3]
    ):
        raise ValueError("unknown native resource source or scope")
    if document.get("status") == "observed":
        if (
            type(document.get("raw_peak")) is not int
            or document["raw_peak"] <= 0
            or type(document.get("peak_memory_bytes")) is not int
            or document["peak_memory_bytes"] != document["raw_peak"] * expected[3]
            or document.get("resource_error") is not None
        ):
            raise ValueError("native peak memory conversion is invalid")
    elif (
        document.get("status") != "not_observed"
        or document.get("peak_memory_bytes") is not None
        or not isinstance(document.get("resource_error"), str)
        or not document["resource_error"]
    ):
        raise ValueError("missing native resource observation needs an explicit reason")
    return document


def observed_capture(command: list[str], *, resource_path: Path, **kwargs) -> dict:
    """The supervisor retains time/output/tree limits; the worker owns native RSS."""
    if resource_path.exists() or not resource_path.parent.is_dir():
        raise ValueError("fresh resource receipt in an existing owned directory required")
    worker = Path(__file__).with_name("owned_child_resources.py")
    wrapper = [sys.executable, str(worker), "--result", str(resource_path), "--", *command]
    value = run_capture(wrapper, **kwargs)
    supervisor = {
        key: value[key] for key in ("command", "process_id", "returncode", "elapsed_seconds")
    }
    value.update(
        command=command,
        resource_supervisor=supervisor,
        resource_artifact=str(resource_path),
        child_resources=None,
        resource_observation_error=None,
        process_identity_scope="resource_supervisor_without_child_observation",
    )
    try:
        if resource_path.stat().st_size > 16384:
            raise ValueError("native resource receipt exceeds its byte budget")
        document = resource_record(load_evidence_json(resource_path), command, supervisor)
        if document["os"] != platform.system():
            raise ValueError("child resource platform differs from the executing host")
        value.update(
            process_id=document["process_id"],
            process_identity_scope="observed_owned_native_child",
            child_resources=document,
        )
    except (OSError, ValueError) as error:
        value["resource_observation_error"] = str(error)
    return value
