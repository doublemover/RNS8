"""Authored native-format Direct-HIP report controls; never measured evidence."""

from pathlib import Path
from diagnostic_companion import build_binding
from report_capture_inputs import load_report_captures
from test_diagnostic_companion import authored_capture_pair
from test_review_reference_support import add_synthetic_full_reference
from test_tile_shape_report import make_cpu_capture, make_direct_capture


def direct_resident_capture(kernel: str, duration: int) -> dict:
    capture = make_direct_capture(128, 128, "default", duration)
    previous = capture["selected_kernel"]
    capture["selected_kernel"] = kernel
    capture["backend_metadata"]["selected_kernel"] = kernel
    capture["backend_metadata"]["autotune_key"] = capture["backend_metadata"][
        "autotune_key"
    ].replace(previous, kernel)
    capture["tile_shape_variant"]["selected_kernel_identity"] = kernel
    capture["tile_shape_variant"]["resource_report_key"] = capture["tile_shape_variant"][
        "resource_report_key"
    ].replace(previous, kernel)
    return capture


def collect_resident_processes(
    root: Path, label: str, kernels: dict[str, tuple[str, int]], change=None
):
    paths, bindings = [], {}
    for mode in ("cpu", *kernels):
        for run in range(2):
            fixture = (
                make_cpu_capture(200) if mode == "cpu" else direct_resident_capture(*kernels[mode])
            )
            fixture["benchmark_execution_mode"] = "persistent_resident_matrices"
            if change is not None:
                change(fixture, mode, run)
            add_synthetic_full_reference(fixture)
            lane = root / f"{label}-{mode}-{run}"
            lane.mkdir()
            _, _, pair = authored_capture_pair(lane, fixture)
            paths.append(pair[0])
            if mode != "cpu":
                bindings[str(pair[0].resolve())] = build_binding(*pair)
    return load_report_captures(paths), paths, bindings
