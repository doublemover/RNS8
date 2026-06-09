"""Focused validators split out of benchmark_schema.core."""

from __future__ import annotations

from typing import Any

from .core import (
    ACCELERATOR_GPU_EVENT_SCOPES,
    DEEP_ACCELERATOR_GPU_EVENT_SCOPE,
    DIRECT_HIP_FINITE_ONESHOT_GPU_EVENT_PHASES,
    DIRECT_HIP_GPU_EVENT_SCOPES,
    DIRECT_HIP_ONESHOT_GPU_EVENT_PHASES,
    DIRECT_HIP_ONESHOT_RESIDENT_FALLBACK_GPU_EVENT_PHASES,
    HIPBLASLT_GPU_EVENT_SCOPES,
    OLD_ACCELERATOR_GPU_EVENT_SCOPE,
    VECTOR_ALU_GPU_EVENT_SCOPES,
    _is_int,
    _is_number,
    wrap64_hip_expected_gemm_event_label,
)

def validate_expected_gpu_event_phases(self, scope: Any, phases: list[str]) -> None:
    backend = self.data.get("backend_selected")
    if self._is_finite_oneshot_capture() and backend == "hip-direct":
        expected = DIRECT_HIP_FINITE_ONESHOT_GPU_EVENT_PHASES
        if phases != expected:
            missing = [phase for phase in expected if phase not in phases]
            extra = [phase for phase in phases if phase not in expected]
            if missing:
                self._error(f"direct-HIP finite one-shot GPU event phase set is incomplete; missing {', '.join(missing)}")
            if extra:
                self._error(f"direct-HIP finite one-shot GPU event phase set contains undeclared phases: {', '.join(extra)}")
            if not missing and not extra:
                self._error("direct-HIP finite one-shot GPU event phase order must match the public API operation order")
        return
    if self._is_direct_hip_bounded_native_input_oneshot_capture() and backend == "hip-direct":
        expected = DIRECT_HIP_ONESHOT_GPU_EVENT_PHASES
        if phases != expected:
            missing = [phase for phase in expected if phase not in phases]
            extra = [phase for phase in phases if phase not in expected]
            if missing:
                self._error(f"direct-HIP one-shot GPU event phase set is incomplete; missing {', '.join(missing)}")
            if extra:
                self._error(f"direct-HIP one-shot GPU event phase set contains undeclared phases: {', '.join(extra)}")
            if not missing and not extra:
                self._error("direct-HIP one-shot GPU event phase order must match the public API operation order")
        return
    if self._is_direct_hip_bounded_resident_fallback_oneshot_capture() and backend == "hip-direct":
        expected = DIRECT_HIP_ONESHOT_RESIDENT_FALLBACK_GPU_EVENT_PHASES
        if phases != expected:
            missing = [phase for phase in expected if phase not in phases]
            extra = [phase for phase in phases if phase not in expected]
            if missing:
                self._error(
                    "direct-HIP one-shot resident fallback GPU event phase set is incomplete; "
                    f"missing {', '.join(missing)}"
                )
            if extra:
                self._error(
                    "direct-HIP one-shot resident fallback GPU event phase set contains undeclared phases: "
                    f"{', '.join(extra)}"
                )
            if not missing and not extra:
                self._error(
                    "direct-HIP one-shot resident fallback GPU event phase order must match the resident pack/GEMM/export order"
                )
        return
    if self._is_direct_hip_bounded_residue_channel_fusion_capture():
        expected = [
            "bounded_uniform_small_i8_a_h2d",
            "bounded_uniform_small_i8_b_h2d",
            "pack",
            "bounded_uniform_small_i8_ab_transient_gemm_kernel_group",
            "rns_gemm",
            "crt_export_status_memset",
            "crt_export_kernel",
            "crt_export_status_d2h",
            "crt_export_d2h",
            "crt_export",
        ]
        if phases != expected:
            missing = [phase for phase in expected if phase not in phases]
            extra = [phase for phase in phases if phase not in expected]
            if missing:
                self._error(
                    "direct-HIP residue-channel fusion GPU event phase set is incomplete; "
                    f"missing {', '.join(missing)}"
                )
            if extra:
                self._error(
                    "direct-HIP residue-channel fusion GPU event phase set contains undeclared phases: "
                    f"{', '.join(extra)}"
                )
            if not missing and not extra:
                self._error("direct-HIP residue-channel fusion GPU event phase order must match the operation order")
        return
    if self._is_direct_hip_bounded_uniform_small_transient_capture():
        expected = [
            "bounded_uniform_small_i8_a_h2d",
            "bounded_uniform_small_i8_b_h2d",
            "pack",
            "bounded_uniform_small_i8_ab_transient_gemm_kernel_group",
            "rns_gemm",
            "crt_export_status_memset",
            "crt_export_kernel",
            "crt_export_status_d2h",
            "crt_export_d2h",
            "crt_export",
        ]
        if phases != expected:
            missing = [phase for phase in expected if phase not in phases]
            extra = [phase for phase in phases if phase not in expected]
            if missing:
                self._error(
                    "direct-HIP bounded uniform-small transient GPU event phase set is incomplete; "
                    f"missing {', '.join(missing)}"
                )
            if extra:
                self._error(
                    "direct-HIP bounded uniform-small transient GPU event phase set contains undeclared phases: "
                    f"{', '.join(extra)}"
                )
            if not missing and not extra:
                self._error(
                    "direct-HIP bounded uniform-small transient GPU event phase order must match the operation order"
                )
        return
    if self._is_streaming_overlap_capture():
        expected = [
            "pack_h2d",
            "pack_kernel",
            "pack",
            "rns_gemm_kernel_group",
            "rns_gemm",
            "crt_export_status_memset",
            "crt_export_kernel",
            "crt_export_status_d2h",
            "crt_export_d2h",
            "crt_export",
        ]
        if phases != expected:
            missing = [phase for phase in expected if phase not in phases]
            extra = [phase for phase in phases if phase not in expected]
            if missing:
                self._error(
                    "direct-HIP streaming-overlap GPU event phase set is incomplete; "
                    f"missing {', '.join(missing)}"
                )
            if extra:
                self._error(
                    "direct-HIP streaming-overlap GPU event phase set contains undeclared phases: "
                    f"{', '.join(extra)}"
                )
            if not missing and not extra:
                self._error("direct-HIP streaming-overlap GPU event phase order must match the pipeline stage order")
        return
    if self._is_direct_hip_bounded_native_a_reuse_b_capture():
        gemm_event = (
            "bounded_uniform_small_i8_ab_colpair_reuse_b_gemm_kernel_group"
            if self._is_direct_hip_bounded_native_a_reuse_b_uniform_small()
            else "bounded_native_a_reuse_b_gemm_kernel_group"
        )
        if self._is_direct_hip_bounded_native_a_reuse_b_u64_large_colpair():
            gemm_event = "bounded_native_a_colpair_reuse_b_gemm_kernel_group"
        expected = [
            "pack_h2d",
            "pack_kernel",
            "pack",
            gemm_event,
            "rns_gemm",
            "crt_export_status_memset",
            "crt_export_kernel",
            "crt_export_status_d2h",
            "crt_export_d2h",
            "crt_export",
        ]
        if phases != expected:
            missing = [phase for phase in expected if phase not in phases]
            extra = [phase for phase in phases if phase not in expected]
            if missing:
                self._error(
                    "direct-HIP bounded native-A reuse-B GPU event phase set is incomplete; "
                    f"missing {', '.join(missing)}"
                )
            if extra:
                self._error(
                    "direct-HIP bounded native-A reuse-B GPU event phase set contains undeclared phases: "
                    f"{', '.join(extra)}"
                )
            if not missing and not extra:
                self._error(
                    "direct-HIP bounded native-A reuse-B GPU event phase order must match the operation order"
                )
        return
    if self._is_direct_hip_bounded_uniform_small_reuse_a_capture():
        expected = [
            "pack_h2d",
            "pack_kernel",
            "pack",
            "bounded_uniform_small_i8_ab_colpair_reuse_a_gemm_kernel_group",
            "rns_gemm",
            "crt_export_status_memset",
            "crt_export_kernel",
            "crt_export_status_d2h",
            "crt_export_d2h",
            "crt_export",
        ]
        if phases != expected:
            missing = [phase for phase in expected if phase not in phases]
            extra = [phase for phase in phases if phase not in expected]
            if missing:
                self._error(
                    "direct-HIP bounded uniform-small reuse-A GPU event phase set is incomplete; "
                    f"missing {', '.join(missing)}"
                )
            if extra:
                self._error(
                    "direct-HIP bounded uniform-small reuse-A GPU event phase set contains undeclared phases: "
                    f"{', '.join(extra)}"
                )
            if not missing and not extra:
                self._error(
                    "direct-HIP bounded uniform-small reuse-A GPU event phase order must match the operation order"
                )
        return
    if self._is_direct_hip_bounded_native_b_reuse_a_u64_large_colpair_capture():
        expected = [
            "pack_h2d",
            "pack_kernel",
            "pack",
            "bounded_native_b_colpair_reuse_a_gemm_kernel_group",
            "rns_gemm",
            "crt_export_status_memset",
            "crt_export_kernel",
            "crt_export_status_d2h",
            "crt_export_d2h",
            "crt_export",
        ]
        if phases != expected:
            missing = [phase for phase in expected if phase not in phases]
            extra = [phase for phase in phases if phase not in expected]
            if missing:
                self._error(
                    "direct-HIP bounded native-B reuse-A GPU event phase set is incomplete; "
                    f"missing {', '.join(missing)}"
                )
            if extra:
                self._error(
                    "direct-HIP bounded native-B reuse-A GPU event phase set contains undeclared phases: "
                    f"{', '.join(extra)}"
                )
            if not missing and not extra:
                self._error(
                    "direct-HIP bounded native-B reuse-A GPU event phase order must match the operation order"
                )
        return
    if self._is_direct_hip_bounded_skinny_gemv_n1_capture():
        expected = [
            "pack_h2d",
            "pack_kernel",
            "pack",
            "rns_gemv_n1_kernel_group",
            "rns_gemm",
            "crt_export_status_memset",
            "crt_export_kernel",
            "crt_export_status_d2h",
            "crt_export_d2h",
            "crt_export",
        ]
        if phases != expected:
            missing = [phase for phase in expected if phase not in phases]
            extra = [phase for phase in phases if phase not in expected]
            if missing:
                self._error(
                    "direct-HIP skinny GEMV N=1 GPU event phase set is incomplete; "
                    f"missing {', '.join(missing)}"
                )
            if extra:
                self._error(
                    "direct-HIP skinny GEMV N=1 GPU event phase set contains undeclared phases: "
                    f"{', '.join(extra)}"
                )
            if not missing and not extra:
                self._error("direct-HIP skinny GEMV N=1 GPU event phase order must match the operation order")
        return
    if self._is_direct_hip_bounded_skinny_gemv_small_n_capture():
        expected = [
            "pack_h2d",
            "pack_kernel",
            "pack",
            "rns_gemv_small_n_kernel_group",
            "rns_gemm",
            "crt_export_status_memset",
            "crt_export_kernel",
            "crt_export_status_d2h",
            "crt_export_d2h",
            "crt_export",
        ]
        if phases != expected:
            missing = [phase for phase in expected if phase not in phases]
            extra = [phase for phase in phases if phase not in expected]
            if missing:
                self._error(
                    "direct-HIP skinny GEMV small-N GPU event phase set is incomplete; "
                    f"missing {', '.join(missing)}"
                )
            if extra:
                self._error(
                    "direct-HIP skinny GEMV small-N GPU event phase set contains undeclared phases: "
                    f"{', '.join(extra)}"
                )
            if not missing and not extra:
                self._error("direct-HIP skinny GEMV small-N GPU event phase order must match the operation order")
        return
    if self._is_direct_hip_native_to_rns_bridge_capture():
        conversion_event = (
            "native_i64_to_rns_kernel"
            if self.data.get("semantics") == "bounded_i64"
            else "native_u64_to_rns_kernel"
        )
        expected = [
            "pack_h2d",
            "pack_kernel",
            "pack",
            conversion_event,
            "rns_gemm_kernel_group",
            "rns_gemm",
            "crt_export_status_memset",
            "crt_export_kernel",
            "crt_export_status_d2h",
            "crt_export_d2h",
            "crt_export",
        ]
        if phases != expected:
            missing = [phase for phase in expected if phase not in phases]
            extra = [phase for phase in phases if phase not in expected]
            if missing:
                self._error(
                    "direct-HIP native-to-RNS bridge GPU event phase set is incomplete; "
                    f"missing {', '.join(missing)}"
                )
            if extra:
                self._error(
                    "direct-HIP native-to-RNS bridge GPU event phase set contains undeclared phases: "
                    f"{', '.join(extra)}"
                )
            if not missing and not extra:
                self._error("direct-HIP native-to-RNS bridge GPU event phase order must match the operation order")
        return
    if self._is_direct_hip_vector_to_rns_chain_capture():
        conversion_event = (
            "native_i64_to_rns_kernel"
            if self.data.get("semantics") == "bounded_i64"
            else "native_u64_to_rns_kernel"
        )
        vector_kernel = (
            "vector_alu_i64_kernel"
            if self.data.get("semantics") == "bounded_i64"
            else "vector_alu_u64_kernel"
        )
        if self._is_direct_hip_vector_to_rns_host_repack_control_capture():
            expected = [
                "vector_alu_pack_a_h2d",
                "vector_alu_pack_b_h2d",
                "pack_h2d",
                "pack_kernel",
                "pack",
                "vector_alu_status_memset",
                vector_kernel,
                "vector_alu_status_d2h",
                "vector_alu_output_d2h",
                "vector_to_rns_host_repack_a",
                "rns_gemm_kernel_group",
                "rns_gemm",
                "crt_export_status_memset",
                "crt_export_kernel",
                "crt_export_status_d2h",
                "crt_export_d2h",
                "crt_export",
            ]
        else:
            expected = [
                "vector_alu_pack_a_h2d",
                "vector_alu_pack_b_h2d",
                "pack_h2d",
                "pack_kernel",
                "pack",
                "vector_alu_status_memset",
                vector_kernel,
                "vector_alu_status_d2h",
                conversion_event,
                "rns_gemm_kernel_group",
                "rns_gemm",
                "crt_export_status_memset",
                "crt_export_kernel",
                "crt_export_status_d2h",
                "crt_export_d2h",
                "crt_export",
            ]
        if phases != expected:
            missing = [phase for phase in expected if phase not in phases]
            extra = [phase for phase in phases if phase not in expected]
            if missing:
                self._error(
                    "direct-HIP vector-to-RNS chain GPU event phase set is incomplete; "
                    f"missing {', '.join(missing)}"
                )
            if extra:
                self._error(
                    "direct-HIP vector-to-RNS chain GPU event phase set contains undeclared phases: "
                    f"{', '.join(extra)}"
                )
            if not missing and not extra:
                self._error("direct-HIP vector-to-RNS chain GPU event phase order must match the operation order")
        return
    if self.data.get("semantics") == "wrap_u64_mod_2_64" and backend == "hip-direct":
        expected = [
            "pack_h2d",
            "pack_kernel",
            "pack",
            wrap64_hip_expected_gemm_event_label(self.data.get("selected_kernel")),
            "rns_gemm",
            "wrap64_export_kernel",
            "wrap64_export_d2h",
            "crt_export",
        ]
        if phases != expected:
            missing = [phase for phase in expected if phase not in phases]
            extra = [phase for phase in phases if phase not in expected]
            if missing:
                self._error(f"direct-HIP wrap64 GPU event phase set is incomplete; missing {', '.join(missing)}")
            if extra:
                self._error(
                    f"direct-HIP wrap64 GPU event phase set contains undeclared phases: {', '.join(extra)}"
                )
            if not missing and not extra:
                self._error("direct-HIP wrap64 GPU event phase order must match the operation order")
        return
    if backend == "hip-vector-alu-int64":
        expected = self._expected_vector_gpu_event_phases()
        if phases != expected:
            missing = [phase for phase in expected if phase not in phases]
            extra = [phase for phase in phases if phase not in expected]
            if missing:
                self._error(f"vector-ALU GPU event phase set is incomplete; missing {', '.join(missing)}")
            if extra:
                self._error(f"vector-ALU GPU event phase set contains undeclared phases: {', '.join(extra)}")
            if not missing and not extra:
                self._error("vector-ALU GPU event phase order must match the native int64 operation order")
        return
    if backend not in {"ck", "rocwmma", "amdgpu-builtins"} or self._is_wrap64_rocwmma_candidate():
        return
    deep_labels = [phase for phase in phases if self._is_deep_accelerator_gpu_event_label(phase)]
    if scope == OLD_ACCELERATOR_GPU_EVENT_SCOPE:
        if deep_labels:
            self._error(
                "deep accelerator GPU event labels require "
                f"timing_metadata.gpu_event_timing_source_scope={DEEP_ACCELERATOR_GPU_EVENT_SCOPE}"
            )
        return
    if scope != DEEP_ACCELERATOR_GPU_EVENT_SCOPE:
        return
    expected = self._expected_accelerator_deep_gpu_event_phases()
    if expected is None or phases == expected:
        return
    missing = [phase for phase in expected if phase not in phases]
    extra = [phase for phase in phases if phase not in expected]
    if missing:
        self._error(f"deep accelerator GPU event phase set is incomplete; missing {', '.join(missing)}")
    if extra:
        self._error(f"deep accelerator GPU event phase set contains undeclared phases: {', '.join(extra)}")
    if not missing and not extra:
        self._error("deep accelerator GPU event phase order must match the selected backend operation order")

def expected_status_event_labels(self) -> list[str]:
    backend = self.data.get("backend_selected")
    semantics = self.data.get("semantics")
    if backend == "hip-vector-alu-int64":
        return ["vector_alu_status_memset", "vector_alu_status_d2h"]
    if semantics in {"exact_wide_signed", "exact_wide_unsigned"}:
        return ["exact_wide_export_status_memset", "exact_wide_export_status_d2h"]
    if semantics in {"bounded_i64", "bounded_u64"}:
        return ["crt_export_status_memset", "crt_export_status_d2h"]
    return []

def known_status_event_labels(self) -> set[str]:
    return {
        "crt_export_status_memset",
        "crt_export_status_d2h",
        "exact_wide_export_status_memset",
        "exact_wide_export_status_d2h",
        "vector_alu_status_memset",
        "vector_alu_status_d2h",
    }

def validate_status_event_consistency(self, phases: list[str], parsed: dict[str, list[float]]) -> None:
    output_policy = self.data.get("output_policy")
    if not isinstance(output_policy, dict):
        return
    handling = output_policy.get("status_handling")
    phase_set = set(phases)
    expected = self._expected_status_event_labels()
    known_present = sorted(self._known_status_event_labels() & phase_set)
    if handling == "required":
        for label in expected:
            if label not in phase_set:
                self._error(f"output_policy.status_handling=required requires GPU event phase {label}")
        return
    if handling == "not_applicable":
        for label in known_present:
            self._error(f"output_policy.status_handling=not_applicable forbids GPU event phase {label}")
        return
    if handling == "structurally_elided":
        for label in known_present:
            values = parsed.get(label, [])
            if any(value != 0.0 for value in values):
                self._error(
                    "output_policy.status_handling=structurally_elided requires "
                    f"gpu_event_timings_us.{label} to be zero-filled when present"
                )

def validate_gpu_events(self) -> None:
    metadata = self.data.get("timing_metadata")
    if not isinstance(metadata, dict):
        return
    enabled = metadata.get("gpu_event_timing")
    repeats = self.data.get("repeats")
    if not isinstance(enabled, bool) or not _is_int(repeats):
        return
    selected_backend = self.data.get("backend_selected")
    residue_current_chain = self._is_residue_current_chain_capture()
    if (
        selected_backend in {"ck", "rocwmma", "amdgpu-builtins", "hip-vector-alu-int64"}
        and enabled is not True
        and not residue_current_chain
    ):
        self._error(f"{selected_backend} captures must include HIP event operation-group timings")
    timings = self.data.get("gpu_event_timings_us")
    summary = self.data.get("gpu_event_timing_summary_us")
    if not enabled:
        if timings is not None:
            self._error("gpu_event_timings_us must be null when gpu_event_timing is false")
        if summary is not None:
            self._error("gpu_event_timing_summary_us must be null when gpu_event_timing is false")
        if metadata.get("gpu_event_phase_order") is not None:
            self._error("timing_metadata.gpu_event_phase_order must be null when events are unavailable")
        if metadata.get("gpu_event_timing_source") is not None:
            self._error("timing_metadata.gpu_event_timing_source must be null when events are unavailable")
        if metadata.get("gpu_event_timing_source_scope") is not None:
            self._error("timing_metadata.gpu_event_timing_source_scope must be null when events are unavailable")
        return
    source = metadata.get("gpu_event_timing_source")
    scope = metadata.get("gpu_event_timing_source_scope")
    if not isinstance(source, str):
        self._error("timing_metadata.gpu_event_timing_source must be a string when events are available")
    elif source != "hipEventElapsedTime":
        self._error("timing_metadata.gpu_event_timing_source must be hipEventElapsedTime")
    if not isinstance(scope, str):
        self._error("timing_metadata.gpu_event_timing_source_scope must be a string when events are available")
    elif selected_backend == "hip-direct" and scope not in DIRECT_HIP_GPU_EVENT_SCOPES:
        expected = ", ".join(sorted(DIRECT_HIP_GPU_EVENT_SCOPES))
        self._error(f"timing_metadata.gpu_event_timing_source_scope must be a known direct-HIP scope: {expected}")
    elif selected_backend == "hipblaslt" and scope not in HIPBLASLT_GPU_EVENT_SCOPES:
        expected = ", ".join(sorted(HIPBLASLT_GPU_EVENT_SCOPES))
        self._error(f"timing_metadata.gpu_event_timing_source_scope must be a known hipBLASLt scope: {expected}")
    elif selected_backend in {"ck", "rocwmma", "amdgpu-builtins"} and scope not in ACCELERATOR_GPU_EVENT_SCOPES:
        expected = ", ".join(sorted(ACCELERATOR_GPU_EVENT_SCOPES))
        self._error(f"timing_metadata.gpu_event_timing_source_scope must be a known accelerator scope: {expected}")
    elif selected_backend == "hip-vector-alu-int64" and scope not in VECTOR_ALU_GPU_EVENT_SCOPES:
        expected = ", ".join(sorted(VECTOR_ALU_GPU_EVENT_SCOPES))
        self._error(f"timing_metadata.gpu_event_timing_source_scope must be a known vector-ALU scope: {expected}")
    if not isinstance(timings, dict):
        self._error("gpu_event_timings_us must be an object when gpu_event_timing is true")
        return
    phases = self._gpu_event_phases(metadata)
    if not phases:
        return
    self._validate_expected_gpu_event_phases(scope, phases)
    phase_set = set(phases)
    timing_keys = set(timings.keys())
    if timing_keys != phase_set:
        for phase in sorted(phase_set - timing_keys):
            self._error(f"gpu_event_timings_us.{phase} must be an array")
        for phase in sorted(timing_keys - phase_set):
            self._error(f"gpu_event_timings_us contains undeclared phase {phase}")
    if isinstance(summary, dict):
        summary_keys = set(summary.keys())
        if summary_keys != phase_set:
            for phase in sorted(phase_set - summary_keys):
                self._error(f"gpu_event_timing_summary_us.{phase} must be an object")
            for phase in sorted(summary_keys - phase_set):
                self._error(f"gpu_event_timing_summary_us contains undeclared phase {phase}")
    parsed: dict[str, list[float]] = {}
    for phase in phases:
        values = timings.get(phase)
        if not isinstance(values, list):
            self._error(f"gpu_event_timings_us.{phase} must be an array")
            continue
        if len(values) != repeats:
            self._error(f"gpu_event_timings_us.{phase} length {len(values)} does not match repeats {repeats}")
        parsed_values: list[float] = []
        for index, value in enumerate(values):
            if not _is_number(value) or float(value) < 0.0:
                self._error(f"gpu_event_timings_us.{phase}[{index}] must be a nonnegative finite number")
            else:
                parsed_values.append(float(value))
        parsed[phase] = parsed_values
    self._validate_status_event_consistency(phases, parsed)
    self._validate_timing_summaries(parsed, "gpu_event_timing_summary_us", phases)
    if self._is_all_zero_direct_hip_adaptive_capture():
        for phase in ["pack_h2d", "pack_kernel", "pack"]:
            values = parsed.get(phase)
            if isinstance(values, list) and any(value != 0.0 for value in values):
                self._error(
                    f"all-zero direct-HIP adaptive captures must report gpu_event_timings_us.{phase} as zero"
                )

def gpu_event_phases(self, metadata: dict[str, Any]) -> list[str]:
    phase_order = metadata.get("gpu_event_phase_order")
    if not isinstance(phase_order, list) or not all(isinstance(item, str) for item in phase_order):
        self._error("timing_metadata.gpu_event_phase_order must be an array of strings when events are available")
        return []
    if len(set(phase_order)) != len(phase_order):
        self._error("timing_metadata.gpu_event_phase_order must not contain duplicates")
        return []
    return list(phase_order)
