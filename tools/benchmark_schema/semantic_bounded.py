from __future__ import annotations

from typing import Any

from .core_shared import *

def validate_bounded_contract(self, ctx: dict[str, Any]) -> None:
    semantics = ctx['semantics']
    prefix = ctx['prefix']
    packed_layout = ctx['packed_layout']
    schedule = ctx['schedule']
    backend_metadata = ctx['backend_metadata']
    k_value = ctx['k_value']
    bound_mode = ctx['bound_mode']
    residue_chain_length = ctx['residue_chain_length']
    residue_output_mode = ctx['residue_output_mode']
    status_check = ctx['status_check']
    prefix_policy = ctx['prefix_policy']
    metadata = ctx['metadata']
    oneshot_capture = self._is_bounded_oneshot_capture()
    native_a_reuse_b_capture = self._is_direct_hip_bounded_native_a_reuse_b_capture()
    streaming_overlap_capture = self._is_streaming_overlap_capture()
    if oneshot_capture:
        if self.data.get("benchmark") != "rns8_bounded_gemm_public_oneshot":
            self._error("one-shot captures must use benchmark=rns8_bounded_gemm_public_oneshot")
        if self._benchmark_execution_mode() != "public_oneshot_transient_native_inputs":
            self._error("one-shot captures must use benchmark_execution_mode=public_oneshot_transient_native_inputs")
        if self.data.get("backend_selected") not in {"cpu-reference", "hip-direct"}:
            self._error("one-shot bounded captures must select cpu-reference or hip-direct")
        if self.data.get("backend_requested") not in {"cpu-reference", "cpu", "hip-direct"}:
            self._error("one-shot bounded captures must request cpu or hip-direct")
        if bound_mode != "global":
            self._error("one-shot bounded captures must use bound_mode=global")
        if residue_chain_length != 1 or residue_output_mode != "host_export":
            self._error("one-shot bounded captures must use host_export residue_chain_length=1")
        if self.data.get("reuse_packed_inputs") is True:
            self._error("one-shot bounded captures must not use packed-input reuse")
        if self.data.get("pack_mode") not in {None, "per_repeat_repack"}:
            self._error("one-shot bounded captures must use pack_mode=per_repeat_repack")
        if self.data.get("prepack_reuse_strategy") not in {None, "none"}:
            self._error("one-shot bounded captures must use prepack_reuse_strategy=none")
        if self.data.get("backend_selected") == "hip-direct":
            native_input_oneshot = self._is_direct_hip_bounded_native_input_oneshot_capture()
            resident_fallback_oneshot = self._is_direct_hip_bounded_resident_fallback_oneshot_capture()
            if not native_input_oneshot and not resident_fallback_oneshot:
                self._error(
                    "direct-HIP one-shot bounded captures must use either prefix-9 native-input metadata "
                    "or selected-prefix resident fallback metadata"
                )
            expected_kernel = DIRECT_HIP_BOUNDED_ONESHOT_RESIDENT_FALLBACK_KERNEL
            expected_epilogue = DIRECT_HIP_BOUNDED_ONESHOT_RESIDENT_FALLBACK_EPILOGUE
            expected_workspace = DIRECT_HIP_BOUNDED_ONESHOT_RESIDENT_FALLBACK_WORKSPACE
            if native_input_oneshot:
                expected_kernel = DIRECT_HIP_BOUNDED_ONESHOT_KERNEL_V1
                large_oneshot_shape = all(int(self.data.get(dim, 0)) >= 512 for dim in ("m", "n", "k"))
                if large_oneshot_shape:
                    expected_kernel = DIRECT_HIP_BOUNDED_ONESHOT_KERNEL_LARGE_COLPAIR_V2
                expected_epilogue = DIRECT_HIP_BOUNDED_ONESHOT_EPILOGUE
                expected_workspace = DIRECT_HIP_BOUNDED_ONESHOT_WORKSPACE
            if self.data.get("selected_kernel") != expected_kernel:
                self._error(
                    "direct-HIP one-shot bounded captures must use "
                    f"selected_kernel={expected_kernel}"
                )
            if isinstance(backend_metadata, dict):
                if backend_metadata.get("epilogue_mode") != expected_epilogue:
                    self._error(
                        "direct-HIP one-shot bounded captures must use "
                        f"backend_metadata.epilogue_mode={expected_epilogue}"
                    )
                if backend_metadata.get("workspace_mode") != expected_workspace:
                    self._error(
                        "direct-HIP one-shot bounded captures must use "
                        f"backend_metadata.workspace_mode={expected_workspace}"
                    )
                if backend_metadata.get("isa_evidence") != DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE:
                    self._error(
                        "direct-HIP one-shot bounded captures must use "
                        f"backend_metadata.isa_evidence={DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE}"
                    )
            metadata = self.data.get("timing_metadata")
            if native_input_oneshot and isinstance(metadata, dict) and metadata.get("gpu_event_timing") is True:
                expected_scope = "direct_hip_oneshot_default_stream_operation_groups"
                if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                    self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
            if resident_fallback_oneshot and isinstance(metadata, dict) and metadata.get("gpu_event_timing") is True:
                expected_scope = "direct_hip_oneshot_resident_fallback_default_stream_operation_groups"
                if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                    self._error(
                        "direct-HIP one-shot captures must use "
                        f"timing_metadata.gpu_event_timing_source_scope={expected_scope} "
                        "for resident fallback metadata"
                    )
    if native_a_reuse_b_capture:
        if bound_mode != "global":
            self._error("direct-HIP bounded native-A reuse-B captures must use bound_mode=global")
        if residue_chain_length != 1 or residue_output_mode != "host_export":
            self._error("direct-HIP bounded native-A reuse-B captures must use host_export residue_chain_length=1")
        uniform_small = self._is_direct_hip_bounded_native_a_reuse_b_uniform_small()
        expected_kernel = (
            DIRECT_HIP_BOUNDED_UNIFORM_SMALL_NATIVE_A_REUSE_B_KERNELS[semantics]
            if uniform_small
            else DIRECT_HIP_BOUNDED_NATIVE_A_REUSE_B_KERNELS[semantics]
        )
        if self._is_direct_hip_bounded_native_a_reuse_b_u64_large_colpair():
            expected_kernel = DIRECT_HIP_BOUNDED_NATIVE_A_REUSE_B_U64_LARGE_COLPAIR_KERNEL
        expected_epilogue = (
            DIRECT_HIP_BOUNDED_UNIFORM_SMALL_NATIVE_A_REUSE_B_EPILOGUE
            if uniform_small
            else DIRECT_HIP_BOUNDED_NATIVE_A_REUSE_B_EPILOGUE
        )
        expected_input_profile = "uniform-small" if uniform_small else "adaptive-bands"
        if self.data.get("selected_kernel") != expected_kernel:
            self._error(
                "direct-HIP bounded native-A reuse-B captures must use "
                f"selected_kernel={expected_kernel}"
            )
        if isinstance(backend_metadata, dict):
            if backend_metadata.get("epilogue_mode") != expected_epilogue:
                self._error(
                    "direct-HIP bounded native-A reuse-B captures must use "
                    f"backend_metadata.epilogue_mode={expected_epilogue}"
                )
            if backend_metadata.get("workspace_mode") != DIRECT_HIP_BOUNDED_NATIVE_A_REUSE_B_WORKSPACE:
                self._error(
                    "direct-HIP bounded native-A reuse-B captures must use "
                    f"backend_metadata.workspace_mode={DIRECT_HIP_BOUNDED_NATIVE_A_REUSE_B_WORKSPACE}"
                )
            if backend_metadata.get("isa_evidence") != DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE:
                self._error(
                    "direct-HIP bounded native-A reuse-B captures must use "
                    f"backend_metadata.isa_evidence={DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE}"
                )
            autotune_key = backend_metadata.get("autotune_key")
            if isinstance(autotune_key, str):
                normalized_key = f";{autotune_key};"
                required_parts = {
                    "kernel": expected_kernel,
                    "epilogue": expected_epilogue,
                    "input_profile": expected_input_profile,
                }
                for key, value in required_parts.items():
                    if f";{key}={value};" not in normalized_key:
                        self._error(
                            "direct-HIP bounded native-A reuse-B backend_metadata.autotune_key "
                            f"must include {key}={value}"
                        )
    if self._is_direct_hip_bounded_uniform_small_reuse_a_capture():
        if bound_mode != "global":
            self._error("direct-HIP bounded uniform-small reuse-A captures must use bound_mode=global")
        if residue_chain_length != 1 or residue_output_mode != "host_export":
            self._error(
                "direct-HIP bounded uniform-small reuse-A captures must use host_export residue_chain_length=1"
            )
        expected_kernel = DIRECT_HIP_BOUNDED_UNIFORM_SMALL_REUSE_A_KERNELS[semantics]
        expected_epilogue = DIRECT_HIP_BOUNDED_UNIFORM_SMALL_REUSE_A_EPILOGUE
        if self.data.get("selected_kernel") != expected_kernel:
            self._error(
                "direct-HIP bounded uniform-small reuse-A captures must use "
                f"selected_kernel={expected_kernel}"
            )
        if isinstance(backend_metadata, dict):
            if backend_metadata.get("epilogue_mode") != expected_epilogue:
                self._error(
                    "direct-HIP bounded uniform-small reuse-A captures must use "
                    f"backend_metadata.epilogue_mode={expected_epilogue}"
                )
            if backend_metadata.get("workspace_mode") != DIRECT_HIP_BOUNDED_UNIFORM_SMALL_REUSE_A_WORKSPACE:
                self._error(
                    "direct-HIP bounded uniform-small reuse-A captures must use "
                    f"backend_metadata.workspace_mode={DIRECT_HIP_BOUNDED_UNIFORM_SMALL_REUSE_A_WORKSPACE}"
                )
            if backend_metadata.get("isa_evidence") != DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE:
                self._error(
                    "direct-HIP bounded uniform-small reuse-A captures must use "
                    f"backend_metadata.isa_evidence={DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE}"
                )
            autotune_key = backend_metadata.get("autotune_key")
            if isinstance(autotune_key, str):
                normalized_key = f";{autotune_key};"
                required_parts = {
                    "kernel": expected_kernel,
                    "epilogue": expected_epilogue,
                    "input_profile": "uniform-small",
                }
                for key, value in required_parts.items():
                    if f";{key}={value};" not in normalized_key:
                        self._error(
                            "direct-HIP bounded uniform-small reuse-A backend_metadata.autotune_key "
                            f"must include {key}={value}"
                        )
    if self._is_direct_hip_bounded_residue_channel_fusion_capture():
        if bound_mode != "global":
            self._error("direct-HIP residue-channel fusion captures must use bound_mode=global")
        if residue_chain_length != 1 or residue_output_mode != "host_export":
            self._error(
                "direct-HIP residue-channel fusion captures must use host_export residue_chain_length=1"
            )
        if self.data.get("reuse_packed_inputs") is True:
            self._error("direct-HIP residue-channel fusion captures must not use packed-input reuse")
        expected_kernel = DIRECT_HIP_BOUNDED_RESIDUE_CHANNEL_FUSION_KERNELS[semantics]
        expected_epilogue = DIRECT_HIP_BOUNDED_RESIDUE_CHANNEL_FUSION_EPILOGUE
        if self.data.get("selected_kernel") != expected_kernel:
            self._error(
                "direct-HIP residue-channel fusion captures must use "
                f"selected_kernel={expected_kernel}"
            )
        if isinstance(backend_metadata, dict):
            if backend_metadata.get("epilogue_mode") != expected_epilogue:
                self._error(
                    "direct-HIP residue-channel fusion captures must use "
                    f"backend_metadata.epilogue_mode={expected_epilogue}"
                )
            if backend_metadata.get("workspace_mode") != DIRECT_HIP_BOUNDED_RESIDUE_CHANNEL_FUSION_WORKSPACE:
                self._error(
                    "direct-HIP residue-channel fusion captures must use "
                    f"backend_metadata.workspace_mode={DIRECT_HIP_BOUNDED_RESIDUE_CHANNEL_FUSION_WORKSPACE}"
                )
            if backend_metadata.get("isa_evidence") != DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE:
                self._error(
                    "direct-HIP residue-channel fusion captures must use "
                    f"backend_metadata.isa_evidence={DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE}"
                )
            autotune_key = backend_metadata.get("autotune_key")
            if isinstance(autotune_key, str):
                normalized_key = f";{autotune_key};"
                required_parts = {
                    "kernel": expected_kernel,
                    "epilogue": expected_epilogue,
                    "input_profile": "uniform-small",
                    "execution": "residue_channel_fusion_native_inputs",
                }
                for key, value in required_parts.items():
                    if f";{key}={value};" not in normalized_key:
                        self._error(
                            "direct-HIP residue-channel fusion backend_metadata.autotune_key "
                            f"must include {key}={value}"
                        )
    if self._is_direct_hip_bounded_uniform_small_transient_capture():
        if bound_mode != "global":
            self._error("direct-HIP bounded uniform-small transient captures must use bound_mode=global")
        if residue_chain_length != 1 or residue_output_mode != "host_export":
            self._error(
                "direct-HIP bounded uniform-small transient captures must use host_export residue_chain_length=1"
            )
        if self.data.get("reuse_packed_inputs") is True:
            self._error("direct-HIP bounded uniform-small transient captures must not use packed-input reuse")
        expected_kernel = DIRECT_HIP_BOUNDED_UNIFORM_SMALL_TRANSIENT_KERNELS[semantics]
        expected_epilogue = DIRECT_HIP_BOUNDED_UNIFORM_SMALL_TRANSIENT_EPILOGUE
        if self.data.get("selected_kernel") != expected_kernel:
            self._error(
                "direct-HIP bounded uniform-small transient captures must use "
                f"selected_kernel={expected_kernel}"
            )
        if isinstance(backend_metadata, dict):
            if backend_metadata.get("epilogue_mode") != expected_epilogue:
                self._error(
                    "direct-HIP bounded uniform-small transient captures must use "
                    f"backend_metadata.epilogue_mode={expected_epilogue}"
                )
            if backend_metadata.get("workspace_mode") != DIRECT_HIP_BOUNDED_UNIFORM_SMALL_TRANSIENT_WORKSPACE:
                self._error(
                    "direct-HIP bounded uniform-small transient captures must use "
                    f"backend_metadata.workspace_mode={DIRECT_HIP_BOUNDED_UNIFORM_SMALL_TRANSIENT_WORKSPACE}"
                )
            if backend_metadata.get("isa_evidence") != DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE:
                self._error(
                    "direct-HIP bounded uniform-small transient captures must use "
                    f"backend_metadata.isa_evidence={DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE}"
                )
            autotune_key = backend_metadata.get("autotune_key")
            if isinstance(autotune_key, str):
                normalized_key = f";{autotune_key};"
                required_parts = {
                    "kernel": expected_kernel,
                    "epilogue": expected_epilogue,
                    "input_profile": "uniform-small",
                    "execution": "transient_uniform_small_i8_ab_inputs",
                }
                for key, value in required_parts.items():
                    if f";{key}={value};" not in normalized_key:
                        self._error(
                            "direct-HIP bounded uniform-small transient backend_metadata.autotune_key "
                            f"must include {key}={value}"
                        )
    if self._is_direct_hip_bounded_native_b_reuse_a_u64_large_colpair_capture():
        if bound_mode != "global":
            self._error("direct-HIP bounded native-B reuse-A captures must use bound_mode=global")
        if residue_chain_length != 1 or residue_output_mode != "host_export":
            self._error("direct-HIP bounded native-B reuse-A captures must use host_export residue_chain_length=1")
        expected_kernel = DIRECT_HIP_BOUNDED_NATIVE_B_REUSE_A_U64_LARGE_COLPAIR_KERNEL
        expected_epilogue = DIRECT_HIP_BOUNDED_NATIVE_B_REUSE_A_EPILOGUE
        if self.data.get("selected_kernel") != expected_kernel:
            self._error(
                "direct-HIP bounded native-B reuse-A captures must use "
                f"selected_kernel={expected_kernel}"
            )
        if isinstance(backend_metadata, dict):
            if backend_metadata.get("epilogue_mode") != expected_epilogue:
                self._error(
                    "direct-HIP bounded native-B reuse-A captures must use "
                    f"backend_metadata.epilogue_mode={expected_epilogue}"
                )
            if backend_metadata.get("workspace_mode") != DIRECT_HIP_BOUNDED_NATIVE_B_REUSE_A_WORKSPACE:
                self._error(
                    "direct-HIP bounded native-B reuse-A captures must use "
                    f"backend_metadata.workspace_mode={DIRECT_HIP_BOUNDED_NATIVE_B_REUSE_A_WORKSPACE}"
                )
            if backend_metadata.get("isa_evidence") != DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE:
                self._error(
                    "direct-HIP bounded native-B reuse-A captures must use "
                    f"backend_metadata.isa_evidence={DIRECT_HIP_RECIPROCAL_ISA_EVIDENCE}"
                )
            autotune_key = backend_metadata.get("autotune_key")
            if isinstance(autotune_key, str):
                normalized_key = f";{autotune_key};"
                required_parts = {
                    "kernel": expected_kernel,
                    "epilogue": expected_epilogue,
                    "input_profile": "adaptive-bands",
                }
                for key, value in required_parts.items():
                    if f";{key}={value};" not in normalized_key:
                        self._error(
                            "direct-HIP bounded native-B reuse-A backend_metadata.autotune_key "
                            f"must include {key}={value}"
                        )
    if _is_int(prefix) and prefix <= 0:
        self._error(f"{semantics} captures must use a positive prefix")
    expected_native_layout = (
        "native_i64_rowmajor_v1" if semantics == "bounded_i64" else "native_u64_rowmajor_v1"
    )
    if self._is_vector_alu_runtime_capture():
        if packed_layout != expected_native_layout:
            self._error(f"{semantics} runtime vector captures must use packed_layout_version={expected_native_layout}")
    elif packed_layout is not None:
        self._error(f"{semantics} captures must use packed_layout_version=null")
    if self._is_direct_hip_native_to_rns_bridge_capture():
        metadata = self.data.get("timing_metadata")
        if self.data.get("benchmark") != "rns8_bounded_gemm_native_to_rns_bridge":
            self._error("native-to-RNS bridge captures must use benchmark=rns8_bounded_gemm_native_to_rns_bridge")
        if self.data.get("backend_requested") != "auto":
            self._error("native-to-RNS bridge captures must use backend_requested=auto")
        if self.data.get("pack_mode") != "per_repeat_repack":
            self._error("native-to-RNS bridge captures must use pack_mode=per_repeat_repack")
        if self.data.get("reuse_packed_inputs") is not False:
            self._error("native-to-RNS bridge captures must not use packed-input reuse")
        if isinstance(metadata, dict) and metadata.get("native_to_rns_bridge_forced") is not True:
            self._error("native-to-RNS bridge captures must set timing_metadata.native_to_rns_bridge_forced=true")
    if self._is_direct_hip_vector_to_rns_chain_capture():
        metadata = self.data.get("timing_metadata")
        host_repack_control = self._is_direct_hip_vector_to_rns_host_repack_control_capture()
        expected_benchmark = (
            "rns8_bounded_gemm_vector_to_rns_chain_host_repack_control"
            if host_repack_control
            else "rns8_bounded_gemm_vector_to_rns_chain"
        )
        if self.data.get("benchmark") != expected_benchmark:
            self._error(
                f"vector-to-RNS chain captures must use benchmark={expected_benchmark}"
            )
        if self.data.get("backend_requested") != "auto":
            self._error("vector-to-RNS chain captures must use backend_requested=auto")
        pack_mode = self.data.get("pack_mode")
        if pack_mode not in {"per_repeat_repack", "prepacked_reuse_b"}:
            self._error("vector-to-RNS chain captures must use pack_mode=per_repeat_repack or prepacked_reuse_b")
        expected_reuse = pack_mode == "prepacked_reuse_b"
        if self.data.get("reuse_packed_inputs") is not expected_reuse:
            self._error("vector-to-RNS chain captures must set reuse_packed_inputs to match consumer-B reuse")
        expected_strategy = "persistent_matrix_residency" if expected_reuse else "none"
        if self.data.get("prepack_reuse_strategy") != expected_strategy:
            self._error(
                f"vector-to-RNS chain captures must use prepack_reuse_strategy={expected_strategy}"
            )
        if isinstance(metadata, dict):
            if metadata.get("vector_to_rns_chain") is not True:
                self._error("vector-to-RNS chain captures must set timing_metadata.vector_to_rns_chain=true")
            expected_control_mode = (
                "host_export_repack_control" if host_repack_control else "fused_device_native_to_rns"
            )
            if metadata.get("vector_to_rns_chain_control_mode") != expected_control_mode:
                self._error(
                    "vector-to-RNS chain captures must set "
                    f"timing_metadata.vector_to_rns_chain_control_mode={expected_control_mode}"
                )
            if metadata.get("native_to_rns_bridge_forced") is not False:
                self._error(
                    "vector-to-RNS chain captures must set timing_metadata.native_to_rns_bridge_forced=false"
                )
            if metadata.get("pack_mode") != pack_mode:
                self._error("vector-to-RNS chain captures must keep timing_metadata.pack_mode in sync")
            if metadata.get("prepack_reuse_strategy") != expected_strategy:
                self._error(
                    "vector-to-RNS chain captures must keep timing_metadata.prepack_reuse_strategy in sync"
                )
    if streaming_overlap_capture:
        metadata = self.data.get("timing_metadata")
        if self.data.get("benchmark") != "rns8_streaming_overlap_resident_b_pipeline":
            self._error(
                "streaming-overlap captures must use benchmark=rns8_streaming_overlap_resident_b_pipeline"
            )
        if self.data.get("benchmark_execution_mode") != "benchmark_streaming_overlap_resident_b_pipeline":
            self._error(
                "streaming-overlap captures must use "
                "benchmark_execution_mode=benchmark_streaming_overlap_resident_b_pipeline"
            )
        if self.data.get("backend_selected") != "hip-direct" or self.data.get("backend_requested") != "hip-direct":
            self._error("streaming-overlap captures must request and select hip-direct")
        if bound_mode != "global":
            self._error("streaming-overlap captures must use bound_mode=global")
        if residue_chain_length != 1 or residue_output_mode != "host_export":
            self._error("streaming-overlap captures must use host_export residue_chain_length=1")
        if self.data.get("pack_mode") != "prepacked_reuse_b":
            self._error("streaming-overlap captures must use pack_mode=prepacked_reuse_b")
        operands = self.data.get("prepack_reuse_operands")
        if self.data.get("reuse_packed_inputs") is not True or operands != ["B"]:
            self._error("streaming-overlap captures must reuse packed B")
        if isinstance(operands, list) and "A" in operands:
            self._error("streaming-overlap captures must not reuse packed A")
        if self.data.get("prepack_reuse_strategy") != "persistent_matrix_residency":
            self._error("streaming-overlap captures must keep B resident through persistent_matrix_residency")
        if isinstance(metadata, dict) and metadata.get("gpu_event_timing") is True:
            expected_scope = "direct_hip_streaming_overlap_multistream_operation_groups"
            if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
    if residue_chain_length > 1:
        if residue_output_mode == "residue_current_rns":
            expected_epilogue_type = "residue_current_rns_output"
        else:
            expected_epilogue_type = "crt_export"
            if self._benchmark_execution_mode() not in {
                "residue_chain_final_host_export",
                "residue_chain_independent_final_host_export",
            }:
                self._error(
                    "bounded residue-chain final-export captures must use "
                    "benchmark_execution_mode=residue_chain_final_host_export or "
                    "residue_chain_independent_final_host_export"
                )
        if bound_mode != "global":
            self._error("bounded residue chains must use bound_mode=global")
        if self.data.get("backend_selected") in {"hip-vector-alu-int64"}:
            self._error("bounded residue chains must not select hip-vector-alu-int64")
        if self.data.get("m") != self.data.get("n") or self.data.get("n") != self.data.get("k"):
            self._error("bounded residue chains must use square m=n=k shapes")
    else:
        expected_epilogue_type = (
            "direct_int64_export" if self.data.get("backend_selected") == "hip-vector-alu-int64" else "crt_export"
        )
        if residue_output_mode != "host_export":
            self._error("bounded host-export captures must use residue_output_mode=host_export")
    if self.data.get("epilogue_type") != expected_epilogue_type:
        self._error(f"{semantics} captures must use {expected_epilogue_type} epilogue")
    if bound_mode == "global":
        expected_bound_kind = "global_max_abs" if semantics == "bounded_i64" else "global_max_unsigned"
        if self.data.get("bound_kind") != expected_bound_kind:
            self._error(f"{semantics} captures must use bound_kind={expected_bound_kind}")
        if self.data.get("tile_bounds_u64") is not None:
            self._error(f"{semantics} global captures must use tile_bounds_u64=null")
        if self.data.get("backend_selected") == "hip-direct" and not oneshot_capture:
            metadata = self.data.get("timing_metadata")
            if isinstance(metadata, dict) and metadata.get("gpu_event_timing") is True:
                if streaming_overlap_capture:
                    expected_scope = "direct_hip_streaming_overlap_multistream_operation_groups"
                elif self._is_direct_hip_native_to_rns_bridge_capture():
                    expected_scope = "direct_hip_native_to_rns_bridge_default_stream_operation_groups"
                elif self._is_direct_hip_vector_to_rns_chain_capture():
                    expected_scope = (
                        "direct_hip_vector_native_host_repack_chain_default_stream_operation_groups"
                        if self._is_direct_hip_vector_to_rns_host_repack_control_capture()
                        else "direct_hip_vector_native_to_rns_chain_default_stream_operation_groups"
                    )
                else:
                    expected_scope = "direct_hip_default_stream_backend_operation_groups"
                if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                    self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
        if self.data.get("backend_selected") == "hipblaslt":
            metadata = self.data.get("timing_metadata")
            if isinstance(metadata, dict) and metadata.get("gpu_event_timing") is True:
                expected_scope = "hipblaslt_baseline_default_stream_backend_operation_groups"
                if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                    self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
            if isinstance(backend_metadata, dict):
                if backend_metadata.get("epilogue_mode") != "separate_i32_scratch_reduce_then_crt_export":
                    self._error(
                        "hipBLASLt bounded captures must use "
                        "backend_metadata.epilogue_mode=separate_i32_scratch_reduce_then_crt_export"
                    )
        if isinstance(schedule, dict) and _is_int(prefix):
            if prefix_policy == "minimum_proven":
                if schedule.get("min_selected_prefix") != schedule.get("max_selected_prefix"):
                    self._error(f"{semantics} minimum_proven global captures must use one selected prefix")
                if schedule.get("max_selected_prefix") > prefix:
                    self._error(f"{semantics} selected schedule prefix must be <= prefix")
            elif schedule.get("min_selected_prefix") != prefix or schedule.get("max_selected_prefix") != prefix:
                self._error(f"{semantics} captures must use fixed selected schedule prefix equal to prefix")
            if schedule.get("prefix_group_count") != 1:
                self._error(f"{semantics} captures must use one fixed prefix group")
            if schedule.get("adaptive_execution_applied") is True:
                self._error(f"{semantics} global captures must not apply adaptive execution")
    elif bound_mode == "per_tile":
        expected_bound_kind = "per_tile_max_abs" if semantics == "bounded_i64" else "per_tile_max_unsigned"
        if self.data.get("bound_kind") != expected_bound_kind:
            self._error(f"{semantics} per-tile captures must use bound_kind={expected_bound_kind}")
        if self.data.get("backend_selected") not in {
            "cpu-reference",
            "hip-direct",
            "ck",
            "rocwmma",
            "amdgpu-builtins",
            "hip-vector-alu-int64",
        }:
            self._error(
                "per-tile adaptive captures must select cpu-reference, hip-direct, ck, rocwmma, amdgpu-builtins, "
                "or hip-vector-alu-int64 backend"
            )
        if self.data.get("bound") != 0:
            self._error("per-tile adaptive captures must use bound=0")
        self._validate_v4_tile_bounds(semantics, schedule)
        if prefix_policy == "fixed_requested":
            if isinstance(schedule, dict) and _is_int(prefix):
                if schedule.get("min_selected_prefix") != prefix or schedule.get("max_selected_prefix") != prefix:
                    self._error("fixed-requested per-tile captures must use selected schedule prefix equal to prefix")
                if schedule.get("prefix_group_count") != 1:
                    self._error("fixed-requested per-tile captures must use one selected prefix group")
                if schedule.get("adaptive_prefix_active") is not False or schedule.get("adaptive_skip_active") is not False:
                    self._error("fixed-requested per-tile captures must not set adaptive prefix flags")
                if schedule.get("adaptive_execution_applied") is not False:
                    self._error("fixed-requested per-tile captures must not apply adaptive execution")
        else:
            self._validate_v4_adaptive_schedule(prefix, schedule)
