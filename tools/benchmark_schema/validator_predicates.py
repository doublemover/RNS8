from __future__ import annotations

from .core_shared import *

class ValidatorPredicatesMixin:
    def _is_wrap64_rocwmma_candidate(self) -> bool:
        return (
            self.data.get("semantics") == "wrap_u64_mod_2_64"
            and self.data.get("backend_selected") == "rocwmma"
            and self.data.get("selected_kernel") == WRAP64_ROCWMMA_CANDIDATE_KERNEL
            and self.data.get("backend_requested") == "rocwmma-wrap64-candidate"
        )

    def _is_vector_alu_runtime_capture(self) -> bool:
        return (
            self.data.get("backend_selected") == "hip-vector-alu-int64"
            and self.data.get("benchmark") == "rns8_bounded_gemm_hip_vector_alu_int64_runtime"
        )

    def _benchmark_execution_mode(self) -> str:
        mode = self.data.get("benchmark_execution_mode")
        if mode is None:
            metadata = self.data.get("timing_metadata")
            if isinstance(metadata, dict):
                mode = metadata.get("benchmark_execution_mode")
        if isinstance(mode, str):
            return mode
        if self._is_wrap64_rocwmma_candidate():
            return "internal_wrap64_rocwmma_candidate"
        if self._is_vector_alu_runtime_capture():
            return "public_runtime_vector_alu_native_buffers"
        if self.data.get("benchmark") == "rns8_hip_graph_replay_resident_rns_chain":
            return "hip_graph_replay_resident_rns_chain"
        if self.data.get("benchmark") == "rns8_hip_graph_replay_bounded_pack_gemm_export":
            return "hip_graph_replay_bounded_pack_gemm_export"
        if self.data.get("benchmark") == "rns8_hip_graph_replay_finite_u8_pack_gemm_export":
            return "hip_graph_replay_finite_u8_pack_gemm_export"
        if self.data.get("benchmark") == "rns8_hip_graph_replay_wrap64_pack_gemm_export":
            return "hip_graph_replay_wrap64_pack_gemm_export"
        if self.data.get("benchmark") in {"rns8_bounded_gemm_public_oneshot", "rns8_finite_u8_public_oneshot"}:
            return "public_oneshot_transient_native_inputs"
        if self.data.get("backend_selected") == "hip-vector-alu-int64":
            return "benchmark_owned_vector_alu_native_buffers"
        return "persistent_resident_matrices"

    def _is_bounded_oneshot_capture(self) -> bool:
        return (
            self._benchmark_execution_mode() == "public_oneshot_transient_native_inputs"
            or self.data.get("benchmark") == "rns8_bounded_gemm_public_oneshot"
        ) and self.data.get("semantics") in {"bounded_i64", "bounded_u64"}

    def _is_direct_hip_bounded_native_input_oneshot_capture(self) -> bool:
        return (
            self._is_bounded_oneshot_capture()
            and self.data.get("backend_selected") == "hip-direct"
            and self.data.get("selected_prefix") == self.data.get("prefix") == 9
        )

    def _is_direct_hip_bounded_resident_fallback_oneshot_capture(self) -> bool:
        selected = self.data.get("selected_prefix")
        requested = self.data.get("prefix")
        schedule = self.data.get("schedule_metadata")
        schedule_selected = schedule.get("max_selected_prefix") if isinstance(schedule, dict) else None
        return (
            self._is_bounded_oneshot_capture()
            and self.data.get("backend_selected") == "hip-direct"
            and _is_int(selected)
            and _is_int(requested)
            and selected > 0
            and selected < requested
            and selected == schedule_selected
        )

    def _is_finite_oneshot_capture(self) -> bool:
        return (
            self._benchmark_execution_mode() == "public_oneshot_transient_native_inputs"
            or self.data.get("benchmark") == "rns8_finite_u8_public_oneshot"
        ) and self.data.get("semantics") in {"finite_ring_u8", "finite_field_u8"}

    def _is_public_oneshot_capture(self) -> bool:
        return self._is_bounded_oneshot_capture() or self._is_finite_oneshot_capture()

    def _is_direct_hip_bounded_native_a_reuse_b_capture(self) -> bool:
        return (
            self.data.get("backend_selected") == "hip-direct"
            and self.data.get("semantics") in {"bounded_i64", "bounded_u64"}
            and self.data.get("pack_mode") == "prepacked_reuse_b"
            and self.data.get("prepack_reuse_strategy") == "persistent_matrix_residency"
            and self._benchmark_execution_mode()
            in {"transient_native_a_resident_b_reuse", "transient_uniform_small_i8_a_resident_i8_b_reuse"}
        )

    def _is_direct_hip_bounded_native_a_reuse_b_uniform_small(self) -> bool:
        semantics = self.data.get("semantics")
        distribution = self.data.get("input_distribution")
        return (
            (semantics == "bounded_i64" and distribution == "signed_uniform_-16_16")
            or (semantics == "bounded_u64" and distribution == "unsigned_uniform_0_16")
        )

    def _is_direct_hip_bounded_native_a_reuse_b_u64_large_colpair(self) -> bool:
        return (
            self.data.get("semantics") == "bounded_u64"
            and not self._is_direct_hip_bounded_native_a_reuse_b_uniform_small()
            and _is_int(self.data.get("m"))
            and _is_int(self.data.get("n"))
            and _is_int(self.data.get("k"))
            and self.data.get("m") >= 512
            and self.data.get("n") >= 512
            and self.data.get("k") >= 512
        )

    def _is_direct_hip_bounded_uniform_small_reuse_a_capture(self) -> bool:
        return (
            self.data.get("backend_selected") == "hip-direct"
            and self.data.get("semantics") in {"bounded_i64", "bounded_u64"}
            and self.data.get("pack_mode") == "prepacked_reuse_a"
            and self.data.get("prepack_reuse_strategy") == "persistent_matrix_residency"
            and self._benchmark_execution_mode() == "transient_uniform_small_i8_b_resident_i8_a_reuse"
            and self._is_direct_hip_bounded_native_a_reuse_b_uniform_small()
        )

    def _is_direct_hip_bounded_uniform_small_transient_capture(self) -> bool:
        return (
            self.data.get("backend_selected") == "hip-direct"
            and self.data.get("semantics") in {"bounded_i64", "bounded_u64"}
            and self.data.get("pack_mode") == "per_repeat_repack"
            and self.data.get("prepack_reuse_strategy") == "none"
            and self._benchmark_execution_mode() == "transient_uniform_small_i8_ab_inputs"
            and self._is_direct_hip_bounded_native_a_reuse_b_uniform_small()
        )

    def _is_direct_hip_native_to_rns_bridge_capture(self) -> bool:
        return (
            self.data.get("backend_selected") == "hip-direct"
            and self.data.get("semantics") in {"bounded_i64", "bounded_u64"}
            and self._benchmark_execution_mode() == "auto_native_to_rns_bridge"
        )

    def _is_direct_hip_vector_to_rns_chain_capture(self) -> bool:
        return (
            self.data.get("backend_selected") == "hip-direct"
            and self.data.get("semantics") in {"bounded_i64", "bounded_u64"}
            and self._benchmark_execution_mode()
            in {
                "vector_native_to_direct_rns_chain",
                "vector_native_host_export_repack_direct_rns_chain",
            }
        )

    def _is_direct_hip_vector_to_rns_host_repack_control_capture(self) -> bool:
        return (
            self.data.get("backend_selected") == "hip-direct"
            and self.data.get("semantics") in {"bounded_i64", "bounded_u64"}
            and self._benchmark_execution_mode() == "vector_native_host_export_repack_direct_rns_chain"
        )

    def _is_direct_hip_bounded_residue_channel_fusion_capture(self) -> bool:
        return (
            self.data.get("backend_selected") == "hip-direct"
            and self.data.get("semantics") in {"bounded_i64", "bounded_u64"}
            and self.data.get("benchmark") == "rns8_bounded_gemm_residue_channel_fusion_experiment"
            and self._benchmark_execution_mode() == "residue_channel_fusion_native_inputs"
            and self.data.get("pack_mode") == "per_repeat_repack"
            and self.data.get("prepack_reuse_strategy") == "none"
        )

    def _is_host_api_batch_capture(self) -> bool:
        return self._benchmark_execution_mode() == "benchmark_host_api_batch"

    def _is_grouped_dispatch_capture(self) -> bool:
        return self._benchmark_execution_mode() == "benchmark_grouped_dispatch_evidence"

    def _is_hip_graph_replay_capture(self) -> bool:
        return self._benchmark_execution_mode() in {
            "hip_graph_replay_resident_rns_chain",
            "hip_graph_replay_bounded_pack_gemm_export",
            "hip_graph_replay_finite_u8_pack_gemm_export",
            "hip_graph_replay_wrap64_pack_gemm_export",
        }

    def _is_streaming_overlap_capture(self) -> bool:
        overlap = self.data.get("streaming_overlap")
        return (
            isinstance(overlap, dict)
            and overlap.get("requested") is True
            and overlap.get("capture_status") == "executed"
        )

    def _is_residue_chain_final_export_capture(self) -> bool:
        return (
            self._residue_chain_length() > 1
            and self._residue_output_mode() == "host_export"
            and self.data.get("residue_chain_final_export") is True
            and self._benchmark_execution_mode()
            in {"residue_chain_final_host_export", "residue_chain_independent_final_host_export"}
        )

    def _is_residue_chain_independent_final_export_capture(self) -> bool:
        return (
            self._residue_chain_length() > 1
            and self._residue_output_mode() == "host_export"
            and self.data.get("residue_chain_final_export") is True
            and self.data.get("residue_chain_independent_final_export") is True
            and self._benchmark_execution_mode() == "residue_chain_independent_final_host_export"
        )

    def _is_direct_hip_bounded_native_b_reuse_a_u64_large_colpair_capture(self) -> bool:
        return (
            self.data.get("backend_selected") == "hip-direct"
            and self.data.get("semantics") == "bounded_u64"
            and self.data.get("pack_mode") == "prepacked_reuse_a"
            and self.data.get("prepack_reuse_strategy") == "persistent_matrix_residency"
            and self._benchmark_execution_mode() == "transient_native_b_resident_a_reuse"
            and not self._is_direct_hip_bounded_native_a_reuse_b_uniform_small()
            and _is_int(self.data.get("m"))
            and _is_int(self.data.get("n"))
            and _is_int(self.data.get("k"))
            and self.data.get("m") >= 512
            and self.data.get("n") >= 512
            and self.data.get("k") >= 512
        )

    def _is_direct_hip_finite_native_a_reuse_b_capture(self) -> bool:
        return (
            self.data.get("backend_selected") == "hip-direct"
            and self.data.get("semantics") in {"finite_ring_u8", "finite_field_u8"}
            and self.data.get("pack_mode") == "prepacked_reuse_b"
            and self.data.get("prepack_reuse_strategy") == "persistent_matrix_residency"
        )

