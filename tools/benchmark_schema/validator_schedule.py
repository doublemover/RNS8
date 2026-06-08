from __future__ import annotations

from typing import Any

from .core_shared import *

class ValidatorScheduleMixin:
    def _validate_phase_availability(self, metadata: dict[str, Any]) -> None:
        availability = metadata.get("phase_availability")
        if not isinstance(availability, dict):
            self._error("timing_metadata.phase_availability must be an object")
            return
        scheduling = availability.get("scheduling")
        if not isinstance(scheduling, dict):
            self._error("timing_metadata.phase_availability.scheduling must be an object")
        else:
            if scheduling.get("timed") is not True:
                self._error("timing_metadata.phase_availability.scheduling.timed must be true")
            if scheduling.get("timing_key") != "scheduling":
                self._error("timing_metadata.phase_availability.scheduling.timing_key must be scheduling")
            expected_scope = (
                "benchmark_static_wrap64_rocwmma_candidate_schedule"
                if self._is_wrap64_rocwmma_candidate()
                else "one_time_schedule_info_query"
            )
            if scheduling.get("scope") != expected_scope:
                self._error(f"timing_metadata.phase_availability.scheduling.scope must be {expected_scope}")
            if not isinstance(scheduling.get("reason"), str) or not scheduling.get("reason"):
                self._error("timing_metadata.phase_availability.scheduling.reason must be a nonempty string")

        global_bound_scan = availability.get(GLOBAL_BOUND_TIMING_PHASE)
        input_scan = self.data.get("bound_source") == "input_scan"
        global_input_scan = input_scan and self.data.get("bound_mode", "global") == "global"
        if global_input_scan and not isinstance(global_bound_scan, dict):
            self._error("timing_metadata.phase_availability.global_bound_scan must be an object for input_scan captures")
        elif global_bound_scan is not None:
            if not isinstance(global_bound_scan, dict):
                self._error("timing_metadata.phase_availability.global_bound_scan must be an object")
            else:
                expected_timed = global_input_scan
                expected_key = GLOBAL_BOUND_TIMING_PHASE if global_input_scan else None
                expected_scope = (
                    "input_row_column_abs_summary" if global_input_scan else "not_applicable_static_profile"
                )
                if global_bound_scan.get("timed") is not expected_timed:
                    self._error(
                        "timing_metadata.phase_availability.global_bound_scan.timed must be "
                        f"{str(expected_timed).lower()}"
                    )
                if global_bound_scan.get("timing_key") != expected_key:
                    self._error(
                        "timing_metadata.phase_availability.global_bound_scan.timing_key must be "
                        f"{expected_key}"
                    )
                if global_bound_scan.get("scope") != expected_scope:
                    self._error(
                        "timing_metadata.phase_availability.global_bound_scan.scope must be "
                        f"{expected_scope}"
                    )
                if not isinstance(global_bound_scan.get("reason"), str) or not global_bound_scan.get("reason"):
                    self._error("timing_metadata.phase_availability.global_bound_scan.reason must be a nonempty string")

        per_tile = self.data.get("bound_mode") == "per_tile"
        tile_bound_scan = availability.get(PER_TILE_TIMING_PHASE)
        if per_tile and not isinstance(tile_bound_scan, dict):
            self._error("timing_metadata.phase_availability.tile_bound_scan must be an object for per-tile captures")
        elif tile_bound_scan is not None:
            if not isinstance(tile_bound_scan, dict):
                self._error("timing_metadata.phase_availability.tile_bound_scan must be an object")
            else:
                expected_timed = per_tile
                expected_key = PER_TILE_TIMING_PHASE if per_tile else None
                expected_scope = "exact_seeded_input_prepass" if per_tile else "not_applicable_global_bound"
                if tile_bound_scan.get("timed") is not expected_timed:
                    self._error(
                        "timing_metadata.phase_availability.tile_bound_scan.timed must be "
                        f"{str(expected_timed).lower()}"
                    )
                if tile_bound_scan.get("timing_key") != expected_key:
                    self._error(
                        "timing_metadata.phase_availability.tile_bound_scan.timing_key must be "
                        f"{expected_key}"
                    )
                if tile_bound_scan.get("scope") != expected_scope:
                    self._error(
                        "timing_metadata.phase_availability.tile_bound_scan.scope must be "
                        f"{expected_scope}"
                    )
                if not isinstance(tile_bound_scan.get("reason"), str) or not tile_bound_scan.get("reason"):
                    self._error("timing_metadata.phase_availability.tile_bound_scan.reason must be a nonempty string")

        reuse_packed = self.data.get("reuse_packed_inputs") is True
        prepack = availability.get("prepack_setup")
        if reuse_packed and not isinstance(prepack, dict):
            self._error("timing_metadata.phase_availability.prepack_setup must be an object for prepacked reuse")
        elif prepack is not None:
            if not isinstance(prepack, dict):
                self._error("timing_metadata.phase_availability.prepack_setup must be an object")
            else:
                expected_timed = reuse_packed
                expected_key = "prepack_setup_us" if reuse_packed else None
                expected_scope = "one_time_before_warmups" if reuse_packed else "not_requested_per_repeat_repack"
                if prepack.get("timed") is not expected_timed:
                    self._error(
                        f"timing_metadata.phase_availability.prepack_setup.timed must be {str(expected_timed).lower()}"
                    )
                if prepack.get("timing_key") != expected_key:
                    self._error(
                        f"timing_metadata.phase_availability.prepack_setup.timing_key must be {expected_key}"
                    )
                if prepack.get("scope") != expected_scope:
                    self._error(
                        f"timing_metadata.phase_availability.prepack_setup.scope must be {expected_scope}"
                    )
                if not isinstance(prepack.get("reason"), str) or not prepack.get("reason"):
                    self._error("timing_metadata.phase_availability.prepack_setup.reason must be a nonempty string")

        reduction = availability.get("reduction")
        if not isinstance(reduction, dict):
            self._error("timing_metadata.phase_availability.reduction must be an object")
            return
        if reduction.get("timed") is not False:
            self._error("timing_metadata.phase_availability.reduction.timed must be false")
        if reduction.get("timing_key") is not None:
            self._error("timing_metadata.phase_availability.reduction.timing_key must be null")
        if self.data.get("semantics") == "wrap_u64_mod_2_64":
            expected_scope = "not_applicable_wrap64_byte_limb"
        elif self.data.get("backend_selected") == "hipblaslt":
            expected_scope = "separate_hipblaslt_i32_scratch_residue_reduce"
        elif self.data.get("backend_selected") == "hip-vector-alu-int64":
            expected_scope = (
                "not_applicable_native_vector_output"
                if self._is_vector_alu_runtime_capture()
                else "not_applicable_direct_int64_export"
            )
        else:
            expected_scope = "fused_into_rns_gemm"
        if reduction.get("scope") != expected_scope:
            self._error(f"timing_metadata.phase_availability.reduction.scope must be {expected_scope}")
        if not isinstance(reduction.get("reason"), str) or not reduction.get("reason"):
            self._error("timing_metadata.phase_availability.reduction.reason must be a nonempty string")

    def _timing_phases(self) -> list[str]:
        phases = list(TIMING_PHASES)
        if self.data.get("bound_source") == "input_scan" and self.data.get("bound_mode", "global") == "global":
            phases.insert(0, GLOBAL_BOUND_TIMING_PHASE)
        if self.data.get("bound_mode") == "per_tile":
            phases.insert(phases.index("matrix_alloc"), PER_TILE_TIMING_PHASE)
        return phases

    def _residue_chain_length(self) -> int:
        value = self.data.get("residue_chain_length", 1)
        if not _is_int(value) or value < 1:
            self._error("residue_chain_length must be a positive integer")
            return 1
        return int(value)

    def _residue_output_mode(self) -> str:
        value = self.data.get("residue_output_mode", "host_export")
        if not isinstance(value, str):
            self._error("residue_output_mode must be a string")
            return "host_export"
        if value not in {"host_export", "residue_current_rns"}:
            self._error("residue_output_mode must be host_export or residue_current_rns")
            return "host_export"
        return value

    def _is_residue_current_chain_capture(self) -> bool:
        return (
            self._residue_output_mode() == "residue_current_rns"
            or self.data.get("epilogue_type") == "residue_current_rns_output"
        )

    def _validate_tile_value(self, key: str, value: Any) -> None:
        if not _is_int(value):
            self._error(f"{key} must be an integer")
            return
        if self._is_wrap64_rocwmma_candidate():
            if value != 16:
                self._error(f"{key} must be 16 for rocWMMA wrap64 candidate captures")
            return
        if value < 64 or value > 512 or (value & (value - 1)) != 0:
            self._error(f"{key} must be a power of two from 64 through 512")

    def _validate_schedule_metadata(self) -> None:
        self._validate_tile_value("tile_m", self.data.get("tile_m"))
        self._validate_tile_value("tile_n", self.data.get("tile_n"))
        schedule = self._require("schedule_metadata", "dict")
        if not isinstance(schedule, dict):
            return
        expected_source = (
            "rns8_bench_wrap64_rocwmma_candidate_static_schedule"
            if self._is_wrap64_rocwmma_candidate()
            else "rns8_get_plan_schedule_info"
        )
        if schedule.get("source") != expected_source:
            self._error(f"schedule_metadata.source must be {expected_source}")
        schedule_bound_kind = schedule.get("bound_kind")
        if schedule_bound_kind is not None:
            if not isinstance(schedule_bound_kind, str) or schedule_bound_kind not in BOUND_KINDS:
                self._error(f"schedule_metadata.bound_kind must be one of {sorted(BOUND_KINDS)}")
            elif schedule_bound_kind != self.data.get("bound_kind"):
                self._error("schedule_metadata.bound_kind must match bound_kind")
        for key in ["effective_bound", "lhs_bound", "rhs_bound"]:
            value = schedule.get(key)
            if value is not None and (not _is_int(value) or value < 0):
                self._error(f"schedule_metadata.{key} must be a nonnegative integer")
        bound_contract = schedule.get("bound_contract")
        if bound_contract is not None and not isinstance(bound_contract, str):
            self._error("schedule_metadata.bound_contract must be a string")
        if schedule_bound_kind == "input_range_and_k":
            if schedule.get("bound_contract") != "input_range_and_k_derived_output_bound":
                self._error("input-range schedules must declare the derived output bound contract")
            for key in ["effective_bound", "lhs_bound", "rhs_bound"]:
                if not _is_int(schedule.get(key)):
                    self._error(f"input-range schedules require schedule_metadata.{key}")
        elif schedule_bound_kind is not None:
            for key in ["lhs_bound", "rhs_bound"]:
                if _is_int(schedule.get(key)) and schedule.get(key) != 0:
                    self._error(f"non-input-range schedules must use schedule_metadata.{key}=0")
        for key in [
            "tile_m",
            "tile_n",
            "tile_rows",
            "tile_cols",
            "tile_count",
            "min_required_prefix",
            "max_required_prefix",
            "min_selected_prefix",
            "max_selected_prefix",
            "prefix_group_count",
            "range_bit_length",
        ]:
            if not _is_int(schedule.get(key)):
                self._error(f"schedule_metadata.{key} must be an integer")
        for key in ["adaptive_prefix_active", "adaptive_skip_active", "adaptive_execution_applied"]:
            if not isinstance(schedule.get(key), bool):
                self._error(f"schedule_metadata.{key} must be a boolean")
        if schedule.get("tile_m") != self.data.get("tile_m"):
            self._error("schedule_metadata.tile_m must match tile_m")
        if schedule.get("tile_n") != self.data.get("tile_n"):
            self._error("schedule_metadata.tile_n must match tile_n")
        tile_rows = schedule.get("tile_rows")
        tile_cols = schedule.get("tile_cols")
        tile_count = schedule.get("tile_count")
        if _is_int(tile_rows) and _is_int(tile_cols) and _is_int(tile_count):
            if tile_rows <= 0 or tile_cols <= 0 or tile_count != tile_rows * tile_cols:
                self._error("schedule_metadata tile grid must have positive rows/cols and matching tile_count")
        min_required = schedule.get("min_required_prefix")
        max_required = schedule.get("max_required_prefix")
        min_selected = schedule.get("min_selected_prefix")
        max_selected = schedule.get("max_selected_prefix")
        if _is_int(min_required) and _is_int(max_required) and min_required > max_required:
            self._error("schedule_metadata min_required_prefix must be <= max_required_prefix")
        if _is_int(min_selected) and _is_int(max_selected) and min_selected > max_selected:
            self._error("schedule_metadata min_selected_prefix must be <= max_selected_prefix")
        flags = schedule.get("flags")
        zero_count = schedule.get("zero_output_tile_count")
        zero_fraction = schedule.get("zero_output_tile_fraction")
        zero_planes = schedule.get("zero_output_selected_residue_planes")
        zero_active = schedule.get("zero_output_skip_active")
        zero_a_rows = schedule.get("zero_a_row_proof_count")
        zero_b_cols = schedule.get("zero_b_col_proof_count")
        zero_row_col_products = schedule.get("zero_row_col_product_count")
        planner_zero_a_rows = schedule.get("planner_zero_a_row_count")
        planner_zero_b_cols = schedule.get("planner_zero_b_col_count")
        planner_zero_row_col_products = schedule.get("planner_zero_row_col_product_count")
        if flags is not None:
            if not _is_int(flags) or flags < 0:
                self._error("schedule_metadata.flags must be a nonnegative integer")
            elif flags & ~TILE_SCHEDULE_KNOWN_FLAGS:
                self._error("schedule_metadata.flags contains unknown tile schedule flags")
        for key, value in [
            ("zero_a_row_proof_count", zero_a_rows),
            ("zero_b_col_proof_count", zero_b_cols),
            ("zero_row_col_product_count", zero_row_col_products),
            ("planner_zero_a_row_count", planner_zero_a_rows),
            ("planner_zero_b_col_count", planner_zero_b_cols),
            ("planner_zero_row_col_product_count", planner_zero_row_col_products),
        ]:
            if not _is_int(value) or value < 0:
                self._error(f"schedule_metadata.{key} must be a nonnegative integer")
        if zero_count is not None:
            if not _is_int(zero_count) or zero_count < 0:
                self._error("schedule_metadata.zero_output_tile_count must be a nonnegative integer")
            elif _is_int(tile_count) and zero_count > tile_count:
                self._error("schedule_metadata.zero_output_tile_count must be <= tile_count")
        if zero_fraction is not None:
            if not _is_number(zero_fraction) or zero_fraction < 0.0 or zero_fraction > 1.0:
                self._error("schedule_metadata.zero_output_tile_fraction must be between 0 and 1")
            elif _is_int(zero_count) and _is_int(tile_count) and tile_count > 0:
                expected = zero_count / tile_count
                if abs(float(zero_fraction) - expected) > 0.000001:
                    self._error("schedule_metadata.zero_output_tile_fraction must match zero_output_tile_count/tile_count")
        if zero_planes is not None:
            if not _is_int(zero_planes) or zero_planes < 0:
                self._error("schedule_metadata.zero_output_selected_residue_planes must be a nonnegative integer")
        if zero_active is not None:
            if not isinstance(zero_active, bool):
                self._error("schedule_metadata.zero_output_skip_active must be a boolean")
            elif _is_int(zero_count) and zero_active != (zero_count > 0):
                self._error("schedule_metadata.zero_output_skip_active must match zero_output_tile_count > 0")
        if _is_int(zero_count):
            if zero_count > 0 and (not _is_int(flags) or (flags & TILE_SCHEDULE_ZERO_OUTPUT) == 0):
                self._error("schedule_metadata zero_output_tile_count requires ZERO_OUTPUT schedule flag")
            if zero_count == 0 and _is_int(flags) and (flags & TILE_SCHEDULE_ZERO_OUTPUT) != 0:
                self._error("schedule_metadata ZERO_OUTPUT flag requires zero_output_tile_count > 0")
        elif _is_int(flags) and (flags & TILE_SCHEDULE_ZERO_OUTPUT) != 0:
            self._error("schedule_metadata ZERO_OUTPUT flag requires zero_output_tile_count")
        if _is_int(zero_planes) and _is_int(zero_count) and zero_count == 0 and zero_planes != 0:
            self._error("schedule_metadata.zero_output_selected_residue_planes must be zero when no zero tiles are skipped")
        if _is_int(zero_a_rows) and _is_int(planner_zero_a_rows) and zero_a_rows != planner_zero_a_rows:
            self._error("schedule_metadata planner_zero_a_row_count must match zero_a_row_proof_count")
        if _is_int(zero_b_cols) and _is_int(planner_zero_b_cols) and zero_b_cols != planner_zero_b_cols:
            self._error("schedule_metadata planner_zero_b_col_count must match zero_b_col_proof_count")
        if (
            _is_int(zero_row_col_products)
            and _is_int(planner_zero_row_col_products)
            and zero_row_col_products != planner_zero_row_col_products
        ):
            self._error("schedule_metadata planner_zero_row_col_product_count must match zero_row_col_product_count")
        row_col_counts_valid = (
            _is_int(zero_a_rows)
            and _is_int(zero_b_cols)
            and _is_int(zero_row_col_products)
            and _is_int(planner_zero_a_rows)
            and _is_int(planner_zero_b_cols)
            and _is_int(planner_zero_row_col_products)
        )
        if row_col_counts_valid:
            m_value = self.data.get("m")
            n_value = self.data.get("n")
            if _is_int(m_value) and zero_a_rows > m_value:
                self._error("schedule_metadata.zero_a_row_proof_count must be <= m")
            if _is_int(n_value) and zero_b_cols > n_value:
                self._error("schedule_metadata.zero_b_col_proof_count must be <= n")
            if _is_int(m_value) and _is_int(n_value):
                expected_products = zero_a_rows * n_value + (m_value - zero_a_rows) * zero_b_cols
                if zero_row_col_products != expected_products:
                    self._error("schedule_metadata.zero_row_col_product_count must match zero row/column product coverage")
                if zero_row_col_products > m_value * n_value:
                    self._error("schedule_metadata.zero_row_col_product_count must be <= m*n")
            row_col_flag = _is_int(flags) and (flags & TILE_SCHEDULE_ZERO_ROW_COL_PRODUCT) != 0
            if zero_row_col_products > 0 and not row_col_flag:
                self._error("schedule_metadata zero_row_col_product_count requires ZERO_ROW_COL_PRODUCT schedule flag")
            if zero_row_col_products == 0 and row_col_flag:
                self._error("schedule_metadata ZERO_ROW_COL_PRODUCT flag requires zero_row_col_product_count > 0")
            per_tile_bounded = (
                self.data.get("semantics") in {"bounded_i64", "bounded_u64"}
                and self.data.get("bound_mode") == "per_tile"
            )
            if not per_tile_bounded and (
                zero_a_rows != 0
                or zero_b_cols != 0
                or zero_row_col_products != 0
                or planner_zero_a_rows != 0
                or planner_zero_b_cols != 0
                or planner_zero_row_col_products != 0
                or row_col_flag
            ):
                self._error("non-per-tile bounded captures must use zero row/column proof counts of 0")

    def _validate_bound_discovery_metadata(self) -> None:
        bound_source = self.data.get("bound_source")
        discovery = self.data.get("bound_discovery")
        semantics = self.data.get("semantics")
        bound_mode = self.data.get("bound_mode", "global")
        if bound_source is not None:
            if not isinstance(bound_source, str) or bound_source not in BOUND_SOURCES:
                self._error(f"bound_source must be one of {sorted(BOUND_SOURCES)}")
        if discovery is None:
            if bound_source == "input_scan":
                self._error("input_scan captures must include bound_discovery")
            return
        if semantics not in {"bounded_i64", "bounded_u64"}:
            self._error("bound_discovery is only valid for bounded captures")
            return
        if not isinstance(discovery, dict):
            self._error("bound_discovery must be an object or null")
            return

        source = discovery.get("source")
        if not isinstance(source, str) or source not in BOUND_DISCOVERY_SOURCES:
            self._error(f"bound_discovery.source must be one of {sorted(BOUND_DISCOVERY_SOURCES)}")
        static_bound = discovery.get("static_bound")
        selected_bound = discovery.get("selected_bound")
        top_bound = self.data.get("bound")
        for key in ["static_bound", "selected_bound"]:
            value = discovery.get(key)
            if not _is_int(value) or value < 0:
                self._error(f"bound_discovery.{key} must be a nonnegative integer")
        if _is_int(selected_bound) and _is_int(top_bound) and selected_bound != top_bound:
            self._error("bound_discovery.selected_bound must match bound")
        if _is_int(static_bound) and _is_int(selected_bound) and selected_bound > static_bound and static_bound != 0:
            self._error("bound_discovery.selected_bound must not exceed static_bound")

        if source == "static_profile_contract":
            if bound_source not in {None, "static_profile"}:
                self._error("static_profile_contract captures must use bound_source=static_profile")
            for key in [
                "discovered_global_bound",
                "candidate_row_sum_col_max",
                "candidate_row_max_col_sum",
                "row_abs_sum_max",
                "row_abs_max",
                "col_abs_sum_max",
                "col_abs_max",
                "zero_row_count",
                "zero_col_count",
            ]:
                if discovery.get(key) is not None:
                    self._error(f"static_profile_contract captures must use bound_discovery.{key}=null")
            return

        if source == "input_exact_tile_bounds":
            if bound_source != "input_scan":
                self._error("input_exact_tile_bounds captures must use bound_source=input_scan")
            if bound_mode != "per_tile":
                self._error("input_exact_tile_bounds captures must use bound_mode=per_tile")
            for key in [
                "discovered_global_bound",
                "candidate_row_sum_col_max",
                "candidate_row_max_col_sum",
                "row_abs_sum_max",
                "row_abs_max",
                "col_abs_sum_max",
                "col_abs_max",
                "zero_row_count",
                "zero_col_count",
            ]:
                if discovery.get(key) is not None:
                    self._error(f"input_exact_tile_bounds captures must use bound_discovery.{key}=null")
            tile_bounds = self.data.get("tile_bounds_u64")
            if not isinstance(tile_bounds, dict):
                self._error("input_exact_tile_bounds captures must include tile_bounds_u64")
            elif tile_bounds.get("source") != "exact_seeded_input_prepass":
                self._error("input_exact_tile_bounds captures must use tile_bounds_u64.source=exact_seeded_input_prepass")
            if _is_int(selected_bound) and selected_bound != 0:
                self._error("input_exact_tile_bounds captures must use selected_bound=0")
            return

        if bound_source != "input_scan":
            self._error("input_row_column_abs_summary captures must use bound_source=input_scan")
        if bound_mode != "global":
            self._error("input_row_column_abs_summary captures must use bound_mode=global")
        for key in [
            "discovered_global_bound",
            "candidate_row_sum_col_max",
            "candidate_row_max_col_sum",
            "row_abs_sum_max",
            "row_abs_max",
            "col_abs_sum_max",
            "col_abs_max",
            "zero_row_count",
            "zero_col_count",
        ]:
            value = discovery.get(key)
            if not _is_int(value) or value < 0:
                self._error(f"bound_discovery.{key} must be a nonnegative integer")
        discovered = discovery.get("discovered_global_bound")
        candidate_a = discovery.get("candidate_row_sum_col_max")
        candidate_b = discovery.get("candidate_row_max_col_sum")
        if _is_int(discovered) and _is_int(candidate_a) and _is_int(candidate_b):
            expected = min(candidate_a, candidate_b)
            if _is_int(static_bound) and static_bound != 0:
                expected = min(expected, static_bound)
            if discovered != expected:
                self._error("bound_discovery.discovered_global_bound must equal the minimum safe candidate bound")
        if _is_int(discovered) and _is_int(selected_bound) and discovered != selected_bound:
            self._error("input_scan bound_discovery.discovered_global_bound must match selected_bound")
        if _is_int(discovery.get("zero_row_count")) and _is_int(self.data.get("m")):
            if discovery.get("zero_row_count") > self.data.get("m"):
                self._error("bound_discovery.zero_row_count must be <= m")
        if _is_int(discovery.get("zero_col_count")) and _is_int(self.data.get("n")):
            if discovery.get("zero_col_count") > self.data.get("n"):
                self._error("bound_discovery.zero_col_count must be <= n")

    def _validate_prefix_policy_metadata(self) -> None:
        present = [field for field in PREFIX_POLICY_FIELDS if field in self.data]
        if present and len(present) != len(PREFIX_POLICY_FIELDS):
            missing = sorted(PREFIX_POLICY_FIELDS - set(present))
            self._error(f"prefix policy metadata fields must be complete; missing {missing}")
            return
        if not present:
            return

        semantics = self.data.get("semantics")
        bound_mode = self.data.get("bound_mode", "global")
        schedule = self.data.get("schedule_metadata")
        prefix = self.data.get("prefix")
        selected = self.data.get("selected_prefix")
        requested = self.data.get("requested_max_prefix")
        policy = self.data.get("contract_prefix_policy")
        planes_requested = self.data.get("residue_planes_requested")
        planes_selected = self.data.get("residue_planes_selected")
        planes_skipped = self.data.get("residue_planes_skipped")
        skip_fraction = self.data.get("residue_plane_skip_fraction")

        for key in [
            "selected_prefix",
            "requested_max_prefix",
            "residue_planes_requested",
            "residue_planes_selected",
            "residue_planes_skipped",
        ]:
            if not _is_int(self.data.get(key)) or self.data.get(key) < 0:
                self._error(f"{key} must be a nonnegative integer")
        if not _is_number(skip_fraction) or float(skip_fraction) < 0.0 or float(skip_fraction) > 1.0:
            self._error("residue_plane_skip_fraction must be a number in [0, 1]")
        if not isinstance(policy, str) or policy not in CONTRACT_PREFIX_POLICIES:
            self._error(f"contract_prefix_policy must be one of {sorted(CONTRACT_PREFIX_POLICIES)}")
        if _is_int(prefix) and _is_int(requested) and requested != prefix:
            self._error("requested_max_prefix must match prefix")

        if semantics in NON_RNS_PREFIX_SEMANTICS:
            if selected != 0 or requested != 0 or planes_requested != 0 or planes_selected != 0 or planes_skipped != 0:
                self._error("non-RNS captures must report zero prefix policy plane counts")
            if policy != "semantic_specific_no_rns_prefix":
                self._error("non-RNS captures must use contract_prefix_policy=semantic_specific_no_rns_prefix")
            if _is_number(skip_fraction) and not _close(float(skip_fraction), 0.0):
                self._error("non-RNS captures must use residue_plane_skip_fraction=0")
            return

        if semantics in RNS_PREFIX_SEMANTICS:
            if _is_int(prefix) and prefix <= 0:
                self._error("RNS captures with prefix policy metadata must use prefix>0")
            if _is_int(selected) and _is_int(prefix) and selected > prefix:
                self._error("selected_prefix must be <= prefix")
            if isinstance(schedule, dict):
                if schedule.get("max_selected_prefix") != selected:
                    self._error("selected_prefix must match schedule_metadata.max_selected_prefix")
                if (
                    _is_int(selected)
                    and selected > 0
                    and _is_int(schedule.get("min_selected_prefix"))
                    and schedule.get("min_selected_prefix") > selected
                ):
                    self._error("schedule_metadata.min_selected_prefix must be <= selected_prefix")
            expected_skipped = max(int(requested) - int(selected), 0) if _is_int(requested) and _is_int(selected) else None
            if expected_skipped is not None and planes_skipped != expected_skipped:
                self._error("residue_planes_skipped must equal requested_max_prefix - selected_prefix")
            if _is_int(requested) and planes_requested != requested:
                self._error("residue_planes_requested must match requested_max_prefix")
            if _is_int(selected) and planes_selected != selected:
                self._error("residue_planes_selected must match selected_prefix")
            if _is_int(requested) and requested > 0 and expected_skipped is not None:
                expected_fraction = float(expected_skipped) / float(requested)
                if _is_number(skip_fraction) and not _close(float(skip_fraction), expected_fraction):
                    self._error("residue_plane_skip_fraction must match skipped/requested")
            if policy == "semantic_specific_no_rns_prefix":
                self._error("RNS captures must not use semantic_specific_no_rns_prefix")
            if policy == "per_tile_minimum" and bound_mode != "per_tile":
                self._error("contract_prefix_policy=per_tile_minimum requires bound_mode=per_tile")
            if policy in {"minimum_proven", "fixed_requested_residue_chain"} and bound_mode != "global":
                self._error(f"contract_prefix_policy={policy} requires bound_mode=global")
            if policy in {"fixed_requested", "fixed_requested_residue_chain"} and _is_int(selected) and _is_int(prefix):
                if selected != prefix or planes_skipped != 0:
                    self._error(f"contract_prefix_policy={policy} requires selected_prefix=prefix")
            if policy == "minimum_proven" and isinstance(schedule, dict):
                if schedule.get("adaptive_execution_applied") is True:
                    self._error("global minimum_proven captures must not apply adaptive execution")
                if schedule.get("prefix_group_count") != 1:
                    self._error("global minimum_proven captures must use one uniform selected prefix group")
            return

        self._error(f"prefix policy metadata is not supported for semantics {semantics}")

    def _validate_semantic_contract(self) -> None:
        from .semantic_contracts import validate_semantic_contract

        validate_semantic_contract(self)

    def _validate_v4_tile_bounds(self, semantics: Any, schedule: Any) -> None:
        tile_bounds = self.data.get("tile_bounds_u64")
        if not isinstance(tile_bounds, dict):
            self._error("tile_bounds_u64 must be an object for per-tile adaptive captures")
            return
        for key in ["source", "pattern", "order"]:
            if not isinstance(tile_bounds.get(key), str) or not tile_bounds.get(key):
                self._error(f"tile_bounds_u64.{key} must be a nonempty string")
        expected_pattern = "exact_output_tile_max_abs_v1" if semantics == "bounded_i64" else "exact_output_tile_max_unsigned_v1"
        if tile_bounds.get("source") != "exact_seeded_input_prepass":
            self._error("tile_bounds_u64.source must be exact_seeded_input_prepass")
        if tile_bounds.get("pattern") != expected_pattern:
            self._error(f"tile_bounds_u64.pattern must be {expected_pattern}")
        if tile_bounds.get("order") != "row_major_output_tiles":
            self._error("tile_bounds_u64.order must be row_major_output_tiles")
        for key in ["count", "min", "max", "hash_u64"]:
            if not _is_int(tile_bounds.get(key)):
                self._error(f"tile_bounds_u64.{key} must be an integer")
        count = tile_bounds.get("count")
        minimum = tile_bounds.get("min")
        maximum = tile_bounds.get("max")
        if _is_int(count) and count <= 0:
            self._error("tile_bounds_u64.count must be positive")
        if _is_int(minimum) and minimum < 0:
            self._error("tile_bounds_u64.min must be nonnegative")
        if _is_int(maximum) and maximum < 0:
            self._error("tile_bounds_u64.max must be nonnegative")
        if _is_int(minimum) and _is_int(maximum) and minimum > maximum:
            self._error("tile_bounds_u64.min must be <= max")
        if semantics == "bounded_i64" and _is_int(maximum) and maximum > 2**63:
            self._error("bounded_i64 tile_bounds_u64.max must be <= 2^63")
        if _is_int(tile_bounds.get("hash_u64")) and tile_bounds.get("hash_u64") < 0:
            self._error("tile_bounds_u64.hash_u64 must be nonnegative")
        if isinstance(schedule, dict) and _is_int(count) and _is_int(schedule.get("tile_count")):
            if count != schedule.get("tile_count"):
                self._error("tile_bounds_u64.count must match schedule_metadata.tile_count")

    def _validate_v4_adaptive_schedule(self, prefix: Any, schedule: Any) -> None:
        if not isinstance(schedule, dict):
            return
        selected_backend_for_schedule = self.data.get("backend_selected")
        vector_runtime_comparator = (
            selected_backend_for_schedule == "hip-vector-alu-int64"
            and self._is_vector_alu_runtime_capture()
        )
        if vector_runtime_comparator:
            if schedule.get("adaptive_execution_applied") is not False:
                self._error(
                    "per-tile adaptive vector runtime captures must set "
                    "schedule_metadata.adaptive_execution_applied=false"
                )
        elif schedule.get("adaptive_execution_applied") is not True:
            self._error("per-tile adaptive captures must set schedule_metadata.adaptive_execution_applied=true")
        selected_kernel = self.data.get("selected_kernel")
        if not isinstance(selected_kernel, str) or not selected_kernel:
            self._error("per-tile adaptive captures must report selected_kernel")
        else:
            selected_backend = self.data.get("backend_selected")
            zero_output_tiles = (
                _is_int(schedule.get("zero_output_tile_count")) and schedule.get("zero_output_tile_count") > 0
            )
            zero_row_col_products = (
                _is_int(schedule.get("zero_row_col_product_count"))
                and schedule.get("zero_row_col_product_count") > 0
            )
            if zero_output_tiles and zero_row_col_products:
                direct_hip_expected_kernel = DIRECT_HIP_ADAPTIVE_ZERO_TILE_ROW_COL_SKIP_KERNEL_V1
            elif zero_output_tiles:
                direct_hip_expected_kernel = DIRECT_HIP_ADAPTIVE_ZERO_SKIP_KERNEL_V3
            elif zero_row_col_products:
                direct_hip_expected_kernel = DIRECT_HIP_ADAPTIVE_ZERO_ROW_COL_SKIP_KERNEL_V1
            else:
                direct_hip_expected_kernel = DIRECT_HIP_ADAPTIVE_GROUPED_SCHEDULE_KERNEL_V3
            amdgpu_expected_kernel = "amdgpu_builtin_cdna3_mfma_i32_16x16x32_i8_centered_epilogue_v1"
            device = self.data.get("device")
            target = device.get("gcn_arch") if isinstance(device, dict) else None
            if isinstance(target, str) and target.startswith("gfx110"):
                amdgpu_expected_kernel = "amdgpu_builtin_rdna3_wmma_i32_16x16x16_iu8_centered_epilogue_v1"
            elif target in {"gfx1200", "gfx1201"}:
                amdgpu_expected_kernel = "amdgpu_builtin_rdna4_wmma_i32_16x16x16_iu8_centered_epilogue_v1"
            elif (
                isinstance(target, str)
                and target.startswith("gfx942")
                and _is_int(self.data.get("m"))
                and _is_int(self.data.get("n"))
                and _is_int(self.data.get("k"))
                and self.data.get("m") >= 128
                and self.data.get("n") >= 128
                and self.data.get("k") >= 128
            ):
                amdgpu_expected_kernel = "amdgpu_builtin_cdna3_mfma_i32_32x32x16_i8_centered_epilogue_v1"
            expected_kernels = {
                "hip-direct": direct_hip_expected_kernel,
                "ck": "ck_wmma_cshuffle_tiled_i8_i32_default_moduli_static_centered_epilogue_v3",
                "rocwmma": "rocwmma_i8_i32_signed_tiled_mod251_255_256_hot_residue_v2",
                "amdgpu-builtins": amdgpu_expected_kernel,
            }
            expected_kernel = expected_kernels.get(selected_backend)
            if expected_kernel is not None and selected_kernel != expected_kernel:
                self._error(f"per-tile adaptive {selected_backend} captures must use selected_kernel={expected_kernel}")
            if selected_backend == "hip-vector-alu-int64" and selected_kernel not in VECTOR_ALU_SELECTED_KERNELS:
                self._error("per-tile adaptive hip-vector-alu-int64 captures must use a known vector-ALU selected_kernel")
        prefix_group_count = schedule.get("prefix_group_count")
        max_selected = schedule.get("max_selected_prefix")
        min_selected = schedule.get("min_selected_prefix")
        adaptive_prefix_expected = _is_int(prefix_group_count) and prefix_group_count > 1
        if isinstance(schedule.get("adaptive_prefix_active"), bool) and schedule.get("adaptive_prefix_active") != adaptive_prefix_expected:
            self._error("schedule_metadata.adaptive_prefix_active must match prefix_group_count > 1")
        adaptive_skip_expected = _is_int(max_selected) and _is_int(prefix) and max_selected < prefix
        if isinstance(schedule.get("adaptive_skip_active"), bool) and schedule.get("adaptive_skip_active") != adaptive_skip_expected:
            self._error("schedule_metadata.adaptive_skip_active must match max_selected_prefix < prefix")
        if _is_int(prefix_group_count) and prefix_group_count <= 0:
            self._error("per-tile adaptive captures must use at least one prefix group")
        if _is_int(min_selected) and min_selected <= 0:
            self._error("schedule_metadata.min_selected_prefix must be positive for per-tile adaptive captures")
        if _is_int(max_selected) and _is_int(prefix) and max_selected > prefix:
            self._error("schedule_metadata.max_selected_prefix must be <= prefix")
        if schedule.get("adaptive_prefix_active") is not True and schedule.get("adaptive_skip_active") is not True:
            self._error("per-tile adaptive captures must apply prefix grouping or prefix skipping")
        metadata = self.data.get("timing_metadata")
        if isinstance(metadata, dict):
            if self.data.get("backend_selected") == "hip-direct":
                if metadata.get("gpu_event_timing") is not True:
                    self._error("direct-HIP per-tile adaptive captures must include HIP event timings")
                expected_scope = "direct_hip_bounded_adaptive_default_stream_backend_operation_groups"
                if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                    self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
            elif self.data.get("backend_selected") == "ck":
                if metadata.get("gpu_event_timing") is not True:
                    self._error("CK per-tile adaptive captures must include HIP event operation-group timings")
                expected_scope = "accelerator_backend_default_stream_deep_kernel_events_with_direct_hip_pack_export"
                if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                    self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
            elif self.data.get("backend_selected") == "rocwmma":
                if metadata.get("gpu_event_timing") is not True:
                    self._error("rocWMMA per-tile adaptive captures must include HIP event operation-group timings")
                expected_scope = "accelerator_backend_default_stream_deep_kernel_events_with_direct_hip_pack_export"
                if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                    self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
            elif self.data.get("backend_selected") == "amdgpu-builtins":
                if metadata.get("gpu_event_timing") is not True:
                    self._error("AMDGPU builtin per-tile adaptive captures must include HIP event operation-group timings")
                expected_scope = "accelerator_backend_default_stream_deep_kernel_events_with_direct_hip_pack_export"
                if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                    self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
            elif self.data.get("backend_selected") == "hip-vector-alu-int64":
                if metadata.get("gpu_event_timing") is not True:
                    self._error("vector-ALU per-tile adaptive captures must include HIP event operation-group timings")
                expected_scope = "vector_alu_default_stream_native_int64_operation_groups"
                if metadata.get("gpu_event_timing_source_scope") != expected_scope:
                    self._error(f"timing_metadata.gpu_event_timing_source_scope must be {expected_scope}")
        backend_metadata = self.data.get("backend_metadata")
        if isinstance(backend_metadata, dict):
            expected_epilogues = {
                "ck": "ck_fused_i32_to_centered_residue_then_crt_export",
                "rocwmma": "rocwmma_fused_i32_to_centered_residue_then_crt_export",
                "amdgpu-builtins": "amdgpu_builtin_fused_i32_to_centered_residue_then_crt_export",
                "hip-vector-alu-int64": "direct_int64_export",
            }
            expected_epilogue = expected_epilogues.get(
                self.data.get("backend_selected"), "fused_centered_residue_then_crt_export"
            )
            if backend_metadata.get("epilogue_mode") != expected_epilogue:
                self._error(
                    f"per-tile adaptive captures must use backend_metadata.epilogue_mode={expected_epilogue}"
                )
            expected_workspaces = {
                "cpu-reference": "host_reference_workspace",
                "rocwmma": "resident_device_buffers_with_rocwmma_pack_workspace",
                "amdgpu-builtins": "resident_device_buffers_direct_amdgpu_builtin_matrix_core_no_dense_pack_workspace",
                "hip-vector-alu-int64": (
                    "native_device_i64_u64_buffers"
                    if self._is_vector_alu_runtime_capture()
                    else "benchmark_owned_device_buffers"
                ),
            }
            expected_workspace = expected_workspaces.get(
                self.data.get("backend_selected"), "resident_device_buffers_with_active_prefix_tiled_schedule"
            )
            if self.data.get("backend_selected") == "ck":
                if backend_metadata.get("workspace_mode") not in {
                    "resident_device_buffers_with_ck_canonical_pack_workspace",
                    "resident_device_buffers_with_ck_centered_pack_workspace",
                }:
                    self._error("per-tile adaptive CK captures must use a known CK pack workspace mode")
            elif backend_metadata.get("workspace_mode") != expected_workspace:
                self._error(
                    f"per-tile adaptive captures must use backend_metadata.workspace_mode={expected_workspace}"
                )
            expected_isas = {
                "cpu-reference": "not_applicable_cpu",
                "ck": "ck_cshuffle_int8_matrix_isa_gate_no_divide",
                "rocwmma": "rocwmma_i8_matrix_isa_gate_no_divide",
                "amdgpu-builtins": "amdgpu_builtin_matrix_isa_gate_no_divide",
                "hip-vector-alu-int64": "source_level_192bit_limb_accumulator_no_matrix_engine",
            }
            expected_isa = expected_isas.get(
                self.data.get("backend_selected"), "rns8_hip_direct_reciprocal_isa_gate"
            )
            if backend_metadata.get("isa_evidence") != expected_isa:
                self._error(
                    f"per-tile adaptive captures must use backend_metadata.isa_evidence={expected_isa}"
                )

