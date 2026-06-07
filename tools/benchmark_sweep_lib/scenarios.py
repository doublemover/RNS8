from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from metadata_registry_constants import (
    GROUPED_DISPATCH_EXECUTION_STRATEGIES,
    PROMOTION_SCOPES,
    SCENARIO_REVIEW_MODES,
)

from .config import ScenarioItem, SweepCase

def _optional_tuple(value: Any, *, label: str) -> tuple[Any, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise SystemExit(f"{label} must be a nonempty list")
    return tuple(value)


def _tuple_or_default(value: Any, *, label: str, default: tuple[Any, ...]) -> tuple[Any, ...]:
    converted = _optional_tuple(value, label=label)
    return default if converted is None else converted


def _string_tuple_or_default(value: Any, *, label: str, default: tuple[str, ...]) -> tuple[str, ...]:
    values = _tuple_or_default(value, label=label, default=default)
    for item in values:
        if not isinstance(item, str) or not item:
            raise SystemExit(f"{label} must contain nonempty strings")
    return tuple(values)


def _required_string(raw: dict[str, Any], key: str, *, label: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{label}.{key} must be a nonempty string")
    return value


def _optional_string(raw: dict[str, Any], key: str, *, label: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{label}.{key} must be a nonempty string when present")
    return value


def _required_int(raw: dict[str, Any], key: str, *, label: str, positive: bool = False) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SystemExit(f"{label}.{key} must be an integer")
    if positive and value <= 0:
        raise SystemExit(f"{label}.{key} must be positive")
    return value


def _optional_int(raw: dict[str, Any], key: str, *, label: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise SystemExit(f"{label}.{key} must be an integer when present")
    return value


def _bool_or_default(raw: dict[str, Any], key: str, *, label: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise SystemExit(f"{label}.{key} must be boolean")
    return value


def _case_from_data(value: Any, cases: dict[str, SweepCase], *, label: str) -> SweepCase:
    if isinstance(value, str):
        if value not in cases:
            raise SystemExit(f"{label} references unknown case alias {value!r}")
        return cases[value]
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a case alias string or object")
    case_label = f"{label}"
    return SweepCase(
        _required_string(value, "name", label=case_label),
        _required_int(value, "m", label=case_label, positive=True),
        _required_int(value, "n", label=case_label, positive=True),
        _required_int(value, "k", label=case_label, positive=True),
        tile_m=_required_int(value, "tile_m", label=case_label, positive=True),
        tile_n=_required_int(value, "tile_n", label=case_label, positive=True),
        bound_mode=_required_string(value, "bound_mode", label=case_label),
        input_profile=_required_string(value, "input_profile", label=case_label),
        require_adaptive=_bool_or_default(value, "require_adaptive", label=case_label, default=False),
        promotable=_bool_or_default(value, "promotable", label=case_label, default=True),
    )


def _finite_moduli(value: Any, *, label: str) -> tuple[int | None, ...]:
    values = _tuple_or_default(value, label=label, default=(None,))
    for item in values:
        if item is not None and (not isinstance(item, int) or isinstance(item, bool) or item <= 0):
            raise SystemExit(f"{label} must contain positive integers or null")
    return values


def _exact_wide_limb_counts(value: Any, *, label: str) -> tuple[int | None, ...]:
    values = _tuple_or_default(value, label=label, default=(None,))
    for item in values:
        if item is not None and (not isinstance(item, int) or isinstance(item, bool) or item <= 0):
            raise SystemExit(f"{label} must contain positive integers or null")
    return values


def _metadata(raw: dict[str, Any], *, label: str) -> dict[str, Any] | None:
    metadata = raw.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise SystemExit(f"{label}.metadata must be an object")
    if isinstance(metadata, dict):
        strategy = metadata.get("grouped_strategy_expectation")
        if strategy is not None and strategy not in GROUPED_DISPATCH_EXECUTION_STRATEGIES:
            raise SystemExit(f"{label} has unregistered grouped_strategy_expectation={strategy!r}")
    return metadata


def _scenario_review_mode(raw: dict[str, Any], *, label: str) -> str:
    value = _required_string(raw, "review_mode_expectation", label=label)
    if value not in SCENARIO_REVIEW_MODES:
        raise SystemExit(f"{label}.review_mode_expectation must be a registered scenario review mode")
    return value


def _promotion_eligibility(raw: dict[str, Any], *, label: str) -> str:
    value = _required_string(raw, "promotion_eligibility", label=label)
    if value not in PROMOTION_SCOPES:
        raise SystemExit(f"{label}.promotion_eligibility must be a registered promotion scope")
    return value


def load_scenario_data_family(path: Path, cases: dict[str, SweepCase] | None = None) -> list[ScenarioItem]:
    cases = cases or {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid scenario JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise SystemExit(f"{path}: scenario data must be an object with schema_version=1")
    family = data.get("family")
    if not isinstance(family, str) or not family:
        raise SystemExit(f"{path}: scenario data must declare a nonempty family")
    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        raise SystemExit(f"{path}: scenario data must declare items")

    items: list[ScenarioItem] = []
    for index, raw in enumerate(raw_items):
        label = f"{path}: items[{index}]"
        if not isinstance(raw, dict):
            raise SystemExit(f"{label} must be an object")
        grouped_tasks = raw.get("grouped_dispatch_tasks", 1)
        if not isinstance(grouped_tasks, int) or isinstance(grouped_tasks, bool) or grouped_tasks <= 0:
            raise SystemExit(f"{label}.grouped_dispatch_tasks must be a positive integer")
        resident_candidate = (
            _required_string(raw, "resident_redesign_candidate", label=label)
            if "resident_redesign_candidate" in raw
            else ""
        )
        resident_dimensions = _string_tuple_or_default(
            raw.get("resident_redesign_dimensions"),
            label=f"{label}.resident_redesign_dimensions",
            default=(),
        )
        if resident_dimensions and not resident_candidate:
            raise SystemExit(f"{label}.resident_redesign_dimensions requires resident_redesign_candidate")
        if resident_candidate and not resident_dimensions:
            raise SystemExit(f"{label}.resident_redesign_candidate requires resident_redesign_dimensions")
        items.append(
            ScenarioItem(
                family,
                _required_string(raw, "name", label=label),
                _required_string(raw, "semantics", label=label),
                _case_from_data(raw.get("case"), cases, label=f"{label}.case"),
                _required_string(raw, "evidence_scope", label=label),
                _required_string(raw, "output_domain", label=label),
                _required_string(raw, "rationale", label=label),
                _scenario_review_mode(raw, label=label),
                _promotion_eligibility(raw, label=label),
                backends=_optional_tuple(raw.get("backends"), label=f"{label}.backends"),
                pack_mode=_required_string(raw, "pack_mode", label=label) if "pack_mode" in raw else "per_repeat_repack",
                finite_moduli=_finite_moduli(raw.get("finite_moduli"), label=f"{label}.finite_moduli"),
                exact_wide_limb_counts=_exact_wide_limb_counts(
                    raw.get("exact_wide_limb_counts"),
                    label=f"{label}.exact_wide_limb_counts",
                ),
                residue_chain_length=_required_int(raw, "residue_chain_length", label=label, positive=True)
                if "residue_chain_length" in raw
                else 1,
                residue_chain_final_export=_bool_or_default(
                    raw,
                    "residue_chain_final_export",
                    label=label,
                    default=False,
                ),
                residue_chain_independent_final_export=_bool_or_default(
                    raw,
                    "residue_chain_independent_final_export",
                    label=label,
                    default=False,
                ),
                output_ld_padding=_required_int(raw, "output_ld_padding", label=label)
                if "output_ld_padding" in raw
                else 0,
                host_api_batch_size=_required_int(raw, "host_api_batch_size", label=label, positive=True)
                if "host_api_batch_size" in raw
                else 1,
                oneshot=_bool_or_default(raw, "oneshot", label=label, default=False),
                native_to_rns_bridge=_bool_or_default(raw, "native_to_rns_bridge", label=label, default=False),
                vector_to_rns_chain=_bool_or_default(raw, "vector_to_rns_chain", label=label, default=False),
                vector_to_rns_chain_host_repack_control=_bool_or_default(
                    raw,
                    "vector_to_rns_chain_host_repack_control",
                    label=label,
                    default=False,
                ),
                prefix_policy=_optional_string(raw, "prefix_policy", label=label),
                max_prefix=_optional_int(raw, "max_prefix", label=label),
                bound_source=_optional_string(raw, "bound_source", label=label),
                next_op_hint=_optional_string(raw, "next_op_hint", label=label),
                residue_channel_fusion=_bool_or_default(
                    raw,
                    "residue_channel_fusion",
                    label=label,
                    default=False,
                ),
                modulus_set=_required_string(raw, "modulus_set", label=label) if "modulus_set" in raw else "default",
                tile_shape_variant=_required_string(raw, "tile_shape_variant", label=label)
                if "tile_shape_variant" in raw
                else "default",
                export_variant=_required_string(raw, "export_variant", label=label)
                if "export_variant" in raw
                else "default",
                reconstruction_variant=_required_string(raw, "reconstruction_variant", label=label)
                if "reconstruction_variant" in raw
                else "default_garner",
                grouped_dispatch_tasks=grouped_tasks,
                hip_graph_replay=_bool_or_default(raw, "hip_graph_replay", label=label, default=False),
                workload_proxy=_required_string(raw, "workload_proxy", label=label)
                if "workload_proxy" in raw
                else "none",
                resident_lifetime=_bool_or_default(raw, "resident_lifetime", label=label, default=False),
                workspace_arena=_bool_or_default(raw, "workspace_arena", label=label, default=False),
                adaptive_grouped_scheduler=_bool_or_default(
                    raw,
                    "adaptive_grouped_scheduler",
                    label=label,
                    default=False,
                ),
                streaming_overlap=_bool_or_default(raw, "streaming_overlap", label=label, default=False),
                k_block_policy=_required_string(raw, "k_block_policy", label=label)
                if "k_block_policy" in raw
                else "auto",
                resident_redesign_candidate=resident_candidate,
                resident_redesign_dimensions=resident_dimensions,
                release_gate=_required_string(raw, "release_gate", label=label) if "release_gate" in raw else "none",
                verification_amortization=_required_string(raw, "verification_amortization", label=label)
                if "verification_amortization" in raw
                else "none",
                error_detection_policy=_required_string(raw, "error_detection_policy", label=label)
                if "error_detection_policy" in raw
                else "none",
                cpu_small_shape_selector=_required_string(raw, "cpu_small_shape_selector", label=label)
                if "cpu_small_shape_selector" in raw
                else "none",
                incremental_result_cache=_required_string(raw, "incremental_result_cache", label=label)
                if "incremental_result_cache" in raw
                else "none",
                include_wrap64_candidate=_bool_or_default(
                    raw,
                    "include_wrap64_candidate",
                    label=label,
                    default=False,
                ),
                metadata=_metadata(raw, label=label),
            )
        )
    return items


def load_scenario_data_catalog(directory: Path) -> dict[str, list[ScenarioItem]]:
    catalog: dict[str, list[ScenarioItem]] = {}
    for path in sorted(directory.glob("*.json")):
        items = load_scenario_data_family(path)
        if not items:
            continue
        family = items[0].family
        if family in catalog:
            raise SystemExit(f"{path}: duplicate scenario family {family!r}")
        catalog[family] = items
    if not catalog:
        raise SystemExit(f"{directory}: no scenario data files found")
    return catalog


