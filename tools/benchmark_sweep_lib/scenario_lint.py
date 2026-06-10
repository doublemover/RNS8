from __future__ import annotations

from pathlib import Path

from .config import (
    BOUNDED_BACKENDS,
    EXACT_WIDE_BACKENDS,
    FINITE_BACKENDS,
    WRAP64_BACKENDS,
)
from .scenarios import load_scenario_data_catalog


RELEASE_REVIEW_SCOPES = {
    "release_review_candidate",
    "execution_path_evidence",
}

VALID_OUTPUT_DOMAINS = {
    "host_export",
    "exact_wide_limb_host",
    "finite_u8_canonical_host",
    "finite_u8_canonical_host_export",
    "wrap64_host",
    "low64_wrap_u64_host_output",
    "residue_current_rns",
    "rns_residue_current",
    "residue_current_then_final_export",
    "residue_current_then_final_limb_export",
    "native_i64_u64_host",
    "native_i64_u64_current",
    "native_i64_host_output",
    "native_u64_host",
    "native_u64_host_output",
    "exact_wide_signed_limbs",
    "exact_wide_unsigned_limbs",
}


def validate_scenario_catalog(directory: Path) -> list[str]:
    catalog = load_scenario_data_catalog(directory)
    errors: list[str] = []

    for family, items in catalog.items():
        for item in items:
            label = f"{family}/{item.name}"

            if (
                item.promotion_eligibility in RELEASE_REVIEW_SCOPES
                and item.review_mode_expectation not in {"release", "release-or-smoke"}
            ):
                errors.append(
                    f"{label}: promotion_eligibility={item.promotion_eligibility} "
                    f"requires review_mode_expectation=release, got {item.review_mode_expectation}"
                )

            if item.output_domain not in VALID_OUTPUT_DOMAINS:
                errors.append(f"{label}: unregistered output_domain={item.output_domain}")

            CHAIN_FAMILIES = {"rns-chain", "rns-chain-final-output", "rns-chain-final-output-broader", "fhe-lattice-proxies", "fhe-lattice-proxy-starfoundry",
                             "exact-wide-output-chain", "exact-wide-output-chain-broader",
                             "direct-hip-reuse-expansion", "residue-channel-fusion"}
            if (item.output_domain in {"residue_current_rns", "rns_residue_current", "residue_current_then_final_export", "residue_current_then_final_limb_export"}
                and item.next_op_hint is None
                and item.family not in CHAIN_FAMILIES):
                errors.append(f"{label}: residue_current_rns output requires next_op_hint")

            GRAPH_ONLY_FAMILIES = {"hip-graph-replay"}
            if item.hip_graph_replay and item.family not in GRAPH_ONLY_FAMILIES:
                has_non_graph = any(
                    other.name != item.name and not other.hip_graph_replay
                    and other.semantics == item.semantics
                    and other.case == item.case
                    for other in items
                )
                if not has_non_graph:
                    errors.append(
                        f"{label}: hip_graph_replay requires a non-graph baseline "
                        f"in the same scenario family with matching semantics/shape"
                    )

            REUSE_ONLY_FAMILIES = {"repeated-b", "reuse-contract", "direct-hip-reuse-expansion", "exact-wide-output-chain", "exact-wide-output-chain-broader", "hip-graph-replay", "fhe-lattice-proxies", "fhe-lattice-proxy-starfoundry", "resident-lifetime-arena"}
            if item.pack_mode in {"prepacked_reuse", "prepacked_reuse_a", "prepacked_reuse_b"} and item.family not in REUSE_ONLY_FAMILIES:
                has_non_reuse = any(
                    other.name != item.name
                    and other.pack_mode == "per_repeat_repack"
                    and other.semantics == item.semantics
                    and other.case == item.case
                    for other in items
                )
                if not has_non_reuse:
                    errors.append(
                        f"{label}: prepacked_reuse requires a per_repeat_repack baseline "
                        f"in the same scenario family with matching semantics/shape"
                    )

            if item.backends is not None:
                if item.semantics in {"bounded-i64", "bounded-u64"}:
                    supported = list(BOUNDED_BACKENDS) + ["auto"]
                elif item.semantics in {"exact-wide-signed", "exact-wide-unsigned"}:
                    supported = list(EXACT_WIDE_BACKENDS) + ["auto"]
                elif item.semantics in {"finite-u8-ring", "finite-u8-field"}:
                    supported = list(FINITE_BACKENDS) + ["auto"]
                elif item.semantics == "wrap-u64":
                    supported = list(WRAP64_BACKENDS) + ["auto"]
                else:
                    supported = []
                for backend in item.backends:
                    if backend not in supported:
                        errors.append(
                            f"{label}: backend={backend} not supported for semantics={item.semantics}"
                        )

            if item.prefix_policy == "fixed-requested" and item.max_prefix is not None:
                if item.max_prefix < 1 or item.max_prefix > 28:
                    errors.append(f"{label}: max_prefix={item.max_prefix} out of valid range 1..28")

    return errors
