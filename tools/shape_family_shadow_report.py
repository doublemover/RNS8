#!/usr/bin/env python3
"""Build non-routing shape-family AUTO recommendations from reviewed cache entries."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import install_autotune_cache


DEFAULT_OUT_DIR = Path("temp") / "shape-family-shadow"


@dataclass(frozen=True)
class ShapeQuery:
    semantic_contract: str
    m: int
    n: int
    k: int
    target_id: str
    finite_modulus: int = 0
    layout: str = "row_major"
    target_family: str = "unknown"
    signedness: str = "unknown"
    output_contract: str = "default"
    export_selector: str = "default"
    limb_count: int = 0


def target_family_for_id(target_id: str) -> str:
    if target_id == "cpu":
        return "cpu"
    if target_id.startswith("gfx94") or target_id in {"gfx90a", "gfx950"}:
        return "cdna"
    if target_id.startswith("gfx10") or target_id.startswith("gfx11") or target_id.startswith("gfx12"):
        return "rdna"
    return "unknown"


def signedness_for_semantics(semantic: str) -> str:
    if semantic in {"bounded_i64", "exact_wide_signed"}:
        return "signed"
    if semantic in {"bounded_u64", "exact_wide_unsigned", "wrap_u64_mod_2_64"}:
        return "unsigned"
    if semantic.startswith("finite_"):
        return "finite"
    return "unknown"


def output_contract_for_semantics(semantic: str) -> str:
    if semantic == "exact_wide_signed":
        return "exact_wide_signed_limbs"
    if semantic == "exact_wide_unsigned":
        return "exact_wide_unsigned_limbs"
    if semantic.startswith("finite_"):
        return "finite_u8_canonical_host_export"
    if semantic == "wrap_u64_mod_2_64":
        return "native_u64_host"
    return "native_i64_u64_host"


def _require_positive_int(fields: dict[str, str], name: str) -> int:
    try:
        value = int(fields[name], 10)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def parse_query(text: str) -> ShapeQuery:
    fields: dict[str, str] = {}
    for raw_part in text.replace(",", ";").split(";"):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"query part {part!r} must be key=value")
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip()
    semantic = fields.get("semantic_contract") or fields.get("semantics")
    target = fields.get("target_id") or fields.get("target")
    if not semantic:
        raise ValueError("query must include semantic_contract or semantics")
    if not target:
        raise ValueError("query must include target_id or target")
    finite_modulus = int(fields.get("finite_modulus", "0"), 10)
    if finite_modulus < 0:
        raise ValueError("finite_modulus must be >= 0")
    limb_count = int(fields.get("limb_count", "0"), 10)
    if limb_count < 0:
        raise ValueError("limb_count must be >= 0")
    return ShapeQuery(
        semantic_contract=semantic,
        m=_require_positive_int(fields, "m"),
        n=_require_positive_int(fields, "n"),
        k=_require_positive_int(fields, "k"),
        target_id=target,
        finite_modulus=finite_modulus,
        layout=fields.get("layout", "row_major"),
        target_family=fields.get("target_family", target_family_for_id(target)),
        signedness=fields.get("signedness", signedness_for_semantics(semantic)),
        output_contract=fields.get("output_contract", output_contract_for_semantics(semantic)),
        export_selector=fields.get("export_selector", "default"),
        limb_count=limb_count,
    )


def shape_bucket(m: int, n: int, k: int) -> str:
    if n == 1:
        return "skinny_n1"
    max_dim = max(m, n, k)
    if max_dim <= 128:
        return "small_1_128"
    if max_dim <= 1024:
        return "medium_129_1024"
    if max_dim <= 4096:
        return "large_1025_4096"
    return "very_large_over_4096"


def query_key(query: ShapeQuery) -> str:
    return (
        f"target={query.target_id};target_family={query.target_family};semantics={query.semantic_contract};"
        f"signedness={query.signedness};finite_modulus={query.finite_modulus};layout={query.layout};"
        f"output_contract={query.output_contract};export_selector={query.export_selector};"
        f"limb_count={query.limb_count};bucket={shape_bucket(query.m, query.n, query.k)}"
    )


def entry_query(entry: dict[str, Any]) -> ShapeQuery:
    shape = entry["shape"]
    semantic = str(entry["semantic_contract"])
    target = str(entry["target_id"])
    return ShapeQuery(
        semantic_contract=semantic,
        m=int(shape["m"]),
        n=int(shape["n"]),
        k=int(shape["k"]),
        target_id=target,
        finite_modulus=int(entry.get("finite_modulus") or 0),
        layout=str(entry.get("layout", "row_major")),
        target_family=str(entry.get("target_family") or target_family_for_id(target)),
        signedness=str(entry.get("signedness") or signedness_for_semantics(semantic)),
        output_contract=str(entry.get("output_contract") or output_contract_for_semantics(semantic)),
        export_selector=str(entry.get("export_selector") or "default"),
        limb_count=int(entry.get("limb_count") or 0),
    )


def boundary_matches(query: ShapeQuery, basis: ShapeQuery) -> bool:
    return (
        basis.semantic_contract == query.semantic_contract
        and basis.target_id == query.target_id
        and basis.target_family == query.target_family
        and basis.signedness == query.signedness
        and basis.finite_modulus == query.finite_modulus
        and basis.layout == query.layout
        and basis.output_contract == query.output_contract
        and basis.export_selector == query.export_selector
        and basis.limb_count == query.limb_count
    )


def exact_cache_key_for(query: ShapeQuery, entry: dict[str, Any]) -> bool:
    basis = entry_query(entry)
    return boundary_matches(query, basis) and basis.m == query.m and basis.n == query.n and basis.k == query.k


def same_family(query: ShapeQuery, entry: dict[str, Any]) -> bool:
    basis = entry_query(entry)
    return boundary_matches(query, basis) and shape_bucket(basis.m, basis.n, basis.k) == shape_bucket(query.m, query.n, query.k)


def entry_distance(query: ShapeQuery, entry: dict[str, Any]) -> float:
    basis = entry_query(entry)
    ratios = [
        math.log2(max(query.m, basis.m) / max(1, min(query.m, basis.m))),
        math.log2(max(query.n, basis.n) / max(1, min(query.n, basis.n))),
        math.log2(max(query.k, basis.k) / max(1, min(query.k, basis.k))),
    ]
    median = entry.get("measured_medians_us", {}).get("end_to_end")
    timing_bias = float(median) * 1e-9 if isinstance(median, (int, float)) else 0.0
    return sum(ratios) + timing_bias


def query_payload(query: ShapeQuery) -> dict[str, Any]:
    return {
        "semantic_contract": query.semantic_contract,
        "m": query.m,
        "n": query.n,
        "k": query.k,
        "target_id": query.target_id,
        "target_family": query.target_family,
        "finite_modulus": query.finite_modulus,
        "layout": query.layout,
        "signedness": query.signedness,
        "output_contract": query.output_contract,
        "export_selector": query.export_selector,
        "limb_count": query.limb_count,
        "shape_bucket": shape_bucket(query.m, query.n, query.k),
    }


def recommendation_for(query: ShapeQuery, entries: list[dict[str, Any]]) -> dict[str, Any]:
    exact = [entry for entry in entries if exact_cache_key_for(query, entry)]
    family = [entry for entry in entries if same_family(query, entry)]
    candidates = exact or family
    candidate = min(candidates, key=lambda entry: entry_distance(query, entry)) if candidates else None
    representative_entries = sorted(family, key=lambda entry: entry_distance(query, entry))
    blockers: list[str] = ["shape_family_shadow_only_no_routing_change"]
    if candidate is None:
        blockers.append("missing_same_family_reviewed_entry")
    elif not exact:
        blockers.extend(["exact_query_not_reviewed", "representative_matrix_requires_same_target_layout_contract_review"])
    else:
        blockers.append("exact_cache_hit_already_owned_by_current_AUTO_cache")
    if query.target_family == "unknown":
        blockers.append("target_family_not_classified")
    basis = entry_query(candidate) if candidate else None
    return {
        "query": query_payload(query),
        "family_key": query_key(query),
        "would_recommend": candidate is not None,
        "recommendation_is_exact_cache_hit": bool(exact),
        "selector_explanation": "exact reviewed cache hit"
        if exact
        else ("nearest same-family reviewed cache entry" if candidate else "no same-family reviewed cache entry"),
        "representative_matrix": {
            "reviewed_entry_count": len(representative_entries),
            "exact_entry_count": len(exact),
            "shapes": [
                f"{item['shape']['m']}x{item['shape']['n']}x{item['shape']['k']}"
                for item in representative_entries[:8]
            ],
            "backends": sorted({str(item["selected_backend"]) for item in representative_entries}),
            "targets": sorted({str(item["target_id"]) for item in representative_entries}),
        },
        "recommended_backend": candidate.get("selected_backend") if candidate else None,
        "recommended_kernel": candidate.get("selected_kernel") if candidate else None,
        "basis_cache_key": candidate.get("key") if candidate else None,
        "basis_shape": {
            "m": basis.m,
            "n": basis.n,
            "k": basis.k,
            "shape_bucket": shape_bucket(basis.m, basis.n, basis.k),
        }
        if basis
        else None,
        "basis_end_to_end_median_us": candidate.get("measured_medians_us", {}).get("end_to_end") if candidate else None,
        "promotion_eligible": False,
        "promotion_blockers": blockers,
    }


def family_summary(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        families.setdefault(query_key(entry_query(entry)), []).append(entry)
    rows = []
    for key, items in sorted(families.items()):
        rows.append(
            {
                "family_key": key,
                "reviewed_entry_count": len(items),
                "backends": sorted({str(item["selected_backend"]) for item in items}),
                "targets": sorted({str(item["target_id"]) for item in items}),
                "shapes": sorted({f"{item['shape']['m']}x{item['shape']['n']}x{item['shape']['k']}" for item in items}),
            }
        )
    return rows


def build_report(cache_path: Path, queries: list[ShapeQuery]) -> dict[str, Any]:
    entries = install_autotune_cache.read_cache(cache_path)
    families = family_summary(entries)
    return {
        "schema": "rns8_shape_family_shadow_report_v2",
        "policy": "non_routing_shape_family_recommendations_require_exact_review_before_AUTO",
        "cache_path": str(cache_path),
        "reviewed_cache_entry_count": len(entries),
        "family_count": len(families),
        "boundary_fields": [
            "target_id",
            "target_family",
            "semantic_contract",
            "signedness",
            "finite_modulus",
            "layout",
            "output_contract",
            "export_selector",
            "limb_count",
        ],
        "families": families,
        "recommendations": [recommendation_for(query, entries) for query in queries],
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "shape-family-shadow-report.json"
    md_path = out_dir / "shape-family-shadow-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# RNS8 Shape-Family AUTO Shadow Report",
        "",
        f"- Policy: `{report['policy']}`",
        f"- Reviewed cache entries: `{report['reviewed_cache_entry_count']}`",
        "",
        "| semantic | shape | target | output | recommendation | exact hit | blockers |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["recommendations"]:
        query = row["query"]
        lines.append(
            "| `{semantic}` | `{shape}` | `{target}` | `{output}` | `{backend}` / `{kernel}` | `{exact}` | `{blockers}` |".format(
                semantic=query["semantic_contract"],
                shape=f"{query['m']}x{query['n']}x{query['k']}",
                target=query["target_id"],
                output=query["output_contract"],
                backend=row["recommended_backend"],
                kernel=row["recommended_kernel"],
                exact=row["recommendation_is_exact_cache_hit"],
                blockers=", ".join(row["promotion_blockers"]),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="reviewed autotune cache JSON")
    parser.add_argument("--query", action="append", default=[], help="semicolon/comma key=value shape query")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    queries = [parse_query(text) for text in args.query]
    report = build_report(args.cache, queries)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for label, path in write_outputs(report, args.out_dir).items():
            print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
