from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import CSV_FIELDS

def csv_value(value: Any) -> str | int | float | None:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def write_outputs(database: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evidence_database.json"
    json_path.write_text(json.dumps(database, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = out_dir / "evidence_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in database["rows"]:
            writer.writerow({field: csv_value(row.get(field)) for field in CSV_FIELDS})

    markdown_path = out_dir / "evidence_summary.md"
    write_markdown(database, markdown_path)
    return {
        "evidence_database": str(json_path),
        "evidence_rows_csv": str(csv_path),
        "evidence_summary": str(markdown_path),
    }


def format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, int):
        return str(value)
    if value is None:
        return ""
    return str(value)


def format_isa_brief(row: dict[str, Any]) -> str:
    if not row.get("isa_report_count"):
        return ""
    parts = [f"reports={row.get('isa_report_count')}"]
    for key, label in (
        ("isa_matrix_instruction_count", "matrix"),
        ("isa_dense_integer_matrix_instruction_count", "dense_int"),
        ("isa_sparse_integer_matrix_instruction_count", "sparse_int"),
        ("isa_wmma_count", "wmma"),
        ("isa_mfma_count", "mfma"),
        ("isa_global_store_count", "stores"),
    ):
        value = row.get(key)
        if value is not None:
            parts.append(f"{label}={value}")
    return ";".join(parts)


def append_count_table(
    lines: list[str],
    *,
    heading: str,
    value_heading: str,
    counts: Counter[str],
) -> None:
    if not counts:
        return
    lines.extend(["", heading, "", f"| {value_heading} | captures |", "|---|---:|"])
    for name, count in sorted(counts.items()):
        lines.append(f"| {name} | {count} |")


def scenario_metadata_counts(database: dict[str, Any], row_key: str) -> Counter[str]:
    return Counter(str(row.get(row_key)) for row in database["rows"] if row.get(row_key))


def append_roofline_priority_table(lines: list[str], heading: str, priority: list[dict[str, Any]]) -> None:
    if not priority:
        return
    lines.extend(
        [
            "",
            heading,
            "",
            "| rank | target | scenario | semantics | target id | captures | bottleneck us | e2e us | median share | median GOP/s | median AI ops/B | backends | hint |",
            "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for item in priority:
        lines.append(
            "| {rank} | {target} | {scenario} | {semantics} | {target_id} | {captures} | {bottleneck_us} | {e2e_us} | {share} | {gops} | {ai} | {backends} | {hint} |".format(
                rank=item.get("rank"),
                target=item.get("roofline_target"),
                scenario=item.get("scenario_family"),
                semantics=item.get("semantics"),
                target_id=item.get("target_id"),
                captures=item.get("captures"),
                bottleneck_us=format_number(item.get("total_bottleneck_us")),
                e2e_us=format_number(item.get("total_end_to_end_us")),
                share=format_number(item.get("median_bottleneck_share")),
                gops=format_number(item.get("median_measured_gops")),
                ai=format_number(item.get("median_arithmetic_intensity_ops_per_byte")),
                backends=",".join(item.get("backends") or []),
                hint=item.get("optimization_hint"),
            )
        )


def write_markdown(database: dict[str, Any], path: Path) -> None:
    lines = [
        "# RNS8 Evidence Database Summary",
        "",
        f"- schema_version: `{database.get('schema_version')}`",
        f"- generated_utc: `{database.get('generated_utc')}`",
        f"- captures: `{database.get('capture_count')}`",
        f"- skipped_invalid_capture_count: `{database.get('summary', {}).get('skipped_invalid_capture_count', 0)}`",
        f"- isa_report_count: `{database.get('summary', {}).get('isa_report_count', 0)}`",
        f"- captures_with_isa_resources: `{database.get('summary', {}).get('captures_with_isa_resources', 0)}`",
        "",
        "## Bottlenecks",
        "",
        "| class | captures |",
        "|---|---:|",
    ]
    for name, count in database["summary"]["bottleneck_counts"].items():
        lines.append(f"| {name} | {count} |")
    append_count_table(
        lines,
        heading="## Pack Split Dominance",
        value_heading="dominant operand",
        counts=Counter(database.get("summary", {}).get("pack_split_dominant_counts") or {}),
    )
    skipped = database.get("skipped_invalid_captures") or []
    if skipped:
        lines.extend(["", "## Skipped Invalid Captures", "", "| capture | error |", "|---|---|"])
        for item in skipped[:40]:
            lines.append(f"| {item.get('capture_path')} | {item.get('error')} |")
        if len(skipped) > 40:
            lines.append(f"| ... | {len(skipped) - 40} additional skipped captures |")
    append_roofline_priority_table(
        lines,
        "## GPU Roofline Priority",
        database.get("summary", {}).get("gpu_roofline_priority") or [],
    )
    append_roofline_priority_table(
        lines,
        "## Roofline Priority",
        database.get("summary", {}).get("roofline_priority") or [],
    )
    lines.extend(["", "## Scenario Families", "", "| family | captures |", "|---|---:|"])
    for name, count in database["summary"]["scenario_counts"].items():
        lines.append(f"| {name} | {count} |")
    metadata_tables = (
        ("source_role", "scenario_source_role"),
        ("workflow_name", "scenario_workflow_name"),
        ("reuse_profile", "scenario_reuse_profile"),
        ("lowering_role", "scenario_lowering_role"),
        ("output_domain_requirement", "scenario_output_domain_requirement"),
        ("large_shape_role", "scenario_large_shape_role"),
        ("promotion_scope", "scenario_promotion_scope"),
        ("grouping_role", "scenario_grouping_role"),
        ("bridge_role", "scenario_bridge_role"),
        ("modulus_role", "scenario_modulus_role"),
        ("prime_or_composite", "scenario_prime_or_composite"),
    )
    if any(scenario_metadata_counts(database, row_key) for _, row_key in metadata_tables):
        lines.extend(["", "## Scenario Metadata"])
        for label, row_key in metadata_tables:
            append_count_table(
                lines,
                heading=f"### {label}",
                value_heading=label,
                counts=scenario_metadata_counts(database, row_key),
            )
    isa_rows = [row for row in database["rows"] if row.get("isa_report_count")]
    if isa_rows:
        lines.extend(
            [
                "",
                "## ISA Resources",
                "",
                "| backend | target | captures | reports | matrix | dense int | sparse int | WMMA | MFMA | SMFMAC | SWMMAC | global stores | LDS mentions | LDS bytes | scratch bytes | waits | instructions | VGPR | SGPR | occupancy |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in isa_rows:
            groups[(str(row.get("backend")), ",".join(row.get("isa_report_targets") or []))].append(row)
        for (backend, target), grouped_rows in sorted(groups.items()):
            representative = grouped_rows[0]
            lines.append(
                "| {backend} | {target} | {captures} | {reports} | {matrix} | {dense_int} | {sparse_int} | {wmma} | {mfma} | {smfmac} | {swmmac} | {stores} | {lds} | {lds_bytes} | {scratch} | {waits} | {instructions} | {vgpr} | {sgpr} | {occupancy} |".format(
                    backend=backend,
                    target=target,
                    captures=len(grouped_rows),
                    reports=representative.get("isa_report_count"),
                    matrix=format_number(representative.get("isa_matrix_instruction_count")),
                    dense_int=format_number(representative.get("isa_dense_integer_matrix_instruction_count")),
                    sparse_int=format_number(representative.get("isa_sparse_integer_matrix_instruction_count")),
                    wmma=format_number(representative.get("isa_wmma_count")),
                    mfma=format_number(representative.get("isa_mfma_count")),
                    smfmac=format_number(representative.get("isa_smfmac_count")),
                    swmmac=format_number(representative.get("isa_swmmac_count")),
                    stores=format_number(representative.get("isa_global_store_count")),
                    lds=format_number(representative.get("isa_lds_mentions")),
                    lds_bytes=format_number(representative.get("isa_lds_bytes")),
                    scratch=format_number(representative.get("isa_scratch_bytes")),
                    waits=format_number(representative.get("isa_wait_instructions")),
                    instructions=format_number(representative.get("isa_instruction_lines")),
                    vgpr=format_number(representative.get("isa_vgpr_count")),
                    sgpr=format_number(representative.get("isa_sgpr_count")),
                    occupancy=format_number(representative.get("isa_occupancy")),
                )
            )
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| scenario | semantics | backend | kernel | shape | bottleneck | pack split | e2e us | GOP/s | AI ops/B | ISA | blockers |",
            "|---|---|---|---|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    rows = sorted(
        database["rows"],
        key=lambda row: (
            str(row.get("scenario_family")),
            str(row.get("semantics")),
            float(row.get("median_end_to_end_us") or 0.0),
        ),
    )
    for row in rows:
        shape = f"{row.get('m')}x{row.get('n')}x{row.get('k')}"
        blockers = ",".join(str(item) for item in row.get("promotion_blockers") or [])
        lines.append(
            "| {scenario} | {semantics} | {backend} | {kernel} | {shape} | {bottleneck} | {pack_split} | {e2e} | {gops} | {ai} | {isa} | {blockers} |".format(
                scenario=row.get("scenario_family"),
                semantics=row.get("semantics"),
                backend=row.get("backend"),
                kernel=row.get("selected_kernel"),
                shape=shape,
                bottleneck=row.get("bottleneck_class"),
                pack_split=row.get("pack_split_dominant_operand") or "",
                e2e=format_number(row.get("median_end_to_end_us")),
                gops=format_number(row.get("measured_gops")),
                ai=format_number(row.get("arithmetic_intensity_ops_per_byte")),
                isa=format_isa_brief(row),
                blockers=blockers or "",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


