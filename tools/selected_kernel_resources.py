"""Selected static kernel resources with immutable artifact and symbol custody."""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any
from evidence_database_lib.isa import isa_index_from_reports, lookup_isa_resources
from target_qualification import EvidenceStore, file_digest


def _number(value: Any) -> float | None:
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


@dataclass
class SelectedStaticResources:
    """Own resource inputs once; capture filenames are never qualification keys."""

    store: EvidenceStore = field(default_factory=EvidenceStore)
    reports: list[tuple[Path, dict]] = field(default_factory=list)
    inventory: list[dict] = field(default_factory=list)
    index: dict = field(default_factory=dict)

    def document(self, reference: str | dict, parent: Path) -> tuple[Path, dict]:
        if isinstance(reference, str):
            path = (parent / reference).resolve(strict=True)
            reference = dict(path=str(path), sha256=file_digest(path))
        return self.store.json(reference, parent)

    def include_report(self, path: Path, report: dict) -> None:
        if "reports" in report:
            rows = report["reports"]
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError("counter report inventory must contain object rows")
            for row in rows:
                self.include_report(path, row)
            return
        self.inventory.append(
            dict(
                path=str(path),
                sha256=self.store.hashes[path],
                resource_summary=report.get("resource_summary"),
            )
        )
        if isinstance(report.get("symbols"), list):
            self.reports.append((path, report))
        summaries = report.get("isa_summaries", [])
        if not isinstance(summaries, list):
            raise ValueError("ISA input summaries must be a list")
        # Read original hash-bound inputs, never flattened resource verdicts.
        for summary in summaries:
            if not isinstance(summary, dict):
                raise ValueError("ISA input summaries must be objects")
            isa_path, isa_report = self.document(
                dict(path=summary.get("path"), sha256=summary.get("sha256")), path.parent
            )
            self.reports.append((isa_path, isa_report))


def load_static_resource_manifest(path: Path | None) -> SelectedStaticResources:
    resources = SelectedStaticResources()
    if path is None:
        return resources
    manifest_path, data = resources.document(str(path.resolve()), path.parent)
    if isinstance(data.get("captures"), list):
        entries = data["captures"]
    else:
        entries = list(data.values())
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("tile resource entries must be objects")
        if isinstance(entry.get("resource_summary"), dict):
            resources.inventory.append(dict(source="unbound_inline_inventory", record=entry))
        for key in ("gpu_counter_report", "counter_report", "isa_summary", "resource_report"):
            reference = entry.get(key)
            if reference is None:
                continue
            report_path, report = resources.document(reference, manifest_path.parent)
            resources.include_report(report_path, report)
    resources.index = isa_index_from_reports(resources.reports)
    return resources


def load_static_counter_report(path: Path | None) -> SelectedStaticResources:
    resources = SelectedStaticResources()
    if path is not None:
        source, data = resources.document(str(path.resolve()), path.parent)
        reports = data.get("reports", [data])
        if not isinstance(reports, list) or any(not isinstance(row, dict) for row in reports):
            raise ValueError("counter report inventory must contain object rows")
        for report in reports:
            resources.include_report(source, report)
        resources.index = isa_index_from_reports(resources.reports)
    return resources


def selected_resource_status(
    capture: dict, resources: SelectedStaticResources, *, backend: str, target: str | None
) -> dict:
    association = lookup_isa_resources(resources.index, backend, target, capture)
    selected = association["isa_resource_reports"]
    fields = {
        "vgpr": "isa_vgpr_count",
        "sgpr": "isa_sgpr_count",
        "lds_bytes": "isa_lds_bytes",
        "scratch_bytes": "isa_scratch_bytes",
        "occupancy": "isa_occupancy",
    }
    summary = {}
    for destination, source in fields.items():
        values = [_number(report.get(source)) for report in selected]
        # One missing symbol cannot borrow another symbol's resource metadata.
        if values and all(value is not None and value >= 0 for value in values):
            summary[destination] = min(values) if destination == "occupancy" else max(values)
        else:
            summary[destination] = None
    return _resource_status(
        dict(
            resource_summary=summary,
            selected_isa=association,
            evidence_status=dict(
                isa_resource_status="selected_static_binding"
                if association["isa_association"]["qualified"]
                else "unqualified",
                profiler_counter_status="not_used_as_static_resource_proof",
                occupancy_scope="static_reported_value_not_measured_wave_occupancy",
            ),
        )
    )


def _resource_status(resource: dict) -> dict:
    summary = resource.get("resource_summary") or {}
    selected = resource.get("selected_isa") or {}
    association = selected.get("isa_association") or {}
    present = association.get("qualified") is True
    registers = all(
        _number(summary.get(key)) is not None and summary[key] > 0 for key in ("vgpr", "sgpr")
    )
    lds = _number(summary.get("lds_bytes")) is not None and summary["lds_bytes"] >= 0
    occupancy = _number(summary.get("occupancy")) is not None and summary["occupancy"] > 0
    return dict(
        resource_summary=summary,
        evidence_status=resource.get("evidence_status", {}),
        selected_isa=selected,
        resource_evidence_present=present,
        register_pressure_present=registers,
        lds_signal_present=lds,
        occupancy_present=occupancy,
        complete_for_promotion=present and registers and lds and occupancy,
    )
