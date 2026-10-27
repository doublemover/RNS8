"""Non-overlapping event accounting; reported raw events remain untouched.

Relationships are declared by the producer's collection contract, never inferred
from equal numeric values. Unknown detail stays diagnostic. A category sum is
GPU service time, not elapsed wall time when streams overlap.
"""
from __future__ import annotations
import math
from .io import event_medians

# Alternative complete child sets. More detailed sets precede aggregate aliases.
CHILDREN = {
    "pack": [("pack_h2d", "pack_kernel"), ("finite_pack_h2d", "finite_pack_kernel"),
             ("vector_alu_pack_a_h2d", "vector_alu_pack_b_h2d")],
    "rns_gemm": [("hipblaslt_pack_transpose_centered", "hipblaslt_int8_i32_matmul", "hipblaslt_i32_to_residue_reduce"),
                 ("hipblaslt_int8_i32_matmul", "hipblaslt_i32_to_residue_reduce"),
                 ("rns_gemm_kernel_group", "direct_hip_zero_output_tile_memset"),
                 ("rns_gemm_prepacked_b_kernel_group", "direct_hip_zero_output_tile_memset"),
                 ("rns_gemm_kernel_group",), ("rns_gemm_prepacked_b_kernel_group",)],
    "crt_export": [("crt_export_status_memset", "crt_export_kernel", "crt_export_status_d2h", "crt_export_d2h"),
                   ("exact_wide_export_status_memset", "exact_wide_export_kernel", "exact_wide_export_status_d2h", "exact_wide_export_d2h"),
                   ("exact_wide_export_kernel", "exact_wide_export_d2h"),
                   ("finite_export_kernel", "finite_export_d2h"),
                   ("vector_alu_status_d2h", "vector_alu_output_d2h")],
}
CATEGORIES = ("status", "transfer", "pack", "compute", "export", "other")


def category(name):
    if name.endswith("_unattributed"): return "other"
    if "status" in name: return "status"
    if any(part in name for part in ("h2d", "d2h", "memcpy")): return "transfer"
    if "pack" in name or "transpose" in name: return "pack"
    if any(part in name for part in ("gemm", "matmul", "kernel_group")): return "compute"
    if any(part in name for part in ("export", "reduce", "centered")): return "export"
    return "other"


def upper_median(values):
    return sorted(values)[math.ceil((len(values)-1)*0.5)]


def disjoint_event_cover(capture):
    raw = capture.get("gpu_event_timings_us") or {}
    medians = event_medians(capture)
    relationships = {key: list(value) for key, value in CHILDREN.items()}
    # Prefix detail and backend-wide detail describe the same instructions.
    # Retain the backend-wide totals, never add the prefix subtotals again.
    for parent in ("rns_gemm_kernel_group", "rns_gemm_prepacked_b_kernel_group"):
        relationships[parent] = [
            ("ck_pack_a_kernel", "ck_pack_b_kernel", "ck_wmma_cshuffle_matmul", "ck_copy_centered_kernel", "ck_add_centered_kernel"),
            ("rocwmma_pack_a_kernel", "rocwmma_pack_b_kernel", "rocwmma_matmul_kernel"),
            ("rocwmma_pack_a_prepacked_b_kernel", "rocwmma_matmul_prepacked_b_kernel")]
    selected, residuals, notes = {}, {}, []
    def choose(name):
        values = raw.get(name)
        candidates = relationships.get(name, ())
        for children in candidates:
            if not all(child in medians for child in children): continue
            child_raw = [raw.get(child) for child in children]
            if (not isinstance(values, list) or not values or
                any(not isinstance(v, list) or len(v) != len(values) for v in child_raw)):
                continue # no unverifiable sum-of-marginal-medians subtraction
            if any(not isinstance(x, (int,float)) or not math.isfinite(x) or x < 0
                   for v in [values, *child_raw] for x in v): continue
            sums = [sum(v[i] for v in child_raw) for i in range(len(values))]
            if any(s > p + max(1e-6, abs(p)*1e-5) for s,p in zip(sums,values)):
                notes.append(name+":child_sum_exceeds_parent_fallback")
                continue
            remainder = [max(0.0,p-s) for p,s in zip(values,sums)]
            for child in children: choose(child)
            if any(value > 1e-6 for value in remainder): residuals[name+"_unattributed"] = remainder
            return
        if name in medians: selected[name] = medians[name]
    roots = [name for name in ("pack", "rns_gemm", "crt_export") if name in medians]
    for root in roots: choose(root)
    if not roots:
        notes.append("no_declared_operation_roots_unknown_coverage")
    selected.update({name:upper_median(values) for name,values in residuals.items()})
    # Aggregate corresponding disjoint per-repeat children before summary.
    category_samples = {}
    all_samples = {name:raw.get(name) for name in selected if name not in residuals}
    all_samples.update(residuals)
    counts = {len(v) for v in all_samples.values() if isinstance(v,list) and v}
    aligned = bool(all_samples) and len(counts)==1 and all(isinstance(v,list) and v for v in all_samples.values())
    categories = {name:0.0 for name in CATEGORIES}
    if aligned:
        count = next(iter(counts))
        for name in CATEGORIES:
            values = [sum(v[i] for label,v in all_samples.items() if category(label)==name) for i in range(count)]
            category_samples[name] = values
            categories[name] = upper_median(values)
    else:
        for name,value in selected.items(): categories[category(name)] += value
        notes.append("marginal_median_diagnostics_only_no_aligned_raw_cover")
    return dict(selected_events=selected, category_medians_us=categories, category_samples_us=category_samples,
                roots=roots, omitted_diagnostic_events=sorted(set(medians)-set(selected)), notes=notes,
                coverage="declared_operation_roots" if roots else "unknown",
                accounting_unit="disjoint_gpu_service_time_not_wall_time_fraction",
                streams_may_overlap=bool((capture.get("streaming_overlap") or {}).get("requested")))
