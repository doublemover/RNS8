#!/usr/bin/env python3
"""Build an ignored RNS8 benchmark evidence database from schema v4 captures."""

from __future__ import annotations

from evidence_database_lib.cli import main, parse_args
from evidence_database_lib.io import (
    discover_capture_paths,
    discover_isa_report_paths,
    event_medians,
    load_review_index,
    load_scenario_index,
    load_validated_captures,
    median_phase,
    normalized_capture_path,
)
from evidence_database_lib.isa import (
    aggregate_isa_resources,
    load_isa_index,
    lookup_isa_resources,
    normalized_backend,
    normalized_target,
    summarize_isa_report,
)
from evidence_database_lib.outputs import write_markdown, write_outputs
from evidence_database_lib.rows import build_database, build_row
from evidence_database_lib.work_model import (
    build_roofline_priority,
    classify_bottleneck,
    estimate_work,
    roofline_target_for_row,
)


if __name__ == "__main__":
    raise SystemExit(main())
