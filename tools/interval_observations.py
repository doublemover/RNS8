"""Disjoint interval accounting within one explicitly bound clock domain."""

from __future__ import annotations


def interval_summary(intervals):
    if not isinstance(intervals, list) or len(intervals) > 100000:
        raise ValueError("bounded same-clock interval list required")
    if any(
        not isinstance(pair, (tuple, list))
        or len(pair) != 2
        or any(type(value) is not int or value < 0 for value in pair)
        or pair[1] < pair[0]
        for pair in intervals
    ):
        raise ValueError("ordered nonnegative native interval endpoints required")
    ordered = sorted(intervals)
    service = sum(end - start for start, end in ordered)
    union, start, end = 0, None, None
    for first, last in ordered:
        if start is None:
            start, end = first, last
        elif first <= end:
            end = max(end, last)
        else:
            union += end - start
            start, end = first, last
    if start is not None:
        union += end - start
    span = max(last for _, last in ordered) - ordered[0][0] if ordered else 0
    return dict(
        count=len(ordered),
        service_ns=service,
        interval_union_ns=union,
        span_ns=span,
        uncovered_span_ns=span - union,
        overlap_service_ns=service - union,
    )


def weighted_peak(intervals, *, include_endpoints=False):
    """Peak simultaneous weight in one clock; endpoint overlap is explicit."""
    if (
        type(include_endpoints) is not bool
        or not isinstance(intervals, list)
        or len(intervals) > 100000
    ):
        raise ValueError("bounded weighted intervals and explicit endpoint policy required")
    events = []
    for row in intervals:
        if (
            not isinstance(row, (tuple, list))
            or len(row) != 3
            or any(type(value) is not int or value < 0 for value in row)
            or row[1] < row[0]
        ):
            raise ValueError("ordered nonnegative weighted interval required")
        first, last, weight = row
        if first == last and not include_endpoints:
            continue
        events.append((first, 0 if include_endpoints else 1, weight))
        events.append((last, 1 if include_endpoints else 0, -weight))
    live = peak = 0
    for _, _, delta in sorted(events):
        live += delta
        peak = max(peak, live)
    return peak
