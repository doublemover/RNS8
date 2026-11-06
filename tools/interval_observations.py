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
