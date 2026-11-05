"""Exact-type scalar predicates shared by offline evidence consumers."""

from __future__ import annotations

import math
from typing import TypeGuard


def demand(value: object, why: str) -> None:
    if not value:
        raise ValueError(why)


def integer(value: object, low: int, high: int) -> TypeGuard[int]:
    return type(value) is int and low <= value <= high


def positive(value: object) -> TypeGuard[int | float]:
    # Exact types intentionally reject bool and numeric subclasses. Huge JSON
    # integers cannot represent supported finite timing observations either.
    if type(value) is int or type(value) is float:
        try:
            return math.isfinite(value) and value > 0
        except OverflowError:
            return False
    return False
