"""Exact-type scalar predicates shared by offline evidence consumers."""

from __future__ import annotations

import math
from typing import Container, TypeGuard


def demand(value: object, why: str) -> None:
    if not value:
        raise ValueError(why)


def integer(value: object, low: int, high: int) -> TypeGuard[int]:
    return type(value) is int and low <= value <= high


def finite_number(value: object) -> TypeGuard[int | float]:
    # Exact types intentionally reject bool and numeric subclasses. Huge JSON
    # integers cannot represent supported finite timing observations either.
    if type(value) is int or type(value) is float:
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
    return False


def positive(value: object) -> TypeGuard[int | float]:
    return finite_number(value) and value > 0


def text_choice(value: object, choices: Container[str]) -> TypeGuard[str]:
    """Reject wrong-type JSON values before a hashed enum lookup."""
    return type(value) is str and value in choices
