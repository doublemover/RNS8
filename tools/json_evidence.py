"""Strict JSON primitives for content-bound offline evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import NoReturn, cast

type JsonValue = bool | int | float | str | None | list[JsonValue] | dict[str, JsonValue]


def parse_evidence_json(text: str) -> JsonValue:
    def pairs(items: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate evidence JSON key: " + key)
            result[key] = value
        return result

    def invalid_constant(value: str) -> NoReturn:
        raise ValueError("nonfinite evidence JSON number: " + value)

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValueError("nonfinite evidence JSON number: " + token)
        return value

    # json.loads constructs only JSON primitives/containers. The hooks reject
    # duplicate keys and nonfinite numbers before this recursive-type cast.
    return cast(
        JsonValue,
        json.loads(
            text, object_pairs_hook=pairs, parse_constant=invalid_constant, parse_float=finite_float
        ),
    )


def load_evidence_json(path: Path) -> JsonValue:
    return parse_evidence_json(Path(path).read_text(encoding="utf-8"))


def same_evidence_value(left: JsonValue, right: JsonValue) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            same_evidence_value(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            same_evidence_value(a, b) for a, b in zip(left, right)
        )
    return left == right
