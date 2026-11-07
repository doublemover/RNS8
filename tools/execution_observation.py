"""Typed native argv/environment observations, independent of caller labels."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from execution_environment_contract import relevant_environment_key

SCHEMA = "rns8_execution_environment_witness_v1"
FIELDS = frozenset(("schema", "state", "arguments", "before", "after"))


def _environment(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict) or len(value) > 256:
        raise ValueError("native environment requires a bounded object")
    result: list[tuple[str, str]] = []
    size = 0
    for key, item in value.items():
        if (
            type(key) is not str
            or not key
            or not relevant_environment_key(key)
            or type(item) is not str
        ):
            raise ValueError("native environment has an unknown or invalid key/value")
        key_size, item_size = len(key.encode("utf-8")), len(item.encode("utf-8"))
        if key_size > 256 or item_size > 4096:
            raise ValueError("native environment has an unknown or invalid key/value")
        size += key_size + item_size
        if size > 65536:
            raise ValueError("native environment exceeds its byte budget")
        result.append((key, item))
    return tuple(sorted(result))


def _arguments(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > 512:
        raise ValueError("native argv requires a bounded string array")
    result: list[str] = []
    size = 0
    for item in value:
        if type(item) is not str:
            raise ValueError("native argv requires a bounded string array")
        item_size = len(item.encode("utf-8"))
        size += item_size
        if item_size > 4096 or size > 65536:
            raise ValueError("native argv exceeds its byte budget")
        result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class ExecutionObservation:
    arguments: tuple[str, ...]
    environment_items: tuple[tuple[str, str], ...]

    @property
    def environment(self) -> dict[str, str]:
        return dict(self.environment_items)


def execution_observation(
    capture: Mapping[str, object], *, required: bool = False
) -> ExecutionObservation | None:
    value = capture.get("execution_environment_witness")
    if value is None:
        if required:
            raise ValueError("native execution environment witness is missing")
        return None
    if not isinstance(value, dict) or set(value) != FIELDS or value.get("schema") != SCHEMA:
        raise ValueError("versioned native execution environment witness required")
    if value["state"] == "not_requested":
        if required or any(value[key] is not None for key in ("arguments", "before", "after")):
            raise ValueError("native execution environment observation was not requested")
        return None
    if value["state"] != "stable":
        raise ValueError("native execution environment changed or did not complete")
    arguments = _arguments(value["arguments"])
    before, after = (
        _environment(value["before"]),
        _environment(value["after"]),
    )
    if before != after:
        raise ValueError("native execution environment contradicts its stable state")
    return ExecutionObservation(arguments, before)


def validate_execution_request(
    capture: Mapping[str, object], request: Mapping[str, object], *, required: bool = False
) -> ExecutionObservation | None:
    arguments = request.get("arguments")
    count = arguments.count("--record-execution-environment") if isinstance(arguments, list) else 0
    if count > 1:
        raise ValueError("native environment observation flag is ambiguous")
    observed = execution_observation(capture, required=required or count == 1)
    if observed is not None and count != 1:
        raise ValueError("native environment witness has no explicit observation request")
    if observed is not None and (
        list(observed.arguments) != request.get("arguments")
        or observed.environment != request.get("environment")
    ):
        raise ValueError("native argv/environment differs from its bound execution request")
    return observed


def execution_observation_blockers(capture: Mapping[str, object]) -> list[str]:
    request = capture.get("request_identity")
    try:
        if isinstance(request, dict):
            validate_execution_request(capture, request)
        elif execution_observation(capture) is not None:
            return ["native_execution_request_missing"]
    except (ValueError, TypeError):
        return ["native_execution_observation_unqualified"]
    return []
