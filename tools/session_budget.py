"""Operator forecasts and quoted linear costs for an already available session."""

from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR, localcontext
import hashlib
import math
from pathlib import Path
import re

from json_evidence import parse_evidence_json, same_evidence_value

SCHEMA = "rns8_session_budget_v1"
FORECAST_SCHEMA = "rns8_session_stage_forecast_v1"
MAX_BYTES = 1024**2


def decimal_value(value, name, *, positive=False):
    if type(value) is not str or not re.fullmatch(r"[0-9]{1,15}(?:\.[0-9]{1,9})?", value):
        raise ValueError("explicit nonnegative decimal string required: " + name)
    result = Decimal(value)
    if positive and not result:
        raise ValueError("positive decimal required: " + name)
    return result


def decimal_text(value):
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def load_forecast(path, commands):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if not 1 <= len(raw) <= MAX_BYTES:
        raise ValueError("stage forecast exceeds its byte allowance")
    value = parse_evidence_json(raw.decode("utf-8"))
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "stages"}
        or value["schema"] != FORECAST_SCHEMA
    ):
        raise ValueError("maintained operator stage forecast required")
    stages = value["stages"]
    if not isinstance(stages, list) or len(stages) > 4096 or commands and not stages:
        raise ValueError("bounded nonempty stage forecast required")
    names = [item["name"] for item in commands]
    if len(set(names)) != len(names):
        raise ValueError("forecast cannot bind duplicate command names")
    indexed = {}
    for row in stages:
        if not isinstance(row, dict) or set(row) != {
            "name",
            "lower_seconds",
            "expected_seconds",
            "upper_seconds",
        }:
            raise ValueError("exact typed stage forecast fields required")
        name = row["name"]
        if (
            type(name) is not str
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,256}", name)
            or name in indexed
        ):
            raise ValueError("invalid or repeated forecast stage name")
        lower, expected, upper = (
            decimal_value(row[key], key)
            for key in ("lower_seconds", "expected_seconds", "upper_seconds")
        )
        if not 0 <= lower <= expected <= upper <= 604800 or not upper:
            raise ValueError("ordered positive stage upper forecast of at most one week required")
        indexed[name] = dict(
            name=name,
            lower_seconds=decimal_text(lower),
            expected_seconds=decimal_text(expected),
            upper_seconds=decimal_text(upper),
        )
    if set(indexed) != set(names):
        missing, extra = sorted(set(names) - indexed.keys()), sorted(indexed.keys() - set(names))
        raise ValueError(
            "forecast command inventory differs; missing=" + repr(missing) + " extra=" + repr(extra)
        )
    return dict(
        input=dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)),
        stages=[indexed[name] for name in names],
        basis="operator_supplied_unqualified_forecast_not_a_runtime_measurement",
        scheduling="existing_sequential_command_dependencies",
    )


def plan_budget(
    commands,
    *,
    total_seconds,
    reserve_seconds,
    forecast_path=None,
    price_per_hour=None,
    maximum_cost=None,
    currency=None,
    price_scope="allocation",
    selected_devices=0,
    already_incurred_cost=None,
):
    if (
        any(
            type(value) not in (int, float) or not math.isfinite(value) or not value > 0
            for value in (total_seconds, reserve_seconds)
        )
        or total_seconds <= reserve_seconds
    ):
        raise ValueError("finite session time exceeding closeout reserve required")
    if type(selected_devices) is not int or not 0 <= selected_devices <= 65536:
        raise ValueError("explicit selected-device count required for budget")
    if type(price_scope) is not str or price_scope not in {"allocation", "per-selected-device"}:
        raise ValueError("explicit supported quoted price scope required")
    quoted = (
        any(
            value is not None
            for value in (price_per_hour, maximum_cost, currency, already_incurred_cost)
        )
        or price_scope != "allocation"
    )
    forecast = None if forecast_path is None else load_forecast(forecast_path, commands)
    quote, blockers = None, []
    effective = Decimal(str(total_seconds))
    with localcontext() as context:
        context.prec = 80
        if quoted:
            if (
                price_per_hour is None
                or maximum_cost is None
                or type(currency) is not str
                or not re.fullmatch(r"[A-Z]{3}", currency)
            ):
                raise ValueError(
                    "price, maximum cost and three-letter currency must be supplied together"
                )
            rate = decimal_value(price_per_hour, "price per hour", positive=True)
            ceiling = decimal_value(maximum_cost, "maximum cost", positive=True)
            incurred = decimal_value(
                "0" if already_incurred_cost is None else already_incurred_cost,
                "already incurred cost",
            )
            if incurred > ceiling:
                raise ValueError("already incurred cost exceeds maximum cost")
            if price_scope == "per-selected-device" and not selected_devices:
                raise ValueError("per-device quote requires an explicit nonempty device selection")
            effective_rate = rate * (
                selected_devices if price_scope == "per-selected-device" else 1
            )
            affordable = ((ceiling - incurred) * 3600 / effective_rate).to_integral_value(
                rounding=ROUND_FLOOR
            )
            effective = min(effective, affordable)
            quote = dict(
                currency=currency,
                price_per_hour=decimal_text(rate),
                price_scope=price_scope,
                selected_devices=selected_devices,
                effective_hourly_rate=decimal_text(effective_rate),
                maximum_cost=decimal_text(ceiling),
                already_incurred_cost=decimal_text(incurred),
                affordable_whole_seconds=decimal_text(affordable),
                rate_basis="operator_supplied_continuous_linear_quote",
                scope="owned_execution_time_not_provider_billing_or_idle_reservation_time",
            )
            if forecast is None:
                blockers.append("quoted-cost execution requires a complete operator stage forecast")
        if effective <= Decimal(str(reserve_seconds)):
            blockers.append("time/cost allowance does not cover the closeout reserve")
        totals = None
        if forecast is not None:
            totals = {
                key: sum((Decimal(row[key]) for row in forecast["stages"]), Decimal(0))
                + Decimal(str(reserve_seconds))
                for key in ("lower_seconds", "expected_seconds", "upper_seconds")
            }
            if totals["upper_seconds"] > effective:
                blockers.append(
                    "all selected stages plus closeout exceed the effective upper time/cost forecast"
                )
            if quote is not None:
                quote["forecast_upper_cost"] = decimal_text(
                    Decimal(quote["already_incurred_cost"])
                    + totals["upper_seconds"] * Decimal(quote["effective_hourly_rate"]) / 3600
                )
    return dict(
        schema=SCHEMA,
        requested_total_seconds=float(total_seconds),
        effective_total_seconds=float(effective),
        reserve_seconds=float(reserve_seconds),
        forecast=forecast,
        total_forecast_seconds=None
        if totals is None
        else {key: decimal_text(value) for key, value in totals.items()},
        quote=quote,
        blockers=blockers,
        state="blocked" if blockers else "planned_unqualified",
        hardware_provisioned=False,
        reservation_extended=False,
        automatic_shutdown=False,
        production_latency=False,
        promotion_allowed=False,
    )


def replay_budget(value, commands):
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("maintained session budget receipt required")
    if not {"requested_total_seconds", "reserve_seconds", "quote", "forecast"} <= value.keys():
        raise ValueError("incomplete session budget fields")
    quote, forecast = value.get("quote"), value.get("forecast")
    if quote is not None and (
        not isinstance(quote, dict)
        or not {
            "price_per_hour",
            "maximum_cost",
            "currency",
            "price_scope",
            "selected_devices",
            "already_incurred_cost",
        }
        <= quote.keys()
    ):
        raise ValueError("incomplete typed session quote fields")
    if forecast is not None and (
        not isinstance(forecast, dict)
        or not isinstance(forecast.get("input"), dict)
        or "path" not in forecast["input"]
    ):
        raise ValueError("incomplete typed forecast input identity")
    rebuilt = plan_budget(
        commands,
        total_seconds=value["requested_total_seconds"],
        reserve_seconds=value["reserve_seconds"],
        forecast_path=None if forecast is None else forecast["input"]["path"],
        **(
            {}
            if quote is None
            else dict(
                price_per_hour=quote["price_per_hour"],
                maximum_cost=quote["maximum_cost"],
                currency=quote["currency"],
                price_scope=quote["price_scope"],
                selected_devices=quote["selected_devices"],
                already_incurred_cost=quote["already_incurred_cost"],
            )
        ),
    )
    if not same_evidence_value(value, rebuilt):
        raise ValueError("session budget source/forecast/quote changed")
    return rebuilt


def stage_fits(forecast, remaining_seconds):
    if type(remaining_seconds) not in (int, float) or not math.isfinite(remaining_seconds):
        raise ValueError("finite actual remaining session time required")
    if forecast is not None and (not isinstance(forecast, dict) or "upper_seconds" not in forecast):
        raise ValueError("typed stage upper forecast required")
    return remaining_seconds > 0 and (
        forecast is None
        or Decimal(str(remaining_seconds))
        >= decimal_value(forecast["upper_seconds"], "stage upper", positive=True)
    )
