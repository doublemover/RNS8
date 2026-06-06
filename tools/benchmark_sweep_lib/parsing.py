from __future__ import annotations

from .config import INPUT_PROFILES, SweepCase

def parse_int(text: str, label: str) -> int:
    try:
        value = int(text, 10)
    except ValueError as exc:
        raise SystemExit(f"{label} must be an integer, got {text!r}") from exc
    if value <= 0:
        raise SystemExit(f"{label} must be positive, got {value}")
    return value


def parse_case(value: str, *, adaptive: bool = False, promotable: bool = True) -> SweepCase:
    name = ""
    body = value
    if ":" in value:
        name, body = value.split(":", 1)
    parts = [part.strip() for part in body.replace("x", ",").split(",") if part.strip()]
    min_expected = 5 if adaptive else 3
    max_expected = 6 if adaptive else 3
    if len(parts) < min_expected or len(parts) > max_expected:
        shape = "NAME:M,N,K,TILE_M,TILE_N[,INPUT_PROFILE]" if adaptive else "NAME:M,N,K"
        raise SystemExit(f"case must use {shape}, got {value!r}")
    m = parse_int(parts[0], "m")
    n = parse_int(parts[1], "n")
    k = parse_int(parts[2], "k")
    tile_m = parse_int(parts[3], "tile_m") if adaptive else 128
    tile_n = parse_int(parts[4], "tile_n") if adaptive else 128
    input_profile = parts[5] if adaptive and len(parts) == 6 else "uniform-small"
    if input_profile not in INPUT_PROFILES:
        raise SystemExit(f"adaptive input profile must be one of {sorted(INPUT_PROFILES)}, got {input_profile!r}")
    if not name:
        prefix = "adaptive" if adaptive else "shape"
        name = f"{prefix}-{m}x{n}x{k}"
    return SweepCase(
        name=name,
        m=m,
        n=n,
        k=k,
        tile_m=tile_m,
        tile_n=tile_n,
        bound_mode="per-tile" if adaptive else "global",
        input_profile=input_profile,
        require_adaptive=adaptive,
        promotable=promotable,
    )


