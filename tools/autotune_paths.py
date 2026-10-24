"""Shared Python spelling of the runtime's platform cache-location contract."""
from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Mapping


def default_autotune_cache_path(environ: Mapping[str, str] | None = None,
                               platform: str | None = None) -> Path:
    env = os.environ if environ is None else environ
    platform = os.name if platform is None else platform
    override = env.get("RNS8_AUTOTUNE_CACHE_PATH")
    if override:
        return Path(override)
    suffix = Path("rns8-gemm") / "autotune.json"
    if platform == "nt":
        if env.get("LOCALAPPDATA"):
            return Path(env["LOCALAPPDATA"]) / suffix
        if env.get("USERPROFILE"):
            return Path(env["USERPROFILE"]) / "AppData" / "Local" / suffix
    else:
        if env.get("XDG_CACHE_HOME"):
            return Path(env["XDG_CACHE_HOME"]) / suffix
        if env.get("HOME"):
            return Path(env["HOME"]) / ".cache" / suffix
    return suffix
