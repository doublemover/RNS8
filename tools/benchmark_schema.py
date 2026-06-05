#!/usr/bin/env python3
"""Compatibility CLI and import wrapper for benchmark schema validation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_schema_package() -> ModuleType:
    package_dir = Path(__file__).with_suffix("")
    package_init = package_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "_rns8_benchmark_schema",
        package_init,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load benchmark schema package from {package_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_schema = _load_schema_package()

BenchmarkSchemaError = _schema.BenchmarkSchemaError
load_capture = _schema.load_capture
main = _schema.main
schema_version = _schema.schema_version
validate_capture = _schema.validate_capture
validate_capture_file = _schema.validate_capture_file
validation_errors = _schema.validation_errors

__all__ = [
    "BenchmarkSchemaError",
    "load_capture",
    "main",
    "schema_version",
    "validate_capture",
    "validate_capture_file",
    "validation_errors",
]


if __name__ == "__main__":
    raise SystemExit(main())
