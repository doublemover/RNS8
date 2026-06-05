#!/usr/bin/env python3
"""Self-test for the checked-in metadata registry."""

from __future__ import annotations

import metadata_registry


def main() -> int:
    registry = metadata_registry.load_registry()
    metadata_registry.validate_registry(registry)
    errors = metadata_registry.check_generated(
        metadata_registry.PYTHON_CONSTANTS_PATH,
        metadata_registry.render_python_constants(registry),
    )
    errors.extend(
        metadata_registry.check_generated(
            metadata_registry.CPP_HEADER_PATH,
            metadata_registry.render_cpp_header(registry),
        )
    )
    if errors:
        raise SystemExit("\n".join(errors))
    print("metadata registry self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
