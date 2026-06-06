#!/usr/bin/env python3
"""Report RNS8 local development dependencies."""

from __future__ import annotations

from check_dependencies_lib.accelerators import *
from check_dependencies_lib.cli import main
from check_dependencies_lib.config import *
from check_dependencies_lib.discovery import *
from check_dependencies_lib.human import print_human
from check_dependencies_lib.readiness import *
from check_dependencies_lib.report import build_report
from check_dependencies_lib.system import *


if __name__ == "__main__":
    raise SystemExit(main())
