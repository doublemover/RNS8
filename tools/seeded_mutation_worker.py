"""Execute one isolated production-module overlay through native unittest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--variant", choices=("baseline", "mutant"), required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--line", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--test-module", action="append", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        if sys.flags.optimize or sys.gettrace() is not None:
            raise ValueError(
                "unoptimized isolated mutation interpreter without an existing tracer required"
            )
        dependencies = json.loads((root / "metadata/dependencies.json").read_text(encoding="utf-8"))
        selected = next(
            row["selected"]["version"]
            for row in dependencies["components"]
            if row["name"] == "python"
        )
        if selected != ".".join(map(str, sys.version_info[:3])):
            raise ValueError("selected Python dependency version required for mutation execution")
        if not re.fullmatch(r"[a-z][a-z0-9_]+", args.module) or args.module in sys.modules:
            raise ValueError("fresh first-party production module required")
        overlay = args.overlay.resolve(strict=True)
        if overlay.name != args.module + ".py" or not overlay.is_relative_to(root / "temp"):
            raise ValueError("owned isolated production-module overlay required")
        if hashlib.sha256(overlay.read_bytes()).hexdigest() != args.sha256:
            raise ValueError("production overlay changed before import")
        spec = importlib.util.spec_from_file_location(args.module, overlay)
        if spec is None or spec.loader is None:
            raise ValueError("production overlay loader unavailable")
        module = importlib.util.module_from_spec(spec)
        sys.modules[args.module] = module
        spec.loader.exec_module(module)
        observed = set()
        active = None

        def observer(test_id):
            nonlocal active
            active = test_id

        def trace(frame, event, arg):
            if frame.f_code.co_filename != str(overlay):
                return None
            if event == "line" and frame.f_lineno == args.line and active:
                observed.add(active)
            return trace

        import python_test_inventory

        command = ["--output", str(args.output)]
        for name in args.test_module:
            command.extend(["--module", name])
        sys.settrace(trace)
        try:
            status = python_test_inventory.main(command, observer=observer)
        finally:
            sys.settrace(None)
        if hashlib.sha256(overlay.read_bytes()).hexdigest() != args.sha256:
            raise ValueError("production overlay changed during native tests")
        value = dict(
            schema="rns8_production_mutation_observation_v1",
            process_id=os.getpid(),
            seed=args.seed,
            variant=args.variant,
            overlay=str(overlay),
            overlay_sha256=args.sha256,
            source_line=args.line,
            observed_test_ids=sorted(observed),
            scope="actual_python_production_module_main_thread_not_native_or_gpu_code",
            python_version=list(sys.version_info[:3]),
        )
        output = args.observation.resolve()
        if not output.is_relative_to(root / "temp"):
            raise ValueError("owned mutation observation output required")
        with output.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
        return status
    except (OSError, ValueError, TypeError, ImportError) as error:
        print("production mutation worker: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
