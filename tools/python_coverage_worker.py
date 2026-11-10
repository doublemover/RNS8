"""Native Coverage.py package/data inspector run by the selected isolated Python."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import sys

from host_instrumentation_session import check_bookends, reference
from python_coverage_contract import raw_data


def package_identity():
    import coverage

    prefix = Path(sys.prefix).resolve()
    package = Path(coverage.__file__).resolve()
    if not package.is_relative_to(prefix) or sys.prefix == sys.base_prefix:
        raise ValueError("coverage requires an isolated prefix without inherited system package")
    distribution = metadata.distribution("coverage")
    files = list(distribution.files or [])
    if not files or len(files) > 8192:
        raise ValueError("bounded installed coverage distribution required")
    refs = []
    for entry in files:
        if Path(entry).suffix == ".pyc":
            continue
        path = Path(distribution.locate_file(entry)).resolve(strict=True)
        if not path.is_relative_to(prefix):
            raise ValueError("installed coverage file escaped isolated prefix")
        refs.append(reference(path))
    if sum(row["bytes"] for row in refs) > 512 * 1024**2:
        raise ValueError("installed coverage package exceeds byte budget")
    return dict(
        schema="rns8_python_coverage_package_identity_v1",
        version=coverage.__version__,
        interpreter=reference(Path(sys.executable)),
        base_interpreter=reference(Path(getattr(sys, "_base_executable", sys.executable))),
        python_version=list(sys.version_info[:3]),
        assertions_enabled=sys.flags.optimize == 0,
        package=reference(package),
        prefix_configuration=reference(prefix / "pyvenv.cfg"),
        installed_files=refs,
    )


def inspect_data(directory: Path, marker_source: Path | None):
    import coverage

    refs = raw_data(directory)
    for row in refs:
        data = coverage.CoverageData(basename=row["path"])
        data.read()
        measured = data.measured_files()
        if not data.has_arcs() or not measured:
            raise ValueError("each process data file must contain real branch measurements")
        row["has_arcs"] = True
        row["measured_file_count"] = len(measured)
        row["marker_lines_executed"] = []
        if marker_source:
            matches = [name for name in measured if Path(name).resolve() == marker_source.resolve()]
            if len(matches) > 1:
                raise ValueError("duplicate native process marker source")
            if matches:
                row["marker_lines_executed"] = data.lines(matches[0]) or []
        check_bookends(row)
    return dict(schema="rns8_python_coverage_data_validation_v1", files=refs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("identity", "data"))
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--marker-source", type=Path)
    args = parser.parse_args(argv)
    try:
        if sys.flags.optimize:
            raise ValueError("Python assertions are stripped")
        if args.action == "identity":
            result = package_identity()
        elif args.directory is not None:
            result = inspect_data(args.directory, args.marker_source)
        else:
            raise ValueError("native process data directory required")
        print(json.dumps(result))
        return 0
    except (OSError, ValueError, TypeError, ImportError) as error:
        print("coverage inspector: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
