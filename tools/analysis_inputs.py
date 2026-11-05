"""Faithful configured host/HIP compilation contexts; never execution evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re

from benchmark_request import file_digest
from compiled_binary_identity import canonical_digest
from json_evidence import load_evidence_json
from quality_inventory import ROOT, inventory


@dataclass(frozen=True)
class Compilation:
    identity: str
    kind: str
    source: Path
    directory: Path
    native_record_json: str
    environment: tuple[tuple[str, str], ...]
    generated_inputs: tuple[Path, ...]
    targets: tuple[str, ...]
    registered_targets: tuple[str, ...]
    generated_source: bool

    def native_record(self) -> dict:
        # This is a fresh copy of a record already admitted by the strict reader.
        return json.loads(self.native_record_json)


@dataclass(frozen=True)
class AnalysisInputs:
    build: Path
    configuration: str
    contexts: tuple[Compilation, ...]
    excluded_records: tuple[tuple[str, str], ...]
    artifact_hashes: tuple[tuple[str, str], ...]
    compiler: Path


def cache_values(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        left, value = line.split("=", 1)
        if ":" not in left:
            raise ValueError("malformed CMake cache record")
        key, _ = left.split(":", 1)
        if key in result:
            raise ValueError("duplicate CMake cache identity: " + key)
        result[key] = value
    return result


def _absolute(value: object) -> Path:
    if type(value) is not str or not value or "$<" in value:
        raise ValueError("analysis input has an absent/unresolved configured path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("analysis compilation paths must be explicit absolute paths")
    return path.resolve()


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(type(item) is not str for item in value):
        raise ValueError("analysis input requires an explicit string array")
    return tuple(value)


def load_inputs(build: Path, *, root: Path = ROOT) -> AnalysisInputs:
    build, root = build.resolve(strict=True), root.resolve(strict=True)
    cache_path = build / "CMakeCache.txt"
    cache = cache_values(cache_path)
    if _absolute(cache.get("CMAKE_HOME_DIRECTORY")) != root:
        raise ValueError("analysis build belongs to another source tree")
    if cache.get("CMAKE_GENERATOR") not in {"Ninja", "Unix Makefiles"}:
        raise ValueError("analysis requires a supported single-configuration compilation database generator")
    if cache.get("CMAKE_UNITY_BUILD", "OFF").upper() not in {"OFF", "NO", "FALSE", "0", ""}:
        raise ValueError("unity compilation cannot claim individual source coverage")
    if cache.get("CMAKE_EXPORT_COMPILE_COMMANDS", "OFF").upper() not in {"ON", "YES", "TRUE", "1"}:
        raise ValueError("analysis build did not enable exported host commands")
    configuration = cache.get("CMAKE_BUILD_TYPE", "")
    if re.fullmatch(r"[A-Za-z0-9_-]+", configuration) is None:
        raise ValueError("analysis requires one explicit named build configuration")
    database = build / "compile_commands.json"
    target_path = build / ("rns8-analysis-targets-" + configuration + ".json")
    hip_path = build / ("rns8-hip-compile-commands-" + configuration + ".json")
    target_document, hip_document = load_evidence_json(target_path), load_evidence_json(hip_path)
    if (not isinstance(target_document, dict)
            or target_document.get("schema") != "rns8_first_party_analysis_targets_v1"
            or not isinstance(hip_document, dict)
            or hip_document.get("schema") != "rns8_custom_hip_compile_commands_v1"):
        raise ValueError("build lacks maintained configured first-party/custom HIP inventories")
    tracked = {(root / row.path).resolve(): row for row in inventory(root) if row.role == "cpp"}
    owners: dict[Path, set[str]] = defaultdict(set)
    registered = target_document.get("targets")
    if not isinstance(registered, list) or not registered:
        raise ValueError("configured first-party target inventory is empty")
    names = set()
    for target in registered:
        if (not isinstance(target, dict) or type(target.get("name")) is not str
                or target["name"] in names or target.get("configuration") != configuration):
            raise ValueError("configured target identity is duplicated or contradictory")
        names.add(target["name"])
        for source in _strings(target.get("sources")):
            path = _absolute(source)
            if path.suffix.lower() not in {".cpp", ".c", ".cc", ".cxx"}:
                continue  # Headers/fragments and linked external objects are not invented TUs.
            if path not in tracked and not path.is_relative_to(build):
                raise ValueError("registered first-party source is outside tracked/generated inventory: " + source)
            owners[path].add(target["name"])
    records = load_evidence_json(database)
    if not isinstance(records, list) or not records:
        raise ValueError("host compilation database is empty")
    contexts, excluded, seen, actual = [], [], set(), Counter()

    def add(record, *, kind, environment=(), generated=(), targets=(), registered_targets=()):
        if not isinstance(record, dict):
            raise ValueError("compilation database contains a non-object record")
        source, directory = _absolute(record.get("file")), _absolute(record.get("directory"))
        arguments, command = record.get("arguments"), record.get("command")
        if arguments is not None:
            argv = _strings(arguments)
            if not argv or not argv[0] or command is not None:
                raise ValueError("compilation context needs one unambiguous native argv/command representation")
        elif type(command) is not str or not command:
            raise ValueError("compilation context lacks its actual native compiler command")
        value = dict(record=record, kind=kind, environment=environment,
                     generated_inputs=[str(path) for path in generated], targets=targets)
        key = canonical_digest(value)
        if key in seen:
            raise ValueError("duplicate equivalent compilation context hides the coverage denominator")
        seen.add(key)
        contexts.append(Compilation(key, kind, source, directory,
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False),
            tuple(environment), tuple(generated), tuple(targets), tuple(registered_targets),
            source.is_relative_to(build)))

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("host compilation database contains a non-object record")
        source = _absolute(record.get("file"))
        if source not in owners:
            if source in tracked:
                raise ValueError("host database has an unregistered first-party translation unit")
            excluded.append((str(source), "external_or_unregistered_not_first_party_coverage"))
            continue
        actual[source] += 1
        add(record, kind="host", registered_targets=sorted(owners[source]),
            generated=(build / "generated" / configuration / "rns8_build_manifest.hpp",))
    expected = {source: len(targets) for source, targets in owners.items()}
    if dict(actual) != expected:
        raise ValueError("exported host contexts do not cover every registered target/source pair")
    hip_rows = hip_document.get("commands")
    if not isinstance(hip_rows, list):
        raise ValueError("custom HIP command inventory is malformed")
    hip_enabled = cache.get("RNS8_ENABLE_HIP", "OFF").upper() in {"ON", "YES", "TRUE", "1"}
    if bool(hip_rows) != hip_enabled:
        raise ValueError("custom HIP command inventory contradicts the configured backend state")
    hip_compilers = set()
    for row in hip_rows:
        if not isinstance(row, dict) or row.get("configuration") != configuration:
            raise ValueError("custom HIP command configuration differs from selected build")
        source = _absolute(row.get("file"))
        if source not in tracked and not source.is_relative_to(build):
            raise ValueError("custom HIP source is outside first-party/generated inventory")
        argv = _strings(row.get("arguments"))
        if not argv:
            raise ValueError("custom HIP compiler argv is empty")
        hip_compilers.add(_absolute(argv[0]))
        if any(re.match(r"^(?:-D|/D)(?:__HIP_DEVICE_COMPILE__|__AMDGCN__|__gfx[0-9a-f]+__)(?:=|$)", argument) for argument in argv):
            raise ValueError("configured HIP analysis must not fabricate compiler device/target macros")
        targets = _strings(row.get("targets"))
        if (not targets or len(set(targets)) != len(targets)
                or targets != tuple(cache.get("RNS8_AMDGPU_TARGETS", "").split(";"))
                or any(argv.count("--offload-arch=" + target) != 1 for target in targets)):
            raise ValueError("custom HIP targets differ from the real compiler argv")
        environment = []
        for item in _strings(row.get("environment")):
            key, separator, value = item.partition("=")
            if not separator or not key or any(name == key for name, _ in environment):
                raise ValueError("custom HIP environment is missing or ambiguous")
            environment.append((key, value))
        generated = tuple(_absolute(path) for path in _strings(row.get("generated_inputs")))
        native = {key: row[key] for key in ("file", "directory", "arguments", "output")}
        add(native, kind="hip", environment=environment, generated=generated, targets=targets)
    if not contexts:
        raise ValueError("zero selected first-party compilation contexts cannot pass analysis")
    compiler = _absolute(cache.get("CMAKE_CXX_COMPILER"))
    artifacts = list(dict.fromkeys([cache_path, database, target_path, hip_path, compiler, *sorted(hip_compilers)]))
    return AnalysisInputs(build, configuration, tuple(contexts), tuple(excluded),
        tuple((str(path), file_digest(path)) for path in artifacts), compiler)
