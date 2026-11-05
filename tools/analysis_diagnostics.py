"""Pinned LLVM23 DiagnosticsYaml adapter; fixes are retained, never applied.

Producer: llvmorg-23.1.4 clang/include/clang/Tooling/DiagnosticsYaml.h.
"""

from __future__ import annotations

from pathlib import Path

from compiled_binary_identity import canonical_digest

SUPPORTED_TOOL = "23.1.4"
MAX_DOCUMENT_BYTES = 16 * 1024**2
MAX_RECORDS = 100000


def _object(value: object, required: set[str], optional: set[str]) -> dict:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - optional:
        raise ValueError("unsupported LLVM diagnostic object fields")
    return value


def _text(value: object) -> str:
    if type(value) is not str:
        raise ValueError("LLVM diagnostic text must remain a string")
    return value


def _unsigned(value: object) -> int:
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError("LLVM source offsets/lengths must be nonnegative integers")
    return value


def _rows(value: object) -> list:
    if not isinstance(value, list) or len(value) > MAX_RECORDS:
        raise ValueError("LLVM diagnostic inventory exceeds its record budget")
    return value


def _location(path: str, offset: int, base: Path) -> dict:
    result = dict(reported_path=path, byte_offset=offset, path=None, line=None,
                  column_byte=None, state="unlocated")
    if not path or path.startswith("<"):
        return result
    source = (base / path).resolve()
    result["path"] = str(source)
    if not source.is_file():
        result["state"] = "reported_file_unavailable"
        return result
    if offset > source.stat().st_size:
        raise ValueError("LLVM diagnostic offset extends beyond its source file")
    # Offset is a byte offset, including CRLF/UTF-8 bytes; never reinterpret it
    # as a Python character index when computing a display location.
    if offset > MAX_DOCUMENT_BYTES:
        result["state"] = "display_location_exceeds_read_budget"
        return result
    with source.open("rb") as stream:
        prefix = stream.read(offset)
    result.update(state="located", line=prefix.count(b"\n") + 1,
                  column_byte=len(prefix.rsplit(b"\n", 1)[-1]) + 1)
    return result


def _message(value: object, base: Path) -> dict:
    value = _object(value, {"Message", "Replacements"}, {"FilePath", "FileOffset", "Ranges"})
    text = _text(value["Message"])
    location = _location(_text(value.get("FilePath", "")), _unsigned(value.get("FileOffset", 0)), base)
    replacements = []
    for row in _rows(value["Replacements"]):
        row = _object(row, {"FilePath", "Offset", "Length", "ReplacementText"}, set())
        replacements.append(dict(file=_text(row["FilePath"]), offset=_unsigned(row["Offset"]),
                                 length=_unsigned(row["Length"]), text=_text(row["ReplacementText"])))
    ranges = []
    for row in _rows(value.get("Ranges", [])):
        row = _object(row, {"FilePath", "FileOffset", "Length"}, set())
        ranges.append(dict(file=_text(row["FilePath"]), offset=_unsigned(row["FileOffset"]),
                           length=_unsigned(row["Length"])))
    return dict(message=text, location=location, replacements=replacements, ranges=ranges)


def diagnostic_document(document: object, *, source: Path, directory: Path,
                        tool_version: str = SUPPORTED_TOOL) -> list[dict]:
    if tool_version != SUPPORTED_TOOL:
        raise ValueError("unsupported LLVM diagnostic adapter version")
    document = _object(document, {"MainSourceFile", "Diagnostics"}, set())
    main = (directory / _text(document["MainSourceFile"])).resolve()
    if main != source.resolve():
        raise ValueError("native LLVM diagnostics belong to another translation unit")
    result = []
    for row in _rows(document["Diagnostics"]):
        row = _object(row, {"DiagnosticName", "DiagnosticMessage"}, {"Notes", "Level", "BuildDirectory"})
        check, level = _text(row["DiagnosticName"]), _text(row.get("Level", "Warning"))
        if not check or level not in {"Warning", "Error", "Remark"}:
            raise ValueError("LLVM diagnostic has an unknown check/level")
        native_directory = _text(row.get("BuildDirectory", ""))
        base = (directory / native_directory).resolve() if native_directory else directory
        observation = dict(check=check, level=level, main_source=str(main),
            message=_message(row["DiagnosticMessage"], base),
            notes=[_message(note, base) for note in _rows(row.get("Notes", []))],
            parse_failure=check == "clang-diagnostic-error")
        observation["identity_sha256"] = canonical_digest(observation)
        result.append(observation)
    return result


def read_diagnostics(path: Path, *, source: Path, directory: Path,
                     tool_version: str = SUPPORTED_TOOL) -> list[dict]:
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_DOCUMENT_BYTES:
        raise ValueError("native LLVM diagnostic document is empty or oversized")
    try:
        from ruamel.yaml import YAML
        from ruamel.yaml.error import YAMLError
        from ruamel.yaml.events import AliasEvent, CollectionStartEvent, CollectionEndEvent
    except ImportError as error:
        raise ValueError("install the locked optional analyzer diagnostic decoder") from error
    text = raw.decode("utf-8", errors="strict")
    count = depth = 0
    # The maintained producer does not emit YAML aliases/anchors or deeply
    # recursive schemas. Reject them before constructing even safe YAML data.
    try:
        for event in YAML(typ="safe", pure=True).parse(text):
            count += 1
            if isinstance(event, AliasEvent) or getattr(event, "anchor", None) is not None:
                raise ValueError("LLVM diagnostic YAML aliases/anchors are unsupported")
            if isinstance(event, CollectionStartEvent):
                depth += 1
            elif isinstance(event, CollectionEndEvent):
                depth -= 1
            if count > MAX_RECORDS or depth > 64:
                raise ValueError("LLVM diagnostic YAML exceeds its structural budget")
        loader = YAML(typ="safe", pure=True)
        loader.allow_duplicate_keys = False
        document = loader.load(text)
    except ValueError:
        raise
    except YAMLError as error:
        raise ValueError("malformed native LLVM diagnostic YAML") from error
    return diagnostic_document(document, source=source, directory=directory, tool_version=tool_version)
