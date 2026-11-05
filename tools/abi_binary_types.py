"""Optional, pinned native-Linux ELF type comparison; never symbol-only proof."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from abi_surface import (
    Object,
    Surface,
    command_output,
    declaration_sha256,
    object_value,
    sha256,
)
from json_evidence import load_evidence_json, same_evidence_value

SCHEMA = "rns8_linux_elf_abi_baseline_v1"


def complete_types(xml: str, surface: Surface) -> Object:
    # ABIXML is produced by the explicitly selected local abidw executable, not
    # an arbitrary remote XML input. Reject declarations without real type data.
    if "<!DOCTYPE" in xml or "<!ENTITY" in xml:
        raise ValueError("ABI XML must not contain external definitions")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as error:
        raise ValueError("malformed ELF ABI XML") from error
    if root.tag != "abi-corpus":
        raise ValueError("one ELF ABI corpus required")
    nodes = list(root.iter())
    identifiers = {node.get("id"): node for node in nodes if node.get("id")}
    if any(
        node.get("type-id") not in identifiers for node in nodes if node.get("type-id")
    ):
        raise ValueError("ELF debug type graph contains unresolved references")
    declarations: dict[str, list[ET.Element]] = {}
    for node in nodes:
        if node.get("name"):
            declarations.setdefault(node.get("name"), []).append(node)

    def resolved(node: ET.Element) -> ET.Element:
        seen: set[str] = set()
        while node.tag == "typedef-decl":
            identifier = node.get("type-id")
            if not identifier or identifier in seen or identifier not in identifiers:
                raise ValueError("ABI type alias is unresolved or cyclic")
            seen.add(identifier)
            node = identifiers[identifier]
        return node

    for name, fields in surface.records.items():
        candidates = [
            resolved(node)
            for node in declarations.get(name, [])
            if node.tag in ("class-decl", "union-decl", "typedef-decl")
        ]
        complete = [
            node
            for node in candidates
            if node.tag in ("class-decl", "union-decl")
            and int(node.get("size-in-bits", "0")) > 0
            and node.get("is-declaration-only") != "yes"
            and all(
                member.get("layout-offset-in-bits") is not None
                for member in node.findall("data-member")
                if node.tag == "class-decl"
            )
            and {
                child.get("name")
                for member in node.findall("data-member")
                for child in member.findall("var-decl")
            }
            >= {field[0] for field in fields}
        ]
        if not complete:
            raise ValueError("ELF debug types omit the complete public record " + name)
    for name, constants in surface.enums.items():
        candidates = [
            resolved(node)
            for node in declarations.get(name, [])
            if node.tag in ("enum-decl", "typedef-decl")
        ]
        if not any(
            node.tag == "enum-decl"
            and int(node.get("size-in-bits", "0")) == 32
            and node.find("underlying-type") is not None
            and {
                item.get("name"): int(item.get("value"))
                for item in node.findall("enumerator")
            }
            == constants
            for node in candidates
        ):
            raise ValueError("ELF debug types omit the complete public enum " + name)
    for name in surface.functions:
        if not any(
            node.tag == "function-decl"
            and node.find("return") is not None
            and all(
                parameter.get("type-id") in identifiers
                for parameter in node.findall("parameter")
            )
            and node.find("return").get("type-id") in identifiers
            for node in declarations.get(name, [])
        ):
            raise ValueError(
                "ELF debug types omit the public function signature " + name
            )
    return dict(
        records=len(surface.records),
        enums=len(surface.enums),
        functions=len(surface.functions),
        type_information_complete=True,
        symbol_only=False,
    )


def pinned_tool(path: Path, expected: str, output: Path) -> Object:
    path = path.resolve(strict=True)
    version = command_output([str(path), "--version"], output).strip()
    # Accept the tool's own program prefix, but not substring matches (2.8
    # must not silently accept 2.8.1, another release or a vendor suffix).
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", expected):
        raise ValueError("pin the complete numeric libabigail release")
    if not re.search(r"(?<![\w.])" + re.escape(expected) + r"(?![\w.+-])", version):
        raise ValueError("libabigail tool version does not match its explicit pin")
    return dict(
        path=str(path), sha256=sha256(path), version=version, release_pin=expected
    )


def compare_elf(
    *,
    binary: Path,
    surface: Surface,
    native_tuple: str,
    compiler_model: Object,
    abidw: Path,
    abidiff: Path,
    version: str,
    baseline: Path,
    output: Path,
) -> Object:
    if sys.platform != "linux" or not native_tuple.startswith("linux-"):
        raise ValueError("libabigail qualification requires native Linux ELF")
    with binary.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError("libabigail input is not an ELF binary")
    expected = object_value(load_evidence_json(baseline))
    if expected.get("schema") != SCHEMA:
        raise ValueError("a content-bound, like-for-like ELF ABI baseline is required")
    if not isinstance(expected.get("binary_sha256"), str) or not re.fullmatch(
        "[0-9a-f]{64}", expected["binary_sha256"]
    ):
        raise ValueError("ELF baseline must identify its reviewed binary content")
    for name, value in dict(
        tuple=native_tuple,
        compiler_model=compiler_model,
        release_pin=version,
        declaration_sha256=declaration_sha256(surface),
    ).items():
        if not same_evidence_value(expected.get(name), value):
            raise ValueError(
                "ELF baseline differs in target/compiler/configuration: " + name
            )
    baseline_xml = (baseline.parent / str(expected.get("xml"))).resolve(strict=True)
    if not baseline_xml.is_relative_to(baseline.parent.resolve(strict=True)):
        raise ValueError("ELF baseline XML escaped its evidence directory")
    if sha256(baseline_xml) != expected.get("xml_sha256"):
        raise ValueError("ELF baseline XML content changed")
    complete_types(baseline_xml.read_text(encoding="utf-8"), surface)
    dw = pinned_tool(abidw, version, output / "abidw-version.txt")
    diff = pinned_tool(abidiff, version, output / "abidiff-version.txt")
    xml_path = output / "candidate.abixml"
    binary_hash = sha256(binary)
    xml = command_output(
        [str(abidw.resolve(strict=True)), "--no-show-locs", str(binary)],
        xml_path,
        timeout=600,
    )
    completeness = complete_types(xml, surface)
    command = [str(abidiff.resolve(strict=True)), str(baseline_xml), str(xml_path)]
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=600,
        check=False,
    )
    report = output / "abidiff.txt"
    report.write_text(process.stdout, encoding="utf-8")
    report.with_suffix(".stderr").write_text(process.stderr, encoding="utf-8")
    if sha256(binary) != binary_hash or sha256(baseline_xml) != expected.get(
        "xml_sha256"
    ):
        raise ValueError("ELF inputs changed during comparison")
    # abidiff uses a bitmask: error/usage/ABI-change/incompatible-change are all
    # non-success. No difference may be silently suppressed into a passing row.
    if process.returncode != 0:
        raise ValueError("ELF ABI comparison failed or changed; inspect " + str(report))
    return dict(
        status="complete_native_elf_types_match",
        scope="ELF_C_binary_types_not_PDB_or_GPU_ABI",
        baseline_sha256=sha256(baseline),
        xml_sha256=sha256(xml_path),
        binary_sha256=binary_hash,
        abidw=dw,
        abidiff=diff,
        completeness=completeness,
        command=command,
    )
