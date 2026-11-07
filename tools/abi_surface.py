"""Compiler-derived inventory of the installed RNS8 C declaration surface."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess

from json_evidence import JsonValue, parse_evidence_json, same_evidence_value

SCHEMA = "rns8_public_c_surface_v1"
type Object = dict[str, JsonValue]


def object_value(value: JsonValue) -> Object:
    if not isinstance(value, dict):
        raise ValueError("ABI evidence requires an object")
    return value


def children(node: Object):
    values = node.get("inner", [])
    if not isinstance(values, list):
        raise ValueError("malformed frontend children")
    for value in values:
        yield object_value(value)


def walk(node: Object):
    yield node
    for child in children(node):
        yield from walk(child)


def spelling(value: JsonValue) -> str:
    if not isinstance(value, str):
        raise ValueError("frontend type/name spelling missing")
    # This normalizes frontend pretty-printing, never discovers declarations.
    value = re.sub(r"\b(?:struct|enum|union)\s+(rns8_\w+)", r"\1", value)
    return value.replace("(void)", "()")


def qualified_type(node: Object) -> str:
    return spelling(object_value(node.get("type")).get("qualType"))


def enum_constants(node: Object) -> dict[str, int]:
    result: dict[str, int] = {}
    previous = -1
    for child in children(node):
        if child.get("kind") != "EnumConstantDecl":
            continue
        constants = [
            item for item in walk(child) if item.get("kind") == "ConstantExpr" and "value" in item
        ]
        if not constants and list(children(child)):
            raise ValueError("frontend omitted a folded enum initializer")
        previous = int(spelling(constants[0]["value"]), 0) if constants else previous + 1
        name = spelling(child.get("name"))
        if name in result:
            raise ValueError("duplicate enum constant: " + name)
        result[name] = previous
    if not result:
        raise ValueError("empty public enum inventory")
    return result


@dataclass(frozen=True)
class Surface:
    schema: str
    headers: dict[str, str]
    enums: dict[str, dict[str, int]]
    constants: dict[str, int]
    records: dict[str, list[list[str]]]
    record_kinds: dict[str, str]
    handles: list[str]
    callbacks: dict[str, str]
    aliases: dict[str, str]
    functions: dict[str, str]
    header_functions: dict[str, str]
    visibility: dict[str, list[str]]
    macros: dict[str, str]

    def to_json(self) -> Object:
        return asdict(self)


def collect_surface(
    ast: Object, headers: dict[str, str], macros: dict[str, str], *, cpp: bool
) -> Surface:
    if ast.get("kind") != "TranslationUnitDecl":
        raise ValueError("a real Clang translation-unit AST is required")
    (
        enums,
        constants,
        records,
        kinds,
        callbacks,
        aliases,
        functions,
        inline,
        visibility,
    ) = {}, {}, {}, {}, {}, {}, {}, {}, {}
    incomplete: set[str] = set()
    typedefs: dict[str, Object] = {}
    nodes = list(walk(ast))
    tag_aliases: dict[str, str] = {}
    for node in nodes:
        name = node.get("name")
        if (
            node.get("kind") != "TypedefDecl"
            or not isinstance(name, str)
            or not name.startswith("rns8_")
        ):
            continue
        typedefs[name] = node
        roots = list(children(node))
        while len(roots) == 1:
            root = roots[0]
            if root.get("kind") in (
                "PointerType",
                "FunctionProtoType",
                "ConstantArrayType",
            ):
                break
            declaration = root.get("decl") or root.get("ownedTagDecl")
            if isinstance(declaration, dict) and declaration.get("kind") in (
                "EnumDecl",
                "RecordDecl",
                "CXXRecordDecl",
            ):
                identity = declaration.get("id")
                if isinstance(identity, str):
                    tag_aliases.setdefault(identity, name)
                break
            roots = list(children(root))
    for node in nodes:
        name = tag_aliases.get(str(node.get("id")), node.get("name"))
        kind = node.get("kind")
        if not isinstance(name, str) or not name.startswith("rns8_"):
            if kind == "FunctionDecl" and any(
                child.get("kind") in ("DLLImportAttr", "DLLExportAttr", "VisibilityAttr")
                for child in children(node)
            ):
                raise ValueError(
                    "exported declarations must use the public rns8_ namespace: " + str(name)
                )
            if kind == "EnumDecl" and any(
                str(child.get("name", "")).startswith("RNS8_") for child in children(node)
            ):
                if node.get("name"):
                    raise ValueError(
                        "public enum outside the RNS8 namespace needs an explicit public alias"
                    )
                for constant, value in enum_constants(node).items():
                    if constant.startswith("RNS8_"):
                        constants[constant] = value
            continue
        if kind == "EnumDecl":
            if cpp:
                underlying = object_value(node.get("fixedUnderlyingType"))
                if underlying.get("qualType") != "uint32_t":
                    raise ValueError("public enum lacks fixed uint32_t C++ backing: " + name)
            values = enum_constants(node)
            if any(value < 0 or value > 0xFFFFFFFF for value in values.values()):
                raise ValueError("public enum exceeds the supported uint32 ABI: " + name)
            if name in enums and enums[name] != values:
                raise ValueError("conflicting public enum: " + name)
            enums[name] = values
        elif kind in ("RecordDecl", "CXXRecordDecl"):
            if not node.get("completeDefinition"):
                incomplete.add(name)
                continue
            fields = []
            for field in children(node):
                if field.get("kind") != "FieldDecl":
                    continue
                if field.get("isBitfield") or not field.get("name"):
                    raise ValueError(
                        "public bitfield/anonymous field needs explicit ABI support: " + name
                    )
                field_type = qualified_type(field)
                if "(unnamed" in field_type or "anonymous" in field_type:
                    raise ValueError(
                        "anonymous public member type needs explicit ABI support: " + name
                    )
                fields.append([spelling(field["name"]), field_type])
            if not fields:
                raise ValueError("empty complete public record: " + name)
            if name in records and records[name] != fields:
                raise ValueError("conflicting public record: " + name)
            records[name] = fields
            kinds[name] = spelling(node.get("tagUsed"))
        elif kind == "TypedefDecl":
            typedefs[name] = node
        elif kind == "FunctionDecl":
            target = (
                inline if node.get("storageClass") == "static" or node.get("inline") else functions
            )
            signature = qualified_type(node)
            if name in target and target[name] != signature:
                raise ValueError("overloaded/conflicting C public symbol: " + name)
            mangled = node.get("mangledName")
            if target is functions and mangled != name:
                raise ValueError(
                    "public function lacks ordinary C linkage/calling convention: " + name
                )
            target[name] = signature
            visibility[name] = sorted(
                {
                    str(child["kind"])
                    for child in children(node)
                    if str(child.get("kind", "")).endswith("Attr")
                }
            )
        elif kind == "VarDecl":
            raise ValueError("public data export needs an explicit inventory/probe policy: " + name)
    handles = sorted(incomplete - records.keys())
    for name, node in typedefs.items():
        if name in records or name in enums or name in handles:
            continue
        if any(child.get("kind") == "FunctionProtoType" for child in walk(node)):
            callbacks[name] = qualified_type(node)
        else:
            aliases[name] = qualified_type(node)
    if not enums or not records or not functions:
        raise ValueError("empty/incomplete public C surface")
    return Surface(
        SCHEMA,
        dict(sorted(headers.items())),
        dict(sorted(enums.items())),
        dict(sorted(constants.items())),
        dict(sorted(records.items())),
        dict(sorted(kinds.items())),
        handles,
        dict(sorted(callbacks.items())),
        dict(sorted(aliases.items())),
        dict(sorted(functions.items())),
        dict(sorted(inline.items())),
        dict(sorted(visibility.items())),
        dict(sorted(macros.items())),
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_output(command: list[str], output: Path, *, timeout: float = 180) -> str:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=timeout,
        check=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.stdout, encoding="utf-8")
    output.with_suffix(output.suffix + ".stderr").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise ValueError(
            "ABI frontend failed; see " + str(output.with_suffix(output.suffix + ".stderr"))
        )
    return result.stdout


def frontend_surface(
    include: Path, clang: Path, output: Path, *, cpp: bool, flags: tuple[str, ...] = ()
) -> tuple[Surface, Object]:
    include, clang, output = (
        include.resolve(strict=True),
        clang.resolve(strict=True),
        output.resolve(),
    )
    headers = sorted((include / "rns8").rglob("*.h"))
    if not headers or not (include / "rns8" / "rns8.h").is_file():
        raise ValueError("complete installed RNS8 C headers required")
    output.mkdir(parents=True, exist_ok=True)
    extension = "cpp" if cpp else "c"
    source = output / ("surface." + extension)
    source.write_text(
        "".join('#include "' + path.as_posix() + '"\n' for path in headers),
        encoding="utf-8",
    )
    base = [
        str(clang),
        "-x",
        "c++" if cpp else "c",
        "-std=c++17" if cpp else "-std=c17",
        "-I" + str(include),
        *flags,
    ]
    ast_command = [*base, "-fsyntax-only", "-Xclang", "-ast-dump=json", str(source)]
    ast = object_value(
        parse_evidence_json(command_output(ast_command, output / (extension + "-ast.json")))
    )
    macros_text = command_output(
        [*base, "-dM", "-E", str(source)], output / (extension + "-macros.txt")
    )
    macros: dict[str, str] = {}
    for line in macros_text.splitlines():
        if not line.startswith("#define RNS8_"):
            continue
        name, _, value = line[len("#define ") :].partition(" ")
        if "(" in name:
            raise ValueError("function-like public macro needs an explicit ABI policy: " + name)
        if value and name not in (
            "RNS8_API",
            "RNS8_ENUM_BASE",
            "RNS8_STATIC",
            "RNS8_BUILDING_LIBRARY",
        ):
            macros[name] = value
    surface = collect_surface(
        ast,
        {path.relative_to(include).as_posix(): sha256(path) for path in headers},
        macros,
        cpp=cpp,
    )
    receipt: Object = dict(
        clang=str(clang),
        clang_sha256=sha256(clang),
        version=command_output([str(clang), "--version"], output / "clang-version.txt").strip(),
        target=command_output(
            [str(clang), "-print-target-triple"], output / "clang-target.txt"
        ).strip(),
        language="c++17" if cpp else "c17",
        command=ast_command,
        include_root=str(include),
        scope="declaration_frontend_only_not_native_link_or_runtime_qualification",
    )
    return surface, receipt


def comparison_projection(surface: Object) -> Object:
    required = {
        "schema",
        "headers",
        "enums",
        "constants",
        "records",
        "record_kinds",
        "handles",
        "callbacks",
        "aliases",
        "functions",
        "header_functions",
        "visibility",
        "macros",
    }
    if set(surface) != required or surface.get("schema") != SCHEMA:
        raise ValueError("unknown/incomplete public ABI surface schema")
    if any(not surface.get(key) for key in ("headers", "enums", "records", "functions")):
        raise ValueError("empty public ABI inventory is not compatible")
    # Header hashes and platform import/export attributes bind each receipt but
    # do not make a comment edit, or static/shared declaration mode, an ABI break.
    return {key: value for key, value in surface.items() if key not in ("headers", "visibility")}


def surface_diff(expected: Object, actual: Object) -> list[Object]:
    left, right = comparison_projection(expected), comparison_projection(actual)
    changes: list[Object] = []
    for category in sorted(left):
        a, b = left[category], right[category]
        if isinstance(a, dict) and isinstance(b, dict):
            for name in sorted(a.keys() | b.keys()):
                if not same_evidence_value(a.get(name), b.get(name)):
                    changes.append(
                        dict(
                            category=category,
                            name=name,
                            change="added"
                            if name not in a
                            else "removed"
                            if name not in b
                            else "changed",
                            before=a.get(name),
                            after=b.get(name),
                        )
                    )
        elif not same_evidence_value(a, b):
            changes.append(dict(category=category, change="changed", before=a, after=b))
    return changes


def declaration_sha256(surface: Surface) -> str:
    projection = comparison_projection(surface.to_json())
    return hashlib.sha256(
        json.dumps(projection, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
