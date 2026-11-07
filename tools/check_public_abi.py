"""Inventory/probe the real public C surface; native checks require an installed prefix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from abi_surface import (
    Surface,
    Object,
    frontend_surface,
    command_output,
    object_value,
    walk,
    enum_constants,
    sha256,
    surface_diff,
    declaration_sha256,
)
from abi_probes import (
    metric_frontend_source,
    native_probe_source,
    installed_consumer_cmake,
    library_inventory_source,
    LAYOUT_SCHEMA,
)
from json_evidence import load_evidence_json, parse_evidence_json, same_evidence_value
from compiler_context import invocation_path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_SCHEMA = "rns8_reviewed_public_abi_v1"
DEFAULT_REGISTRY = ROOT / "metadata" / "public-c-abi.json"
DEFAULT_LIBRARY_INVENTORY = ROOT / "src" / "core" / "public_abi_inventory.inc"


def write_json(path: Path, value: Object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def frontend_metrics(
    surface: Surface, include: Path, clang: Path, output: Path, *, cpp: bool
) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    source, names = metric_frontend_source(surface, include)
    path = output / ("metrics.cpp" if cpp else "metrics.c")
    path.write_text(source, encoding="utf-8")
    command = [
        str(clang.resolve(strict=True)),
        "-x",
        "c++" if cpp else "c",
        "-std=c++17" if cpp else "-std=c17",
        "-I" + str(include.resolve(strict=True)),
        "-fsyntax-only",
        "-Xclang",
        "-ast-dump=json",
        str(path),
    ]
    ast = object_value(parse_evidence_json(command_output(command, output / "metrics-ast.json")))
    nodes = [
        node
        for node in walk(ast)
        if node.get("kind") == "EnumDecl" and node.get("name") == "rns8_abi_frontend_metrics"
    ]
    if len(nodes) != 1:
        raise ValueError("frontend did not emit the complete ABI metric inventory")
    values = enum_constants(nodes[0])
    expected = {"rns8_abi_metric_" + str(index) for index in range(len(names))}
    if set(values) != expected:
        raise ValueError("frontend metric inventory is incomplete")
    return {name: values["rns8_abi_metric_" + str(index)] for index, name in enumerate(names)}


def tuple_key(target: str, metrics: dict[str, int]) -> str:
    if metrics.get("sizeof:void_pointer") != 8 or metrics.get("CHAR_BIT") != 8:
        raise ValueError("supported native ABI probes require 64-bit pointers and 8-bit bytes")
    architecture = target.split("-", 1)[0]
    if architecture not in ("x86_64", "aarch64", "arm64"):
        raise ValueError("ABI target architecture has no reviewed policy")
    system = "windows" if "windows" in target else "linux" if "linux" in target else None
    if system is None:
        raise ValueError("ABI target must be native Windows or Linux")
    model = (
        "llp64"
        if metrics.get("sizeof:long") == 4
        else "lp64"
        if metrics.get("sizeof:long") == 8
        else None
    )
    if model is None:
        raise ValueError("unsupported native data model")
    return system + "-" + architecture + "-" + model


def discover(include: Path, clang: Path, output: Path) -> tuple[Surface, Object, dict[str, int]]:
    cpp, receipt = frontend_surface(include, clang, output / "cpp", cpp=True)
    c, _ = frontend_surface(include, clang, output / "c", cpp=False)
    differences = surface_diff(cpp.to_json(), c.to_json())
    if differences:
        write_json(output / "c-cpp-declaration-diff.json", dict(changes=differences))
        raise ValueError("C17 and C++17 public declaration inventories differ")
    metrics = frontend_metrics(cpp, include, clang, output / "cpp-layout", cpp=True)
    c_metrics = frontend_metrics(c, include, clang, output / "c-layout", cpp=False)
    if metrics != c_metrics:
        write_json(output / "c-cpp-layout-diff.json", dict(c=c_metrics, cpp=metrics))
        raise ValueError("C17/C++17 frontend layouts differ")
    return cpp, receipt, metrics


def registry(path: Path) -> Object:
    value = object_value(load_evidence_json(path))
    if (
        set(value) != {"schema", "surface", "layout_baselines"}
        or value.get("schema") != REGISTRY_SCHEMA
    ):
        raise ValueError("complete reviewed ABI registry required")
    object_value(value.get("surface"))
    if not object_value(value.get("layout_baselines")):
        raise ValueError("an empty ABI baseline never proves compatibility")
    return value


def baseline_metrics(value: Object, key: str) -> dict[str, int]:
    baselines = object_value(value.get("layout_baselines"))
    if key not in baselines:
        raise ValueError(
            "no reviewed layout baseline for native tuple "
            + key
            + "; author/review it on that target first"
        )
    entry = object_value(baselines[key])
    metrics = object_value(entry.get("metrics"))
    if not metrics or any(
        type(item) is not int or not 0 <= item <= 0xFFFFFFFFFFFFFFFF for item in metrics.values()
    ):
        raise ValueError("complete unsigned64 ABI layout baseline required for " + key)
    return dict(metrics)


def author(args) -> Object:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    surface, receipt, metrics = discover(args.include, args.clang, output)
    key = tuple_key(str(receipt["target"]), metrics)
    entry: Object = dict(
        metrics=metrics,
        provenance=dict(
            kind="compiler_frontend_candidate_not_native_consumer_qualification",
            compiler="clang",
            version=receipt["version"],
            executable_sha256=receipt["clang_sha256"],
            target=receipt["target"],
            language_pair=["c17", "c++17"],
        ),
    )
    if args.registry.is_file():
        result = registry(args.registry)
        result["surface"] = surface.to_json()
        object_value(result["layout_baselines"])[key] = entry
    else:
        result = dict(
            schema=REGISTRY_SCHEMA,
            surface=surface.to_json(),
            layout_baselines={key: entry},
        )
    write_json(args.registry, result)
    args.library_inventory.write_text(library_inventory_source(surface), encoding="utf-8")
    write_json(
        output / "author-receipt.json",
        dict(
            schema=REGISTRY_SCHEMA,
            frontend=receipt,
            registry_sha256=sha256(args.registry),
            tuple=key,
            qualified_native_consumers=False,
        ),
    )
    return dict(
        status="authored_frontend_candidate",
        tuple=key,
        registry=str(args.registry),
        enum_count=len(surface.enums),
        record_count=len(surface.records),
        symbol_count=len(surface.functions),
    )


def check_source(args) -> Object:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    expected = registry(args.registry)
    surface, receipt, metrics = discover(args.include, args.clang, output)
    differences = surface_diff(object_value(expected["surface"]), surface.to_json())
    if args.library_inventory.read_text(encoding="utf-8") != library_inventory_source(surface):
        raise ValueError(
            "compiled ABI inventory is stale; regenerate and review it with the public declaration registry"
        )
    key = tuple_key(str(receipt["target"]), metrics)
    layout = baseline_metrics(expected, key)
    changed_layout = {
        name: dict(before=layout.get(name), after=metrics.get(name))
        for name in sorted(layout.keys() | metrics.keys())
        if layout.get(name) != metrics.get(name)
    }
    result: Object = dict(
        schema="rns8_public_abi_source_review_v1",
        frontend=receipt,
        status="declarations_and_frontend_layout_match"
        if not differences and not changed_layout
        else "changed",
        declaration_changes=differences,
        layout_changes=changed_layout,
        tuple=key,
        qualified_native_consumers=False,
        binary_exports_checked=False,
    )
    write_json(output / "source-review.json", result)
    if differences or changed_layout:
        raise ValueError(
            "public ABI changed; inspect the source-review receipt and deliberately update/version the API"
        )
    return result


def installed_config(prefix: Path) -> Path:
    candidates = [
        *prefix.glob("lib*/cmake/RNS8/RNS8Config.cmake"),
        *prefix.glob("lib/*/cmake/RNS8/RNS8Config.cmake"),
    ]
    paths = {path.resolve(strict=True) for path in candidates}
    if len(paths) != 1:
        raise ValueError(
            "one installed RNS8 package configuration is required; no build-tree fallback"
        )
    result = paths.pop()
    if not result.is_relative_to(prefix):
        raise ValueError("installed package configuration escaped its prefix")
    return result.parent


def binary_exports(
    binary: Path, llvm_readobj: Path | None, llvm_nm: Path | None, output: Path
) -> set[str]:
    if os.name == "nt":
        if llvm_readobj is None:
            raise ValueError("an explicit LLVM PE export reader is required")
        text = command_output(
            [str(llvm_readobj.resolve(strict=True)), "--coff-exports", str(binary)],
            output,
        )
        exports = {
            line.split("Name:", 1)[1].strip()
            for line in text.splitlines()
            if line.strip().startswith("Name:")
        }
    else:
        if llvm_nm is None:
            raise ValueError("an explicit LLVM ELF dynamic-symbol reader is required")
        text = command_output(
            [
                str(llvm_nm.resolve(strict=True)),
                "--dynamic",
                "--defined-only",
                "--extern-only",
                "--format=posix",
                str(binary),
            ],
            output,
        )
        exports = {line.split()[0].split("@", 1)[0] for line in text.splitlines() if line.split()}
    public = {name for name in exports if name.startswith("rns8_")}
    if not public:
        raise ValueError("empty public export inventory cannot qualify a shared library")
    return public


def native_run(command: list[str], output: Path, env: dict[str, str] | None = None) -> Object:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        env=env,
        timeout=180,
        check=False,
    )
    output.write_text(result.stdout, encoding="utf-8")
    output.with_suffix(output.suffix + ".stderr").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise ValueError("installed ABI consumer failed: " + str(output))
    value = object_value(parse_evidence_json(result.stdout))
    if value.get("schema") != LAYOUT_SCHEMA or value.get("native_device_raw_enum_cases") != 9:
        raise ValueError("installed ABI consumer did not emit complete layout/CPU smoke evidence")
    if value.get("all_external_enum_paths_covered") is not False:
        raise ValueError("installed ABI consumer overstates behavioral scope")
    for name in ("library_compiler_id", "library_runtime_model", "declaration_sha256"):
        if not isinstance(value.get(name), str) or not value[name]:
            raise ValueError("installed ABI consumer omitted library provenance: " + name)
    return value


def native_build_profile(build: Path, compilers: dict[str, Path]) -> Object:
    """Preserve actual compiler definitions and flags, rather than guessed defaults."""
    commands_path = build / "compile_commands.json"
    commands = load_evidence_json(commands_path)
    if not isinstance(commands, list) or len(commands) < 2:
        raise ValueError("both native consumer compile commands are required")
    filenames = {Path(str(object_value(item).get("file"))).name for item in commands}
    if not {"probe.c", "probe.cpp"} <= filenames:
        raise ValueError("both native consumer languages are missing from compile commands")
    definitions = list(build.glob("CMakeFiles/*/CMakeCCompiler.cmake"))
    definitions += list(build.glob("CMakeFiles/*/CMakeCXXCompiler.cmake"))
    if len(definitions) != 2:
        raise ValueError("CMake native compiler identities are incomplete")
    cache_path = build / "CMakeCache.txt"
    cache = cache_path.read_text(encoding="utf-8")
    texts = [
        line.split("=", 1)[1]
        for line in cache.splitlines()
        if "=" in line and not line.startswith(("//", "#"))
    ]
    for item in commands:
        entry = object_value(item)
        command = entry.get("command")
        arguments = entry.get("arguments")
        if isinstance(command, str):
            texts.append(command)
        elif isinstance(arguments, list) and all(isinstance(word, str) for word in arguments):
            texts.extend(arguments)
        else:
            raise ValueError("native compile command has no actual arguments")
    reject_abi_flags(texts)
    return dict(
        compilers={
            name: dict(
                path=str(path.resolve(strict=True)),
                invocation=str(invocation_path(path)),
                sha256=sha256(path),
            )
            for name, path in compilers.items()
        },
        compile_commands=commands,
        compile_commands_sha256=sha256(commands_path),
        compiler_definitions={
            str(path.relative_to(build)): dict(
                sha256=sha256(path), content=path.read_text(encoding="utf-8")
            )
            for path in definitions
        },
        cmake_cache_sha256=sha256(cache_path),
        packing_policy="platform_default",
        enum_policy="fixed_u32_no_short_enums",
    )


def reject_abi_flags(texts: list[str]) -> None:
    import re

    # Check the actual commands and inherited cache flags, including clang-cl.
    # Fixed typedefs do not make unsupported global packing/enum switches safe.
    if any(
        re.search(
            r"(?i)(?:^|[\s\";])(?:-fshort-enums|-fpack-struct(?:=\S+)?|/Zp\d*|-Zp\d*)(?=$|[\s\";])",
            text,
        )
        for text in texts
    ):
        raise ValueError("unsupported enum/packing flags in native ABI configuration")


def negative_consumers(
    surface: Surface,
    include: Path,
    layout: dict[str, int],
    source: Path,
    prefix: Path,
    linkage: str,
    configuration: Path,
) -> list[str]:
    """Seed errors in disposable consumers, without modifying installed artifacts."""
    probe = native_probe_source(surface, include, layout)
    # A different field type of the same size must fail the generated type
    # checks, independently of offsetof/sizeof. Only the scratch copy is edited.
    copied = source / "mutated-include"
    shutil.copytree(include, copied)
    public = copied / "rns8" / "public_abi.h"
    contents = public.read_text(encoding="utf-8")
    anchor = "uint32_t abi_version;"
    if contents.count(anchor) != 1:
        raise ValueError("ABI member negative control anchor is ambiguous")
    public.write_text(contents.replace(anchor, "float abi_version;"), encoding="utf-8")
    bodies = {
        "packing": "#pragma pack(push,1)\n" + probe + "\n#pragma pack(pop)\n",
        "enum": probe.replace(
            "#include <stdio.h>",
            "#undef RNS8_SUCCESS\n#define RNS8_SUCCESS 0x7fffffffu\n#include <stdio.h>",
            1,
        ),
        "member": native_probe_source(surface, copied, layout),
        "missing_export": "#define rns8_get_public_abi_info rns8_abi_missing_export\n" + probe,
    }
    if bodies["enum"] == probe:
        raise ValueError("ABI enum negative control was not inserted")
    targets = []
    cmake = installed_consumer_cmake(prefix, linkage, configuration)
    for name, body in bodies.items():
        for language in ("c", "cpp"):
            filename = "negative_" + name + "." + language
            (source / filename).write_text(body, encoding="utf-8")
            target = "rns8_abi_negative_" + name + "_" + language
            imported = "rns8::rns8_static" if linkage == "static" else "rns8::rns8"
            cmake += (
                "\nadd_executable(" + target + " EXCLUDE_FROM_ALL " + filename + ")\n"
                "target_link_libraries(" + target + " PRIVATE " + imported + ")\n"
                "set_target_properties("
                + target
                + " PROPERTIES C_STANDARD 17 C_STANDARD_REQUIRED YES "
                "CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES C_EXTENSIONS NO CXX_EXTENSIONS NO)\n"
            )
            targets.append(target)
    (source / "CMakeLists.txt").write_text(cmake, encoding="utf-8")
    return targets


def run_negative_consumers(cmake: Path, build: Path, targets: list[str], output: Path) -> Object:
    results = {}
    for target in targets:
        command = [
            str(cmake),
            "--build",
            str(build),
            "--config",
            "Release",
            "--target",
            target,
            "--verbose",
        ]
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=300,
            check=False,
        )
        log = output / (target + ".log")
        log.write_text(process.stdout, encoding="utf-8")
        log.with_suffix(".stderr").write_text(process.stderr, encoding="utf-8")
        if process.returncode == 0:
            raise ValueError("seeded ABI mismatch unexpectedly compiled/linked: " + target)
        diagnostics = (process.stdout + "\n" + process.stderr).lower()
        if "_missing_export_" in target:
            explained = "rns8_abi_missing_export" in diagnostics and any(
                word in diagnostics
                for word in (
                    "unresolved external",
                    "undefined reference",
                    "undefined symbol",
                )
            )
        elif "_member_" in target:
            explained = "public member type" in diagnostics
        else:
            explained = "static assertion" in diagnostics or "error c2338" in diagnostics
        if not explained:
            raise ValueError(
                "negative ABI build failed without the expected mismatch diagnostic: " + target
            )
        # The successful ordinary build establishes toolchain readiness. Keep
        # the failed command and diagnostics for inspection, never execute it.
        results[target] = dict(
            expected_rejection=True,
            returncode=process.returncode,
            command=command,
            log_sha256=sha256(log),
        )
    return dict(
        scope="packing_changed_enum_same_size_field_and_removed_export",
        controls=results,
    )


def native_consumer_configure(
    cmake: Path, source: Path, build: Path, cc: Path, cxx: Path, generator: str
) -> list[str]:
    return [
        str(invocation_path(cmake)),
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        generator,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DCMAKE_C_COMPILER=" + str(invocation_path(cc)),
        "-DCMAKE_CXX_COMPILER=" + str(invocation_path(cxx)),
    ]


def check_installed(args) -> Object:
    prefix = args.prefix.resolve(strict=True)
    configuration = installed_config(prefix)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    include = prefix / "include"
    expected = registry(args.registry)
    surface, frontend, metrics = discover(include, args.clang, output / "frontend")
    differences = surface_diff(object_value(expected["surface"]), surface.to_json())
    if differences:
        write_json(output / "declaration-diff.json", dict(changes=differences))
        raise ValueError("installed declarations differ from the reviewed public inventory")
    key = tuple_key(str(frontend["target"]), metrics)
    layout = baseline_metrics(expected, key)
    if layout != metrics:
        raise ValueError("installed frontend layout differs from the reviewed native tuple")
    source = output / "consumer"
    source.mkdir()
    probe = native_probe_source(surface, include, layout)
    for extension in ("c", "cpp"):
        (source / ("probe." + extension)).write_text(probe, encoding="utf-8")
    negative_targets = negative_consumers(
        surface, include, layout, source, prefix, args.linkage, configuration
    )
    build = output / "build"
    configure = native_consumer_configure(
        args.cmake, source, build, args.cc, args.cxx, args.generator
    )
    command_output(configure, output / "configure.log", timeout=300)
    profile = native_build_profile(build, dict(c=args.cc, cpp=args.cxx))
    markers = list(build.glob("installed-binary-*.txt"))
    binaries = {
        Path(path.read_text(encoding="utf-8").strip()).resolve(strict=True) for path in markers
    }
    if len(binaries) != 1:
        raise ValueError("installed target binary was not resolved uniquely")
    binary = binaries.pop()
    if not binary.is_relative_to(prefix):
        raise ValueError("consumer resolved a build-tree/outside-prefix RNS8 binary")
    identity = sha256(binary)
    command_output(
        [
            str(args.cmake.resolve(strict=True)),
            "--build",
            str(build),
            "--config",
            "Release",
            "--verbose",
        ],
        output / "build.log",
        timeout=300,
    )
    environment = os.environ.copy()
    directories = [str(prefix / "bin"), str(binary.parent)]
    environment["PATH"] = os.pathsep.join([*directories, environment.get("PATH", "")])
    if os.name != "nt":
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(binary.parent), environment.get("LD_LIBRARY_PATH", "")]
        )
    results = {}
    for language in ("c", "cpp"):
        executable = (
            "rns8_abi_" + args.linkage + "_" + language + (".exe" if os.name == "nt" else "")
        )
        candidates = [
            path.resolve(strict=True)
            for path in (build / executable, build / "Release" / executable)
            if path.is_file()
        ]
        if len(candidates) != 1:
            raise ValueError("one native ABI consumer executable required for " + language)
        result = native_run([str(candidates[0])], output / (language + "-native.json"), environment)
        if not same_evidence_value(result.get("metrics"), layout):
            raise ValueError(
                "native " + language + " compiler layout differs from installed/reviewed headers"
            )
        if result.get("declaration_sha256") != declaration_sha256(surface):
            raise ValueError("native library declaration digest does not match installed headers")
        results[language] = dict(
            executable=str(candidates[0]),
            sha256=sha256(candidates[0]),
            observation=result,
        )
    negatives = run_negative_consumers(args.cmake, build, negative_targets, output)
    export_receipt = None
    if args.linkage == "shared":
        exports = binary_exports(binary, args.llvm_readobj, args.llvm_nm, output / "exports.txt")
        missing = sorted(set(surface.functions) - exports)
        unexpected = sorted(exports - set(surface.functions))
        if missing or unexpected:
            write_json(
                output / "export-diff.json",
                dict(missing=missing, unexpected=unexpected),
            )
            raise ValueError("shared exports differ from the complete public C symbol inventory")
        reader = args.llvm_readobj if os.name == "nt" else args.llvm_nm
        export_receipt = dict(
            symbols=sorted(exports), reader=str(reader), reader_sha256=sha256(reader)
        )
    if sha256(binary) != identity or any(
        sha256(include / name) != digest for name, digest in surface.headers.items()
    ):
        raise ValueError("installed headers/binary changed while checking ABI")
    elf = None
    elf_arguments = (args.abidw, args.abidiff, args.abigail_version, args.elf_baseline)
    if any(item is not None for item in elf_arguments):
        if not all(item is not None for item in elf_arguments) or args.linkage != "shared":
            raise ValueError(
                "ELF comparison needs shared linkage, both pinned tools, version and baseline"
            )
        from abi_binary_types import compare_elf

        observations = [object_value(entry["observation"]) for entry in results.values()]
        models = {
            (item["library_compiler_id"], item["library_runtime_model"]) for item in observations
        }
        if len(models) != 1:
            raise ValueError(
                "native consumers observed different library compiler/runtime identities"
            )
        compiler_id, runtime_model = models.pop()
        elf = compare_elf(
            binary=binary,
            surface=surface,
            native_tuple=key,
            compiler_model=dict(
                compiler_id=compiler_id,
                runtime_model=runtime_model,
                packing_policy=profile["packing_policy"],
                enum_policy=profile["enum_policy"],
            ),
            abidw=args.abidw,
            abidiff=args.abidiff,
            version=args.abigail_version,
            baseline=args.elf_baseline,
            output=output,
        )
    result: Object = dict(
        schema="rns8_installed_public_abi_v1",
        status="native_layout_link_and_smoke_passed",
        tuple=key,
        prefix=str(prefix),
        linkage=args.linkage,
        frontend=frontend,
        headers=surface.headers,
        binary=str(binary),
        binary_sha256=identity,
        registry_sha256=sha256(args.registry),
        configure_command=configure,
        native_build_profile=profile,
        negative_controls=negatives,
        consumers=results,
        exports=export_receipt,
        elf_type_comparison=elf,
        all_external_enum_paths_covered=False,
        scope="installed_C_layout_signatures_exports_link_and_native_device_CPU_smoke_not_all_runtime_or_GPU_behavior",
    )
    write_json(output / "abi-receipt.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    clang_default = shutil.which("clang")
    for name in ("author", "source", "installed"):
        child = sub.add_parser(name)
        child.add_argument(
            "--clang",
            type=Path,
            required=clang_default is None,
            default=Path(clang_default) if clang_default else None,
        )
        child.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
        child.add_argument(
            "--output", type=Path, required=True, help="fresh ignored output directory"
        )
        if name != "installed":
            child.add_argument("--include", type=Path, default=ROOT / "include")
            child.add_argument("--library-inventory", type=Path, default=DEFAULT_LIBRARY_INVENTORY)
        else:
            child.add_argument("--prefix", type=Path, required=True)
            child.add_argument("--linkage", choices=("static", "shared"), required=True)
            for option, names in (
                ("cc", ("cl", "clang", "cc") if os.name == "nt" else ("cc", "clang")),
                (
                    "cxx",
                    ("cl", "clang++", "c++") if os.name == "nt" else ("c++", "clang++"),
                ),
                ("cmake", ("cmake",)),
            ):
                found = next((path for name in names if (path := shutil.which(name))), None)
                child.add_argument(
                    "--" + option,
                    type=Path,
                    required=found is None,
                    default=Path(found) if found else None,
                )
            child.add_argument("--generator", default="Ninja")
            child.add_argument("--llvm-readobj", type=Path)
            child.add_argument("--llvm-nm", type=Path)
            child.add_argument("--abidw", type=Path)
            child.add_argument("--abidiff", type=Path)
            child.add_argument("--abigail-version")
            child.add_argument("--elf-baseline", type=Path)
    args = parser.parse_args()
    try:
        result = (
            author(args)
            if args.action == "author"
            else check_source(args)
            if args.action == "source"
            else check_installed(args)
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
