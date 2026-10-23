"""Generate immutable artifact source identity; invoked by every build, outside timing."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def dependency_tree_identity(root):
    """Hash actual include contents, following compiler-visible directory links.

    No timestamp cache: edits to an external checkout must invalidate the build
    even when the RNS8 checkout and package-manager version records are unchanged.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"configured dependency tree is missing: {root}")
    digest = hashlib.sha256()
    files, byte_count = 0, 0

    def visit(directory, ancestors):
        nonlocal files, byte_count
        resolved = directory.resolve(strict=True)
        if resolved in ancestors:
            raise ValueError(f"dependency directory cycle: {directory}")
        ancestors = ancestors | {resolved}
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if path.name in (".git", "__pycache__"):
                continue
            if path.is_dir():
                visit(path, ancestors)
            elif path.is_file():
                with path.open("rb") as stream:
                    content = hashlib.sha256()
                    size = 0
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        content.update(chunk)
                        size += len(chunk)
                    content_hash = content.hexdigest()
                record = [path.relative_to(root).as_posix(), size, content_hash]
                digest.update(json.dumps(record, ensure_ascii=True, separators=(",", ":")).encode() + b"\n")
                files += 1
                byte_count += size
            else:
                raise ValueError(f"unreadable dependency entry: {path}")

    visit(root, set())
    if not files:
        raise ValueError(f"configured dependency tree is empty: {root}")
    return dict(root=str(root.resolve()), sha256=digest.hexdigest(), files=files, bytes=byte_count)


def manifest_provenance_complete(manifest, backend=None):
    def hexadecimal(value, length):
        return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)

    if not isinstance(manifest, dict) or type(manifest.get("version")) is not int or manifest["version"] != 2:
        return False
    if (manifest.get("source_dirty") is not False or
            not hexadecimal(manifest.get("source_commit"), 40) or
            not hexadecimal(manifest.get("source_digest"), 64) or
            not hexadecimal(manifest.get("dependency_digest"), 64) or
            manifest.get("linkage") != "static-rns8" or
            manifest.get("identity_scope") != "source-config-and-external-headers"):
        return False
    configuration, dependencies = manifest.get("configuration"), manifest.get("external_dependencies")
    if not isinstance(configuration, str) or not configuration or not isinstance(dependencies, dict):
        return False
    options = {}
    for line in configuration.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            if key in options:
                return False
            options[key] = value
    if options.get("garner_table", "OFF").upper() in ("ON", "TRUE", "1", "YES"):
        return False
    declared = {key.removeprefix("dependency_tree.") for key in options if key.startswith("dependency_tree.")}
    if set(dependencies) != declared:
        return False
    selected = {"rocwmma": "ROCWMMA", "ck": "CK", "hipblaslt": "HIPBLASLT"}.get(backend)
    if selected and selected not in declared:
        return False
    for dependency in ("ROCWMMA", "CK", "HIPBLASLT"):
        if options.get(dependency.lower(), "OFF").upper() in ("ON", "TRUE", "1", "YES") and dependency not in declared:
            return False
    for name, entry in dependencies.items():
        if (not isinstance(entry, dict) or not isinstance(entry.get("root"), str) or not entry["root"] or
                not options.get("dependency_tree." + name) or not hexadecimal(entry.get("sha256"), 64) or
                type(entry.get("files")) is not int or entry["files"] <= 0 or
                type(entry.get("bytes")) is not int or entry["bytes"] < 0):
            return False
    return True


def manifest_dependencies_compatible(manifests):
    if not all(manifest_provenance_complete(m) for m in manifests):
        return False
    if len({m["dependency_digest"] for m in manifests}) > 1:
        return False
    seen = {}
    for manifest in manifests:
        for name, entry in manifest["external_dependencies"].items():
            identity = (entry["sha256"], entry["files"], entry["bytes"])
            if name in seen and seen[name] != identity:
                return False
            seen[name] = identity
    return True


def generate(source, config):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], stderr=subprocess.DEVNULL)
    try:
        commit = git("rev-parse", "HEAD").decode().strip()
        dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    digest = hashlib.sha256()
    # Include untracked implementation inputs as well as tracked files.
    paths = [source / "CMakeLists.txt"]
    for folder in ("src", "include", "cmake", "benchmarks", "tools"):
        paths.extend(p for p in (source / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    for path in sorted(paths):
        digest.update(path.relative_to(source).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    configuration = config.read_text(encoding="utf-8")
    dependency_digest = "unavailable"
    external_dependencies = {}
    for line in configuration.splitlines():
        if line.startswith("vcpkg_installed="):
            status = Path(line.partition("=")[2]) / "vcpkg" / "status"
            if status.is_file():
                dependency_digest = hashlib.sha256(status.read_bytes()).hexdigest()
        if line.startswith("dependency_tree."):
            key, separator, value = line.partition("=")
            name = key.removeprefix("dependency_tree.")
            if not separator or not value or not name or name in external_dependencies:
                raise ValueError(f"invalid or duplicate dependency tree: {line}")
            external_dependencies[name] = dependency_tree_identity(value)
    return dict(version=2, source_commit=commit, source_dirty=dirty,
                source_digest=digest.hexdigest(), configuration=configuration, dependency_digest=dependency_digest,
                external_dependencies=external_dependencies,
                linkage="static-rns8", identity_scope="source-config-and-external-headers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    args = parser.parse_args()
    manifest = generate(args.source, args.config)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    text = "#pragma once\n#define RNS8_COMPILED_COMMIT " + json.dumps(manifest["source_commit"]) + "\n"
    text += "#define RNS8_BUILD_MANIFEST_JSON " + json.dumps(encoded) + "\n"
    identity = hashlib.sha256(encoded.encode()).hexdigest()
    # COFF carries this record through static libraries. Mixing a stale host or
    # HIP object with a different source/configuration is a link error, even if
    # someone replaced an object with a newer timestamp outside the build graph.
    text += '#if defined(_MSC_VER) && !defined(__HIP_DEVICE_COMPILE__)\n'
    text += '#pragma detect_mismatch("rns8_build_identity", "' + identity + '")\n#endif\n'
    args.header.parent.mkdir(parents=True, exist_ok=True)
    if not args.header.exists() or args.header.read_text() != text:
        args.header.write_text(text)
    manifest_path = args.header.with_suffix(".json")
    if not manifest_path.exists() or manifest_path.read_text() != encoded:
        manifest_path.write_text(encoded)


if __name__ == "__main__":
    main()
