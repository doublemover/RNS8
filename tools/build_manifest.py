"""Generate immutable artifact source identity; invoked by every build, outside timing."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


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
    for line in configuration.splitlines():
        if line.startswith("vcpkg_installed="):
            status = Path(line.partition("=")[2]) / "vcpkg" / "status"
            if status.is_file():
                dependency_digest = hashlib.sha256(status.read_bytes()).hexdigest()
    return dict(version=1, source_commit=commit, source_dirty=dirty,
                source_digest=digest.hexdigest(), configuration=configuration, dependency_digest=dependency_digest,
                linkage="static-rns8", identity_scope="source-and-config")


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
