"""Generate immutable artifact source identity; invoked by every build, outside timing."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from json_evidence import load_evidence_json

TRUE_OPTIONS = frozenset(('ON', 'TRUE', '1', 'YES', 'Y'))
FALSE_OPTIONS = frozenset(('OFF', 'FALSE', '0', 'NO', 'N', ''))

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


def hexadecimal(value, length):
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def configuration_options(configuration, strict=False):
    options = {}
    for line in configuration.splitlines():
        key, separator, value = line.partition('=')
        if separator:
            if not key or key in options:
                raise ValueError('duplicate or empty manifest configuration key')
            options[key] = value
        elif strict and line.strip():
            raise ValueError('manifest configuration line has no value')
    return options


def dependency_file_identity(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'configured dependency file is missing: {path}')
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
    return dict(path=str(path.resolve()), sha256=digest.hexdigest(), bytes=size)


def dependency_content_identity(options, trees, files, receipts):
    # Paths remain recorded in configuration/entries; the content record binds
    # selected bytes and ABI inputs without making tree relocation a content edit.
    selection = {key: value for key, value in options.items()
                 if not key.startswith(('dependency_tree.', 'dependency_file.', 'dependency_receipt.'))
                 and key != 'vcpkg_installed'}
    return dict(schema='rns8_dependency_content_v1',
        resolution_kind=options.get('dependency_resolution_kind', 'unknown'), selection=selection,
        trees={name: {key: entry[key] for key in ('sha256', 'files', 'bytes')} for name, entry in trees.items()},
        files={name: {key: entry[key] for key in ('sha256', 'bytes')} for name, entry in files.items()},
        receipts={name: {key: entry[key] for key in ('sha256', 'bytes')} for name, entry in receipts.items()})


def dependency_content_digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def _content_identity_valid(manifest, options, diagnostic):
    groups = {}
    for field, prefix in (('external_files', 'dependency_file.'), ('dependency_receipts', 'dependency_receipt.')):
        entries = manifest.get(field)
        declared = {key.removeprefix(prefix) for key in options if key.startswith(prefix)}
        if not isinstance(entries, dict) or set(entries) != declared:
            return False
        for name, entry in entries.items():
            if (not name or not options[prefix + name] or not isinstance(entry, dict) or
                    set(entry) != {'path', 'sha256', 'bytes'} or not isinstance(entry['path'], str) or not entry['path'] or
                    not hexadecimal(entry['sha256'], 64) or type(entry['bytes']) is not int or entry['bytes'] <= 0):
                return False
        groups[field] = entries
    expected = dependency_content_identity(options, manifest['external_dependencies'],
        groups['external_files'], groups['dependency_receipts'])
    try:
        if (not isinstance(manifest.get('dependency_identity'), dict) or
                dependency_content_digest(manifest['dependency_identity']) != dependency_content_digest(expected) or
                manifest.get('dependency_digest') != dependency_content_digest(expected)):
            return False
    except (TypeError, ValueError):
        return False
    # Diagnostic shape is distinct from qualification: missing policy/resolution
    # evidence remains visible and cannot be repaired with an invented digest.
    if diagnostic:
        return expected['resolution_kind'] in ('unknown', 'native', 'vcpkg')
    kind, receipts, files = expected['resolution_kind'], expected['receipts'], expected['files']
    trees = expected['trees']
    if kind not in ('native', 'vcpkg') or 'BOOST' not in trees or not any(name.startswith('JSON_') for name in trees):
        return False
    if not options.get('compiler') or not options.get('compiler_path') or 'flags' not in options:
        return False
    for option in ('hip', 'hipblaslt', 'ck', 'rocwmma', 'rccl', 'rocshmem', 'distributed_mpi', 'gmp', 'flint', 'garner_table'):
        if options.get(option, 'OFF').upper() not in TRUE_OPTIONS | FALSE_OPTIONS:
            return False
    if options.get('hip', 'OFF').upper() in TRUE_OPTIONS:
        if not all(options.get(key) for key in ('hip_compiler', 'hip_version', 'targets')):
            return False
    for name, key in (('POLICY', 'dependency_policy_sha256'), ('BUILD_TOOLS', 'bootstrap_requirements_sha256')):
        if name not in receipts or not hexadecimal(options.get(key), 64) or receipts[name]['sha256'] != options[key]:
            return False
    if not hexadecimal(options.get('dependency_resolution_sha256'), 64):
        return False
    if kind == 'vcpkg' and 'VCPKG_STATUS' not in receipts:
        return False
    for option, required_trees, required_files in (
            ('hip', ('HIP',), ('HIP_RUNTIME', 'HIP_COMPILER')),
            ('hipblaslt', ('HIPBLASLT',), ('HIPBLASLT_LINK', 'HIPBLASLT_RUNTIME')),
            ('rccl', ('RCCL',), ('RCCL',)), ('rocshmem', ('ROCSHMEM',), ('ROCSHMEM',)),
            ('gmp', (), ('GMP', 'GMP_HEADER')), ('flint', (), ('FLINT', 'FLINT_HEADER'))):
        if options.get(option, 'OFF').upper() in TRUE_OPTIONS:
            if any(name not in trees for name in required_trees) or any(name not in files for name in required_files):
                return False
    if options.get('distributed_mpi', 'OFF').upper() in TRUE_OPTIONS:
        if not any(name.startswith('MPI_') for name in trees) or not any(name.startswith('MPI_') for name in files):
            return False
    return True


def _manifest_provenance(manifest, backend, diagnostic):

    if not isinstance(manifest, dict) or type(manifest.get("version")) is not int or manifest["version"] not in (2, 3):
        return False
    version = manifest['version']
    scope = 'source-config-and-selected-dependency-content' if version == 3 else 'source-config-and-external-headers'
    if ((type(manifest.get('source_dirty')) is not bool or (manifest['source_dirty'] and not diagnostic)) or
            not hexadecimal(manifest.get("source_commit"), 40) or
            not hexadecimal(manifest.get("source_digest"), 64) or
            (not hexadecimal(manifest.get("dependency_digest"), 64) and not
             (diagnostic and version == 2 and manifest.get('dependency_digest') == 'unavailable')) or
            manifest.get("linkage") != "static-rns8" or
            manifest.get("identity_scope") != scope):
        return False
    configuration, dependencies = manifest.get("configuration"), manifest.get("external_dependencies")
    if not isinstance(configuration, str) or not configuration or not isinstance(dependencies, dict):
        return False
    try:
        options = configuration_options(configuration, strict=version == 3)
    except ValueError:
        return False
    if not diagnostic and options.get("garner_table", "OFF").upper() in TRUE_OPTIONS:
        return False
    declared = {key.removeprefix("dependency_tree.") for key in options if key.startswith("dependency_tree.")}
    if set(dependencies) != declared:
        return False
    selected = {"rocwmma": "ROCWMMA", "ck": "CK", "hipblaslt": "HIPBLASLT"}.get(backend)
    if selected and selected not in declared:
        return False
    if version == 3:
        if backend in ('hip-direct', 'amdgpu-builtins', 'rocwmma', 'ck', 'hipblaslt') and options.get('hip', 'OFF').upper() not in TRUE_OPTIONS:
            return False
        option = 'builtins' if backend == 'amdgpu-builtins' else backend if selected else None
        if option and options.get(option, 'OFF').upper() not in TRUE_OPTIONS:
            return False
    for dependency in ("ROCWMMA", "CK", "HIPBLASLT"):
        if options.get(dependency.lower(), "OFF").upper() in TRUE_OPTIONS and dependency not in declared:
            return False
    for name, entry in dependencies.items():
        if (not name or not isinstance(entry, dict) or not isinstance(entry.get("root"), str) or not entry["root"] or
                not options.get("dependency_tree." + name) or not hexadecimal(entry.get("sha256"), 64) or
                type(entry.get("files")) is not int or entry["files"] <= 0 or
                type(entry.get("bytes")) is not int or entry["bytes"] < 0):
            return False
    return version == 2 or _content_identity_valid(manifest, options, diagnostic)


def manifest_provenance_complete(manifest, backend=None):
    """Qualifying immutable content evidence; schema2 retains its legacy contract."""
    return _manifest_provenance(manifest, backend, False)


def manifest_diagnostic_shape_valid(manifest, backend=None):
    """Validate diagnostic identity without asserting clean/complete qualification."""
    return _manifest_provenance(manifest, backend, True)


def manifest_dependencies_compatible(manifests):
    if not manifests or not all(manifest_provenance_complete(m) for m in manifests):
        return False
    if len({m['version'] for m in manifests}) != 1:
        return False # no implicit migration between legacy status and content digests
    if manifests[0]['version'] == 2 and len({m["dependency_digest"] for m in manifests}) > 1:
        return False
    seen = {}
    for manifest in manifests:
        for name, entry in manifest["external_dependencies"].items():
            identity = (entry["sha256"], entry["files"], entry["bytes"])
            if name in seen and seen[name] != identity:
                return False
            seen[name] = identity
    if manifests[0]['version'] == 3:
        if len({m['dependency_identity']['resolution_kind'] for m in manifests}) != 1:
            return False
        # Optional backend sets can differ. Every shared selected dependency and
        # policy/package receipt must agree; each full digest still binds its set.
        seen = {}
        for manifest in manifests:
            for group in ('files', 'receipts'):
                for name, entry in manifest['dependency_identity'][group].items():
                    key, identity = (group, name), (entry['sha256'], entry['bytes'])
                    if key in seen and seen[key] != identity:
                        return False
                    seen[key] = identity
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
    options = configuration_options(configuration, strict=True)
    groups = {}
    for field, prefix, read in (('external_dependencies', 'dependency_tree.', dependency_tree_identity),
            ('external_files', 'dependency_file.', dependency_file_identity),
            ('dependency_receipts', 'dependency_receipt.', dependency_file_identity)):
        groups[field] = {}
        for key, value in options.items():
            if key.startswith(prefix):
                name = key.removeprefix(prefix)
                if not name or not value:
                    raise ValueError(f'invalid dependency declaration: {key}')
                groups[field][name] = read(value)
    identity = dependency_content_identity(options, groups['external_dependencies'],
        groups['external_files'], groups['dependency_receipts'])
    return dict(version=3, source_commit=commit, source_dirty=dirty,
                source_digest=digest.hexdigest(), configuration=configuration,
                dependency_digest=dependency_content_digest(identity), dependency_identity=identity,
                **groups, linkage="static-rns8", identity_scope="source-config-and-selected-dependency-content")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--header", type=Path)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument('--check-manifest', type=Path)
    checks.add_argument('--check-build', type=Path, help='check the unique generated configuration manifest before device work')
    parser.add_argument('--backend', action='append', default=[], choices=(
        'cpu-reference', 'hip-direct', 'amdgpu-builtins', 'ck', 'rocwmma', 'hipblaslt'))
    args = parser.parse_args(argv)
    if args.check_manifest or args.check_build:
        if any((args.source, args.config, args.header)):
            parser.error('manifest checks cannot be combined with generation inputs')
        try:
            path = args.check_manifest
            if args.check_build:
                generated = args.check_build / 'generated'
                candidates = sorted(generated.glob('*/rns8_build_manifest.json'))
                unconfigured = generated / 'rns8_build_manifest.json'
                if unconfigured.is_file():
                    candidates.append(unconfigured)
                if len(candidates) != 1:
                    raise ValueError('expected exactly one built configuration manifest; use --check-manifest for an explicit configuration')
                path = candidates[0]
            manifest = load_evidence_json(path)
            if not all(manifest_provenance_complete(manifest, backend) for backend in (args.backend or [None])):
                raise ValueError('manifest has incomplete, dirty, inconsistent or backend-incompatible provenance')
            print(json.dumps(dict(schema='rns8_build_manifest_admission_v1', manifest=str(path),
                source_commit=manifest['source_commit'], dependency_digest=manifest['dependency_digest'],
                backends=args.backend, provenance_complete=True, device_work_performed=False, promotion_eligible=False)))
            return 0
        except (OSError, ValueError, TypeError) as error:
            print('manifest admission failed: ' + str(error), file=sys.stderr)
            return 1
    if not all((args.source, args.config, args.header)) or args.backend:
        parser.error('generation requires --source, --config and --header; --backend applies only to checks')
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
