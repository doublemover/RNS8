"""Provenance regressions use an isolated repository, never the user's checkout."""

import json
import contextlib
import copy
import hashlib
import io
import os
import sys
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from build_manifest import (
    generate,
    dependency_tree_identity,
    manifest_provenance_complete,
    manifest_dependencies_compatible,
    manifest_diagnostic_shape_valid,
    main as manifest_main,
)
from json_evidence import load_evidence_json


def dependency_fixture(root, kind="native"):
    """Real content hashes of synthetic files; no dependency binary is executed."""
    source = root / "project"
    source.mkdir()
    (source / "CMakeLists.txt").write_text("project(fixture)\n", encoding="utf-8")
    external = root / "native packages with spaces"
    external.mkdir()
    options = {
        "config": "Release",
        "hip": "ON",
        "ck": "OFF",
        "dependency_resolution_kind": kind,
        "vcpkg_installed": "",
        "dependency_resolution_sha256": "e" * 64,
        "compiler": "synthetic compiler 1",
        "compiler_path": str(external / "HIP_COMPILER.synthetic"),
        "flags": "-O2",
        "hip_compiler": str(external / "HIP_COMPILER.synthetic"),
        "hip_version": "synthetic compiler 1",
        "targets": "gfx1100",
    }
    for name in ("BOOST", "JSON_1", "HIP"):
        directory = external / name
        directory.mkdir()
        (directory / "fixture.hpp").write_text("fixture header " + name, encoding="utf-8")
        options["dependency_tree." + name] = str(directory)
    for name in ("HIP_RUNTIME", "HIP_COMPILER"):
        file = external / (name + ".synthetic")
        file.write_text("identity fixture; never loaded or executed " + name, encoding="utf-8")
        options["dependency_file." + name] = str(file)
    for name, key in (
        ("POLICY", "dependency_policy_sha256"),
        ("BUILD_TOOLS", "bootstrap_requirements_sha256"),
    ):
        file = external / (name + ".receipt")
        file.write_text("synthetic locked receipt " + name, encoding="utf-8")
        options["dependency_receipt." + name] = str(file)
        options[key] = hashlib.sha256(file.read_bytes()).hexdigest()
    if kind == "vcpkg":
        status = external / "vcpkg" / "status"
        status.parent.mkdir()
        status.write_text("Package: fixture\nVersion: 1\n", encoding="utf-8")
        options["dependency_receipt.VCPKG_STATUS"] = str(status)
    config = root / "configuration.txt"
    config.write_text(
        "".join(f"{key}={value}\n" for key, value in options.items()), encoding="utf-8"
    )
    return source, config, options


def clean_fixture_generate(source, config):
    def git(command, **kwargs):
        return b"c" * 40 if "rev-parse" in command else b""

    with patch("build_manifest.subprocess.check_output", side_effect=git):
        return generate(source, config)


def dependency_scratch():
    scratch = Path(__file__).resolve().parents[1] / "temp"
    scratch.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="manifest-content-", dir=scratch)


class BuildManifestTest(unittest.TestCase):
    def test_native_identity_ignores_ambient_vcpkg_and_supports_cpu_only_selection(self):
        with dependency_scratch() as directory:
            root = Path(directory)
            source, config, options = dependency_fixture(root)
            first = clean_fixture_generate(source, config)
            ambient = root / "ambient"
            (ambient / "vcpkg").mkdir(parents=True)
            (ambient / "vcpkg" / "status").write_text(
                "must never become a native receipt", encoding="utf-8"
            )
            with contextlib.chdir(ambient):
                self.assertEqual(first, clean_fixture_generate(source, config))
            for key in (
                "dependency_tree.HIP",
                "dependency_file.HIP_RUNTIME",
                "dependency_file.HIP_COMPILER",
            ):
                options.pop(key)
            options["hip"] = "OFF"
            config.write_text(
                "".join(f"{key}={value}\n" for key, value in options.items()), encoding="utf-8"
            )
            cpu = clean_fixture_generate(source, config)
            self.assertTrue(manifest_provenance_complete(cpu, "cpu-reference"))
            self.assertFalse(manifest_provenance_complete(cpu, "hip-direct"))
            self.assertTrue(manifest_dependencies_compatible([first, cpu]))
            options.pop("dependency_receipt.POLICY")
            config.write_text(
                "".join(f"{key}={value}\n" for key, value in options.items()), encoding="utf-8"
            )
            missing = clean_fixture_generate(source, config)
            self.assertFalse(manifest_provenance_complete(missing))
            self.assertTrue(manifest_diagnostic_shape_valid(missing))

    def test_host_only_manifest_admission_is_strict_and_requires_unique_build_configuration(self):
        with dependency_scratch() as directory:
            root = Path(directory)
            source, config, _ = dependency_fixture(root)
            manifest = clean_fixture_generate(source, config)
            release = root / "build" / "generated" / "Release" / "rns8_build_manifest.json"
            release.parent.mkdir(parents=True)
            release.write_text(json.dumps(manifest), encoding="utf-8")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertEqual(
                    manifest_main(
                        ["--check-build", str(root / "build"), "--backend", "hip-direct"]
                    ),
                    0,
                )
            receipt = json.loads(out.getvalue())
            self.assertTrue(receipt["provenance_complete"])
            self.assertFalse(receipt["device_work_performed"])
            self.assertFalse(receipt["promotion_eligible"])
            debug = release.parents[1] / "Debug" / "rns8_build_manifest.json"
            debug.parent.mkdir()
            debug.write_text(json.dumps(manifest), encoding="utf-8")
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(manifest_main(["--check-build", str(root / "build")]), 1)
                self.assertEqual(
                    manifest_main(["--check-manifest", str(release), "--backend", "hip-direct"]), 0
                )
                self.assertEqual(
                    manifest_main(["--check-manifest", str(release), "--backend", "ck"]), 1
                )
                release.write_text(json.dumps(dict(manifest, source_dirty=True)), encoding="utf-8")
                self.assertEqual(manifest_main(["--check-manifest", str(release)]), 1)
                with self.assertRaises(SystemExit):
                    manifest_main(["--source", str(source)])
                with self.assertRaises(SystemExit):
                    manifest_main(["--check-manifest", str(release), "--source", str(source)])
            for text in ('{"version":3,"version":2}', '{"version":NaN}', '{"version":Infinity}'):
                release.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_evidence_json(release)

    def test_generated_native_and_vcpkg_content_manifests_reach_real_collector_admission(self):
        from test_collect_native_calibration import case_for, capture_for
        from collect_native_calibration import validate_capture

        for kind in ("native", "vcpkg"):
            with self.subTest(kind=kind), dependency_scratch() as directory:
                source, config, _ = dependency_fixture(Path(directory), kind)
                manifest = clean_fixture_generate(source, config)
                self.assertEqual(manifest["version"], 3)
                self.assertEqual(manifest["dependency_identity"]["resolution_kind"], kind)
                self.assertEqual("VCPKG_STATUS" in manifest["dependency_receipts"], kind == "vcpkg")
                self.assertTrue(manifest_provenance_complete(manifest))
                self.assertFalse(manifest_provenance_complete(manifest, "ck"))
                case = case_for("wmma")
                capture = capture_for(case)
                capture["build_manifest"] = manifest
                self.assertEqual(
                    validate_capture(capture, case, "fixture-run", 3, 1)["build_manifest"], manifest
                )
                self.assertTrue(
                    manifest_dependencies_compatible([manifest, copy.deepcopy(manifest)])
                )

    def test_native_content_mutations_and_missing_selected_evidence_fail_closed(self):
        with dependency_scratch() as directory:
            source, config, options = dependency_fixture(Path(directory))
            first = clean_fixture_generate(source, config)
            header = Path(options["dependency_tree.BOOST"]) / "fixture.hpp"
            header.write_text("different selected Boost", encoding="utf-8")
            changed = clean_fixture_generate(source, config)
            self.assertEqual(first["source_digest"], changed["source_digest"])
            self.assertNotEqual(first["dependency_digest"], changed["dependency_digest"])
            self.assertFalse(manifest_dependencies_compatible([first, changed]))
            library = Path(options["dependency_file.HIP_RUNTIME"])
            library.write_text("changed selected runtime", encoding="utf-8")
            runtime = clean_fixture_generate(source, config)
            self.assertNotEqual(runtime["dependency_digest"], changed["dependency_digest"])
            self.assertFalse(manifest_dependencies_compatible([changed, runtime]))
            policy = Path(options["dependency_receipt.POLICY"])
            policy.write_text("new locked policy", encoding="utf-8")
            stale = clean_fixture_generate(source, config)
            self.assertFalse(manifest_provenance_complete(stale))
            self.assertTrue(manifest_diagnostic_shape_valid(stale))
            options["dependency_policy_sha256"] = hashlib.sha256(policy.read_bytes()).hexdigest()
            config.write_text(
                "".join(f"{key}={value}\n" for key, value in options.items()), encoding="utf-8"
            )
            refreshed = clean_fixture_generate(source, config)
            self.assertTrue(manifest_provenance_complete(refreshed))
            self.assertFalse(manifest_dependencies_compatible([runtime, refreshed]))
            library.unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                clean_fixture_generate(source, config)

    def test_content_tampering_diagnostics_and_legacy_migration_are_distinct(self):
        from test_capture_identity_support import synthetic_identity

        capture = {}
        synthetic_identity(capture)
        legacy = capture["build_manifest"]
        self.assertTrue(manifest_provenance_complete(legacy))
        with dependency_scratch() as directory:
            source, config, _ = dependency_fixture(Path(directory))
            manifest = clean_fixture_generate(source, config)
            self.assertFalse(manifest_dependencies_compatible([manifest, legacy]))
            for mutate in (
                lambda m: m.update(dependency_digest="a" * 64),
                lambda m: m["dependency_identity"]["trees"]["BOOST"].update(files=True),
                lambda m: m["dependency_receipts"].pop("POLICY"),
                lambda m: m["external_files"].pop("HIP_RUNTIME"),
                lambda m: m["external_dependencies"].pop("BOOST"),
                lambda m: m.update(configuration=m["configuration"] + "hip=OFF\n"),
                lambda m: m.update(identity_scope="source-config-and-external-headers"),
            ):
                bad = copy.deepcopy(manifest)
                mutate(bad)
                self.assertFalse(manifest_provenance_complete(bad))
                self.assertFalse(manifest_diagnostic_shape_valid(bad))
            for changes in (dict(source_dirty=True), dict(source_commit="unknown")):
                bad = dict(manifest, **changes)
                self.assertFalse(manifest_provenance_complete(bad))
            dirty = dict(manifest, source_dirty=True)
            self.assertTrue(manifest_diagnostic_shape_valid(dirty))
            legacy = dict(legacy, dependency_digest="unavailable")
            self.assertFalse(manifest_provenance_complete(legacy))
            self.assertTrue(manifest_diagnostic_shape_valid(legacy))
            options = config.read_text(encoding="utf-8").replace("ck=OFF", "ck=ON")
            ck = Path(directory) / "CK"
            ck.mkdir()
            (ck / "fixture.hpp").write_text("selected CK", encoding="utf-8")
            config.write_text(options + f"dependency_tree.CK={ck}\n", encoding="utf-8")
            selected = clean_fixture_generate(source, config)
            self.assertTrue(manifest_provenance_complete(selected, "ck"))
            self.assertNotEqual(selected["dependency_digest"], manifest["dependency_digest"])
            self.assertTrue(manifest_dependencies_compatible([manifest, selected]))

    def test_source_identity_changes_without_committing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL
                )

            git("init")
            (root / "src").mkdir()
            source = root / "src" / "a.cpp"
            source.write_text("int a=1;")
            (root / "CMakeLists.txt").write_text("project(test)")
            config = root / "configuration.txt"
            config.write_text("config=Release")
            git("add", ".")
            git(
                "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "A"
            )
            a = generate(root, config)
            self.assertFalse(a["source_dirty"])
            source.write_text("int a=2;")
            b = generate(root, config)
            self.assertEqual(a["source_commit"], b["source_commit"])
            self.assertTrue(b["source_dirty"])
            self.assertNotEqual(a["source_digest"], b["source_digest"])
            # Saved artifact metadata remains immutable when checkout changes.
            saved = json.dumps(a)
            git("add", ".")
            git(
                "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "B"
            )
            self.assertNotEqual(a["source_commit"], generate(root, config)["source_commit"])
            self.assertEqual(json.loads(saved), a)

    def test_external_header_edits_additions_and_deletions_change_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "project"
            source.mkdir()
            (source / "CMakeLists.txt").write_text("project(test)")
            headers = root / "external"
            headers.mkdir()
            header = headers / "vendor.hpp"
            header.write_text("int vendor=1;")
            config = source / "config.txt"
            config.write_text(f"dependency_tree.vendor={headers}\n")
            first = generate(source, config)
            header.write_text("int vendor=2;")
            second = generate(source, config)
            self.assertEqual(first["source_digest"], second["source_digest"])
            self.assertNotEqual(first["external_dependencies"], second["external_dependencies"])
            extra = headers / "extra.inc"
            extra.write_bytes(b"binary\x00header")
            third = generate(source, config)
            self.assertEqual(third["external_dependencies"]["vendor"]["files"], 2)
            self.assertNotEqual(second["external_dependencies"], third["external_dependencies"])
            extra.unlink()
            self.assertEqual(second, generate(source, config))
            config.write_text(
                f"dependency_tree.vendor={headers}\ndependency_tree.vendor={headers}\n"
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                generate(source, config)

    def test_dependency_identity_fails_closed_for_missing_empty_or_cyclic_trees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "missing"):
                dependency_tree_identity(root / "missing")
            with self.assertRaisesRegex(ValueError, "empty"):
                dependency_tree_identity(root)
            (root / "header.hpp").write_text("int x;")
            try:
                (root / "cycle").symlink_to(root, target_is_directory=True)
            except OSError:
                return  # Windows may not grant unprivileged symlink creation.
            with self.assertRaisesRegex(ValueError, "cycle"):
                dependency_tree_identity(root)

    def test_release_admission_requires_bound_external_dependencies(self):
        import copy
        from test_capture_identity_support import synthetic_identity

        capture = {}
        synthetic_identity(capture)
        manifest = capture["build_manifest"]
        self.assertTrue(manifest_provenance_complete(manifest, "ck"))
        missing = copy.deepcopy(manifest)
        missing["configuration"] = "ck=OFF"
        missing["external_dependencies"] = {}
        self.assertTrue(manifest_provenance_complete(missing, "cpu-reference"))
        self.assertFalse(manifest_provenance_complete(missing, "ck"))
        self.assertTrue(manifest_dependencies_compatible([manifest, copy.deepcopy(manifest)]))
        changed = copy.deepcopy(manifest)
        changed["external_dependencies"]["CK"]["sha256"] = "f" * 64
        self.assertTrue(manifest_provenance_complete(changed))
        self.assertFalse(manifest_dependencies_compatible([manifest, changed]))
        for mutate in (
            lambda m: m.update(version=1),
            lambda m: m["external_dependencies"].pop("CK"),
            lambda m: m["external_dependencies"]["CK"].update(files=True),
            lambda m: m["external_dependencies"]["CK"].update(sha256="z" * 64),
            lambda m: m.update(configuration="ck=ON"),
        ):
            changed = copy.deepcopy(manifest)
            mutate(changed)
            self.assertFalse(manifest_provenance_complete(changed))

    @unittest.skipUnless(os.name == "nt", "COFF link identity requires MSVC")
    def test_mixed_object_link_is_rejected(self):
        from msvc_env import run_in_msvc_environment

        scratch = Path(__file__).resolve().parents[1] / "temp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            (root / "CMakeLists.txt").write_text("project(witness)")
            config = root / "config.txt"
            generator = Path(__file__).with_name("build_manifest.py").resolve()
            for name in ("a", "b"):
                config.write_text("config=" + name)
                subprocess.run(
                    [
                        sys.executable,
                        str(generator),
                        "--source",
                        str(root),
                        "--config",
                        str(config),
                        "--header",
                        str(root / (name + ".hpp")),
                    ],
                    check=True,
                )
            (root / "a.cpp").write_text('#include "a.hpp"\nint f(){return 1;}')
            (root / "b.cpp").write_text('#include "b.hpp"\nint f(); int main(){return f();}')
            result = run_in_msvc_environment(
                ["cl", "/nologo", "/EHsc", "a.cpp", "b.cpp", "/Fe:mixed.exe"],
                cwd=root,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("LNK2038", result.stdout)


if __name__ == "__main__":
    unittest.main()
