#!/usr/bin/env python3
"""Offline dependency-policy, provenance and resolved-inventory fixtures."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from check_dependencies_lib import version_policy as p

ROOT = Path(__file__).resolve().parents[1]


class Policy(unittest.TestCase):
    def setUp(self):
        self.policy = p.load_policy(ROOT / "metadata/dependencies.json")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for filename in ("vcpkg.json", ".python-version", "scripts/requirements-build-tools.txt", ".github/workflows/ci.yml", ".github/workflows/perf-regression.yml", "metadata/dependencies.json"):
            target = self.root / filename; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / filename, target)

    def test_current_repository_is_consistent_but_not_qualified(self):
        self.assertEqual(p.check_repository(ROOT, self.policy), [])
        self.assertFalse(self.policy["promotion_allowed"])
        self.assertEqual(self.policy["status"], "candidate_pending_qualification")

    def test_inventory_only_components_are_not_install_requirements(self):
        inventory = [item for item in self.policy["components"] if item["role"] == "inventory_only"]
        self.assertTrue({"numpy", "pandas", "scipy", "PyYAML"}.issubset({item["name"] for item in inventory}))
        self.assertTrue(all(item["selected"] is None for item in inventory))

    def test_sdk_candidate_keeps_prior_source_fallback_pins(self):
        selected = {item["name"]: item for item in self.policy["components"]}
        self.assertEqual(selected["ck"]["selected"]["version"], "1.1.0")
        self.assertEqual(selected["rocwmma"]["selected"]["version"], "2.1.0")
        self.assertEqual(self.policy["gpu_sdk_candidate"]["status"], "local_release_validated_broader_platforms_pending")
        self.assertTrue(self.policy["gpu_sdk_candidate"]["installed"])
        self.assertFalse(self.policy["gpu_sdk_candidate"]["qualified"])
        invalid = deepcopy(self.policy)
        del next(item for item in invalid["components"] if item["name"] == "ck")["exception"]
        self.assertTrue(p.validate_policy(invalid))

    def test_malformed_policy_records_fail_closed(self):
        mutations = [lambda d: d.update(schema="unsupported"), lambda d: d.update(promotion_allowed=True),
            lambda d: d["components"].append(deepcopy(d["components"][0])),
            lambda d: d["components"][0].update(role=[]),
            lambda d: d["components"][0]["latest_stable"].update(source=None),
            lambda d: d["components"][0]["latest_stable"].update(version="3.16.0rc1"),
            lambda d: d["components"][0]["selected"].update(version="latest"),
            lambda d: d["components"][0].update(selected=None),
            lambda d: d["vcpkg"].update(baseline="main"),
            lambda d: d["vcpkg"]["ports"]["catch2"].update(manifest_sha256=None),
            lambda d: d["vcpkg"]["ports"]["catch2"].update(port_version=-1),
            lambda d: d["vcpkg"]["ports"]["catch2"].update(dependencies=[{"name": []}]),
            lambda d: d["bootstrap"]["artifacts"][0].update(url="http://untrusted.invalid/wheel"),
            lambda d: d["bootstrap"]["artifacts"][0].update(sha256="bad"),
            lambda d: d["bootstrap"].update(artifacts=None)]
        for mutate in mutations:
            data = deepcopy(self.policy); mutate(data)
            with self.subTest(mutation=mutate): self.assertTrue(p.validate_policy(data))

    def test_declared_closure_preserves_host_and_platform_conditions(self):
        ports = self.policy["vcpkg"]["ports"]
        self.assertGreater(len(ports), 20)
        self.assertIn({"name": "pthreads", "platform": "windows"}, ports["flint"]["dependencies"])
        self.assertTrue(any(isinstance(item, dict) and item.get("host") for item in ports["gmp"]["dependencies"]))
        bad = deepcopy(self.policy); del bad["vcpkg"]["ports"]["pthreads"]
        self.assertTrue(p.validate_policy(bad))

    def test_manifest_baseline_and_port_revision_drift(self):
        manifest = json.loads((self.root / "vcpkg.json").read_text())
        for mutate in (lambda d: d.update({"builtin-baseline": "a" * 40}),
                       lambda d: d["overrides"][0].update(version="0.0.0"),
                       lambda d: d["overrides"][-1].update({"port-version": 99})):
            changed = deepcopy(manifest); mutate(changed)
            (self.root / "vcpkg.json").write_text(json.dumps(changed))
            self.assertTrue(p.check_repository(self.root, self.policy))

    def test_requirement_version_and_hash_tampering(self):
        path = self.root / "scripts/requirements-build-tools.txt"; original = path.read_text()
        for changed in (original.replace("cmake==4.4.4", "cmake>=3.22,<4"),
                        original.replace(self.policy["bootstrap"]["artifacts"][0]["sha256"], "0" * 64)):
            path.write_text(changed)
            self.assertTrue(p.check_repository(self.root, self.policy))

    def test_action_and_python_selection_drift(self):
        path = self.root / ".github/workflows/ci.yml"
        path.write_text(path.read_text().replace("3d3c42e5aac5ba805825da76410c181273ba90b1", "v7"))
        self.assertTrue(p.check_repository(self.root, self.policy))
        (self.root / ".python-version").write_text("3.10\n")
        self.assertTrue(any(".python-version" in error for error in p.check_repository(self.root, self.policy)))

    def status(self, extra=""):
        root = self.root / "installed with spaces"
        metadata = root / "vcpkg"; (metadata / "info").mkdir(parents=True, exist_ok=True)
        (metadata / "status").write_text("Package: gmp\nVersion: 6.3.0\nPort-Version: 5\nArchitecture: x64-windows\nAbi: fixture-abi\nStatus: install ok installed\nDepends: vcpkg-make:x64-windows\n\n"
            "Package: gmp\nVersion: 6.3.0\nPort-Version: 5\nArchitecture: x64-windows\nFeature: fat\nStatus: install ok installed\n\n"
            "Package: gmp\nVersion: 6.2.0\nArchitecture: arm64-windows\nStatus: install ok installed\n\n" + extra)
        (metadata / "info/gmp_6.3.0_x64-windows.list").write_text("x64-windows/include/gmp.h\n")
        return root

    def test_complete_installed_status_keeps_triplets_features_abis(self):
        receipt = p.installed_vcpkg_receipt(self.status())
        self.assertEqual(len(receipt["packages"]), 3)
        self.assertEqual({item["feature"] for item in receipt["packages"]}, {"core", "fat"})
        self.assertEqual({item["triplet"] for item in receipt["packages"]}, {"x64-windows", "arm64-windows"})
        self.assertFalse(receipt["runtime_abi_validated"])
        self.assertRegex(receipt["status_sha256"], "^[0-9a-f]{64}$")
        self.assertEqual(len(receipt["package_metadata_sha256"]), 1)

    def test_duplicate_installed_package_identity_is_rejected(self):
        root = self.status("Package: gmp\nVersion: 6.3.0\nArchitecture: x64-windows\nStatus: install ok installed\n")
        with self.assertRaisesRegex(ValueError, "duplicate installed"): p.installed_vcpkg_receipt(root)

    def test_missing_receipt_never_counts_as_installed(self):
        receipt = p.installed_vcpkg_receipt(self.root / "missing")
        self.assertEqual(receipt["status"], "unavailable"); self.assertEqual(receipt["packages"], [])

    def test_selected_triplet_port_revision_and_optional_feature_gate(self):
        installed = self.status()
        cache = self.root / "CMakeCache.txt"; cache.write_text("VCPKG_INSTALLED_DIR:PATH=" + str(installed))
        report = {"system": {"python": "3.15.1"}, "commands": {"cmake": {"detail": "cmake version 4.4.4"}},
                  "dependency_resolution": {"vcpkg": {"selected": {"paths": {"triplet": "x64-windows"}}}, "feature_policy": {"RNS8_ENABLE_GMP": True}}}
        receipt = p.evaluate_report(self.root, self.policy, report, cache)
        item = next(item for item in receipt["components"] if item["name"] == "gmp")
        self.assertEqual(item["observed_version"], "6.3.0"); self.assertEqual(item["observed_port_version"], 5)
        self.assertTrue(item["matches_selected_pin"]); self.assertTrue(item["required_for_selected_configuration"])
        self.assertFalse(receipt["promotion_allowed"])
        changed = deepcopy(self.policy); changed["vcpkg"]["ports"]["gmp"]["port_version"] = 6
        item = next(item for item in p.evaluate_report(self.root, changed, report, cache)["components"] if item["name"] == "gmp")
        self.assertTrue(item["matches_selected_version"]); self.assertFalse(item["matches_selected_pin"])

    def test_offline_cli_check_executes_no_installer(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools/check_dependencies.py"), "--check-lock", "--json"],
                                capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "consistent_candidate_not_qualified"); self.assertFalse(report["promotion_allowed"])

    def test_old_explicit_reference_root_cannot_borrow_new_manifest_version(self):
        installed = self.status()
        classic = self.root / "old classic install"; (classic / "vcpkg").mkdir(parents=True)
        (classic / "vcpkg/status").write_text("Package: gmp\nVersion: 6.3.0\nPort-Version: 3\nArchitecture: x64-windows\nStatus: install ok installed\n")
        prefix = classic / "x64-windows"; prefix.mkdir()
        cache = self.root / "CMakeCache.txt"; cache.write_text("VCPKG_INSTALLED_DIR:PATH=" + str(installed))
        report = {"dependency_resolution": {"vcpkg": {"selected": {"paths": {"triplet": "x64-windows"}}},
                  "optional_components": {"gmp": {"selected": {"canonical_root": str(prefix)}}},
                  "feature_policy": {"RNS8_ENABLE_GMP": True}}}
        receipt = p.evaluate_report(self.root, self.policy, report, cache)
        gmp = next(item for item in receipt["components"] if item["name"] == "gmp")
        self.assertEqual(gmp["observed_port_version"], 3)
        self.assertTrue(gmp["matches_selected_version"]); self.assertFalse(gmp["matches_selected_pin"])
        (classic / "vcpkg/status").unlink()
        gmp = next(item for item in p.evaluate_report(self.root, self.policy, report, cache)["components"] if item["name"] == "gmp")
        self.assertIsNone(gmp["matches_selected_pin"])

    def test_selected_source_version_and_revision_are_distinct_gates(self):
        source = self.root / "selected ck source"; source.mkdir()
        (source / "CMakeLists.txt").write_text("set(version 1.1.0)\n")
        report = {"dependency_resolution": {"optional_components": {"ck": {"selected": {
            "canonical_root": str(source), "version": None, "revision": "d9272218c4c59a58e41d3d346362cdaa707c30ce"}}},
            "feature_policy": {"RNS8_ENABLE_CK": True}}}
        receipt = p.evaluate_report(self.root, self.policy, report)
        ck = next(item for item in receipt["components"] if item["name"] == "ck")
        self.assertTrue(ck["matches_selected_pin"]); self.assertTrue(ck["required_for_selected_configuration"])
        report["dependency_resolution"]["optional_components"]["ck"]["selected"]["revision"] = "0" * 40
        ck = next(item for item in p.evaluate_report(self.root, self.policy, report)["components"] if item["name"] == "ck")
        self.assertTrue(ck["matches_selected_version"]); self.assertFalse(ck["matches_selected_pin"])

    def test_windows_git_security_revision_is_not_dropped(self):
        report = {"commands": {"git": {"detail": "git version 2.56.0.windows.1"}},
                  "dependency_resolution": {"request": {"host": "Windows"}}}
        git = next(item for item in p.evaluate_report(self.root, self.policy, report)["components"] if item["name"] == "git")
        self.assertEqual(git["selected"]["version"], "2.56.0.windows.3")
        self.assertFalse(git["matches_selected_pin"])
        report["commands"]["git"]["detail"] = "git version 2.56.0.windows.3"
        git = next(item for item in p.evaluate_report(self.root, self.policy, report)["components"] if item["name"] == "git")
        self.assertTrue(git["matches_selected_pin"])


if __name__ == "__main__": unittest.main()
