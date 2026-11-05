"""Authored compilation contexts; no configured compiler or analyzer executes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis_inputs import load_inputs
from quality_inventory import SourceFile


class AnalysisInputs(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.build = self.root / "build"
        self.build.mkdir()
        (self.root / "src").mkdir()
        self.source = self.root / "src/with spaces λ.cpp"
        self.source.write_text("// Authored source, not compiled.\n")
        self.compiler = self.root / "authored compiler"
        self.compiler.write_bytes(b"NOT EXECUTABLE")
        self.cache = {"CMAKE_HOME_DIRECTORY": str(self.root), "CMAKE_GENERATOR": "Ninja",
            "CMAKE_BUILD_TYPE": "Debug", "CMAKE_UNITY_BUILD": "OFF",
            "CMAKE_EXPORT_COMPILE_COMMANDS": "ON", "CMAKE_CXX_COMPILER": str(self.compiler),
            "RNS8_ENABLE_HIP": "OFF", "RNS8_AMDGPU_TARGETS": "gfx1100"}
        self.targets = dict(schema="rns8_first_party_analysis_targets_v1", targets=[
            dict(name=name, configuration="Debug", sources=[str(self.source)])
            for name in ("rns8_static", "rns8_shared")])
        self.database = [dict(directory=str(self.build), file=str(self.source),
            arguments=[str(self.compiler), "-D" + name, "-Ipath with spaces", '-DQUOTED="bytes"',
                       "-c", str(self.source), "-o", str(self.build / (name + ".obj"))])
            for name in ("static", "shared")]
        self.hip = dict(schema="rns8_custom_hip_compile_commands_v1", commands=[])
        self.files = (SourceFile(self.source.relative_to(self.root).as_posix(), "cpp"),)

    def write(self):
        (self.build / "CMakeCache.txt").write_text("\n".join(key + ":STRING=" + value for key, value in self.cache.items()))
        for name, value in (("compile_commands.json", self.database),
                            ("rns8-analysis-targets-Debug.json", self.targets),
                            ("rns8-hip-compile-commands-Debug.json", self.hip)):
            (self.build / name).write_text(json.dumps(value))

    def read(self):
        self.write()
        with patch("analysis_inputs.inventory", return_value=self.files):
            return load_inputs(self.build, root=self.root)

    def test_static_shared_contexts_are_distinct_without_requoting_or_mutable_aliases(self):
        value = self.read()
        self.assertEqual(len(value.contexts), 2)
        self.assertNotEqual(value.contexts[0].identity, value.contexts[1].identity)
        self.assertEqual(value.contexts[0].native_record(), self.database[0])
        altered = value.contexts[0].native_record()
        altered["arguments"].append("MUTATION")
        self.assertEqual(value.contexts[0].native_record(), self.database[0])
        self.assertEqual(set(value.contexts[0].registered_targets), {"rns8_static", "rns8_shared"})

    def test_native_shell_command_is_preserved_not_reparsed_as_posix_argv(self):
        for index, row in enumerate(self.database):
            row.pop("arguments")
            row["command"] = '"C:\\compiler path\\cl.exe" /DQUOTED="a b" /Fo' + str(index) + ".obj /c source.cpp"
        value = self.read()
        self.assertEqual(value.contexts[0].native_record(), self.database[0])

    def test_omitted_duplicate_or_unregistered_context_cannot_shrink_the_denominator(self):
        original = copy.deepcopy(self.database)
        for mutation in ("missing", "duplicate", "unexpected"):
            self.database = copy.deepcopy(original)
            if mutation == "missing":
                self.database.pop()
            elif mutation == "duplicate":
                self.database[1] = copy.deepcopy(self.database[0])
            else:
                self.targets["targets"] = []
            with self.assertRaises(ValueError):
                self.read()

    def test_unsupported_generator_unity_wrong_root_and_missing_export_are_rejected(self):
        for key, value in (("CMAKE_GENERATOR", "Visual Studio 18 2026"), ("CMAKE_UNITY_BUILD", "ON"),
                           ("CMAKE_HOME_DIRECTORY", str(self.root.parent)), ("CMAKE_EXPORT_COMPILE_COMMANDS", "OFF")):
            previous = self.cache[key]
            self.cache[key] = value
            with self.assertRaises(ValueError):
                self.read()
            self.cache[key] = previous

    def test_custom_hip_records_keep_targets_environment_and_generated_requirements(self):
        source = self.root / "src/real.hip"
        source.write_text("// Authored HIP source, never compiled.\n")
        self.files += (SourceFile("src/real.hip", "cpp"),)
        self.cache["RNS8_ENABLE_HIP"] = "ON"
        row = dict(file=str(source), directory=str(self.build), output=str(self.build / "real.obj"),
            configuration="Debug", arguments=[str(self.compiler), "--offload-arch=gfx1100", "-c", str(source)],
            targets=["gfx1100"], environment=["HIP_PATH=SDK with spaces", "EXPLICIT=a;b"],
            generated_inputs=[str(self.build / "generated/Debug/manifest.hpp")])
        self.hip["commands"] = [row]
        value = self.read()
        self.assertEqual(value.contexts[-1].environment, (("HIP_PATH", "SDK with spaces"), ("EXPLICIT", "a;b")))
        self.assertEqual(value.contexts[-1].targets, ("gfx1100",))
        self.assertFalse(value.contexts[-1].generated_inputs[0].exists())
        row["targets"] = ["gfx942"]
        with self.assertRaisesRegex(ValueError, "targets"):
            self.read()


if __name__ == "__main__":
    unittest.main()
