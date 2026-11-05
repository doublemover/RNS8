"""Source and loader controls for ordinary, independently owned contract cases."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PythonCaseStructure(unittest.TestCase):
    def test_production_schema_namespaces_are_explicit_and_import_in_fresh_processes(self):
        paths=sorted((ROOT/"tools/benchmark_schema").glob("*.py"))
        self.assertGreater(len(paths),10)
        for path in paths:
            tree=ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom):
                    self.assertFalse(any(alias.name=="*" for alias in node.names),str(path))
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                    self.assertNotIn(node.func.id,("globals","exec","eval"),str(path))
        for name in ("semantic_wrap64","semantic_finite","semantic_exact_wide","semantic_bounded",
                     "gpu_event_validation","reuse_timing","core"):
            result=subprocess.run([sys.executable,"-c","import importlib,sys; importlib.import_module(sys.argv[1])",
                "benchmark_schema."+name],cwd=ROOT/"tools",capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_every_migrated_case_retains_assertions_and_loads_independently(self):
        inventory = json.loads((ROOT / "metadata/python-contract-case-inventory.json").read_text())
        self.assertEqual(inventory["schema"], "rns8_python_contract_case_inventory_v1")
        self.assertEqual(len(inventory["cases"]), 11)
        identifiers = set()
        for row in inventory["cases"]:
            path = ROOT / "tools" / (row["module"] + ".py")
            source = ast.parse(path.read_text())
            self.assertGreaterEqual(sum(isinstance(node, ast.Assert) for node in ast.walk(source)),
                                    row.get("minimum_direct_assertions", row["original_assertions"]))
            references = {node.id for node in ast.walk(source) if isinstance(node, ast.Name)}
            self.assertTrue(set(row.get("contract_helpers", [])).issubset(references))
            module = importlib.import_module(row["module"])
            suite = unittest.defaultTestLoader.loadTestsFromModule(module)
            self.assertEqual(suite.countTestCases(), 1)
            cls, method = row["case"].split(".")
            identifier = getattr(module, cls)(method).id()
            self.assertNotIn(identifier, identifiers)
            identifiers.add(identifier)
        self.assertEqual(len(identifiers), 11)

    def test_migrated_sources_have_no_dynamic_execution_or_wildcard_globals(self):
        paths = list((ROOT / "tools").glob("test_benchmark_schema*.py"))
        paths += list((ROOT / "tools").glob("test_benchmark_sweep*.py"))
        for path in paths:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, {"exec", "eval"}, str(path))
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn("*", {name.name for name in node.names}, str(path))
                    self.assertNotIn(node.module, {"test_benchmark_schema", "test_benchmark_sweep",
                                                   "test_benchmark_schema_support"}, str(path))

    def test_fixture_import_graph_is_acyclic(self):
        paths = list((ROOT / "tools").glob("test_benchmark_schema_support_*.py"))
        names = {path.stem for path in paths}
        graph = {path.stem: {node.module for node in ast.walk(ast.parse(path.read_text()))
                            if isinstance(node, ast.ImportFrom) and node.module in names} for path in paths}
        visited, active = set(), set()

        def visit(name):
            self.assertNotIn(name, active, "fixture dependency cycle: " + name)
            if name in visited:
                return
            active.add(name)
            for dependency in graph[name]:
                visit(dependency)
            active.remove(name)
            visited.add(name)

        for name in graph:
            visit(name)

    def test_sweep_cases_have_no_waived_behavioral_checks(self):
        for path in (ROOT / "tools").glob("test_benchmark_sweep_case_*.py"):
            tree = ast.parse(path.read_text())
            self.assertFalse(any(isinstance(node, ast.Pass) for node in ast.walk(tree)), str(path))

    def test_command_factories_do_not_share_mutable_namespaces(self):
        from test_benchmark_sweep_support import exact_command_args, scenario_command_args
        first, second = exact_command_args(), exact_command_args()
        first.backends.append("authored_mutation")
        self.assertEqual(second.backends, ["cpu", "hip-direct"])
        scenario = scenario_command_args()
        scenario.scenario.append("authored_mutation")
        self.assertEqual(scenario_command_args().scenario, ["repeated-b"])
        self.assertEqual(exact_command_args().semantics, ["exact_wide_signed"])


if __name__ == "__main__":
    unittest.main()
