"""Documentation checker regressions; no build, GPU execution or real network."""
import copy
from email.message import Message
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from check_documentation import (
    ROOT, anchors, check, command_errors, document_paths, external_status,
    links, local_target, prose, slug, template_errors,
)


class MarkdownTest(unittest.TestCase):
    def test_github_headings_unicode_formatting_duplicates_and_fences(self):
        text = "# Hello **world**!\n## Hello world!\n## `rns8_api` / output\n## 512³ → i64\n"
        text += "```cpp\n# hidden\n```\n~~~\n# hidden2\n~~~\nUnderlined title\n---\n<a id='explicit'></a>"
        self.assertEqual(anchors(text), {"hello-world", "hello-world-1", "rns8_api--output",
                                         "512³--i64", "underlined-title", "explicit"})
        self.assertEqual(slug("17.1 INT4"), "171-int4")
        self.assertNotIn("hidden", prose(text))

    def test_links_reference_definitions_balanced_urls_and_images(self):
        text = "[source](src/a.cpp) ![asset](<images/with space.png>)\n"
        text += "[paper](https://example.test/a_(b)) [named][id]\n[id]: target.md#section\nBare https://example.test/plain.\n"
        text += "[missing][oops]\n<a href='other.html#here'><img src='figure.svg'></a>\n"
        text += "```md\n[ignored](missing.md)\n```"
        targets = [target for _, target in links(text)]
        for expected in ("src/a.cpp", "images/with space.png", "https://example.test/a_(b)",
                         "target.md#section", "MISSING_REFERENCE:oops", "other.html#here", "figure.svg", "https://example.test/plain"):
            self.assertIn(expected, targets)
        self.assertNotIn("missing.md", targets)

    def test_inline_code_is_not_reference_link_syntax(self):
        text = "`[plane][column][K-word]` ``[ignored](missing.md)`` [real](README.md)"
        self.assertEqual([target for _, target in links(text)], ["README.md"])

    def test_html_links_and_explicit_anchors(self):
        text = '<h1 id="a">Title</h1><a name="old" href="../README.md#intro">Go</a><img src="img.png">'
        self.assertEqual(anchors(text, html=True), {"a", "old"})
        self.assertEqual([target for _, target in links(text, html=True)], ["../README.md#intro", "img.png"])

    def test_paths_spaces_percent_decoding_and_escape_rejection(self):
        source = ROOT / "docs/README.md"
        self.assertEqual(local_target(ROOT, source, "../README.md#intro"), (ROOT / "README.md", "intro"))
        self.assertEqual(local_target(ROOT, source, "a%20b.md#two%20words"), (ROOT / "docs/a b.md", "two words"))
        self.assertIsNone(local_target(ROOT, source, "https://example.test/a"))
        with self.assertRaises(ValueError):
            local_target(ROOT, source, "../../escape.md")


class RepositoryTest(unittest.TestCase):
    def test_inventory_covers_every_first_party_document(self):
        report = check()
        self.assertEqual(report["errors"], [])
        paths = document_paths(ROOT)
        self.assertIn("docs/archive/dashboard.html", paths)
        self.assertIn("src/backend_rocwmma/README.md", paths)
        self.assertIn(".github/ISSUE_TEMPLATE/hardware_validation_offer.yml", paths)
        self.assertIn("LICENSE", paths)
        self.assertFalse(any("third_party/rocm/" in path for path in paths))
        inventory = json.loads((ROOT / "docs/documentation-inventory.json").read_text(encoding="utf-8"))
        self.assertEqual(inventory["schema"], "rns8_documentation_inventory_v1")
        for record in inventory["documents"]:
            with self.subTest(path=record["path"]):
                self.assertEqual(record["inbound_links"], report["inbound_links"][record["path"]])
                self.assertEqual(record["anchors"], report["anchors"].get(record["path"], []))

    def test_missing_inventory_record_and_stale_derived_metadata_fail(self):
        original = Path.read_text
        inventory_path = ROOT / "docs/documentation-inventory.json"
        value = json.loads(inventory_path.read_text(encoding="utf-8"))
        def run_with(data):
            def read(path, *args, **kwargs):
                return json.dumps(data) if path == inventory_path else original(path, *args, **kwargs)
            with patch.object(Path, "read_text", read):
                return check()["errors"]
        missing = copy.deepcopy(value)
        missing["documents"].pop()
        self.assertTrue(any("inventory mismatch" in error for error in run_with(missing)))
        stale = copy.deepcopy(value)
        stale["documents"][0]["anchors"] = ["made-up-anchor"]
        self.assertTrue(any("stale derived" in error for error in run_with(stale)))
        duplicate = copy.deepcopy(value)
        duplicate["documents"].append(duplicate["documents"][0])
        self.assertIn("duplicate documentation inventory entries", run_with(duplicate))

    def test_missing_target_anchor_owner_and_script_fail(self):
        original = Path.read_text
        readme = ROOT / "README.md"
        inventory_path = ROOT / "docs/documentation-inventory.json"
        def read(path, *args, **kwargs):
            value = original(path, *args, **kwargs)
            if path == readme:
                return value + "\n[bad](docs/correctness.md#absent-anchor) [missing](docs/missing.md)\npython tools/no-such-script.py\n"
            if path == inventory_path:
                data = json.loads(value)
                data["documents"][0]["owner"] = "missing-owner.cpp"
                return json.dumps(data)
            return value
        with patch.object(Path, "read_text", read):
            errors = check()["errors"]
        for fragment in ("missing anchor", "missing target", "missing referenced script", "missing capability/evidence owner"):
            self.assertTrue(any(fragment in error for error in errors), fragment)

    def test_static_presets_scripts_targets_and_bench_options(self):
        presets = {"configurePresets": {"configure"}, "buildPresets": {"build"}, "testPresets": {"test"}}
        valid = "cmake --preset configure\ncmake --build --preset build\nctest --preset test\npython tools/check_documentation.py\nrns8-example-bounded-i64-oneshot\nrns8-bench --backend hip-direct --m 32"
        self.assertEqual(command_errors(ROOT, "fixture", valid, presets), [])
        malformed = "cmake --preset build\ncmake --build --preset test\nctest --preset configure\npython tools/missing.py\nrns8-example-missing\nrns8-bench --fictional-option true"
        self.assertEqual(len(command_errors(ROOT, "fixture", malformed, presets)), 6)

    def test_documentation_legal_and_funding_dispositions(self):
        data = json.loads((ROOT / "docs/documentation-inventory.json").read_text(encoding="utf-8"))
        records = {record["path"]: record for record in data["documents"]}
        for path in ("LICENSE", "NOTICE", "SECURITY.md", "CHANGELOG.md"):
            self.assertEqual(records[path]["disposition"], "preserve")
        self.assertEqual(records["docs/research/resident-output-api-draft.md"]["disposition"], "redirect")
        self.assertEqual(records["docs/archive/dashboard.html"]["disposition"], "historical")

    def test_readme_required_section_order_and_execution_boundaries(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        headings = ["Hardware tested", "Measured RX", "What it computes", "How it works",
                    "Build and run", "Use it", "Backends"]
        positions = [text.index("## " + heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("## Build without a GPU", text)
        self.assertNotIn("ctest --", text)
        records = re.findall(r"\]\((docs/local-validation-\d{8}\.md)\)", text)
        self.assertTrue(records, "README needs a dated local validation record")
        for path in set(records):
            record = ROOT / path
            receipts = re.findall(r"\]\((\.\./metadata/local-validation-\d{8}\.json)\)", record.read_text(encoding="utf-8"))
            self.assertTrue(receipts, "validation must bind its actual compiled artifacts")
            for receipt in receipts:
                artifacts = json.loads((record.parent / receipt).read_text(encoding="utf-8"))["artifacts"]
                self.assertTrue(artifacts)
                for artifact in artifacts.values():
                    self.assertRegex(artifact["build_commit"], r"^[0-9a-f]{7,40}$")
                    self.assertRegex(artifact["sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("Internal checked native-byte WMMA", text)
        self.assertIn("Public AMDGPU builtins", text)
        self.assertIn("examples/workloads/README.md", text)
        self.assertIn("application performance and other", text)


class HardwareOfferTest(unittest.TestCase):
    def test_form_notice_and_fields(self):
        self.assertEqual(template_errors(ROOT), [])

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML optional for issue-form schema parse; static notices/fields always checked")
    def test_yaml_schema_and_all_choices_are_strings(self):
        import yaml
        form = yaml.safe_load((ROOT / ".github/ISSUE_TEMPLATE/hardware_validation_offer.yml").read_text(encoding="utf-8"))
        self.assertIsInstance(form["name"], str)
        self.assertIsInstance(form["description"], str)
        fields = [field for field in form["body"] if field["type"] != "markdown"]
        self.assertEqual(len(fields), 5)
        for field in fields:
            self.assertIn(field["type"], ("input", "textarea", "dropdown"))
            self.assertTrue(field["validations"]["required"])
            self.assertIsInstance(field["attributes"]["label"], str)
            for option in field["attributes"].get("options", []):
                self.assertIsInstance(option, str)


class ExternalTest(unittest.TestCase):
    def test_success_fetch_is_bounded_and_fragment_is_not_requested(self):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        with patch("check_documentation.urlopen", return_value=response) as fetch:
            result = external_status("https://example.test/a#section", timeout=3)
        self.assertEqual(result["status"], "reachable")
        self.assertEqual(fetch.call_args.args[0].full_url, "https://example.test/a")
        self.assertEqual(fetch.call_args.kwargs["timeout"], 3)
        response.read.assert_called_once_with(1024)

    def test_429_retry_after_is_bounded_and_unresolved_is_not_pass(self):
        headers = Message()
        headers["Retry-After"] = "600"
        error = HTTPError("https://example.test", 429, "rate limited", headers, None)
        with patch("check_documentation.urlopen", side_effect=error) as fetch, patch("check_documentation.time.sleep") as sleep:
            result = external_status("https://example.test")
        self.assertEqual(fetch.call_count, 2)
        sleep.assert_called_once_with(5)
        self.assertEqual(result, dict(url="https://example.test", status="unresolved", http_status=429))

    def test_404_is_failed_without_retry_and_network_errors_unresolved(self):
        with patch("check_documentation.urlopen", side_effect=HTTPError("https://example.test", 404, "missing", {}, None)) as fetch:
            self.assertEqual(external_status("https://example.test")["status"], "failed")
        self.assertEqual(fetch.call_count, 1)
        for error in (URLError("offline"), TimeoutError("timeout"), OSError("network")):
            with patch("check_documentation.urlopen", side_effect=error) as fetch:
                self.assertEqual(external_status("https://example.test")["status"], "unresolved")
                self.assertEqual(fetch.call_count, 2)


if __name__ == "__main__":
    unittest.main()
