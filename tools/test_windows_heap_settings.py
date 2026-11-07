"""Recovery and interruption controls using an in-memory registry only."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from windows_heap_settings import (
    configuration,
    temporary_configuration,
    restore_journal,
    write_journal,
)


def absent(*, key=False):
    return dict(key_exists=key, value_exists=False, type=None, value=None)


def dword(value):
    return dict(key_exists=True, value_exists=True, type="REG_DWORD", value=value)


class Registry:
    def __init__(self, before):
        self.value = copy.deepcopy(before)
        self.enables = 0
        self.restores = 0
        self.fail_enable = False
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.closed = True

    def snapshot(self):
        return copy.deepcopy(self.value)

    def enable(self, before):
        assert self.value == before
        self.enables += 1
        self.value = dword(1)
        if self.fail_enable:
            raise OSError("controlled flush failure after mutation")

    def restore(self, before):
        self.restores += 1
        if self.value != dword(1) and self.value != before:
            raise ValueError("controlled concurrent setting change")
        self.value = copy.deepcopy(before)
        return "restored_exact_prior_state"


class HeapSettings(unittest.TestCase):
    def lease(self, registry, documents, *, authorized=True, journal=None):
        return temporary_configuration(
            configuration("rns8-bench.exe"),
            "journal",
            authorized=authorized,
            registry_factory=lambda policy: registry,
            journal=journal
            or (lambda path, value: documents.append((path.name, copy.deepcopy(value)))),
        )

    def test_plan_and_missing_authorization_never_open_the_registry(self):
        policy = configuration("rns8-bench.exe")
        self.assertEqual(policy["value"], "TracingFlags")
        self.assertEqual(policy["registry_view_bits"], 64)
        for value in (False, 1, None):
            registry = Registry(absent())
            documents = []
            with self.assertRaisesRegex(ValueError, "authorization"):
                with self.lease(registry, documents, authorized=value):
                    self.fail("unapproved setting mutation")
            self.assertFalse(documents)
            self.assertEqual(registry.enables, 0)
            self.assertFalse(registry.closed)
        for image in ("../app.exe", r"C:\app.exe", "app", "x;evil.exe", "x\0.exe"):
            with self.assertRaises(ValueError):
                configuration(image)

    def test_prior_state_is_journaled_before_write_and_restored_after_success_or_interrupt(self):
        for before in (absent(), absent(key=True), dword(0), dword(1)):
            for failure in (None, ValueError, KeyboardInterrupt):
                registry = Registry(before)
                documents = []

                def journal(path, value, *, registry=registry, before=before, documents=documents):
                    if path.name.endswith("before.json"):
                        self.assertEqual(registry.value, before)
                        self.assertEqual(registry.enables, 0)
                    documents.append((path.name, copy.deepcopy(value)))

                def run(registry=registry, documents=documents, failure=failure, journal=journal):
                    with self.lease(registry, documents, journal=journal):
                        self.assertEqual(registry.value, dword(1))
                        if failure:
                            raise failure("controlled workload interruption")

                with self.subTest(before=before, failure=failure):
                    if failure:
                        with self.assertRaises(failure):
                            run()
                    else:
                        run()
                    self.assertEqual(registry.value, before)
                    self.assertTrue(registry.closed)
                    self.assertEqual(registry.enables, 0 if before == dword(1) else 1)
                    self.assertEqual(
                        [n for n, _ in documents],
                        ["heap-settings-before.json", "heap-settings-after.json"],
                    )
                    self.assertTrue(documents[-1][1]["prior_value_restored"])

    def test_failed_prejournal_never_mutates_and_failed_native_flush_still_restores(self):
        registry = Registry(absent())

        def broken(path, value):
            raise OSError("controlled full journal volume")

        with self.assertRaises(OSError):
            with self.lease(registry, [], journal=broken):
                self.fail("work continued after backup failed")
        self.assertEqual(registry.enables, 0)
        self.assertEqual(registry.restores, 0)
        self.assertTrue(registry.closed)
        registry = Registry(dword(0))
        registry.fail_enable = True
        documents = []
        with self.assertRaisesRegex(OSError, "flush"):
            with self.lease(registry, documents):
                self.fail("failed configuration continued")
        self.assertEqual(registry.value, dword(0))
        self.assertEqual(registry.restores, 1)
        self.assertTrue(documents[-1][1]["prior_value_restored"])

    def test_unknown_prior_values_and_concurrent_changes_are_never_overwritten(self):
        for before in (dword(2), dict(dword(1), type="REG_SZ"), dict(dword(1), value=True)):
            registry = Registry(before)
            documents = []
            with self.assertRaises(ValueError):
                with self.lease(registry, documents):
                    self.fail("unknown prior setting was overwritten")
            self.assertFalse(documents)
            self.assertEqual(registry.enables, 0)
        registry = Registry(dword(0))
        documents = []
        with self.assertRaisesRegex(ValueError, "concurrent"):
            with self.lease(registry, documents):
                registry.value = dword(2)
        self.assertEqual(registry.value, dword(2))
        self.assertEqual(documents[-1][1]["state"], "restore_incomplete")
        self.assertFalse(documents[-1][1]["prior_value_restored"])

    def test_explicit_crash_recovery_uses_saved_prior_value_and_retains_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = root / "prior.json"
            write_journal(
                saved,
                dict(
                    schema="rns8_windows_heap_settings_journal_v1",
                    policy=configuration("rns8-bench.exe"),
                    prior=dword(0),
                    owner_pid=123,
                    mutation_may_follow=True,
                ),
            )
            registry = Registry(dword(1))
            documents = []

            def journal(path, value):
                documents.append((path.name, copy.deepcopy(value)))

            with self.assertRaisesRegex(ValueError, "authorization"):
                restore_journal(
                    saved,
                    root,
                    registry_factory=lambda policy: self.fail("unapproved registry read"),
                )
            report = restore_journal(
                saved,
                root,
                authorized=True,
                registry_factory=lambda policy: registry,
                journal=journal,
            )
            self.assertEqual(registry.value, dword(0))
            self.assertTrue(registry.closed)
            self.assertTrue(report["prior_value_restored"])
            self.assertEqual(report["original_owner_pid"], 123)
            self.assertEqual(
                [n for n, _ in documents],
                ["heap-settings-recovery-before.json", "heap-settings-recovery-after.json"],
            )
            registry = Registry(dword(2))
            with self.assertRaisesRegex(ValueError, "concurrent"):
                restore_journal(
                    saved,
                    root,
                    authorized=True,
                    registry_factory=lambda policy: registry,
                    journal=journal,
                )
            self.assertEqual(registry.value, dword(2))
            self.assertFalse(documents[-1][1]["prior_value_restored"])


if __name__ == "__main__":
    unittest.main()
