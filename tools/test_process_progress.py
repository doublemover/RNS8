"""Authored bounded child-process controls; no library/GPU workloads."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from process_progress import run_capture, capture_succeeded
from gpu_qualification import validate_result


class BoundedProcessCapture(unittest.TestCase):
    def test_incremental_inventory_does_not_block_child_deadlines_and_closes_its_cursor(self):
        closed = []

        def slow_inventory(*args, **kwargs):
            try:
                for _ in range(10000):
                    time.sleep(0.001)
                    yield Path.cwd(), None
            finally:
                closed.append(True)

        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parents[1] / "temp"
        ) as directory:
            with patch("process_progress.artifact_entries", side_effect=slow_inventory):
                result = run_capture(
                    [sys._base_executable, "-c", "import time; time.sleep(10)"],
                    cwd=Path.cwd(),
                    label="authored-slow-inventory",
                    artifact_roots=[Path(directory)],
                    max_artifact_bytes=1024,
                    timeout_seconds=0.2,
                )
        self.assertTrue(result["timed_out"])
        self.assertLess(result["elapsed_seconds"], 3)
        self.assertEqual(closed, [True])

    def test_expected_information_exit_does_not_waive_capture_integrity(self):
        result = dict(returncode=1, timed_out=False)
        self.assertTrue(capture_succeeded(result, expected_exit_code=1))
        self.assertFalse(capture_succeeded(result))
        self.assertFalse(capture_succeeded(result, expected_exit_code=True))
        for change in (
            dict(timed_out=True),
            dict(interrupted=True),
            dict(capture_error="read failed"),
            dict(limit_exceeded="stdout_stderr_bytes"),
            dict(returncode=True),
        ):
            with self.subTest(change=change):
                self.assertFalse(capture_succeeded(dict(result, **change), expected_exit_code=1))

    def test_started_pid_observer_can_cancel_its_owned_child(self):
        cancelled, observed = threading.Event(), []

        def started(pid):
            observed.append(pid)
            cancelled.set()

        result = run_capture(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            cwd=Path.cwd(),
            label="authored-supervised-child",
            timeout_seconds=5,
            on_start=started,
            cancel_event=cancelled,
        )
        self.assertEqual(observed, [result["process_id"]])
        self.assertTrue(result["interrupted"])
        self.assertIn("supervisor", result["capture_error"])
        self.assertFalse(capture_succeeded(result))

    def test_start_observer_failure_cleans_up_and_does_not_claim_readiness(self):
        def failed(pid):
            raise ValueError("authored start observer rejected " + str(pid))

        result = run_capture(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            cwd=Path.cwd(),
            label="authored-start-observer",
            timeout_seconds=5,
            on_start=failed,
        )
        self.assertIn("start observer rejected", result["capture_error"])
        self.assertFalse(capture_succeeded(result))

    def test_streams_pid_unicode_and_final_output_are_preserved(self):
        result = run_capture(
            [
                # Windows venv redirectors own a different PID from the Python
                # process executing this body. Check a native child directly.
                sys._base_executable,
                "-c",
                "import os,sys; print(os.getpid()); print('unicode \\u03b1'); print('stderr',file=sys.stderr)",
            ],
            cwd=Path.cwd(),
            label="authored-output",
            timeout_seconds=10,
        )
        self.assertTrue(capture_succeeded(result))
        self.assertEqual(int(result["stdout"].splitlines()[0]), result["process_id"])
        self.assertIn("unicode \u03b1", result["stdout"])
        self.assertIn("stderr", result["stderr"])

    def test_stdout_and_stderr_share_a_hard_retention_budget(self):
        result = run_capture(
            [
                sys.executable,
                "-c",
                "import os; os.write(1,b'a'*200000); os.write(2,b'b'*200000)",
            ],
            cwd=Path.cwd(),
            label="authored-output-limit",
            timeout_seconds=10,
            max_output_bytes=1024,
        )
        self.assertFalse(capture_succeeded(result))
        self.assertEqual(result["limit_exceeded"], "stdout_stderr_bytes")
        self.assertEqual(result["captured_output_bytes"], 1024)
        self.assertLessEqual(len(result["stdout"]) + len(result["stderr"]), 1024)
        self.assertTrue(validate_result({}, result))

    def test_normal_zero_exit_cannot_hide_a_capture_error(self):
        result = dict(
            returncode=0,
            timed_out=False,
            limit_exceeded="stdout_stderr_bytes",
            stdout="{}",
            stderr="",
            process_id=1,
        )
        self.assertFalse(capture_succeeded(result))
        self.assertTrue(validate_result({}, result))
        result.update(limit_exceeded=None, capture_error="owned output did not close")
        self.assertFalse(capture_succeeded(result))

    def test_artifact_budget_is_relative_to_child_cwd_and_preserves_partial_files(self):
        scratch = Path.cwd() / "temp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root = Path(temporary)
            result = run_capture(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import time; p=Path('artifacts'); p.mkdir(); (p/'partial.bin').write_bytes(b'a'*65536); time.sleep(10)",
                ],
                cwd=root,
                label="authored-artifact-limit",
                timeout_seconds=5,
                artifact_roots=[Path("artifacts")],
                max_artifact_bytes=1024,
            )
            self.assertFalse(capture_succeeded(result))
            self.assertEqual(result["limit_exceeded"], "owned_artifact_bytes")
            self.assertTrue((root / "artifacts" / "partial.bin").is_file())
            self.assertGreater(result["observed_artifact_bytes"], 1024)

    def test_timeout_and_missing_command_never_pass(self):
        result = run_capture(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            cwd=Path.cwd(),
            label="authored-timeout",
            timeout_seconds=0.2,
        )
        self.assertFalse(capture_succeeded(result))
        self.assertTrue(result["timed_out"])
        missing = run_capture(
            [str(Path.cwd() / ("missing-authored-command-" + str(os.getpid())))],
            cwd=Path.cwd(),
            label="authored-missing",
            timeout_seconds=1,
        )
        self.assertFalse(capture_succeeded(missing))
        self.assertIsNone(missing["process_id"])

    def test_lossy_decoding_cannot_be_a_successful_capture(self):
        result = run_capture(
            [sys.executable, "-c", "import os; os.write(1,b'\\xff')"],
            cwd=Path.cwd(),
            label="authored-invalid-utf8",
            timeout_seconds=5,
        )
        self.assertFalse(capture_succeeded(result))
        self.assertIn("UTF-8", result["capture_error"])

    def test_opt_in_interrupt_preserves_partial_output_and_cannot_pass(self):
        original_event = threading.Event
        original_wait = original_event.wait
        events = []

        def event():
            value = original_event()
            events.append(value)
            return value

        def wait(value, timeout=None):
            if (
                events
                and value is events[0]
                and threading.current_thread() is threading.main_thread()
            ):
                if not original_wait(value, 5):
                    raise AssertionError("authored child output did not arrive")
                raise KeyboardInterrupt
            return original_wait(value, timeout)

        with (
            patch("process_progress.threading.Event", side_effect=event),
            patch.object(original_event, "wait", new=wait),
        ):
            result = run_capture(
                [
                    sys.executable,
                    "-c",
                    "import time; print('retained before interrupt',flush=True); time.sleep(30)",
                ],
                cwd=Path.cwd(),
                label="authored-interrupt",
                timeout_seconds=10,
                return_on_interrupt=True,
            )
        self.assertTrue(result["interrupted"])
        self.assertIn("retained before interrupt", result["stdout"])
        self.assertFalse(capture_succeeded(result))
        for malformed in (0, "false", True):
            self.assertFalse(
                capture_succeeded(dict(returncode=0, timed_out=False, interrupted=malformed))
            )

    def test_nonfinite_and_missing_budgets_fail_before_launch(self):
        for kwargs in (
            dict(timeout_seconds=float("nan")),
            dict(heartbeat_seconds=float("inf")),
            dict(max_output_bytes=0),
            dict(artifact_roots=[Path("output")]),
            dict(return_on_interrupt=1),
        ):
            with self.assertRaises(ValueError):
                run_capture(
                    ["must-not-launch"],
                    cwd=Path.cwd(),
                    label="authored-invalid",
                    **kwargs,
                )


if __name__ == "__main__":
    unittest.main()
