"""Authored Linux resource/plan controls; no host observation or tool executes."""

from __future__ import annotations

import copy
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import session_resource_contract as resources
import gpu_qualification as qualification

GIB = 1024**3


def observed():
    rows = [
        dict(
            path="/sys/fs/cgroup/allocation/job",
            cpu_max="150000 100000",
            memory_max=str(16 * GIB),
            memory_current=str(4 * GIB),
        ),
        dict(
            path="/sys/fs/cgroup/allocation",
            cpu_max="max 100000",
            memory_max=str(32 * GIB),
            memory_current=str(24 * GIB),
        ),
    ]
    return dict(
        system="Linux",
        allowed_cpu_count=8,
        mem_available_bytes=64 * GIB,
        disk_available_bytes=100 * GIB,
        available_inodes=10000,
        cgroup_state="observed",
        cgroup_observations=rows,
        cgroup_limits=resources.cgroup_limits(rows),
        amdgpu_loaded=True,
        kfd_access=True,
        render_access=True,
        target_inventory_state="observed",
        kfd_nodes=[
            dict(node_id=3, gpu_id=101, properties=dict(vendor_id=4098, gfx_target_version=90402))
        ],
    )


class ResourceContract(unittest.TestCase):
    def test_nested_cgroup_quota_and_shared_parent_memory_are_not_waived(self):
        value = resources.evaluate(
            observed(),
            jobs=2,
            minimum_free_bytes=2 * GIB,
            peak_job_memory_bytes=3 * GIB,
            memory_reserve_bytes=GIB,
            gpu_required=True,
            architecture="gfx942",
        )
        self.assertEqual(value["state"], "resource_access_checks_passed")
        self.assertEqual(value["effective_build_cpu_ceiling"], 2)
        self.assertEqual(value["observed_memory_headroom_bytes"], 8 * GIB)
        self.assertEqual(value["required_memory_bytes"], 7 * GIB)
        self.assertFalse(value["memory_is_reserved"])
        self.assertFalse(value["selected_runtime_device_mapping_verified"])
        self.assertFalse(value["hardware_qualified"])
        value = resources.evaluate(
            observed(),
            jobs=3,
            minimum_free_bytes=2 * GIB,
            peak_job_memory_bytes=3 * GIB,
            memory_reserve_bytes=GIB,
        )
        self.assertEqual(value["state"], "blocked")
        self.assertEqual(len(value["blockers"]), 2)

    def test_unknown_resource_model_and_gpu_access_do_not_become_hardware_credit(self):
        value = resources.evaluate(observed(), jobs=1, minimum_free_bytes=GIB)
        self.assertTrue(any("model not supplied" in item for item in value["warnings"]))
        for mode in ("kfd", "render", "driver", "target", "cgroup", "disk", "inodes"):
            data = observed()
            if mode in {"kfd", "render"}:
                data[mode + "_access"] = False
            elif mode == "driver":
                data["amdgpu_loaded"] = False
            elif mode == "target":
                data["kfd_nodes"][0]["properties"]["gfx_target_version"] = 90010
            elif mode == "cgroup":
                data["cgroup_state"] = "unavailable"
            elif mode == "disk":
                data["disk_available_bytes"] = GIB - 1
            else:
                data["available_inodes"] = 0
            with self.subTest(mode=mode):
                value = resources.evaluate(
                    data, jobs=1, minimum_free_bytes=GIB, gpu_required=True, architecture="gfx942"
                )
                self.assertEqual(value["state"], "blocked")
                self.assertFalse(value["hardware_qualified"])

    def test_actual_visible_cgroup_mount_root_and_ancestor_resolution(self):
        mountinfo = "31 20 0:28 /tenant /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
        self.assertEqual(
            resources.cgroup_directories("0::/tenant/job/step\n", mountinfo),
            [Path("/sys/fs/cgroup/job/step"), Path("/sys/fs/cgroup/job"), Path("/sys/fs/cgroup")],
        )
        for membership, mounts in (
            ("1:cpu:/tenant/job", mountinfo),
            ("0::/other/job", mountinfo),
            ("0::/tenant/../escape", mountinfo),
            ("0::/tenant/job", mountinfo + mountinfo),
        ):
            with self.subTest(membership=membership), self.assertRaises(ValueError):
                resources.cgroup_directories(membership, mounts)

    def test_malformed_and_resealed_derived_resources_are_rejected(self):
        for mode in ("bool", "quota", "missing_current", "derived"):
            data = observed()
            if mode == "bool":
                data["mem_available_bytes"] = True
            elif mode == "quota":
                data["cgroup_observations"][0]["cpu_max"] = "150000 0"
            elif mode == "missing_current":
                data["cgroup_observations"][0]["memory_current"] = None
            else:
                data["cgroup_limits"]["memory_headroom_bytes"] = 64 * GIB
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                resources.evaluate(data, jobs=1, minimum_free_bytes=GIB)
        with (
            patch.object(resources, "observe", side_effect=AssertionError("must not observe")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                resources.main(
                    [
                        "--output-root",
                        "unused",
                        "--jobs",
                        "0",
                        "--minimum-free-bytes",
                        "1",
                        "--architecture",
                        "gfx942",
                    ]
                ),
                1,
            )

    def test_kfd_source_fields_remain_in_their_native_namespace(self):
        value = resources.kfd_properties(
            "gfx_target_version 90402\nvendor_id 4098\nlocation_id 1024\nunknown_future_field 7\n"
        )
        self.assertEqual(value, dict(gfx_target_version=90402, vendor_id=4098, location_id=1024))
        for text in (
            "gfx_target_version 90402\ngfx_target_version 90402\n",
            "vendor_id -1\n",
            "gfx_target_version unknown\n",
        ):
            with self.assertRaises(ValueError):
                resources.kfd_properties(text)

    def test_plan_precedes_compilation_and_consumer_recomputes_native_policy(self):
        with patch.object(resources, "observe", side_effect=AssertionError("must not observe")):
            plan = qualification.command_plan(
                "mi300x",
                ["hip-direct"],
                Path("temp/plan"),
                2,
                [7],
                resource_policy=dict(minimum_free_bytes=GIB, peak_job_memory_bytes=3 * GIB),
            )
        item = plan[0]
        self.assertEqual(item["name"], "linux-resource-access-preflight")
        self.assertTrue(plan[1]["name"].endswith("-configure"))
        self.assertEqual(plan[1]["depends_on"], [item["name"]])
        with patch.object(resources.os, "getpid", return_value=777):
            value = resources.evaluate(observed(), **item["host_resources"])
        result = dict(
            returncode=0, timed_out=False, process_id=777, stdout=json.dumps(value), stderr=""
        )
        self.assertEqual(qualification.validate_result(item, result), [])
        for key, changed in (
            ("effective_build_cpu_ceiling", 16),
            ("hardware_qualified", 0),
            ("process_id", 778),
        ):
            altered = copy.deepcopy(value)
            altered[key] = changed
            with self.subTest(key=key):
                self.assertTrue(
                    qualification.validate_result(item, dict(result, stdout=json.dumps(altered)))
                )


if __name__ == "__main__":
    unittest.main()
