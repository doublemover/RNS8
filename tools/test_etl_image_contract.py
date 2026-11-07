"""Authored TDH image observations; no ETL decoder, file fixture or profiler runs."""

from __future__ import annotations

import copy
import unittest

from etl_image_contract import image_catalog
from etl_process_contract import process_contract
from test_etl_process_contract import native_metadata


def image(opcode=10, stamp=210, **fields):
    return dict(
        dict(
            pid=333,
            opcode=opcode,
            time_filetime_100ns=stamp,
            image_base=0x180000000,
            image_size=4096,
            image_checksum=123,
            image_timestamp=456,
            image_path_utf8=r"C:\app.exe",
            event_version=2,
            pointer_bytes=8,
        ),
        **fields,
    )


def read(rows, *, change=None):
    metadata = native_metadata()
    metadata.update(
        image_events=rows, observed_image_events=len(rows), observed_events=max(10, len(rows) + 2)
    )
    if change:
        change(metadata)
    process = process_contract(
        metadata,
        pid=333,
        parent_pid=444,
        executable="app.exe",
        decoder_pid=555,
        source_sha256="a" * 64,
        file_bytes=4096,
    )
    return image_catalog(metadata, process)


class NativeImages(unittest.TestCase):
    def test_native_load_unload_pairs_preserve_generation_and_span_identity(self):
        report = read(
            [
                image(),
                image(2, 400),
                image(10, 500, image_path_utf8=r"C:\β\module.dll"),
                image(2, 800, image_path_utf8=r"C:\β\module.dll"),
            ]
        )
        self.assertEqual([row["generation_id"] for row in report["generations"]], [0, 2])
        self.assertEqual(
            [row["observed_lifetime_100ns"] for row in report["generations"]], [190, 300]
        )
        self.assertTrue(
            all(row["state"] == "observed_load_unload_pair" for row in report["generations"])
        )
        self.assertFalse(report["unresolved"])
        self.assertFalse(report["loaded_PDB_identity_verified"])
        self.assertFalse(report["complete_image_history_proven"])
        self.assertEqual(report["CSV_module_or_IP_join"], "unbound")

    def test_rundown_is_a_snapshot_and_cannot_create_load_unload_lifetimes(self):
        report = read([image(3, 210), image(4, 800)])
        self.assertEqual(len(report["rundown_snapshots"]), 2)
        self.assertFalse(report["generations"])
        self.assertTrue(report["raw_module_presence_observed"])
        self.assertFalse(report["complete_image_history_proven"])

    def test_unmatched_replaced_and_conflicting_generations_remain_explicit(self):
        report = read(
            [image(2, 210), image(10, 300), image(10, 500), image(2, 800, image_checksum=999)]
        )
        self.assertEqual(
            [row["state"] for row in report["generations"]],
            ["replacement_without_observed_unload", "conflicting_unload_metadata"],
        )
        self.assertEqual(
            {row["reason"] for row in report["unresolved"]},
            {
                "unmatched_image_unload",
                "prior_image_unload_unobserved",
                "unload_differs_from_observed_live_image",
            },
        )
        self.assertTrue(all(row["unload"] is None for row in report["generations"]))

    def test_equal_timestamp_events_do_not_receive_an_invented_order(self):
        report = read([image(10, 300), image(2, 300)])
        self.assertFalse(report["generations"])
        self.assertEqual(len(report["unresolved"]), 2)
        self.assertTrue(
            all(
                row["reason"] == "same_base_same_timestamp_order_unresolved"
                for row in report["unresolved"]
            )
        )

    def test_malformed_native_types_spans_names_windows_and_counts_fail(self):
        for fields in (
            dict(pid=334),
            dict(pointer_bytes=True),
            dict(image_size=0),
            dict(image_base=(1 << 64) - 1),
            dict(image_checksum=True),
            dict(image_path_utf8="bad\0path"),
            dict(image_path_utf8="x" * 4096),
            dict(time_filetime_100ns=1001),
            dict(opcode=1),
            dict(image_base=(1 << 32) - 1, image_size=2, pointer_bytes=4),
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                read([image(**fields)])
        for key in ("observed_image_events", "undecoded_image_events"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                read([image()], change=lambda value, key=key: value.update({key: True}))
        with self.assertRaises(ValueError):
            read([image()], change=lambda value: value.update(undecoded_image_events=1))

    def test_record_and_cumulative_path_budgets_are_enforced(self):
        with self.assertRaises(ValueError):
            read([image() for _ in range(4097)])
        path = "C:\\" + "β" * 4000
        with self.assertRaisesRegex(ValueError, "path budget"):
            read(
                [
                    image(
                        stamp=210 + i,
                        image_base=0x180000000 + i * 4096,
                        image_path_utf8=path + str(i),
                    )
                    for i in range(132)
                ]
            )

    def test_loss_and_partial_process_identity_do_not_turn_into_loaded_symbol_proof(self):
        report = read([image()], change=lambda value: value["header"].update(events_lost=1))
        self.assertFalse(report["selected_process_identity_complete"])
        self.assertFalse(report["loaded_PDB_identity_verified"])
        self.assertFalse(report["promotion_allowed"])
        metadata = native_metadata()
        process = dict(
            schema="rns8_etl_selected_process_custody_v1",
            selected_pid=334,
            process_identity_complete=False,
        )
        with self.assertRaises(ValueError):
            image_catalog(copy.deepcopy(metadata), process)


if __name__ == "__main__":
    unittest.main()
