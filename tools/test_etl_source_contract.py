"""Authored native module/PDB controls; no decoder or symbolizer is executed."""

from __future__ import annotations

import copy
import unittest
import uuid

from etl_source_contract import (
    module_points,
    decode_lookup,
    source_report,
    frame_address_catalog,
    address_module_candidates,
    symbolized_report,
    source_observations,
)
from test_etl_cpu_contract import read as read_cpu, cpu_metadata
from test_etl_image_contract import read as read_images, image
from test_etl_stack_contract import read as read_stacks, stack_metadata, fragment
from test_etl_heap_contract import heap_metadata
from test_etl_schedule_contract import schedule_metadata

GUID = uuid.UUID("e93dca29-156c-43bb-b53b-280622752b35")


def inputs():
    cpu = read_cpu(cpu_metadata(), ranges=True)
    images = read_images([image(image_size=8192), image(2, 800, image_size=8192)])
    pair = dict(
        executable=dict(image_size=8192, image_checksum=123, image_timestamp=456),
        symbols=dict(path=r"C:\app.pdb", guid=str(GUID), age=1),
    )
    return cpu, images, [dict(recorded_image_path=r"C:\app.exe", pair=pair)]


def lookup():
    request = dict(pdb=r"C:\app.pdb", dbghelp=r"C:\dbghelp.dll", image_size=8192, rvas=[0x1234])
    document = dict(
        schema="rns8_native_pdb_symbols_v1",
        process_id=555,
        reader_source_sha256="a" * 64,
        pdb_path=request["pdb"],
        dbghelp_path=request["dbghelp"],
        pdb_guid_bytes_le=list(GUID.bytes_le),
        pdb_age=1,
        image_size=8192,
        lookup="explicit_local_PDB_RVAs_not_a_live_process",
        inlined_callchain_resolved=False,
        source_contents_verified=False,
        symbols=[
            dict(
                rva=0x1234,
                symbol=dict(name="rns8_proof", start_rva=0x1200, displacement=0x34),
                source=dict(
                    path=r"C:\src\proof.cpp", line=73, start_rva=0x1230, displacement=4, text=None
                ),
            )
        ],
    )
    return request, document


def decode(document, request=None):
    return decode_lookup(
        document,
        request=request or lookup()[0],
        pair=inputs()[2][0]["pair"],
        process_id=555,
        source_sha256="a" * 64,
    )


class NativeSources(unittest.TestCase):
    def test_closed_image_RVA_lookup_and_same_file_phase_counts_remain_conditional(self):
        cpu, images, modules = inputs()
        candidates = module_points(cpu, images, modules)
        self.assertEqual([p["rva"] for p in candidates["points"]], [0x1234, 0x1234])
        self.assertFalse(candidates["unresolved"])
        request, document = lookup()
        values = decode(document, request)
        report = source_report(cpu, candidates, {(0, rva): v for rva, v in values.items()})
        self.assertEqual([r["phase"] for r in report["selected_sites"]], ["proof", "public_call"])
        self.assertEqual(sum(r["samples"] for r in report["selected_sites"]), 2)
        self.assertIsNone(report["CPU_duration_ns"])
        self.assertFalse(report["loaded_symbols_verified"])
        self.assertFalse(report["source_contents_verified"])

    def test_reused_module_base_is_joined_by_actual_lifetime(self):
        cpu, _, modules = inputs()
        images = read_images(
            [
                image(image_size=8192),
                image(2, 500, image_size=8192),
                image(10, 600, image_size=8192),
                image(2, 800, image_size=8192),
            ]
        )
        result = module_points(cpu, images, modules)
        self.assertEqual([p["image_generation_id"] for p in result["points"]], [0, 2])

    def test_ambiguous_missing_rundown_and_changed_image_fingerprints_withhold_attribution(self):
        for kind in ("rundown", "unclosed", "overlap", "undecoded", "changed", "wrong_path"):
            cpu, images, modules = inputs()
            if kind == "rundown":
                images = read_images([image(3, 210), image(4, 800)])
            elif kind == "unclosed":
                images = read_images([image(image_size=8192)])
            elif kind == "overlap":
                images["generations"].append(copy.deepcopy(images["generations"][0]))
                images["generations"][-1].update(
                    unload=None, state="load_observed_unload_unobserved"
                )
            elif kind == "undecoded":
                images["undecoded_image_events"] = 1
            elif kind == "changed":
                modules[0]["pair"]["executable"]["image_checksum"] = 124
            else:
                modules[0]["recorded_image_path"] = r"C:\other.exe"
            with self.subTest(kind=kind):
                result = module_points(cpu, images, modules)
                self.assertFalse(result["points"])
                self.assertEqual(len(result["unresolved"]), 2)

    def test_no_sample_or_symbol_inventory_can_be_duplicated_or_dropped(self):
        cpu, images, modules = inputs()
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            module_points(cpu, images, modules * 2)
        cpu["samples"].append(cpu["samples"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            module_points(cpu, images, modules)
        cpu, images, modules = inputs()
        candidates = module_points(cpu, images, modules)
        with self.assertRaisesRegex(ValueError, "inventory"):
            source_report(cpu, candidates, {})

    def test_lookup_requires_actual_opened_GUID_age_process_source_and_RVA_identity(self):
        for key, value in (
            ("pdb_guid_bytes_le", [True] * 16),
            ("pdb_guid_bytes_le", [0] * 16),
            ("pdb_age", 2),
            ("process_id", True),
            ("process_id", 556),
            ("reader_source_sha256", "b" * 64),
            ("pdb_path", r"C:\other.pdb"),
            ("image_size", True),
            ("source_contents_verified", True),
            ("inlined_callchain_resolved", True),
        ):
            request, document = lookup()
            document[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                decode(document, request)
        for key, value in (("rva", True), ("rva", 1), ("symbol", []), ("source", [])):
            _, document = lookup()
            document["symbols"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                decode(document)

    def test_line_and_symbol_displacements_and_native_errors_are_not_guessed(self):
        for field, key, value in (
            ("source", "displacement", 3),
            ("symbol", "start_rva", -1),
            ("source", "line", True),
            ("source", "text", "invented"),
            ("symbol", "name", "a\0b"),
        ):
            _, document = lookup()
            document["symbols"][0][field][key] = value
            with self.subTest(field=field, key=key), self.assertRaises(ValueError):
                decode(document)
        _, document = lookup()
        row = document["symbols"][0]
        row.update(symbol=None, source=None, symbol_error=487, source_error=126)
        self.assertIsNone(decode(document)[0x1234]["source"])
        del row["source_error"]
        with self.assertRaises(ValueError):
            decode(document)

    def test_native_frames_deduplicate_address_epoch_lookup_but_preserve_recursive_event_counts(
        self,
    ):
        metadata = stack_metadata()
        metadata["stack_events"][1]["frames"] = [0x180001400]
        cpu = read_cpu(metadata, ranges=True)
        stacks = read_stacks(metadata, ranges=True)
        _, images, modules = inputs()
        catalog = frame_address_catalog(stacks, images)
        self.assertEqual(catalog["native_bound_frame_observations"], 3)
        self.assertEqual(catalog["unique_address_epoch_candidates"], 2)
        candidates = address_module_candidates(catalog["addresses"], 333, images, modules)
        leaves = module_points(cpu, images, modules)
        keys = {
            (p["module_id"], p["rva"])
            for group in (leaves["points"], candidates["points"])
            for p in group
        }
        values = {
            key: dict(
                rva=key[1],
                symbol=dict(name="proof", start_rva=key[1], displacement=0),
                source=dict(
                    path=r"C:\src\proof.cpp", line=73, start_rva=key[1], displacement=0, text=None
                ),
            )
            for key in keys
        }
        report = symbolized_report(cpu, leaves, catalog, candidates, values)
        self.assertEqual(len(report["native_stack_fragment_sources"]["lookup_table"]), 2)
        self.assertEqual(report["samples"][0]["stack_fragment_records"], [11, 12])
        self.assertEqual(len(report["stack_source_sites"]), 1)
        site = report["stack_source_sites"][0]
        self.assertEqual(
            (
                site["native_event_count"],
                site["native_fragment_count"],
                site["native_frame_slot_observations"],
            ),
            (1, 2, 3),
        )
        self.assertEqual(site["phase"], "proof")
        self.assertEqual(site["scope"], "native_measured_phase")
        self.assertFalse(
            report["native_stack_fragment_sources"]["native_fragment_concatenation_performed"]
        )
        observations = list(source_observations(report))
        self.assertEqual(len(observations), 2)
        self.assertEqual(len(list(observations[0]["callchain"])), 3)
        self.assertFalse(report["source_contents_verified"])
        self.assertFalse(report["loaded_symbols_verified"])
        with self.assertRaisesRegex(ValueError, "inventory"):
            symbolized_report(cpu, leaves, catalog, candidates, {})

    def test_address_deduplication_never_crosses_an_observed_module_reuse_boundary(self):
        metadata = stack_metadata()
        metadata["stack_events"] = [
            fragment(11, frames=[0x180001400]),
            fragment(12, qpc=6500, frames=[0x180001400]),
        ]
        metadata["stack_frames"] = 2
        stacks = read_stacks(metadata)
        _, _, modules = inputs()
        images = read_images(
            [
                image(image_size=8192),
                image(2, 500, image_size=8192),
                image(10, 600, image_size=8192),
                image(2, 800, image_size=8192),
            ]
        )
        catalog = frame_address_catalog(stacks, images)
        self.assertEqual(len(catalog["addresses"]), 2)
        self.assertNotEqual(
            catalog["addresses"][0]["image_map_epoch"], catalog["addresses"][1]["image_map_epoch"]
        )
        candidates = address_module_candidates(catalog["addresses"], 333, images, modules)
        self.assertEqual([p["image_generation_id"] for p in candidates["points"]], [0, 2])

    def test_unresolved_leaf_keeps_valid_conditional_native_caller_fragments(self):
        metadata = stack_metadata()
        metadata["sample_events"][0]["instruction_pointer"] = 0xFFFF800000000000
        cpu = read_cpu(metadata, ranges=True)
        stacks = read_stacks(metadata, ranges=True)
        _, images, modules = inputs()
        catalog = frame_address_catalog(stacks, images)
        candidates = address_module_candidates(catalog["addresses"], 333, images, modules)
        leaves = module_points(cpu, images, modules)
        self.assertEqual(len(leaves["unresolved"]), 1)
        keys = {
            (p["module_id"], p["rva"])
            for group in (leaves["points"], candidates["points"])
            for p in group
        }
        values = {
            key: dict(
                rva=key[1],
                symbol=dict(name="caller", start_rva=key[1], displacement=0),
                source=None,
            )
            for key in keys
        }
        report = symbolized_report(cpu, leaves, catalog, candidates, values)
        self.assertEqual(len(report["samples"]), 2)
        self.assertIsNone(report["samples"][0]["symbol"])
        self.assertEqual(report["samples"][0]["stack_fragment_records"], [11, 12])
        self.assertTrue(report["native_stack_fragment_symbolization_observed"])
        self.assertTrue(report["unresolved"])
        self.assertFalse(report["inlined_or_unwound_stacks_resolved"])

    def test_allocator_and_scheduler_frames_keep_native_role_and_measured_phase(self):
        for role in ("heap_event", "schedule_out"):
            metadata = heap_metadata() if role == "heap_event" else schedule_metadata()
            qpc = 6000 if role == "heap_event" else 3500
            metadata.update(
                stack_events=[fragment(20, qpc=qpc, frames=[0x180001400, 0x180001400])],
                observed_stack_events=1,
                stack_frames=2,
            )
            cpu = read_cpu(metadata, ranges=True)
            stacks = read_stacks(
                metadata, heap=role == "heap_event", schedule=role == "schedule_out", ranges=True
            )
            _, images, modules = inputs()
            catalog = frame_address_catalog(stacks, images)
            frames = address_module_candidates(catalog["addresses"], 333, images, modules)
            leaves = module_points(cpu, images, modules)
            keys = {
                (p["module_id"], p["rva"])
                for group in (leaves["points"], frames["points"])
                for p in group
            }
            values = {
                key: dict(
                    rva=key[1],
                    symbol=dict(name="native_call", start_rva=key[1], displacement=0),
                    source=None,
                )
                for key in keys
            }
            report = symbolized_report(cpu, leaves, catalog, frames, values)
            with self.subTest(role=role):
                self.assertEqual(len(report["stack_source_sites"]), 1)
                site = report["stack_source_sites"][0]
                self.assertEqual(site["source_kind"], role)
                self.assertEqual(site["phase"], "public_call")
                self.assertEqual(site["native_event_count"], 1)
                self.assertEqual(site["native_frame_slot_observations"], 2)
                self.assertEqual(len(list(source_observations(report))), 3)


if __name__ == "__main__":
    unittest.main()
