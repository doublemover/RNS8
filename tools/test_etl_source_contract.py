"""Authored native module/PDB controls; no decoder or symbolizer is executed."""

from __future__ import annotations

import copy
import unittest
import uuid

from etl_source_contract import module_points, decode_lookup, source_report
from test_etl_cpu_contract import read as read_cpu, cpu_metadata
from test_etl_image_contract import read as read_images, image

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


if __name__ == "__main__":
    unittest.main()
