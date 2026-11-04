from __future__ import annotations

import unittest

from ck_kernel_contract import CK_KERNEL_FAMILIES, ck_kernel_family, ck_kernel_matches, ck_selected_kernel
from install_autotune_cache import reviewed_kernel_supported_for_contract
from metadata_registry_constants import CK_SELECTED_KERNELS


class CkKernelContracts(unittest.TestCase):
    def test_current_families_are_registered_and_admitted(self) -> None:
        for family in CK_KERNEL_FAMILIES:
            for q in (0, 127, 251, 255, 256):
                kernel = ck_selected_kernel(family, q)
                self.assertIn(kernel, CK_SELECTED_KERNELS)
                self.assertEqual(ck_kernel_family(kernel), family)
                self.assertTrue(ck_kernel_matches(kernel, q))
                self.assertTrue(reviewed_kernel_supported_for_contract(
                    "ck", "finite_ring_u8" if q else "bounded_i64", kernel, q, 64, 64, 64
                ))
            self.assertTrue(ck_kernel_matches(ck_selected_kernel(family, tiled=True), allow_tiled=True))
            self.assertFalse(ck_kernel_matches(ck_selected_kernel(family, tiled=True)))
            self.assertFalse(ck_kernel_matches(ck_selected_kernel(family, 251), 255))
            self.assertFalse(ck_kernel_matches(f"{family}_unknown"))

    def test_unknown_family_and_encoding_are_rejected(self) -> None:
        self.assertIsNone(ck_kernel_family("ck_xdl_cshuffle_u8_i32_v4_default_moduli_centered_epilogue"))
        with self.assertRaises(ValueError):
            ck_selected_kernel("ck_unknown")
        with self.assertRaises(ValueError):
            ck_selected_kernel(CK_KERNEL_FAMILIES[0], 257)
        with self.assertRaises(ValueError):
            ck_selected_kernel(CK_KERNEL_FAMILIES[0], 251, tiled=True)


if __name__ == "__main__":
    unittest.main()
