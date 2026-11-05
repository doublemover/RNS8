"""Current CK lowering labels; historical reports retain their original labels."""

from __future__ import annotations

CK_KERNEL_FAMILIES = (
    "ck_wmma_cshuffle_s8_i32_v4",
    "ck_wmma_cshuffle_u8_i32_v4",
    "ck_xdl_cshuffle_s8_i32_v4",
)


def ck_selected_kernel(family: str, modulus: int = 0, *, tiled: bool = False) -> str:
    if family not in CK_KERNEL_FAMILIES:
        raise ValueError("unknown CK physical implementation")
    if modulus:
        if not 2 <= modulus <= 256 or tiled:
            raise ValueError("invalid CK finite kernel contract")
        tag = f"mod{modulus}" if modulus in {251, 255, 256} else "static_modulus"
        return f"{family}_finite_u8_{tag}_centered_epilogue"
    return f"{family}_{'tiled_' if tiled else ''}default_moduli_centered_epilogue"


def ck_kernel_family(kernel: object) -> str | None:
    if not isinstance(kernel, str):
        return None
    return next((family for family in CK_KERNEL_FAMILIES if kernel.startswith(f"{family}_")), None)


def ck_kernel_matches(kernel: str, modulus: int = 0, *, allow_tiled: bool = False) -> bool:
    if modulus and not 2 <= modulus <= 256:
        return False
    family = ck_kernel_family(kernel)
    return family is not None and (
        kernel == ck_selected_kernel(family, modulus)
        or (not modulus and allow_tiled and kernel == ck_selected_kernel(family, tiled=True))
    )
