"""Read the same bounded source GEMM inventory compiled into native admission."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import isqrt
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "src/backend_common/source_gemm_contract.inc"
POLICIES = {
    "resolve_first",
    "host_cpu",
    "hip_target",
    "feature_detected_hip",
    "feature_detected_cpu",
}


@dataclass(frozen=True)
class Backend:
    enum: str
    name: str
    target_policy: str
    semantics: tuple[bool, ...]
    finite_minimum: int
    finite_maximum: int


@dataclass(frozen=True)
class Contract:
    version: int
    semantics: tuple[tuple[str, str], ...]
    backends: tuple[Backend, ...]
    source_sha256: str


def parse_contract(raw: bytes) -> Contract:
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= 32768:
        raise ValueError("bounded source GEMM inventory bytes required")
    text = re.sub(r"//[^\n]*", "", raw.decode("utf-8"))
    pattern = re.compile(r"RNS8_GEMM_SOURCE_(CONTRACT|SEMANTIC|BACKEND)\s*\((.*?)\)", re.S)
    cursor, version, semantics, backends = 0, None, [], []
    for match in pattern.finditer(text):
        if text[cursor : match.start()].strip():
            raise ValueError("unsupported source GEMM inventory syntax")
        cursor = match.end()
        mode, body = match.groups()
        if mode == "CONTRACT":
            if version is not None or semantics or backends or body.strip() != "1":
                raise ValueError("source GEMM inventory version/order differs")
            version = 1
        elif mode == "SEMANTIC":
            row = re.fullmatch(r'\s*(RNS8_[A-Z0-9_]+)\s*,\s*"([a-z0-9-]+)"\s*', body)
            if version is None or backends or row is None:
                raise ValueError("invalid source GEMM semantic record")
            semantics.append(row.groups())
        else:
            row = re.fullmatch(
                r'\s*(RNS8_BACKEND_[A-Z0-9_]+)\s*,\s*"([a-z0-9-]+)"\s*,\s*([a-z_]+)\s*,\s*'
                r"((?:true|false)(?:\s*,\s*(?:true|false)){6})\s*,\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*",
                body,
            )
            if version is None or len(semantics) != 7 or row is None:
                raise ValueError("invalid source GEMM backend record")
            enum, name, policy, flags, minimum, maximum = row.groups()
            bits = tuple(word.strip() == "true" for word in flags.split(","))
            minimum, maximum = int(minimum), int(maximum)
            domain = (2, 256) if any(bits[5:]) else (0, 0)
            if policy not in POLICIES or (minimum, maximum) != domain:
                raise ValueError("unsupported source GEMM target/modulus policy")
            backends.append(Backend(enum, name, policy, bits, minimum, maximum))
    if text[cursor:].strip() or version is None or len(semantics) != 7 or len(backends) != 10:
        raise ValueError("incomplete source GEMM inventory")
    if any(len(set(row[index] for row in semantics)) != len(semantics) for index in (0, 1)) or any(
        len({getattr(row, key) for row in backends}) != len(backends) for key in ("enum", "name")
    ):
        raise ValueError("duplicate source GEMM inventory identities")
    return Contract(version, tuple(semantics), tuple(backends), hashlib.sha256(raw).hexdigest())


def load_contract() -> Contract:
    with CONTRACT.open("rb") as stream:
        return parse_contract(stream.read(32769))


def admission(
    contract: Contract, backend: str, semantics: str, *, modulus=0, architecture=None
) -> dict:
    if (
        type(backend) is not str
        or type(semantics) is not str
        or type(modulus) is not int
        or not 0 <= modulus <= 256
    ):
        raise ValueError("explicit typed source operation identity required")
    row = next((row for row in contract.backends if row.name == backend), None)
    index = next((i for i, (_, name) in enumerate(contract.semantics) if name == semantics), None)
    if row is None or index is None:
        raise ValueError("unknown source backend or semantic name")
    hip = row.target_policy in {"hip_target", "feature_detected_hip"}
    if hip and (
        type(architecture) is not str or not re.fullmatch(r"gfx[0-9a-f]{3,5}", architecture)
    ):
        raise ValueError("explicit source candidate GPU target required")
    eligible, reason = row.semantics[index], "source_semantics_implemented"
    if not eligible:
        reason = "source_backend_does_not_implement_semantics"
    elif semantics.startswith("finite-"):
        if not row.finite_minimum <= modulus <= row.finite_maximum:
            eligible, reason = False, "finite_modulus_outside_source_domain"
        elif semantics == "finite-u8-field" and any(
            modulus % divisor == 0 for divisor in range(2, isqrt(modulus) + 1)
        ):
            eligible, reason = False, "finite_field_requires_prime_modulus"
    elif modulus:
        raise ValueError("nonfinite source cases cannot carry a finite modulus")
    return dict(
        schema="rns8_source_gemm_admission_v1",
        contract_version=contract.version,
        source_sha256=contract.source_sha256,
        backend=backend,
        semantics=semantics,
        modulus=modulus,
        requested_architecture=architecture,
        source_eligible=eligible,
        reason=reason,
        target_policy=row.target_policy,
        target_support="requires_native_compile_and_runtime_detection"
        if hip
        else "requires_host_provider_admission",
        compiled_available=None,
        runtime_available=None,
        hardware_qualified=False,
        promotion_allowed=False,
    )
