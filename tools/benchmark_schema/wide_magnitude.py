"""Exact integer checks for public single-GEMM magnitude receipts (not timing qualification)."""
import math

WIDE_PROFILES = {"wide-two-limb-constant": (40, False), "wide-three-limb-constant": (62, False),
                 "wide-two-limb-mixed": (40, True), "wide-three-limb-mixed": (62, True)}
MODULI = (256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193, 191, 181, 179, 173, 167)


def validate_magnitude(data: dict) -> list[str]:
    profile = data.get("input_profile", "uniform-small")
    magnitude = data.get("exact_wide_magnitude")
    if magnitude is None:
        return ["wide magnitude profile requires complete magnitude receipt"] if profile in WIDE_PROFILES else []
    errors = []
    try:
        if not isinstance(magnitude, dict) or magnitude["schema"] != "exact_wide_magnitude_v1":
            raise ValueError("magnitude schema")
        if magnitude["oracle_complete"] is not True or magnitude["oracle"] != "boost_cpp_int_full_integer_gemm_all_limbs":
            raise ValueError("full exact integer oracle receipt required")
        if data.get("residue_chain_length", 1) != 1:
            raise ValueError("single-GEMM magnitude receipt cannot describe a dependent chain")
        signed = data["semantics"] == "exact_wide_signed"
        if data["semantics"] not in {"exact_wide_signed", "exact_wide_unsigned"}:
            raise ValueError("magnitude semantic contract")
        def decimal(key):
            value = magnitude[key]
            if not isinstance(value, str) or not value.isdecimal(): raise ValueError(key)
            return int(value)
        def number(key):
            value = magnitude[key]
            if type(value) is not int or value < 0: raise ValueError(key)
            return value
        a, b, bound, product = [decimal(key) for key in ("a_max_abs", "b_max_abs", "admitted_abs_bound", "basis_product")]
        prefix = data.get("selected_prefix", data["prefix"])
        if type(prefix) is not int or not 1 <= prefix <= len(MODULI) or product != math.prod(MODULI[:prefix]):
            raise ValueError("basis product does not match selected prefix")
        # Batch maxima may belong to different tasks: max per-task bound is no
        # larger than the product of global maxima, and covers every output.
        if bound > data["k"] * a * b or product <= bound * (2 if signed else 1):
            raise ValueError("invalid strict range certificate")
        limbs = data["exact_wide_limb_count"]
        if bound >= (1 << (64 * limbs - int(signed))): raise ValueError("limb width cannot hold admitted bound")
        if number("a_max_magnitude_bits") != a.bit_length() or number("b_max_magnitude_bits") != b.bit_length():
            raise ValueError("input magnitude bit lengths")
        low, high = number("output_min_magnitude_bits"), number("output_max_magnitude_bits")
        if low > high or high > bound.bit_length() or number("output_max_magnitude_limbs") != (high + 63) // 64:
            raise ValueError("output magnitude bits/limbs")
        if number("storage_bits") != 64 * limbs: raise ValueError("storage width")
        tasks = data.get("host_api_batch", {}).get("tasks_per_measured_repeat", 1)
        tasks = max(tasks, data.get("grouped_dispatch", {}).get("task_count", 1))
        expected_outputs = data["m"] * data["n"] * tasks
        if number("checked_outputs") != expected_outputs or number("checked_limbs") != expected_outputs * limbs:
            raise ValueError("oracle must check every output limb, not a sample or checksum only")
        if number("negative_outputs") > expected_outputs or (not signed and number("negative_outputs")):
            raise ValueError("output signedness")
        if profile in WIDE_PROFILES:
            exponent, mixed = WIDE_PROFILES[profile]
            if not all((1 << exponent) <= value <= (1 << exponent) + (1023 if mixed else 0) for value in (a, b)):
                raise ValueError("input maxima do not match magnitude profile")
            if not mixed and (low != (data["k"] << (2 * exponent)).bit_length() or low != high):
                raise ValueError("constant profile must retain its mathematically known wide output magnitude")
        if magnitude["timing_scope"] != "outside_all_repeated_public_call_samples": raise ValueError("oracle timing scope")
        for key in ("admission_us", "oracle_us"):
            if type(magnitude[key]) not in (int, float) or not math.isfinite(magnitude[key]) or magnitude[key] < 0:
                raise ValueError("invalid preflight/postflight timing")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        errors.append(f"invalid exact-wide magnitude receipt: {exc}")
    return errors
