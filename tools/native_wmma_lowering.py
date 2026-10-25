"""Single structured lowering for emitted HIP and bounded logical interpretation.

This is not an AMD machine-code interpreter. Matrix lane mapping and native
reduction semantics are explicit trusted contracts, bound to dependency hashes.
Every index, predicate, slab loop and accumulator lifetime below is emitted.
"""
import copy
import hashlib
import json

SCHEMA = "rns8_native_wmma_lowering_v1"
OPS = {"+": 2, "*": 2, "//": 2, "%": 2, "<": 2, "and": 2, "min": 2}
VARIABLES = {"bx", "by", "bz", "lane", "word", "byte", "reg", "channel", "kk",
             "m", "n", "k", "lda", "ldb", "ldc", "prefix"}


def expr(op, a, b):
    return [op, a, b]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def expression(node, variables=None):
    """Emit or evaluate the SAME small, total integer expression language."""
    if type(node) is int:
        if not 0 <= node <= 2147483647:
            raise ValueError("integer literal outside supported nonnegative INT32 domain")
        return str(node) if variables is None else node
    if isinstance(node, str) and node in VARIABLES:
        return f"static_cast<int64_t>({node})" if variables is None else variables[node]
    if not isinstance(node, list) or len(node) != 3 or node[0] not in OPS:
        raise ValueError("unsupported expression node: " + repr(node))
    op = node[0]
    a, b = (expression(n, variables) for n in node[1:])
    if variables is None:
        if op == "min":
            return f"(({a}) < ({b}) ? ({a}) : ({b}))"
        symbol = {"//": "/", "and": "&&"}.get(op, op)
        return f"({a} {symbol} {b})"
    if op == "+": return a + b
    if op == "*": return a * b
    if op == "//": return a // b
    if op == "%": return a % b
    if op == "<": return int(a < b)
    if op == "and": return int(bool(a) and bool(b))
    if op == "min": return min(a, b)
    raise ValueError("unsupported operation")


def recipe(group, unsigned_inputs):
    q = expr("+", "kk", expr("+", expr("*", "word", 4), "byte"))
    ar = expr("+", expr("*", "by", 16), expr("%", "lane", 16))
    bc = expr("+", expr("*", "bx", 16), expr("%", "lane", 16))
    row = expr("+", expr("*", "by", 16), expr("+", expr("*", "reg", 2), expr("//", "lane", 16)))
    plane = expr("+", expr("*", "bz", group), "channel")
    return {"schema": SCHEMA, "group": group, "unsigned_inputs": unsigned_inputs,
            "tile": [16, 16, 16], "wave": 32, "words": 4, "bytes": 4, "registers": 8,
            "slab_k": 65536, "slab_step": 1, "slab_repetitions": 1, "k_step": 16,
            "accumulator_initialized": True, "reset_each_slab": False,
            "carry": "centered_remainder", "encoding": "native64_centered_default_basis",
            "instruction": "rdna3_signed_iu8_16x16x16_wave32", "output": "centered_i8",
            "a_index": expr("+", expr("*", ar, "lda"), q),
            "b_index": expr("+", expr("*", q, "ldb"), bc),
            "a_mask": expr("and", expr("<", ar, "m"), expr("<", q, "k")),
            "b_mask": expr("and", expr("<", bc, "n"), expr("<", q, "k")),
            "plane": plane, "modulus_index": plane, "plane_mask": expr("<", plane, "prefix"),
            "c_index": expr("+", expr("*", expr("+", expr("*", plane, "m"), row), "ldc"), bc),
            "c_mask": expr("and", expr("<", row, "m"), expr("<", bc, "n"))}


def validate(ir):
    baseline = recipe(ir.get("group"), ir.get("unsigned_inputs"))
    if set(ir) != set(baseline) or ir.get("schema") != SCHEMA:
        raise ValueError("unknown lowering schema or fields")
    if type(ir["group"]) is not int or ir["group"] not in (1, 2, 4) or type(ir["unsigned_inputs"]) is not bool:
        raise ValueError("unsupported group/input domain")
    for key in ("tile", "wave", "words", "bytes", "registers", "encoding", "instruction", "output", "carry"):
        if ir[key] != baseline[key]: raise ValueError("unsupported lowering node: " + key)
    for key in ("slab_k", "slab_step", "slab_repetitions", "k_step"):
        if type(ir[key]) is not int or not 1 <= ir[key] <= 65536:
            raise ValueError("invalid bounded loop: " + key)
    if ir["slab_k"] % 16 or ir["k_step"] % 16:
        raise ValueError("unsupported partial-instruction slab schedule")
    for key in ("accumulator_initialized", "reset_each_slab"):
        if type(ir[key]) is not bool: raise ValueError("invalid accumulator lifetime")
    def variables(node):
        if isinstance(node, str): return {node}
        if isinstance(node, list): return variables(node[1]) | variables(node[2])
        return set()
    for key in ("a_index", "b_index", "a_mask", "b_mask", "plane", "modulus_index", "plane_mask", "c_index", "c_mask"):
        expression(ir[key])
        allowed = ({"bx", "by", "lane", "word", "byte", "kk", "m", "n", "k", "lda", "ldb", "ldc", "prefix"}
                   if key.startswith(("a_", "b_")) else
                   {"bz", "channel", "prefix"} if key in ("plane", "modulus_index", "plane_mask") else
                   {"bx", "by", "bz", "channel", "lane", "reg", "m", "n", "k", "lda", "ldb", "ldc", "prefix"})
        if not variables(ir[key]) <= allowed:
            raise ValueError("unsupported out-of-scope/dependent expression: " + key)


def render(ir):
    """Kernel body, not a wrapper around an independently implemented tile."""
    validate(ir)
    e = lambda key: expression(ir[key])
    native = "uint64_t" if ir["unsigned_inputs"] else "int64_t"
    center = "center_native_u64" if ir["unsigned_inputs"] else "center_native_i64"
    init = "{}" if ir["accumulator_initialized"] else ""
    reset = "for (int channel=0; channel<Group; ++channel) acc[channel] = accumulator{};" if ir["reset_each_slab"] else ""
    return f'''#if defined(__HIP_DEVICE_COMPILE__)
  using fragment = int __attribute__((ext_vector_type(4)));
  using accumulator = int __attribute__((ext_vector_type(8)));
  using Native = {native};
  namespace finite = rns8::detail::finite_u8;
  constexpr int Group = {ir['group']};
  const auto* A = static_cast<const Native*>(a);
  const auto* B = static_cast<const Native*>(b);
  const int64_t bx=blockIdx.x, by=blockIdx.y, bz=blockIdx.z;
  const int lane=threadIdx.x & 31;
  accumulator acc[Group]{init};
  finite::Native64Reducer reducers[Group]{{}};
  for (int channel=0; channel<Group; ++channel)
    if ({e('plane_mask')}) reducers[channel] = finite::native64_reducer(rns8_default_moduli_device[{e('modulus_index')}]);
  for (int64_t slab=0; slab<k; slab+={ir['slab_k'] * ir['slab_step']}) {{
    {reset}
    const int64_t end = slab+{ir['slab_k']} < k ? slab+{ir['slab_k']} : k;
    for (int repeat=0; repeat<{ir['slab_repetitions']}; ++repeat)
      for (int64_t kk=slab; kk<end; kk+={ir['k_step']}) {{
        fragment av[Group]{{}}, bv[Group]{{}};
        for (int word=0; word<4; ++word) {{
          uint32_t ap[Group]{{}}, bp[Group]{{}};
          for (int byte=0; byte<4; ++byte) {{
            const Native raw_a = {e('a_mask')} ? A[{e('a_index')}] : Native{{}};
            const Native raw_b = {e('b_mask')} ? B[{e('b_index')}] : Native{{}};
            for (int channel=0; channel<Group; ++channel) if ({e('plane_mask')}) {{
              ap[channel] |= uint32_t(uint8_t(finite::{center}(raw_a,reducers[channel]))) << (8*byte);
              bp[channel] |= uint32_t(uint8_t(finite::{center}(raw_b,reducers[channel]))) << (8*byte);
            }}
          }}
          for (int channel=0; channel<Group; ++channel) {{
            av[channel][word]=static_cast<int>(ap[channel]);
            bv[channel][word]=static_cast<int>(bp[channel]);
          }}
        }}
        for (int channel=0; channel<Group; ++channel) if ({e('plane_mask')})
          acc[channel] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,av[channel],true,bv[channel],acc[channel],false);
      }}
    for (int channel=0; channel<Group; ++channel) if ({e('plane_mask')})
      for (int reg=0; reg<8; ++reg) {{
        const int modulus=static_cast<int>(reducers[channel].modulus);
        int carried=acc[channel][reg] % modulus;
        if (carried<0) carried+=modulus;
        if (carried>=(modulus+1)/2) carried-=modulus;
        acc[channel][reg]=carried;
      }}
  }}
  for (int channel=0; channel<Group; ++channel) if ({e('plane_mask')}) {{
    const int modulus=static_cast<int>(reducers[channel].modulus);
    for (int reg=0; reg<8; ++reg) if ({e('c_mask')}) {{
      int value=acc[channel][reg] % modulus;
      if (value<0) value+=modulus;
      if (value>=(modulus+1)/2) value-=modulus;
      c[{e('c_index')}] = static_cast<int8_t>(value);
    }}
  }}
#endif'''


def mutations(ir):
    result = {}
    for name in ("wrong_stride", "omitted_tail_mask", "uninitialized_partial", "stale_partial",
                 "wrong_modulus", "missing_slab", "duplicate_slab", "duplicate_writer"):
        value = copy.deepcopy(ir)
        if name == "wrong_stride":
            def replace(x):
                return "ldb" if x == "lda" else [replace(y) for y in x] if isinstance(x, list) else x
            value["a_index"] = replace(value["a_index"])
        elif name == "omitted_tail_mask": value["a_mask"] = 1
        elif name == "uninitialized_partial": value["accumulator_initialized"] = False
        elif name == "stale_partial": value["reset_each_slab"] = True
        elif name == "wrong_modulus": value["modulus_index"] = expr("+", value["plane"], 1)
        elif name == "missing_slab": value["slab_step"] = 2
        elif name == "duplicate_slab": value["slab_repetitions"] = 2
        elif name == "duplicate_writer": value["c_index"] = 0
        result[name] = value
    return result
