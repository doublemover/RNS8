from dataclasses import replace

from arithmetic_composition import LayoutMapping, Node, verify_composition, verify_layout


def main() -> int:
    byte = Node("byte", "input", interval=(-128, 127), bits=8)
    dot = Node("dot", "dot", ("byte", "byte"), count=65536)
    assert verify_composition((byte, dot))["algebraically_verified"]
    combined = Node("combined", "add", ("dot", "dot"))
    assert not verify_composition((byte, dot, combined))["algebraically_verified"]
    residue = Node("residue", "canonical_reduce", ("dot",), modulus=256, bits=8, signed=False)
    safe = Node("safe", "add", ("residue", "residue"))
    assert verify_composition((byte, dot, residue, safe))["algebraically_verified"]
    premature = Node("divide", "exact_divide", ("residue",), factor=2)
    assert not verify_composition((byte, dot, residue, premature))["algebraically_verified"]
    even = Node("even", "input", interval=(-256, 254), multiple=2)
    divide = Node("divide", "exact_divide", ("even",), factor=2, bits=8)
    assert verify_composition((even, divide))["algebraically_verified"]
    scaled = Node("scaled", "rescale", ("dot",), factor=2, bits=64)
    mixed = Node("mixed", "add", ("dot", "scaled"), bits=64)
    assert not verify_composition((byte, dot, scaled, mixed))["algebraically_verified"]
    overflow = replace(dot, count=131072)
    final_zero = Node("zero", "input", interval=(0, 0))
    assert not verify_composition((byte, overflow, final_zero))["algebraically_verified"]
    carry = Node("carry", "input", interval=(0, (1 << 30)))
    carried = replace(dot, inputs=("byte", "byte", "carry"))
    assert not verify_composition((byte, carry, carried))["algebraically_verified"]
    assert not verify_composition((dot, byte))["algebraically_verified"]
    mapping = LayoutMapping((0,1,4,5),(0,2,1,3),6,4)
    layout = Node("layout","layout",("byte",),bits=8,layout=mapping)
    result=verify_composition((byte,layout))
    assert result["algebraically_verified"]
    assert result["verdicts"]["physical_layout_correctness"] == "proven"
    assert result["verdicts"]["target_execution"] == "not_run"
    for broken in (replace(mapping,source_offsets=(0,0,4,5)),
                   replace(mapping,destination_offsets=(0,1,2,4)),
                   replace(mapping,destination_offsets=(0,1)),
                   replace(mapping,out_of_place=False),
                   replace(mapping,source_offsets=(False,1,4,5))):
        assert not verify_layout(broken)
        assert not verify_composition((byte,replace(layout,layout=broken)))["algebraically_verified"]
    assert not verify_composition((byte,replace(layout,layout=None)))["algebraically_verified"]
    changed=replace(layout,layout=replace(mapping,destination_offsets=(3,2,1,0)))
    assert verify_composition((byte,changed))["sha256"] != result["sha256"]
    print("arithmetic composition self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
