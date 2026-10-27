"""Generate compiler-facing scalar encoding fixtures with exact certificates."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from arithmetic_certificate import verify
from encoding_catalogue import awe_certificate, schoolbook_certificate, generate_header


def generate(directory):
    variants = {f"{name}_{m}": factory(m) for name, factory in
                (("awe6", awe_certificate), ("base13", schoolbook_certificate)) for m in (251, 256)}
    original = awe_certificate(256)
    variants["exact_divide"] = replace(original, denominator=2,
        planes=tuple(replace(p, lhs=tuple(2 * x for x in p.lhs)) for p in original.planes),
        instruction=replace(original.instruction, operand_min=-8, operand_max=8))
    sources = {name: generate_header(c, name) for name, c in variants.items()}
    directory.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        (directory / (name + ".hpp")).write_text(source, encoding="utf-8", newline="\n")
    aggregate = "#pragma once\n" + "".join(f'#include "{name}.hpp"\n' for name in sources)
    (directory / "encodings.hpp").write_text(aggregate, encoding="utf-8", newline="\n")
    manifest = {name: {"source_sha256": hashlib.sha256(sources[name].encode()).hexdigest(),
                       "proof": verify(c)} for name, c in variants.items()}
    (directory / "encodings.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                                               encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    generate(parser.parse_args().out_dir)
