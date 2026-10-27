"""Regenerate source identity constants without running arithmetic proofs."""
import hashlib
from pathlib import Path


def rendered(root: Path) -> str:
    paths = {"wmma": "wrap64_signed_radix_certificate.json",
             **{target: "wrap64_signed_mfma_"+target+".json" for target in ("gfx90a", "gfx942", "gfx950")}}
    prefix = '#pragma once\nnamespace rns8::detail {\n// SHA256 of the complete tracked certificate bytes; never an executed proof receipt.\n'
    return prefix+''.join('inline constexpr const char* wrap64_certificate_'+name+'="'+hashlib.sha256((root/'tools/data'/path).read_bytes()).hexdigest()+'";\n' for name,path in paths.items())+'}\n'


if __name__ == '__main__':
    root=Path(__file__).resolve().parents[1]
    (root/'src/backend_common/wrap64_certificate_identity.hpp').write_text(rendered(root),encoding='utf-8')
