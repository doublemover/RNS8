"""Safe plain-text cells for generated Markdown tables."""
from __future__ import annotations


def markdown_cell(value: object) -> str:
    return str(value if value is not None else "").replace("&", "&amp;").replace(
        "<", "&lt;").replace(">", "&gt;").replace("|", "&#124;").replace(
        chr(96), "&#96;").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
