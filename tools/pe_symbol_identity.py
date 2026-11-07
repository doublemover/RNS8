"""Bound PE32+ RSDS and MSF7 PDB GUID/age identities without loading symbols."""

from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import uuid

MAX_BYTES = 512 * 1024**2
MSF_MAGIC = b"Microsoft C/C++ MSF 7.00\r\n\x1aDS\0\0\0"
SOURCES = [
    "https://learn.microsoft.com/en-us/windows/win32/debug/pe-format",
    "https://llvm.org/docs/PDB/MsfFile.html",
    "https://llvm.org/docs/PDB/PdbStream.html",
]


class Image:
    def __init__(self, path, max_bytes):
        self.path = Path(path).resolve(strict=True)
        if type(max_bytes) is not int or not 4096 <= max_bytes <= MAX_BYTES:
            raise ValueError("bounded PE/PDB byte budget required")
        if not self.path.is_file():
            raise ValueError("regular PE/PDB file required")
        self.size = self.path.stat().st_size
        if not 64 <= self.size <= max_bytes:
            raise ValueError("PE/PDB file exceeds its byte envelope")
        self.stream = self.path.open("rb")

    def __enter__(self):
        try:
            self.digest = self.hash()
            return self
        except BaseException:
            self.stream.close()
            raise

    def __exit__(self, *error):
        self.stream.close()

    def read(self, offset, length):
        if offset < 0 or length < 0 or offset > self.size or length > self.size - offset:
            raise ValueError("PE/PDB structure extends outside the actual file")
        self.stream.seek(offset)
        value = self.stream.read(length)
        if len(value) != length:
            raise ValueError("PE/PDB file truncated during inspection")
        return value

    def hash(self):
        self.stream.seek(0)
        digest, total = hashlib.sha256(), 0
        while block := self.stream.read(min(1024**2, self.size + 1 - total)):
            total += len(block)
            if total > self.size:
                raise ValueError("PE/PDB file grew during inspection")
            digest.update(block)
        if total != self.size:
            raise ValueError("PE/PDB file size changed during inspection")
        return digest.hexdigest()

    def finish(self):
        if self.hash() != self.digest or self.path.stat().st_size != self.size:
            raise ValueError("PE/PDB content changed during inspection")
        return dict(path=str(self.path), sha256=self.digest, bytes=self.size)


def pe_identity(path, *, max_bytes=MAX_BYTES):
    with Image(path, max_bytes) as image:
        dos = image.read(0, 64)
        offset = struct.unpack_from("<I", dos, 60)[0]
        if dos[:2] != b"MZ" or not 64 <= offset <= 1024**2:
            raise ValueError("native PE DOS header required")
        header = image.read(offset, 24)
        machine, count, _, _, _, optional_size, _ = struct.unpack_from("<HHIIIHH", header, 4)
        if header[:4] != b"PE\0\0" or machine not in (0x8664, 0xAA64) or not 1 <= count <= 96:
            raise ValueError("native x64/ARM64 PE image required")
        if not 168 <= optional_size <= 4096:
            raise ValueError("complete PE32+ debug data directory required")
        optional = image.read(offset + 24, optional_size)
        if (
            struct.unpack_from("<H", optional)[0] != 0x20B
            or struct.unpack_from("<I", optional, 108)[0] < 7
        ):
            raise ValueError("PE32+ debug data directory absent")
        debug_rva, debug_size = struct.unpack_from("<II", optional, 160)
        if not debug_rva or not debug_size or debug_size % 28 or debug_size > 28 * 128:
            raise ValueError("bounded nonempty PE debug directory required")
        sections = []
        for index in range(count):
            row = image.read(offset + 24 + optional_size + 40 * index, 40)
            virtual_bytes, rva, raw_bytes, raw_offset = struct.unpack_from("<IIII", row, 8)
            if rva + max(virtual_bytes, raw_bytes) > 1 << 32:
                raise ValueError("PE section RVA extent overflows")
            if raw_bytes:
                image.read(raw_offset, min(raw_bytes, 1))
                if raw_offset + raw_bytes > image.size:
                    raise ValueError("PE section raw extent exceeds file")
            sections.append((rva, raw_bytes, raw_offset))

        def file_offset(rva, length):
            found = [
                raw + rva - base
                for base, size, raw in sections
                if base <= rva and rva - base <= size and length <= size - (rva - base)
            ]
            if len(found) != 1:
                raise ValueError("ambiguous or missing file-backed PE debug RVA")
            return found[0]

        directory = image.read(file_offset(debug_rva, debug_size), debug_size)
        records = []
        for position in range(0, debug_size, 28):
            _, _, _, _, kind, size, rva, raw = struct.unpack_from("<IIHHIIII", directory, position)
            if kind != 2:
                continue
            if not 25 <= size <= 4096 + 25 or file_offset(rva, size) != raw:
                raise ValueError("invalid PE CodeView extent or RVA/raw disagreement")
            value = image.read(raw, size)
            if value[:4] != b"RSDS" or value[-1:] != b"\0" or b"\0" in value[24:-1]:
                raise ValueError("modern terminated RSDS CodeView record required")
            guid, age = uuid.UUID(bytes_le=value[4:20]), struct.unpack_from("<I", value, 20)[0]
            if not guid.int or not age:
                raise ValueError("nonzero PE PDB GUID/age required")
            records.append(
                dict(
                    guid=str(guid),
                    age=age,
                    recorded_pdb_path=value[24:-1].decode("utf-8", errors="strict"),
                )
            )
        if len(records) != 1:
            raise ValueError("one unambiguous PE RSDS identity required")
        return dict(image.finish(), machine=machine, **records[0])


def pdb_identity(path, *, max_bytes=MAX_BYTES):
    with Image(path, max_bytes) as image:
        header = image.read(0, 56)
        block_size, free_map, blocks, directory_bytes, reserved, block_map = struct.unpack_from(
            "<6I", header, 32
        )
        if header[:32] != MSF_MAGIC or block_size not in (512, 1024, 2048, 4096):
            raise ValueError("supported MSF7 PDB superblock required")
        if (
            blocks * block_size != image.size
            or free_map not in (1, 2)
            or reserved
            or not 8 <= directory_bytes <= 8 * 1024**2
        ):
            raise ValueError("invalid MSF7 file/directory extent")
        count = (directory_bytes + block_size - 1) // block_size
        if (
            not 3 <= block_map < blocks
            or block_map % block_size in (1, 2)
            or count * 4 > block_size
        ):
            raise ValueError("bounded single-block MSF7 directory map required")
        pages = list(
            struct.unpack("<" + "I" * count, image.read(block_map * block_size, count * 4))
        )
        used = {0, 1, 2, block_map}

        def admit(page):
            if not 3 <= page < blocks or page in used or page % block_size in (1, 2):
                raise ValueError("overlapping, reserved or out-of-file MSF7 stream block")
            used.add(page)

        for page in pages:
            admit(page)
        directory = b"".join(image.read(page * block_size, block_size) for page in pages)[
            :directory_bytes
        ]
        streams = struct.unpack_from("<I", directory)[0]
        if not 2 <= streams <= 4096 or 4 + streams * 4 > len(directory):
            raise ValueError("bounded MSF7 stream directory required")
        sizes = struct.unpack_from("<" + "I" * streams, directory, 4)
        cursor, info_page = 4 + streams * 4, None
        for index, size in enumerate(sizes):
            count = 0 if size == 0xFFFFFFFF else (size + block_size - 1) // block_size
            if count > blocks or cursor + count * 4 > len(directory):
                raise ValueError("MSF7 stream block list exceeds directory")
            for position in range(count):
                page = struct.unpack_from("<I", directory, cursor + 4 * position)[0]
                admit(page)
                if index == 1 and position == 0:
                    info_page = page
            cursor += count * 4
        if cursor != len(directory) or sizes[1] < 28 or sizes[1] == 0xFFFFFFFF or info_page is None:
            raise ValueError("complete MSF7 PDB info stream required")
        free_maps = {}
        for page in used:
            interval, bit = divmod(page, block_size * 8)
            if interval not in free_maps:
                free_maps[interval] = image.read(
                    (interval * block_size + free_map) * block_size, block_size
                )
            if free_maps[interval][bit // 8] & (1 << (bit % 8)):
                raise ValueError("referenced MSF7 block is marked free by the active map")
        version, signature, age, guid = struct.unpack(
            "<III16s", image.read(info_page * block_size, 28)
        )
        if version != 20000404 or not age or not any(guid):
            raise ValueError("supported VC70 PDB info identity required")
        return dict(
            image.finish(),
            guid=str(uuid.UUID(bytes_le=guid)),
            age=age,
            info_version=version,
            signature=signature,
        )


def symbol_pair(binary, pdb):
    executable, symbols = pe_identity(binary), pdb_identity(pdb)
    if any(executable[key] != symbols[key] for key in ("guid", "age")):
        raise ValueError("PDB GUID/age does not match the actual PE image")
    # Reopen both files after the pair is inspected; current aliases cannot be
    # switched between the two separately bounded reads.
    for file in (executable, symbols):
        with Image(file["path"], MAX_BYTES) as image:
            if image.digest != file["sha256"]:
                raise ValueError("earlier PE/PDB file changed during pair inspection")
    return dict(
        schema="rns8_pe_pdb_identity_v1",
        executable=executable,
        symbols=symbols,
        identity_match=True,
        source_contents_verified=False,
        native_symbol_resolution_observed=False,
        sources=SOURCES,
    )
