"""Bounded non-following filesystem metadata traversal for owned artifacts."""

from __future__ import annotations

import os
import math
from pathlib import Path
import stat
import time


def artifact_entries(roots, *, seconds: float, missing_ok=False, maximum_entries=100000):
    """Yield (path, file bytes), with None for progress and one metadata read."""
    if (
        type(seconds) not in (float, int)
        or not math.isfinite(seconds)
        or seconds <= 0
        or type(missing_ok) is not bool
        or type(maximum_entries) is not int
        or maximum_entries < 1
    ):
        raise ValueError("positive finite scan and entry budgets required")
    pending = list(roots)
    deadline, count = time.monotonic() + seconds, len(pending)
    if count > maximum_entries:
        raise ValueError("owned artifact tree exceeds its entry budget")
    junction = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)

    def check_time():
        if time.monotonic() >= deadline:
            raise ValueError("owned artifact inventory exceeds its scan time budget")

    while pending:
        check_time()
        entry = pending.pop()
        try:
            # DirEntry retains enumeration metadata on Windows and caches its
            # non-following stat elsewhere. Path predicates would query again.
            info = entry.lstat() if isinstance(entry, Path) else entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_reparse_tag", 0) == junction:
                raise ValueError("owned artifact tree contains a link or junction")
            if stat.S_ISDIR(info.st_mode):
                yield Path(entry), None
                with os.scandir(entry) as children:
                    for child in children:
                        check_time()
                        count += 1
                        if count > maximum_entries:
                            raise ValueError("owned artifact tree exceeds its entry budget")
                        pending.append(child)
                        # Let supervisors yield even while enumerating a wide
                        # directory; None contributes no unobserved file bytes.
                        yield Path(child), None
            elif stat.S_ISREG(info.st_mode):
                yield Path(entry), info.st_size
            else:
                raise ValueError("owned artifact tree contains a nonregular file")
        except FileNotFoundError:
            if not missing_ok:
                raise ValueError("owned artifact tree contains an absent file") from None


def regular_files(roots, *, seconds: float, missing_ok=False, maximum_entries=100000):
    for path, size in artifact_entries(
        roots, seconds=seconds, missing_ok=missing_ok, maximum_entries=maximum_entries
    ):
        if size is not None:
            yield path, size
