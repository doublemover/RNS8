"""Offline logical ordering and block-cache model; never physical XCD placement."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class Descriptor:
    m: int
    n: int
    k: int
    tile_m: int
    tile_n: int
    slab_k: int
    group_size: int
    prefixes: tuple[int, ...]  # Row-major output tiles; zero means proven zero.
    representation: str  # native64 or residue8; no implicit representation reuse.
    order: str
    target: str
    topology_version: str
    cache_domains: int = 1
    concurrent_groups_per_domain: int = 1
    cache_bytes_per_domain: int = 0
    version: int = 1


@dataclass(frozen=True, order=True)
class Work:
    slab: int
    row: int
    col: int
    first_plane: int
    planes: int


class Schedule:
    def __init__(self, desc: Descriptor):
        positive = (desc.m, desc.n, desc.k, desc.tile_m, desc.tile_n,
                    desc.slab_k, desc.group_size, desc.cache_domains,
                    desc.concurrent_groups_per_domain)
        if (any(type(x) is not int or x <= 0 for x in positive)
                or type(desc.cache_bytes_per_domain) is not int
                or desc.cache_bytes_per_domain < 0 or desc.version != 1
                or not desc.target or not desc.topology_version
                or desc.representation not in ("native64", "residue8")
                or desc.order not in ("output-neighbor", "modulus", "same-output")):
            raise ValueError("invalid schedule descriptor")
        self.desc = desc
        self.rows = (desc.m + desc.tile_m - 1) // desc.tile_m
        self.cols = (desc.n + desc.tile_n - 1) // desc.tile_n
        if (len(desc.prefixes) != self.rows * self.cols
                or any(type(p) is not int or not 0 <= p <= 20 for p in desc.prefixes)):
            raise ValueError("one valid prefix is required per output tile")
        self.phases: list[tuple[Work, ...]] = []
        self.inverse: dict[Work, tuple[int, int]] = {}
        for slab in range((desc.k + desc.slab_k - 1) // desc.slab_k):
            work = [Work(slab, row, col, first, min(desc.group_size, prefix - first))
                    for row in range(self.rows) for col in range(self.cols)
                    for prefix in (desc.prefixes[row * self.cols + col],)
                    for first in range(0, prefix, desc.group_size)]

            def key(item: Work):
                # Bucket compatible tail/cost shapes first. Basis indices are
                # immutable; sorting only changes their launch IDs.
                bucket = (self.extents(item)[:2], item.planes)
                local = {"output-neighbor": (item.row, item.first_plane, item.col),
                         "modulus": (item.first_plane, item.row, item.col),
                         "same-output": (item.row, item.col, item.first_plane)}[desc.order]
                return bucket, local

            phase = tuple(sorted(work, key=key))
            self.phases.append(phase)
            for launch, item in enumerate(phase):
                self.inverse[item] = (slab, launch)

    def extents(self, item: Work) -> tuple[int, int, int]:
        d = self.desc
        return (min(d.tile_m, d.m - item.row * d.tile_m),
                min(d.tile_n, d.n - item.col * d.tile_n),
                min(d.slab_k, d.k - item.slab * d.slab_k))

    def decode(self, phase: int, launch: int) -> Work:
        if not 0 <= phase < len(self.phases) or not 0 <= launch < len(self.phases[phase]):
            raise ValueError("launch ID outside schedule")
        return self.phases[phase][launch]

    def encode(self, item: Work) -> tuple[int, int]:
        try:
            return self.inverse[item]
        except KeyError as error:
            raise ValueError("work item outside schedule") from error

    def blocks(self, item: Work):
        """Disjoint logical tile blocks, not cache lines or measured transactions."""
        d = self.desc
        m, n, k = self.extents(item)
        planes = range(item.first_plane, item.first_plane + item.planes)
        source_planes = (-1,) if d.representation == "native64" else planes
        width = 8 if d.representation == "native64" else 1
        for plane in source_planes:
            yield ("a", item.row, item.slab, plane), m * k * width, True
            yield ("b", item.col, item.slab, plane), n * k * width, True
        for plane in planes:
            yield ("c", item.row, item.col, plane), m * n, False


def simulate(schedule: Schedule, hot: bool = False) -> dict:
    """Round-robin domains, fully associative LRU blocks, write allocation.

    Reuse distance counts distinct intervening block bytes in the domain trace.
    The diagnostic trace implementation is deliberately offline, O(accesses^2)
    in the worst case. A block larger than cache capacity bypasses the cache.
    """
    d = schedule.desc
    caches = [OrderedDict() for _ in range(d.cache_domains)]
    footprints = [0] * d.cache_domains
    history = [OrderedDict() for _ in range(d.cache_domains)]

    def run(measure: bool):
        requested_read = requested_write = misses = hits = 0
        source_requested = 0
        unique, source_unique, distances, waves = {}, {}, [], 0
        domain_work = [0] * d.cache_domains
        empty_slots = 0
        for phase in schedule.phases:
            capacity = d.cache_domains * d.concurrent_groups_per_domain
            phase_waves = (len(phase) + capacity - 1) // capacity
            waves += phase_waves
            empty_slots += phase_waves * capacity - len(phase)
            for launch, item in enumerate(phase):
                domain = launch % d.cache_domains
                domain_work[domain] += 1
                cache, trace = caches[domain], history[domain]
                for block, size, source in schedule.blocks(item):
                    unique[block] = size
                    if source:
                        source_unique[block] = size
                        source_requested += size
                        requested_read += size
                    else:
                        requested_write += size
                        if item.slab:
                            requested_read += size
                    if measure and block in trace:
                        distance = 0
                        for recent, recent_size in reversed(trace.items()):
                            if recent == block:
                                break
                            distance += recent_size
                        distances.append(distance)
                    trace[block] = size
                    trace.move_to_end(block)
                    if block in cache:
                        hits += size
                        cache.move_to_end(block)
                    else:
                        misses += size
                        if size <= d.cache_bytes_per_domain:
                            while footprints[domain] + size > d.cache_bytes_per_domain:
                                _, removed = cache.popitem(last=False)
                                footprints[domain] -= removed
                            cache[block] = size
                            footprints[domain] += size
        return dict(requested_read_bytes=requested_read, requested_write_bytes=requested_write,
                    requested_source_bytes=source_requested,
                    unique_source_bytes=sum(source_unique.values()), unique_bytes=sum(unique.values()),
                    modeled_cache_hit_bytes=hits, modeled_cache_miss_bytes=misses,
                    reuse_distance_bytes=distances, scheduling_waves=waves,
                    empty_wave_slots=empty_slots, domain_work_items=domain_work)

    if hot:
        run(False)  # Actual warm trace, not an assumption that everything fits.
    report = run(True)
    report.update(cache_state="warm-replay" if hot else "cold",
                  model="logical-block LRU; round-robin hypothetical domains; phase barriers",
                  target=d.target, topology_version=d.topology_version,
                  physical_affinity_verified=False, promotion_eligible=False,
                  actual_hbm_bytes=None, measured_cache_hit_rate=None)
    return report
