"""Independent physical metadata accounting for immutable native queue schedules."""
from collections.abc import Mapping, Sequence
from typing import TypedDict

IDENTITIES: tuple[str, ...] = ('explicit_round_robin_tile_channel_v1', 'task_major_prefix_directory_tile_channel_v1',
              'compact_round_robin_breakpoint_rank_scan_v1', 'ranked_round_robin_bitmap_prefix_select_v1')


class NativeQueueLayout(TypedDict):
    abi_version: int
    phase_count: int
    rank_directory_bytes: int
    device_bytes: int


def work_counts(tasks: Sequence[Mapping[str, int]]) -> list[int]:
    return [((t['m'] + 15) // 16) * ((t['n'] + 15) // 16) * ((t['prefix'] + t['group'] - 1) // t['group']) for t in tasks]


def layout(tasks: Sequence[Mapping[str, int]], mode: int) -> NativeQueueLayout:
    if type(mode) is not int or not 0 <= mode <= 3:raise ValueError('native schedule mode')
    if not 1 <= len(tasks) <= 4096:raise ValueError('native schedule task count')
    counts = work_counts(tasks);phases = len(set(counts)) if mode in (2, 3) else 0
    directory = (phases - 1) * (2 * ((len(tasks) + 31) // 32) + 1) * 4 if mode == 3 else 0
    payload = len(tasks) * 104 + 4 + (sum(counts) * 16 if mode == 0 else len(tasks) * 16 + phases * 16 + directory)
    return NativeQueueLayout(abi_version=2 if mode == 3 else 1, phase_count=phases, rank_directory_bytes=directory, device_bytes=payload)


class NativeQueueDispatch(TypedDict):
    policy: int
    group: int
    unsigned_inputs: int
    encoded_b: int


def dispatch_hint(case: Mapping) -> NativeQueueDispatch:
    policy = case.get('dispatch_policy', 0)
    if type(policy) is not int or not 0 <= policy <= 2:
        raise ValueError('native dispatch policy')
    tasks = case['tasks']
    encoded = case['encoded_b']
    first = tasks[0]
    signedness_key = 'unsigned_inputs' if 'unsigned_inputs' in first else 'unsigned'
    uniform = policy != 2 and (not encoded or policy == 1) and all(
        t['group'] == first['group'] and t[signedness_key] == first[signedness_key] for t in tasks)
    return NativeQueueDispatch(policy=policy, group=first['group'] if uniform else 0,
                               unsigned_inputs=int(first[signedness_key]) if uniform else 0,
                               encoded_b=int(encoded) if uniform else 0)


def validate_dispatch_capture(case: Mapping, actual: object) -> None:
    # Old captures without an explicit policy retain their historical identity.
    if actual is None and 'dispatch_policy' not in case:
        return
    if not isinstance(actual, dict) or actual != dispatch_hint(case) or any(type(v) is not int for v in actual.values()):
        raise ValueError('selected native dispatch specialization')
