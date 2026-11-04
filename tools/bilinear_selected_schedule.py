"""Independently lower pinned immutable expression DAGs, before slot allocation."""
from functools import lru_cache
import hashlib
import struct


def lower(inputs, nodes, root):
    if not 1 <= inputs <= 48 or len(nodes) > 90 or not 0 <= root < inputs + len(nodes):
        raise ValueError('selected DAG envelope')
    for index, (left, right, sa, sb, halve) in enumerate(nodes):
        if not 0 <= left < inputs + index or not -1 <= right < inputs + index or sa not in (-1, 1) or sb not in (-1, 1) or (halve and right >= 0):
            raise ValueError('selected DAG definition')
    needed = {root}
    for index in reversed(range(len(nodes))):
        if inputs + index in needed:
            left, right, *_ = nodes[index];needed.add(left)
            if right >= 0:needed.add(right)
    selected = [index for index in range(len(nodes)) if inputs + index in needed]
    loads = [index for index in range(inputs) if index in needed]
    last = {}
    for emitted, index in enumerate(selected):
        left, right, *_ = nodes[index];last[left] = emitted
        if right >= 0:last[right] = emitted
    last[root] = len(selected)
    locations = {node: index for index, node in enumerate(loads)}
    occupied = set(locations.values());slots = len(loads);steps = []
    for emitted, index in enumerate(selected):
        left, right, sa, sb, halve = nodes[index];a = locations[left];b = locations[right] if right >= 0 else a
        for source in {left, right} - {-1}:
            if last[source] == emitted:occupied.remove(locations[source])
        target = next(slot for slot in range(48) if slot not in occupied)
        occupied.add(target);locations[inputs + index] = target;slots = max(slots, target + 1)
        steps.append((target, a, b, sa, sb if right >= 0 else 0, int(halve)))
    return dict(input_ids=loads, steps=steps, slots=slots, result=locations[root])


@lru_cache(maxsize=2)
def schedules(rank48):
    result = []
    if rank48:
        from rank48_slp import programs
        for stage in programs(include_dag=True)[:2]:
            dag = stage['immutable_dag'];nodes = []
            for a, b, op in dag['nodes']:
                nodes.append((a, b, -1 if op == 3 else 1, -1 if op == 1 else 1, op == 2))
            result.append([lower(stage['inputs'], nodes, root) for root in dag['results']])
    else:
        from rank23_october55 import ARTIFACT, parse
        stages = parse(ARTIFACT.read_bytes())
        for side in ('U', 'V'):
            stage = stages[side]
            nodes = [(node['left'], node['right'], node['left_sign'], node['right_sign'], False) for node in stage['nodes']]
            result.append([lower(stage['inputs'], nodes, root) for root in stage['outputs']])
    return result


def identity_and_work(rank48):
    data = bytearray(b'rns8_selected_bilinear_v1\0')
    def append(value):data.extend(struct.pack('<I', value & 0xffffffff))
    append(48 if rank48 else 23)
    work = dict(binary=[0, 0], halving=[0, 0], input_loads=[0, 0])
    for side, plans in enumerate(schedules(rank48)):
        for plan in plans:
            for value in (len(plan['input_ids']), len(plan['steps']), plan['slots'], plan['result']):append(value)
            for value in plan['input_ids']:append(value)
            work['input_loads'][side] += len(plan['input_ids'])
            for step in plan['steps']:
                for value in step:append(value)
                work['binary'][side] += step[4] != 0;work['halving'][side] += step[5]
    return hashlib.sha256(data).hexdigest(), work
