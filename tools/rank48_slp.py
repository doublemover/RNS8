"""Strictly import pinned linear programs, prove their maps, and allocate live slots."""
import ast
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

from rank48_certificate import DATA, ROOT, load


def compile_stage(text, inputs, outputs):
    names = {f'i{i}': i for i in range(inputs)}
    vectors = [[Fraction(i == j) for j in range(inputs)] for i in range(inputs)]
    nodes = []

    def expression(value):
        if isinstance(value, ast.Name) and value.id in names:
            return names[value.id]
        if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
            left, right, op = expression(value.operand), -1, 3
        elif isinstance(value, ast.BinOp) and isinstance(value.op, (ast.Add, ast.Sub)):
            left, right = expression(value.left), expression(value.right)
            op = int(isinstance(value.op, ast.Sub))
        elif (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div) and
              isinstance(value.right, ast.Constant) and type(value.right.value) is int and value.right.value == 2):
            left, right, op = expression(value.left), -1, 2
        else:
            raise ValueError('unadmitted SLP expression')
        source = vectors[left]
        vector = ([x / 2 for x in source] if op == 2 else [-x for x in source] if op == 3 else
                  [x + (-y if op else y) for x, y in zip(source, vectors[right])])
        node = len(vectors);vectors.append(vector);nodes.append((left, right, op))
        return node

    for line in text.splitlines():
        match = re.fullmatch(r'([toz][0-9]+):=(.+);', line.strip())
        if not match:
            raise ValueError('invalid assignment envelope')
        names[match[1]] = expression(ast.parse(match[2], mode='eval').body)
    if any(f'o{i}' not in names for i in range(outputs)):
        raise ValueError('incomplete output map')
    results = [names[f'o{i}'] for i in range(outputs)]
    last = [-1] * len(vectors)
    for index, (left, right, _) in enumerate(nodes):
        last[left] = index
        if right >= 0:last[right] = index
    for node in results:last[node] = len(nodes)
    locations = list(range(inputs)) + [-1] * len(nodes)
    free = []
    slots = inputs
    steps = []
    for index, (left, right, op) in enumerate(nodes):
        a, b = locations[left], locations[right] if right >= 0 else 0
        for node in set((left, right)):
            if node >= 0 and last[node] == index:free.append(locations[node])
        target = free.pop() if free else slots
        if target == slots:slots += 1
        locations[inputs + index] = target
        steps.append((target, a, b, op))
    return dict(inputs=inputs, outputs=outputs, slots=slots, steps=steps,
                results=[locations[node] for node in results], matrix=[vectors[node] for node in results])


def programs():
    manifest = json.loads((DATA/'provenance.json').read_text(encoding='utf-8'))
    left, right, output = load()
    result = []
    for suffix, inputs, outputs, expected in [('L',16,48,left),('R',16,48,right),('P',48,16,output)]:
        name = f'4x4x4_48_204_{suffix}.slp';data = (DATA/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest['files'][name]['sha256']:
            raise ValueError('SLP artifact digest changed')
        stage = compile_stage(data.decode('ascii'), inputs, outputs)
        if stage['matrix'] != expected:raise ValueError('SLP differs from certified coefficient map')
        result.append(stage)
    count = sum(len(p['steps']) for p in result)
    halves = sum(step[3] == 2 for p in result for step in p['steps'])
    if count != 204 or halves != 4:raise ValueError(f'canonical program operation count changed: {count}/{halves}')
    return result


def header(stages):
    identity = [{k:v for k,v in stage.items() if k != 'matrix'} for stage in stages]
    digest = hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).hexdigest()
    capacity = max(p['slots'] for p in stages)
    steps = max(len(p['steps']) for p in stages)
    lines = ['// Generated from pinned PLinOpt SLPs by tools/rank48_slp.py; see third_party/rank48.',
             '#pragma once','#include <cstdint>','namespace rns8::detail::rank48::slp {',
             f'inline constexpr unsigned capacity={capacity}, max_steps={steps};',
             'struct Step {uint8_t target,left,right,operation;};',
             'struct Plan {unsigned inputs,outputs,count,slots;Step steps[max_steps];uint8_t results[48];};',
             'inline constexpr const char* schedule_sha256="'+digest+'";']
    if capacity > 255:raise ValueError('slot index overflow')
    for name, stage in zip(('u','v','w'),stages):
        lines += [f'inline constexpr Plan {name}{{{stage["inputs"]},{stage["outputs"]},{len(stage["steps"])},{stage["slots"]},{{']
        lines += ['  {'+','.join(map(str,step))+'},' for step in stage['steps']]
        lines += ['},{'+','.join(map(str,stage['results']))+'}};']
    lines += ['#if defined(__HIPCC__)','#define RNS8_R48_HD __host__ __device__','#else','#define RNS8_R48_HD','#endif',
              'RNS8_R48_HD inline int center(int value,int q){',
              '  value%=q;', '  if(value<0)value+=q;', '  return value>=(q+1)/2?value-q:value;', '}',
              'RNS8_R48_HD inline void evaluate(const Plan& plan,int8_t* slots,int q){',
              '  for(unsigned i=0;i<plan.count;++i){const auto op=plan.steps[i];const int a=slots[op.left],b=slots[op.right];',
              '    const int value=op.operation==2?a*((q+1)/2):op.operation==3?-a:op.operation==1?a-b:a+b;',
              '    slots[op.target]=int8_t(center(value,q));}',
              '}', '#undef RNS8_R48_HD','}','']
    return '\n'.join(lines)


if __name__ == '__main__':
    stages = programs()
    (ROOT/'src/backend_common/rank48_slp.hpp').write_text(header(stages),encoding='utf-8',newline='\n')
    print(json.dumps(dict(operations=204,halvings=4,slots=[p['slots'] for p in stages],exact_linear_maps='verified')))
