"""Collect matched event/production pairs without merging or editing raw captures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import uuid
from benchmark_sweep_lib.execution import run_command
from benchmark_references import DEFAULT_INDEX,DEFAULT_CACHE,prepare,cached_arguments
from diagnostic_companion import build_binding, normalize_arguments


def paired_plan(binary, arguments, output, runs, collection_id, reference_directory=None):
    if type(runs) is not int or not 2 <= runs <= 100:
        raise ValueError('2..100 independent production processes required')
    args = cached_arguments(normalize_arguments(arguments),reference_directory)
    if '--sustained-batches' in args:
        index=args.index('--sustained-batches')
        if index+1==len(args) or args[index+1]!='0':
            raise ValueError('pair ordinary production latency first; instrumented sustained mode is not supported')
    rows = []
    for run in range(runs):
        order = ('production', 'instrumented') if run % 2 == 0 else ('instrumented', 'production')
        paths = {mode: output/f'{collection_id}-{run}-{mode}.json' for mode in order}
        rows.append(dict(independent_run=run, captures=[dict(mode=mode, output=str(paths[mode]),
            command=[str(binary), *args, '--host-timing', mode]) for mode in order],
            production=str(paths['production']), diagnostic=str(paths['instrumented']),
            binding=str(output/f'{collection_id}-{run}-binding.json')))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-cache',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--reference-cache-index',type=Path,default=DEFAULT_INDEX)
    parser.add_argument('--runs', type=int, default=2); parser.add_argument('--timeout-seconds', type=float, default=600)
    parser.add_argument('--plan-only', action='store_true'); parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(); arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    plan = paired_plan(args.bench, arguments, args.output, args.runs, str(uuid.uuid4()),DEFAULT_CACHE if args.reference_cache else None)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'pair-plan.json').open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(dict(schema='rns8_diagnostic_pair_collection_v1', pairs=plan, promotion_eligible=False), indent=2)+'\n')
    if args.plan_only:
        return 0
    if args.reference_cache:prepare(args.reference_cache_index,DEFAULT_CACHE)
    for row in plan:
        for capture in row['captures']:
            path = Path(capture['output'])
            if path.exists():
                raise ValueError('refusing immutable observation overwrite')
            if not run_command(capture['command'], path, args.timeout_seconds):
                raise RuntimeError('capture failed: '+str(path))
        binding = build_binding(row['production'], row['diagnostic'])
        with Path(row['binding']).open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(binding, indent=2, sort_keys=True)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
