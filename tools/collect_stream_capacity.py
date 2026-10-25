"""Explicit independent physical streaming capacity gates; never a resumed sweep."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid

from benchmark_sweep_lib.execution import run_command
from benchmark_schema.wide_magnitude import MODULI
from build_manifest import manifest_provenance_complete
from qualify_large_addresses import strict_json

M,N,K,P=65537,3,65537,9
BASE=2**20
MAXIMUM=BASE+7
BOUND=K*MAXIMUM*MAXIMUM
MINIMUM_PREFIX=next(p for p in range(1,len(MODULI)+1) if math.prod(MODULI[:p])>2*BOUND)
TILES,PANELS=257,17


def expected_checksums():
    oracle={}
    for r in range(15):
        for c in range(N):
            values=[(BASE+(3*r+7*q)%15-7)*(1 if c%2 else -1)*(BASE+(11*q+5*c+3)%15-7) for q in range(15)]
            oracle[r,c]=sum(values)*(K//15)+sum(values[:K%15])
    initial=sum(oracle[r%15,c]*(r*N+c+1) for r in range(M) for c in range(N))
    changed=initial+sum((1 if c%2 else -1)*(BASE+(5*c+3)%15-7)*(c+1) for c in range(N))
    return [str(initial),str(initial),str(changed)]


def validate(data,mode,run_id,device_id=0):
    if not isinstance(data,dict):raise ValueError('capacity receipt must be an object')
    expected=dict(schema='rns8_stream_physical_capacity_v1',mode=mode,run_id=run_id,m=M,n=N,k=K,prefix=P,
        semantic_contract='bounded_i64_explicit_k_amax_bmax',max_abs_a=MAXIMUM,max_abs_b=MAXIMUM,
        full_expression_abs_bound=BOUND,minimum_proven_prefix=MINIMUM_PREFIX,
        all_owned_device_and_pinned_payload_retired=True,
        independent_oracle_inside_timer=False,promotion_eligible=False,process_rss_measured=False,device_id=device_id)
    for key,value in expected.items():
        if type(data.get(key)) is not type(value) or data[key]!=value:raise ValueError(f'invalid physical capacity field: {key}')
    if data.get('ordered_moduli')!=list(MODULI[:P]) or data.get('basis_product_decimal')!=str(math.prod(MODULI[:P])):
        raise ValueError('incorrect ordered basis/range capacity')
    for key in ('actual_total_device_bytes','caller_payload_bytes','actual_caller_capacity_bytes','caller_budget_bytes',
            'available_host_bytes_before','host_reserve_bytes','a_native_allocation_bytes','full_a_residue_bytes',
            'device_budget_bytes','library_host_budget_bytes','minimum_full_a_residue_bytes'):
        if type(data.get(key)) is not int or data[key]<=0:raise ValueError(f'unknown physical capacity: {key}')
    expected_caller=8*(M*(K+3)+K*(N+3)+M*(N+3)+M*N+15*(K+3))
    if (data['a_native_allocation_bytes']!=M*(K+3)*8 or data['full_a_residue_bytes']!=M*K*P or
            data['minimum_full_a_residue_bytes']!=M*K*MINIMUM_PREFIX or
            data['caller_payload_bytes']!=expected_caller or
            min(data['a_native_allocation_bytes'],data['minimum_full_a_residue_bytes'])<=data['actual_total_device_bytes'] or
            data['actual_caller_capacity_bytes']<data['caller_payload_bytes'] or
            data['actual_caller_capacity_bytes']>data['caller_budget_bytes'] or
            data['host_reserve_bytes']!=8*2**30 or data['available_host_bytes_before']<
            data['caller_payload_bytes']+data['library_host_budget_bytes']+data['host_reserve_bytes']):
        raise ValueError('physical capacity and explicit RAM/device budgets do not reconcile')
    manifest=data.get('build_manifest')
    if not manifest_provenance_complete(manifest,'hip-direct'):
        raise ValueError('missing clean source-bound compiled provenance')
    options=dict(line.split('=',1) for line in manifest['configuration'].splitlines() if '=' in line)
    if options.get('hip','OFF').upper() not in ('ON','TRUE','1','YES'):
        raise ValueError('physical receipt requires a HIP-enabled artifact')
    if not isinstance(data.get('target'),str) or not data['target'].startswith('gfx') or not data.get('device'):
        raise ValueError('missing actual GPU target')
    for key in ('hip_runtime_version','hip_driver_version'):
        if type(data.get(key)) is not int or data[key]<=0:raise ValueError('unknown HIP/runtime identity')
    r=data.get('queried_resources')
    if not isinstance(r,dict) or r.get('output_tiles')!=TILES or r.get('k_panels_per_tile')!=PANELS:
        raise ValueError('incorrect complete tile/panel coverage geometry')
    for key in ('device_bytes','host_bytes','pinned_host_bytes','ordinary_host_bytes'):
        if type(r.get(key)) is not int or r[key]<0:raise ValueError('unknown resource capacity')
    if (r['device_bytes']>data['device_budget_bytes'] or r['host_bytes']>data['library_host_budget_bytes'] or
            r['host_bytes']!=r['pinned_host_bytes']+r['ordinary_host_bytes']):raise ValueError('resource query escaped budget')
    phases=data.get('phases');checksums=expected_checksums()
    if not isinstance(phases,list) or len(phases)!=3:raise ValueError('need cold/warm/changed-generation phases')
    for i,phase in enumerate(phases):
        if not isinstance(phase,dict):raise ValueError('capacity phase must be an object')
        fixed=dict(source_a_generation=2 if i==2 else 1,epoch=i+1,checked_outputs=M*N,
            all_output_and_padding_exact=True,weighted_checksum=checksums[i],completed_tiles=TILES,
            completed_panels=TILES*PANELS,h2d_payload_bytes=8*(M*K+TILES*K*N),
            d2h_payload_bytes=8*M*N+4*TILES,h2d_transfers=2*TILES*PANELS,d2h_transfers=2*TILES)
        for key,value in fixed.items():
            if type(phase.get(key)) is not type(value) or phase[key]!=value:raise ValueError(f'incomplete/wrong physical phase: {key}')
        elapsed=phase.get('completed_host_call_ns')
        if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<=0:raise ValueError('invalid completed-call timer')
        device,pinned=(phase.get(k) for k in ('observed_requested_device_peak_delta','observed_requested_pinned_peak_delta'))
        if (type(device) is not int or type(pinned) is not int or device<r['device_bytes'] or pinned<r['pinned_host_bytes'] or
                device>data['device_budget_bytes'] or pinned+r['ordinary_host_bytes']>data['library_host_budget_bytes']):
            raise ValueError('observed requested peak escaped budget')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--caller-budget-bytes',type=int,required=True);p.add_argument('--device-budget-bytes',type=int,required=True)
    p.add_argument('--host-budget-bytes',type=int,required=True);p.add_argument('--device',type=int,default=0)
    p.add_argument('--runs',type=int,default=2);p.add_argument('--timeout-seconds',type=float,default=900)
    a=p.parse_args()
    if not (0<=a.device<=63 and 1<=a.runs<=5 and all(0<v<=2**64-1 for v in
            (a.caller_budget_bytes,a.device_budget_bytes,a.host_budget_bytes)) and math.isfinite(a.timeout_seconds) and 0<a.timeout_seconds<=3600):
        p.error('invalid budgets/device or bounded process count/timeout')
    binary=a.binary.resolve(strict=True);a.output.mkdir(parents=True,exist_ok=False)
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    root=Path(__file__).resolve().parents[1]
    sources=set()
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.suffix=='.py' and path.is_relative_to(root/'tools'):sources.add(path)
    source_identity={p.relative_to(root).as_posix():digest(p) for p in sorted(sources)}
    def sources_unchanged():
        return all(digest(root/name)==value for name,value in source_identity.items())
    pinned=digest(binary);records=[];identity=None
    result=dict(schema='rns8_stream_capacity_collection_v1',executable_sha256=pinned,records=records,
        requested_processes=2*a.runs,campaign_resumed=False,promotion_eligible=False,status='pending',
        collector_source_files=source_identity,python_version=sys.version)
    def save():
        (a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    save()
    for run in range(a.runs):
        for mode in (('serial','overlap') if run%2==0 else ('overlap','serial')):
            record=dict(run=run,mode=mode,run_id=str(uuid.uuid4()));records.append(record)
            output=a.output/f'{mode}-{run:02d}.json'
            command=[str(binary),str(a.device),mode,str(a.caller_budget_bytes),str(a.device_budget_bytes),str(a.host_budget_bytes),record['run_id']]
            record['command']=command;save()
            try:
                if digest(binary)!=pinned or not sources_unchanged():raise ValueError('pinned executable/collector sources changed')
                if not run_command(command,output,a.timeout_seconds):
                    failed=output.with_suffix('.failed.json');record['failure']=strict_json(failed.read_text())
                    result['status']='unrun' if record['failure'].get('returncode')==77 else 'failed';save()
                    return 77 if result['status']=='unrun' else 1
                data=strict_json(output.read_text(encoding='utf-8'));validate(data,mode,record['run_id'],a.device)
                if digest(binary)!=pinned or not sources_unchanged() or data['artifact_capture']['executable_sha256']!=pinned or not data['artifact_capture']['unchanged_during_run']:
                    raise ValueError('changed executable identity')
                scope=(data['target'],data['device'],data['actual_total_device_bytes'],data['hip_runtime_version'],
                    data['hip_driver_version'],data['build_manifest']['source_digest'],data['build_manifest']['dependency_digest'])
                if identity is not None and scope!=identity:raise ValueError('mixed device/build qualification cohort')
                identity=scope;record.update(capture=output.name,capture_sha256=digest(output),execution_id=data['execution_id'])
            except (ValueError,KeyError,TypeError,OSError) as error:
                record['error']=str(error);result['status']='failed';save();return 1
            save();print(f'capacity {mode} process {run+1}/{a.runs}: exact',flush=True)
    result['status']='all_exact';save();return 0


if __name__=='__main__':raise SystemExit(main())
