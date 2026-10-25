"""Strict measured-price file contract shared by collection and held-out review."""
import hashlib
import json
import math
import re


def price_identity(document):
    return hashlib.sha256(json.dumps(document,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def price_cells(document):
    keys={'schema','cost_schema_version','identity','training_run_ids','training_execution_ids','source_artifacts','dataset','cache_state','timing_source','device','runtime_version','cells','qualified','promotion_eligible'}
    if type(document) is not dict or set(document)!=keys or document['schema']!='rns8_product_stage_prices_v1' or type(document['cost_schema_version']) is not int or document['cost_schema_version']!=2 or document['qualified'] is not False or document['promotion_eligible'] is not False:
        raise ValueError('stage price ABI')
    if document['dataset']!='all_one_inputs_exact_integer_dag_v1' or document['cache_state']!='hot_repeated_stage' or document['timing_source']!='host_monotonic_completed_backend_call':raise ValueError('stage price scope')
    if type(document['device']) is not str or not document['device'] or type(document['runtime_version']) is not int or not 1<=document['runtime_version']<=2**31-1:raise ValueError('stage price device/runtime')
    identity=document['identity']
    def digest(v):return type(v) is str and re.fullmatch('[0-9a-f]{64}',v) is not None
    if type(identity) is not dict or set(identity)!={'graph_sha256','build_sha256','target','node_count','minimum_prefixes'} or not all(digest(identity[k]) for k in ('graph_sha256','build_sha256')) or type(identity['target']) is not str or not identity['target']:
        raise ValueError('stage price identity')
    n=identity['node_count'];minimum=identity['minimum_prefixes']
    if type(n) is not int or not 1<=n<=64 or type(minimum) is not list or len(minimum)!=n or any(type(p) is not int or not 1<=p<=20 for p in minimum):raise ValueError('stage price exact prefixes')
    for name in ('training_run_ids','training_execution_ids'):
        ids=document[name]
        if type(ids) is not list or not 5<=len(ids)<=100 or any(type(s) is not str or not s for s in ids) or len(set(ids))!=len(ids):raise ValueError('stage training processes')
    if len(document['training_run_ids'])!=len(document['training_execution_ids']):raise ValueError('stage training process mapping')
    artifacts=document['source_artifacts']
    if type(artifacts) is not list or len(artifacts)!=len(document['training_run_ids']) or any(type(a) is not dict or set(a)!={'executable_sha256','capture_sha256'} or not all(digest(v) for v in a.values()) for a in artifacts):raise ValueError('stage price artifacts')
    if type(document['cells']) is not list or not 1<=len(document['cells'])<=640000:raise ValueError('stage price cells')
    result={}
    for cell in document['cells']:
        if type(cell) is not dict or set(cell)!={'node','kind','capacity','old_prefix','prefix','cost_ns','interval_ns'}:raise ValueError('stage price cell ABI')
        if any(type(cell[k]) is not int for k in ('node','capacity','old_prefix','prefix','cost_ns')):raise ValueError('stage price integer')
        node,cap,old,p,cost=(cell[k] for k in ('node','capacity','old_prefix','prefix','cost_ns'))
        if not 0<=node<n or not 1<=p<=20 or not 0<=old<=20 or not 0<=cap<=20 or not 1<=cost<2**64-1:raise ValueError('stage price range')
        kind=cell['kind']
        valid=(kind=='uniform_gemm' and cap==old==0 and p>=minimum[node] or kind=='lift' and cap==0 and minimum[node]<=old<p or
            kind=='packed_gemm' and old==0 and minimum[node]<=p<=cap or kind=='pack' and old<p<=cap or
            kind=='publish' and cap==old==0 and node==n-1)
        if not valid:raise ValueError('stage price physical axes')
        interval=cell['interval_ns']
        if type(interval) is not list or len(interval)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in interval) or interval[0]>interval[1] or not interval[0]<=cost<=math.ceil(interval[1]):raise ValueError('stage price interval')
        key=(node,kind,cap,old,p)
        if key in result:raise ValueError('duplicate stage price')
        result[key]=cell
    return result
