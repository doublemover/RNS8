"""Explicit constant native-input datasets shared by product collectors."""
def inputs(case):
    values=case.get('input_values',[1,1])
    low,high=(0,2**64-1) if case['unsigned_inputs'] else (-2**63,2**63-1)
    if type(values) is not list or len(values)!=2 or any(type(v) is not int or not low<=v<=high for v in values):
        raise ValueError('two in-range integer dataset inputs required')
    return values

def identity(case):
    a,b=inputs(case)
    return 'all_one_inputs_exact_integer_dag_v1' if (a,b)==(1,1) else f'constant_native64_dag_v1;A={a};B={b}'
