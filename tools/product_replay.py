"""Checked, explicitly scoped cold/steady/training lifetime estimates."""
LIMIT=2**64-1
FIELDS=dict(expected_replays=1,retain_external_b=False,require_amortization=False,
    preparation_ns=0,calibration_ns=0,orchestration_ns=0,publication_ns=0,
    baseline_replay_ns=LIMIT,baseline_preparation_ns=0)

def policy(case,training_ns=0,publish_ns=0):
    extra=case.get('replay_policy',{})
    if type(extra) is not dict or set(extra)-set(FIELDS):raise ValueError('replay policy fields')
    result=FIELDS|extra
    for k,v in result.items():
        if k in ('retain_external_b','require_amortization'):
            if type(v) is not bool:raise ValueError('replay boolean')
        elif type(v) is not int or not 0<=v<=LIMIT:raise ValueError('replay integer')
    if not 1<=result['expected_replays']<LIMIT:raise ValueError('expected replay lifetime')
    result['calibration_ns']=max(training_ns,result['calibration_ns'])
    result['publication_ns']=max(publish_ns,result['publication_ns'])
    return result

def estimate(p,first,steady):
    def checked(n):
        if not 0<=n<LIMIT:raise ValueError('unknown/overflowed replay cost')
        return n
    common=checked(p['orchestration_ns']+p['publication_ns'])
    first=checked(first+common);steady=checked(steady+common)
    once=checked(p['preparation_ns']+p['calibration_ns']);n=p['expected_replays']
    lifetime=checked(once+first+(n-1)*steady);status=0;break_even=0
    if p['baseline_replay_ns']<LIMIT and p['baseline_preparation_ns']<LIMIT:
        base=p['baseline_preparation_ns']+n*p['baseline_replay_ns']
        if base<LIMIT:status=1 if lifetime<base else 2
        if base>=LIMIT:return dict(first_ns=first,steady_ns=steady,once_ns=once,lifetime_ns=lifetime,
            amortized_ns=(lifetime+n-1)//n,break_even_replays=0,amortization_status=0)
        if steady<p['baseline_replay_ns'] and first>=steady:
            premium=checked(once+first-steady)-p['baseline_preparation_ns']
            break_even=1 if premium<0 else premium//(p['baseline_replay_ns']-steady)+1
        elif once+first<p['baseline_preparation_ns']+p['baseline_replay_ns']:break_even=1
    return dict(first_ns=first,steady_ns=steady,once_ns=once,lifetime_ns=lifetime,
        amortized_ns=(lifetime+n-1)//n,break_even_replays=break_even,amortization_status=status)
