"""Independent exact model of versioned native calibration fixtures; no GPU access."""
from functools import lru_cache
from modulus_set_search import DEFAULT_MODULI
MASK=(1<<64)-1
PROFILES=('full-width-v1','uniform-small-v2','mixed-tiles-v2','zero-tiles-v2','localized-outlier-v2')

def identity(profile,seed):
    if profile=='full-width-v1' and seed==3135:return 'full_width_native64_mt19937_64_seed3135_forced_extrema_v1'
    return f'native_distribution_v2:{profile}:{seed}'

def word(value):
    value=(value+0x9e3779b97f4a7c15)&MASK
    value=((value^(value>>30))*0xbf58476d1ce4e5b9)&MASK
    value=((value^(value>>27))*0x94d049bb133111eb)&MASK
    return value^(value>>31)

def mt19937_64(seed):
    state=[seed]
    for i in range(1,312):state.append((6364136223846793005*(state[-1]^(state[-1]>>62))+i)&MASK)
    while True:
        for i in range(312):
            value=(state[i]&0xffffffff80000000)|(state[(i+1)%312]&0x7fffffff)
            state[i]=state[(i+156)%312]^(value>>1)^(0xb5026f5aa96619e9 if value&1 else 0)
        for value in state:
            value^=(value>>29)&0x5555555555555555
            value^=(value<<17)&0x71d67fffeda60000
            value^=(value<<37)&0xfff7eee000000000
            yield value^(value>>43)

def inputs(m,n,k,unsigned,profile,seed):
    if profile not in PROFILES:raise ValueError('unknown dataset profile')
    if profile=='full-width-v1':
        rng=mt19937_64(seed);a=[next(rng) for _ in range(m*k)];b=[next(rng) for _ in range(k*n)]
        a[0]=1<<63;b[0]=MASK;return a,b
    def sample(index,axis,rhs):
        bits=word((index+(seed<<32)+(0x123456789abcdef0 if rhs else 0))&MASK)
        magnitude=1+bits%7
        if profile=='mixed-tiles-v2' and (axis//16)%2:magnitude=(1<<48)+(bits&65535)
        if profile=='zero-tiles-v2' and (axis//16)%2:magnitude=0
        return (-magnitude)&MASK if not unsigned and bits>>63 else magnitude
    a=[sample(row*k+q,row,False) for row in range(m) for q in range(k)]
    b=[sample(q*n+col,col,True) for q in range(k) for col in range(n)]
    if profile=='localized-outlier-v2':a[(m//2)*k+k//2]=1<<60;b[(k//2)*n+n//2]=1<<60
    return a,b

@lru_cache(maxsize=128)
def model(m,n,k,unsigned,group,profile,seed,adaptive):
    a,b=inputs(m,n,k,unsigned,profile,seed)
    def magnitude(bits):return ((-bits)&MASK) if not unsigned and bits>>63 else bits
    def summaries(rhs):
        count=n if rhs else m;values=b if rhs else a;out=[]
        for start in range(0,count,16):
            maximum=l1=0
            for axis in range(start,min(count,start+16)):
                row=[magnitude(values[q*n+axis if rhs else axis*k+q]) for q in range(k)]
                maximum=max(maximum,max(row));l1=max(l1,sum(row))
            out.append((maximum,l1))
        return out
    rows,cols=summaries(False),summaries(True);hist=[0]*21
    for am,al1 in rows:
        for bm,bl1 in cols:
            bound=min(am*bm*k,am*bl1,bm*al1);threshold=bound*(1 if unsigned else 2);prefix=0;product=1
            if bound:
                while prefix==0 or product<=threshold:
                    if prefix==20:raise ValueError('dataset exceeds certified output capacity')
                    product*=DEFAULT_MODULI[prefix];prefix+=1
            hist[prefix]+=1
    groups=sum(count*((prefix+group-1)//group) for prefix,count in enumerate(hist))
    fixed=len(rows)*len(cols)*((20+group-1)//group)
    buckets=sum(bool(x) for x in hist[1:])
    if ((profile=='mixed-tiles-v2' and buckets<3) or (profile=='localized-outlier-v2' and buckets<2)
            or (profile=='zero-tiles-v2' and (not hist[0] or not buckets))):raise ValueError('dataset degenerated advertised buckets')
    if profile!='full-width-v1' and groups>=fixed:raise ValueError('savings fixture has no group-work savings')
    digest=1469598103934665603
    for values in (a,b):
        for bits in values:
            for byte in range(8):digest=((digest^((bits>>(byte*8))&255))*1099511628211)&MASK
    channels=sum(p*count for p,count in enumerate(hist))
    return dict(schema_version=2,method='axis_l1_exact_cpp_int_and_production_helper_crosscheck',
                input_hash_fnv1a64=str(digest),required_prefix_histogram=hist,adaptive_executed_prefix_histogram=list(hist),
                zero_tiles=hist[0],required_channels=channels,adaptive_executed_channels=channels,
                adaptive_group_work_items=groups,fixed_capacity_group_work_items=fixed,
                globally_required_prefix=max(p for p,x in enumerate(hist) if x),
                source_work_not_instruction_or_speedup=True,actual_candidate_group_work_items=groups if adaptive else fixed,
                global_minimal_completed_output_control_implemented=False)
