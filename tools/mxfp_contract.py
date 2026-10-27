"""CDNA4 source contract and exact dyadic proof; never a hardware receipt."""
from fractions import Fraction
from floating_contract import decode_element,decode_scale
import hashlib
import json

# selector: (format name, radix bits, digit count, minimum/maximum scale)
FORMATS={4:('ocp_e2m1',2,4,-1,1),2:('ocp_e2m3',3,3,0,3),
         3:('ocp_e3m2',3,3,-2,4),0:('ocp_e4m3',4,2,-5,9)}
PAIRS={3:(4,4),4:(2,2),5:(3,3),6:(0,0),7:(0,4),8:(4,0),9:(2,4),10:(4,2)}
NAMES=('mxfp4','mxfp6e2','mxfp6e3','mxfp8','mxfp8-fp4','mxfp4-fp8','mxfp6-fp4','mxfp4-fp6')

def mode_contract(mode):
    materialized=mode.endswith('-materialized')
    base=mode.removesuffix('-materialized')
    dynamic=base.endswith('-dynamic');base=base.removesuffix('-dynamic')
    if base not in NAMES:raise ValueError('MX format mode')
    identity=NAMES.index(base)+3;a,b=PAIRS[identity]
    return dict(identity=identity,format='gfx950_'+base,method=int(materialized),dynamic=dynamic,
                product_count=FORMATS[a][2]*FORMATS[b][2],lhs=a,rhs=b)

def modes():
    return tuple(name+dynamic+materialized for name in NAMES for dynamic in ('','-dynamic') for materialized in ('','-materialized'))

def scale_schedule(identity,k,dynamic):
    if identity not in PAIRS or type(k) is not int or k<=0:raise ValueError('shape/format')
    blocks=(k+127)//128
    result=[]
    for operand,selector in enumerate(PAIRS[identity]):
        *_,low,high=FORMATS[selector]
        result.extend(((low,high,0) if operand==0 else (high,low,0))[i%3] if dynamic else 0 for i in range(blocks))
    return result

def digits(value,selector):
    _,bits,count,_,_=FORMATS[selector];radix=1<<bits;out=[]
    for index in range(count):
        low=value if index+1==count else (value+radix//2)%radix-radix//2
        out.append(low);value=(value-low)//radix
    if value:raise ValueError('digit overflow')
    return out

def exact_code(value,selector,scale):
    name,*_=FORMATS[selector]
    target=Fraction(value)/decode_scale(scale+127)
    bits={'ocp_e2m1':4,'ocp_e2m3':6,'ocp_e3m2':6,'ocp_e4m3':8}[name]
    for code in range(1<<bits):
        try:decoded=decode_element(name,code)
        except ValueError:continue
        if decoded==target:return code
    raise ValueError('nonexact operand scale')

def certificate(identity,source_sha256,compiler):
    if identity not in PAIRS or len(source_sha256)!=64 or any(c not in '0123456789abcdef' for c in source_sha256) or not compiler:raise ValueError('certificate binding')
    operands=[]
    for selector in PAIRS[identity]:
        name,bits,count,low,high=FORMATS[selector]
        alphabet=sorted({d for value in range(-128,128) for d in digits(value,selector)})
        for value in range(-128,128):
            if sum(d*(1<<(i*bits)) for i,d in enumerate(digits(value,selector)))!=value:raise ValueError('reconstruction')
        for scale in range(low,high+1):
            for value in alphabet:
                code=exact_code(value,selector,scale)
                if decode_element(name,code)*decode_scale(scale+127)!=value:raise ValueError('physical scale')
        operands.append(dict(selector=selector,format=name,radix_bits=bits,digits=count,alphabet=alphabet,scale_range=[low,high]))
    a,b=operands
    bound=128*max(map(abs,a['alphabet']))*max(map(abs,b['alphabet']))
    if bound>2**24:raise ValueError('accumulator integer grid')
    result=dict(schema='rns8_mxfp_integer_model_v1',format_id=identity,operands=operands,
        safe_k=128,proper_partial_bound=bound,product_count=a['digits']*b['digits'],
        instruction='__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4',target='gfx950',
        scale_format='ocp_e8m0_bias127',scale_policy='inverse_physical_encoding_reset_every128',
        reduction='each_product_to_integer_modulus_before_weighted_integer_merge',
        source_sha256=source_sha256,compiler=compiler,arithmetic_model_proven=True,
        target_qualified=False,performance_qualified=False,promotion_eligible=False)
    result['certificate_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result


def main():
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--format',type=int,choices=tuple(PAIRS),required=True)
    p.add_argument('--compiler',required=True);p.add_argument('--source',type=Path,required=True)
    a=p.parse_args();result=certificate(a.format,hashlib.sha256(a.source.read_bytes()).hexdigest(),a.compiler)
    print(json.dumps(result,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
