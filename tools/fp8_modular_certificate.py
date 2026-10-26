"""The conditional gfx942 FNUZ four-product model; never a target receipt."""
from dataclasses import asdict
import argparse
import hashlib
import json
import re
from pathlib import Path
from arithmetic_certificate import ArithmeticCertificate,InstructionContract,ProductPlane,verify
from floating_contract import FloatingContract,FloatingBlock,bind_artifacts

def encode(value):
    if type(value) is not int or not -8<=value<=8:raise ValueError('exact FNUZ alphabet')
    if not value:return 0
    magnitude=abs(value);exponent=magnitude.bit_length()-1
    return (128 if value<0 else 0)|((exponent+8)<<3)|((magnitude<<(3-exponent))-8)

def certificate(modulus,compiler,source_sha256):
    if type(modulus) is not int or not 2<=modulus<=256:raise ValueError('finite modulus')
    if type(source_sha256) is not str or re.fullmatch('[0-9a-f]{64}',source_sha256) is None:raise ValueError('source identity')
    canonical=range(modulus)
    values=[x if x<(modulus+1)//2 else x-modulus for x in canonical]
    low=tuple((x+8)%16-8 for x in values);high=tuple((x-l)//16 for x,l in zip(values,low))
    planes=[]
    for a,wa in ((low,1),(high,16)):
        for b,wb in ((low,1),(high,16)):
            planes.append(ProductPlane(a,b,wa*wb,tuple(map(encode,a)),tuple(map(encode,b))))
    f=FloatingContract('amd_e4m3_fnuz',(FloatingBlock(65536,127,127,0),),block_k=65536,
        reduction='fold',destination_bits=64,lowering_sha256='0'*64,proof_artifact_sha256='0'*64)
    i=InstructionContract('gfx942','mfma_f32_16x16x32_fp8_fp8',-8,8,-2**24,2**24,compiler,
        flags=('source_sha256='+source_sha256,'FNUZ_bias8','FP32_integer_accumulation_assumption'),floating=f)
    # This checker conservatively prices unreduced weighted recombination in
    # INT64. The actual kernel reduces each exact partial before weighting.
    return bind_artifacts(ArithmeticCertificate(modulus,tuple(planes),1,65536,0,0,-2**63,2**63-1,i))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--modulus',type=int,required=True)
    p.add_argument('--compiler',required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    c=certificate(a.modulus,a.compiler,hashlib.sha256(a.source.read_bytes()).hexdigest());result=verify(c)
    print(json.dumps(dict(schema='rns8_fp8_modular_model_v1',certificate=asdict(c),verdict=result,
        target_qualified=False,promotion_eligible=False),indent=2))
    return 0 if not result['proof_failures'] else 1

if __name__=='__main__':raise SystemExit(main())
