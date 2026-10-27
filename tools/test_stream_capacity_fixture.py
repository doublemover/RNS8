"""Small CLI gates must not allocate the explicit physical-capacity dataset."""
import json
import subprocess
import sys


def main():
    binary=sys.argv[1]
    def run(*args):
        return subprocess.run([binary,*args],capture_output=True,text=True,timeout=30)
    result=run('--help')
    assert result.returncode==0 and '--describe' in result.stdout
    result=run('--describe')
    assert result.returncode==0,result.stderr
    d=json.loads(result.stdout)
    assert d['schema']=='rns8_stream_capacity_description_v1' and d['gpu_execution'] is False
    assert d['a_native_bytes']>32*2**30 and d['full_a_residue_bytes']>36*2**30
    assert d['caller_bytes']<40*2**30
    for name in ('serial','overlap'):
        r=d[name]
        assert r['device_bytes']<=64*2**20 and r['host_bytes']<=64*2**20
        assert r['output_tiles']==257 and r['k_panels_per_tile']==17
        assert r['host_bytes']==r['pinned_host_bytes']+r['ordinary_host_bytes']
    assert d['serial']['pinned_host_bytes']==0
    assert d['overlap']['pinned_host_bytes']>0 and d['overlap']['event_count']==8
    # The zero caller budget rejects before host-memory/device queries or any
    # caller allocation, regardless of the current machine's GPU/RAM.
    result=run('0','serial','0','67108864','67108864','insufficient-budget')
    assert result.returncode==77 and 'caller budget' in result.stderr and not result.stdout
    result=run('0','invented-mode','0','67108864','67108864','bad-mode')
    assert result.returncode==1 and not result.stdout
    return 0


if __name__=='__main__':
    raise SystemExit(main())
